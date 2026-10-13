from types import SimpleNamespace

import numpy as np
import pytest
from electromagnetic_fields.dipole_fields.laser_pulses import SineSquareLaser
from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.angular_momentum import angular_matrix_element

from tdmcscf.grid.spherical import DipoleFieldTerm, SphericalDipoleField


@pytest.fixture(scope="module")
def grid():
    return FEMDVR(
        np.array([0.0, 1.0, 3.0, 6.0]),
        np.array([9, 10, 11]),
        Linear_map,
        GaussLegendreLobatto,
        symmetrize=False,
    )


def _constant(value):
    return lambda time: value


def _direction_matrix(orbital_lm, axis):
    functions = (
        lambda theta, phi: np.sin(theta) * np.cos(phi),
        lambda theta, phi: np.sin(theta) * np.sin(phi),
        lambda theta, phi: np.cos(theta),
    )
    matrix = np.zeros((len(orbital_lm), len(orbital_lm)), dtype=complex)
    for i, (l1, m1) in enumerate(orbital_lm):
        for j, (l2, m2) in enumerate(orbital_lm):
            matrix[i, j] = angular_matrix_element(
                l1, m1, l2, m2, functions[axis], order=15
            )
    return matrix


@pytest.mark.parametrize("axis", range(3))
def test_length_action_for_each_cartesian_axis(grid, axis):
    components = [None, None, None]
    components[axis] = _constant(1.7)
    operator = SphericalDipoleField.from_components(
        grid,
        2,
        gauge="length",
        x=components[0],
        y=components[1],
        z=components[2],
    )
    rng = np.random.default_rng(12)
    orbitals = rng.normal(size=(2,) + operator.orbital_shape) + 1j * rng.normal(
        size=(2,) + operator.orbital_shape
    )
    direction = _direction_matrix(operator.orbital_lm, axis)
    expected = 1.7 * operator.r[None, :, None] * np.einsum(
        "ij,paj->pai", direction, orbitals
    )
    np.testing.assert_allclose(operator.apply(orbitals, 0.3), expected, atol=2e-14)


@pytest.mark.parametrize("axis", range(3))
def test_velocity_action_on_an_s_orbital(grid, axis):
    components = [None, None, None]
    components[axis] = _constant(2.0)
    operator = SphericalDipoleField.from_components(
        grid,
        1,
        gauge="velocity",
        x=components[0],
        y=components[1],
        z=components[2],
        charge=-2.5,
        mass=4.0,
    )
    orbitals = np.zeros(operator.orbital_shape, dtype=complex)
    i00 = operator.orbital_lm.index((0, 0))
    orbitals[:, i00] = operator.r * (6.0 - operator.r)

    derivative = 6.0 - 2.0 * operator.r
    reduced_radial_gradient = derivative - orbitals[:, i00] / operator.r
    direction = _direction_matrix(operator.orbital_lm, axis)[:, i00]
    momentum = -1j * reduced_radial_gradient[:, None] * direction[None, :]
    expected = -(operator.charge / operator.mass) * 2.0 * momentum
    np.testing.assert_allclose(operator.apply(orbitals, 1.2), expected, atol=2e-11)


def test_sum_of_polarized_terms_and_component_convenience(grid):
    terms = [
        DipoleFieldTerm(lambda t: 2.0 * t, (1.0, 0.0, 0.0)),
        DipoleFieldTerm(lambda t: 1.0 - t, (0.0, 3.0, 0.0)),
        DipoleFieldTerm(lambda t: -0.5, (1.0, 0.0, 2.0)),
    ]
    summed = SphericalDipoleField(grid, 1, gauge="length", fields=terms)
    np.testing.assert_allclose(summed.field_vector(0.25), [0.0, 2.25, -1.0])

    components = SphericalDipoleField.from_components(
        grid,
        1,
        gauge="length",
        x=lambda t: t,
        z=lambda t: 2.0 * t,
    )
    np.testing.assert_allclose(components.field_vector(0.25), [0.25, 0.0, 0.5])


def test_electromagnetic_fields_profile_and_single_orbital_shape(grid):
    pulse = SineSquareLaser(
        E0=0.4,
        omega=1.3,
        ncycles=2,
        gauge="length",
        phase=0.2,
        t0=0.1,
    )
    operator = SphericalDipoleField.from_profile(
        grid, 1, pulse, (1.0, 2.0, 0.0), gauge="length"
    )
    time = 0.7
    np.testing.assert_allclose(
        operator.field_vector(time), pulse(time) * np.array([1.0, 2.0, 0.0])
    )
    orbital = np.ones(operator.orbital_shape)
    original = orbital.copy()
    assert operator(orbital, time).shape == orbital.shape
    np.testing.assert_array_equal(orbital, original)


def test_length_and_velocity_actions_are_hermitian_in_quadrature_metric(grid):
    rng = np.random.default_rng(9)
    for gauge in ("length", "velocity"):
        operator = SphericalDipoleField.from_profile(
            grid,
            2,
            _constant(0.7),
            (1.0, -0.3, 0.8),
            gauge=gauge,
            charge=1.4,
            mass=2.1,
        )
        left = rng.normal(size=operator.orbital_shape) + 1j * rng.normal(
            size=operator.orbital_shape
        )
        right = rng.normal(size=operator.orbital_shape) + 1j * rng.normal(
            size=operator.orbital_shape
        )
        weights = operator.radial_weights[:, None]
        lhs = np.sum(weights * left.conj() * operator.apply(right, 0.2))
        rhs = np.sum(weights * operator.apply(left, 0.2).conj() * right)
        np.testing.assert_allclose(lhs, rhs, atol=2e-11, rtol=2e-12)


@pytest.mark.parametrize("gauge", ["", "Length", "acceleration"])
def test_rejects_invalid_gauge(grid, gauge):
    with pytest.raises(ValueError, match="gauge"):
        SphericalDipoleField.from_profile(grid, 1, _constant(1), gauge=gauge)


@pytest.mark.parametrize("polarization", [(0, 0, 0), (1, 2), (1, 2, np.nan), (1, 0, 1j)])
def test_rejects_invalid_polarization(polarization):
    with pytest.raises(ValueError, match="polarization"):
        DipoleFieldTerm(_constant(1), polarization)


def test_rejects_invalid_fields_and_profile_values(grid):
    with pytest.raises(ValueError, match="at least one"):
        SphericalDipoleField.from_components(grid, 1, gauge="length")
    with pytest.raises(TypeError, match="DipoleFieldTerm"):
        SphericalDipoleField(grid, 1, gauge="length", fields=[_constant(1)])
    operator = SphericalDipoleField.from_profile(
        grid, 1, lambda time: np.ones(2), gauge="length"
    )
    with pytest.raises(ValueError, match="finite real scalar"):
        operator.field_vector(0.0)


@pytest.mark.parametrize("charge", [np.inf, np.nan, 1j, True])
def test_rejects_invalid_charge(grid, charge):
    with pytest.raises((TypeError, ValueError), match="charge"):
        SphericalDipoleField.from_profile(
            grid, 1, _constant(1), gauge="length", charge=charge
        )


def test_rejects_invalid_d1(grid):
    bad_grid = SimpleNamespace(r=grid.r, weights=grid.weights, D1=np.eye(3))
    with pytest.raises(ValueError, match="D1"):
        SphericalDipoleField.from_profile(
            bad_grid, 1, _constant(1), gauge="length"
        )
