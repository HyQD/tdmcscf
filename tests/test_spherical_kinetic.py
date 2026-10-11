from types import SimpleNamespace

import numpy as np
import pytest

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.angular_momentum import LM_to_I

from tdmcscf.grid.spherical import SphericalKineticEnergy


@pytest.fixture(scope="module")
def grid():
    return FEMDVR(
        np.array([0.0, 1.0, 3.0, 6.0]),
        np.array([9, 10, 11]),
        Linear_map,
        GaussLegendreLobatto,
        symmetrize=False,
    )


@pytest.mark.parametrize("mass", [1.0, 2.5])
def test_analytic_polynomial_action_in_every_channel(grid, mass):
    kinetic = SphericalKineticEnergy(grid, l_max=3, mass=mass)
    r = kinetic.r
    u = np.column_stack([r ** (l + 1) * (6 - r) for l, m in kinetic.orbital_lm])
    expected = np.column_stack([(l + 1) * r**l / mass for l, m in kinetic.orbital_lm])
    # These smooth polynomials vanish at both boundaries. This check includes
    # finite-element bridge nodes, where the assembled derivative must agree.
    np.testing.assert_allclose(kinetic.apply(u), expected, atol=4e-10, rtol=1e-10)
    batch = np.array([u, (1 + 2j) * u])
    np.testing.assert_allclose(
        kinetic.apply(batch),
        np.array([expected, (1 + 2j) * expected]),
        atol=1e-9,
        rtol=1e-10,
    )


def test_mass_scaling_and_weighted_hermiticity(grid):
    unit = SphericalKineticEnergy(grid, l_max=2)
    heavy = SphericalKineticEnergy(grid, l_max=2, mass=7.3)
    rng = np.random.default_rng(7)
    a = rng.normal(size=unit.orbital_shape) + 1j * rng.normal(size=unit.orbital_shape)
    b = rng.normal(size=unit.orbital_shape) + 1j * rng.normal(size=unit.orbital_shape)
    original = a.copy()
    np.testing.assert_allclose(heavy.apply(a), unit.apply(a) / 7.3, atol=1e-11)
    w = unit.radial_weights[:, None]
    np.testing.assert_allclose(
        np.sum(w * a.conj() * heavy.apply(b)),
        np.sum(w * heavy.apply(a).conj() * b),
        atol=1e-10,
        rtol=1e-12,
    )
    energy = np.sum(w * a.conj() * heavy.apply(a))
    assert energy.real > 0
    assert abs(energy.imag) < 1e-10
    np.testing.assert_array_equal(a, original)


def test_dirichlet_box_spectrum(grid):
    mass = 2.3
    kinetic = SphericalKineticEnergy(grid, l_max=0, mass=mass)
    # Apply to unit nodal vectors, then transform only for this eigensolve
    # to the Euclidean metric using the quadrature weights.
    n = len(kinetic.r)
    matrix = kinetic.apply(np.eye(n)[:, :, None])[:, :, 0].T
    root_w = np.sqrt(kinetic.radial_weights)
    symmetric = root_w[:, None] * matrix / root_w[None, :]
    np.testing.assert_allclose(symmetric, symmetric.T, atol=1e-12)
    energies = np.linalg.eigvalsh(symmetric)
    expected = (np.arange(1, 5) * np.pi / 6) ** 2 / (2 * mass)
    np.testing.assert_allclose(energies[:4], expected, atol=1e-10, rtol=2e-8)


def test_fixed_m_channels_and_grid_lib_ordering(grid):
    kinetic = SphericalKineticEnergy(grid, l_max=3, m_max=1)
    for i, (l, m) in enumerate(kinetic.orbital_lm):
        assert LM_to_I(l, m, 3, 1) == i
    u = np.zeros(kinetic.orbital_shape, dtype=complex)
    channel = kinetic.orbital_lm.index((2, -1))
    u[:, channel] = (1 + 1j) * kinetic.r**3 * (6 - kinetic.r)
    result = kinetic.apply(u)
    np.testing.assert_allclose(
        result[:, channel], (1 + 1j) * 3 * kinetic.r**2, atol=2e-10
    )
    np.testing.assert_array_equal(np.delete(result, channel, axis=1), 0)


def test_construction_needs_no_angular_or_coulomb_setup(grid, monkeypatch):
    import tdmcscf.grid.spherical.mean_fields as fields

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Kinetic energy must not construct transforms or Coulomb responses."
        )

    monkeypatch.setattr(fields, "radial_Coulomb", forbidden)
    monkeypatch.setattr(fields, "AngularTransform", forbidden)
    kinetic = SphericalKineticEnergy(grid, 2)
    assert kinetic.mass == 1
    assert kinetic.apply(np.zeros(kinetic.orbital_shape)).shape == kinetic.orbital_shape


@pytest.mark.parametrize("mass", [0.0, -1.0, np.inf, np.nan, 1j, True, [1.0]])
def test_rejects_invalid_mass(grid, mass):
    with pytest.raises((ValueError, TypeError), match="mass"):
        SphericalKineticEnergy(grid, 1, mass=mass)


def test_rejects_incompatible_grid_and_arrays(grid):
    kinetic = SphericalKineticEnergy(grid, 1)
    with pytest.raises(ValueError, match="shape"):
        kinetic.apply(np.zeros((len(grid.r), 4)))
    with pytest.raises(ValueError, match="finite"):
        kinetic.apply(np.full(kinetic.orbital_shape, np.nan))
    with pytest.raises(ValueError, match="m_max"):
        SphericalKineticEnergy(grid, 1, m_max=2)
    shifted = SimpleNamespace(r=grid.r + 1, weights=grid.weights, D2=grid.D2)
    with pytest.raises(ValueError, match="start at zero"):
        SphericalKineticEnergy(shifted, 1)
