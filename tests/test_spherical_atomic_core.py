from types import SimpleNamespace

import numpy as np
import pytest

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.angular_momentum import LM_to_I

from tdmcscf.grid.spherical import AtomicCoreHamiltonian, SphericalCoulombPotential


@pytest.fixture(scope="module")
def grid():
    return FEMDVR(
        np.array([0.0, 1.0, 3.0, 6.0]),
        np.array([9, 10, 11]),
        Linear_map,
        GaussLegendreLobatto,
        symmetrize=False,
    )


def test_potential_is_nodal_multiplication_without_derivatives(grid):
    radial = SimpleNamespace(r=grid.r, weights=grid.weights)
    with np.errstate(divide="raise", invalid="raise"):
        potential = SphericalCoulombPotential(radial, 3, m_max=1, Z=1.7)
    r = potential.r
    u = np.column_stack([r ** (l + 1) * (6 - r) for l, _ in potential.orbital_lm])
    expected = np.column_stack([-1.7 * r**l * (6 - r) for l, _ in potential.orbital_lm])
    original = u.copy()
    np.testing.assert_allclose(potential.values, -1.7 / r)
    np.testing.assert_allclose(potential.apply(u), expected)
    np.testing.assert_allclose(
        potential.apply(np.array([u, (1 + 2j) * u])),
        np.array([expected, (1 + 2j) * expected]),
    )
    np.testing.assert_array_equal(u, original)
    assert potential.Z == 1.7
    assert potential.apply(u).dtype == np.complex128


@pytest.mark.parametrize("m_max", [None, 0, 1])
def test_core_action_matrices_and_channel_order(grid, m_max):
    core = AtomicCoreHamiltonian(grid, 3, m_max, Z=2.3, mass=1.7)
    rng = np.random.default_rng(12)
    u = rng.normal(size=(2, *core.orbital_shape)) + 1j * rng.normal(
        size=(2, *core.orbital_shape)
    )
    original = u.copy()
    result = core.apply(u)
    np.testing.assert_allclose(result, core.kinetic.apply(u) + core.potential.apply(u))
    np.testing.assert_allclose(core.apply(u[0]), result[0])
    for i, (l, m) in enumerate(core.orbital_lm):
        assert LM_to_I(l, m, core.l_max, core.m_max) == i
        matrix = core.radial_matrix(l)
        np.testing.assert_allclose(result[:, :, i], u[:, :, i] @ matrix.T, atol=1e-11)
        np.testing.assert_allclose(
            core.kinetic.apply(u)[:, :, i],
            u[:, :, i] @ core.kinetic.radial_matrix(l).T,
            atol=1e-11,
        )
        metric_matrix = core.radial_weights[:, None] * matrix
        np.testing.assert_allclose(metric_matrix, metric_matrix.T, atol=1e-11)
    # A fixed-m orbital stays in its channel.
    fixed = np.zeros(core.orbital_shape, dtype=complex)
    fixed[:, 0] = u[0, :, 0]
    np.testing.assert_array_equal(core.apply(fixed)[:, 1:], 0)
    # Returned matrices can be modified without changing either operator.
    core.radial_matrix(0)[:] = 0
    core.kinetic.radial_matrix(0)[:] = 0
    np.testing.assert_array_equal(core.apply(u), result)
    np.testing.assert_array_equal(u, original)


@pytest.mark.parametrize("mass,Z", [(1.0, 1.0), (2.5, 1.7)])
def test_hydrogenic_energies_and_nodal_eigenvectors(mass, Z):
    grid = FEMDVR(
        np.array([0.0, 1.0, 3.0, 7.0, 15.0, 30.0, 60.0, 100.0]) / (mass * Z),
        np.full(7, 17),
        Linear_map,
        GaussLegendreLobatto,
        symmetrize=False,
    )
    core = AtomicCoreHamiltonian(grid, 2, Z=Z, mass=mass)
    root_w = np.sqrt(core.radial_weights)
    for l in range(3):
        matrix = core.radial_matrix(l)
        symmetric = root_w[:, None] * matrix / root_w[None, :]
        np.testing.assert_allclose(symmetric, symmetric.T, atol=1e-10)
        energies, vectors = np.linalg.eigh(symmetric)
        n = np.arange(l + 1, l + 3)
        np.testing.assert_allclose(energies[:2], -mass * Z**2 / (2 * n**2), atol=2e-10)
        nodal = vectors[:, :2] / root_w[:, None]
        np.testing.assert_allclose(
            nodal.T @ (core.radial_weights[:, None] * nodal), np.eye(2), atol=1e-12
        )
        u = np.zeros((2, *core.orbital_shape), dtype=complex)
        u[:, :, core.orbital_lm.index((l, 0))] = nodal.T
        np.testing.assert_allclose(
            core.apply(u), energies[:2, None, None] * u, atol=2e-10
        )


def test_mass_scales_only_kinetic_energy(grid):
    unit = AtomicCoreHamiltonian(grid, 1, Z=2)
    heavy = AtomicCoreHamiltonian(grid, 1, Z=2, mass=3)
    assert unit.mass == 1
    assert unit.Z == 2
    np.testing.assert_array_equal(unit.potential.values, heavy.potential.values)
    np.testing.assert_allclose(
        heavy.radial_matrix(1),
        unit.kinetic.radial_matrix(1) / 3 + np.diag(unit.potential.values),
    )


@pytest.mark.parametrize("m", [-1, 0, 1])
def test_central_core_implements_fixed_m_core_protocol(grid, m):
    core = AtomicCoreHamiltonian(grid, 3, m_max=2, Z=2.3)
    rng = np.random.default_rng(14 + m)
    shape = core.fixed_m_shape(m)
    orbital = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    expected = (core.fixed_m_matrix(m) @ orbital.ravel()).reshape(shape)

    np.testing.assert_allclose(core.apply_fixed_m(orbital, m), expected)
    symmetric = core.fixed_m_matrix(m, representation="symmetric")
    np.testing.assert_allclose(symmetric, symmetric.conj().T, atol=2.0e-12)
    transformed = core.to_symmetric_fixed_m(orbital, m)
    np.testing.assert_allclose(
        core.from_symmetric_fixed_m(transformed, m),
        orbital,
    )


@pytest.mark.parametrize("Z", [0, -1, np.nan, np.inf, 1j, True, [1.0]])
def test_invalid_charge(grid, Z):
    for operator in (SphericalCoulombPotential, AtomicCoreHamiltonian):
        with pytest.raises((TypeError, ValueError), match="Z"):
            operator(grid, 1, Z=Z)


def test_invalid_inputs(grid):
    core = AtomicCoreHamiltonian(grid, 1, Z=1)
    for operator in (core, core.kinetic):
        for l in (-1, 2, 0.5, True):
            with pytest.raises((TypeError, ValueError), match="l"):
                operator.radial_matrix(l)
    for operator in (core, core.potential):
        with pytest.raises(ValueError, match="shape"):
            operator.apply(np.zeros((len(grid.r), 4)))
        with pytest.raises(ValueError, match="finite"):
            operator.apply(np.full(core.orbital_shape, np.nan))
    with pytest.raises(ValueError, match="mass"):
        AtomicCoreHamiltonian(grid, 1, Z=1, mass=0)
    with pytest.raises(TypeError, match="Z"):
        AtomicCoreHamiltonian(grid, 1)
    with pytest.raises(ValueError, match="m_max"):
        SphericalCoulombPotential(grid, 1, 2, Z=1)
    shifted = SimpleNamespace(r=grid.r + 1, weights=grid.weights)
    with pytest.raises(ValueError, match="start at zero"):
        SphericalCoulombPotential(shifted, 1, Z=1)
