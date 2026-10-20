import numpy as np
import pytest

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from integrator_lib import CrankNicolson

from tdmcscf.dft import LDAVWN, PBE
from tdmcscf.grid.spherical import (
    AtomicCoreHamiltonian,
    SphericalMeanFieldBuilder,
    SphericalRestrictedTDKS,
    SphericalXCBuilder,
)


@pytest.fixture(scope="module")
def radial_grid():
    return FEMDVR(
        np.array([0.0, 2.0, 6.0]),
        np.array([9, 10]),
        Linear_map,
        GaussLegendreLobatto,
        symmetrize=False,
    )


def normalized_orbitals(core, count=1, seed=29):
    rng = np.random.default_rng(seed)
    shape = (count,) + core.orbital_shape
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    values *= (core.r * np.exp(-core.r / 2))[None, :, None]
    sqrt_weights = np.repeat(np.sqrt(core.radial_weights), shape[-1])
    weighted = values.reshape(count, -1).T * sqrt_weights[:, None]
    orthogonal, _ = np.linalg.qr(weighted)
    return (orthogonal / sqrt_weights[:, None]).T.reshape(shape)


def equation(radial_grid, gauge="canonical", functional=None):
    core = AtomicCoreHamiltonian(radial_grid, 1, m_max=0, Z=2)
    hartree = SphericalMeanFieldBuilder(
        radial_grid, 1, m_max=0, L_max=2, M_max=0
    )
    xc = SphericalXCBuilder(
        radial_grid,
        1,
        m_max=0,
        functional=LDAVWN() if functional is None else functional,
    )
    return SphericalRestrictedTDKS(
        core,
        hartree,
        xc,
        orbital_gauge=gauge,
        orbital_m=[0],
    )


def test_mean_field_is_hartree_plus_local_xc(radial_grid):
    tdks = equation(radial_grid)
    orbitals = normalized_orbitals(tdks.core)
    fields = tdks._hartree_fields(orbitals)
    expected = tdks._hartree_action(orbitals, fields)
    expected += tdks.xc.build(orbitals).apply(orbitals)
    np.testing.assert_allclose(tdks.mean_field_action(orbitals), expected, atol=3e-12)


@pytest.mark.parametrize("gauge", ["canonical", "orthogonal"])
def test_complete_tdks_equation_in_each_orbital_gauge(radial_grid, gauge):
    tdks = equation(radial_grid, gauge)
    orbitals = normalized_orbitals(tdks.core)
    time = 0.3
    hamiltonian = tdks.hamiltonian_action(orbitals, time)
    rhs = tdks.linear_rhs(orbitals, time) + tdks.nonlinear_rhs(orbitals, time)
    if gauge == "canonical":
        expected = -1j * hamiltonian
    else:
        occupied = tdks.overlaps(orbitals, hamiltonian)
        projected = hamiltonian - np.einsum(
            "qai,qp->pai", orbitals, occupied, optimize=True
        )
        expected = -1j * projected
        np.testing.assert_allclose(
            tdks.overlaps(orbitals, rhs), 0.0, atol=3e-12
        )
    np.testing.assert_allclose(rhs, expected, atol=3e-12)


@pytest.mark.parametrize("gauge", ["canonical", "orthogonal"])
def test_frozen_density_rhs_is_linear_and_matches_reference(radial_grid, gauge):
    tdks = equation(radial_grid, gauge)
    reference = normalized_orbitals(tdks.core)
    trial_a = normalized_orbitals(tdks.core, seed=31)
    trial_b = normalized_orbitals(tdks.core, seed=37)
    alpha = 0.3 - 0.2j
    beta = -0.4 + 0.1j
    time = 0.2
    frozen = tdks.freeze_rhs(reference)
    np.testing.assert_allclose(
        frozen(alpha * trial_a + beta * trial_b, time),
        alpha * frozen(trial_a, time) + beta * frozen(trial_b, time),
        atol=4e-12,
    )
    full_rhs = tdks.linear_rhs(reference, time) + tdks.nonlinear_rhs(
        reference, time
    )
    np.testing.assert_allclose(frozen(reference, time), full_rhs, atol=4e-12)
    np.testing.assert_allclose(
        tdks.freeze_density_rhs(reference)(trial_a, time),
        frozen(trial_a, time),
        atol=4e-12,
    )


def test_one_density_frozen_midpoint_step_preserves_norm(radial_grid):
    tdks = equation(radial_grid, "canonical")
    orbitals = normalized_orbitals(tdks.core)
    propagated = CrankNicolson(rtol=1e-12, atol=1e-14)(
        orbitals,
        0.0,
        0.01,
        tdks.freeze_rhs(orbitals),
    )
    np.testing.assert_allclose(tdks.orbital_norms(propagated), 1.0, atol=3e-11)


def test_fixed_m_frozen_rhs_accepts_full_space_krylov_vectors(radial_grid):
    core = AtomicCoreHamiltonian(radial_grid, 1, m_max=1, Z=2)
    hartree = SphericalMeanFieldBuilder(
        radial_grid, 1, m_max=1, L_max=2, M_max=2
    )
    xc = SphericalXCBuilder(
        radial_grid, 1, m_max=1, functional=LDAVWN()
    )
    tdks = SphericalRestrictedTDKS(
        core,
        hartree,
        xc,
        orbital_gauge="canonical",
        orbital_m=[0],
    )
    reference = normalized_orbitals(core)
    channel_m = np.array([m for _, m in core.orbital_lm])
    reference[:, :, channel_m != 0] = 0.0
    reference /= np.sqrt(tdks.orbital_norms(reference))[:, None, None]

    frozen = tdks.freeze_rhs(reference)
    full_space_trial = normalized_orbitals(core, seed=43)
    projected_trial = full_space_trial.copy()
    projected_trial[:, :, channel_m != 0] = 0.0
    actual = frozen(full_space_trial, 0.2)
    expected = frozen(projected_trial, 0.2)
    np.testing.assert_allclose(actual, expected, atol=3e-12)
    np.testing.assert_array_equal(actual[:, :, channel_m != 0], 0.0)

    propagated = CrankNicolson(rtol=1e-12, atol=1e-14)(
        reference,
        0.0,
        0.01,
        frozen,
    )
    np.testing.assert_array_equal(propagated[:, :, channel_m != 0], 0.0)
    np.testing.assert_allclose(tdks.orbital_norms(propagated), 1.0, atol=3e-11)


def test_pbe_instantaneous_and_frozen_actions_are_finite(radial_grid):
    tdks = equation(radial_grid, functional=PBE())
    orbitals = normalized_orbitals(tdks.core)
    instantaneous = tdks.mean_field_action(orbitals)
    frozen = tdks.freeze_rhs(orbitals)(orbitals, 0.2)
    assert np.all(np.isfinite(instantaneous))
    assert np.all(np.isfinite(frozen))
    np.testing.assert_allclose(
        frozen,
        tdks.linear_rhs(orbitals, 0.2) + tdks.nonlinear_rhs(orbitals, 0.2),
        atol=5e-12,
    )
