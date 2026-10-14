import numpy as np
import pytest

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)

from tdmcscf.grid.spherical import (
    AtomicCoreHamiltonian,
    ShiftedAtomicCorePreconditioner,
    SphericalDipoleField,
    SphericalMeanFieldBuilder,
    SphericalRestrictedTDHF,
)
from tdmcscf.integrators import CrankNicolson, RK4, Strang2


@pytest.fixture(scope="module")
def radial_grid():
    return FEMDVR(
        np.array([0.0, 2.0, 6.0]),
        np.array([9, 10]),
        Linear_map,
        GaussLegendreLobatto,
        symmetrize=False,
    )


def normalized_orbitals(core, count=1, seed=8):
    rng = np.random.default_rng(seed)
    shape = (count,) + core.orbital_shape
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    values *= (core.r * np.exp(-core.r / 2))[None, :, None]
    sqrt_weights = np.repeat(np.sqrt(core.radial_weights), shape[-1])
    weighted = values.reshape(count, -1).T * sqrt_weights[:, None]
    orthogonal, _ = np.linalg.qr(weighted)
    return (orthogonal / sqrt_weights[:, None]).T.reshape(shape)


def test_closed_shell_mean_field_is_2J_minus_K(radial_grid):
    core = AtomicCoreHamiltonian(radial_grid, 0, Z=2)
    builder = SphericalMeanFieldBuilder(
        radial_grid, 0, L_max=0, M_max=0
    )
    tdhf = SphericalRestrictedTDHF(core, builder, orbital_m=[0])
    orbital = normalized_orbitals(core)
    fields = builder.build(orbital, orbital_m=[0])
    expected = fields.apply(orbital, (0, 0), orbital_m=[0])
    np.testing.assert_allclose(tdhf.mean_field_action(orbital), expected, atol=2e-13)


def test_canonical_and_orthogonal_gauge_equations(radial_grid):
    core = AtomicCoreHamiltonian(radial_grid, 1, m_max=0, Z=2)
    builder = SphericalMeanFieldBuilder(
        radial_grid, 1, m_max=0, L_max=2, M_max=0
    )
    orbitals = normalized_orbitals(core, count=2)
    canonical = SphericalRestrictedTDHF(
        core, builder, orbital_gauge="canonical"
    )
    orthogonal = SphericalRestrictedTDHF(
        core, builder, orbital_gauge="orthogonal"
    )
    time = 0.4

    fock = canonical.fock_action(orbitals, time)
    canonical_rhs = canonical.linear_rhs(orbitals, time) + canonical.nonlinear_rhs(
        orbitals, time
    )
    np.testing.assert_allclose(canonical_rhs, -1j * fock, atol=3e-12)

    occupied_fock = orthogonal.overlaps(orbitals, fock)
    projected = fock - np.einsum(
        "qai,qp->pai", orbitals, occupied_fock, optimize=True
    )
    orthogonal_rhs = orthogonal.linear_rhs(
        orbitals, time
    ) + orthogonal.nonlinear_rhs(orbitals, time)
    np.testing.assert_allclose(orthogonal_rhs, -1j * projected, atol=3e-12)
    np.testing.assert_allclose(
        orthogonal.overlaps(orbitals, orthogonal_rhs), 0, atol=3e-12
    )


def test_orbital_norms_and_fixed_m_validation(radial_grid):
    core = AtomicCoreHamiltonian(radial_grid, 1, m_max=0, Z=2)
    builder = SphericalMeanFieldBuilder(
        radial_grid, 1, m_max=0, L_max=2, M_max=0
    )
    orbitals = normalized_orbitals(core, count=2)
    tdhf = SphericalRestrictedTDHF(core, builder, orbital_m=[0, 0])
    np.testing.assert_allclose(tdhf.orbital_norms(orbitals), 1, atol=2e-13)
    with pytest.raises(ValueError, match="one value"):
        SphericalRestrictedTDHF(core, builder, orbital_m=[0]).mean_field_action(
            orbitals
        )


def test_shifted_core_preconditioner_matches_channel_solves(radial_grid):
    core = AtomicCoreHamiltonian(radial_grid, 2, m_max=1, Z=3)
    step_size = 0.07
    preconditioner = ShiftedAtomicCorePreconditioner(core, step_size)
    orbitals = normalized_orbitals(core, count=2)
    result = preconditioner(orbitals)
    expected = np.empty_like(orbitals)
    identity = np.eye(len(core.r))
    for channel, (l, _) in enumerate(core.orbital_lm):
        matrix = identity + 0.5j * step_size * core.radial_matrix(l)
        expected[:, :, channel] = np.linalg.solve(
            matrix, orbitals[:, :, channel].T
        ).T
    np.testing.assert_allclose(result, expected, atol=2e-13, rtol=2e-13)


def test_one_strang_step_couples_s_and_p_channels_and_preserves_norm(radial_grid):
    core = AtomicCoreHamiltonian(radial_grid, 1, m_max=0, Z=2)
    builder = SphericalMeanFieldBuilder(
        radial_grid, 1, m_max=0, L_max=2, M_max=0
    )
    field = SphericalDipoleField.from_profile(
        radial_grid,
        1,
        lambda time: 1e-3,
        polarization=(0.0, 0.0, 1.0),
        m_max=0,
        gauge="length",
    )
    tdhf = SphericalRestrictedTDHF(
        core,
        builder,
        dipole_field=field,
        orbital_gauge="orthogonal",
        orbital_m=[0],
    )
    orbitals = np.zeros((1,) + core.orbital_shape, dtype=complex)
    s_channel = core.orbital_lm.index((0, 0))
    p_channel = core.orbital_lm.index((1, 0))
    orbitals[0, :, s_channel] = core.r * np.exp(-core.r)
    orbitals /= np.sqrt(tdhf.orbital_norms(orbitals))[0]

    step_size = 0.01
    preconditioner = ShiftedAtomicCorePreconditioner(core, step_size / 2)
    integrator = Strang2(
        CrankNicolson(
            rtol=1e-12,
            atol=1e-14,
            preconditioner=preconditioner,
        ),
        RK4(),
    )
    propagated = integrator(
        orbitals,
        0.0,
        step_size,
        tdhf.linear_rhs,
        tdhf.nonlinear_rhs,
    )
    assert np.linalg.norm(propagated[0, :, p_channel]) > 0
    np.testing.assert_allclose(tdhf.orbital_norms(propagated), 1, atol=2e-11)


@pytest.mark.parametrize("gauge", ["bad", None, 1])
def test_rejects_invalid_orbital_gauge(radial_grid, gauge):
    core = AtomicCoreHamiltonian(radial_grid, 0, Z=2)
    builder = SphericalMeanFieldBuilder(radial_grid, 0)
    with pytest.raises(ValueError, match="orbital_gauge"):
        SphericalRestrictedTDHF(core, builder, orbital_gauge=gauge)
