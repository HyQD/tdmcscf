import numpy as np
import pytest

from tdmcscf.grid.spherical import (
    FixedDensityAxialFock,
    SphericalDiatomicHartreeFock,
    SphericalRestrictedTDHF,
)
from tdmcscf.systems import AxialDiatomic


def _solver(
    *,
    system=None,
    occupation=None,
    l_max=2,
    m_max=0,
    points=7,
    **options,
):
    if system is None:
        system = AxialDiatomic("H", "H", half_bond_length=0.7)
    if occupation is None:
        occupation = {0: 1}
    parameters = dict(
        nodes=[0.0, system.half_bond_length, 2.0, 5.0, 10.0],
        n_points=points,
        l_max=l_max,
        m_max=m_max,
        max_iterations=20,
        conv_tol_grad=1.0e-7,
        eigensolver_tolerance=1.0e-10,
    )
    parameters.update(options)
    return SphericalDiatomicHartreeFock(
        system,
        occupation,
        **parameters,
    )


def test_fixed_density_fock_matches_tdhf_action_on_occupied_orbitals():
    solver = _solver(max_iterations=1)
    symmetric = solver._initial_orbitals()
    occupied, orbital_m = solver._full_nodal(symmetric)
    fock = FixedDensityAxialFock(
        solver.core,
        solver.mean_fields,
        occupied,
        orbital_m,
    )
    tdhf = SphericalRestrictedTDHF(
        solver.core,
        solver.mean_fields,
        orbital_gauge="canonical",
        orbital_m=orbital_m,
    )

    np.testing.assert_allclose(
        fock.apply(occupied, orbital_m),
        tdhf.fock_action(occupied, 0.0),
        atol=4.0e-12,
        rtol=4.0e-12,
    )


def test_fixed_density_fock_matches_tdhf_for_multiple_signed_m_sectors():
    system = AxialDiatomic("Li", "Li", half_bond_length=0.7)
    solver = _solver(
        system=system,
        occupation={0: 1, 1: 1},
        l_max=1,
        m_max=1,
        points=5,
        max_iterations=1,
    )
    symmetric = solver._initial_orbitals()
    occupied, orbital_m = solver._full_nodal(symmetric)
    fock = FixedDensityAxialFock(
        solver.core,
        solver.mean_fields,
        occupied,
        orbital_m,
    )
    tdhf = SphericalRestrictedTDHF(
        solver.core,
        solver.mean_fields,
        orbital_gauge="canonical",
        orbital_m=orbital_m,
    )

    assert orbital_m == (-1, 0, 1)
    np.testing.assert_allclose(
        fock.apply(occupied, orbital_m),
        tdhf.fock_action(occupied, 0.0),
        atol=5.0e-11,
        rtol=5.0e-11,
    )


def test_fixed_density_fock_matches_tdhf_density_frozen_trial_action():
    solver = _solver(max_iterations=1)
    symmetric = solver._initial_orbitals()
    occupied, orbital_m = solver._full_nodal(symmetric)
    fock = FixedDensityAxialFock(
        solver.core,
        solver.mean_fields,
        occupied,
        orbital_m,
    )
    tdhf = SphericalRestrictedTDHF(
        solver.core,
        solver.mean_fields,
        orbital_gauge="canonical",
        orbital_m=orbital_m,
    )
    rng = np.random.default_rng(41)
    fixed_shape = solver.core.fixed_m_shape(0)
    fixed_trial = rng.normal(size=fixed_shape)
    fixed_trial = fixed_trial + 1j * rng.normal(size=fixed_shape)
    trial = np.zeros((1, *solver.orbital_shape), dtype=complex)
    indices = [
        i for i, (_, magnetic) in enumerate(solver.orbital_lm) if magnetic == 0
    ]
    trial[0][:, indices] = fixed_trial
    frozen_rhs = tdhf.freeze_density_rhs(occupied)

    np.testing.assert_allclose(
        fock.apply(trial, [0]),
        1j * frozen_rhs(trial, 0.0),
        atol=4.0e-12,
        rtol=4.0e-12,
    )


def test_h2_rhf_converges_and_returns_consistent_metadata():
    solver = _solver()
    result = solver.solve()

    assert result.converged
    assert result.iterations < 15
    assert result.energy == pytest.approx(-1.116994754073, abs=2.0e-9)
    assert result.energy == pytest.approx(
        result.electronic_energy + result.nuclear_repulsion_energy,
        abs=2.0e-13,
    )
    assert result.orbital_m == (0,)
    assert result.orbital_labels == ((0, 0),)
    assert result.diagonalization == "matrix-free"
    np.testing.assert_allclose(result.l_populations.sum(axis=1), 1.0)
    overlap = np.einsum(
        "pai,a,qai->pq",
        result.orbitals.conj(),
        solver.radial_weights,
        result.orbitals,
        optimize=True,
    )
    np.testing.assert_allclose(overlap, np.eye(1), atol=2.0e-12)


def test_heteronuclear_heh_plus_converges():
    system = AxialDiatomic(
        "He",
        "H",
        half_bond_length=0.75,
        charge=1,
        label="HeH+",
    )
    solver = _solver(
        system=system,
        l_max=2,
        nodes=[0.0, 0.75, 2.0, 5.0, 10.0],
    )
    result = solver.solve()

    assert result.converged
    assert result.energy == pytest.approx(-2.682744051984, abs=3.0e-9)
    assert result.nuclear_repulsion_energy == pytest.approx(4.0 / 3.0)
    assert np.any(result.l_populations[0, 1::2] > 1.0e-5)


def test_occupation_and_cutoff_validation():
    neutral_h2 = AxialDiatomic("H", "H", half_bond_length=0.7)
    with pytest.raises(ValueError, match="electron count"):
        _solver(system=neutral_h2, occupation={0: 2})

    lithium = AxialDiatomic("Li", "Li", half_bond_length=0.7)
    with pytest.raises(ValueError, match="m_max"):
        _solver(
            system=lithium,
            occupation={0: 1, 1: 1},
            l_max=2,
            m_max=0,
        )
    with pytest.raises(ValueError, match="coulomb_M_max"):
        _solver(
            system=lithium,
            occupation={0: 1, 1: 1},
            l_max=2,
            m_max=1,
            coulomb_M_max=1,
        )
