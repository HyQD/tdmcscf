import numpy as np
import pytest
from scipy.linalg import eigh

from grid_lib.pseudospectral_grids import (
    FEMDVR,
    GaussLegendreLobatto,
    Linear_map,
)

from tdmcscf.grid.spherical import AxialCoreHamiltonian
from tdmcscf.systems import AxialDiatomic


grid_tdse = pytest.importorskip("grid_tdse.spherical.one_radial")


@pytest.mark.parametrize("m", [-1, 0, 1])
def test_fixed_m_operator_and_states_match_grid_tdse(m):
    nodes = np.asarray([0.0, 1.0, 3.0, 6.0])
    n_points = np.asarray([8, 8, 8])
    l_max = 3
    m_max = 1
    system = AxialDiatomic("H", "He", half_bond_length=1.0, charge=2)
    grid = FEMDVR(
        nodes,
        n_points,
        Linear_map,
        GaussLegendreLobatto,
        symmetrize=False,
    )
    core = AxialCoreHamiltonian(
        grid,
        l_max,
        m_max,
        system=system,
        nuclear_L_max=2 * l_max,
    )
    reference_system = grid_tdse.OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=2.0,
        half_bond_length=1.0,
        particle_mass=1.0,
    )
    reference = grid_tdse.OneElectronDiatomicHamiltonian(
        reference_system,
        nodes=nodes,
        n_points=n_points,
        l_max=l_max,
        m_max=m_max,
        nuclear_L_max=2 * l_max,
    )

    np.testing.assert_allclose(
        core.fixed_m_matrix(m),
        reference.fixed_m_matrix(m),
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    core_symmetric = core.fixed_m_matrix(m, representation="symmetric")
    reference_symmetric = reference.fixed_m_matrix(
        m, representation="symmetric"
    )
    core_energies, core_vectors = eigh(core_symmetric)
    reference_energies, reference_vectors = eigh(reference_symmetric)
    np.testing.assert_allclose(
        core_energies[:3],
        reference_energies[:3],
        atol=3.0e-13,
        rtol=3.0e-13,
    )

    for index in range(3):
        overlap = np.vdot(reference_vectors[:, index], core_vectors[:, index])
        phase = 1.0 if overlap == 0.0 else overlap.conjugate() / abs(overlap)
        np.testing.assert_allclose(
            phase * core_vectors[:, index],
            reference_vectors[:, index],
            atol=2.0e-11,
            rtol=2.0e-11,
        )
