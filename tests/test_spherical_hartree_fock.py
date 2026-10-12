import numpy as np
import pytest

from tdmcscf.grid.spherical import (
    SphericalAtomicHartreeFock,
    make_atomic_femdvr_grid,
)


def test_grid_accepts_explicit_and_uniform_elements():
    explicit = make_atomic_femdvr_grid(r_max=8, nodes=[0, 1, 3, 8], n_points=[7, 8, 9])
    np.testing.assert_array_equal(explicit.nodes, [0, 1, 3, 8])
    np.testing.assert_array_equal(explicit.n_points, [7, 8, 9])
    uniform = make_atomic_femdvr_grid(r_max=8, n_elements=4, n_points=7)
    np.testing.assert_allclose(uniform.nodes, np.linspace(0, 8, 5))
    np.testing.assert_array_equal(uniform.n_points, np.full(4, 7))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"nodes": [0, 2, 1], "n_points": 7}, "nodes"),
        ({"r_max": 9, "nodes": [0, 2, 8], "n_points": 7}, "r_max"),
        ({"r_max": 8, "n_elements": 3, "n_points": [7, 7]}, "n_points"),
        ({"r_max": 8, "n_elements": 3, "n_points": 2}, "three"),
    ],
)
def test_grid_rejects_inconsistent_controls(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        make_atomic_femdvr_grid(**kwargs)


@pytest.mark.parametrize(
    "atom,nodes,n_points,reference",
    [
        ("He", [0, 1, 3, 7, 15, 30], 15, -2.861679995612),
        ("Be", [0, 0.5, 1.5, 4, 10, 25], 17, -14.573023168316),
        ("Ne", [0, 0.3, 1, 3, 7, 15, 30], 17, -128.547098109378),
    ],
)
def test_closed_shell_atomic_reference_energies(atom, nodes, n_points, reference):
    result = SphericalAtomicHartreeFock(
        atom,
        nodes=nodes,
        n_points=n_points,
        max_iterations=40,
        conv_tol_e=1e-10,
        conv_tol_grad=1e-8,
    ).solve()
    assert result.converged
    assert result.iterations < 20
    assert result.energy == pytest.approx(reference, abs=2e-8)
    assert result.history[-1].energy_change < 1e-10
    assert result.history[-1].gradient_norm < 1e-8


def test_result_is_weighted_orthonormal_and_embedded_in_requested_basis():
    solver = SphericalAtomicHartreeFock(
        "Ne",
        nodes=[0, 0.5, 2, 6, 15, 30],
        n_points=13,
        l_max=3,
        m_max=2,
        coulomb_l_max=2,
        max_iterations=40,
    )
    result = solver.solve()
    weights = solver.radial_weights
    overlap = np.einsum(
        "paI,qaI,a->pq",
        result.orbitals.conj(),
        result.orbitals,
        weights,
    )
    np.testing.assert_allclose(overlap, np.eye(5), atol=2e-12)
    assert result.orbitals.shape == (5, *solver.orbital_shape)
    assert result.orbital_quantum_numbers == (
        (1, 0, 0),
        (2, 0, 0),
        (2, 1, -1),
        (2, 1, 0),
        (2, 1, 1),
    )
    radial_row = {(1, 0): 0, (2, 0): 1, (2, 1): 0}
    for orbital, (n, l, m) in zip(result.orbitals, result.orbital_quantum_numbers):
        channel = solver.orbital_lm.index((l, m))
        np.testing.assert_allclose(
            orbital[:, channel], result.radial_orbitals[l][radial_row[n, l]]
        )
        np.testing.assert_array_equal(np.delete(orbital, channel, axis=1), 0)
    assert result.l_max == 3
    assert result.m_max == 2
    assert result.coulomb_l_max == 2
    for matrix in solver.fock_matrices(result.radial_orbitals).values():
        weighted = solver.radial_weights[:, None] * matrix
        np.testing.assert_allclose(weighted, weighted.T.conj(), atol=2e-11)


def test_diis_is_optional():
    solver = SphericalAtomicHartreeFock(
        "He",
        nodes=[0, 1, 4, 12, 30],
        n_points=13,
        diis=False,
        damping=0.35,
        max_iterations=80,
    )
    result = solver.solve()
    assert result.converged


def test_rejects_basis_that_omits_occupied_channels():
    with pytest.raises(ValueError, match="occupied"):
        SphericalAtomicHartreeFock("Ne", r_max=20, n_points=9, m_max=0)
