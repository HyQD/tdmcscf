import numpy as np
import pytest

from tdmcscf.dft import LDAVWN, PBE, XAlphaExchange
from tdmcscf.grid.spherical import (
    SphericalAtomicKohnSham,
    SphericalKohnSham,
)
from tdmcscf.systems import HarmonicOscillator


HELIUM_GRID = {
    "nodes": [0.0, 1.0, 4.0, 12.0, 30.0],
    "n_points": 13,
}


def test_helium_lda_vwn_ground_state_and_energy_decomposition():
    solver = SphericalAtomicKohnSham(
        "He",
        functional=LDAVWN(),
        **HELIUM_GRID,
        max_iterations=40,
        conv_tol_grad=1e-9,
    )
    result = solver.solve()

    assert result.converged
    assert result.iterations < 15
    assert result.energy == pytest.approx(-2.8348356240528, abs=2e-11)
    assert result.energy == pytest.approx(
        result.kinetic_energy
        + result.external_energy
        + result.hartree_energy
        + result.xc_energy,
        abs=2e-13,
    )
    density_xc_potential = np.dot(
        solver.radial_weights,
        result.radial_density * result.xc_potential,
    )
    eigenvalue_energy = (
        2.0 * np.sum(result.orbital_energies)
        - result.hartree_energy
        + result.xc_energy
        - density_xc_potential
    )
    assert result.energy == pytest.approx(eigenvalue_energy, abs=3e-10)
    assert np.dot(solver.radial_weights, result.radial_density) == pytest.approx(
        2.0, abs=2e-12
    )
    np.testing.assert_allclose(
        result.density,
        result.radial_density / (4.0 * np.pi * solver.r**2),
    )
    np.testing.assert_allclose(
        result.radial_density,
        solver.radial_density(result.radial_orbitals),
        atol=2e-13,
    )
    assert result.hartree_potential[-1] == pytest.approx(
        2.0 / solver.r[-1], abs=2e-12
    )
    assert result.configuration.symbol == "He"
    assert result.functional.name == "LDA-VWN5"
    assert result.initial_guess == "core"
    assert result.xc_quadrature_order is None


def test_kohn_sham_matrices_are_hermitian_in_radial_metric():
    solver = SphericalAtomicKohnSham(
        "Be", functional=XAlphaExchange(alpha=0.7), **HELIUM_GRID
    )
    initial = {
        l: values / solver._sqrt_weights[None, :]
        for l, values in solver._initial_orbitals().items()
    }
    for matrix in solver.kohn_sham_matrices(initial).values():
        weighted = solver.radial_weights[:, None] * matrix
        np.testing.assert_allclose(weighted, weighted.T.conj(), atol=2e-12)


def test_returned_orbitals_are_weighted_orthonormal_and_can_be_embedded():
    solver = SphericalAtomicKohnSham(
        "Ne",
        functional=LDAVWN(),
        nodes=[0, 0.5, 2, 6, 15, 30],
        n_points=13,
        l_max=3,
        m_max=2,
        max_iterations=40,
    )
    result = solver.solve()
    overlap = np.einsum(
        "paI,qaI,a->pq",
        result.orbitals.conj(),
        result.orbitals,
        solver.radial_weights,
    )
    np.testing.assert_allclose(overlap, np.eye(5), atol=3e-12)
    assert result.orbitals.shape == (5, *solver.orbital_shape)
    assert result.l_max == 3
    assert result.m_max == 2
    assert result.coulomb_l_max == 0


def test_generic_solver_accepts_an_arbitrary_central_system():
    system = HarmonicOscillator(0.5, occupation={0: 1})
    result = SphericalKohnSham(
        system,
        functional=XAlphaExchange(alpha=0.0),
        nodes=[0, 1, 3, 7, 14],
        n_points=11,
        max_iterations=50,
    ).solve()
    assert result.converged
    assert result.system is system
    assert np.isfinite(result.energy)


def test_poisson_method_is_selectable():
    common = {
        "functional": LDAVWN(),
        **HELIUM_GRID,
        "max_iterations": 40,
    }
    inverse = SphericalAtomicKohnSham("He", **common, poisson_method="inverse")
    direct = SphericalAtomicKohnSham("He", **common, poisson_method="solve")
    np.testing.assert_allclose(
        direct.radial_coulomb, inverse.radial_coulomb, atol=1e-12, rtol=1e-12
    )
    assert direct.solve().energy == pytest.approx(inverse.solve().energy, abs=2e-12)


def test_solver_rejects_an_invalid_functional():
    with pytest.raises(TypeError, match="functional"):
        SphericalAtomicKohnSham(
            "He", functional=object(), r_max=10, n_elements=2, n_points=7
        )


@pytest.mark.parametrize("initial_guess", [None, "unknown"])
def test_solver_rejects_an_invalid_initial_guess(initial_guess):
    error = TypeError if initial_guess is None else ValueError
    with pytest.raises(error, match="initial_guess"):
        SphericalAtomicKohnSham(
            "He",
            functional=LDAVWN(),
            r_max=10,
            n_elements=2,
            n_points=7,
            initial_guess=initial_guess,
        )


def test_semilocal_solver_rejects_insufficient_xc_quadrature():
    with pytest.raises(ValueError, match="at least n_points"):
        SphericalAtomicKohnSham(
            "He",
            functional=PBE(),
            **HELIUM_GRID,
            xc_quadrature_order=12,
        )


def test_helium_pbe_ground_state_and_density_gradient():
    solver = SphericalAtomicKohnSham(
        "He",
        functional=PBE(),
        **HELIUM_GRID,
        max_iterations=40,
        conv_tol_grad=1e-9,
    )
    result = solver.solve()
    assert result.converged
    assert result.energy == pytest.approx(-2.8929348668321, abs=3e-11)
    assert result.functional.name == "PBE"
    assert result.initial_guess == "lda"
    np.testing.assert_array_equal(result.xc_quadrature_order, [26, 26, 26, 26])
    assert result.density_gradient_squared.shape == result.density.shape
    assert np.all(result.density_gradient_squared >= 0)
    assert np.all(np.isfinite(result.xc_potential))
    assert result.xc_matrix.shape == (len(solver.r), len(solver.r))
    weighted_xc = solver.radial_weights[:, None] * result.xc_matrix
    np.testing.assert_allclose(weighted_xc, weighted_xc.T, atol=2e-12)


def test_pbe_weak_matrix_is_discrete_xc_energy_derivative():
    solver = SphericalAtomicKohnSham(
        "He", functional=PBE(), **HELIUM_GRID, max_iterations=40
    )
    result = solver.solve()
    orbital = result.radial_orbitals[0][0]
    direction = np.sin(np.linspace(0.0, np.pi, len(solver.r)))
    expected = 4.0 * np.dot(
        solver.radial_weights * direction,
        result.xc_matrix @ orbital,
    ).real

    def xc_energy(values):
        state = solver._assemble({0: values[None, :]})
        return state["xc_energy"]

    step = 1e-6
    finite_difference = (
        xc_energy(orbital + step * direction)
        - xc_energy(orbital - step * direction)
    ) / (2.0 * step)
    assert finite_difference == pytest.approx(expected, abs=5e-10)


@pytest.mark.parametrize(
    ("atom", "expected_energy"),
    [
        ("Ne", -128.8664391466599),
        ("Mg", -199.9553089506940),
    ],
)
def test_pbe_weak_form_converges_smoothly_for_closed_shell_atoms(
    atom, expected_energy
):
    result = SphericalAtomicKohnSham(
        atom,
        functional=PBE(),
        nodes=np.linspace(0.0, 20.0, 11),
        n_points=21,
        max_iterations=40,
        conv_tol_grad=1e-8,
        initial_guess="auto",
        poisson_method="solve",
    ).solve()
    assert result.converged
    assert result.iterations < 15
    assert result.energy == pytest.approx(expected_energy, abs=3e-10)
