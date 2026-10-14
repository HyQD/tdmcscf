from types import MappingProxyType

import numpy as np
import pytest

from tdmcscf.grid.spherical import (
    ShiftedSphericalCorePreconditioner,
    SphericalCoreHamiltonian,
    SphericalHartreeFock,
    SphericalMeanFieldBuilder,
    SphericalRestrictedTDHF,
    make_spherical_femdvr_grid,
)
from tdmcscf.systems import (
    Atomic,
    ClosedShellOccupation,
    CustomCentralPotential,
    GaussianWell,
    HarmonicOscillator,
    closed_shell_atom,
)


def test_closed_shell_occupation_derives_counts_and_quantum_numbers():
    occupation = ClosedShellOccupation({1: 1, 0: 2})
    assert isinstance(occupation.radial_orbitals_per_l, MappingProxyType)
    assert dict(occupation.radial_orbitals_per_l) == {0: 2, 1: 1}
    assert occupation.occupied_l == (0, 1)
    assert occupation.occupied_l_max == 1
    assert occupation.n_doubly_occupied == 5
    assert occupation.electron_count == 10
    assert occupation.radial_orbital_labels == ((0, 0), (1, 0), (0, 1))
    assert occupation.orbital_quantum_numbers == (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, -1),
        (0, 1, 0),
        (0, 1, 1),
    )
    with pytest.raises(TypeError):
        occupation.radial_orbitals_per_l[2] = 1


@pytest.mark.parametrize(
    "value",
    [{}, {-1: 1}, {0: 0}, {0.5: 1}, {0: 1.5}, [(0, 1)]],
)
def test_closed_shell_occupation_rejects_invalid_input(value):
    with pytest.raises((TypeError, ValueError)):
        ClosedShellOccupation(value)


def test_atomic_system_accepts_labels_nuclear_charges_and_explicit_occupations():
    by_label = Atomic("Ne")
    by_charge = Atomic(10)
    by_keyword = Atomic(nuclear_charge=10)
    assert by_label.configuration is closed_shell_atom("Ne")
    assert by_charge.configuration is by_label.configuration
    assert by_keyword.configuration is by_label.configuration
    assert by_label.nuclear_charge == 10
    assert by_label.charge == 0
    assert by_label.electron_count == 10
    assert Atomic(17, charge=-1).label == "Cl-"

    beryllium_ion = Atomic(4, charge=2, occupation={0: 1})
    assert beryllium_ion.configuration is None
    assert beryllium_ion.electron_count == 2
    assert beryllium_ion.label == "Z=4, charge=+2"
    with pytest.raises(ValueError, match="occupation"):
        Atomic(4, charge=0, occupation={0: 1})


def test_named_and_custom_potentials_use_the_requested_mass_and_parameters():
    r = np.array([0.5, 1.0, 2.0])
    harmonic = HarmonicOscillator(0.5, mass=2.0, occupation={0: 1})
    np.testing.assert_allclose(
        harmonic.potential_values(r), 0.5 * 2.0 * 0.5**2 * r**2
    )
    assert harmonic.mass == 2.0

    gaussian = GaussianWell(4.0, 0.5, occupation={0: 1})
    np.testing.assert_allclose(
        gaussian.potential_values(r), -4.0 * np.exp(-0.5 * r**2)
    )
    custom = CustomCentralPotential(
        lambda nodes: 3.0,
        occupation={0: 1},
        label="constant",
    )
    np.testing.assert_array_equal(custom.potential_values(r), np.full(3, 3.0))
    assert custom.label == "constant"


@pytest.mark.parametrize(
    "factory,match",
    [
        (lambda: HarmonicOscillator(0, occupation={0: 1}), "omega"),
        (lambda: GaussianWell(0, 1, occupation={0: 1}), "V0"),
        (lambda: GaussianWell(1, 0, occupation={0: 1}), "a"),
        (lambda: CustomCentralPotential(None, occupation={0: 1}), "callable"),
    ],
)
def test_systems_reject_invalid_parameters(factory, match):
    with pytest.raises((TypeError, ValueError), match=match):
        factory()


def test_generic_core_uses_the_system_potential_and_mass():
    grid = make_spherical_femdvr_grid(nodes=[0, 1, 3, 6], n_points=9)
    system = HarmonicOscillator(0.7, mass=2.0, occupation={0: 1})
    core = SphericalCoreHamiltonian(
        grid,
        2,
        m_max=1,
        potential=system.potential,
        mass=system.mass,
    )
    np.testing.assert_allclose(
        core.potential.values, 0.5 * system.mass * system.omega**2 * core.r**2
    )
    assert core.mass == 2.0
    values = np.ones(core.orbital_shape)
    np.testing.assert_allclose(
        core.apply(values), core.kinetic.apply(values) + core.potential.apply(values)
    )


def test_generic_core_is_accepted_by_tdhf_and_its_preconditioner():
    grid = make_spherical_femdvr_grid(nodes=[0, 1, 3, 6], n_points=9)
    system = GaussianWell(4.0, 0.5, occupation={0: 1})
    core = SphericalCoreHamiltonian(
        grid,
        0,
        potential=system.potential,
        mass=system.mass,
    )
    tdhf = SphericalRestrictedTDHF(
        core,
        SphericalMeanFieldBuilder(grid, 0),
    )
    orbitals = np.ones((1, *core.orbital_shape), dtype=complex)
    np.testing.assert_allclose(
        tdhf.one_body_action(orbitals, 0.0), core.apply(orbitals)
    )
    preconditioner = ShiftedSphericalCorePreconditioner(core, 0.01)
    assert preconditioner(orbitals).shape == orbitals.shape


def test_generic_hartree_fock_supports_named_and_custom_central_potentials():
    harmonic = HarmonicOscillator(0.5, occupation={0: 1})
    custom = CustomCentralPotential(
        lambda r: 0.5 * 0.5**2 * r**2,
        occupation={0: 1},
    )
    common = dict(
        nodes=[0, 2, 5, 10, 20],
        n_points=11,
        max_iterations=30,
        conv_tol_grad=1e-8,
        poisson_method="solve",
    )
    harmonic_result = SphericalHartreeFock(harmonic, **common).solve()
    custom_result = SphericalHartreeFock(custom, **common).solve()
    assert harmonic_result.converged
    assert custom_result.converged
    assert harmonic_result.energy == pytest.approx(2.0384388717504045, abs=2e-10)
    assert custom_result.energy == pytest.approx(harmonic_result.energy, abs=2e-12)
    assert harmonic_result.orbital_quantum_numbers == ((0, 0, 0),)
    weights = harmonic_result.grid.weights[1:-1]
    overlap = np.einsum(
        "paI,qaI,a->pq",
        harmonic_result.orbitals.conj(),
        harmonic_result.orbitals,
        weights,
    )
    np.testing.assert_allclose(overlap, np.eye(1), atol=2e-12)

    gaussian_result = SphericalHartreeFock(
        GaussianWell(4.0, 0.5, occupation={0: 1}), **common
    ).solve()
    assert gaussian_result.converged
    assert gaussian_result.energy < 0
