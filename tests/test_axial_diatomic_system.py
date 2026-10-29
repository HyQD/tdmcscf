import numpy as np
import pytest

from tdmcscf.systems import AxialClosedShellOccupation, AxialDiatomic


def test_axial_closed_shell_occupation_expands_signed_sectors():
    occupation = AxialClosedShellOccupation({0: 2, 1: 1, 2: 2})

    assert dict(occupation.orbitals_per_abs_m) == {0: 2, 1: 1, 2: 2}
    assert dict(occupation.orbitals_per_m) == {
        -2: 2,
        -1: 1,
        0: 2,
        1: 1,
        2: 2,
    }
    assert occupation.occupied_m_max == 2
    assert occupation.n_doubly_occupied == 8
    assert occupation.electron_count == 16
    assert occupation.orbital_m == (-2, -2, -1, 0, 0, 1, 2, 2)
    assert occupation == AxialClosedShellOccupation(occupation)


@pytest.mark.parametrize(
    "value",
    [{}, {True: 1}, {-1: 1}, {0: 0}, {0: 1.5}, [(0, 1)]],
)
def test_invalid_axial_closed_shell_occupation(value):
    with pytest.raises((TypeError, ValueError)):
        AxialClosedShellOccupation(value)


def test_h2_plus_geometry_and_derived_properties():
    system = AxialDiatomic(
        "H",
        "H",
        half_bond_length=0.7,
        charge=1,
        label="H2+",
    )

    assert system.label == "H2+"
    assert system.nuclear_symbols == ("H", "H")
    assert system.total_nuclear_charge == 2.0
    assert system.electron_count == 1
    assert system.mass == 1.0
    assert system.internuclear_distance == pytest.approx(1.4)
    assert system.nuclear_repulsion_energy == pytest.approx(1.0 / 1.4)
    np.testing.assert_array_equal(
        system.nuclear_positions,
        [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
    )
    np.testing.assert_array_equal(system.nuclear_charges, [1.0, 1.0])
    np.testing.assert_array_equal(system.nuclear_dipole, [0.0, 0.0, 0.0])


def test_heteronuclear_system_accepts_distance_and_effective_charges():
    system = AxialDiatomic(
        1.5,
        np.float64(2.5),
        internuclear_distance=2.4,
        charge=2,
        mass=0.8,
    )

    assert system.half_bond_length == pytest.approx(1.2)
    assert system.electron_count == 2
    assert system.mass == pytest.approx(0.8)
    assert system.nuclear_repulsion_energy == pytest.approx(1.5 * 2.5 / 2.4)
    np.testing.assert_allclose(system.nuclear_dipole, [0.0, 0.0, 1.2])
    assert system.geometry_dict()["electron_count"] == 2


def test_nuclear_arrays_are_read_only():
    system = AxialDiatomic("H", "He", half_bond_length=1.0, charge=1)

    assert not system.nuclear_positions.flags.writeable
    assert not system.nuclear_charges.flags.writeable
    assert not system.nuclear_dipole.flags.writeable


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({}, "exactly one"),
        (
            {"half_bond_length": 1.0, "internuclear_distance": 2.0},
            "exactly one",
        ),
        ({"half_bond_length": 0.0}, "half_bond_length"),
        ({"half_bond_length": 1.0, "charge": 0.5}, "charge"),
        ({"half_bond_length": 1.0, "charge": 3}, "negative"),
    ],
)
def test_invalid_geometry_or_charge(arguments, match):
    with pytest.raises((TypeError, ValueError), match=match):
        AxialDiatomic("H", "H", **arguments)


def test_invalid_symbol_and_noninteger_effective_electron_count():
    with pytest.raises(ValueError, match="Unknown element"):
        AxialDiatomic("Xx", "H", half_bond_length=1.0)
    with pytest.raises(ValueError, match="integer electron count"):
        AxialDiatomic(1.2, 1.3, half_bond_length=1.0)
