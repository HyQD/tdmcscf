import pytest

from tdmcscf.systems import (
    AtomicConfiguration,
    ClosedSubshell,
    atomic_configurations,
    closed_shell_atom,
)


@pytest.mark.parametrize(
    "label,Z,charge,electrons,n_docc,l_max",
    [
        ("H-", 1, -1, 2, 1, 0),
        ("Ne", 10, 0, 10, 5, 1),
        ("Ti4+", 22, 4, 18, 9, 1),
        ("Kr", 36, 0, 36, 18, 2),
    ],
)
def test_catalog_closed_shell_counts(label, Z, charge, electrons, n_docc, l_max):
    atom = closed_shell_atom(label)
    assert atom.nuclear_charge == Z
    assert atom.charge == charge
    assert atom.electron_count == electrons
    assert atom.n_doubly_occupied == n_docc
    assert atom.occupied_l_max == l_max
    assert atom.occupied_m_max == l_max
    assert len(atom.orbital_quantum_numbers) == n_docc
    for subshell in atom.subshells:
        states = [
            state
            for state in atom.orbital_quantum_numbers
            if state[:2] == (subshell.n, subshell.l)
        ]
        assert [state[2] for state in states] == list(
            range(-subshell.l, subshell.l + 1)
        )


def test_explicit_configuration_and_read_only_catalog():
    lithium_ion = AtomicConfiguration("Li+", 3, 1, (ClosedSubshell(1, 0),))
    assert closed_shell_atom(lithium_ion) is lithium_ion
    assert lithium_ion.orbital_quantum_numbers == ((1, 0, 0),)
    with pytest.raises(TypeError):
        atomic_configurations()["Li+"] = lithium_ion


def test_rejects_inconsistent_or_non_closed_configurations():
    with pytest.raises(ValueError, match="electrons"):
        AtomicConfiguration("bad", 4, 0, (ClosedSubshell(1, 0),))
    with pytest.raises(ValueError, match="unique"):
        AtomicConfiguration("bad", 4, 0, (ClosedSubshell(1, 0), ClosedSubshell(1, 0)))
    with pytest.raises(ValueError, match="Unknown"):
        closed_shell_atom("Xe")
