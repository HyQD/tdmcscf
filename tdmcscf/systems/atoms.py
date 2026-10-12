"""Closed-shell atomic and ionic configurations."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, order=True)
class ClosedSubshell:
    """A completely occupied nonrelativistic ``(n, l)`` subshell."""

    n: int
    l: int

    def __post_init__(self):
        if isinstance(self.n, bool) or not isinstance(self.n, int):
            raise TypeError("n must be an integer.")
        if isinstance(self.l, bool) or not isinstance(self.l, int):
            raise TypeError("l must be an integer.")
        if self.n < 1:
            raise ValueError("n must be positive.")
        if self.l < 0 or self.l >= self.n:
            raise ValueError("A subshell must satisfy 0 <= l < n.")

    @property
    def electron_count(self):
        """Number of electrons in the complete subshell."""
        return 2 * (2 * self.l + 1)

    @property
    def spatial_orbital_count(self):
        """Number of doubly occupied spatial orbitals."""
        return 2 * self.l + 1


@dataclass(frozen=True)
class AtomicConfiguration:
    """A closed-shell atom or ion, independent of its numerical basis."""

    symbol: str
    nuclear_charge: int
    charge: int
    subshells: tuple[ClosedSubshell, ...]

    def __post_init__(self):
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a nonempty string.")
        if isinstance(self.nuclear_charge, bool) or not isinstance(
            self.nuclear_charge, int
        ):
            raise TypeError("nuclear_charge must be an integer.")
        if self.nuclear_charge <= 0:
            raise ValueError("nuclear_charge must be positive.")
        if isinstance(self.charge, bool) or not isinstance(self.charge, int):
            raise TypeError("charge must be an integer.")
        subshells = tuple(self.subshells)
        if not subshells or not all(isinstance(s, ClosedSubshell) for s in subshells):
            raise ValueError("subshells must contain ClosedSubshell objects.")
        if len(set(subshells)) != len(subshells):
            raise ValueError("subshells must be unique.")
        object.__setattr__(self, "subshells", subshells)
        if self.electron_count != self.nuclear_charge - self.charge:
            raise ValueError(
                "The closed subshells must contain nuclear_charge - charge electrons."
            )

    @property
    def electron_count(self):
        return sum(subshell.electron_count for subshell in self.subshells)

    @property
    def n_doubly_occupied(self):
        return sum(subshell.spatial_orbital_count for subshell in self.subshells)

    @property
    def occupied_l_max(self):
        return max(subshell.l for subshell in self.subshells)

    @property
    def occupied_m_max(self):
        return self.occupied_l_max

    @property
    def orbital_quantum_numbers(self):
        """Occupied spatial orbitals in subshell order and increasing m."""
        return tuple(
            (subshell.n, subshell.l, m)
            for subshell in self.subshells
            for m in range(-subshell.l, subshell.l + 1)
        )


def _configuration(symbol, nuclear_charge, charge, *nl):
    return AtomicConfiguration(
        symbol,
        nuclear_charge,
        charge,
        tuple(ClosedSubshell(n, l) for n, l in nl),
    )


_CONFIGURATIONS = {
    "H-": _configuration("H-", 1, -1, (1, 0)),
    "He": _configuration("He", 2, 0, (1, 0)),
    "Be": _configuration("Be", 4, 0, (1, 0), (2, 0)),
    "F-": _configuration("F-", 9, -1, (1, 0), (2, 0), (2, 1)),
    "Ne": _configuration("Ne", 10, 0, (1, 0), (2, 0), (2, 1)),
    "Mg": _configuration("Mg", 12, 0, (1, 0), (2, 0), (2, 1), (3, 0)),
    "Cl-": _configuration("Cl-", 17, -1, (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)),
    "Ar": _configuration("Ar", 18, 0, (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)),
    "Ti4+": _configuration("Ti4+", 22, 4, (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)),
    "Ca": _configuration("Ca", 20, 0, (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0)),
    "Zn": _configuration(
        "Zn",
        30,
        0,
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (4, 0),
        (3, 2),
    ),
    "Kr": _configuration(
        "Kr",
        36,
        0,
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (4, 0),
        (3, 2),
        (4, 1),
    ),
}


def atomic_configurations():
    """Return a read-only mapping of built-in closed-shell configurations."""
    return MappingProxyType(_CONFIGURATIONS)


def closed_shell_atom(atom):
    """Resolve a catalog label or validate an explicit configuration."""
    if isinstance(atom, AtomicConfiguration):
        return atom
    if not isinstance(atom, str):
        raise TypeError("atom must be a configuration or catalog label.")
    try:
        return _CONFIGURATIONS[atom]
    except KeyError as error:
        available = ", ".join(_CONFIGURATIONS)
        raise ValueError(
            f"Unknown closed-shell atom or ion {atom!r}: {available}."
        ) from error
