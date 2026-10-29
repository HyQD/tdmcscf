"""Clamped, midpoint-centred axial diatomic systems."""

from collections.abc import Mapping
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np


_ELEMENT_SYMBOLS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
)
_NUCLEAR_CHARGES = {
    symbol: index for index, symbol in enumerate(_ELEMENT_SYMBOLS, start=1)
}


class AxialClosedShellOccupation:
    r"""Closed-shell spatial-orbital counts in absolute-``m`` sectors.

    ``orbitals_per_abs_m[0]`` is the number of doubly occupied sigma
    orbitals. For ``abs(m) > 0``, the count is present in both signed sectors
    ``+m`` and ``-m``. The electron count is therefore

    ``2 * (n_0 + 2 * sum_{m>0} n_m)``.
    """

    def __init__(self, orbitals_per_abs_m):
        if isinstance(orbitals_per_abs_m, AxialClosedShellOccupation):
            counts = dict(orbitals_per_abs_m.orbitals_per_abs_m)
        elif isinstance(orbitals_per_abs_m, Mapping):
            counts = dict(orbitals_per_abs_m)
        else:
            raise TypeError("orbitals_per_abs_m must be a mapping.")
        if not counts:
            raise ValueError("orbitals_per_abs_m cannot be empty.")
        checked = {}
        for magnetic, count in counts.items():
            if isinstance(magnetic, (bool, np.bool_)) or not isinstance(
                magnetic, Integral
            ):
                raise TypeError("Magnetic sectors must be nonnegative integers.")
            if isinstance(count, (bool, np.bool_)) or not isinstance(
                count, Integral
            ):
                raise TypeError("Orbital counts must be positive integers.")
            magnetic = int(magnetic)
            count = int(count)
            if magnetic < 0:
                raise ValueError("Magnetic sectors must be nonnegative integers.")
            if count <= 0:
                raise ValueError("Orbital counts must be positive integers.")
            checked[magnetic] = count
        self._counts = MappingProxyType(dict(sorted(checked.items())))

    @property
    def orbitals_per_abs_m(self):
        return self._counts

    @property
    def occupied_m_max(self):
        return max(self._counts)

    @property
    def orbitals_per_m(self):
        signed = {}
        for magnetic, count in self._counts.items():
            signed[magnetic] = count
            if magnetic:
                signed[-magnetic] = count
        return MappingProxyType(dict(sorted(signed.items())))

    @property
    def n_doubly_occupied(self):
        return sum(
            (1 if magnetic == 0 else 2) * count
            for magnetic, count in self._counts.items()
        )

    @property
    def electron_count(self):
        return 2 * self.n_doubly_occupied

    @property
    def orbital_labels(self):
        return tuple(
            (index, magnetic)
            for magnetic, count in self.orbitals_per_m.items()
            for index in range(count)
        )

    @property
    def orbital_m(self):
        return tuple(magnetic for _, magnetic in self.orbital_labels)

    def __repr__(self):
        return (
            "AxialClosedShellOccupation(orbitals_per_abs_m="
            f"{dict(self._counts)!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, AxialClosedShellOccupation):
            return NotImplemented
        return dict(self._counts) == dict(other._counts)


def _positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real scalar.")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite real scalar.")
    return value


def _nucleus(value, name):
    if isinstance(value, str):
        symbol = value.strip()
        try:
            return float(_NUCLEAR_CHARGES[symbol]), symbol
        except KeyError as error:
            available = ", ".join(_ELEMENT_SYMBOLS)
            raise ValueError(
                f"Unknown element symbol {value!r}. Available symbols: {available}."
            ) from error
    return _positive_real(value, name), None


class AxialDiatomic:
    r"""Two clamped nuclei at ``(0,0,-a)`` and ``(0,0,+a)``.

    Nuclear centres may be supplied as positive charges or as chemical
    symbols through krypton. The spherical expansion origin is the bond
    midpoint. ``charge`` is the total molecular charge and determines the
    electron count, but no orbital occupation is attached to this geometry.
    """

    def __init__(
        self,
        nucleus1,
        nucleus2,
        *,
        half_bond_length=None,
        internuclear_distance=None,
        charge=0,
        mass=1.0,
        label=None,
    ):
        if (half_bond_length is None) == (internuclear_distance is None):
            raise ValueError(
                "Supply exactly one of half_bond_length or "
                "internuclear_distance."
            )
        if half_bond_length is None:
            distance = _positive_real(
                internuclear_distance, "internuclear_distance"
            )
            half_bond_length = 0.5 * distance
        else:
            half_bond_length = _positive_real(
                half_bond_length, "half_bond_length"
            )
            distance = 2.0 * half_bond_length

        charge1, symbol1 = _nucleus(nucleus1, "nucleus1")
        charge2, symbol2 = _nucleus(nucleus2, "nucleus2")
        if isinstance(charge, (bool, np.bool_)) or not isinstance(
            charge, Integral
        ):
            raise TypeError("charge must be an integer.")
        charge = int(charge)
        electron_count = charge1 + charge2 - charge
        rounded_count = round(electron_count)
        if not np.isclose(electron_count, rounded_count, rtol=0.0, atol=1e-12):
            raise ValueError(
                "nuclear_charge1 + nuclear_charge2 - charge must be an "
                "integer electron count."
            )
        if rounded_count < 0:
            raise ValueError("charge cannot imply a negative electron count.")

        if label is None:
            left = symbol1 if symbol1 is not None else f"Z={charge1:g}"
            right = symbol2 if symbol2 is not None else f"Z={charge2:g}"
            suffix = "" if charge == 0 else f", charge={charge:+d}"
            label = f"{left}{right} ({distance:g} bohr{suffix})"
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be a nonempty string.")

        self.nuclear_charge1 = charge1
        self.nuclear_charge2 = charge2
        self.nuclear_symbols = (symbol1, symbol2)
        self.half_bond_length = half_bond_length
        self.internuclear_distance = distance
        self.charge = charge
        self.electron_count = int(rounded_count)
        self.mass = _positive_real(mass, "mass")
        self.label = label.strip()

        positions = np.asarray(
            [
                [0.0, 0.0, -half_bond_length],
                [0.0, 0.0, half_bond_length],
            ]
        )
        charges = np.asarray([charge1, charge2])
        dipole = np.sum(charges[:, None] * positions, axis=0)
        for values in (positions, charges, dipole):
            values.setflags(write=False)
        self.nuclear_positions = positions
        self.nuclear_charges = charges
        self.nuclear_dipole = dipole

    @property
    def total_nuclear_charge(self):
        return self.nuclear_charge1 + self.nuclear_charge2

    @property
    def nuclear_repulsion_energy(self):
        return (
            self.nuclear_charge1
            * self.nuclear_charge2
            / self.internuclear_distance
        )

    def geometry_dict(self):
        """Return a JSON-serializable description of the clamped geometry."""

        return {
            "type": "AxialDiatomic",
            "label": self.label,
            "nuclear_charges": self.nuclear_charges.tolist(),
            "nuclear_symbols": list(self.nuclear_symbols),
            "nuclear_positions": self.nuclear_positions.tolist(),
            "half_bond_length": self.half_bond_length,
            "internuclear_distance": self.internuclear_distance,
            "charge": self.charge,
            "electron_count": self.electron_count,
            "mass": self.mass,
        }
