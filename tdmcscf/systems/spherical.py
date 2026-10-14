"""Closed-shell electronic systems in central one-body potentials."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from .atoms import AtomicConfiguration, closed_shell_atom


def _positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real scalar.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real scalar.")
    return value


class ClosedShellOccupation:
    r"""Numbers of occupied radial orbitals in each angular-momentum sector.

    ``radial_orbitals_per_l[l] = n_l`` represents ``n_l`` radial orbitals,
    each present for every ``m=-l,...,l`` and doubly occupied. Consequently,

    ``electron_count = 2 * sum_l (2*l + 1) * n_l``.
    """

    def __init__(self, radial_orbitals_per_l):
        if isinstance(radial_orbitals_per_l, ClosedShellOccupation):
            counts = dict(radial_orbitals_per_l.radial_orbitals_per_l)
        elif isinstance(radial_orbitals_per_l, Mapping):
            counts = dict(radial_orbitals_per_l)
        else:
            raise TypeError("radial_orbitals_per_l must be a mapping.")
        if not counts:
            raise ValueError("radial_orbitals_per_l cannot be empty.")
        checked = {}
        for l, count in counts.items():
            if isinstance(l, (bool, np.bool_)) or not isinstance(l, Integral):
                raise TypeError("Angular momenta must be nonnegative integers.")
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
                raise TypeError("Radial-orbital counts must be positive integers.")
            l = int(l)
            count = int(count)
            if l < 0:
                raise ValueError("Angular momenta must be nonnegative integers.")
            if count <= 0:
                raise ValueError("Radial-orbital counts must be positive integers.")
            checked[l] = count
        self._counts = MappingProxyType(dict(sorted(checked.items())))

    @property
    def radial_orbitals_per_l(self):
        return self._counts

    @property
    def occupied_l(self):
        return tuple(self._counts)

    @property
    def occupied_l_max(self):
        return max(self._counts)

    @property
    def n_doubly_occupied(self):
        return sum((2 * l + 1) * count for l, count in self._counts.items())

    @property
    def electron_count(self):
        return 2 * self.n_doubly_occupied

    @property
    def radial_orbital_labels(self):
        """Return ``(n_r, l)`` labels with zero-based radial indices."""
        return tuple(
            (n_r, l)
            for l, count in self._counts.items()
            for n_r in range(count)
        )

    @property
    def orbital_quantum_numbers(self):
        """Return occupied ``(n_r, l, m)`` labels."""
        return tuple(
            (n_r, l, m)
            for n_r, l in self.radial_orbital_labels
            for m in range(-l, l + 1)
        )

    def __repr__(self):
        return (
            "ClosedShellOccupation(radial_orbitals_per_l="
            f"{dict(self._counts)!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, ClosedShellOccupation):
            return NotImplemented
        return dict(self._counts) == dict(other._counts)


def _occupation(value):
    if isinstance(value, ClosedShellOccupation):
        return value
    return ClosedShellOccupation(value)


@dataclass(frozen=True)
class _CoulombPotential:
    Z: float

    def __call__(self, r):
        return -self.Z / np.asarray(r)


@dataclass(frozen=True)
class _HarmonicOscillatorPotential:
    omega: float
    mass: float

    def __call__(self, r):
        return 0.5 * self.mass * self.omega**2 * np.asarray(r) ** 2


@dataclass(frozen=True)
class _GaussianWellPotential:
    V0: float
    a: float

    def __call__(self, r):
        return -self.V0 * np.exp(-self.a * np.asarray(r) ** 2)


class SphericalElectronicSystem:
    """Closed-shell electrons moving in a time-independent central potential."""

    def __init__(self, *, potential, occupation, mass=1.0, label=None):
        if not callable(potential):
            raise TypeError("potential must be callable as potential(r).")
        self.potential = potential
        self.occupation = _occupation(occupation)
        self.mass = _positive_real(mass, "mass")
        if label is None:
            label = type(self).__name__
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a nonempty string.")
        self.label = label

    @property
    def n_doubly_occupied(self):
        return self.occupation.n_doubly_occupied

    @property
    def electron_count(self):
        return self.occupation.electron_count

    @property
    def occupied_l_max(self):
        return self.occupation.occupied_l_max

    @property
    def occupied_m_max(self):
        return self.occupied_l_max

    @property
    def radial_orbital_labels(self):
        return self.occupation.radial_orbital_labels

    @property
    def orbital_quantum_numbers(self):
        return self.occupation.orbital_quantum_numbers

    def potential_values(self, r):
        """Evaluate and validate the real central potential on radial nodes."""
        nodes = np.asarray(r, dtype=float)
        values = np.asarray(self.potential(nodes))
        if values.ndim == 0:
            values = np.full(nodes.shape, values)
        if values.shape != nodes.shape:
            raise ValueError(
                "potential(r) must return a scalar or match the shape of r."
            )
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError("potential(r) must return finite real values.")
        return values.astype(float)


class Atomic(SphericalElectronicSystem):
    """Closed-shell atom or ion in the central potential ``-Z/r``.

    A catalog system may be selected by label, by an ``AtomicConfiguration``,
    or by nuclear charge and charge. For systems outside the catalog, provide
    an explicit closed-shell ``occupation``.
    """

    def __init__(
        self,
        atom=None,
        *,
        nuclear_charge=None,
        charge=None,
        occupation=None,
        mass=1.0,
        label=None,
    ):
        if atom is not None and nuclear_charge is not None:
            raise ValueError("Supply atom or nuclear_charge, not both.")
        reference = nuclear_charge if atom is None else atom
        if reference is None:
            raise TypeError("atom or nuclear_charge is required.")

        configuration = None
        if occupation is None:
            configuration = closed_shell_atom(reference, charge=charge)
            Z = configuration.nuclear_charge
            ionic_charge = configuration.charge
            occupation = ClosedShellOccupation(
                {
                    l: sum(subshell.l == l for subshell in configuration.subshells)
                    for l in {subshell.l for subshell in configuration.subshells}
                }
            )
            system_label = configuration.symbol
        else:
            if isinstance(reference, (bool, np.bool_)) or not isinstance(
                reference, Integral
            ):
                raise TypeError(
                    "An atom with an explicit occupation requires an integer "
                    "nuclear charge."
                )
            Z = int(reference)
            if Z <= 0:
                raise ValueError("nuclear_charge must be positive.")
            occupation = _occupation(occupation)
            inferred_charge = Z - occupation.electron_count
            if charge is None:
                ionic_charge = inferred_charge
            else:
                if isinstance(charge, (bool, np.bool_)) or not isinstance(
                    charge, Integral
                ):
                    raise TypeError("charge must be an integer.")
                ionic_charge = int(charge)
                if ionic_charge != inferred_charge:
                    raise ValueError(
                        "occupation must contain nuclear_charge - charge electrons."
                    )
            system_label = f"Z={Z}, charge={ionic_charge:+d}"

        self.configuration = configuration
        self.nuclear_charge = Z
        self.charge = ionic_charge
        super().__init__(
            potential=_CoulombPotential(float(Z)),
            occupation=occupation,
            mass=mass,
            label=system_label if label is None else label,
        )

    @property
    def radial_orbital_labels(self):
        if self.configuration is None:
            return super().radial_orbital_labels
        return tuple(
            (subshell.n, subshell.l)
            for subshell in self.configuration.subshells
        )

    @property
    def orbital_quantum_numbers(self):
        if self.configuration is None:
            return super().orbital_quantum_numbers
        return self.configuration.orbital_quantum_numbers


class HarmonicOscillator(SphericalElectronicSystem):
    r"""Closed-shell electrons in ``V(r) = mass*omega**2*r**2/2``."""

    def __init__(self, omega, *, occupation, mass=1.0, label=None):
        mass = _positive_real(mass, "mass")
        omega = _positive_real(omega, "omega")
        self.omega = omega
        super().__init__(
            potential=_HarmonicOscillatorPotential(omega, mass),
            occupation=occupation,
            mass=mass,
            label="harmonic oscillator" if label is None else label,
        )


class GaussianWell(SphericalElectronicSystem):
    r"""Closed-shell electrons in ``V(r) = -V0*exp(-a*r**2)``."""

    def __init__(self, V0, a, *, occupation, mass=1.0, label=None):
        self.V0 = _positive_real(V0, "V0")
        self.a = _positive_real(a, "a")
        super().__init__(
            potential=_GaussianWellPotential(self.V0, self.a),
            occupation=occupation,
            mass=mass,
            label="Gaussian well" if label is None else label,
        )


class CustomCentralPotential(SphericalElectronicSystem):
    """Closed-shell electronic system defined by a callable ``potential(r)``."""

    def __init__(self, potential, *, occupation, mass=1.0, label="custom"):
        super().__init__(
            potential=potential,
            occupation=occupation,
            mass=mass,
            label=label,
        )
