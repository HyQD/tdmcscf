r"""Restricted time-dependent Hartree--Fock actions in spherical coordinates."""

import numpy as np

from ._validation import _orbitals
from .atomic_core import AtomicCoreHamiltonian
from .dipole_fields import SphericalDipoleField
from .mean_fields import SphericalMeanFieldBuilder


class SphericalRestrictedTDHF:
    r"""Closed-shell spherical TDHF orbital equation.

    Parameters
    ----------
    core : AtomicCoreHamiltonian
        Time-independent kinetic and central nuclear operator.
    mean_fields : SphericalMeanFieldBuilder
        Builder whose orbital channels match ``core``. Its ``L_max`` and
        ``M_max`` determine the electron--electron Coulomb truncation.
    dipole_field : SphericalDipoleField, optional
        Time-dependent electric-dipole interaction. Its orbital channels must
        match ``core``. If omitted, the one-body action is field free.
    orbital_gauge : {"orthogonal", "canonical"}, optional
        The orthogonal gauge propagates ``i*dot(Phi) = Q F Phi``. The
        canonical gauge propagates ``i*dot(Phi) = F Phi``. Defaults to the
        orthogonal gauge.
    orbital_m : sequence of int, optional
        Definite magnetic quantum number of every occupied spatial orbital.
        Supplying it enables the polar-only mean-field path. The number of
        entries is checked when an orbital array is supplied.

    Notes
    -----
    For closed-shell spatial orbitals, the Fock action is

    ``F phi_p = h phi_p + sum_j (2 J_j - K_j) phi_p``.

    The split equation uses ``h`` as its linear part. In the orthogonal gauge
    the nonlinear action is ``G Phi - Phi (Phi**dagger F Phi)``, so the sum
    of the linear and nonlinear actions is exactly ``Q F Phi``.
    """

    def __init__(
        self,
        core,
        mean_fields,
        *,
        dipole_field=None,
        orbital_gauge="orthogonal",
        orbital_m=None,
    ):
        if not isinstance(core, AtomicCoreHamiltonian):
            raise TypeError("core must be an AtomicCoreHamiltonian.")
        if not isinstance(mean_fields, SphericalMeanFieldBuilder):
            raise TypeError("mean_fields must be a SphericalMeanFieldBuilder.")
        if mean_fields.orbital_shape != core.orbital_shape:
            raise ValueError("core and mean_fields must use the same orbital space.")
        if mean_fields.orbital_lm != core.orbital_lm:
            raise ValueError("core and mean_fields must use the same channel order.")
        if dipole_field is not None:
            if not isinstance(dipole_field, SphericalDipoleField):
                raise TypeError("dipole_field must be a SphericalDipoleField.")
            if dipole_field.orbital_shape != core.orbital_shape:
                raise ValueError(
                    "core and dipole_field must use the same orbital space."
                )
            if dipole_field.orbital_lm != core.orbital_lm:
                raise ValueError(
                    "core and dipole_field must use the same channel order."
                )
        if orbital_gauge not in {"orthogonal", "canonical"}:
            raise ValueError("orbital_gauge must be 'orthogonal' or 'canonical'.")

        if orbital_m is None:
            magnetic = None
        else:
            magnetic = np.asarray(orbital_m)
            if magnetic.ndim != 1 or magnetic.dtype.kind not in "iu":
                raise ValueError(
                    "orbital_m must be a one-dimensional integer sequence."
                )
            if np.any(np.abs(magnetic) > core.m_max):
                raise ValueError("orbital_m contains a value outside the orbital basis.")
            magnetic = tuple(int(m) for m in magnetic)

        self.core = core
        self.mean_fields = mean_fields
        self.dipole_field = dipole_field
        self.orbital_gauge = orbital_gauge
        self.orbital_m = magnetic
        self.orbital_shape = core.orbital_shape
        self.orbital_lm = core.orbital_lm
        self.radial_weights = core.radial_weights

    def _orbitals(self, orbitals):
        values, _ = _orbitals(orbitals, self.orbital_shape, allow_single=False)
        if self.orbital_m is not None and len(self.orbital_m) != len(values):
            raise ValueError("orbital_m must contain one value per occupied orbital.")
        return values

    def overlaps(self, left, right):
        """Return the occupied-space overlap matrix ``left**dagger right``."""
        left = self._orbitals(left)
        right = self._orbitals(right)
        if len(left) != len(right):
            raise ValueError("left and right must contain the same number of orbitals.")
        return np.einsum(
            "qai,a,pai->qp",
            left.conj(),
            self.radial_weights,
            right,
            optimize=True,
        )

    def orbital_norms(self, orbitals):
        """Return the radial-quadrature norm of every spatial orbital."""
        values = self._orbitals(orbitals)
        return np.einsum(
            "pai,a,pai->p",
            values.conj(),
            self.radial_weights,
            values,
            optimize=True,
        ).real

    def one_body_action(self, orbitals, time):
        """Apply the atomic core and optional dipole interaction."""
        values = self._orbitals(orbitals)
        result = self.core.apply(values)
        if self.dipole_field is not None:
            result += self.dipole_field.apply(values, time)
        return result

    def mean_field_action(self, orbitals):
        r"""Apply ``sum_j (2 J_j - K_j)`` to every occupied orbital."""
        values = self._orbitals(orbitals)
        fields = self.mean_fields.build(values, orbital_m=self.orbital_m)
        result = np.zeros_like(values)
        for j in range(len(values)):
            result += 2 * fields.apply(
                values,
                (j, j),
                orbital_m=self.orbital_m,
            )
            for p in range(len(values)):
                target_m = None if self.orbital_m is None else self.orbital_m[j]
                result[p] -= fields.apply(
                    values[j],
                    (j, p),
                    orbital_m=target_m,
                )
        return result

    def fock_action(self, orbitals, time):
        """Apply the complete closed-shell Fock operator."""
        values = self._orbitals(orbitals)
        return self.one_body_action(values, time) + self.mean_field_action(values)

    def linear_rhs(self, orbitals, time):
        """Return ``-i h(t) Phi`` for the implicit linear substep."""
        return -1j * self.one_body_action(orbitals, time)

    def nonlinear_rhs(self, orbitals, time):
        """Return the nonlinear time derivative in the selected orbital gauge."""
        values = self._orbitals(orbitals)
        mean_field = self.mean_field_action(values)
        if self.orbital_gauge == "canonical":
            action = mean_field
        else:
            fock = self.one_body_action(values, time) + mean_field
            occupied_fock = self.overlaps(values, fock)
            projection = np.einsum(
                "qai,qp->pai", values, occupied_fock, optimize=True
            )
            action = mean_field - projection
        return -1j * action
