r"""Spin-restricted time-dependent Kohn--Sham orbital actions."""

import numpy as np

from .local_density import SphericalXCBuilder
from .tdhf import SphericalRestrictedTDHF


class SphericalRestrictedTDKS(SphericalRestrictedTDHF):
    r"""Adiabatic spin-unpolarized TDKS equation in spherical coordinates.

    The orbital Hamiltonian is

    ``H[n](t) = h(t) + V_H[n](t) + V_xc[n](t)``.

    The Hartree potential uses ``mean_fields`` and the local XC potential uses
    ``xc``. Both are rebuilt from the instantaneous spin-summed density.
    Canonical and orthogonal occupied-orbital gauges are supported in the
    same way as by :class:`SphericalRestrictedTDHF`.
    """

    def __init__(self, core, mean_fields, xc, **kwargs):
        if not isinstance(xc, SphericalXCBuilder):
            raise TypeError("xc must be a SphericalXCBuilder.")
        self.xc = xc
        super().__init__(core, mean_fields, **kwargs)
        if xc.orbital_shape != core.orbital_shape:
            raise ValueError("core and xc must use the same orbital space.")
        if xc.orbital_lm != core.orbital_lm:
            raise ValueError("core and xc must use the same channel order.")

    def _hartree_fields(self, orbitals):
        pairs = tuple((p, p) for p in range(len(orbitals)))
        return self.mean_fields.build(
            orbitals,
            pairs=pairs,
            orbital_m=self.orbital_m,
        )

    def _hartree_action(self, orbitals, fields):
        values = self._orbitals(orbitals)
        result = np.zeros_like(values)
        for p in range(len(values)):
            result += 2.0 * fields.apply(
                values,
                (p, p),
                orbital_m=self.orbital_m,
            )
        return result

    def mean_field_action(self, orbitals):
        """Apply the instantaneous Hartree and adiabatic XC potentials."""
        values = self._orbitals(orbitals)
        hartree_fields = self._hartree_fields(values)
        xc_potential = self.xc.build(values)
        return self._hartree_action(values, hartree_fields) + xc_potential.apply(
            values
        )

    def hamiltonian_action(self, orbitals, time):
        """Apply the complete time-dependent Kohn--Sham Hamiltonian."""
        values = self._orbitals(orbitals)
        return self.one_body_action(values, time) + self.mean_field_action(values)

    def fock_action(self, orbitals, time):
        """Compatibility alias for :meth:`hamiltonian_action`."""
        return self.hamiltonian_action(orbitals, time)

    def freeze_rhs(self, reference_orbitals):
        r"""Freeze ``V_H[n] + V_xc[n]`` and return a linear orbital RHS."""
        reference = self._orbitals(reference_orbitals).copy()
        hartree_fields = self._hartree_fields(reference)
        xc_potential = self.xc.build(reference)
        reference_mean_field = self._hartree_action(
            reference, hartree_fields
        ) + xc_potential.apply(reference)
        occupied_hamiltonian_cache = {}

        def frozen_rhs(trial_orbitals, time):
            values = self._orbitals(trial_orbitals)
            action = self.one_body_action(values, time)
            action += self._hartree_action(values, hartree_fields)
            action += xc_potential.apply(values)
            if self.orbital_gauge == "orthogonal":
                key = float(time)
                if key not in occupied_hamiltonian_cache:
                    reference_action = self.one_body_action(reference, time)
                    reference_action += reference_mean_field
                    occupied_hamiltonian_cache[key] = self.overlaps(
                        reference, reference_action
                    )
                action -= np.einsum(
                    "qai,qp->pai",
                    values,
                    occupied_hamiltonian_cache[key],
                    optimize=True,
                )
            return -1j * action

        return frozen_rhs

    def freeze_density_rhs(self, reference_orbitals):
        """Freeze the density; equivalent to :meth:`freeze_rhs` for TDKS."""
        return self.freeze_rhs(reference_orbitals)
