r"""Spin-restricted time-dependent Kohn--Sham orbital actions."""

import numpy as np

from .local_density import SphericalXCBuilder
from .tdhf import SphericalRestrictedTDHF


class SphericalRestrictedTDKS(SphericalRestrictedTDHF):
    r"""Adiabatic spin-unpolarized TDKS equation in spherical coordinates.

    The orbital Hamiltonian is

    ``H[n](t) = h(t) + V_H[n](t) + V_xc[n](t)``.

    The Hartree potential uses ``mean_fields`` and the local or semilocal XC
    operator uses ``xc``. Both are rebuilt from the instantaneous spin-summed
    density.
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
        self._channel_m = np.array([m for _, m in self.orbital_lm])

    def _project_fixed_m(self, orbitals):
        """Project solver vectors onto their declared magnetic subspaces."""
        values = self._orbitals(orbitals)
        if self.orbital_m is None:
            return values
        projected = np.zeros_like(values)
        for p, m in enumerate(self.orbital_m):
            selected = self._channel_m == m
            projected[p, :, selected] = values[p, :, selected]
        return projected

    def _hartree_fields(self, orbitals):
        values = self._project_fixed_m(orbitals)
        pairs = tuple((p, p) for p in range(len(values)))
        return self.mean_fields.build(
            values,
            pairs=pairs,
            orbital_m=self.orbital_m,
        )

    def _hartree_action(self, orbitals, fields):
        values = self._project_fixed_m(orbitals)
        result = np.zeros_like(values)
        for p in range(len(values)):
            result += 2.0 * fields.apply(
                values,
                (p, p),
            )
        return self._project_fixed_m(result)

    def mean_field_action(self, orbitals):
        """Apply the instantaneous Hartree and adiabatic XC operators."""
        values = self._project_fixed_m(orbitals)
        hartree_fields = self._hartree_fields(values)
        xc_potential = self.xc.build(values)
        return self._project_fixed_m(
            self._hartree_action(values, hartree_fields)
            + xc_potential.apply(values)
        )

    def hamiltonian_action(self, orbitals, time):
        """Apply the complete time-dependent Kohn--Sham Hamiltonian."""
        values = self._project_fixed_m(orbitals)
        return self._project_fixed_m(
            self.one_body_action(values, time) + self.mean_field_action(values)
        )

    def fock_action(self, orbitals, time):
        """Compatibility alias for :meth:`hamiltonian_action`."""
        return self.hamiltonian_action(orbitals, time)

    def freeze_rhs(self, reference_orbitals):
        r"""Freeze the Hartree--XC operator and return a linear orbital RHS."""
        reference = self._project_fixed_m(reference_orbitals).copy()
        hartree_fields = self._hartree_fields(reference)
        xc_potential = self.xc.build(reference)
        reference_mean_field = self._project_fixed_m(
            self._hartree_action(reference, hartree_fields)
            + xc_potential.apply(reference)
        )
        occupied_hamiltonian_cache = {}

        def frozen_rhs(trial_orbitals, time):
            values = self._project_fixed_m(trial_orbitals)
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
            return -1j * self._project_fixed_m(action)

        return frozen_rhs

    def freeze_density_rhs(self, reference_orbitals):
        """Freeze the density; equivalent to :meth:`freeze_rhs` for TDKS."""
        return self.freeze_rhs(reference_orbitals)

    def linear_rhs(self, orbitals, time):
        """Return the fixed-m-projected one-body time derivative."""
        values = self._project_fixed_m(orbitals)
        return -1j * self._project_fixed_m(
            self.one_body_action(values, time)
        )

    def nonlinear_rhs(self, orbitals, time):
        """Return the nonlinear TDKS derivative in the selected gauge."""
        values = self._project_fixed_m(orbitals)
        mean_field = self.mean_field_action(values)
        if self.orbital_gauge == "canonical":
            action = mean_field
        else:
            hamiltonian = self.one_body_action(values, time) + mean_field
            occupied_hamiltonian = self.overlaps(values, hamiltonian)
            action = mean_field - np.einsum(
                "qai,qp->pai",
                values,
                occupied_hamiltonian,
                optimize=True,
            )
        return -1j * self._project_fixed_m(action)
