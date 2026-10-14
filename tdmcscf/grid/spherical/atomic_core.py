"""Spherical one-body Hamiltonians for time-independent central potentials."""

import numpy as np

from .kinetic import SphericalKineticEnergy
from .potentials import SphericalCoulombPotential, SphericalRadialPotential


class SphericalCoreHamiltonian:
    r"""Combine kinetic energy with an arbitrary real central potential.

    The Hamiltonian is

    ``Hcore = -laplacian/(2*mass) + V(r)``.

    ``potential`` must be callable as ``potential(r)``. The operator acts on
    nodal reduced radial values and provides one radial matrix for every ``l``.
    """

    def __init__(
        self,
        radial_grid,
        l_max,
        m_max=None,
        *,
        potential,
        mass=1.0,
    ):
        self.kinetic = SphericalKineticEnergy(
            radial_grid, l_max, m_max, mass=mass
        )
        self.potential = SphericalRadialPotential(
            radial_grid, l_max, m_max, potential=potential
        )
        self._set_metadata()

    def _set_metadata(self):
        if self.potential.orbital_lm != self.kinetic.orbital_lm:
            raise ValueError("kinetic and potential operators must use one basis.")
        self.l_max = self.kinetic.l_max
        self.m_max = self.kinetic.m_max
        self.r = self.kinetic.r
        self.radial_weights = self.kinetic.radial_weights
        self.orbital_lm = self.kinetic.orbital_lm
        self.orbital_shape = self.kinetic.orbital_shape

    @property
    def mass(self):
        """Particle or reduced mass fixed at construction."""
        return self.kinetic.mass

    def apply(self, orbitals):
        """Return ``Hcore*u`` with the same single-orbital or batch shape."""
        return self.kinetic.apply(orbitals) + self.potential.apply(orbitals)

    def radial_matrix(self, l):
        """Return the nodal radial block for ``0 <= l <= l_max``.

        The quadrature metric ``S = diag(radial_weights)`` satisfies
        ``H dagger S = S H``. Transform with ``sqrt(S) H invsqrt(S)`` before
        using a conventional Hermitian eigensolver.
        """
        matrix = self.kinetic.radial_matrix(l)
        matrix[np.diag_indices_from(matrix)] += self.potential.values
        return matrix


class AtomicCoreHamiltonian(SphericalCoreHamiltonian):
    """Compatibility wrapper for ``-laplacian/(2*mass) - Z/r``."""

    def __init__(self, radial_grid, l_max, m_max=None, *, Z, mass=1.0):
        self.kinetic = SphericalKineticEnergy(
            radial_grid, l_max, m_max, mass=mass
        )
        self.potential = SphericalCoulombPotential(
            radial_grid, l_max, m_max, Z=Z
        )
        self._set_metadata()

    @property
    def Z(self):
        """Nuclear charge fixed at construction."""
        return self.potential.Z
