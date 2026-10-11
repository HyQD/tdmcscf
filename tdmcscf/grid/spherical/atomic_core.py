"""Atomic core Hamiltonian for a central point nucleus."""

import numpy as np

from .kinetic import SphericalKineticEnergy
from .potentials import SphericalCoulombPotential


class AtomicCoreHamiltonian:
    """Combine kinetic energy and nuclear attraction in atomic units.

    Construct with ``(radial_grid, l_max, m_max=None, *, Z, mass=1.0)``.
    ``Z`` is a required positive finite real charge; ``mass`` is a positive
    finite particle or reduced mass in electron-mass units. The fixed real
    grid and zero endpoint convention match ``SphericalKineticEnergy``.

    The independently usable ``kinetic`` and ``potential`` operators act on
    reduced radial nodal values. Hcore = -laplacian/(2*mass) - Z/r preserves
    every angular channel. ``radial_matrix(l)`` returns its nodal radial
    block, independent of m, without allocating a full angular/radial matrix.
    No mean field, orbital projector, or time-evolution factor is included.
    """

    def __init__(self, radial_grid, l_max, m_max=None, *, Z, mass=1.0):
        self.potential = SphericalCoulombPotential(radial_grid, l_max, m_max, Z=Z)
        self.kinetic = SphericalKineticEnergy(radial_grid, l_max, m_max, mass=mass)
        self.l_max = self.kinetic.l_max
        self.m_max = self.kinetic.m_max
        self.r = self.kinetic.r
        self.radial_weights = self.kinetic.radial_weights
        self.orbital_lm = self.kinetic.orbital_lm
        self.orbital_shape = self.kinetic.orbital_shape

    @property
    def Z(self):
        """Nuclear charge fixed at construction."""
        return self.potential.Z

    @property
    def mass(self):
        """Particle or reduced mass fixed at construction."""
        return self.kinetic.mass

    def apply(self, orbitals):
        """Return Hcore*u with the same single-orbital or batch shape.

        Accept (n_r, n_lm) or (n_orbitals, n_r, n_lm) nodal arrays.
        Real and complex inputs are supported; output is complex128.
        Input arrays are not modified.
        """
        return self.kinetic.apply(orbitals) + self.potential.apply(orbitals)

    def radial_matrix(self, l):
        """Return an independent nodal radial block for 0 <= l <= l_max.

        The quadrature metric S = diag(radial_weights) satisfies H† S = S H.
        For a Hermitian eigensolver, first transform to sqrt(S) H invsqrt(S).
        """
        matrix = self.kinetic.radial_matrix(l)
        matrix[np.diag_indices_from(matrix)] += self.potential.values
        return matrix
