"""Spherical one-body Hamiltonians for time-independent central potentials."""

import numpy as np

from .core import (
    _fixed_m_metric_weights,
    _fixed_m_orbitals,
    _inner_product,
    _signed_m,
    _transform_fixed_m,
)
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

    def fixed_m_shape(self, m):
        """Return the radial-first shape of one signed-``m`` block."""

        m = _signed_m(m, self.m_max)
        return (len(self.r), self.l_max - abs(m) + 1)

    def apply_fixed_m(self, orbitals, m):
        """Apply the central core Hamiltonian in one signed-``m`` block."""

        m = _signed_m(m, self.m_max)
        shape = self.fixed_m_shape(m)
        values = _fixed_m_orbitals(orbitals, shape)
        result = np.empty(values.shape, dtype=np.complex128)
        for channel, l in enumerate(range(abs(m), self.l_max + 1)):
            result[..., :, channel] = np.einsum(
                "ab,...b->...a",
                self.radial_matrix(l),
                values[..., :, channel],
                optimize=True,
            )
        return result

    def fixed_m_metric_weights(self, m):
        """Return flattened radial-first metric entries for one block."""

        n_l = self.fixed_m_shape(m)[1]
        return _fixed_m_metric_weights(self.radial_weights, n_l)

    def fixed_m_matrix(self, m, *, representation="strong"):
        """Return a strong or Euclidean-Hermitian signed-``m`` matrix."""

        m = _signed_m(m, self.m_max)
        if representation not in {"strong", "symmetric"}:
            raise ValueError("representation must be 'strong' or 'symmetric'.")
        n_r, n_l = self.fixed_m_shape(m)
        matrix = np.zeros((n_r * n_l, n_r * n_l))
        for channel, l in enumerate(range(abs(m), self.l_max + 1)):
            indices = np.arange(n_r) * n_l + channel
            matrix[np.ix_(indices, indices)] = self.radial_matrix(l)
        if representation == "strong":
            return matrix
        root = np.sqrt(self.fixed_m_metric_weights(m))
        return root[:, None] * matrix / root[None, :]

    def to_symmetric_fixed_m(self, orbitals, m):
        """Map nodal fixed-``m`` values to the Euclidean representation."""

        return _transform_fixed_m(
            orbitals,
            self.fixed_m_shape(m),
            self.radial_weights,
            inverse=False,
        )

    def from_symmetric_fixed_m(self, orbitals, m):
        """Map Euclidean fixed-``m`` values to nodal values."""

        return _transform_fixed_m(
            orbitals,
            self.fixed_m_shape(m),
            self.radial_weights,
            inverse=True,
        )

    def inner_product(self, left, right):
        """Return the radial-quadrature inner product."""

        return _inner_product(left, right, self.radial_weights)


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
