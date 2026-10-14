"""Central one-body potentials acting on reduced radial nodal orbitals."""

import numpy as np
from grid_lib.spherical_coordinates.potentials import Coulomb

from ._validation import _integer, _orbitals, _positive_real, _radial_data


class SphericalRadialPotential:
    """Nodal multiplication by a real, time-independent central potential.

    ``potential`` is called with the positive interior radial nodes and must
    return a finite real scalar or an array with the same shape. Arrays contain
    nodal reduced radial values, with no absorbed quadrature weights.
    """

    def __init__(self, radial_grid, l_max, m_max=None, *, potential):
        self.l_max = _integer(l_max, "l_max")
        self.m_max = self.l_max if m_max is None else _integer(m_max, "m_max")
        if self.m_max > self.l_max:
            raise ValueError("m_max cannot exceed l_max.")
        if not callable(potential):
            raise TypeError("potential must be callable as potential(r).")
        r, weights = _radial_data(radial_grid)
        self.r = r[1:-1].copy()
        self.radial_weights = weights[1:-1].copy()
        self.orbital_lm = tuple(
            (l, m)
            for m in range(-self.m_max, self.m_max + 1)
            for l in range(abs(m), self.l_max + 1)
        )
        self.orbital_shape = (len(self.r), len(self.orbital_lm))
        values = np.asarray(potential(self.r))
        if values.ndim == 0:
            values = np.full(self.r.shape, values)
        if values.shape != self.r.shape:
            raise ValueError(
                "potential(r) must return a scalar or an array matching r."
            )
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError("potential(r) must return finite real values.")
        self.values = values.astype(float)
        self.potential = potential

    def apply(self, orbitals):
        """Return ``V*u`` while preserving a single or batched input shape."""
        values, single = _orbitals(orbitals, self.orbital_shape, allow_single=True)
        result = self.values[:, None] * values
        return result[0] if single else result


class SphericalCoulombPotential(SphericalRadialPotential):
    """Nuclear attraction ``-Z/r`` for a nucleus at the origin."""

    def __init__(self, radial_grid, l_max, m_max=None, *, Z):
        self._Z = _positive_real(Z, "Z")
        super().__init__(
            radial_grid,
            l_max,
            m_max,
            potential=Coulomb(self.Z),
        )

    @property
    def Z(self):
        """Nuclear charge fixed at construction."""
        return self._Z
