"""Central one-body potentials acting on reduced radial nodal orbitals."""

from grid_lib.spherical_coordinates.potentials import Coulomb

from ._validation import _integer, _orbitals, _positive_real, _radial_data


class SphericalCoulombPotential:
    """Nuclear attraction -Z/r for a nucleus at the origin, in atomic units.

    Parameters
    ----------
    radial_grid
        Fixed real radial grid with full ``r`` and ``weights`` arrays,
        starting at zero. Both endpoints are removed, as for kinetic energy.
        No derivative matrix is required.
    l_max : int
        Maximum orbital angular momentum.
    m_max : int, optional
        Retain abs(m) <= m_max. Defaults to l_max.
    Z : positive real scalar
        Required nuclear charge; noninteger effective charges are allowed.

    Notes
    -----
    ``values`` contains -Z/r at the interior nodes. Multiplication preserves
    every (l,m) channel in grid-lib's M-major ordering. Arrays contain nodal
    reduced radial values, with no absorbed quadrature weights. No angular
    transform, mass factor, or extra overlap inverse is needed. The excluded
    origin is never evaluated, so no regularization of the Coulomb potential
    is introduced.
    """

    def __init__(self, radial_grid, l_max, m_max=None, *, Z):
        self.l_max = _integer(l_max, "l_max")
        self.m_max = self.l_max if m_max is None else _integer(m_max, "m_max")
        if self.m_max > self.l_max:
            raise ValueError("m_max cannot exceed l_max.")
        self._Z = _positive_real(Z, "Z")
        r, weights = _radial_data(radial_grid)
        self.r = r[1:-1].copy()
        self.radial_weights = weights[1:-1].copy()
        self.orbital_lm = tuple(
            (l, m)
            for m in range(-self.m_max, self.m_max + 1)
            for l in range(abs(m), self.l_max + 1)
        )
        self.orbital_shape = (len(self.r), len(self.orbital_lm))
        self.values = Coulomb(self.Z)(self.r)

    @property
    def Z(self):
        """Nuclear charge fixed at construction."""
        return self._Z

    def apply(self, orbitals):
        """Return Vu for (n_r, n_lm) or (n_orbitals, n_r, n_lm) arrays.

        Preserve the input shape, return complex128, and leave inputs unchanged.
        """
        values, single = _orbitals(orbitals, self.orbital_shape, allow_single=True)
        result = self.values[:, None] * values
        return result[0] if single else result
