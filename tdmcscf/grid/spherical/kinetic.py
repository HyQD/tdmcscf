r"""Action of -laplacian/(2*mass) on reduced radial nodal orbitals."""

from numbers import Real

import numpy as np

from ._validation import _integer, _orbitals


class SphericalKineticEnergy:
    r"""Spherical kinetic energy for a fixed real radial grid in atomic units.

    Parameters
    ----------
    radial_grid
        Grid with full ``r``, ``weights``, and nodal ``D2`` arrays, such as
        FEMDVR constructed with ``symmetrize=False``. It must start at zero.
        The first and last nodes are removed, imposing zero reduced radial
        orbitals at both endpoints. Use the nodal D2, not D2_symmetric.
    l_max : int
        Maximum orbital angular momentum.
    m_max : int, optional
        Retain abs(m) <= m_max. Defaults to l_max.
    mass : positive real scalar, optional
        Particle or reduced mass in electron-mass units. Defaults to 1.
        The value is fixed at construction and scales both radial and
        centrifugal contributions.

    Notes
    -----
    For phi = sum_lm u_lm(r)*Y_lm(Omega)/r, apply returns the reduced radial
    coefficients of T*phi:

        (Tu)_a,lm = -sum_b D2_ab*u_b,lm/(2*mass)
                    + l*(l+1)*u_a,lm/(2*mass*r_a**2).

    ``orbital_lm`` uses grid-lib's M-major LM_to_I ordering. Arrays contain
    nodal values with no absorbed quadrature weights. The quadrature metric
    is diag(radial_weights), with respect to which the kinetic operator is
    Hermitian. No additional overlap inverse or quadrature weights are
    applied: grid-lib's nodal D2 already includes the overlap inverse.

    The operator preserves every (l,m) channel, so general and fixed-m
    orbitals use the same implementation. Construction allocates one radial
    matrix and a radial/channel centrifugal array, without angular transforms
    or Coulomb matrices. No potential, orbital Q projector, or time-evolution
    factor is included.
    """

    def __init__(self, radial_grid, l_max, m_max=None, *, mass=1.0):
        self.l_max = _integer(l_max, "l_max")
        self.m_max = self.l_max if m_max is None else _integer(m_max, "m_max")
        if self.m_max > self.l_max:
            raise ValueError("m_max cannot exceed l_max.")
        if isinstance(mass, (bool, np.bool_)) or not isinstance(mass, Real):
            raise TypeError("mass must be a positive finite real scalar.")
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("mass must be a positive finite real scalar.")
        self._mass = float(mass)

        r = np.asarray(radial_grid.r)
        weights = np.asarray(radial_grid.weights)
        D2 = np.asarray(radial_grid.D2)
        if (
            r.ndim != 1
            or len(r) < 3
            or np.iscomplexobj(r)
            or not np.all(np.isfinite(r))
            or r[0] != 0
            or np.any(np.diff(r) <= 0)
        ):
            raise ValueError(
                "The radial grid must be real, increasing, and start at zero."
            )
        if (
            weights.shape != r.shape
            or np.iscomplexobj(weights)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError("Radial quadrature weights must be real and positive.")
        if (
            D2.shape != (len(r), len(r))
            or np.iscomplexobj(D2)
            or not np.all(np.isfinite(D2))
        ):
            raise ValueError(
                "D2 must be a finite real nodal matrix on the full radial grid."
            )

        self.r = r[1:-1].copy()
        self.radial_weights = weights[1:-1].copy()
        self.orbital_lm = tuple(
            (l, m)
            for m in range(-self.m_max, self.m_max + 1)
            for l in range(abs(m), self.l_max + 1)
        )
        self.orbital_shape = (len(self.r), len(self.orbital_lm))
        angular = np.array([l * (l + 1) for l, _ in self.orbital_lm])
        self._radial = -D2[1:-1, 1:-1].copy() / (2 * self.mass)
        self._centrifugal = angular[None, :] / (2 * self.mass * self.r[:, None] ** 2)

    @property
    def mass(self):
        """Fixed particle or reduced mass in electron-mass units."""
        return self._mass

    def apply(self, orbitals):
        """Return Tu for one orbital or a batch, preserving the input shape.

        Accept (n_r, n_lm) or (n_orbitals, n_r, n_lm) nodal arrays.
        Real and complex inputs are supported; output is complex128.
        Input arrays are not modified.
        """
        values, single = _orbitals(orbitals, self.orbital_shape, allow_single=True)
        result = self._radial @ values + self._centrifugal * values
        return result[0] if single else result
