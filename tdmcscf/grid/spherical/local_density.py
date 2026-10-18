r"""Local spin-unpolarized XC potentials in spherical coordinates."""

import numpy as np
from grid_lib.spherical_coordinates import GaussLegendreFourierGrid

from ._angular import AngularTransform
from ._validation import _integer, _orbitals, _radial_data
from .kohn_sham import _functional_values, _local_functional


class SphericalXCBuilder:
    r"""Evaluate and apply a local XC functional on an angular grid.

    Orbital arrays contain reduced radial coefficients ``u_p,lm(r)``. The
    physical spin-summed density evaluated by this class is

    ``n(r, Omega) = 2 * sum_p |sum_lm u_p,lm(r) Y_lm(Omega)|**2 / r**2``.

    Local nonlinear functionals are not band limited. ``angular_bandlimit``
    therefore controls numerical oversampling rather than an XC-potential
    truncation. It defaults to ``4*l_max`` and must be at least ``2*l_max``.
    """

    def __init__(
        self,
        radial_grid,
        l_max,
        m_max=None,
        *,
        functional,
        angular_grid=None,
        angular_bandlimit=None,
    ):
        self.l_max = _integer(l_max, "l_max")
        self.m_max = self.l_max if m_max is None else _integer(m_max, "m_max")
        if self.m_max > self.l_max:
            raise ValueError("m_max cannot exceed l_max.")
        self.functional = _local_functional(functional)

        r, weights = _radial_data(radial_grid)
        self.r = r[1:-1].copy()
        self.radial_weights = weights[1:-1].copy()

        if angular_grid is not None and angular_bandlimit is not None:
            raise ValueError(
                "Supply angular_grid or angular_bandlimit, but not both."
            )
        if angular_grid is None:
            if angular_bandlimit is None:
                angular_bandlimit = 4 * self.l_max
            angular_bandlimit = _integer(
                angular_bandlimit, "angular_bandlimit"
            )
            if angular_bandlimit < 2 * self.l_max:
                raise ValueError("angular_bandlimit must be at least 2*l_max.")
            angular_grid = GaussLegendreFourierGrid.from_bandlimit(
                angular_bandlimit
            )
        elif not isinstance(angular_grid, GaussLegendreFourierGrid):
            raise TypeError("angular_grid must be a GaussLegendreFourierGrid.")

        if (
            angular_grid.n_theta < 2 * self.l_max + 1
            or angular_grid.n_phi < 4 * self.m_max + 1
        ):
            raise ValueError(
                "angular_grid is too small for orbital-density construction "
                "and projected local-potential action."
            )
        self.angular_grid = angular_grid
        self.angular_bandlimit = angular_grid.n_theta - 1
        self._transform = AngularTransform(
            self.l_max, self.m_max, self.angular_grid
        )
        self.orbital_lm = self._transform.channels
        self.orbital_shape = (len(self.r), len(self.orbital_lm))

    def _orbitals(self, orbitals, *, allow_single=False):
        return _orbitals(
            orbitals,
            self.orbital_shape,
            allow_single=allow_single,
        )

    def density(self, orbitals):
        """Return ``n(r, theta, phi)`` for occupied spatial orbitals."""
        values, _ = self._orbitals(orbitals)
        angular_values = self._transform.synthesize(values)
        return 2.0 * np.sum(np.abs(angular_values) ** 2, axis=0) / (
            self.r[:, None, None] ** 2
        )

    def build(self, orbitals):
        """Evaluate the density and local XC energy density and potential."""
        density = self.density(orbitals)
        evaluated = _functional_values(self.functional, density)
        return SphericalXCPotential(
            self,
            density,
            np.asarray(evaluated.energy_density, dtype=float),
            np.asarray(evaluated.potential, dtype=float),
        )


class SphericalXCPotential:
    """A local XC potential evaluated from one fixed orbital density."""

    def __init__(self, builder, density, energy_density, values):
        self._builder = builder
        self.density = density
        self.energy_density = energy_density
        self.values = values

    def apply(self, orbitals):
        """Multiply by the fixed local potential and project to the basis."""
        orbitals, single = self._builder._orbitals(
            orbitals, allow_single=True
        )
        angular_values = self._builder._transform.synthesize(orbitals)
        result = self._builder._transform.analyze(
            angular_values * self.values
        )
        return result[0] if single else result
