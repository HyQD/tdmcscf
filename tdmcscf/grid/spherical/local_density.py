r"""Local and semilocal spin-unpolarized XC potentials on spherical grids."""

from dataclasses import dataclass

import numpy as np
from grid_lib.spherical_coordinates import (
    GaussLegendreFourierGrid,
    SphericalNodalDifferentialOperator,
)
from scipy.special import sph_harm_y

from ._angular import AngularTransform
from ._validation import _integer, _orbitals, _radial_data
from .xc import (
    _functional_values,
    _local_functional,
    _make_xc_quadrature,
    _quadrature_orders,
    _requires_gradient,
    evaluate_spherical_functional,
)


@dataclass(frozen=True)
class _SemilocalElement:
    """Frozen semilocal coefficients on one element quadrature."""

    quadrature: object
    density_derivative: np.ndarray
    flux_radial: np.ndarray
    flux_polar: np.ndarray
    flux_azimuthal: np.ndarray


class SphericalXCBuilder:
    r"""Evaluate and apply a local or semilocal XC functional on an angular grid.

    Orbital arrays contain reduced radial coefficients ``u_p,lm(r)``. The
    physical spin-summed density evaluated by this class is

    ``n(r, Omega) = 2 * sum_p |sum_lm u_p,lm(r) Y_lm(Omega)|**2 / r**2``.

    Nonlinear functionals are not band limited. ``angular_bandlimit``
    therefore controls angular oversampling rather than an XC-potential
    truncation. It defaults to ``4*l_max`` and must be at least ``2*l_max``.
    For a semilocal functional, ``xc_quadrature_order`` controls the
    Gauss--Legendre radial overintegration and defaults to twice the number
    of FEM-DVR points in each element.
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
        xc_quadrature_order=None,
    ):
        self.l_max = _integer(l_max, "l_max")
        self.m_max = self.l_max if m_max is None else _integer(m_max, "m_max")
        if self.m_max > self.l_max:
            raise ValueError("m_max cannot exceed l_max.")
        self.functional = _local_functional(functional)

        r, weights = _radial_data(radial_grid)
        self._radial_grid = radial_grid
        self._full_r = r.copy()
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
        self._differential = None
        self.xc_quadrature_order = None
        self._xc_quadrature = ()
        if _requires_gradient(self.functional):
            self._differential = SphericalNodalDifferentialOperator(
                radial_grid,
                self.angular_grid,
                radial_indices=slice(1, -1),
            )
            self.xc_quadrature_order = _quadrature_orders(
                xc_quadrature_order, radial_grid.n_points
            )
            self._xc_quadrature = _make_xc_quadrature(
                radial_grid, self.xc_quadrature_order
            )
            theta = self.angular_grid.theta[:, None]
            phi = self.angular_grid.phi[None, :]
            polar_basis = []
            azimuthal_basis = []
            for l, m in self.orbital_lm:
                _, derivatives = sph_harm_y(
                    l, m, theta, phi, diff_n=1
                )
                polar_basis.append(derivatives[..., 0])
                azimuthal_basis.append(
                    derivatives[..., 1]
                    / self._differential.sin_theta[:, None]
                )
            self._polar_basis = np.asarray(polar_basis)
            self._azimuthal_basis = np.asarray(azimuthal_basis)

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

    def _full_density(self, orbitals, density):
        """Restore regular endpoint density values for differentiation."""
        full_density = np.zeros(
            (len(self._full_r),) + self.angular_grid.shape
        )
        full_density[1:-1] = density
        s_channel = self.orbital_lm.index((0, 0))
        origin_slopes = (
            orbitals[:, :, s_channel] @ self._radial_grid.D1[0, 1:-1]
        )
        full_density[0] = (
            2.0 * np.sum(np.abs(origin_slopes) ** 2) / (4.0 * np.pi)
        )
        return full_density

    def _full_orbitals(self, orbitals):
        full = np.zeros(
            (len(orbitals), len(self._full_r), len(self.orbital_lm)),
            dtype=np.complex128,
        )
        full[:, 1:-1] = orbitals
        return full

    def _physical_orbitals(self, full_orbitals, quadrature):
        local = full_orbitals[:, quadrature.indices]
        coefficients = np.einsum(
            "qi,oic->oqc", quadrature.basis, local
        )
        radial_coefficients = np.einsum(
            "qi,oic->oqc", quadrature.basis_derivative, local
        )
        reduced = self._transform.synthesize(coefficients)
        reduced_radial = self._transform.synthesize(radial_coefficients)
        reduced_polar = np.einsum(
            "oqc,ctp->oqtp", coefficients, self._polar_basis
        )
        reduced_azimuthal = np.einsum(
            "oqc,ctp->oqtp", coefficients, self._azimuthal_basis
        )
        radius = quadrature.r[None, :, None, None]
        physical = reduced / radius
        return physical, (
            reduced_radial / radius - reduced / radius**2,
            reduced_polar / radius**2,
            reduced_azimuthal / radius**2,
        )

    def _semilocal_elements(self, orbitals):
        full_orbitals = self._full_orbitals(orbitals)
        elements = []
        energy = 0.0
        for quadrature in self._xc_quadrature:
            physical, gradient = self._physical_orbitals(
                full_orbitals, quadrature
            )
            density = 2.0 * np.sum(np.abs(physical) ** 2, axis=0)
            gradient_terms = tuple(
                4.0 * np.real(physical.conj() * component)
                for component in gradient
            )
            density_gradient = [
                np.sum(component, axis=0) for component in gradient_terms
            ]
            for index in (1, 2):
                cancellation_scale = np.sum(
                    np.abs(gradient_terms[index]), axis=0
                )
                roundoff = (
                    np.abs(density_gradient[index])
                    <= 100.0
                    * np.finfo(float).eps
                    * cancellation_scale
                )
                density_gradient[index] = np.where(
                    roundoff, 0.0, density_gradient[index]
                )
            density_gradient = tuple(density_gradient)
            gradient_squared = sum(
                component**2 for component in density_gradient
            )
            gradient_squared = np.maximum(gradient_squared, 0.0)
            evaluated = _functional_values(
                self.functional, density, gradient_squared
            )
            factor = 2.0 * np.asarray(evaluated.sigma_derivative)
            elements.append(
                _SemilocalElement(
                    quadrature,
                    np.asarray(evaluated.density_derivative),
                    factor * density_gradient[0],
                    factor * density_gradient[1],
                    factor * density_gradient[2],
                )
            )
            angular_energy = self.angular_grid.integrate(
                np.asarray(evaluated.energy_density)
            )
            energy += np.dot(
                quadrature.weights,
                quadrature.r**2 * angular_energy,
            )
        return tuple(elements), float(energy)

    def _derivative_projection(self, values, basis):
        return np.einsum(
            "oqab,cab,ab->oqc",
            values,
            basis.conj(),
            self.angular_grid.weights,
        )

    def _apply_semilocal(self, orbitals, elements):
        full_orbitals = self._full_orbitals(orbitals)
        weak_action = np.zeros_like(full_orbitals)
        for element in elements:
            quadrature = element.quadrature
            physical, gradient = self._physical_orbitals(
                full_orbitals, quadrature
            )
            flux = (
                element.flux_radial,
                element.flux_polar,
                element.flux_azimuthal,
            )
            local_action = element.density_derivative[None] * physical
            for component, coefficient in zip(gradient, flux, strict=True):
                local_action += coefficient[None] * component

            scalar_projection = self._transform.analyze(local_action)
            radial_projection = self._transform.analyze(
                element.flux_radial[None] * physical
            )
            polar_projection = self._derivative_projection(
                element.flux_polar[None] * physical,
                self._polar_basis,
            )
            azimuthal_projection = self._derivative_projection(
                element.flux_azimuthal[None] * physical,
                self._azimuthal_basis,
            )

            basis = quadrature.basis
            derivative = quadrature.basis_derivative
            weights = quadrature.weights
            radius = quadrature.r
            contribution = np.einsum(
                "qi,q,oqc->oic",
                basis,
                weights * radius,
                scalar_projection,
            )
            contribution += np.einsum(
                "qi,q,oqc->oic",
                derivative,
                weights * radius,
                radial_projection,
            )
            contribution -= np.einsum(
                "qi,q,oqc->oic",
                basis,
                weights,
                radial_projection,
            )
            contribution += np.einsum(
                "qi,q,oqc->oic",
                basis,
                weights,
                polar_projection + azimuthal_projection,
            )
            weak_action[:, quadrature.indices] += contribution
        return weak_action[:, 1:-1] / self.radial_weights[None, :, None]

    def build(self, orbitals):
        """Evaluate the density and XC energy density and Euler potential."""
        values, _ = self._orbitals(orbitals)
        density = self.density(values)
        full_density = None
        if self._differential is not None:
            full_density = self._full_density(values, density)
        evaluated = evaluate_spherical_functional(
            self.functional,
            density,
            full_density=full_density,
            differential_operator=self._differential,
        )
        semilocal_elements = ()
        if self._differential is None:
            energy = np.dot(
                self.radial_weights,
                self.r**2
                * self.angular_grid.integrate(evaluated.energy_density),
            )
        else:
            semilocal_elements, energy = self._semilocal_elements(values)
        return SphericalXCPotential(
            self,
            density,
            evaluated.energy_density,
            evaluated.potential,
            evaluated.gradient_squared,
            semilocal_elements,
            float(energy),
        )


class SphericalXCPotential:
    """A frozen local or weak-form semilocal XC operator."""

    def __init__(
        self,
        builder,
        density,
        energy_density,
        values,
        gradient_squared=None,
        semilocal_elements=(),
        energy=0.0,
    ):
        self._builder = builder
        self.density = density
        self.energy_density = energy_density
        self.values = values
        self.gradient_squared = gradient_squared
        self._semilocal_elements = semilocal_elements
        self.energy = energy

    def apply(self, orbitals):
        """Apply the frozen XC operator and project to the orbital basis."""
        orbitals, single = self._builder._orbitals(
            orbitals, allow_single=True
        )
        if self._semilocal_elements:
            result = self._builder._apply_semilocal(
                orbitals, self._semilocal_elements
            )
        else:
            angular_values = self._builder._transform.synthesize(orbitals)
            result = self._builder._transform.analyze(
                angular_values * self.values
            )
        return result[0] if single else result
