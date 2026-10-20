"""Validation and coordinate-specific evaluation of spherical XC functionals."""

from dataclasses import dataclass

import numpy as np
from grid_lib.spherical_coordinates import SphericalVectorField
from scipy.special import roots_legendre


@dataclass(frozen=True)
class _ElementQuadrature:
    """Overintegration data for one radial finite element."""

    indices: np.ndarray
    r: np.ndarray
    weights: np.ndarray
    basis: np.ndarray
    basis_derivative: np.ndarray


def _quadrature_orders(value, element_points):
    """Resolve one semilocal quadrature order per radial element."""
    element_points = np.asarray(element_points, dtype=int)
    if value is None:
        return 2 * element_points
    values = np.asarray(value)
    if values.ndim == 0:
        if isinstance(value, (bool, np.bool_)) or not np.issubdtype(
            values.dtype, np.integer
        ):
            raise TypeError("xc_quadrature_order must contain integers.")
        values = np.full(element_points.shape, int(values))
    elif values.shape == element_points.shape and np.issubdtype(
        values.dtype, np.integer
    ):
        values = values.astype(int)
    else:
        raise ValueError(
            "xc_quadrature_order must be one integer or one integer per "
            "element."
        )
    if np.any(values < element_points):
        raise ValueError(
            "xc_quadrature_order must be at least n_points in every element."
        )
    return values


def _lagrange_table(nodes, barycentric_weights, points, radial_scale):
    differences = points[:, None] - nodes[None, :]
    terms = barycentric_weights[None, :] / differences
    basis = terms / np.sum(terms, axis=1)[:, None]
    logarithmic_derivative = np.sum(basis / differences, axis=1)
    basis_derivative = basis * (
        logarithmic_derivative[:, None] - 1.0 / differences
    )
    return basis, basis_derivative / radial_scale


def _make_xc_quadrature(grid, orders):
    """Build element-local Gauss--Legendre overintegration tables."""
    quadratures = []
    for element, (dvr, order) in enumerate(
        zip(grid.dvr, orders, strict=True)
    ):
        points, weights = roots_legendre(int(order))
        lower = grid.nodes[element]
        upper = grid.nodes[element + 1]
        radial_scale = 0.5 * (upper - lower)
        r = lower + radial_scale * (points + 1.0)
        weights = radial_scale * weights
        basis, basis_derivative = _lagrange_table(
            dvr.x,
            1.0 / dvr.PN_x2,
            points,
            radial_scale,
        )
        start = grid.edge_indices[element]
        stop = grid.edge_indices[element + 1] + 1
        quadratures.append(
            _ElementQuadrature(
                np.arange(start, stop),
                r,
                weights,
                basis,
                basis_derivative,
            )
        )
    return tuple(quadratures)


@dataclass(frozen=True)
class SphericalFunctionalValues:
    """XC energy density, Euler potential, and squared density gradient."""

    energy_density: np.ndarray
    potential: np.ndarray
    gradient_squared: np.ndarray | None


def _requires_gradient(functional):
    return bool(getattr(functional, "requires_gradient", False))


def _checked_array(values, shape, description):
    array = np.asarray(values)
    if (
        array.shape != shape
        or np.iscomplexobj(array)
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            f"functional.evaluate must return a finite real {description} "
            "array matching density."
        )
    return array.astype(float, copy=False)


def _functional_values(functional, density, gradient_squared=None):
    """Evaluate and validate a local or semilocal functional."""
    density = np.asarray(density)
    if _requires_gradient(functional):
        if gradient_squared is None:
            raise ValueError(
                "A semilocal functional requires gradient_squared."
            )
        gradient_squared = np.asarray(gradient_squared)
        if gradient_squared.shape != density.shape:
            raise ValueError("gradient_squared must match density.")
        values = functional.evaluate(density, gradient_squared)
        for name, description in (
            ("energy_density", "energy_density"),
            ("density_derivative", "density_derivative"),
            ("sigma_derivative", "sigma_derivative"),
        ):
            _checked_array(getattr(values, name, None), density.shape, description)
    else:
        values = functional.evaluate(density)
        for name in ("energy_density", "potential"):
            _checked_array(getattr(values, name, None), density.shape, name)
    return values


def _local_functional(functional):
    """Validate the established local-or-semilocal functional protocol."""
    if not callable(getattr(functional, "evaluate", None)):
        raise TypeError("functional must provide evaluate(...).")
    density = np.array([0.0, 1.0])
    if _requires_gradient(functional):
        _functional_values(functional, density, np.zeros_like(density))
    else:
        _functional_values(functional, density)
    return functional


def _real_potential(values):
    values = np.asarray(values)
    if np.iscomplexobj(values):
        scale = max(1.0, float(np.max(np.abs(values.real))))
        if np.max(np.abs(values.imag)) > 5e-12 * scale:
            raise RuntimeError(
                "The XC potential acquired a non-negligible imaginary part."
            )
        values = values.real
    if not np.all(np.isfinite(values)):
        raise RuntimeError("The XC potential contains non-finite values.")
    return np.asarray(values, dtype=float)


def evaluate_spherical_functional(
    functional,
    density,
    *,
    full_density=None,
    differential_operator=None,
):
    """Construct a local or semilocal XC Euler potential.

    ``density`` is defined on the active positive-radius nodes. Semilocal
    evaluation additionally requires ``full_density`` and a compatible
    spherical nodal differential operator. The weighted gradient adjoint
    supplies the discrete counterpart of the negative divergence.
    """
    density = np.asarray(density)
    if not _requires_gradient(functional):
        values = _functional_values(functional, density)
        return SphericalFunctionalValues(
            _checked_array(
                values.energy_density, density.shape, "energy_density"
            ),
            _checked_array(values.potential, density.shape, "potential"),
            None,
        )

    if differential_operator is None or full_density is None:
        raise ValueError(
            "Semilocal evaluation requires full_density and "
            "differential_operator."
        )
    gradient = differential_operator.gradient(full_density)
    gradient_squared = differential_operator.gradient_norm_squared(full_density)
    gradient_squared = _real_potential(gradient_squared)
    values = _functional_values(functional, density, gradient_squared)
    factor = 2.0 * np.asarray(values.sigma_derivative)
    flux = SphericalVectorField(
        factor * gradient.radial,
        factor * gradient.polar,
        factor * gradient.azimuthal,
    )
    potential = np.asarray(values.density_derivative) + (
        differential_operator.gradient_adjoint(flux)
    )
    return SphericalFunctionalValues(
        _checked_array(values.energy_density, density.shape, "energy_density"),
        _real_potential(potential),
        gradient_squared,
    )
