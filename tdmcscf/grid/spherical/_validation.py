"""Shared validation for spherical nodal orbital arrays."""

from numbers import Real

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _orbitals(values, shape, allow_single=False):
    array = np.asarray(values)
    single = allow_single and array.ndim == 2
    if single:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1:] != shape or array.shape[0] == 0:
        raise ValueError(
            f"Orbitals must have shape (n_orbitals, {shape[0]}, {shape[1]})"
            + (f" or {shape}." if allow_single else ".")
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("Orbital values must be numeric.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Orbital values must be finite.")
    return np.asarray(array, dtype=np.complex128), single


def _positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real scalar.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real scalar.")
    return float(value)


def _radial_data(radial_grid):
    """Validate full real radial nodes and their quadrature weights."""
    r = np.asarray(radial_grid.r)
    weights = np.asarray(radial_grid.weights)
    if (
        r.ndim != 1
        or len(r) < 3
        or np.iscomplexobj(r)
        or not np.all(np.isfinite(r))
        or r[0] != 0
        or np.any(np.diff(r) <= 0)
    ):
        raise ValueError("The radial grid must be real, increasing, and start at zero.")
    if (
        weights.shape != r.shape
        or np.iscomplexobj(weights)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("Radial quadrature weights must be real and positive.")
    return r, weights
