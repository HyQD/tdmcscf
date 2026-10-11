"""Shared validation for spherical nodal orbital arrays."""

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
