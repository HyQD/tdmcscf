"""Shared validation helpers for time integrators."""

from numbers import Real

import numpy as np


def finite_real(value, name):
    """Return *value* as float after validating a finite real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    return float(value)


def positive_real(value, name, *, allow_zero=False):
    """Return a validated positive (or nonnegative) real scalar."""
    value = finite_real(value, name)
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def state_array(state):
    """Return a finite complex state array without modifying the input."""
    array = np.asarray(state)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("state must be a numeric array.")
    if array.ndim == 0:
        raise ValueError("state must have at least one dimension.")
    if not np.all(np.isfinite(array)):
        raise ValueError("state must contain only finite values.")
    return np.asarray(array, dtype=np.complex128)


def matching_array(value, shape, name):
    """Validate a finite numeric array with a prescribed shape."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must return a numeric array.")
    if array.shape != shape:
        raise ValueError(f"{name} must preserve the state shape {shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must return only finite values.")
    return np.asarray(array, dtype=np.complex128)
