"""Shared protocol and validation for spherical one-body core operators."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SphericalCoreOperator(Protocol):
    """Structural interface required from a spherical one-body core."""

    l_max: int
    m_max: int
    r: np.ndarray
    radial_weights: np.ndarray
    orbital_lm: tuple
    orbital_shape: tuple

    def apply(self, orbitals): ...

    def fixed_m_shape(self, m): ...

    def apply_fixed_m(self, orbitals, m): ...

    def fixed_m_matrix(self, m, *, representation="strong"): ...


def validate_spherical_core_operator(core):
    """Validate and return an object implementing the core protocol."""

    if not isinstance(core, SphericalCoreOperator):
        raise TypeError("core must implement the SphericalCoreOperator protocol.")
    r = np.asarray(core.r)
    weights = np.asarray(core.radial_weights)
    if r.ndim != 1 or weights.shape != r.shape or len(r) == 0:
        raise ValueError("core radial nodes and weights must be one-dimensional.")
    if core.orbital_shape != (len(r), len(core.orbital_lm)):
        raise ValueError("core orbital_shape is inconsistent with its metadata.")
    return core


def _signed_m(value, m_max):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("m must be an integer.")
    value = int(value)
    if abs(value) > m_max:
        raise ValueError("abs(m) cannot exceed m_max.")
    return value


def _fixed_m_orbitals(orbitals, shape):
    values = np.asarray(orbitals)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("orbitals must be numeric.")
    if values.ndim < 2 or values.shape[-2:] != shape:
        raise ValueError(f"orbitals must have trailing shape {shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("orbitals must contain only finite values.")
    return values


def _fixed_m_metric_weights(radial_weights, n_l):
    return np.repeat(np.asarray(radial_weights), n_l)


def _transform_fixed_m(orbitals, shape, radial_weights, *, inverse):
    values = _fixed_m_orbitals(orbitals, shape)
    weight_shape = (1,) * (values.ndim - 2) + (len(radial_weights), 1)
    root = np.sqrt(radial_weights).reshape(weight_shape)
    return values / root if inverse else values * root


def _inner_product(left, right, radial_weights):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("left and right must have the same shape.")
    if left.ndim < 2 or left.shape[-2] != len(radial_weights):
        raise ValueError("the penultimate axis must be the radial grid.")
    weight_shape = (1,) * (left.ndim - 2) + (len(radial_weights), 1)
    return np.sum(
        left.conj() * right * np.asarray(radial_weights).reshape(weight_shape)
    )
