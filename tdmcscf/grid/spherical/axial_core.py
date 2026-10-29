"""One-body core Hamiltonians for midpoint-centred axial systems."""

import numpy as np

from grid_lib.spherical_coordinates import (
    SphericalMultipolePotential,
    radial_Coulomb,
)

from tdmcscf.systems import AxialDiatomic

from ._validation import _integer
from .core import (
    _fixed_m_metric_weights,
    _fixed_m_orbitals,
    _inner_product,
    _signed_m,
    _transform_fixed_m,
)
from .kinetic import SphericalKineticEnergy


class AxialCoreHamiltonian:
    r"""Kinetic energy plus a clamped axial two-centre Coulomb potential."""

    _NUCLEAR_METHODS = {"analytic", "poisson"}

    def __init__(
        self,
        radial_grid,
        l_max,
        m_max=None,
        *,
        system,
        nuclear_L_max=None,
        nuclear_method="analytic",
        poisson_method="inverse",
    ):
        if not isinstance(system, AxialDiatomic):
            raise TypeError("system must be an AxialDiatomic.")
        self.system = system
        self.radial_grid = radial_grid
        self.kinetic = SphericalKineticEnergy(
            radial_grid,
            l_max,
            m_max,
            mass=system.mass,
        )
        self.l_max = self.kinetic.l_max
        self.m_max = self.kinetic.m_max
        self.nuclear_L_max = (
            2 * self.l_max
            if nuclear_L_max is None
            else _integer(nuclear_L_max, "nuclear_L_max")
        )
        self.nuclear_M_max = 0
        if nuclear_method not in self._NUCLEAR_METHODS:
            choices = ", ".join(sorted(self._NUCLEAR_METHODS))
            raise ValueError(f"nuclear_method must be one of: {choices}.")
        if poisson_method not in {"inverse", "solve"}:
            raise ValueError("poisson_method must be 'inverse' or 'solve'.")
        self.nuclear_method = nuclear_method
        self.poisson_method = poisson_method
        self._validate_nuclear_element_boundary()

        self.r = self.kinetic.r
        self.radial_weights = self.kinetic.radial_weights
        self.orbital_lm = self.kinetic.orbital_lm
        self.orbital_shape = self.kinetic.orbital_shape
        self.potential = self._make_potential()
        if self.potential.orbital_lm != self.orbital_lm:
            raise ValueError("kinetic and potential operators must use one basis.")

    @property
    def mass(self):
        return self.kinetic.mass

    @property
    def nuclear_repulsion_energy(self):
        return self.system.nuclear_repulsion_energy

    def _validate_nuclear_element_boundary(self):
        try:
            boundaries = np.asarray(self.radial_grid.nodes, dtype=float)
        except AttributeError as error:
            raise TypeError(
                "radial_grid must expose finite-element boundaries as nodes."
            ) from error
        radius = self.system.half_bond_length
        scale = max(1.0, radius, float(np.max(np.abs(boundaries))))
        tolerance = 64.0 * np.finfo(float).eps * scale
        if not np.any(
            np.isclose(
                boundaries[1:-1],
                radius,
                rtol=64.0 * np.finfo(float).eps,
                atol=tolerance,
            )
        ):
            raise ValueError(
                "half_bond_length must coincide with an internal FEM "
                "element boundary."
            )

    def _make_potential(self):
        arguments = dict(
            l_max=self.l_max,
            m_max=self.m_max,
            L_max=self.nuclear_L_max,
            M_max=self.nuclear_M_max,
        )
        if self.nuclear_method == "analytic":
            return SphericalMultipolePotential.from_point_charges(
                self.r,
                self.system.nuclear_positions,
                self.system.nuclear_charges,
                **arguments,
            )
        radial_coulomb = radial_Coulomb(
            self.radial_grid,
            self.nuclear_L_max + 1,
            method=self.poisson_method,
        )
        return SphericalMultipolePotential.from_point_charges_poisson(
            self.r,
            radial_coulomb,
            self.radial_weights,
            self.system.nuclear_positions,
            self.system.nuclear_charges,
            **arguments,
        )

    def apply(self, orbitals):
        """Apply the core Hamiltonian to full spherical orbital arrays."""

        return self.kinetic.apply(orbitals) + self.potential.apply(orbitals)

    def fixed_m_shape(self, m):
        """Return the radial-first shape of one signed-``m`` block."""

        m = _signed_m(m, self.m_max)
        return (len(self.r), self.l_max - abs(m) + 1)

    def apply_fixed_m(self, orbitals, m):
        """Apply one conserved signed-``m`` Hamiltonian block."""

        m = _signed_m(m, self.m_max)
        shape = self.fixed_m_shape(m)
        values = _fixed_m_orbitals(orbitals, shape)
        result = np.empty(values.shape, dtype=np.complex128)
        for channel, l in enumerate(range(abs(m), self.l_max + 1)):
            result[..., :, channel] = np.einsum(
                "ab,...b->...a",
                self.kinetic.radial_matrix(l),
                values[..., :, channel],
                optimize=True,
            )
        result += np.einsum(
            "rij,...rj->...ri",
            self.potential.fixed_m_matrix(m),
            values,
            optimize=True,
        )
        return result

    def fixed_m_metric_weights(self, m):
        """Return flattened radial-first metric entries for one block."""

        n_l = self.fixed_m_shape(m)[1]
        return _fixed_m_metric_weights(self.radial_weights, n_l)

    def fixed_m_matrix(self, m, *, representation="strong"):
        """Return a strong or Euclidean-Hermitian signed-``m`` matrix."""

        m = _signed_m(m, self.m_max)
        if representation not in {"strong", "symmetric"}:
            raise ValueError("representation must be 'strong' or 'symmetric'.")
        n_r, n_l = self.fixed_m_shape(m)
        dtype = np.result_type(
            self.potential.coefficients.dtype,
            np.complex128,
        )
        matrix = np.zeros((n_r * n_l, n_r * n_l), dtype=dtype)
        for channel, l in enumerate(range(abs(m), self.l_max + 1)):
            indices = np.arange(n_r) * n_l + channel
            matrix[np.ix_(indices, indices)] += self.kinetic.radial_matrix(l)
        pointwise = self.potential.fixed_m_matrix(m)
        for radial_index in range(n_r):
            indices = radial_index * n_l + np.arange(n_l)
            matrix[np.ix_(indices, indices)] += pointwise[radial_index]
        if representation == "strong":
            return matrix
        root = np.sqrt(self.fixed_m_metric_weights(m))
        return root[:, None] * matrix / root[None, :]

    def to_symmetric_fixed_m(self, orbitals, m):
        """Map nodal fixed-``m`` values to the Euclidean representation."""

        return _transform_fixed_m(
            orbitals,
            self.fixed_m_shape(m),
            self.radial_weights,
            inverse=False,
        )

    def from_symmetric_fixed_m(self, orbitals, m):
        """Map Euclidean fixed-``m`` values to nodal values."""

        return _transform_fixed_m(
            orbitals,
            self.fixed_m_shape(m),
            self.radial_weights,
            inverse=True,
        )

    def inner_product(self, left, right):
        """Return the radial-quadrature inner product."""

        return _inner_product(left, right, self.radial_weights)
