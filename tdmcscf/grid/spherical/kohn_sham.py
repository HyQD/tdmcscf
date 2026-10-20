r"""Spin-restricted Kohn--Sham ground states in central potentials."""

from dataclasses import dataclass

import numpy as np
from grid_lib.pseudospectral_grids.femdvr import FEMDVR

from tdmcscf.dft import LDAVWN
from tdmcscf.systems import (
    Atomic,
    AtomicConfiguration,
    SphericalElectronicSystem,
    closed_shell_atom,
)

from .atomic_core import AtomicCoreHamiltonian
from .hartree_fock import SCFIteration, SphericalHartreeFock
from .xc import (
    _functional_values,
    _local_functional,
    _make_xc_quadrature,
    _quadrature_orders,
    _requires_gradient,
    evaluate_spherical_functional,
)


@dataclass
class KohnShamResult:
    """Converged restricted Kohn--Sham state and energy decomposition."""

    system: SphericalElectronicSystem
    grid: FEMDVR
    energy: float
    orbital_energies: np.ndarray
    orbitals: np.ndarray
    orbital_quantum_numbers: tuple[tuple[int, int, int], ...]
    radial_orbitals: dict[int, np.ndarray]
    radial_orbital_energies: dict[int, np.ndarray]
    converged: bool
    iterations: int
    history: tuple[SCFIteration, ...]
    l_max: int
    m_max: int
    coulomb_l_max: int
    poisson_method: str
    functional: object
    initial_guess: str
    xc_quadrature_order: np.ndarray | None
    density: np.ndarray
    radial_density: np.ndarray
    density_gradient_squared: np.ndarray | None
    hartree_potential: np.ndarray
    xc_potential: np.ndarray
    xc_matrix: np.ndarray
    kinetic_energy: float
    external_energy: float
    hartree_energy: float
    xc_energy: float


@dataclass
class AtomicKohnShamResult(KohnShamResult):
    """Atomic Kohn--Sham result retaining the spectroscopic configuration."""

    configuration: AtomicConfiguration


class SphericalKohnSham(SphericalHartreeFock):
    r"""Spin-unpolarized closed-shell Kohn--Sham solver.

    The complete-shell occupation makes the density and effective potential
    spherical.  One radial Kohn--Sham matrix is therefore diagonalized for
    each occupied angular momentum ``l``:

    ``H_l = T_l + V_ext + V_H[n] + V_xc[n]``.

    ``functional`` is a coordinate-independent local or semilocal functional
    acting on the total spin-summed density. Local functionals return their
    pointwise energy density and potential. Semilocal functionals return the
    partial derivatives used to construct a quadrature-consistent Euler
    potential. The built-in choices live in :mod:`tdmcscf.dft`.

    The remaining grid, representation, and SCF controls have the same
    meaning as for :class:`SphericalHartreeFock`.  Only the ``L=0`` Coulomb
    response is required for the spherical Hartree potential.
    """

    result_type = KohnShamResult

    def __init__(
        self,
        system,
        *,
        functional,
        r_max=None,
        nodes=None,
        n_elements=None,
        n_points=15,
        l_max=None,
        m_max=None,
        poisson_method="inverse",
        max_iterations=100,
        conv_tol_e=1e-10,
        conv_tol_grad=1e-8,
        damping=0.2,
        diis=True,
        diis_start=3,
        diis_size=8,
        initial_guess="auto",
        xc_quadrature_order=None,
    ):
        self.functional = _local_functional(functional)
        if not isinstance(initial_guess, str):
            raise TypeError("initial_guess must be 'auto', 'core', or 'lda'.")
        normalized_guess = initial_guess.lower().replace("_", "-")
        if normalized_guess not in {"auto", "core", "lda"}:
            raise ValueError("initial_guess must be 'auto', 'core', or 'lda'.")
        if normalized_guess == "auto":
            normalized_guess = (
                "lda" if _requires_gradient(self.functional) else "core"
            )
        self.initial_guess = normalized_guess
        self._last_ks_state = None
        super().__init__(
            system,
            r_max=r_max,
            nodes=nodes,
            n_elements=n_elements,
            n_points=n_points,
            l_max=l_max,
            m_max=m_max,
            coulomb_l_max=0,
            poisson_method=poisson_method,
            max_iterations=max_iterations,
            conv_tol_e=conv_tol_e,
            conv_tol_grad=conv_tol_grad,
            damping=damping,
            diis=diis,
            diis_start=diis_start,
            diis_size=diis_size,
        )
        self.xc_quadrature_order = None
        self._xc_quadrature = ()
        if _requires_gradient(self.functional):
            self.xc_quadrature_order = _quadrature_orders(
                xc_quadrature_order, self.grid.n_points
            )
            self._xc_quadrature = _make_xc_quadrature(
                self.grid, self.xc_quadrature_order
            )

    def _initial_orbitals(self):
        if self.initial_guess == "core":
            return super()._initial_orbitals()

        gradient_tolerance = self.conv_tol_grad
        if gradient_tolerance is not None:
            gradient_tolerance = max(gradient_tolerance, 1e-8)
        lda_solver = SphericalKohnSham(
            self.system,
            functional=LDAVWN(),
            nodes=self.grid.nodes,
            n_points=self.grid.n_points,
            l_max=self.l_max,
            m_max=self.m_max,
            poisson_method=self.poisson_method,
            max_iterations=self.max_iterations,
            conv_tol_e=self.conv_tol_e,
            conv_tol_grad=gradient_tolerance,
            damping=self.damping,
            diis=self.diis,
            diis_start=self.diis_start,
            diis_size=self.diis_size,
            initial_guess="core",
        )
        lda_result = lda_solver.solve()
        return {
            l: values * self._sqrt_weights[None, :]
            for l, values in lda_result.radial_orbitals.items()
        }

    def _checked_radial_orbitals(self, radial_orbitals):
        if not isinstance(radial_orbitals, dict):
            raise TypeError("radial_orbitals must be a dictionary keyed by l.")
        if set(radial_orbitals) != set(self._counts_by_l):
            raise ValueError("radial_orbitals must contain every occupied l.")
        checked = {}
        for l, count in self._counts_by_l.items():
            values = np.asarray(radial_orbitals[l])
            expected = (count, len(self.r))
            if values.shape != expected:
                raise ValueError(f"radial_orbitals[{l}] must have shape {expected}.")
            if not np.issubdtype(values.dtype, np.number) or not np.all(
                np.isfinite(values)
            ):
                raise ValueError("Radial orbitals must be finite numeric arrays.")
            checked[l] = np.asarray(values, dtype=np.complex128)
        return checked

    def radial_density(self, radial_orbitals):
        r"""Return ``P(r) = 4*pi*r**2*n(r)`` on the interior radial nodes."""
        checked = self._checked_radial_orbitals(radial_orbitals)
        density = np.zeros(len(self.r))
        for l, orbitals in checked.items():
            density += 2.0 * (2 * l + 1) * np.sum(np.abs(orbitals) ** 2, axis=0)
        return density

    def density(self, radial_orbitals):
        """Return the physical spin-summed density ``n(r)``."""
        return self.radial_density(radial_orbitals) / (4.0 * np.pi * self.r**2)

    def _radial_density_derivative(self, radial_orbitals, radial_density):
        """Differentiate the physical density through reduced orbitals."""
        derivative = self.grid.D1[1:-1, 1:-1]
        radial_density_derivative = np.zeros(len(self.r))
        for l, orbitals in radial_orbitals.items():
            orbital_derivatives = orbitals @ derivative.T
            radial_density_derivative += 4.0 * (2 * l + 1) * np.sum(
                np.real(orbitals.conj() * orbital_derivatives), axis=0
            )
        return (
            radial_density_derivative - 2.0 * radial_density / self.r
        ) / (4.0 * np.pi * self.r**2)

    def _semilocal_xc(self, radial_orbitals, radial_density, density):
        """Return the weak-form GGA matrix and diagnostic Euler potential."""
        full_orbitals = {}
        for l, orbitals in radial_orbitals.items():
            full = np.zeros(
                (len(orbitals), len(self.grid.r)), dtype=np.complex128
            )
            full[:, 1:-1] = orbitals
            full_orbitals[l] = full

        weak_matrix = np.zeros((len(self.grid.r), len(self.grid.r)))
        energy = 0.0
        for quadrature in self._xc_quadrature:
            radial = np.zeros(len(quadrature.r))
            radial_derivative = np.zeros(len(quadrature.r))
            for l, orbitals in full_orbitals.items():
                local = orbitals[:, quadrature.indices]
                values = local @ quadrature.basis.T
                derivatives = local @ quadrature.basis_derivative.T
                degeneracy = 2.0 * (2 * l + 1)
                radial += degeneracy * np.sum(np.abs(values) ** 2, axis=0)
                radial_derivative += 2.0 * degeneracy * np.sum(
                    np.real(values.conj() * derivatives), axis=0
                )

            quadrature_density = radial / (4.0 * np.pi * quadrature.r**2)
            quadrature_density_derivative = (
                radial_derivative - 2.0 * radial / quadrature.r
            ) / (4.0 * np.pi * quadrature.r**2)
            evaluated = _functional_values(
                self.functional,
                quadrature_density,
                quadrature_density_derivative**2,
            )
            density_part = np.asarray(evaluated.density_derivative)
            flux = (
                2.0
                * np.asarray(evaluated.sigma_derivative)
                * quadrature_density_derivative
            )
            basis = quadrature.basis
            basis_derivative = quadrature.basis_derivative
            weights = quadrature.weights
            local_matrix = basis.T @ (
                (weights * density_part)[:, None] * basis
            )
            local_matrix += basis_derivative.T @ (
                (weights * flux)[:, None] * basis
            )
            local_matrix += basis.T @ (
                (weights * flux)[:, None] * basis_derivative
            )
            local_matrix -= basis.T @ (
                (2.0 * weights * flux / quadrature.r)[:, None] * basis
            )
            weak_matrix[np.ix_(quadrature.indices, quadrature.indices)] += (
                local_matrix
            )
            energy += np.dot(
                weights,
                4.0
                * np.pi
                * quadrature.r**2
                * np.asarray(evaluated.energy_density),
            )

        matrix = (
            weak_matrix[1:-1, 1:-1] / self.radial_weights[:, None]
        )

        # Evaluate nodal quantities for diagnostics and sampled output.
        density_derivative = self._radial_density_derivative(
            radial_orbitals, radial_density
        )
        gradient_squared = density_derivative**2
        values = _functional_values(
            self.functional, density, gradient_squared
        )
        density_part = np.asarray(values.density_derivative)
        sigma_part = np.asarray(values.sigma_derivative)
        flux = 2.0 * sigma_part * density_derivative

        # Retain a pointwise strong-form Euler potential for diagnostics. The
        # SCF equations use the symmetric weak matrix constructed above.
        full_radial_flux = np.zeros(len(self.grid.r))
        full_radial_flux[1:-1] = self.r**2 * flux
        potential = density_part - (
            self.grid.D1[1:-1] @ full_radial_flux
        ) / self.r**2
        return values, gradient_squared, potential, matrix, float(energy)

    def _assemble(self, radial_orbitals):
        checked = self._checked_radial_orbitals(radial_orbitals)
        radial_density = np.zeros(len(self.r))
        for l, orbitals in checked.items():
            radial_density += (
                2.0 * (2 * l + 1) * np.sum(np.abs(orbitals) ** 2, axis=0)
            )
        density = radial_density / (4.0 * np.pi * self.r**2)

        # radial_Coulomb[0] contains 4*pi times the weighted monopole kernel.
        hartree = self.radial_coulomb[0] @ (radial_density / (4.0 * np.pi))
        if _requires_gradient(self.functional):
            xc, gradient_squared, xc_potential, xc_matrix, xc_energy = (
                self._semilocal_xc(checked, radial_density, density)
            )
            xc_energy_density = np.asarray(xc.energy_density)
        else:
            xc = evaluate_spherical_functional(self.functional, density)
            gradient_squared = None
            xc_potential = xc.potential
            xc_energy_density = xc.energy_density
            xc_matrix = np.diag(xc_potential)
            xc_energy = np.dot(
                self.radial_weights,
                4.0 * np.pi * self.r**2 * xc_energy_density,
            )

        matrices = {}
        for l in self._occupied_l:
            matrix = self.core.radial_matrix(l).astype(np.complex128)
            matrix[np.diag_indices_from(matrix)] += hartree
            matrix += xc_matrix
            matrices[l] = matrix

        return {
            "radial_orbitals": checked,
            "radial_density": radial_density,
            "density": density,
            "density_gradient_squared": gradient_squared,
            "hartree_potential": hartree,
            "xc_potential": xc_potential,
            "xc_matrix": xc_matrix,
            "xc_energy_density": xc_energy_density,
            "xc_energy": float(xc_energy),
            "matrices": matrices,
        }

    def kohn_sham_matrices(self, radial_orbitals):
        """Construct nodal radial Kohn--Sham matrices for every occupied l."""
        return self._assemble(radial_orbitals)["matrices"]

    # Retain the hook used by the shared Roothaan/DIIS implementation.
    def fock_matrices(self, radial_orbitals):
        return self.kohn_sham_matrices(radial_orbitals)

    def _evaluate(self, symmetric_orbitals):
        nodal = {
            l: values / self._sqrt_weights[None, :]
            for l, values in symmetric_orbitals.items()
        }
        state = self._assemble(nodal)
        hamiltonian = {
            l: self._to_symmetric(matrix)
            for l, matrix in state["matrices"].items()
        }

        errors = []
        kinetic_energy = 0j
        for l in self._occupied_l:
            one_body_density = (
                symmetric_orbitals[l].conj().T @ symmetric_orbitals[l]
            )
            commutator = (
                hamiltonian[l] @ one_body_density
                - one_body_density @ hamiltonian[l]
            )
            errors.append(np.sqrt(2 * l + 1) * commutator.ravel())
            kinetic = self._to_symmetric(self.core.kinetic.radial_matrix(l))
            for orbital in symmetric_orbitals[l]:
                kinetic_energy += 2.0 * (2 * l + 1) * np.vdot(
                    orbital, kinetic @ orbital
                )

        external_energy = np.dot(
            self.radial_weights,
            state["radial_density"] * self.core.potential.values,
        )
        hartree_energy = 0.5 * np.dot(
            self.radial_weights,
            state["radial_density"] * state["hartree_potential"],
        )
        xc_energy = state["xc_energy"]
        if abs(kinetic_energy.imag) > 1e-8:
            raise RuntimeError(
                "The Kohn--Sham kinetic energy acquired a non-negligible "
                "imaginary part."
            )
        kinetic_energy = float(kinetic_energy.real)
        external_energy = float(external_energy)
        hartree_energy = float(hartree_energy)
        xc_energy = float(xc_energy)
        energy = kinetic_energy + external_energy + hartree_energy + xc_energy
        error = np.concatenate(errors)
        self._last_ks_state = {
            "density": state["density"].copy(),
            "radial_density": state["radial_density"].copy(),
            "density_gradient_squared": (
                None
                if state["density_gradient_squared"] is None
                else state["density_gradient_squared"].copy()
            ),
            "hartree_potential": state["hartree_potential"].copy(),
            "xc_potential": state["xc_potential"].copy(),
            "xc_matrix": state["xc_matrix"].copy(),
            "kinetic_energy": kinetic_energy,
            "external_energy": external_energy,
            "hartree_energy": hartree_energy,
            "xc_energy": xc_energy,
        }
        return energy, float(np.linalg.norm(error)), hamiltonian, error

    def _result_metadata(self):
        if self._last_ks_state is None:
            raise RuntimeError("No Kohn--Sham state has been evaluated.")
        return {
            "functional": self.functional,
            "initial_guess": self.initial_guess,
            "xc_quadrature_order": (
                None
                if self.xc_quadrature_order is None
                else self.xc_quadrature_order.copy()
            ),
            **self._last_ks_state,
        }


class SphericalAtomicKohnSham(SphericalKohnSham):
    """Restricted Kohn--Sham solver for a built-in closed-shell atom or ion."""

    result_type = AtomicKohnShamResult

    def __init__(
        self,
        atom,
        *,
        functional,
        charge=None,
        r_max=None,
        nodes=None,
        n_elements=None,
        n_points=15,
        l_max=None,
        m_max=None,
        poisson_method="inverse",
        mass=1.0,
        max_iterations=100,
        conv_tol_e=1e-10,
        conv_tol_grad=1e-8,
        damping=0.2,
        diis=True,
        diis_start=3,
        diis_size=8,
        initial_guess="auto",
        xc_quadrature_order=None,
    ):
        self.configuration = closed_shell_atom(atom, charge=charge)
        super().__init__(
            Atomic(self.configuration, mass=mass),
            functional=functional,
            r_max=r_max,
            nodes=nodes,
            n_elements=n_elements,
            n_points=n_points,
            l_max=l_max,
            m_max=m_max,
            poisson_method=poisson_method,
            max_iterations=max_iterations,
            conv_tol_e=conv_tol_e,
            conv_tol_grad=conv_tol_grad,
            damping=damping,
            diis=diis,
            diis_start=diis_start,
            diis_size=diis_size,
            initial_guess=initial_guess,
            xc_quadrature_order=xc_quadrature_order,
        )

    def _make_core(self):
        return AtomicCoreHamiltonian(
            self.grid,
            self.l_max,
            self.m_max,
            Z=self.configuration.nuclear_charge,
            mass=self.system.mass,
        )

    def _result_metadata(self):
        return {**super()._result_metadata(), "configuration": self.configuration}
