r"""Spin-restricted Kohn--Sham ground states in central potentials."""

from dataclasses import dataclass

import numpy as np
from grid_lib.pseudospectral_grids.femdvr import FEMDVR

from tdmcscf.systems import (
    Atomic,
    AtomicConfiguration,
    SphericalElectronicSystem,
    closed_shell_atom,
)

from .atomic_core import AtomicCoreHamiltonian
from .hartree_fock import SCFIteration, SphericalHartreeFock


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
    density: np.ndarray
    radial_density: np.ndarray
    hartree_potential: np.ndarray
    xc_potential: np.ndarray
    kinetic_energy: float
    external_energy: float
    hartree_energy: float
    xc_energy: float


@dataclass
class AtomicKohnShamResult(KohnShamResult):
    """Atomic Kohn--Sham result retaining the spectroscopic configuration."""

    configuration: AtomicConfiguration


def _functional_values(functional, density):
    values = functional.evaluate(density)
    for name in ("energy_density", "potential"):
        array = np.asarray(getattr(values, name, None))
        if array.shape != density.shape or np.iscomplexobj(array) or not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                "functional.evaluate(density) must return finite real "
                "energy_density and potential arrays matching density."
            )
    return values


def _local_functional(functional):
    if not callable(getattr(functional, "evaluate", None)):
        raise TypeError("functional must provide evaluate(density).")
    _functional_values(functional, np.array([0.0, 1.0]))
    return functional


class SphericalKohnSham(SphericalHartreeFock):
    r"""Spin-unpolarized closed-shell Kohn--Sham solver.

    The complete-shell occupation makes the density and effective potential
    spherical.  One radial Kohn--Sham matrix is therefore diagonalized for
    each occupied angular momentum ``l``:

    ``H_l = T_l + V_ext + V_H[n] + V_xc[n]``.

    ``functional`` is a coordinate-independent local functional acting on
    the total spin-summed density.  It must provide ``evaluate(density)`` and
    return pointwise ``energy_density`` and ``potential`` arrays.  The
    built-in choices live in :mod:`tdmcscf.dft`.

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
    ):
        self.functional = _local_functional(functional)
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
        xc = _functional_values(self.functional, density)
        xc_potential = np.asarray(xc.potential, dtype=float)
        local_potential = hartree + xc_potential

        matrices = {}
        for l in self._occupied_l:
            matrix = self.core.radial_matrix(l).astype(np.complex128)
            matrix[np.diag_indices_from(matrix)] += local_potential
            matrices[l] = matrix

        return {
            "radial_orbitals": checked,
            "radial_density": radial_density,
            "density": density,
            "hartree_potential": hartree,
            "xc_potential": xc_potential,
            "xc_energy_density": np.asarray(xc.energy_density, dtype=float),
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
        xc_energy = np.dot(
            self.radial_weights,
            4.0 * np.pi * self.r**2 * state["xc_energy_density"],
        )
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
            "hartree_potential": state["hartree_potential"].copy(),
            "xc_potential": state["xc_potential"].copy(),
            "kinetic_energy": kinetic_energy,
            "external_energy": external_energy,
            "hartree_energy": hartree_energy,
            "xc_energy": xc_energy,
        }
        return energy, float(np.linalg.norm(error)), hamiltonian, error

    def _result_metadata(self):
        if self._last_ks_state is None:
            raise RuntimeError("No Kohn--Sham state has been evaluated.")
        return {"functional": self.functional, **self._last_ks_state}


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
