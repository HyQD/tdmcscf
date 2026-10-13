"""Symmetry-adapted restricted Hartree--Fock for closed-shell atoms."""

from dataclasses import dataclass

import numpy as np
from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.angular_momentum import (
    LM_to_I,
    get_y_value,
    get_ybar_value,
)
from grid_lib.spherical_coordinates.radial_Coulomb import radial_Coulomb

from tdmcscf.systems import AtomicConfiguration, closed_shell_atom

from ._validation import _integer, _positive_real
from .atomic_core import AtomicCoreHamiltonian


@dataclass(frozen=True)
class SCFIteration:
    """Diagnostics evaluated for one set of occupied orbitals."""

    iteration: int
    energy: float
    energy_change: float
    gradient_norm: float


@dataclass
class AtomicHartreeFockResult:
    """Converged state and diagnostics from an atomic RHF calculation."""

    configuration: AtomicConfiguration
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


class _DIIS:
    def __init__(self, maximum_size):
        self.maximum_size = maximum_size
        self.fock = []
        self.errors = []

    def append(self, fock, error):
        self.fock.append({l: matrix.copy() for l, matrix in fock.items()})
        self.errors.append(error.copy())
        if len(self.fock) > self.maximum_size:
            self.fock.pop(0)
            self.errors.pop(0)

    def extrapolate(self):
        size = len(self.errors)
        system = np.empty((size + 1, size + 1))
        system[-1, :-1] = -1
        system[:-1, -1] = -1
        system[-1, -1] = 0
        for i, left in enumerate(self.errors):
            for j, right in enumerate(self.errors):
                system[i, j] = np.vdot(left, right).real
        rhs = np.zeros(size + 1)
        rhs[-1] = -1
        coefficients = np.linalg.solve(system, rhs)[:-1]
        return {
            l: sum(c * fock[l] for c, fock in zip(coefficients, self.fock))
            for l in self.fock[-1]
        }


def make_atomic_femdvr_grid(*, r_max=None, nodes=None, n_elements=None, n_points=15):
    """Construct the fixed real FEM-DVR grid used by the atomic solver.

    Supply explicit element-boundary ``nodes`` or let the function generate
    uniform boundaries from ``r_max`` and ``n_elements``. If explicit nodes
    and ``r_max`` are both supplied, the final node must equal ``r_max``.
    ``n_points`` may be one integer or one integer per element.
    """
    if nodes is None:
        if r_max is None:
            raise ValueError("r_max is required when nodes are not supplied.")
        r_max = _positive_real(r_max, "r_max")
        n_elements = (
            10 if n_elements is None else _integer(n_elements, "n_elements", minimum=1)
        )
        boundaries = np.linspace(0.0, r_max, n_elements + 1)
    else:
        boundaries = np.asarray(nodes)
        if (
            boundaries.ndim != 1
            or len(boundaries) < 2
            or np.iscomplexobj(boundaries)
            or not np.all(np.isfinite(boundaries))
            or boundaries[0] != 0
            or np.any(np.diff(boundaries) <= 0)
        ):
            raise ValueError(
                "nodes must be finite, increasing element boundaries starting at zero."
            )
        boundaries = boundaries.astype(float)
        if r_max is not None and not np.isclose(
            _positive_real(r_max, "r_max"), boundaries[-1], rtol=0, atol=1e-12
        ):
            raise ValueError("r_max must equal the final explicit node.")
        if (
            n_elements is not None
            and _integer(n_elements, "n_elements", minimum=1) != len(boundaries) - 1
        ):
            raise ValueError("n_elements must agree with the explicit nodes.")

    count = len(boundaries) - 1
    points = np.asarray(n_points)
    if points.ndim == 0:
        if isinstance(n_points, (bool, np.bool_)) or not np.issubdtype(
            points.dtype, np.integer
        ):
            raise TypeError("n_points must contain integers.")
        points = np.full(count, int(points))
    elif points.shape == (count,) and np.issubdtype(points.dtype, np.integer):
        points = points.astype(int)
    else:
        raise ValueError("n_points must be one integer or one integer per element.")
    if np.any(points < 3):
        raise ValueError("Every element must contain at least three points.")
    return FEMDVR(
        boundaries, points, Linear_map, GaussLegendreLobatto, symmetrize=False
    )


class SphericalAtomicHartreeFock:
    r"""Restricted Hartree--Fock for a spherical closed-shell atom or ion.

    The occupied subshells are constrained to share one radial orbital over
    all ``m=-l,...,l``. The rotationally invariant Fock operator is assembled
    and diagonalized once for every occupied ``l``. ``l_max`` and ``m_max``
    control the representation of returned orbitals and may exceed the
    occupied angular momenta for subsequent time-dependent calculations.

    Parameters
    ----------
    atom : str or AtomicConfiguration
        Built-in label or explicit closed-shell configuration.
    r_max, nodes, n_elements, n_points
        FEM-DVR controls accepted by :func:`make_atomic_femdvr_grid`.
    l_max, m_max : int, optional
        Returned spherical-harmonic basis. Both must include every occupied
        magnetic sublevel. Defaults to the largest occupied l.
    coulomb_l_max : int, optional
        Largest retained Coulomb multipole. Defaults to twice the largest
        occupied l, which is exact for the occupied atomic Fock blocks.
    poisson_method : {"inverse", "solve"}, optional
        Linear-algebra method used by grid-lib to construct the radial
        Coulomb matrices from the Poisson equation. Defaults to ``"inverse"``
        for backward compatibility.
    conv_tol_e : float, optional
        Convergence threshold for the absolute change in total energy. This is
        used only when ``conv_tol_grad`` is ``None``; the energy change is
        always recorded in the iteration history.
    conv_tol_grad : float or None, optional
        Convergence threshold for the orbital-gradient norm, represented by
        the weighted Fock--density commutator norm. When supplied, this is the
        sole convergence criterion. Set it to ``None`` to converge on the
        energy change instead.
    damping : float, optional
        Fraction of the preceding Fock matrix retained before DIIS starts.
    diis : bool, optional
        Enable commutator DIIS after ``diis_start`` iterations.
    """

    def __init__(
        self,
        atom,
        *,
        r_max=None,
        nodes=None,
        n_elements=None,
        n_points=15,
        l_max=None,
        m_max=None,
        coulomb_l_max=None,
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
        self.configuration = closed_shell_atom(atom)
        occupied_l_max = self.configuration.occupied_l_max
        self.l_max = occupied_l_max if l_max is None else _integer(l_max, "l_max")
        self.m_max = self.l_max if m_max is None else _integer(m_max, "m_max")
        if self.l_max < occupied_l_max or self.m_max < occupied_l_max:
            raise ValueError("l_max and m_max must include every occupied subshell.")
        if self.m_max > self.l_max:
            raise ValueError("m_max cannot exceed l_max.")
        default_coulomb = 2 * occupied_l_max
        self.coulomb_l_max = (
            default_coulomb
            if coulomb_l_max is None
            else _integer(coulomb_l_max, "coulomb_l_max")
        )
        self.poisson_method = poisson_method
        self.max_iterations = _integer(max_iterations, "max_iterations", minimum=1)
        self.conv_tol_e = _positive_real(conv_tol_e, "conv_tol_e")
        self.conv_tol_grad = (
            None
            if conv_tol_grad is None
            else _positive_real(conv_tol_grad, "conv_tol_grad")
        )
        if isinstance(damping, (bool, np.bool_)) or not isinstance(
            damping, (int, float, np.integer, np.floating)
        ):
            raise TypeError("damping must be a finite real scalar in [0, 1).")
        if not np.isfinite(damping) or damping < 0 or damping >= 1:
            raise ValueError("damping must be a finite real scalar in [0, 1).")
        if not isinstance(diis, (bool, np.bool_)):
            raise TypeError("diis must be boolean.")
        self.damping = float(damping)
        self.diis = bool(diis)
        self.diis_start = _integer(diis_start, "diis_start", minimum=1)
        self.diis_size = _integer(diis_size, "diis_size", minimum=2)

        self.grid = make_atomic_femdvr_grid(
            r_max=r_max,
            nodes=nodes,
            n_elements=n_elements,
            n_points=n_points,
        )
        self.core = AtomicCoreHamiltonian(
            self.grid,
            self.l_max,
            self.m_max,
            Z=self.configuration.nuclear_charge,
            mass=mass,
        )
        self.r = self.core.r
        self.radial_weights = self.core.radial_weights
        self.orbital_lm = self.core.orbital_lm
        self.orbital_shape = self.core.orbital_shape
        self.radial_coulomb = radial_Coulomb(
            self.grid,
            self.coulomb_l_max + 1,
            method=self.poisson_method,
        )
        grouped = {}
        for subshell in self.configuration.subshells:
            grouped.setdefault(subshell.l, []).append(subshell)
        self._subshells_by_l = {
            l: tuple(sorted(subshells)) for l, subshells in grouped.items()
        }
        self._occupied_l = tuple(sorted(self._subshells_by_l))
        self._sqrt_weights = np.sqrt(self.radial_weights)

    def _to_symmetric(self, matrix):
        transformed = self._sqrt_weights[:, None] * matrix / self._sqrt_weights[None, :]
        return 0.5 * (transformed + transformed.T.conj())

    def _initial_orbitals(self):
        orbitals = {}
        for l, subshells in self._subshells_by_l.items():
            values, vectors = np.linalg.eigh(
                self._to_symmetric(self.core.radial_matrix(l))
            )
            del values
            orbitals[l] = vectors[:, : len(subshells)].T
        return orbitals

    def fock_matrices(self, radial_orbitals):
        """Construct nodal radial Fock matrices from occupied radial orbitals.

        ``radial_orbitals[l]`` must have shape ``(n_shells_l, n_r)`` and use
        nodal normalization. The returned dictionary contains an independent
        matrix for each occupied l. The exchange term is a nonlocal radial
        matrix; the complete-shell sum makes every block independent of m.
        """
        checked = {}
        for l, subshells in self._subshells_by_l.items():
            values = np.asarray(radial_orbitals[l])
            expected = (len(subshells), len(self.r))
            if values.shape != expected:
                raise ValueError(f"radial_orbitals[{l}] must have shape {expected}.")
            if not np.issubdtype(values.dtype, np.number) or not np.all(
                np.isfinite(values)
            ):
                raise ValueError("Radial orbitals must be finite numeric arrays.")
            checked[l] = np.asarray(values, dtype=np.complex128)

        blocks = {
            l: self.core.radial_matrix(l).astype(np.complex128)
            for l in self._occupied_l
        }
        direct = {
            l: np.zeros(len(self.r), dtype=np.complex128) for l in self._occupied_l
        }
        exchange = {
            l: np.zeros((len(self.r), len(self.r)), dtype=np.complex128)
            for l in self._occupied_l
        }

        for occupied_l, orbitals in checked.items():
            for occupied in orbitals:
                density = occupied.conj() * occupied
                for occupied_m in range(-occupied_l, occupied_l + 1):
                    for target_l in self._occupied_l:
                        for L in range(self.coulomb_l_max + 1):
                            direct_coefficient = get_y_value(
                                L,
                                0,
                                occupied_l,
                                occupied_m,
                                occupied_l,
                                occupied_m,
                            ) * get_ybar_value(L, 0, target_l, 0, target_l, 0)
                            if direct_coefficient != 0:
                                direct[target_l] += (
                                    2
                                    * direct_coefficient
                                    * (self.radial_coulomb[L] @ density)
                                )

                            if abs(occupied_m) > L:
                                continue
                            exchange_coefficient = get_y_value(
                                L,
                                occupied_m,
                                occupied_l,
                                occupied_m,
                                target_l,
                                0,
                            ) * get_ybar_value(
                                L,
                                occupied_m,
                                target_l,
                                0,
                                occupied_l,
                                occupied_m,
                            )
                            if exchange_coefficient != 0:
                                exchange[target_l] += exchange_coefficient * (
                                    occupied[:, None]
                                    * self.radial_coulomb[L]
                                    * occupied.conj()[None, :]
                                )

        for l in self._occupied_l:
            blocks[l][np.diag_indices_from(blocks[l])] += direct[l]
            blocks[l] -= exchange[l]
        return blocks

    def _evaluate(self, symmetric_orbitals):
        nodal = {
            l: values / self._sqrt_weights[None, :]
            for l, values in symmetric_orbitals.items()
        }
        nodal_fock = self.fock_matrices(nodal)
        fock = {l: self._to_symmetric(matrix) for l, matrix in nodal_fock.items()}
        core = {
            l: self._to_symmetric(self.core.radial_matrix(l)) for l in self._occupied_l
        }
        energy = 0j
        errors = []
        for l in self._occupied_l:
            density = symmetric_orbitals[l].conj().T @ symmetric_orbitals[l]
            commutator = fock[l] @ density - density @ fock[l]
            errors.append(np.sqrt(2 * l + 1) * commutator.ravel())
            for orbital in symmetric_orbitals[l]:
                energy += (2 * l + 1) * (
                    np.vdot(orbital, core[l] @ orbital)
                    + np.vdot(orbital, fock[l] @ orbital)
                )
        if abs(energy.imag) > 1e-8:
            raise RuntimeError(
                "The RHF energy acquired a non-negligible imaginary part."
            )
        error = np.concatenate(errors)
        return float(energy.real), float(np.linalg.norm(error)), fock, error

    @staticmethod
    def _diagonalize(fock, previous):
        updated = {}
        energies = {}
        for l, matrix in fock.items():
            values, vectors = np.linalg.eigh(matrix)
            count = previous[l].shape[0]
            selected = vectors[:, :count].T
            overlap = selected.conj() @ previous[l].T
            for i in range(count):
                if overlap[i, i] != 0:
                    selected[i] *= np.exp(1j * np.angle(overlap[i, i]))
            updated[l] = selected
            energies[l] = values[:count]
        return updated, energies

    def _expanded_result(self, radial, radial_energies):
        rows = {
            (subshell.n, l): i
            for l, subshells in self._subshells_by_l.items()
            for i, subshell in enumerate(subshells)
        }
        orbitals = np.zeros(
            (self.configuration.n_doubly_occupied, *self.orbital_shape),
            dtype=np.complex128,
        )
        energies = []
        for p, (n, l, m) in enumerate(self.configuration.orbital_quantum_numbers):
            row = rows[n, l]
            channel = int(LM_to_I(l, m, self.l_max, self.m_max))
            orbitals[p, :, channel] = radial[l][row]
            energies.append(radial_energies[l][row])
        return orbitals, np.asarray(energies)

    def solve(self):
        """Run the damped Roothaan/DIIS iterations and return the RHF state."""
        orbitals = self._initial_orbitals()
        diis = _DIIS(self.diis_size)
        previous_fock = None
        previous_energy = None
        history = []
        converged = False
        radial_energies = {}

        for iteration in range(1, self.max_iterations + 1):
            energy, gradient_norm, raw_fock, error = self._evaluate(orbitals)
            energy_change = (
                np.inf if previous_energy is None else abs(energy - previous_energy)
            )
            history.append(
                SCFIteration(iteration, energy, energy_change, gradient_norm)
            )
            if self.conv_tol_grad is not None:
                has_converged = gradient_norm < self.conv_tol_grad
            else:
                has_converged = (
                    previous_energy is not None
                    and energy_change < self.conv_tol_e
                )
            if has_converged:
                converged = True
                _, radial_energies = self._diagonalize(raw_fock, orbitals)
                break
            if iteration == self.max_iterations:
                _, radial_energies = self._diagonalize(raw_fock, orbitals)
                break

            diis.append(raw_fock, error)
            effective_fock = raw_fock
            if self.diis and iteration >= self.diis_start and len(diis.fock) >= 2:
                try:
                    effective_fock = diis.extrapolate()
                except np.linalg.LinAlgError:
                    effective_fock = raw_fock
            elif previous_fock is not None and self.damping:
                effective_fock = {
                    l: (1 - self.damping) * raw_fock[l]
                    + self.damping * previous_fock[l]
                    for l in raw_fock
                }
            orbitals, radial_energies = self._diagonalize(effective_fock, orbitals)
            previous_fock = raw_fock
            previous_energy = energy

        radial = {
            l: values / self._sqrt_weights[None, :] for l, values in orbitals.items()
        }
        expanded, energies = self._expanded_result(radial, radial_energies)
        return AtomicHartreeFockResult(
            configuration=self.configuration,
            grid=self.grid,
            energy=history[-1].energy,
            orbital_energies=energies,
            orbitals=expanded,
            orbital_quantum_numbers=self.configuration.orbital_quantum_numbers,
            radial_orbitals=radial,
            radial_orbital_energies=radial_energies,
            converged=converged,
            iterations=len(history),
            history=tuple(history),
            l_max=self.l_max,
            m_max=self.m_max,
            coulomb_l_max=self.coulomb_l_max,
            poisson_method=self.poisson_method,
        )
