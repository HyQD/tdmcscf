"""High-level setup and propagation interface for spherical restricted TDHF."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from numbers import Integral, Real
from pathlib import Path
import warnings

import numpy as np
from grid_lib.spherical_coordinates.utils import mask_function

from integrator_lib import IMCMF1, IMCMF2, CrankNicolson, RK4, Strang2
from tdmcscf.systems import SphericalElectronicSystem

from ._validation import _integer, _positive_real
from .atomic_core import SphericalCoreHamiltonian
from .dipole_fields import DipoleFieldTerm, SphericalDipoleField
from .hartree_fock import (
    HartreeFockResult,
    SphericalHartreeFock,
    make_spherical_femdvr_grid,
)
from .mean_fields import SphericalMeanFieldBuilder
from .preconditioners import ShiftedSphericalCorePreconditioner
from .tdhf import SphericalRestrictedTDHF


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    return value


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _normalized_polarization(value):
    vector = np.asarray(value)
    if (
        vector.shape != (3,)
        or not np.issubdtype(vector.dtype, np.number)
        or np.iscomplexobj(vector)
        or not np.all(np.isfinite(vector))
    ):
        raise ValueError("polarization must contain three finite real components.")
    vector = vector.astype(float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("polarization cannot be the zero vector.")
    return tuple(vector / norm)


def _one_field_term(value):
    if isinstance(value, DipoleFieldTerm):
        field = value.field
        polarization = value.polarization
    elif callable(value):
        field = value
        polarization = (0.0, 0.0, 1.0)
    else:
        try:
            field, polarization = value
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Each field must be callable, a DipoleFieldTerm, or a "
                "(callable, polarization) pair."
            ) from error
        if not callable(field):
            raise TypeError("The first item in a field pair must be callable.")
    return DipoleFieldTerm(field, _normalized_polarization(polarization))


def _field_terms(fields):
    if fields is None:
        return ()
    if isinstance(fields, DipoleFieldTerm) or callable(fields):
        return (_one_field_term(fields),)
    if (
        isinstance(fields, tuple)
        and len(fields) == 2
        and callable(fields[0])
        and np.asarray(fields[1]).shape == (3,)
    ):
        return (_one_field_term(fields),)
    if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
        raise TypeError("fields must be a field term or a sequence of field terms.")
    return tuple(_one_field_term(term) for term in fields)


@dataclass(frozen=True)
class RadialCosineMask:
    r"""Cosine mask equal to one below ``r0`` and zero at ``r_max``.

    Between the two radii the mask is
    ``cos(pi*(r-r0)/(2*(r_max-r0)))**(1/n)``. It is applied after every
    completed propagation step. Orbitals are not renormalized afterwards.
    """

    r0: float
    n: float = 4.0

    def __post_init__(self):
        object.__setattr__(self, "r0", _finite_real(self.r0, "r0"))
        object.__setattr__(self, "n", _positive_real(self.n, "n"))
        if self.r0 < 0:
            raise ValueError("r0 must be nonnegative.")

    def values(self, r, r_max):
        """Return mask values on radial nodes inside ``r_max``."""
        r_max = _positive_real(r_max, "r_max")
        if self.r0 >= r_max:
            raise ValueError("r0 must be smaller than the outer grid radius.")
        nodes = np.asarray(r, dtype=float)
        return mask_function(nodes, r_max, self.r0, self.n)


def estimate_rk4_time_step(kinetic, safety=0.5):
    r"""Estimate a conservative RK4 step from the kinetic-energy spectrum.

    Classical RK4 intersects the imaginary stability axis at
    ``abs(dt*lambda) = 2*sqrt(2)``. The returned estimate multiplies this
    limit by ``safety`` and uses the largest absolute kinetic eigenvalue over
    all retained ``l`` sectors. Other Hamiltonian terms can require a smaller
    time step.
    """
    if not hasattr(kinetic, "radial_matrix") or not hasattr(
        kinetic, "radial_weights"
    ):
        raise TypeError("kinetic must provide radial matrices and weights.")
    safety = _positive_real(safety, "safety")
    if safety > 1:
        raise ValueError("safety cannot exceed one.")
    weights = np.asarray(kinetic.radial_weights)
    sqrt_weights = np.sqrt(weights)
    spectral_radius = 0.0
    for l in range(kinetic.l_max + 1):
        matrix = kinetic.radial_matrix(l)
        symmetric = sqrt_weights[:, None] * matrix / sqrt_weights[None, :]
        symmetric = 0.5 * (symmetric + symmetric.T.conj())
        spectral_radius = max(
            spectral_radius,
            float(np.max(np.abs(np.linalg.eigvalsh(symmetric)))),
        )
    if spectral_radius == 0:
        return np.inf
    return safety * 2 * np.sqrt(2) / spectral_radius


@dataclass(frozen=True)
class TDHFCheckpoint:
    """Saved propagation state and samples."""

    time: float
    step: int
    orbitals: np.ndarray
    times: np.ndarray
    samples: dict[str, np.ndarray]
    metadata: dict
    grid_nodes: np.ndarray
    grid_n_points: np.ndarray
    radial_grid: np.ndarray
    radial_weights: np.ndarray

    @classmethod
    def load(cls, path):
        """Load a checkpoint written by :class:`SphericalTDHF`."""
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            names = metadata["sample_names"]
            samples = {
                name: np.array(data[f"sample_{index}"], copy=True)
                for index, name in enumerate(names)
            }
            return cls(
                time=float(data["time"]),
                step=int(data["step"]),
                orbitals=np.array(data["orbitals"], copy=True),
                times=np.array(data["times"], copy=True),
                samples=samples,
                metadata=metadata,
                grid_nodes=np.array(data["grid_nodes"], copy=True),
                grid_n_points=np.array(data["grid_n_points"], copy=True),
                radial_grid=np.array(data["radial_grid"], copy=True),
                radial_weights=np.array(data["radial_weights"], copy=True),
            )


@dataclass
class TDHFResult:
    """Final orbitals, sampled observables, and propagation metadata."""

    system: SphericalElectronicSystem
    grid: object
    initial_orbitals: np.ndarray
    orbitals: np.ndarray
    times: np.ndarray
    samples: dict[str, np.ndarray]
    time_step: float
    n_steps: int
    l_max: int
    m_max: int
    L_max: int
    M_max: int
    field_gauge: str
    orbital_gauge: str
    integrator: str
    particle_charge: float

    @property
    def orbital_norms(self):
        """Sampled individual orbital norms, if requested."""
        return self.samples.get("orbital_norms")

    @property
    def position_expectations(self):
        """Sampled per-orbital Cartesian position expectations."""
        return self.samples.get("position_expectations")

    @property
    def total_dipole(self):
        """Return the closed-shell Cartesian dipole sampled from positions."""
        if self.position_expectations is None:
            return None
        return 2 * self.particle_charge * self.position_expectations.sum(axis=1)


class SphericalTDHF:
    r"""Configure and run a closed-shell spherical restricted-TDHF calculation.

    ``system`` supplies the central potential, particle mass, and occupation.
    ``initial_state`` may be ``"groundstate"``, a compatible
    :class:`HartreeFockResult`, or a full array of orthonormal occupied
    orbitals. Field entries may be temporal callables, ``DipoleFieldTerm``
    objects, or ``(callable, polarization)`` pairs. Polarization vectors are
    normalized by this interface.

    The supported propagation methods are ``"strang2"``, ``"rk4"``,
    ``"im-cmf1"``, ``"im-cmf2"``, ``"im-cdm1"``, and ``"im-cdm2"``.
    Strang2 uses preconditioned Crank--Nicolson for the linear one-body steps
    and RK4 for the nonlinear step. RK4 rebuilds the mean fields at every
    stage. The IM-CMF methods freeze orbital-pair mean fields, while the
    IM-CDM methods freeze the density matrix entering the Fock operator.
    """

    _BUILTIN_SAMPLERS = {"orbital_norms", "position_expectations"}
    checkpoint_type = TDHFCheckpoint
    result_type = TDHFResult
    progress_label = "TDHF propagation"

    def __init__(
        self,
        system,
        *,
        r_max=None,
        nodes=None,
        n_elements=None,
        n_points=15,
        l_max=None,
        m_max=None,
        L_max=None,
        M_max=None,
        poisson_method="inverse",
        initial_state="groundstate",
        ground_state_options=None,
        fields=None,
        field_gauge="length",
        particle_charge=-1.0,
        orbital_gauge="orthogonal",
        integrator="strang2",
        integrator_options=None,
        initial_time=0.0,
        final_time=None,
        time_step=None,
        mask=None,
        observables=("orbital_norms", "position_expectations"),
        samplers=None,
        sample_interval=1,
        checkpoint_path=None,
        checkpoint_interval=None,
    ):
        if not isinstance(system, SphericalElectronicSystem):
            raise TypeError("system must be a SphericalElectronicSystem.")
        self.system = system
        self.l_max = (
            system.occupied_l_max
            if l_max is None
            else _integer(l_max, "l_max")
        )
        self.m_max = self.l_max if m_max is None else _integer(m_max, "m_max")
        if self.l_max < system.occupied_l_max:
            raise ValueError("l_max must include every occupied subshell.")
        if self.m_max < system.occupied_m_max:
            raise ValueError("m_max must include every occupied magnetic sublevel.")
        if self.m_max > self.l_max:
            raise ValueError("m_max cannot exceed l_max.")
        self.L_max = (
            2 * self.l_max if L_max is None else _integer(L_max, "L_max")
        )
        default_M = min(self.L_max, 2 * self.m_max)
        self.M_max = default_M if M_max is None else _integer(M_max, "M_max")
        if self.L_max > 2 * self.l_max:
            raise ValueError("L_max cannot exceed 2*l_max.")
        if self.M_max > self.L_max or self.M_max > 2 * self.m_max:
            raise ValueError("M_max must not exceed L_max or 2*m_max.")
        if poisson_method not in {"inverse", "solve"}:
            raise ValueError("poisson_method must be 'inverse' or 'solve'.")
        self.poisson_method = poisson_method

        self.initial_time = _finite_real(initial_time, "initial_time")
        if final_time is None:
            raise ValueError("final_time is required.")
        self.final_time = _finite_real(final_time, "final_time")
        if self.final_time <= self.initial_time:
            raise ValueError("final_time must be larger than initial_time.")
        if time_step is None:
            raise ValueError("time_step is required.")
        self.requested_time_step = _positive_real(time_step, "time_step")
        duration = self.final_time - self.initial_time
        self.n_steps = int(np.ceil(duration / self.requested_time_step))
        self.time_step = duration / self.n_steps
        self.propagation_times = np.linspace(
            self.initial_time, self.final_time, self.n_steps + 1
        )

        self.grid = make_spherical_femdvr_grid(
            r_max=r_max,
            nodes=nodes,
            n_elements=n_elements,
            n_points=n_points,
        )
        self.core = SphericalCoreHamiltonian(
            self.grid,
            self.l_max,
            self.m_max,
            potential=system.potential,
            mass=system.mass,
        )
        self.ground_state = None
        self.initial_orbitals = self._initial_orbitals(
            initial_state, ground_state_options
        )

        if field_gauge not in {"length", "velocity"}:
            raise ValueError("field_gauge must be 'length' or 'velocity'.")
        self.field_gauge = field_gauge
        self.particle_charge = _finite_real(particle_charge, "particle_charge")
        self.field_terms = _field_terms(fields)
        self._validate_field_gauges()
        self.dipole_field = None
        if self.field_terms:
            self.dipole_field = SphericalDipoleField(
                self.grid,
                self.l_max,
                self.m_max,
                gauge=self.field_gauge,
                fields=self.field_terms,
                charge=self.particle_charge,
                mass=self.system.mass,
            )
        self.m_conserved = all(
            np.allclose(term.polarization[:2], 0.0, rtol=0.0, atol=1e-15)
            for term in self.field_terms
        )
        orbital_m = self._infer_orbital_m() if self.m_conserved else None

        self.mean_fields = SphericalMeanFieldBuilder(
            self.grid,
            self.l_max,
            self.m_max,
            L_max=self.L_max,
            M_max=self.M_max,
            poisson_method=self.poisson_method,
        )
        self.equation = SphericalRestrictedTDHF(
            self.core,
            self.mean_fields,
            dipole_field=self.dipole_field,
            orbital_gauge=orbital_gauge,
            orbital_m=orbital_m,
        )
        self.orbital_gauge = orbital_gauge

        self.integrator_name = str(integrator).lower()
        self.integrator_options = dict(integrator_options or {})
        self.integrator = self._make_integrator()

        self.mask = mask
        self.mask_values = self._mask_values(mask)
        self.sample_interval = _positive_integer(sample_interval, "sample_interval")
        self.samplers = self._make_samplers(observables, samplers)

        self.checkpoint_path = (
            None if checkpoint_path is None else Path(checkpoint_path)
        )
        self.checkpoint_interval = (
            None
            if checkpoint_interval is None
            else _positive_integer(checkpoint_interval, "checkpoint_interval")
        )
        if self.checkpoint_interval is not None and self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required with checkpoint_interval.")

    def _initial_orbitals(self, initial_state, options):
        if isinstance(initial_state, str):
            if initial_state != "groundstate":
                raise ValueError("initial_state string must be 'groundstate'.")
            settings = dict(options or {})
            reserved = {
                "r_max",
                "nodes",
                "n_elements",
                "n_points",
                "l_max",
                "m_max",
            }
            duplicate = reserved.intersection(settings)
            if duplicate:
                names = ", ".join(sorted(duplicate))
                raise ValueError(
                    "ground_state_options cannot override propagation settings: "
                    f"{names}."
                )
            settings.setdefault("poisson_method", self.poisson_method)
            solver = SphericalHartreeFock(
                self.system,
                nodes=self.grid.nodes,
                n_points=self.grid.n_points,
                l_max=self.l_max,
                m_max=self.m_max,
                **settings,
            )
            result = solver.solve()
            if not result.converged:
                raise RuntimeError("The Hartree--Fock initial state did not converge.")
            self.ground_state = result
            values = result.orbitals
        elif isinstance(initial_state, HartreeFockResult):
            self.ground_state = initial_state
            values = self._expand_hartree_fock_result(initial_state)
        else:
            if options:
                raise ValueError(
                    "ground_state_options apply only to initial_state='groundstate'."
                )
            values = initial_state
        return self._validate_initial_orbitals(values)

    def _expand_hartree_fock_result(self, result):
        source_grid = np.asarray(result.grid.r)
        if source_grid.shape != self.grid.r.shape or not np.allclose(
            source_grid, self.grid.r, rtol=0.0, atol=1e-13
        ):
            raise ValueError("The Hartree--Fock result uses a different radial grid.")
        source = np.asarray(result.orbitals)
        source_channels = tuple(
            (l, m)
            for m in range(-result.m_max, result.m_max + 1)
            for l in range(abs(m), result.l_max + 1)
        )
        expected = (len(source), len(self.core.r), len(source_channels))
        if source.ndim != 3 or source.shape != expected:
            raise ValueError("The Hartree--Fock result has an invalid orbital shape.")
        target = {lm: index for index, lm in enumerate(self.core.orbital_lm)}
        expanded = np.zeros(
            (len(source), *self.core.orbital_shape), dtype=np.complex128
        )
        for index, lm in enumerate(source_channels):
            if lm not in target:
                if np.any(source[:, :, index] != 0):
                    raise ValueError(
                        "The propagation angular basis omits an occupied component."
                    )
                continue
            expanded[:, :, target[lm]] = source[:, :, index]
        return expanded

    def _validate_initial_orbitals(self, orbitals, *, require_orthonormal=True):
        values = np.asarray(orbitals)
        expected = (self.system.n_doubly_occupied, *self.core.orbital_shape)
        if values.shape != expected:
            raise ValueError(f"initial orbitals must have shape {expected}.")
        if not np.issubdtype(values.dtype, np.number) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("initial orbitals must contain finite numbers.")
        values = np.asarray(values, dtype=np.complex128).copy()
        if require_orthonormal:
            overlap = np.einsum(
                "qai,a,pai->qp",
                values.conj(),
                self.core.radial_weights,
                values,
                optimize=True,
            )
            if not np.allclose(
                overlap, np.eye(len(values)), rtol=0.0, atol=1e-9
            ):
                raise ValueError("initial orbitals must be orthonormal.")
        return values

    def _validate_field_gauges(self):
        for term in self.field_terms:
            gauge = getattr(term.field, "gauge", self.field_gauge)
            if gauge != self.field_gauge:
                raise ValueError(
                    f"Field {type(term.field).__name__} uses gauge {gauge!r}, "
                    f"but field_gauge is {self.field_gauge!r}."
                )

    def _infer_orbital_m(self):
        channel_m = np.array([m for _, m in self.core.orbital_lm])
        magnetic = []
        for orbital in self.initial_orbitals:
            weights = np.array(
                [
                    np.einsum(
                        "ai,a,ai->",
                        orbital[:, channel_m == m].conj(),
                        self.core.radial_weights,
                        orbital[:, channel_m == m],
                        optimize=True,
                    ).real
                    for m in range(-self.m_max, self.m_max + 1)
                ]
            )
            present = np.flatnonzero(weights > 1e-12)
            if len(present) != 1:
                return None
            magnetic.append(int(present[0] - self.m_max))
        return tuple(magnetic)

    def _crank_nicolson(self, step_size, options):
        preconditioner = ShiftedSphericalCorePreconditioner(
            self.core, step_size
        )
        return CrankNicolson(preconditioner=preconditioner, **options)

    def _predictor_corrector_integrator(self, options, corrector_steps):
        self.predictor_crank_nicolson = self._crank_nicolson(
            self.time_step / 2, options
        )
        self.corrector_crank_nicolson = self._crank_nicolson(
            self.time_step / corrector_steps, options
        )
        self.crank_nicolson = self.corrector_crank_nicolson
        return IMCMF2(
            self.predictor_crank_nicolson,
            self.corrector_crank_nicolson,
            corrector_steps=corrector_steps,
        )

    def _make_integrator(self):
        options = dict(self.integrator_options)
        if "preconditioner" in options:
            raise ValueError(
                "The preconditioner is constructed from the core Hamiltonian "
                "and cannot be overridden."
            )
        self.corrector_steps = None
        if self.integrator_name == "strang2":
            self.crank_nicolson = self._crank_nicolson(
                self.time_step / 2, options
            )
            return Strang2(self.crank_nicolson, RK4())
        if self.integrator_name in {"im-cmf1", "im-cdm1"}:
            self.crank_nicolson = self._crank_nicolson(
                self.time_step, options
            )
            return IMCMF1(self.crank_nicolson)
        if self.integrator_name in {"im-cmf2", "im-cdm2"}:
            self.corrector_steps = _positive_integer(
                options.pop("corrector_steps", 2), "corrector_steps"
            )
            if self.corrector_steps not in {1, 2}:
                raise ValueError("corrector_steps must be 1 or 2.")
            return self._predictor_corrector_integrator(
                options, self.corrector_steps
            )
        if self.integrator_name == "rk4":
            if options:
                raise ValueError("RK4 does not accept integrator_options.")
            self.crank_nicolson = None
            estimate = estimate_rk4_time_step(self.core.kinetic)
            relation = (
                "exceeds" if self.time_step > estimate else "is below"
            )
            warnings.warn(
                "RK4 requires a small time step. The kinetic-spectrum estimate "
                f"is dt <= {estimate:.6g} a.u.; the configured step "
                f"{self.time_step:.6g} a.u. {relation} this estimate. Field and "
                "mean-field terms can require a smaller step.",
                RuntimeWarning,
                stacklevel=2,
            )
            return RK4()
        choices = (
            "'strang2', 'rk4', 'im-cmf1', 'im-cmf2', 'im-cdm1', "
            "or 'im-cdm2'"
        )
        raise ValueError(f"integrator must be {choices}.")

    def _mask_values(self, mask):
        if mask is None:
            return None
        if isinstance(mask, RadialCosineMask):
            values = mask.values(self.core.r, self.grid.r[-1])
        elif callable(mask):
            values = mask(self.core.r)
        else:
            values = mask
        values = np.asarray(values)
        if (
            values.shape != self.core.r.shape
            or np.iscomplexobj(values)
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values > 1)
        ):
            raise ValueError(
                "mask must provide one finite real value in [0, 1] per radial node."
            )
        return values.astype(float)

    def _make_samplers(self, observables, samplers):
        if isinstance(observables, str):
            observables = (observables,)
        names = tuple(observables)
        unknown = set(names).difference(self._BUILTIN_SAMPLERS)
        if unknown:
            choices = ", ".join(sorted(self._BUILTIN_SAMPLERS))
            raise ValueError(f"Unknown observable; choose from: {choices}.")
        result = {}
        if "orbital_norms" in names:
            result["orbital_norms"] = (
                lambda simulation, orbitals, time: simulation.equation.orbital_norms(
                    orbitals
                )
            )
        if "position_expectations" in names:
            self._position_operators = self._make_position_operators()
            result["position_expectations"] = (
                lambda simulation, orbitals, time: simulation.position_expectations(
                    orbitals
                )
            )
        if samplers is not None:
            if not isinstance(samplers, Mapping):
                raise TypeError("samplers must map names to callables.")
            for name, sampler in samplers.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("sampler names must be nonempty strings.")
                if name in result:
                    raise ValueError(f"Duplicate sampler name: {name!r}.")
                if not callable(sampler):
                    raise TypeError(f"Sampler {name!r} must be callable.")
                result[name] = sampler
        return result

    def _make_position_operators(self):
        return tuple(
            SphericalDipoleField.from_profile(
                self.grid,
                self.l_max,
                lambda time: 1.0,
                polarization=axis,
                m_max=self.m_max,
                gauge="length",
                charge=-1.0,
                mass=self.system.mass,
            )
            for axis in np.eye(3)
        )

    def position_expectations(self, orbitals):
        """Return per-orbital Cartesian position expectations."""
        if not hasattr(self, "_position_operators"):
            self._position_operators = self._make_position_operators()
        values = []
        for operator in self._position_operators:
            action = operator.apply(orbitals, 0.0)
            expectation = np.einsum(
                "pai,a,pai->p",
                np.asarray(orbitals).conj(),
                self.core.radial_weights,
                action,
                optimize=True,
            )
            if np.max(np.abs(expectation.imag)) > 1e-9:
                raise RuntimeError("A position expectation acquired an imaginary part.")
            values.append(expectation.real)
        return np.stack(values, axis=-1)

    def _full_rhs(self, orbitals, time):
        return self.equation.linear_rhs(
            orbitals, time
        ) + self.equation.nonlinear_rhs(orbitals, time)

    def _advance(self, orbitals, time):
        if self.integrator_name == "strang2":
            result = self.integrator(
                orbitals,
                time,
                self.time_step,
                self.equation.linear_rhs,
                self.equation.nonlinear_rhs,
            )
        elif self.integrator_name in {"im-cmf1", "im-cmf2"}:
            result = self.integrator(
                orbitals,
                time,
                self.time_step,
                self.equation.freeze_rhs,
            )
        elif self.integrator_name in {"im-cdm1", "im-cdm2"}:
            result = self.integrator(
                orbitals,
                time,
                self.time_step,
                self.equation.freeze_density_rhs,
            )
        else:
            result = self.integrator(
                orbitals, time, self.time_step, self._full_rhs
            )
        if self.mask_values is not None:
            result = result * self.mask_values[None, :, None]
        return result

    def _sample(self, orbitals, time, storage):
        for name, sampler in self.samplers.items():
            value = np.asarray(sampler(self, orbitals, time))
            if not np.issubdtype(value.dtype, np.number) or not np.all(
                np.isfinite(value)
            ):
                raise ValueError(f"Sampler {name!r} returned non-finite data.")
            if storage[name] and value.shape != storage[name][0].shape:
                raise ValueError(f"Sampler {name!r} changed its output shape.")
            storage[name].append(value.copy())

    def _metadata(self, sample_names):
        system = {
            "type": type(self.system).__name__,
            "label": self.system.label,
            "mass": self.system.mass,
            "occupation": dict(self.system.occupation.radial_orbitals_per_l),
        }
        for name in ("nuclear_charge", "charge", "omega", "V0", "a"):
            if hasattr(self.system, name):
                system[name] = getattr(self.system, name)
        fields = []
        for term in self.field_terms:
            fields.append(
                {
                    "type": type(term.field).__name__,
                    "gauge": getattr(term.field, "gauge", self.field_gauge),
                    "polarization": list(term.polarization),
                    "time_support": list(
                        getattr(term.field, "time_support", (None, None))
                    ),
                }
            )
        return {
            "format_version": 1,
            "system": system,
            "l_max": self.l_max,
            "m_max": self.m_max,
            "L_max": self.L_max,
            "M_max": self.M_max,
            "poisson_method": self.poisson_method,
            "field_gauge": self.field_gauge,
            "orbital_gauge": self.orbital_gauge,
            "integrator": self.integrator_name,
            "corrector_steps": self.corrector_steps,
            "particle_charge": self.particle_charge,
            "initial_time": self.initial_time,
            "final_time": self.final_time,
            "time_step": self.time_step,
            "n_steps": self.n_steps,
            "fields": fields,
            "sample_names": list(sample_names),
        }

    def _result_metadata(self):
        """Additional fields passed to the propagation result."""
        return {}

    def _write_checkpoint(self, path, step, orbitals, times, samples):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "metadata": np.array(json.dumps(self._metadata(samples))),
            "time": np.array(self.propagation_times[step]),
            "step": np.array(step),
            "orbitals": np.asarray(orbitals),
            "times": np.asarray(times),
            "grid_nodes": np.asarray(self.grid.nodes),
            "grid_n_points": np.asarray(self.grid.n_points),
            "radial_grid": np.asarray(self.grid.r),
            "radial_weights": np.asarray(self.grid.weights),
        }
        for index, values in enumerate(samples.values()):
            arrays[f"sample_{index}"] = np.asarray(values)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)

    def _restore(self, checkpoint):
        if not isinstance(checkpoint, self.checkpoint_type):
            checkpoint = self.checkpoint_type.load(checkpoint)
        different_nodes = not np.array_equal(
            checkpoint.grid_nodes, self.grid.nodes
        )
        different_points = not np.array_equal(
            checkpoint.grid_n_points, self.grid.n_points
        )
        if different_nodes or different_points:
            raise ValueError("The checkpoint uses a different FEM-DVR grid.")
        metadata = checkpoint.metadata
        for name in ("l_max", "m_max", "L_max", "M_max", "integrator"):
            attribute = "integrator_name" if name == "integrator" else name
            expected = getattr(self, attribute)
            if metadata[name] != expected:
                raise ValueError(f"The checkpoint has a different {name} setting.")
        if metadata.get("corrector_steps") != self.corrector_steps:
            raise ValueError(
                "The checkpoint uses a different corrector_steps setting."
            )
        if not np.isclose(metadata["time_step"], self.time_step):
            raise ValueError("The checkpoint uses a different time step.")
        if checkpoint.step < 0 or checkpoint.step > self.n_steps:
            raise ValueError("The checkpoint step is outside the propagation interval.")
        expected_time = self.propagation_times[checkpoint.step]
        if not np.isclose(checkpoint.time, expected_time, rtol=0.0, atol=1e-13):
            raise ValueError("The checkpoint time does not match the propagation grid.")
        orbitals = self._validate_initial_orbitals(
            checkpoint.orbitals, require_orthonormal=False
        )
        if set(checkpoint.samples) != set(self.samplers):
            raise ValueError("The checkpoint uses a different set of samplers.")
        storage = {
            name: [np.array(value, copy=True) for value in values]
            for name, values in checkpoint.samples.items()
        }
        return checkpoint.step, orbitals, list(checkpoint.times), storage

    def run(self, *, progress=False, checkpoint=None):
        """Propagate to ``final_time`` and return sampled results.

        Set ``checkpoint`` to a path or :class:`TDHFCheckpoint` to resume a
        compatible interrupted propagation. Checkpoints are also written at
        the configured interval and after the final step.
        """
        if checkpoint is None:
            start_step = 0
            orbitals = self.initial_orbitals.copy()
            sampled_times = [self.initial_time]
            storage = {name: [] for name in self.samplers}
            self._sample(orbitals, self.initial_time, storage)
        else:
            start_step, orbitals, sampled_times, storage = self._restore(
                checkpoint
            )

        steps = range(start_step, self.n_steps)
        if progress:
            from tqdm.auto import tqdm

            steps = tqdm(steps, desc=self.progress_label)
        for step in steps:
            orbitals = self._advance(orbitals, self.propagation_times[step])
            completed = step + 1
            should_sample = (
                completed % self.sample_interval == 0
                or completed == self.n_steps
            )
            if should_sample:
                time = self.propagation_times[completed]
                sampled_times.append(time)
                self._sample(orbitals, time, storage)
            should_checkpoint = (
                self.checkpoint_path is not None
                and self.checkpoint_interval is not None
                and completed % self.checkpoint_interval == 0
            )
            if should_checkpoint:
                self._write_checkpoint(
                    self.checkpoint_path,
                    completed,
                    orbitals,
                    sampled_times,
                    storage,
                )

        samples = {
            name: np.stack(values) for name, values in storage.items()
        }
        if self.checkpoint_path is not None:
            self._write_checkpoint(
                self.checkpoint_path,
                self.n_steps,
                orbitals,
                sampled_times,
                samples,
            )
        return self.result_type(
            system=self.system,
            grid=self.grid,
            initial_orbitals=self.initial_orbitals.copy(),
            orbitals=orbitals,
            times=np.asarray(sampled_times),
            samples=samples,
            time_step=self.time_step,
            n_steps=self.n_steps,
            l_max=self.l_max,
            m_max=self.m_max,
            L_max=self.L_max,
            M_max=self.M_max,
            field_gauge=self.field_gauge,
            orbital_gauge=self.orbital_gauge,
            integrator=self.integrator_name,
            particle_charge=self.particle_charge,
            **self._result_metadata(),
        )
