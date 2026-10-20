"""High-level setup and propagation interface for spherical TDKS."""

from dataclasses import dataclass

import numpy as np

from tdmcscf.systems import SphericalElectronicSystem

from ._validation import _integer
from .interface import SphericalTDHF, TDHFCheckpoint, TDHFResult
from .kohn_sham import KohnShamResult, SphericalKohnSham
from .local_density import SphericalXCBuilder
from .tdks import SphericalRestrictedTDKS
from .xc import _local_functional


def _functional_metadata(functional):
    metadata = {
        "module": type(functional).__module__,
        "type": type(functional).__name__,
        "name": str(getattr(functional, "name", type(functional).__name__)),
    }
    for name in ("alpha", "density_cutoff"):
        if hasattr(functional, name):
            metadata[name] = float(getattr(functional, name))
    components = getattr(functional, "components", None)
    if components is not None:
        metadata["components"] = [
            _functional_metadata(component) for component in components
        ]
    return metadata


@dataclass(frozen=True)
class TDKSCheckpoint(TDHFCheckpoint):
    """Saved TDKS propagation state and samples."""


@dataclass
class TDKSResult(TDHFResult):
    """Final TDKS orbitals, sampled observables, and metadata."""

    functional: object
    xc_angular_bandlimit: int
    xc_quadrature_order: np.ndarray | None


class SphericalTDKS(SphericalTDHF):
    r"""Configure and run adiabatic spin-restricted spherical TDKS.

    ``functional`` is evaluated on the instantaneous total density. The
    interface otherwise follows :class:`SphericalTDHF`: it accepts a computed
    ground state or custom orthonormal orbitals, electric-dipole fields,
    canonical or orthogonal orbital gauges, all supported integrators, masks,
    samplers, and restartable checkpoints.

    ``xc_angular_bandlimit`` controls the angular grid used for nonlinear XC
    evaluation and projection. It defaults to ``4*l_max`` and must be at
    least ``2*l_max``. For semilocal functionals, ``xc_quadrature_order``
    controls radial overintegration and defaults to twice the number of
    FEM-DVR points in each element.
    """

    checkpoint_type = TDKSCheckpoint
    result_type = TDKSResult
    progress_label = "TDKS propagation"

    def __init__(
        self,
        system,
        *,
        functional,
        xc_angular_bandlimit=None,
        xc_quadrature_order=None,
        **kwargs,
    ):
        if not isinstance(system, SphericalElectronicSystem):
            raise TypeError("system must be a SphericalElectronicSystem.")
        self.functional = _local_functional(functional)
        self._requested_xc_angular_bandlimit = xc_angular_bandlimit
        self._requested_xc_quadrature_order = xc_quadrature_order
        requested_l_max = kwargs.get("l_max")
        effective_l_max = (
            system.occupied_l_max
            if requested_l_max is None
            else _integer(requested_l_max, "l_max")
        )
        if xc_angular_bandlimit is not None:
            checked_bandlimit = _integer(
                xc_angular_bandlimit, "xc_angular_bandlimit"
            )
            if checked_bandlimit < 2 * effective_l_max:
                raise ValueError(
                    "xc_angular_bandlimit must be at least 2*l_max."
                )
        super().__init__(system, **kwargs)

        if self._requested_xc_angular_bandlimit is None:
            self.xc_angular_bandlimit = 4 * self.l_max
        else:
            self.xc_angular_bandlimit = _integer(
                self._requested_xc_angular_bandlimit,
                "xc_angular_bandlimit",
            )
        if self.xc_angular_bandlimit < 2 * self.l_max:
            raise ValueError("xc_angular_bandlimit must be at least 2*l_max.")
        self.xc = SphericalXCBuilder(
            self.grid,
            self.l_max,
            self.m_max,
            functional=self.functional,
            angular_bandlimit=self.xc_angular_bandlimit,
            xc_quadrature_order=self._requested_xc_quadrature_order,
        )
        self.xc_quadrature_order = self.xc.xc_quadrature_order
        if (
            self.ground_state is not None
            and self.ground_state.xc_quadrature_order is not None
            and not np.array_equal(
                self.xc_quadrature_order,
                self.ground_state.xc_quadrature_order,
            )
        ):
            raise ValueError(
                "The Kohn-Sham result uses a different "
                "xc_quadrature_order."
            )
        self.equation = SphericalRestrictedTDKS(
            self.core,
            self.mean_fields,
            self.xc,
            dipole_field=self.dipole_field,
            orbital_gauge=self.orbital_gauge,
            orbital_m=self.equation.orbital_m,
        )

    def _initial_orbitals(self, initial_state, options):
        if isinstance(initial_state, str):
            if initial_state != "groundstate":
                raise ValueError("initial_state string must be 'groundstate'.")
            settings = dict(options or {})
            reserved = {
                "functional",
                "r_max",
                "nodes",
                "n_elements",
                "n_points",
                "l_max",
                "m_max",
                "xc_quadrature_order",
            }
            duplicate = reserved.intersection(settings)
            if duplicate:
                names = ", ".join(sorted(duplicate))
                raise ValueError(
                    "ground_state_options cannot override propagation settings: "
                    f"{names}."
                )
            settings.setdefault("poisson_method", self.poisson_method)
            settings["xc_quadrature_order"] = (
                self._requested_xc_quadrature_order
            )
            solver = SphericalKohnSham(
                self.system,
                functional=self.functional,
                nodes=self.grid.nodes,
                n_points=self.grid.n_points,
                l_max=self.l_max,
                m_max=self.m_max,
                **settings,
            )
            result = solver.solve()
            if not result.converged:
                raise RuntimeError("The Kohn--Sham initial state did not converge.")
            self.ground_state = result
            values = result.orbitals
        elif isinstance(initial_state, KohnShamResult):
            if _functional_metadata(initial_state.functional) != (
                _functional_metadata(self.functional)
            ):
                raise ValueError(
                    "The Kohn-Sham result uses a different XC functional."
                )
            if (
                self._requested_xc_quadrature_order is None
                and initial_state.xc_quadrature_order is not None
            ):
                self._requested_xc_quadrature_order = (
                    initial_state.xc_quadrature_order.copy()
                )
            self.ground_state = initial_state
            values = self._expand_hartree_fock_result(initial_state)
        else:
            if options:
                raise ValueError(
                    "ground_state_options apply only to "
                    "initial_state='groundstate'."
                )
            values = initial_state
        return self._validate_initial_orbitals(values)

    def _metadata(self, sample_names):
        metadata = super()._metadata(sample_names)
        metadata.update(
            {
                "method": "TDKS",
                "functional": _functional_metadata(self.functional),
                "xc_angular_bandlimit": self.xc_angular_bandlimit,
                "xc_quadrature_order": (
                    None
                    if self.xc_quadrature_order is None
                    else self.xc_quadrature_order.tolist()
                ),
            }
        )
        return metadata

    def _restore(self, checkpoint):
        if not isinstance(checkpoint, self.checkpoint_type):
            checkpoint = self.checkpoint_type.load(checkpoint)
        expected_functional = _functional_metadata(self.functional)
        if checkpoint.metadata.get("functional") != expected_functional:
            raise ValueError("The checkpoint uses a different XC functional.")
        if (
            checkpoint.metadata.get("xc_angular_bandlimit")
            != self.xc_angular_bandlimit
        ):
            raise ValueError(
                "The checkpoint uses a different xc_angular_bandlimit."
            )
        checkpoint_order = checkpoint.metadata.get("xc_quadrature_order")
        expected_order = (
            None
            if self.xc_quadrature_order is None
            else self.xc_quadrature_order.tolist()
        )
        if checkpoint_order != expected_order:
            raise ValueError(
                "The checkpoint uses a different xc_quadrature_order."
            )
        return super()._restore(checkpoint)

    def _result_metadata(self):
        return {
            "functional": self.functional,
            "xc_angular_bandlimit": self.xc_angular_bandlimit,
            "xc_quadrature_order": (
                None
                if self.xc_quadrature_order is None
                else self.xc_quadrature_order.copy()
            ),
        }
