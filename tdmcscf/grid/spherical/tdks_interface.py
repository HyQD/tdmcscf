"""High-level setup and propagation interface for spherical TDKS."""

from dataclasses import dataclass

from tdmcscf.systems import SphericalElectronicSystem

from ._validation import _integer
from .interface import SphericalTDHF, TDHFCheckpoint, TDHFResult
from .kohn_sham import KohnShamResult, SphericalKohnSham, _local_functional
from .local_density import SphericalXCBuilder
from .tdks import SphericalRestrictedTDKS


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


class SphericalTDKS(SphericalTDHF):
    r"""Configure and run adiabatic spin-restricted spherical TDKS.

    ``functional`` is evaluated on the instantaneous total density. The
    interface otherwise follows :class:`SphericalTDHF`: it accepts a computed
    ground state or custom orthonormal orbitals, electric-dipole fields,
    canonical or orthogonal orbital gauges, all supported integrators, masks,
    samplers, and restartable checkpoints.

    ``xc_angular_bandlimit`` controls the angular grid used for nonlinear XC
    evaluation and projection. It defaults to ``4*l_max`` and must be at
    least ``2*l_max``.
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
        **kwargs,
    ):
        if not isinstance(system, SphericalElectronicSystem):
            raise TypeError("system must be a SphericalElectronicSystem.")
        self.functional = _local_functional(functional)
        self._requested_xc_angular_bandlimit = xc_angular_bandlimit
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
            }
            duplicate = reserved.intersection(settings)
            if duplicate:
                names = ", ".join(sorted(duplicate))
                raise ValueError(
                    "ground_state_options cannot override propagation settings: "
                    f"{names}."
                )
            settings.setdefault("poisson_method", self.poisson_method)
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
        return super()._restore(checkpoint)

    def _result_metadata(self):
        return {
            "functional": self.functional,
            "xc_angular_bandlimit": self.xc_angular_bandlimit,
        }
