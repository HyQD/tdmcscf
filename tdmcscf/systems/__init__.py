"""Physical systems and electronic configurations."""

from .atoms import (
    AtomicConfiguration,
    ClosedSubshell,
    atomic_configurations,
    closed_shell_atom,
)
from .diatomic import AxialDiatomic
from .spherical import (
    Atomic,
    ClosedShellOccupation,
    CustomCentralPotential,
    GaussianWell,
    HarmonicOscillator,
    SphericalElectronicSystem,
)

__all__ = [
    "Atomic",
    "AtomicConfiguration",
    "AxialDiatomic",
    "ClosedShellOccupation",
    "ClosedSubshell",
    "CustomCentralPotential",
    "GaussianWell",
    "HarmonicOscillator",
    "SphericalElectronicSystem",
    "atomic_configurations",
    "closed_shell_atom",
]
