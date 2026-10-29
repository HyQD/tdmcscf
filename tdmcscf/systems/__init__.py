"""Physical systems and electronic configurations."""

from .atoms import (
    AtomicConfiguration,
    ClosedSubshell,
    atomic_configurations,
    closed_shell_atom,
)
from .diatomic import AxialClosedShellOccupation, AxialDiatomic
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
    "AxialClosedShellOccupation",
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
