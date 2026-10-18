r"""Local spin-unpolarized exchange-correlation functionals.

All functionals use the total electron density ``n = n_up + n_down`` and
return the energy density ``f(n)`` and its functional derivative
``v(n) = df/dn`` in atomic units.  They contain no grid or coordinate-system
logic and accept either scalars or NumPy arrays.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np


_DENSITY_ERROR = "density must contain finite, nonnegative real values."
_EXCHANGE_FACTOR = (3.0 / np.pi) ** (1.0 / 3.0)


def _density(value):
    values = np.asarray(value)
    if (
        not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
    ):
        raise ValueError(_DENSITY_ERROR)
    return values.astype(float, copy=False)


def _nonnegative_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative real scalar.")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative real scalar.")
    return value


@dataclass(frozen=True)
class LocalFunctionalValues:
    r"""Pointwise energy density and multiplicative potential.

    ``energy_density`` is ``f(n)`` such that

    ``E_xc[n] = integral f(n(r)) d**3 r``.

    ``potential`` is the local functional derivative ``df/dn``.
    """

    energy_density: np.ndarray
    potential: np.ndarray


@dataclass(frozen=True)
class XAlphaExchange:
    r"""Slater X-alpha exchange for a spin-unpolarized density.

    The convention is

    ``v_x(n) = -(3*alpha/2) * (3/pi)**(1/3) * n**(1/3)``.

    Consequently, ``alpha=2/3`` is Dirac local-density exchange.  ``alpha=0``
    is accepted and provides a convenient zero-XC functional.
    """

    alpha: float = 2.0 / 3.0

    def __post_init__(self):
        object.__setattr__(self, "alpha", _nonnegative_real(self.alpha, "alpha"))

    @property
    def name(self):
        return f"X-alpha(alpha={self.alpha:g})"

    def evaluate(self, density):
        density = _density(density)
        cube_root = np.cbrt(density)
        coefficient = 9.0 * self.alpha * _EXCHANGE_FACTOR / 8.0
        energy_density = -coefficient * density * cube_root
        potential = -1.5 * self.alpha * _EXCHANGE_FACTOR * cube_root
        return LocalFunctionalValues(energy_density, potential)


class DiracExchange(XAlphaExchange):
    r"""Dirac spin-unpolarized LDA exchange (X-alpha with alpha=2/3)."""

    def __init__(self):
        super().__init__(alpha=2.0 / 3.0)

    @property
    def name(self):
        return "Dirac LDA exchange"


class VWN5Correlation:
    r"""VWN5 local correlation for a spin-unpolarized density.

    This is the RPA-parametrized form of Vosko, Wilk, and Nusair commonly
    denoted VWN5.  The zero-density energy density and potential are defined
    by their limiting value, zero.
    """

    A = 0.0310907
    b = 3.72744
    c = 12.9352
    x0 = -0.10498

    def __init__(self, density_cutoff=1e-30):
        self.density_cutoff = _nonnegative_real(
            density_cutoff, "density_cutoff"
        )

    @property
    def name(self):
        return "VWN5 correlation"

    def evaluate(self, density):
        density = _density(density)
        energy_density = np.zeros_like(density)
        potential = np.zeros_like(density)
        positive = density > self.density_cutoff
        if not np.any(positive):
            return LocalFunctionalValues(energy_density, potential)

        n = density[positive]
        rs = np.cbrt(3.0 / (4.0 * np.pi * n))
        x = np.sqrt(rs)
        X = x * x + self.b * x + self.c
        X0 = self.x0**2 + self.b * self.x0 + self.c
        Q = np.sqrt(4.0 * self.c - self.b**2)
        angle = np.arctan(Q / (2.0 * x + self.b))
        prefactor = self.b * self.x0 / X0

        epsilon = self.A * (
            np.log(x * x / X)
            + 2.0 * self.b * angle / Q
            - prefactor
            * (
                np.log((x - self.x0) ** 2 / X)
                + 2.0 * (self.b + 2.0 * self.x0) * angle / Q
            )
        )
        derivative = 2.0 * self.A * (
            1.0 / x
            - (x + self.b) / X
            - prefactor
            * (1.0 / (x - self.x0) - (x + self.b + self.x0) / X)
        )
        energy_density[positive] = n * epsilon
        potential[positive] = epsilon - x * derivative / 6.0
        return LocalFunctionalValues(energy_density, potential)


class SumLocalFunctional:
    """Sum pointwise energy densities and potentials from local components."""

    def __init__(self, *components):
        if not components:
            raise ValueError("At least one local functional is required.")
        for component in components:
            if not callable(getattr(component, "evaluate", None)):
                raise TypeError("Every component must provide evaluate(density).")
        self.components = tuple(components)

    @property
    def name(self):
        return " + ".join(component.name for component in self.components)

    def evaluate(self, density):
        density = _density(density)
        energy_density = np.zeros_like(density)
        potential = np.zeros_like(density)
        for component in self.components:
            values = component.evaluate(density)
            energy_density += values.energy_density
            potential += values.potential
        return LocalFunctionalValues(energy_density, potential)


class XAlphaVWN(SumLocalFunctional):
    """Slater X-alpha exchange plus spin-unpolarized VWN5 correlation."""

    def __init__(self, alpha=2.0 / 3.0):
        exchange = XAlphaExchange(alpha)
        self.alpha = exchange.alpha
        super().__init__(exchange, VWN5Correlation())

    @property
    def name(self):
        return f"X-alpha(alpha={self.alpha:g}) + VWN5"


class LDAVWN(XAlphaVWN):
    """Dirac LDA exchange plus spin-unpolarized VWN5 correlation."""

    def __init__(self):
        super().__init__(alpha=2.0 / 3.0)

    @property
    def name(self):
        return "LDA-VWN5"
