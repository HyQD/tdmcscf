r"""Local and semilocal spin-unpolarized exchange-correlation functionals.

All functionals use the total electron density ``n = n_up + n_down`` and
return the energy density ``f(n)`` and its functional derivative
``v(n) = df/dn`` in atomic units. Semilocal functionals additionally accept
``sigma = grad(n) dot grad(n)`` and return the partial derivatives needed to
construct their Euler potential. They contain no grid or coordinate-system
logic and accept either scalars or NumPy arrays.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np


_DENSITY_ERROR = "density must contain finite, nonnegative real values."
_EXCHANGE_FACTOR = (3.0 / np.pi) ** (1.0 / 3.0)
_PBE_KAPPA = 0.804
_PBE_BETA = 0.06672455060314922
_PBE_MU = _PBE_BETA * np.pi**2 / 3.0
_PBE_GAMMA = (1.0 - np.log(2.0)) / np.pi**2


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


def _density_and_sigma(density, gradient_squared):
    density = _density(density)
    sigma = np.asarray(gradient_squared)
    if (
        not np.issubdtype(sigma.dtype, np.number)
        or np.iscomplexobj(sigma)
        or not np.all(np.isfinite(sigma))
        or np.any(sigma < 0)
    ):
        raise ValueError(
            "gradient_squared must contain finite, nonnegative real values."
        )
    try:
        density, sigma = np.broadcast_arrays(
            density, sigma.astype(float, copy=False)
        )
    except ValueError as error:
        raise ValueError(
            "density and gradient_squared must be broadcastable."
        ) from error
    return density, sigma


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
class SemilocalFunctionalValues:
    r"""Pointwise GGA energy density and its two partial derivatives.

    For ``f(n, sigma)`` with ``sigma = grad(n) dot grad(n)``, the returned
    arrays are ``f``, ``partial f / partial n``, and
    ``partial f / partial sigma``. The coordinate-specific Euler potential is

    ``v = density_derivative - 2*div(sigma_derivative * grad(n))``.
    """

    energy_density: np.ndarray
    density_derivative: np.ndarray
    sigma_derivative: np.ndarray


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


def _pw92_modified_unpolarized(density):
    """Return modified-PW92 epsilon_c and its density derivative.

    This is the unpolarized uniform-gas parametrization used by standard PBE
    and identified as ``LDA_C_PW_MOD`` by LibXC.
    """
    A = 0.0310907
    alpha_1 = 0.21370
    beta_1 = 7.5957
    beta_2 = 3.5876
    beta_3 = 1.6382
    beta_4 = 0.49294

    rs = np.cbrt(3.0 / (4.0 * np.pi * density))
    sqrt_rs = np.sqrt(rs)
    polynomial = (
        beta_1 * sqrt_rs
        + beta_2 * rs
        + beta_3 * rs * sqrt_rs
        + beta_4 * rs**2
    )
    polynomial_derivative = (
        0.5 * beta_1 / sqrt_rs
        + beta_2
        + 1.5 * beta_3 * sqrt_rs
        + 2.0 * beta_4 * rs
    )
    argument = 2.0 * A * polynomial
    argument_derivative = 2.0 * A * polynomial_derivative
    logarithm = np.log1p(1.0 / argument)
    logarithm_derivative = -argument_derivative / (
        argument * (argument + 1.0)
    )
    epsilon = -2.0 * A * (1.0 + alpha_1 * rs) * logarithm
    epsilon_rs = -2.0 * A * (
        alpha_1 * logarithm
        + (1.0 + alpha_1 * rs) * logarithm_derivative
    )
    epsilon_density = -rs * epsilon_rs / (3.0 * density)
    return epsilon, epsilon_density


class PBEExchange:
    r"""Spin-unpolarized Perdew--Burke--Ernzerhof exchange.

    ``evaluate(density, gradient_squared)`` returns the energy density
    ``f_x(n, sigma)`` and its partial derivatives with respect to ``n`` and
    ``sigma = grad(n) dot grad(n)``.
    """

    requires_gradient = True
    kappa = _PBE_KAPPA
    mu = _PBE_MU

    def __init__(self, density_cutoff=1e-30):
        self.density_cutoff = _nonnegative_real(
            density_cutoff, "density_cutoff"
        )

    @property
    def name(self):
        return "PBE exchange"

    def evaluate(self, density, gradient_squared):
        density, sigma = _density_and_sigma(density, gradient_squared)
        energy_density = np.zeros_like(density)
        density_derivative = np.zeros_like(density)
        sigma_derivative = np.zeros_like(density)
        positive = density > self.density_cutoff
        if not np.any(positive):
            return SemilocalFunctionalValues(
                energy_density, density_derivative, sigma_derivative
            )

        n = density[positive]
        selected_sigma = sigma[positive]
        reduced_gradient_factor = 1.0 / (
            4.0 * (3.0 * np.pi**2) ** (2.0 / 3.0)
        )
        s_squared = (
            reduced_gradient_factor * selected_sigma * n ** (-8.0 / 3.0)
        )
        denominator = 1.0 + self.mu * s_squared / self.kappa
        enhancement = 1.0 + self.kappa - self.kappa / denominator
        enhancement_derivative = self.mu / denominator**2
        exchange_coefficient = 0.75 * _EXCHANGE_FACTOR

        energy_density[positive] = (
            -exchange_coefficient * n ** (4.0 / 3.0) * enhancement
        )
        density_derivative[positive] = -exchange_coefficient * np.cbrt(n) * (
            4.0 * enhancement / 3.0
            - 8.0 * s_squared * enhancement_derivative / 3.0
        )
        sigma_derivative[positive] = (
            -exchange_coefficient
            * reduced_gradient_factor
            * n ** (-4.0 / 3.0)
            * enhancement_derivative
        )
        return SemilocalFunctionalValues(
            energy_density, density_derivative, sigma_derivative
        )


class PBECorrelation:
    r"""Spin-unpolarized PBE correlation with modified-PW92 LDA correlation."""

    requires_gradient = True
    beta = _PBE_BETA
    gamma = _PBE_GAMMA

    def __init__(self, density_cutoff=1e-30):
        self.density_cutoff = _nonnegative_real(
            density_cutoff, "density_cutoff"
        )

    @property
    def name(self):
        return "PBE correlation"

    def evaluate(self, density, gradient_squared):
        density, sigma = _density_and_sigma(density, gradient_squared)
        energy_density = np.zeros_like(density)
        density_derivative = np.zeros_like(density)
        sigma_derivative = np.zeros_like(density)
        positive = density > self.density_cutoff
        if not np.any(positive):
            return SemilocalFunctionalValues(
                energy_density, density_derivative, sigma_derivative
            )

        n = density[positive]
        selected_sigma = sigma[positive]
        epsilon, epsilon_density = _pw92_modified_unpolarized(n)
        t_squared_factor = np.pi / (
            16.0 * (3.0 * np.pi**2) ** (1.0 / 3.0)
        )
        t_squared = t_squared_factor * selected_sigma * n ** (-7.0 / 3.0)
        ratio = self.beta / self.gamma
        exponential_minus_one = np.expm1(-epsilon / self.gamma)
        coefficient = ratio / exponential_minus_one
        exponential = exponential_minus_one + 1.0
        coefficient_epsilon = (
            ratio
            * exponential
            / (self.gamma * exponential_minus_one**2)
        )

        product = coefficient * t_squared
        denominator = 1.0 + product + product**2
        inverse_denominator = 1.0 / denominator
        rational = (
            t_squared * inverse_denominator * (1.0 + product)
        )
        rational_t = (1.0 + 2.0 * product) * inverse_denominator**2
        rational_coefficient = -(
            t_squared * inverse_denominator
        ) ** 2 * product * (2.0 + product)
        logarithm_argument = 1.0 + ratio * rational
        correction = self.gamma * np.log(logarithm_argument)
        correction_t = (
            self.gamma * ratio * rational_t / logarithm_argument
        )
        correction_coefficient = (
            self.gamma
            * ratio
            * rational_coefficient
            / logarithm_argument
        )

        t_density = -7.0 * t_squared / (3.0 * n)
        t_sigma = t_squared_factor * n ** (-7.0 / 3.0)
        energy_density[positive] = n * (epsilon + correction)
        density_derivative[positive] = epsilon + correction + n * (
            (1.0 + correction_coefficient * coefficient_epsilon)
            * epsilon_density
            + correction_t * t_density
        )
        sigma_derivative[positive] = n * correction_t * t_sigma
        return SemilocalFunctionalValues(
            energy_density, density_derivative, sigma_derivative
        )


class PBE:
    r"""Standard spin-unpolarized PBE exchange-correlation functional."""

    requires_gradient = True

    def __init__(self, density_cutoff=1e-30):
        self.density_cutoff = _nonnegative_real(
            density_cutoff, "density_cutoff"
        )
        self.components = (
            PBEExchange(self.density_cutoff),
            PBECorrelation(self.density_cutoff),
        )

    @property
    def name(self):
        return "PBE"

    def evaluate(self, density, gradient_squared):
        density, sigma = _density_and_sigma(density, gradient_squared)
        energy_density = np.zeros_like(density)
        density_derivative = np.zeros_like(density)
        sigma_derivative = np.zeros_like(density)
        for component in self.components:
            values = component.evaluate(density, sigma)
            energy_density += values.energy_density
            density_derivative += values.density_derivative
            sigma_derivative += values.sigma_derivative
        return SemilocalFunctionalValues(
            energy_density, density_derivative, sigma_derivative
        )
