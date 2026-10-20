import numpy as np
import pytest

from tdmcscf.dft import (
    DiracExchange,
    LDAVWN,
    PBE,
    PBECorrelation,
    PBEExchange,
    SumLocalFunctional,
    VWN5Correlation,
    XAlphaExchange,
    XAlphaVWN,
)


def test_dirac_exchange_is_xalpha_two_thirds():
    density = np.array([0.0, 1e-8, 0.2, 1.0, 20.0])
    expected = XAlphaExchange(alpha=2.0 / 3.0).evaluate(density)
    actual = DiracExchange().evaluate(density)
    np.testing.assert_allclose(actual.energy_density, expected.energy_density)
    np.testing.assert_allclose(actual.potential, expected.potential)


def test_xalpha_uses_slater_convention():
    density = np.array([0.125, 1.0, 8.0])
    alpha = 0.7
    values = XAlphaExchange(alpha=alpha).evaluate(density)
    factor = (3.0 / np.pi) ** (1.0 / 3.0)
    expected_potential = -1.5 * alpha * factor * np.cbrt(density)
    expected_energy = -9.0 * alpha * factor * density ** (4.0 / 3.0) / 8.0
    np.testing.assert_allclose(values.potential, expected_potential)
    np.testing.assert_allclose(values.energy_density, expected_energy)


@pytest.mark.parametrize(
    "functional",
    [
        XAlphaExchange(alpha=0.71),
        VWN5Correlation(),
        LDAVWN(),
    ],
)
def test_potential_is_derivative_of_local_energy_density(functional):
    density = np.logspace(-10, 2, 17)
    step = 1e-5 * density
    forward = functional.evaluate(density + step).energy_density
    backward = functional.evaluate(density - step).energy_density
    finite_difference = (forward - backward) / (2.0 * step)
    np.testing.assert_allclose(
        functional.evaluate(density).potential,
        finite_difference,
        rtol=2e-8,
        atol=2e-10,
    )


def test_lda_vwn_is_sum_of_exchange_and_correlation():
    density = np.logspace(-8, 1, 12)
    combined = LDAVWN().evaluate(density)
    exchange = DiracExchange().evaluate(density)
    correlation = VWN5Correlation().evaluate(density)
    np.testing.assert_allclose(
        combined.energy_density,
        exchange.energy_density + correlation.energy_density,
    )
    np.testing.assert_allclose(
        combined.potential,
        exchange.potential + correlation.potential,
    )


def test_xalpha_vwn_exposes_configurable_exchange():
    density = np.logspace(-6, 1, 9)
    functional = XAlphaVWN(alpha=0.8)
    expected_exchange = XAlphaExchange(alpha=0.8).evaluate(density)
    expected_correlation = VWN5Correlation().evaluate(density)
    actual = functional.evaluate(density)
    assert functional.alpha == 0.8
    np.testing.assert_allclose(
        actual.energy_density,
        expected_exchange.energy_density + expected_correlation.energy_density,
    )
    np.testing.assert_allclose(
        actual.potential,
        expected_exchange.potential + expected_correlation.potential,
    )


def test_zero_density_has_zero_energy_density_and_potential():
    for functional in (XAlphaExchange(), VWN5Correlation(), LDAVWN()):
        values = functional.evaluate(np.zeros(3))
        np.testing.assert_array_equal(values.energy_density, 0)
        np.testing.assert_array_equal(values.potential, 0)


def test_vwn_low_density_guard_is_finite():
    density = np.array([0.0, np.nextafter(0.0, 1.0), 1e-40, 1e-30])
    values = VWN5Correlation().evaluate(density)
    assert np.all(np.isfinite(values.energy_density))
    assert np.all(np.isfinite(values.potential))
    np.testing.assert_array_equal(values.energy_density, 0)
    np.testing.assert_array_equal(values.potential, 0)


def test_vwn5_matches_independent_libxc_reference_values():
    density = np.array([1e-6, 0.1, 1.0])
    values = VWN5Correlation().evaluate(density)
    expected_energy_per_particle = np.array(
        [-0.0047766175044473, -0.0533972891859498, -0.0715926123067906]
    )
    expected_potential = np.array(
        [-0.0060999077256952, -0.0608120303312615, -0.0799383831759856]
    )
    np.testing.assert_allclose(
        values.energy_density / density,
        expected_energy_per_particle,
        atol=2e-15,
    )
    np.testing.assert_allclose(values.potential, expected_potential, atol=2e-15)


@pytest.mark.parametrize("alpha", [-1, np.inf, np.nan, True, "two thirds"])
def test_xalpha_rejects_invalid_alpha(alpha):
    with pytest.raises((TypeError, ValueError), match="alpha"):
        XAlphaExchange(alpha=alpha)


@pytest.mark.parametrize(
    "density",
    [np.array([-1.0]), np.array([np.nan]), np.array([1.0 + 1.0j])],
)
def test_local_functionals_reject_invalid_density(density):
    with pytest.raises(ValueError, match="density"):
        LDAVWN().evaluate(density)


def test_sum_requires_functional_components():
    with pytest.raises(ValueError, match="At least one"):
        SumLocalFunctional()
    with pytest.raises(TypeError, match="evaluate"):
        SumLocalFunctional(object())


@pytest.mark.parametrize(
    "functional,expected_energy,expected_density_derivative,expected_sigma_derivative",
    [
        (
            PBEExchange(),
            [-0.0343264047916244, -0.7391510592579351],
            [-0.4564729090966313, -0.9839568734437065],
            [-0.09093660937705003, -0.00422644348269713],
        ),
        (
            PBECorrelation(),
            [-0.00527980365779292, -0.07061275944383515],
            [-0.06115010248649831, -0.08022709770881299],
            [0.08991573711031386, 0.00415558445540934],
        ),
        (
            PBE(),
            [-0.03960620844941731, -0.8097638187017703],
            [-0.5176230115831296, -1.0641839711525196],
            [-0.001020872266736167, -0.00007085902728778959],
        ),
    ],
)
def test_pbe_matches_independent_libxc_reference_values(
    functional,
    expected_energy,
    expected_density_derivative,
    expected_sigma_derivative,
):
    density = np.array([0.1, 1.0])
    gradient_squared = np.array([0.0005, 0.14])
    values = functional.evaluate(density, gradient_squared)
    np.testing.assert_allclose(values.energy_density, expected_energy, atol=2e-16)
    np.testing.assert_allclose(
        values.density_derivative, expected_density_derivative, atol=2e-15
    )
    np.testing.assert_allclose(
        values.sigma_derivative, expected_sigma_derivative, atol=2e-15
    )


@pytest.mark.parametrize("functional", [PBEExchange(), PBECorrelation(), PBE()])
def test_pbe_partial_derivatives_are_energy_density_derivatives(functional):
    density = np.logspace(-6, 1, 11)
    gradient_squared = np.logspace(-8, 0, 11)
    density_step = 1e-5 * density
    sigma_step = 1e-5 * gradient_squared
    density_difference = (
        functional.evaluate(density + density_step, gradient_squared).energy_density
        - functional.evaluate(
            density - density_step, gradient_squared
        ).energy_density
    ) / (2.0 * density_step)
    sigma_difference = (
        functional.evaluate(density, gradient_squared + sigma_step).energy_density
        - functional.evaluate(
            density, gradient_squared - sigma_step
        ).energy_density
    ) / (2.0 * sigma_step)
    values = functional.evaluate(density, gradient_squared)
    np.testing.assert_allclose(
        values.density_derivative, density_difference, rtol=3e-8, atol=2e-10
    )
    np.testing.assert_allclose(
        values.sigma_derivative, sigma_difference, rtol=3e-8, atol=5e-10
    )


def test_pbe_is_sum_of_exchange_and_correlation():
    density = np.logspace(-8, 1, 12)
    gradient_squared = np.logspace(-10, 0, 12)
    combined = PBE().evaluate(density, gradient_squared)
    exchange = PBEExchange().evaluate(density, gradient_squared)
    correlation = PBECorrelation().evaluate(density, gradient_squared)
    for name in ("energy_density", "density_derivative", "sigma_derivative"):
        np.testing.assert_allclose(
            getattr(combined, name),
            getattr(exchange, name) + getattr(correlation, name),
        )


def test_pbe_zero_density_is_finite_and_zero():
    values = PBE().evaluate(np.zeros(3), np.array([0.0, 1.0, 1e20]))
    np.testing.assert_array_equal(values.energy_density, 0.0)
    np.testing.assert_array_equal(values.density_derivative, 0.0)
    np.testing.assert_array_equal(values.sigma_derivative, 0.0)


@pytest.mark.parametrize(
    "gradient_squared",
    [np.array([-1.0]), np.array([np.nan]), np.array([1.0 + 1.0j])],
)
def test_pbe_rejects_invalid_gradient_squared(gradient_squared):
    with pytest.raises(ValueError, match="gradient_squared"):
        PBE().evaluate(np.ones(1), gradient_squared)
