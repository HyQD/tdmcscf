import numpy as np
import pytest

from tdmcscf.dft import (
    DiracExchange,
    LDAVWN,
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
