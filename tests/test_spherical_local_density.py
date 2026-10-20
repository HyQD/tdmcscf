import numpy as np
import pytest

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)

from tdmcscf.dft import LDAVWN, LocalFunctionalValues, PBE
from tdmcscf.grid.spherical import (
    SphericalAtomicKohnSham,
    SphericalXCBuilder,
)
from tdmcscf.grid.spherical.xc import evaluate_spherical_functional


@pytest.fixture(scope="module")
def radial_grid():
    return FEMDVR(
        np.array([0.0, 2.0, 6.0]),
        np.array([9, 10]),
        Linear_map,
        GaussLegendreLobatto,
        symmetrize=False,
    )


def normalized_orbitals(builder, count=2, seed=17):
    rng = np.random.default_rng(seed)
    shape = (count,) + builder.orbital_shape
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    values *= (builder.r * np.exp(-builder.r / 2))[None, :, None]
    sqrt_weights = np.repeat(np.sqrt(builder.radial_weights), shape[-1])
    weighted = values.reshape(count, -1).T * sqrt_weights[:, None]
    orthogonal, _ = np.linalg.qr(weighted)
    return (orthogonal / sqrt_weights[:, None]).T.reshape(shape)


def test_physical_density_integrates_to_closed_shell_electron_count(radial_grid):
    builder = SphericalXCBuilder(
        radial_grid, 2, m_max=1, functional=LDAVWN()
    )
    orbitals = normalized_orbitals(builder, count=2)
    density = builder.density(orbitals)
    angular_integral = builder.angular_grid.integrate(density)
    electron_count = np.dot(
        builder.radial_weights,
        builder.r**2 * angular_integral,
    )
    assert electron_count == pytest.approx(4.0, abs=3e-12)
    assert np.all(density >= 0)


def test_s_wave_density_has_expected_radial_form(radial_grid):
    builder = SphericalXCBuilder(
        radial_grid, 0, functional=LDAVWN()
    )
    radial = builder.r * np.exp(-builder.r)
    radial /= np.sqrt(np.dot(builder.radial_weights, radial**2))
    orbitals = radial[None, :, None]
    density = builder.density(orbitals)
    expected = 2.0 * radial**2 / (4.0 * np.pi * builder.r**2)
    np.testing.assert_allclose(density[:, 0, 0], expected, atol=2e-14)


def test_constant_local_potential_multiplies_orbitals(radial_grid):
    class ConstantFunctional:
        name = "constant test functional"

        def evaluate(self, density):
            return LocalFunctionalValues(3.0 * density, np.full_like(density, 3.0))

    builder = SphericalXCBuilder(
        radial_grid,
        2,
        m_max=1,
        functional=ConstantFunctional(),
        angular_bandlimit=6,
    )
    orbitals = normalized_orbitals(builder, count=2)
    potential = builder.build(orbitals)
    np.testing.assert_allclose(potential.apply(orbitals), 3.0 * orbitals, atol=3e-13)
    np.testing.assert_allclose(
        potential.energy_density, 3.0 * potential.density
    )


def test_lda_vwn_potential_action_is_finite(radial_grid):
    builder = SphericalXCBuilder(
        radial_grid, 2, m_max=2, functional=LDAVWN()
    )
    orbitals = normalized_orbitals(builder, count=1)
    potential = builder.build(orbitals)
    assert np.all(np.isfinite(potential.values))
    assert np.all(np.isfinite(potential.apply(orbitals)))


def test_rejects_insufficient_angular_oversampling(radial_grid):
    with pytest.raises(ValueError, match=r"2\*l_max"):
        SphericalXCBuilder(
            radial_grid,
            2,
            functional=LDAVWN(),
            angular_bandlimit=3,
        )


def test_pbe_potential_and_density_gradient_are_finite(radial_grid):
    builder = SphericalXCBuilder(
        radial_grid,
        2,
        m_max=2,
        functional=PBE(),
        angular_bandlimit=8,
    )
    orbitals = normalized_orbitals(builder, count=2)
    potential = builder.build(orbitals)
    np.testing.assert_array_equal(builder.xc_quadrature_order, [18, 20])
    assert potential.gradient_squared.shape == potential.density.shape
    assert np.all(potential.gradient_squared >= 0)
    assert np.all(np.isfinite(potential.energy_density))
    assert np.all(np.isfinite(potential.values))
    assert np.all(np.isfinite(potential.apply(orbitals)))


def test_pbe_weak_action_is_orbital_energy_derivative(radial_grid):
    builder = SphericalXCBuilder(
        radial_grid,
        1,
        m_max=1,
        functional=PBE(),
        angular_bandlimit=6,
    )
    orbitals = normalized_orbitals(builder, count=2)
    direction = normalized_orbitals(builder, count=2, seed=41)
    potential = builder.build(orbitals)
    action = potential.apply(orbitals)
    expected = 4.0 * np.real(
        np.sum(
            builder.radial_weights[None, :, None]
            * direction.conj()
            * action
        )
    )

    step = 1e-6
    energies = [
        builder.build(orbitals + sign * step * direction).energy
        for sign in (-1.0, 1.0)
    ]
    finite_difference = (energies[1] - energies[0]) / (2.0 * step)
    assert finite_difference == pytest.approx(expected, rel=2e-8, abs=2e-9)


def test_pbe_weak_action_is_hermitian(radial_grid):
    builder = SphericalXCBuilder(
        radial_grid,
        1,
        m_max=1,
        functional=PBE(),
        angular_bandlimit=6,
    )
    reference = normalized_orbitals(builder, count=2)
    left = normalized_orbitals(builder, count=1, seed=29)[0]
    right = normalized_orbitals(builder, count=1, seed=31)[0]
    potential = builder.build(reference)
    left_matrix_element = np.sum(
        builder.radial_weights[:, None]
        * left.conj()
        * potential.apply(right)
    )
    right_matrix_element = np.sum(
        builder.radial_weights[:, None]
        * potential.apply(left).conj()
        * right
    )
    assert left_matrix_element == pytest.approx(
        right_matrix_element, rel=2e-12, abs=2e-12
    )


def test_pbe_weak_action_preserves_complete_p_shell(radial_grid):
    builder = SphericalXCBuilder(
        radial_grid,
        1,
        m_max=1,
        functional=PBE(),
        angular_bandlimit=4,
    )
    radial = builder.r * np.exp(-builder.r)
    radial /= np.sqrt(np.dot(builder.radial_weights, radial**2))
    orbitals = np.zeros((3, *builder.orbital_shape), dtype=np.complex128)
    for orbital, m in enumerate((-1, 0, 1)):
        orbitals[orbital, :, builder.orbital_lm.index((1, m))] = radial

    action = builder.build(orbitals).apply(orbitals)
    radial_actions = []
    for orbital, m in enumerate((-1, 0, 1)):
        channel = builder.orbital_lm.index((1, m))
        radial_actions.append(action[orbital, :, channel])
        other_channels = np.arange(len(builder.orbital_lm)) != channel
        np.testing.assert_allclose(
            action[orbital, :, other_channels], 0.0, atol=2e-13
        )
    np.testing.assert_allclose(radial_actions[1], radial_actions[0], atol=2e-13)
    np.testing.assert_allclose(radial_actions[2], radial_actions[0], atol=2e-13)


def test_pbe_time_dependent_action_matches_ground_state_weak_matrix():
    functional = PBE()
    result = SphericalAtomicKohnSham(
        "He",
        functional=functional,
        nodes=[0.0, 3.0, 6.0, 9.0],
        n_points=13,
        l_max=0,
        m_max=0,
        poisson_method="solve",
        max_iterations=100,
        conv_tol_grad=1e-9,
    ).solve()
    assert result.converged
    builder = SphericalXCBuilder(
        result.grid,
        0,
        0,
        functional=functional,
        angular_bandlimit=2,
    )
    actual = builder.build(result.orbitals).apply(result.orbitals)[0, :, 0]
    expected = result.xc_matrix @ result.radial_orbitals[0][0]
    np.testing.assert_allclose(actual, expected, atol=2e-13)


def test_pbe_weighted_adjoint_potential_is_discrete_energy_derivative(radial_grid):
    builder = SphericalXCBuilder(
        radial_grid,
        1,
        m_max=1,
        functional=PBE(),
        angular_bandlimit=4,
    )
    operator = builder._differential
    full_r = radial_grid.r[:, None, None]
    theta, phi = builder.angular_grid.mesh()
    angular = np.sin(theta) * np.cos(phi)
    full_density = np.exp(-full_r) * (1.0 + 0.04 * full_r * angular)
    density = full_density[1:-1]
    direction = (
        builder.r[:, None, None]
        * (radial_grid.r[-1] - builder.r)[:, None, None]
        * np.exp(-builder.r[:, None, None])
        * (1.0 + 0.1 * angular)
    )
    full_direction = np.zeros_like(full_density)
    full_direction[1:-1] = direction

    evaluated = evaluate_spherical_functional(
        builder.functional,
        density,
        full_density=full_density,
        differential_operator=operator,
    )
    expected = operator.integrate(direction * evaluated.potential)
    step = 1e-6
    energies = []
    for sign in (-1.0, 1.0):
        shifted = density + sign * step * direction
        shifted_full = full_density + sign * step * full_direction
        values = evaluate_spherical_functional(
            builder.functional,
            shifted,
            full_density=shifted_full,
            differential_operator=operator,
        )
        energies.append(operator.integrate(values.energy_density))
    finite_difference = (energies[1] - energies[0]) / (2.0 * step)
    assert finite_difference == pytest.approx(expected, rel=2e-8, abs=2e-10)
