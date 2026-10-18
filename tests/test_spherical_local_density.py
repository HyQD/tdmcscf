import numpy as np
import pytest

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)

from tdmcscf.dft import LDAVWN, LocalFunctionalValues
from tdmcscf.grid.spherical import SphericalXCBuilder


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
