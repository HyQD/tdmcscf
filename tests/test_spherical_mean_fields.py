import numpy as np
import pytest
from scipy.special import gamma, gammainc, gammaincc

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates import GaussLegendreFourierGrid
from grid_lib.spherical_coordinates.angular_momentum import (
    LM_to_I,
    get_y,
    get_ybar,
)
from grid_lib.spherical_coordinates.radial_mean_fields import compute_mean_field_einsum

from tdmcscf.grid.spherical import SphericalMeanFieldBuilder


@pytest.fixture(scope="module")
def radial_grid():
    return FEMDVR(
        np.array([0.0, 2.0, 5.0, 10.0, 20.0, 40.0]),
        np.full(5, 21),
        Linear_map,
        GaussLegendreLobatto,
    )


@pytest.fixture(scope="module")
def builder(radial_grid):
    return SphericalMeanFieldBuilder(radial_grid, l_max=2)


def random_orbitals(builder, seed=91, real=False):
    rng = np.random.default_rng(seed)
    shape = (3,) + builder.orbital_shape
    values = rng.normal(size=shape)
    if not real:
        values = values + 1j * rng.normal(size=shape)
    values *= (builder.r * np.exp(-builder.r))[None, :, None]
    sqrt_weights = np.repeat(np.sqrt(builder.radial_weights), shape[-1])
    weighted = values.reshape(3, -1).T * sqrt_weights[:, None]
    orthogonal, _ = np.linalg.qr(weighted)
    return (orthogonal / sqrt_weights[:, None]).T.reshape(shape)


def fixed_m_orbitals(builder):
    values = random_orbitals(builder)
    magnetic = np.array([-1, 0, 1])
    for orbital, m in zip(values, magnetic):
        orbital[:, [mi != m for _, mi in builder.orbital_lm]] = 0
        orbital /= np.sqrt(np.sum(builder.radial_weights[:, None] * abs(orbital) ** 2))
    return values, magnetic


def gaunt_fields(builder, orbitals):
    ybar = get_ybar(builder.l_max, builder.m_max, 2 * builder.l_max, 2 * builder.m_max)
    return compute_mean_field_einsum(
        orbitals, builder.radial_matrices, ybar, 2 * builder.l_max, 2 * builder.m_max
    )


@pytest.mark.parametrize("real", [False, True])
def test_general_fields_match_grid_lib_gaunt_reference(builder, real):
    orbitals = random_orbitals(builder, real=real)
    fields = builder.build(orbitals)
    reference = gaunt_fields(builder, orbitals)
    assert fields.pairs == ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    for ip, (q, s) in enumerate(fields.pairs):
        np.testing.assert_allclose(
            fields.coefficients[ip], reference[:, q, s], atol=2e-13
        )
        np.testing.assert_allclose(
            fields.multipoles((s, q)), reference[:, s, q], atol=2e-13
        )


def test_channel_order_and_restricted_m_match_grid_lib(radial_grid):
    builder = SphericalMeanFieldBuilder(radial_grid, l_max=2, m_max=1)
    for i, (l, m) in enumerate(builder.orbital_lm):
        assert LM_to_I(l, m, 2, 1) == i
    for i, (L, M) in enumerate(builder.multipole_lm):
        assert LM_to_I(L, M, 4, 2) == i
    orbitals = random_orbitals(builder)
    pairs = [(2, 0), (1, 1), (0, 2)]
    fields = builder.build(orbitals, pairs=pairs)
    assert fields.pairs == tuple(pairs)
    reference = gaunt_fields(builder, orbitals)
    np.testing.assert_allclose(
        fields.coefficients, [reference[:, q, s] for q, s in pairs], atol=2e-13
    )
    with pytest.raises(KeyError, match="Neither pair"):
        fields.multipoles((0, 1))


@pytest.mark.parametrize("pair", [(0, 1), (1, 0), (2, 2)])
def test_general_action_matches_independent_gaunt_contraction(builder, pair):
    source = random_orbitals(builder)
    target = random_orbitals(builder, seed=207)
    fields = builder.build(source)
    y = get_y(builder.l_max, builder.m_max, 2 * builder.l_max, 2 * builder.m_max)
    q, s = pair
    reference_field = gaunt_fields(builder, source)[:, q, s]
    expected = np.einsum("kr,kij,prj->pri", reference_field, y, target, optimize=True)
    np.testing.assert_allclose(fields.apply(target, pair), expected, atol=2e-13)
    np.testing.assert_allclose(fields.apply(target[0], pair), expected[0], atol=2e-13)


def test_fixed_m_construction_and_action_use_polar_only_path(builder, monkeypatch):
    orbitals, magnetic = fixed_m_orbitals(builder)
    general = builder.build(orbitals)
    expected_actions = {
        pair: general.apply(orbitals, pair) for pair in [(0, 2), (2, 0), (1, 1)]
    }

    def no_azimuthal_transform(*args, **kwargs):
        raise AssertionError("The fixed-m path must not sample the azimuthal grid.")

    monkeypatch.setattr(
        builder._orbital_transform, "synthesize", no_azimuthal_transform
    )
    monkeypatch.setattr(builder._orbital_transform, "analyze", no_azimuthal_transform)
    monkeypatch.setattr(builder._field_transform, "synthesize", no_azimuthal_transform)
    monkeypatch.setattr(builder._field_transform, "analyze", no_azimuthal_transform)
    fixed = builder.build(orbitals, orbital_m=magnetic)
    np.testing.assert_allclose(fixed.coefficients, general.coefficients, atol=2e-13)
    for pair, expected in expected_actions.items():
        actual = fixed.apply(orbitals, pair, orbital_m=magnetic)
        np.testing.assert_allclose(actual, expected, atol=2e-13)
    # m_in=+1 and M=+2 lies outside the retained orbital channels.
    np.testing.assert_array_equal(fixed.apply(orbitals[2], (0, 2), orbital_m=1), 0)


def test_complex_pair_reversal_gives_adjoint_action_in_radial_metric(builder):
    source = random_orbitals(builder)
    target = random_orbitals(builder, seed=77)
    fields = builder.build(source, pairs=[(0, 1)])
    a, b = target[:2]
    w = builder.radial_weights[:, None]
    left = np.sum(w * a.conj() * fields.apply(b, (0, 1)))
    right = np.sum(w * fields.apply(a, (1, 0)).conj() * b)
    np.testing.assert_allclose(left, right, atol=2e-13)


def test_analytic_hydrogen_1s_field_and_action_in_nodal_convention(radial_grid):
    builder = SphericalMeanFieldBuilder(radial_grid, l_max=0)
    r = builder.r
    u = 2 * r * np.exp(-r)
    np.testing.assert_allclose(np.dot(builder.radial_weights, u**2), 1, atol=1e-12)
    fields = builder.build(u[None, :, None], orbital_m=0)
    potential = (1 - (1 + r) * np.exp(-2 * r)) / r
    np.testing.assert_allclose(
        fields.multipoles((0, 0))[0],
        np.sqrt(4 * np.pi) * potential,
        atol=2e-11,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        fields.apply(u[:, None], (0, 0), orbital_m=0)[:, 0],
        potential * u,
        atol=2e-11,
        rtol=1e-10,
    )


def test_analytic_complex_dipole_field_checks_m_sign_and_radial_normalization(builder):
    r = builder.r
    orbitals = np.zeros((2,) + builder.orbital_shape, dtype=complex)
    orbitals[0, :, builder.orbital_lm.index((0, 0))] = 2 * r * np.exp(-r)
    orbitals[1, :, builder.orbital_lm.index((1, 1))] = (
        1j * 2 / np.sqrt(3) * r**2 * np.exp(-r)
    )
    fields = builder.build(orbitals, pairs=[(0, 1)])
    # d_01^(1,1)(r) = i * (4/sqrt(12*pi)) * r^3 exp(-2r).
    lower = gamma(5) * gammainc(5, 2 * r) / 2**5
    upper = gamma(2) * gammaincc(2, 2 * r) / 2**2
    expected = (
        1j * 4 / np.sqrt(12 * np.pi) * (4 * np.pi / 3) * (lower / r**2 + r * upper)
    )
    index = builder.multipole_lm.index((1, 1))
    np.testing.assert_allclose(
        fields.coefficients[0, index], expected, atol=3e-11, rtol=1e-9
    )
    other = [k for k in range(len(builder.multipole_lm)) if k != index]
    np.testing.assert_allclose(fields.coefficients[0, other], 0, atol=2e-13)


def test_radial_response_is_reused_and_inputs_are_unchanged(builder, monkeypatch):
    import tdmcscf.grid.spherical.mean_fields as module

    orbitals = random_orbitals(builder)
    original = orbitals.copy()

    def no_rebuild(*args, **kwargs):
        raise AssertionError(
            "Radial matrices must not be rebuilt with each orbital update."
        )

    monkeypatch.setattr(module, "radial_Coulomb", no_rebuild)
    first = builder.build(orbitals, pairs=[(0, 1)])
    second = builder.build(2 * orbitals, pairs=[(0, 1)])
    np.testing.assert_allclose(second.coefficients, 4 * first.coefficients, atol=2e-13)
    first.apply(orbitals, (0, 1))
    np.testing.assert_array_equal(orbitals, original)


@pytest.mark.parametrize(
    "pairs", [[(0, 3)], [(-1, 0)], [(True, 0)], [(0, 1), (0, 1)], [(0,)]]
)
def test_rejects_invalid_pair_requests(builder, pairs):
    with pytest.raises((ValueError, TypeError)):
        builder.build(random_orbitals(builder), pairs=pairs)


def test_rejects_wrong_orbital_shape_and_inconsistent_fixed_m_metadata(builder):
    orbitals = random_orbitals(builder)
    with pytest.raises(ValueError, match="shape"):
        builder.build(orbitals[:, :-1])
    with pytest.raises(ValueError, match="finite"):
        builder.build(orbitals * np.nan)
    with pytest.raises(ValueError, match="zero outside"):
        builder.build(orbitals, orbital_m=[-1, 0, 1])
    with pytest.raises(ValueError, match="one integer"):
        builder.build(orbitals, orbital_m=[-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="outside the retained"):
        builder.build(orbitals, orbital_m=3)


@pytest.mark.parametrize(
    "grid", [GaussLegendreFourierGrid(4, 9), GaussLegendreFourierGrid(5, 8)]
)
def test_rejects_angular_aliasing(radial_grid, grid):
    with pytest.raises(ValueError, match="too small"):
        SphericalMeanFieldBuilder(radial_grid, l_max=2, angular_grid=grid)


def test_empty_pair_list_is_supported(builder):
    fields = builder.build(random_orbitals(builder), pairs=[])
    assert fields.pairs == ()
    assert fields.coefficients.shape == (0, len(builder.multipole_lm), len(builder.r))
