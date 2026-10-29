import numpy as np
import pytest
from scipy.linalg import eigh

from grid_lib.pseudospectral_grids import (
    FEMDVR,
    GaussLegendreLobatto,
    Linear_map,
)

from tdmcscf.grid.spherical import (
    AxialCoreHamiltonian,
    SphericalMeanFieldBuilder,
    SphericalRestrictedTDHF,
)
from tdmcscf.systems import AxialDiatomic


def _grid(points=6):
    return FEMDVR(
        np.asarray([0.0, 1.0, 3.0, 6.0]),
        np.asarray([points, points, points]),
        Linear_map,
        GaussLegendreLobatto,
        symmetrize=False,
    )


def _core(
    *,
    points=6,
    l_max=2,
    m_max=1,
    nuclear_L_max=None,
    nuclear_method="analytic",
    system=None,
):
    if system is None:
        system = AxialDiatomic(
            "H",
            "H",
            half_bond_length=1.0,
            charge=1,
            label="H2+",
        )
    return AxialCoreHamiltonian(
        _grid(points),
        l_max,
        m_max,
        system=system,
        nuclear_L_max=nuclear_L_max,
        nuclear_method=nuclear_method,
    )


@pytest.mark.parametrize("m", [-1, 0, 1])
def test_fixed_m_action_matches_explicit_strong_matrix(m):
    core = _core()
    rng = np.random.default_rng(3 + m)
    shape = core.fixed_m_shape(m)
    orbitals = rng.normal(size=(2, *shape))
    orbitals = orbitals + 1j * rng.normal(size=(2, *shape))
    expected = np.asarray(
        [
            (core.fixed_m_matrix(m) @ orbital.ravel()).reshape(shape)
            for orbital in orbitals
        ]
    )

    np.testing.assert_allclose(
        core.apply_fixed_m(orbitals, m),
        expected,
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_full_action_preserves_m_and_matches_fixed_m_action():
    core = _core()
    m = 1
    fixed = np.arange(np.prod(core.fixed_m_shape(m)), dtype=float)
    fixed = fixed.reshape(core.fixed_m_shape(m))
    full = np.zeros(core.orbital_shape)
    indices = [
        index for index, (_, magnetic) in enumerate(core.orbital_lm) if magnetic == m
    ]
    full[:, indices] = fixed

    action = core.apply(full)

    np.testing.assert_allclose(action[:, indices], core.apply_fixed_m(fixed, m))
    complement = np.ones(action.shape[-1], dtype=bool)
    complement[indices] = False
    np.testing.assert_allclose(action[:, complement], 0.0, atol=1.0e-14)


@pytest.mark.parametrize("m", [-1, 0, 1])
def test_fixed_m_matrix_is_hermitian_in_femdvr_metric(m):
    core = _core()
    strong = core.fixed_m_matrix(m)
    weights = core.fixed_m_metric_weights(m)
    symmetric = core.fixed_m_matrix(m, representation="symmetric")

    np.testing.assert_allclose(
        strong.conj().T * weights[None, :],
        weights[:, None] * strong,
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        symmetric,
        symmetric.conj().T,
        atol=3.0e-13,
        rtol=3.0e-13,
    )


def test_metric_transforms_round_trip_for_batched_orbitals():
    core = _core()
    rng = np.random.default_rng(9)
    nodal = rng.normal(size=(3, *core.fixed_m_shape(0)))

    symmetric = core.to_symmetric_fixed_m(nodal, 0)

    np.testing.assert_allclose(core.from_symmetric_fixed_m(symmetric, 0), nodal)
    np.testing.assert_allclose(
        np.sum(np.abs(symmetric) ** 2),
        core.inner_product(nodal, nodal),
    )


def test_axial_blocks_are_degenerate_in_sign_of_m():
    core = _core(l_max=3, m_max=2)

    np.testing.assert_allclose(
        core.fixed_m_matrix(-2),
        core.fixed_m_matrix(2),
        atol=2.0e-14,
        rtol=2.0e-14,
    )


def test_multipoles_above_twice_lmax_do_not_change_matrix():
    complete = _core(l_max=2, m_max=0, nuclear_L_max=4)
    extended = _core(l_max=2, m_max=0, nuclear_L_max=7)

    np.testing.assert_allclose(
        complete.fixed_m_matrix(0),
        extended.fixed_m_matrix(0),
        atol=2.0e-14,
        rtol=2.0e-14,
    )


def test_poisson_energy_converges_to_analytic_energy_with_radial_order():
    errors = []
    for points in (6, 8):
        analytic = _core(
            points=points,
            m_max=0,
            nuclear_method="analytic",
        )
        poisson = _core(
            points=points,
            m_max=0,
            nuclear_method="poisson",
        )
        analytic_energy = eigh(
            analytic.fixed_m_matrix(0, representation="symmetric"),
            eigvals_only=True,
            subset_by_index=(0, 0),
        )[0]
        poisson_energy = eigh(
            poisson.fixed_m_matrix(0, representation="symmetric"),
            eigvals_only=True,
            subset_by_index=(0, 0),
        )[0]
        errors.append(abs(analytic_energy - poisson_energy))

    assert errors[1] < errors[0]
    assert errors[1] < 1.0e-6


def test_homonuclear_symmetry_and_heteronuclear_odd_multipoles():
    homonuclear = _core(l_max=3, m_max=0)
    heteronuclear = _core(
        l_max=3,
        m_max=0,
        system=AxialDiatomic(
            "H",
            "He",
            half_bond_length=1.0,
            charge=1,
        ),
    )

    np.testing.assert_allclose(homonuclear.potential.coefficients[1::2], 0.0)
    assert np.max(np.abs(heteronuclear.potential.coefficients[1::2])) > 0.0


def test_nucleus_must_be_at_internal_element_boundary():
    system = AxialDiatomic("H", "H", half_bond_length=1.2, charge=1)
    with pytest.raises(ValueError, match="internal FEM element boundary"):
        AxialCoreHamiltonian(_grid(), 2, system=system)


def test_h2_plus_coarse_grid_reference_energy():
    core = _core(points=8, l_max=4, m_max=0)
    electronic_energy = eigh(
        core.fixed_m_matrix(0, representation="symmetric"),
        eigvals_only=True,
        subset_by_index=(0, 0),
    )[0]

    assert electronic_energy == pytest.approx(-1.096374461122, abs=2.0e-10)
    assert electronic_energy + core.nuclear_repulsion_energy == pytest.approx(
        -0.596374461122,
        abs=2.0e-10,
    )


def test_axial_core_satisfies_low_level_tdhf_core_protocol():
    core = _core(l_max=2, m_max=0)
    mean_fields = SphericalMeanFieldBuilder(
        core.radial_grid,
        core.l_max,
        core.m_max,
        L_max=4,
        M_max=0,
    )
    equation = SphericalRestrictedTDHF(core, mean_fields, orbital_m=[0])
    orbital = np.zeros((1, *core.orbital_shape), dtype=complex)
    orbital[0, :, core.orbital_lm.index((0, 0))] = core.r * np.exp(-core.r)
    orbital /= np.sqrt(equation.orbital_norms(orbital))[0]

    np.testing.assert_allclose(
        equation.one_body_action(orbital, 0.0),
        core.apply(orbital),
    )
