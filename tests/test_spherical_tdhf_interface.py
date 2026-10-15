import numpy as np
import pytest

from tdmcscf.grid.spherical import (
    RadialCosineMask,
    SphericalHartreeFock,
    SphericalTDHF,
    TDHFCheckpoint,
    estimate_rk4_time_step,
)
from tdmcscf.systems import HarmonicOscillator


class ConstantField:
    def __init__(self, value, gauge="length"):
        self.value = value
        self.gauge = gauge
        self.time_support = (0.0, 1.0)

    def __call__(self, time):
        return self.value


@pytest.fixture(scope="module")
def ground_state_setup():
    system = HarmonicOscillator(0.5, occupation={0: 1})
    calculation = SphericalTDHF(
        system,
        nodes=[0.0, 2.0, 6.0],
        n_points=9,
        l_max=1,
        m_max=1,
        L_max=2,
        M_max=2,
        initial_state="groundstate",
        ground_state_options={
            "max_iterations": 30,
            "conv_tol_grad": 1e-8,
        },
        fields=[(ConstantField(1e-4), (0.0, 0.0, 2.0))],
        field_gauge="length",
        integrator="strang2",
        time_step=0.01,
        final_time=0.02,
        poisson_method="solve",
        samplers={
            "maximum_amplitude": lambda simulation, orbitals, time: np.max(
                np.abs(orbitals)
            )
        },
    )
    return calculation


def test_ground_state_setup_fields_sampling_and_propagation(ground_state_setup):
    calculation = ground_state_setup
    assert calculation.ground_state.converged
    assert calculation.mean_fields.poisson_method == "solve"
    assert calculation.field_terms[0].polarization == (0.0, 0.0, 1.0)
    assert calculation.m_conserved
    assert calculation.equation.orbital_m == (0,)

    result = calculation.run()
    assert result.orbitals.shape == (1, *calculation.core.orbital_shape)
    np.testing.assert_allclose(result.times, [0.0, 0.01, 0.02])
    np.testing.assert_allclose(result.orbital_norms, 1.0, atol=3e-11)
    assert result.position_expectations.shape == (3, 1, 3)
    assert result.total_dipole.shape == (3, 3)
    assert result.samples["maximum_amplitude"].shape == (3,)


def test_rk4_uses_varying_mean_fields_and_warns(ground_state_setup):
    with pytest.warns(RuntimeWarning, match="kinetic-spectrum estimate"):
        calculation = SphericalTDHF(
            ground_state_setup.system,
            nodes=ground_state_setup.grid.nodes,
            n_points=ground_state_setup.grid.n_points,
            l_max=1,
            m_max=1,
            initial_state=ground_state_setup.initial_orbitals,
            fields=[(ConstantField(1e-4), (1.0, 0.0, 0.0))],
            field_gauge="length",
            integrator="rk4",
            time_step=1e-4,
            final_time=2e-4,
        )
    assert not calculation.m_conserved
    assert calculation.equation.orbital_m is None
    assert estimate_rk4_time_step(calculation.core.kinetic) > 0
    result = calculation.run()
    assert np.all(np.isfinite(result.orbitals))


def test_hartree_fock_result_is_embedded_in_larger_angular_space(
    ground_state_setup,
):
    solver = SphericalHartreeFock(
        ground_state_setup.system,
        nodes=ground_state_setup.grid.nodes,
        n_points=ground_state_setup.grid.n_points,
        l_max=0,
        m_max=0,
        max_iterations=30,
        conv_tol_grad=1e-8,
    )
    ground_state = solver.solve()
    calculation = SphericalTDHF(
        ground_state_setup.system,
        nodes=ground_state_setup.grid.nodes,
        n_points=ground_state_setup.grid.n_points,
        l_max=1,
        m_max=1,
        initial_state=ground_state,
        time_step=0.01,
        final_time=0.01,
    )
    assert calculation.initial_orbitals.shape[-1] == 4
    s_channel = calculation.core.orbital_lm.index((0, 0))
    other_channels = np.arange(4) != s_channel
    assert np.all(calculation.initial_orbitals[:, :, other_channels] == 0)


def test_rejects_field_gauge_mismatch_and_nonorthonormal_state(
    ground_state_setup,
):
    common = dict(
        system=ground_state_setup.system,
        nodes=ground_state_setup.grid.nodes,
        n_points=ground_state_setup.grid.n_points,
        l_max=1,
        m_max=1,
        initial_state=ground_state_setup.initial_orbitals,
        time_step=0.01,
        final_time=0.02,
    )
    with pytest.raises(ValueError, match="uses gauge"):
        SphericalTDHF(
            **common,
            fields=ConstantField(1e-4, gauge="velocity"),
            field_gauge="length",
        )
    with pytest.raises(ValueError, match="orthonormal"):
        SphericalTDHF(
            **{**common, "initial_state": 2 * ground_state_setup.initial_orbitals}
        )


def test_mask_checkpoint_and_resume(tmp_path, ground_state_setup):
    checkpoint_path = tmp_path / "tdhf_checkpoint.npz"
    calculation = SphericalTDHF(
        ground_state_setup.system,
        nodes=ground_state_setup.grid.nodes,
        n_points=ground_state_setup.grid.n_points,
        l_max=1,
        m_max=1,
        initial_state=ground_state_setup.initial_orbitals,
        time_step=0.01,
        final_time=0.02,
        mask=RadialCosineMask(r0=0.5),
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
    )
    result = calculation.run()
    assert checkpoint_path.exists()
    assert result.orbital_norms[-1, 0] < result.orbital_norms[0, 0]

    checkpoint = TDHFCheckpoint.load(checkpoint_path)
    assert checkpoint.step == calculation.n_steps
    np.testing.assert_allclose(checkpoint.orbitals, result.orbitals)
    np.testing.assert_allclose(
        checkpoint.samples["orbital_norms"], result.orbital_norms
    )

    resumed = calculation.run(checkpoint=checkpoint)
    np.testing.assert_allclose(resumed.orbitals, result.orbitals)
    np.testing.assert_allclose(resumed.times, result.times)


@pytest.mark.parametrize(
    "mask",
    [RadialCosineMask(7.0), np.array([1.0, 0.5])],
)
def test_rejects_masks_incompatible_with_the_grid(mask, ground_state_setup):
    with pytest.raises(ValueError, match="mask|r0"):
        SphericalTDHF(
            ground_state_setup.system,
            nodes=ground_state_setup.grid.nodes,
            n_points=ground_state_setup.grid.n_points,
            l_max=1,
            m_max=1,
            initial_state=ground_state_setup.initial_orbitals,
            time_step=0.01,
            final_time=0.02,
            mask=mask,
        )


@pytest.mark.parametrize(
    ("integrator_name", "options", "predictor_steps"),
    [
        ("cn-cmf1", {"solver": "gmres", "rtol": 1e-11}, None),
        (
            "cn-cmf2",
            {"solver": "gmres", "rtol": 1e-11, "predictor_steps": 1},
            1,
        ),
        (
            "cn-cmf2",
            {"solver": "gmres", "rtol": 1e-11},
            2,
        ),
    ],
)
def test_cncmf_propagation_options(
    ground_state_setup, integrator_name, options, predictor_steps
):
    calculation = SphericalTDHF(
        ground_state_setup.system,
        nodes=ground_state_setup.grid.nodes,
        n_points=ground_state_setup.grid.n_points,
        l_max=1,
        m_max=1,
        initial_state=ground_state_setup.initial_orbitals,
        fields=[(ConstantField(1e-4), (0.0, 0.0, 1.0))],
        field_gauge="length",
        orbital_gauge="orthogonal",
        integrator=integrator_name,
        integrator_options=options,
        time_step=0.01,
        final_time=0.01,
    )
    result = calculation.run()
    assert calculation.predictor_steps == predictor_steps
    assert np.all(np.isfinite(result.orbitals))
    np.testing.assert_allclose(result.orbital_norms, 1.0, atol=5e-10)
    if predictor_steps is not None:
        assert len(calculation.integrator.last_predictor_iterations) == predictor_steps
        assert len(calculation.integrator.last_corrector_iterations) == 2


@pytest.mark.parametrize("predictor_steps", [0, 3])
def test_cncmf2_rejects_invalid_predictor_steps(
    ground_state_setup, predictor_steps
):
    with pytest.raises(ValueError, match="predictor_steps"):
        SphericalTDHF(
            ground_state_setup.system,
            nodes=ground_state_setup.grid.nodes,
            n_points=ground_state_setup.grid.n_points,
            l_max=1,
            m_max=1,
            initial_state=ground_state_setup.initial_orbitals,
            integrator="cn-cmf2",
            integrator_options={"predictor_steps": predictor_steps},
            time_step=0.01,
            final_time=0.01,
        )


@pytest.mark.parametrize("orbital_gauge", ["canonical", "orthogonal"])
@pytest.mark.parametrize(
    ("integrator_name", "predictor_steps"),
    [("im-cdm1", None), ("im-cdm2", 1)],
)
def test_imcdm_propagation_options(
    ground_state_setup,
    orbital_gauge,
    integrator_name,
    predictor_steps,
):
    calculation = SphericalTDHF(
        ground_state_setup.system,
        nodes=ground_state_setup.grid.nodes,
        n_points=ground_state_setup.grid.n_points,
        l_max=1,
        m_max=1,
        initial_state=ground_state_setup.initial_orbitals,
        fields=[(ConstantField(1e-4), (0.0, 0.0, 1.0))],
        field_gauge="length",
        orbital_gauge=orbital_gauge,
        integrator=integrator_name,
        integrator_options={"solver": "gmres", "rtol": 1e-11},
        time_step=0.01,
        final_time=0.01,
    )
    result = calculation.run()
    assert calculation.predictor_steps == predictor_steps
    assert np.all(np.isfinite(result.orbitals))
    np.testing.assert_allclose(result.orbital_norms, 1.0, atol=5e-10)
    if predictor_steps is not None:
        assert calculation.integrator.predictor_steps == 1
        assert len(calculation.integrator.last_predictor_iterations) == 1
        assert len(calculation.integrator.last_corrector_iterations) == 2


@pytest.mark.parametrize("predictor_steps", [0, 2])
def test_imcdm2_rejects_nonstandard_predictor_steps(
    ground_state_setup, predictor_steps
):
    with pytest.raises(ValueError, match="predictor_steps"):
        SphericalTDHF(
            ground_state_setup.system,
            nodes=ground_state_setup.grid.nodes,
            n_points=ground_state_setup.grid.n_points,
            l_max=1,
            m_max=1,
            initial_state=ground_state_setup.initial_orbitals,
            integrator="im-cdm2",
            integrator_options={"predictor_steps": predictor_steps},
            time_step=0.01,
            final_time=0.01,
        )
