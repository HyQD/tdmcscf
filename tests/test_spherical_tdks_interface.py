import numpy as np
import pytest

from tdmcscf.dft import LDAVWN, XAlphaExchange
from tdmcscf.grid.spherical import (
    RadialCosineMask,
    SphericalKohnSham,
    SphericalTDKS,
    TDKSCheckpoint,
)
from tdmcscf.systems import HarmonicOscillator


class ConstantField:
    gauge = "length"
    time_support = (0.0, 1.0)

    def __init__(self, value):
        self.value = value

    def __call__(self, time):
        return self.value


@pytest.fixture(scope="module")
def calculation():
    system = HarmonicOscillator(0.5, occupation={0: 1})
    return SphericalTDKS(
        system,
        functional=LDAVWN(),
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
        orbital_gauge="canonical",
        integrator="strang2",
        integrator_options={"solver": "gmres", "rtol": 1e-11},
        time_step=0.01,
        final_time=0.02,
        poisson_method="solve",
    )


def test_ground_state_setup_and_tdkohn_sham_propagation(calculation):
    assert calculation.ground_state.converged
    assert calculation.ground_state.functional.name == "LDA-VWN5"
    assert calculation.xc_angular_bandlimit == 4
    assert calculation.m_conserved
    assert calculation.equation.orbital_m == (0,)

    static_local_potential = (
        calculation.ground_state.hartree_potential
        + calculation.ground_state.xc_potential
    )
    expected_mean_field = (
        calculation.initial_orbitals
        * static_local_potential[None, :, None]
    )
    np.testing.assert_allclose(
        calculation.equation.mean_field_action(calculation.initial_orbitals),
        expected_mean_field,
        atol=3e-12,
    )

    result = calculation.run()
    np.testing.assert_allclose(result.times, [0.0, 0.01, 0.02])
    np.testing.assert_allclose(result.orbital_norms, 1.0, atol=5e-10)
    assert result.position_expectations.shape == (3, 1, 3)
    assert result.total_dipole.shape == (3, 3)
    assert result.functional is calculation.functional
    assert result.xc_angular_bandlimit == 4


def test_kohn_sham_result_can_be_embedded_in_larger_angular_space(calculation):
    ground_state = SphericalKohnSham(
        calculation.system,
        functional=LDAVWN(),
        nodes=calculation.grid.nodes,
        n_points=calculation.grid.n_points,
        l_max=0,
        m_max=0,
        max_iterations=30,
    ).solve()
    propagated = SphericalTDKS(
        calculation.system,
        functional=LDAVWN(),
        nodes=calculation.grid.nodes,
        n_points=calculation.grid.n_points,
        l_max=1,
        m_max=1,
        initial_state=ground_state,
        time_step=0.01,
        final_time=0.01,
    )
    assert propagated.initial_orbitals.shape[-1] == 4
    s_channel = propagated.core.orbital_lm.index((0, 0))
    assert np.all(
        propagated.initial_orbitals[
            :, :, np.arange(len(propagated.core.orbital_lm)) != s_channel
        ]
        == 0
    )


@pytest.mark.parametrize("integrator", ["im-cmf1", "im-cdm1"])
def test_density_frozen_integrators(calculation, integrator):
    propagated = SphericalTDKS(
        calculation.system,
        functional=LDAVWN(),
        nodes=calculation.grid.nodes,
        n_points=calculation.grid.n_points,
        l_max=1,
        m_max=1,
        initial_state=calculation.initial_orbitals,
        integrator=integrator,
        integrator_options={"solver": "gmres", "rtol": 1e-11},
        time_step=0.01,
        final_time=0.01,
    )
    result = propagated.run()
    np.testing.assert_allclose(result.orbital_norms, 1.0, atol=5e-10)


def test_tdkohn_sham_checkpoint_and_functional_compatibility(
    tmp_path, calculation
):
    path = tmp_path / "tdks_checkpoint.npz"
    propagated = SphericalTDKS(
        calculation.system,
        functional=LDAVWN(),
        nodes=calculation.grid.nodes,
        n_points=calculation.grid.n_points,
        l_max=1,
        m_max=1,
        initial_state=calculation.initial_orbitals,
        time_step=0.01,
        final_time=0.02,
        mask=RadialCosineMask(0.5),
        checkpoint_path=path,
        checkpoint_interval=1,
    )
    result = propagated.run()
    checkpoint = TDKSCheckpoint.load(path)
    resumed = propagated.run(checkpoint=checkpoint)
    np.testing.assert_allclose(resumed.orbitals, result.orbitals)

    incompatible = SphericalTDKS(
        calculation.system,
        functional=XAlphaExchange(alpha=0.7),
        nodes=calculation.grid.nodes,
        n_points=calculation.grid.n_points,
        l_max=1,
        m_max=1,
        initial_state=calculation.initial_orbitals,
        time_step=0.01,
        final_time=0.02,
    )
    with pytest.raises(ValueError, match="different XC functional"):
        incompatible.run(checkpoint=checkpoint)


def test_rejects_insufficient_xc_angular_bandlimit(calculation):
    with pytest.raises(ValueError, match=r"2\*l_max"):
        SphericalTDKS(
            calculation.system,
            functional=LDAVWN(),
            nodes=calculation.grid.nodes,
            n_points=calculation.grid.n_points,
            l_max=2,
            m_max=1,
            xc_angular_bandlimit=3,
            initial_state=calculation.initial_orbitals,
            time_step=0.01,
            final_time=0.02,
        )
