from __future__ import annotations

import numpy as np
import pytest

from experiments.policy_lcb.continuous import (
    ContinuousPolicyLCBOptimizerSpec,
    ContinuousPolicyLCBSpec,
    continuous_analytic_policy,
    continuous_lcb_loss,
    continuous_lcb_quantile,
    continuous_lcb_slope,
    continuous_noise_seed_for_run,
    continuous_stein_seed,
    evaluate_continuous_policy_lcb_draw,
    evaluate_continuous_policy_lcb_seed,
)
from optimization.helpers import stein_difference_theta_grad


def _spec() -> ContinuousPolicyLCBSpec:
    return ContinuousPolicyLCBSpec(
        policy_domain=(0.0, 1.0),
        deltas=(0.2, 0.05),
        master_noise_seed=20260807,
        master_optimizer_seed=20260808,
        reporting_seed=20260809,
        run_seeds=(101, 102),
        optimizer=ContinuousPolicyLCBOptimizerSpec(
            step_rule="projected_constant",
            enabled_estimators=("first_order", "finite_difference", "stein_difference"),
            starts=(0.1, 0.5, 0.9),
            t_steps=30,
            step_size=0.2,
            sigma=0.05,
            n_grad_samples=32,
        ),
    )


def test_shared_gaussian_lcb_formula_and_gradient() -> None:
    delta = 0.2
    z = 0.75
    policy = 0.4
    quantile = continuous_lcb_quantile(delta)

    assert continuous_lcb_loss(policy, z, delta) == pytest.approx(
        policy * (quantile - 1.0 - z)
    )
    assert continuous_lcb_slope(z, delta) == pytest.approx(quantile - 1.0 - z)


def test_analytic_minimum_is_the_correct_endpoint() -> None:
    assert continuous_analytic_policy(2.0, 0.2) == 1.0
    assert continuous_analytic_policy(-2.0, 0.2) == 0.0


def test_pure_stein_helper_matches_linear_gradient_up_to_sample_second_moment() -> None:
    epsilon = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    slope = 3.0
    estimate = stein_difference_theta_grad(
        lambda theta: slope * float(theta[0]),
        np.asarray([0.4]),
        step=0.05,
        epsilon_samples=epsilon,
    )

    assert estimate == pytest.approx([slope * np.mean(epsilon[:, 0] ** 2)])


@pytest.mark.parametrize("z, expected_policy", [(2.0, 1.0), (-2.0, 0.0)])
def test_all_estimators_converge_to_analytic_endpoint(z: float, expected_policy: float) -> None:
    result = evaluate_continuous_policy_lcb_draw(
        _spec(),
        run_seed=101,
        noise_seed=17,
        z=z,
    )

    assert len(result.start_results) == 2 * 3 * 3
    assert len(result.best_results) == 2 * 3
    assert {row.estimator for row in result.best_results} == {
        "first_order",
        "finite_difference",
        "stein_difference",
    }
    assert all(row.final_policy == pytest.approx(expected_policy) for row in result.best_results)
    assert all(row.optimization_error == pytest.approx(0.0) for row in result.best_results)
    assert all(not row.oracle_violation for row in result.best_results)


def test_problem_draws_vary_but_stein_stream_is_fixed_and_replayable() -> None:
    spec = _spec()
    first = evaluate_continuous_policy_lcb_seed(spec, 101)
    replay = evaluate_continuous_policy_lcb_seed(spec, 101)
    second = evaluate_continuous_policy_lcb_seed(spec, 102)

    assert first == replay
    assert first.noise_seed == continuous_noise_seed_for_run(spec, 101)
    assert second.noise_seed == continuous_noise_seed_for_run(spec, 102)
    assert first.noise_seed != second.noise_seed
    assert first.z != second.z
    assert first.stein_seed == second.stein_seed == continuous_stein_seed(spec)


def test_shared_draw_and_quantile_are_paired_across_conditions() -> None:
    result = evaluate_continuous_policy_lcb_draw(
        _spec(),
        run_seed=101,
        noise_seed=17,
        z=0.25,
    )

    assert {row.z for row in result.start_results} == {0.25}
    for delta in _spec().deltas:
        rows = [row for row in result.start_results if row.delta == delta]
        assert {row.quantile for row in rows} == {continuous_lcb_quantile(delta)}
        assert {row.stein_seed for row in rows} == {result.stein_seed}


def test_spec_rejects_out_of_domain_starts() -> None:
    optimizer = ContinuousPolicyLCBOptimizerSpec(
        step_rule="projected_constant",
        enabled_estimators=("first_order",),
        starts=(-0.1,),
        t_steps=5,
        step_size=0.1,
        sigma=0.05,
        n_grad_samples=4,
    )
    with pytest.raises(ValueError, match="starts"):
        ContinuousPolicyLCBSpec(
            policy_domain=(0.0, 1.0),
            deltas=(0.2,),
            master_noise_seed=1,
            master_optimizer_seed=2,
            reporting_seed=3,
            run_seeds=(4,),
            optimizer=optimizer,
        )

