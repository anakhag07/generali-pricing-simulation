from __future__ import annotations

import csv
import json
import numpy as np
from pathlib import Path
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
    collect_continuous_policy_lcb_outputs,
    evaluate_continuous_policy_lcb_draw,
    evaluate_continuous_policy_lcb_seed,
    load_continuous_policy_lcb_manifest,
    parse_continuous_policy_lcb_manifest,
    run_continuous_policy_lcb_manifest_seed,
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


def _manifest_payload() -> dict[str, object]:
    return {
        "kind": "continuous_policy_lcb",
        "name": "continuous-lcb-test",
        "policy_domain": [0.0, 1.0],
        "true_value": {"type": "identity"},
        "surrogate": {"type": "shared_policy_scaled_gaussian"},
        "deltas": [0.2, 0.05],
        "optimizer": {
            "step_rule": "projected_constant",
            "enabled_estimators": ["first_order", "finite_difference", "stein_difference"],
            "starts": [0.1, 0.5, 0.9],
            "t_steps": 30,
            "step_size": 0.2,
            "sigma": 0.05,
            "n_grad_samples": 32,
        },
        "seeds": {
            "master_noise_seed": 20260807,
            "master_optimizer_seed": 20260808,
            "reporting_seed": 20260809,
            "run_seeds": [101, 102],
        },
        "launch": {"mode": "local", "array": "seed"},
    }


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


def test_manifest_parser_validates_continuous_contract(tmp_path: Path) -> None:
    payload = _manifest_payload()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_continuous_policy_lcb_manifest(path)

    assert manifest.name == "continuous-lcb-test"
    assert manifest.spec == _spec()
    assert manifest.launch.array == "seed"

    payload["surrogate"] = {"type": "policy_scaled_gaussian"}
    with pytest.raises(ValueError, match="surrogate.type"):
        parse_continuous_policy_lcb_manifest(payload)


def test_committed_manifest_matches_the_documented_seed_and_optimizer_contract() -> None:
    manifest_path = (
        Path(__file__).parents[2]
        / "manifests"
        / "continuous_policy_lcb_validation.json"
    )
    manifest = load_continuous_policy_lcb_manifest(manifest_path)

    assert manifest.name == "continuous-policy-lcb-validation"
    assert manifest.spec.policy_domain == (0.0, 1.0)
    assert manifest.spec.run_seeds == tuple(range(101, 126))
    assert manifest.spec.master_noise_seed == 20260807
    assert manifest.spec.master_optimizer_seed == 20260808
    assert manifest.spec.reporting_seed == 20260809
    assert manifest.spec.optimizer.enabled_estimators == (
        "first_order",
        "finite_difference",
        "stein_difference",
    )
    assert manifest.spec.optimizer.starts == (0.1, 0.5, 0.9)
    assert manifest.spec.optimizer.t_steps == 500
    assert manifest.launch.array == "seed"


def test_seed_execution_collection_and_exact_output_tree(tmp_path: Path) -> None:
    manifest = parse_continuous_policy_lcb_manifest(_manifest_payload())
    first = run_continuous_policy_lcb_manifest_seed(manifest, 0, runs_root=tmp_path)
    skipped = run_continuous_policy_lcb_manifest_seed(manifest, 0, runs_root=tmp_path)
    run_continuous_policy_lcb_manifest_seed(manifest, 1, runs_root=tmp_path)
    collected = collect_continuous_policy_lcb_outputs(manifest, runs_root=tmp_path)

    assert first["skipped"] is False
    assert skipped["skipped"] is True
    assert collected["n_start_rows"] == 2 * 2 * 3 * 3
    assert collected["n_best_rows"] == 2 * 2 * 3
    project_dir = manifest.project_dir(tmp_path)
    expected_root_files = {
        "EXPERIMENT.md",
        "seed_draws.csv",
        "seed_start_results.csv",
        "seed_best_results.csv",
        "optimizer_summary.csv",
        "coverage_summary.csv",
        "oracle_summary.csv",
    }
    assert expected_root_files <= {path.name for path in project_dir.iterdir() if path.is_file()}
    for run_seed in manifest.spec.run_seeds:
        result_path = project_dir / "seeds" / f"seed-{run_seed}" / "result.json"
        assert result_path.exists()
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["model"]["optimized_quantity"] == "negative_lcb"
        assert payload["seed_contract"]["stein_stream_varies_across_run_seeds"] is False
        assert (project_dir / "seeds" / f"seed-{run_seed}" / "trajectories.csv").exists()
        assert (project_dir / "plots" / "seeds" / f"seed-{run_seed}.png").exists()
    for filename in (
        "final_policy_median_iqr.png",
        "final_policy_mean_ci95.png",
        "optimization_gap.png",
        "convergence_steps.png",
        "coverage.png",
        "oracle_slack.png",
    ):
        assert (project_dir / "plots" / filename).exists()
    with (project_dir / "seed_best_results.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 12
