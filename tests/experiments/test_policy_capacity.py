from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.policy_capacity import (
    build_policy_capacity_launch_plan,
    initial_theta,
    load_policy_capacity_manifest,
    policy_capacity_tasks,
    split_positions,
    summarize_policy_capacity,
    _with_acceptance_penalty_metrics,
)
from objective.policy import AdditiveChebyshevFeatureMap, SoftmaxPolicy
from reporting.visualization import (
    plot_policy_capacity_action_diagnostics,
    plot_policy_capacity_baseline_adjusted_gains,
    plot_policy_capacity_endpoint_slices,
    plot_policy_capacity_generalization_gap,
    plot_policy_capacity_model_transfer,
    plot_policy_capacity_objective,
    plot_policy_capacity_penalized_gains,
)


MANIFEST = Path(__file__).parents[2] / "manifests" / "policy_capacity_glm_xgb.json"
RESTRICTED_MANIFEST = (
    Path(__file__).parents[2] / "manifests" / "policy_capacity_glm_xgb_u_0_0p16.json"
)
XGB_EXTENSION_MANIFEST = (
    Path(__file__).parents[2]
    / "manifests"
    / "policy_capacity_xgb_u_0_0p16_degree_32.json"
)
XGB_FULL_POLYNOMIAL_MANIFEST = (
    Path(__file__).parents[2]
    / "manifests"
    / "policy_capacity_xgb_u_0_0p16_full_polynomial_degree_3.json"
)


def test_canonical_policy_capacity_manifest_has_gradual_parameter_ladder() -> None:
    manifest = load_policy_capacity_manifest(MANIFEST)

    assert manifest.degrees == (0, 1, 2, 3, 4, 5, 6, 8, 10)
    assert [manifest.parameter_count(degree) for degree in manifest.degrees] == [
        1,
        20,
        39,
        58,
        77,
        96,
        115,
        153,
        191,
    ]
    assert manifest.action_bounds == (-0.1, 0.2)
    np.testing.assert_allclose(manifest.curve_action_grid, np.linspace(-0.1, 0.2, 31), atol=1e-15)
    tasks = policy_capacity_tasks(manifest)
    plan = build_policy_capacity_launch_plan(manifest)
    assert len(tasks) == plan.task_count == 360
    assert (tasks[0].split_seed, tasks[0].optimize_model, tasks[0].degree) == (0, "glm", 0)
    assert (tasks[-1].split_seed, tasks[-1].optimize_model, tasks[-1].degree) == (
        19,
        "xgb",
        10,
    )
    assert plan.slurm_profile is not None
    assert plan.slurm_profile.cpus_per_task == 2
    assert plan.slurm_profile.memory == "16G"
    assert plan.slurm_profile.time == "02:00:00"


def test_initial_theta_represents_zero_action_for_every_customer() -> None:
    manifest = load_policy_capacity_manifest(MANIFEST)
    policy = SoftmaxPolicy(
        feature_map=AdditiveChebyshevFeatureMap(max_degree=6, clip_scale=3.0),
        action_low=-0.1,
        action_high=0.2,
    )
    actions = policy.value(initial_theta(manifest, 6), np.ones((4, 19)))

    np.testing.assert_allclose(actions, 0.0, atol=1e-15)
    np.testing.assert_allclose(initial_theta(manifest, 6)[0], np.log(0.5), atol=1e-15)


def test_restricted_policy_capacity_manifest_uses_canonical_action_range() -> None:
    manifest = load_policy_capacity_manifest(RESTRICTED_MANIFEST)

    assert manifest.name == "policy-capacity-glm-xgb-u-0-0p16"
    assert manifest.action_bounds == (0.0, 0.16)
    assert manifest.initial_u == 0.08
    np.testing.assert_allclose(manifest.curve_action_grid, np.linspace(0.0, 0.16, 17))
    np.testing.assert_allclose(initial_theta(manifest, 10)[0], 0.0, atol=1e-15)
    assert len(policy_capacity_tasks(manifest)) == 360


def test_xgb_extension_manifest_reaches_609_parameters_in_360_small_tasks() -> None:
    manifest = load_policy_capacity_manifest(XGB_EXTENSION_MANIFEST)

    assert manifest.models == ("xgb",)
    assert manifest.degrees == (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 21, 24, 28, 32)
    assert manifest.parameter_count(21) == 400
    assert manifest.parameter_count(32) == 609
    tasks = policy_capacity_tasks(manifest)
    assert len(tasks) == 360
    assert (tasks[0].split_seed, tasks[0].optimize_model, tasks[0].degree) == (0, "xgb", 0)
    assert (tasks[-1].split_seed, tasks[-1].optimize_model, tasks[-1].degree) == (
        19,
        "xgb",
        32,
    )


def test_xgb_full_polynomial_manifest_explodes_capacity_in_80_small_tasks() -> None:
    manifest = load_policy_capacity_manifest(XGB_FULL_POLYNOMIAL_MANIFEST)

    assert manifest.models == ("xgb",)
    assert manifest.basis == "total_degree_polynomial"
    assert manifest.degrees == (0, 1, 2, 3)
    assert [manifest.parameter_count(degree) for degree in manifest.degrees] == [
        1,
        20,
        210,
        1540,
    ]
    tasks = policy_capacity_tasks(manifest)
    assert len(tasks) == 80
    assert (tasks[0].split_seed, tasks[0].optimize_model, tasks[0].degree) == (0, "xgb", 0)
    assert (tasks[-1].split_seed, tasks[-1].optimize_model, tasks[-1].degree) == (
        19,
        "xgb",
        3,
    )


def test_split_positions_are_deterministic_disjoint_and_balanced() -> None:
    manifest = load_policy_capacity_manifest(MANIFEST)
    train, test = split_positions(manifest, 7)
    repeated_train, repeated_test = split_positions(manifest, 7)

    np.testing.assert_array_equal(train, repeated_train)
    np.testing.assert_array_equal(test, repeated_test)
    assert train.size == test.size == 100
    assert set(train).isdisjoint(test)
    assert set(np.concatenate([train, test])) == set(range(200))


def test_summary_and_capacity_plots_use_parameter_count_not_acceptance(tmp_path) -> None:
    rows = []
    degrees = (0, 1, 2, 3, 4, 5, 6, 8, 10)
    for seed in range(20):
        for optimize_model in ("glm", "xgb"):
            for evaluate_model in ("glm", "xgb"):
                for degree in degrees:
                    parameter_count = 1 + 19 * degree
                    offset = 0.2 if optimize_model == evaluate_model else -0.1
                    train_profit = 35.0 + 0.04 * parameter_count + offset + 0.01 * seed
                    test_profit = train_profit - 0.005 * parameter_count
                    rows.append(
                        {
                            "split_seed": seed,
                            "optimize_model": optimize_model,
                            "evaluate_model": evaluate_model,
                            "degree": degree,
                            "parameter_count": parameter_count,
                            "train_objective": -train_profit,
                            "test_objective": -test_profit,
                            "train_profit": train_profit,
                            "test_profit": test_profit,
                            "generalization_gap_profit": test_profit - train_profit,
                            "train_acceptance": 0.9,
                            "test_acceptance": 0.89,
                            "train_acceptance_violation": 0.0,
                            "test_acceptance_violation": 0.0,
                            "train_u_std": 0.02,
                            "test_u_std": 0.021,
                            "train_near_bound_fraction": 0.1,
                            "test_near_bound_fraction": 0.11,
                            "optimizer_runtime_sec": 1.0,
                        }
                    )
    summary = summarize_policy_capacity(pd.DataFrame(rows))
    manifest = load_policy_capacity_manifest(MANIFEST)
    enriched_rows = _with_acceptance_penalty_metrics(pd.DataFrame(rows), manifest)

    assert summary.shape[0] == 36
    assert set(summary["n_splits"]) == {20}
    paths = []
    for family in ("glm", "xgb"):
        paths.extend(
            [
                plot_policy_capacity_objective(summary, tmp_path, family=family),
                plot_policy_capacity_baseline_adjusted_gains(
                    pd.DataFrame(rows),
                    tmp_path,
                    family=family,
                ),
                plot_policy_capacity_action_diagnostics(
                    pd.DataFrame(rows),
                    tmp_path,
                    family=family,
                ),
                plot_policy_capacity_penalized_gains(
                    enriched_rows,
                    tmp_path,
                    family=family,
                ),
                plot_policy_capacity_generalization_gap(summary, tmp_path, family=family),
                plot_policy_capacity_model_transfer(summary, tmp_path, family=family),
            ]
        )
    endpoint_records = []
    for model in ("glm", "xgb"):
        for degree in (0, 5, 10):
            endpoint_records.append(
                {
                    "model": model,
                    "degree": degree,
                    "theta": np.zeros(1 + 19 * degree),
                    "state_dim": 19,
                    "clip_scale": 3.0,
                    "action_low": -0.1,
                    "action_high": 0.2,
                }
            )
    for family in ("glm", "xgb"):
        paths.append(
            plot_policy_capacity_endpoint_slices(
                endpoint_records,
                tmp_path,
                family=family,
            )
        )

    for path in paths:
        assert path.suffix == ".pdf"
        assert path.read_bytes().startswith(b"%PDF")
        assert not path.with_suffix(".png").exists()
