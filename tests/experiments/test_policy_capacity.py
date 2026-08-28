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
)
from objective.policy import AdditiveChebyshevFeatureMap, SoftmaxPolicy
from reporting.visualization import (
    plot_policy_capacity_endpoint_slices,
    plot_policy_capacity_generalization_gap,
    plot_policy_capacity_model_transfer,
    plot_policy_capacity_objective,
)


MANIFEST = Path(__file__).parents[2] / "manifests" / "policy_capacity_glm_xgb.json"


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
                            "optimizer_runtime_sec": 1.0,
                        }
                    )
    summary = summarize_policy_capacity(pd.DataFrame(rows))

    assert summary.shape[0] == 36
    assert set(summary["n_splits"]) == {20}
    paths = [
        plot_policy_capacity_objective(summary, tmp_path),
        plot_policy_capacity_generalization_gap(summary, tmp_path),
        plot_policy_capacity_model_transfer(summary, tmp_path),
    ]
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
    paths.append(plot_policy_capacity_endpoint_slices(endpoint_records, tmp_path))

    for path in paths:
        assert path.suffix == ".pdf"
        assert path.read_bytes().startswith(b"%PDF")
        assert path.with_suffix(".png").exists()
