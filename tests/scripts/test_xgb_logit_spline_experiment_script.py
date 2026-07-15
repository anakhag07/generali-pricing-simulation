from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.launch import LaunchContext
from experiments.results import EstimatorResult, OptimizationTrace
from scripts import run_xgb_logit_spline_experiment as script


def test_default_overrides_run_all_profiles_with_held_out_diagnostics() -> None:
    args = script._parse_args([])

    overrides = script._config_overrides(args)

    assert overrides["n_samples"] is None
    assert overrides["train_fraction"] == 0.8
    assert overrides["test_fraction"] == 0.2
    assert overrides["initial_u"] == 0.08
    assert overrides["sigma"] == 1e-4
    assert overrides["perturbation_space"] == "u"
    assert overrides["enabled_estimators"] == ("first_order", "finite_difference")
    assert overrides["constant_u_baselines"] == (0.0, 0.08, 0.16)
    assert overrides["plot"] is True
    assert overrides["wandb_enabled"] is False


def test_build_config_enables_exact_gradient_diagnostics() -> None:
    args = script._parse_args(
        ["--n-samples", "8", "--t-steps", "1", "--estimators", "first_order", "--quiet"]
    )

    config = script._build_config(args)

    assert config.correctness.gradient_source == "exact"
    assert config.x_fixed.shape[0] == 8
    assert config.train_fraction == 0.8
    assert config.test_fraction == 0.2
    assert config.objective.u_bounds == (0.0, 0.16)
    assert config.objective.policy.action_low == 0.0
    assert config.objective.policy.action_high == 0.16


def test_convergence_rows_include_optimizer_and_true_gradient_diagnostics() -> None:
    trace = OptimizationTrace(
        steps=[0, 1],
        u_values=[0.08, 0.1],
        objective_values=[-1.0, -2.0],
        u_grad_estimates=[np.nan, np.nan],
        theta_grad_norms=[2.0, 0.2],
        true_theta_grad_norms=[1.9, 0.1],
        optimizer_success=True,
        optimizer_status=0,
        optimizer_message="converged",
    )
    final = EstimatorResult(
        theta=np.asarray([0.1]),
        u=0.1,
        value=-2.0,
        time=0.5,
        mean_acceptance=0.75,
    )
    result = SimpleNamespace(results={"first_order": final}, traces={"first_order": trace})

    rows = script._convergence_rows(result)

    assert rows == [
        {
            "estimator": "first_order",
            "optimizer_success": True,
            "optimizer_status": 0,
            "optimizer_message": "converged",
            "steps": 2,
            "final_objective": -2.0,
            "final_true_theta_grad_norm": 0.1,
            "mean_u": 0.1,
            "mean_acceptance": 0.75,
            "runtime_sec": 0.5,
        }
    ]


def test_task_delegates_to_standard_execution(tmp_path, monkeypatch) -> None:
    args = script._parse_args(["--n-samples", "8", "--quiet"])
    config = object()
    fake_result = object()
    run_dir = tmp_path / "run"
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(script, "_build_config", lambda parsed_args: config)
    monkeypatch.setattr(script, "_convergence_rows", lambda result: [{"estimator": "first_order"}])
    monkeypatch.setattr(script, "_print_convergence", lambda rows, output_dir: None)

    def fake_execute(name, run_config, **kwargs):
        calls.append({"name": name, "config": run_config, **kwargs})
        return SimpleNamespace(result=fake_result, run_context=SimpleNamespace(run_dir=run_dir))

    monkeypatch.setattr(script, "execute_experiment_run", fake_execute)
    context = LaunchContext(
        plan_name=script.PROJECT_NAME,
        runs_root=tmp_path,
        sweep_id="sweep",
        sweep_dir=tmp_path / "sweeps" / "sweep",
        tasks_dir=tmp_path / "sweeps" / "sweep" / "tasks",
        launch_mode="local",
        array=False,
        task_index=0,
    )

    payload = script._run_task(0, context, args)

    assert calls[0]["name"] == script.RUN_NAME
    assert calls[0]["config"] is config
    assert calls[0]["runs_root"] == tmp_path
    assert calls[0]["run_metadata"]["preset_name"] == script.BASE_PRESET
    assert payload == {
        "run_dir": str(run_dir),
        "convergence": [{"estimator": "first_order"}],
    }


def test_launch_plan_is_one_cpu_local_task(monkeypatch) -> None:
    args = script._parse_args([])
    monkeypatch.setattr(script, "results_root", lambda: Path("/tmp/results"))

    plan = script._build_launch_plan(args)

    assert plan.task_count == 1
    assert plan.requires_jax is False
    assert plan.default_launch == "local"
    assert plan.default_array is False
    assert plan.runs_root == "/tmp/results/xgb-logit-spline-experiment"
