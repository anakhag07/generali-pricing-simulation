from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from objective.noise import HomoskedasticGaussianNoise, NoisyObjective
from objective.objectives import QuadraticObjective
from scratch import run_quadratic_homoskedastic_sweep as script


def test_full_and_pilot_grid_sizes() -> None:
    full = script._parse_args([])
    pilot = script._parse_args(["--pilot"])
    optax = script._parse_args(["--optimizer", "optax-adam"])
    optax_pilot = script._parse_args(["--pilot", "--optimizer", "optax-adam"])

    full_noise, full_radii, full_seeds = script._resolved_grid(full)
    pilot_noise, pilot_radii, pilot_seeds = script._resolved_grid(pilot)

    assert len(full_noise) * len(full_radii) * len(full_seeds) == 640
    assert len(pilot_noise) * len(pilot_radii) * len(pilot_seeds) == 60
    assert script._project_name(full) == script.PROJECT_NAME
    assert script._project_name(pilot) == script.PILOT_PROJECT_NAME
    assert script._project_name(optax) == script.OPTAX_PROJECT_NAME
    assert script._project_name(optax_pilot) == script.OPTAX_PILOT_PROJECT_NAME
    assert script._resolved_t_steps(full) == 200
    assert script._resolved_t_steps(optax) == 2000


def test_optimizer_cli_overrides_defaults() -> None:
    args = script._parse_args(
        ["--optimizer", "optax-adam", "--t-steps", "300", "--step-size", "0.02"]
    )

    assert args.optimizer == script.OPTAX_ADAM
    assert script._resolved_t_steps(args) == 300
    assert args.step_size == pytest.approx(0.02)
    assert args.array_max_parallel == 2


def test_optax_launch_plan_uses_gpu_array_by_grid_cell() -> None:
    optax = script._parse_args(["--optimizer", "optax-adam"])
    pilot = script._parse_args(["--optimizer", "optax-adam", "--pilot"])
    lbfgsb = script._parse_args([])

    assert len(script._task_specs(optax)) == 32
    assert len(set(script._task_specs(optax))) == 32
    assert len(script._task_specs(pilot)) == 12

    optax_plan = script._build_launch_plan(optax)
    assert optax_plan.name == script.OPTAX_PROJECT_NAME
    assert optax_plan.task_count == 32
    assert optax_plan.requires_jax is True
    assert optax_plan.default_launch == "auto"
    assert optax_plan.default_array is True

    lbfgsb_plan = script._build_launch_plan(lbfgsb)
    assert lbfgsb_plan.requires_jax is False
    assert lbfgsb_plan.default_launch == "local"
    assert lbfgsb_plan.default_array is False


def test_variant_name_round_trip() -> None:
    name = script._variant_name(1e-6, 1e-2)

    assert name == "noise-std-1e-06__fd-radius-0.01"
    assert script._parse_variant(name) == (1e-6, 1e-2)
    assert script._parse_variant("fd-radius-0.01") is None


def test_override_list_covers_noise_by_radius_product() -> None:
    overrides = script._build_override_list(
        dimension=4,
        noise_stds=(0.0, 1e-4),
        fd_radii=(1e-2, 1e-1),
        t_steps=25,
        optimizer=script.L_BFGS_B,
        step_size=0.05,
    )

    assert len(overrides) == 4
    assert [item["_run_name"] for item in overrides] == [
        "noise-std-0__fd-radius-0.01",
        "noise-std-0__fd-radius-0.1",
        "noise-std-0.0001__fd-radius-0.01",
        "noise-std-0.0001__fd-radius-0.1",
    ]
    for item in overrides:
        objective = item["objective"]
        assert isinstance(objective, NoisyObjective)
        assert isinstance(objective.base_objective, QuadraticObjective)
        assert isinstance(objective.noise, HomoskedasticGaussianNoise)
        assert item["enabled_estimators"] == (script.ESTIMATOR,)
        assert item["perturbation_space"] == "theta"
        assert item["step_rule"] == script.L_BFGS_B
        assert item["step_size"] == pytest.approx(0.05)
        assert item["ftol"] == pytest.approx(1e-12)
        assert np.linalg.norm(item["theta0"]) == pytest.approx(1.0)


def test_optax_overrides_use_learning_rate_without_ftol() -> None:
    overrides = script._build_override_list(
        dimension=2,
        noise_stds=(1e-4,),
        fd_radii=(1e-2,),
        t_steps=2000,
        optimizer=script.OPTAX_ADAM,
        step_size=0.05,
    )

    assert len(overrides) == 1
    assert overrides[0]["step_rule"] == script.OPTAX_ADAM
    assert overrides[0]["t_steps"] == 2000
    assert overrides[0]["step_size"] == pytest.approx(0.05)
    assert "ftol" not in overrides[0]


def test_run_grid_varies_only_noise_seed(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_sweep(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(project_dir=tmp_path / "project")

    monkeypatch.setattr(script, "run_sweep", fake_run_sweep)

    project_dir = script._run_grid(
        project_name="project",
        dimension=2,
        noise_stds=(0.0,),
        fd_radii=(0.1,),
        run_seeds=(7, 8),
        t_steps=2000,
        optimizer=script.OPTAX_ADAM,
        step_size=0.05,
    )

    assert project_dir == tmp_path / "project"
    assert calls[0]["base_preset"] == script.BASE_PRESET
    assert calls[0]["run_seeds"] == (7, 8)
    assert calls[0]["vary"] == ("noise",)
    assert calls[0]["anchor_seed"] == 7
    assert calls[0]["per_seed_plots"] is False
    override = calls[0]["override_list"][0]
    assert override["step_rule"] == script.OPTAX_ADAM
    assert override["t_steps"] == 2000
    assert override["step_size"] == pytest.approx(0.05)
    assert "ftol" not in override


def test_grid_task_runs_one_cell_and_returns_seed_summaries(monkeypatch, tmp_path) -> None:
    args = script._parse_args(
        [
            "--optimizer",
            "optax-adam",
            "--noise-stds",
            "0.001",
            "--fd-radii",
            "0.01",
            "--run-seeds",
            "7",
            "8",
        ]
    )
    variant = script._variant_name(0.001, 0.01)
    variant_dir = tmp_path / variant
    variant_dir.mkdir()
    for seed in (7, 8):
        (variant_dir / f"summary-seed-{seed}.json").write_text("{}", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run_grid(**kwargs):
        calls.append(kwargs)
        return tmp_path

    monkeypatch.setattr(script, "_run_grid", fake_run_grid)

    payload = script._run_grid_task(0, SimpleNamespace(), args=args)

    assert calls[0]["noise_stds"] == (0.001,)
    assert calls[0]["fd_radii"] == (0.01,)
    assert calls[0]["run_seeds"] == (7, 8)
    assert payload["variant_name"] == variant
    assert len(payload["summary_paths"]) == 2


def test_rows_from_task_payloads_reads_exact_summaries(tmp_path) -> None:
    paths = []
    for seed in (7, 8):
        path = tmp_path / f"summary-seed-{seed}.json"
        path.write_text(json.dumps(_summary_payload(run_seed=seed)), encoding="utf-8")
        paths.append(str(path))

    rows = script._rows_from_task_payloads(
        [{"noise_std": 1e-4, "fd_radius": 1e-2, "summary_paths": paths}]
    )

    assert [row["run_seed"] for row in rows] == [7, 8]
    assert all(row["noise_to_radius"] == pytest.approx(0.01) for row in rows)


def test_collector_rejects_missing_and_failed_tasks(monkeypatch, tmp_path) -> None:
    args = script._parse_args(
        ["--noise-stds", "0", "--fd-radii", "0.1", "--run-seeds", "7"]
    )
    context = SimpleNamespace(tasks_dir=tmp_path)

    monkeypatch.setattr(script, "read_task_records", lambda unused: [])
    with pytest.raises(RuntimeError, match="missing task indices=\\[0\\]"):
        script._collect_grid_tasks(context, args=args)

    monkeypatch.setattr(
        script,
        "read_task_records",
        lambda unused: [{"task_index": 0, "status": "failed", "error": "boom"}],
    )
    monkeypatch.setattr(
        script,
        "task_payloads",
        lambda unused: (_ for _ in ()).throw(RuntimeError("Cannot collect failed task records")),
    )
    with pytest.raises(RuntimeError, match="Cannot collect failed task records"):
        script._collect_grid_tasks(context, args=args)


def test_main_delegates_optax_to_launch_plan(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_launch_plan(plan, **kwargs):
        calls.append({"plan": plan, **kwargs})

    monkeypatch.setattr(script, "run_launch_plan", fake_run_launch_plan)

    script.main(
        [
            "--optimizer",
            "optax-adam",
            "--noise-stds",
            "0",
            "--fd-radii",
            "0.1",
            "--run-seeds",
            "7",
        ]
    )

    assert calls[0]["plan"].requires_jax is True
    assert calls[0]["plan"].task_count == 1
    assert calls[0]["cwd"] == script.REPO_ROOT
    assert calls[0]["argv"][1:] == [
        "--optimizer",
        "optax-adam",
        "--noise-stds",
        "0",
        "--fd-radii",
        "0.1",
        "--run-seeds",
        "7",
    ]


def test_summary_row_separates_clean_and_noisy_values(tmp_path) -> None:
    summary_path = tmp_path / "summary-seed-7.json"
    summary = _summary_payload(success=False)

    row = script._summary_row(
        summary,
        summary_path,
        noise_std=1e-4,
        fd_radius=1e-2,
    )

    assert row["noise_to_radius"] == pytest.approx(0.01)
    assert row["final_theta_norm"] == pytest.approx(0.5)
    assert row["clean_final_objective"] == pytest.approx(0.125)
    assert row["noisy_final_objective"] == pytest.approx(0.1)
    assert row["exploitation_gap"] == pytest.approx(-0.025)
    assert row["clean_improvement"] == pytest.approx(0.375)
    assert row["optimizer_success"] is False


def test_aggregation_includes_optimizer_failures_and_writes_csvs(tmp_path) -> None:
    rows = [
        script._summary_row(
            _summary_payload(success=success, clean=clean, noisy=noisy, run_seed=seed),
            tmp_path / f"summary-seed-{seed}.json",
            noise_std=1e-4,
            fd_radius=1e-2,
        )
        for success, clean, noisy, seed in (
            (True, 0.1, 0.08, 7),
            (False, 0.2, 0.15, 8),
        )
    ]

    summaries = script._aggregate_rows(rows)

    assert len(summaries) == 1
    assert summaries[0]["n_seeds"] == 2
    assert summaries[0]["optimizer_success_rate"] == pytest.approx(0.5)
    assert summaries[0]["clean_final_objective_median"] == pytest.approx(0.15)

    script._write_outputs(tmp_path, rows, plot=False)
    assert (tmp_path / "quadratic_homoskedastic_finals.csv").exists()
    assert (tmp_path / "quadratic_homoskedastic_summary.csv").exists()


def test_collect_rows_reads_saved_variant_summaries(tmp_path) -> None:
    variant_dir = tmp_path / "noise-std-0.0001__fd-radius-0.01"
    variant_dir.mkdir()
    path = variant_dir / "summary-seed-7.json"
    path.write_text(json.dumps(_summary_payload()), encoding="utf-8")

    rows = script._collect_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["run_seed"] == 7
    assert rows[0]["summary_path"] == str(path)


def _summary_payload(
    *,
    success: bool = True,
    clean: float = 0.125,
    noisy: float = 0.1,
    run_seed: int = 7,
) -> dict:
    theta_norm = np.sqrt(2.0 * clean)
    return {
        "run": {"run_dir": "/tmp/run"},
        "config": {
            "resolved_seed_setup": {"run_seed": run_seed},
            "objective": {
                "type": "NoisyObjective",
                "base_objective": {"type": "QuadraticObjective", "dimension": 2},
            },
        },
        "initial_value": 0.5,
        "estimators": {
            script.ESTIMATOR: {
                "theta": [theta_norm, 0.0],
                "final_value": clean,
                "runtime_sec": 0.01,
                "optimizer_success": success,
                "optimizer_status": 0 if success else 2,
                "optimizer_message": "ok" if success else "abnormal",
            }
        },
        "trace_summary": {
            script.ESTIMATOR: {
                "steps": 4,
                "final_objective": noisy,
            }
        },
    }
