from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts import run_sweep as script


def test_run_sweep_uses_theta_offset_override_list() -> None:
    assert len(script.THETA_OVERRIDE_LIST) == len(script.THETA_OFFSETS)
    assert script.OVERRIDE_LIST is script.THETA_OVERRIDE_LIST
    assert [entry["_run_name"] for entry in script.THETA_OVERRIDE_LIST] == [
        f"theta-offset-{float(offset):g}" for offset in script.THETA_OFFSETS
    ]
    for entry, offset in zip(script.THETA_OVERRIDE_LIST, script.THETA_OFFSETS):
        np.testing.assert_allclose(entry["theta0"], script.BASE_THETA + float(offset))
        assert entry["objective"].noise.std == script.NOISE_STD


def test_run_sweep_uses_noise_override_list() -> None:
    assert len(script.NOISE_OVERRIDE_LIST) == len(script.NOISE_STDS)
    assert [entry["_run_name"] for entry in script.NOISE_OVERRIDE_LIST] == [
        f"noise-std-{float(noise_std):g}" for noise_std in script.NOISE_STDS
    ]
    for entry, noise_std in zip(script.NOISE_OVERRIDE_LIST, script.NOISE_STDS):
        np.testing.assert_allclose(entry["theta0"], np.zeros_like(script.BASE_THETA))
        assert entry["objective"].noise.std == float(noise_std)


def test_run_sweep_sweeps_bind_both_grids() -> None:
    assert script.SWEEPS == (
        (script.THETA_PROJECT_NAME, script.THETA_OVERRIDE_LIST),
        (script.NOISE_PROJECT_NAME, script.NOISE_OVERRIDE_LIST),
    )


def test_run_sweep_main_delegates_to_launch_plan(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_run_launch_plan(plan, *, args, argv):
        calls["plan"] = plan
        calls["args"] = args
        calls["argv"] = argv

    monkeypatch.setattr(script, "run_launch_plan", fake_run_launch_plan)

    script.main([])

    plan = calls["plan"]
    args = calls["args"]
    argv = calls["argv"]
    expected_tasks = (
        len(script.THETA_OVERRIDE_LIST) + len(script.NOISE_OVERRIDE_LIST)
    ) * len(script.RUN_SEEDS)
    assert plan.task_count == expected_tasks
    assert plan.requires_jax is False
    assert plan.name == script.LAUNCH_PLAN_NAME
    assert args.no_sbatch is False
    assert isinstance(argv, list)
    assert "--no-sbatch" not in argv


def test_run_sweep_no_sbatch_passes_launch_local(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_run_launch_plan(plan, *, args, argv):
        calls["args"] = args
        calls["argv"] = argv

    monkeypatch.setattr(script, "run_launch_plan", fake_run_launch_plan)

    script.main(["--no-sbatch"])

    assert calls["args"].no_sbatch is True
    assert "--no-sbatch" in calls["argv"]


def test_run_sweep_serial_path_runs_both_grids(monkeypatch) -> None:
    sweep_calls: list[dict[str, object]] = []

    def fake_run_sweep(**kwargs):
        sweep_calls.append(kwargs)
        return SimpleNamespace(run_results=[object()], summary_rows=[], project_dir="results")

    monkeypatch.setattr(script, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(script, "_variant_is_completed", lambda variant_dir, required_estimators: False)
    monkeypatch.setattr(script, "_regenerate_distance_plots", lambda: None)

    script._run_sweep_serial(SimpleNamespace())

    assert len(sweep_calls) == 2
    theta_call, noise_call = sweep_calls
    common = {
        "base_preset": script.BASE_PRESET,
        "run_seeds": script.RUN_SEEDS,
        "vary": script.VARY,
        "anchor_seed": script.ANCHOR_SEED,
        "fixed": script.FIXED_SEEDS,
        "display_keys": script.DISPLAY_KEYS,
    }
    assert {k: theta_call[k] for k in common} == common
    assert theta_call["project_name"] == script.THETA_PROJECT_NAME
    assert [entry["_run_name"] for entry in theta_call["override_list"]] == [
        entry["_run_name"] for entry in script.THETA_OVERRIDE_LIST
    ]
    assert {k: noise_call[k] for k in common} == common
    assert noise_call["project_name"] == script.NOISE_PROJECT_NAME
    assert [entry["_run_name"] for entry in noise_call["override_list"]] == [
        entry["_run_name"] for entry in script.NOISE_OVERRIDE_LIST
    ]


def test_run_sweep_skips_completed_variants(monkeypatch) -> None:
    calls: list[object] = []

    def fake_variant_is_completed(variant_dir, required_estimators):
        return variant_dir.name in {"theta-offset-0", "theta-offset-0.01"}

    def fake_run_sweep(**kwargs):
        calls.append(kwargs["override_list"])
        return SimpleNamespace(run_results=[], summary_rows=[], project_dir="results")

    monkeypatch.setattr(script, "_variant_is_completed", fake_variant_is_completed)
    monkeypatch.setattr(script, "run_sweep", fake_run_sweep)

    script._run_missing_sweep(
        base_preset=script.BASE_PRESET,
        project_name=script.THETA_PROJECT_NAME,
        override_list=script.THETA_OVERRIDE_LIST,
        run_seeds=script.RUN_SEEDS,
        vary=script.VARY,
        anchor_seed=script.ANCHOR_SEED,
        fixed=script.FIXED_SEEDS,
        display_keys=script.DISPLAY_KEYS,
        required_estimators=script.REQUIRED_ESTIMATORS,
    )

    assert len(calls) == 1
    remaining_names = [entry["_run_name"] for entry in calls[0]]
    assert "theta-offset-0" not in remaining_names
    assert "theta-offset-0.01" not in remaining_names
