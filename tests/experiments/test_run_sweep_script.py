from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts import run_sweep as script


def test_run_sweep_uses_theta_offset_override_list() -> None:
    assert len(script.OVERRIDE_LIST) == len(script.THETA_OFFSETS)
    assert [entry["_run_name"] for entry in script.OVERRIDE_LIST] == [
        "theta-offset-0",
        "theta-offset-0.01",
        "theta-offset-0.1",
        "theta-offset-0.25",
        "theta-offset-0.5",
        "theta-offset-1",
        "theta-offset-2",
        "theta-offset-5",
        "theta-offset-10",
    ]
    for entry, offset in zip(script.OVERRIDE_LIST, script.THETA_OFFSETS):
        np.testing.assert_allclose(entry["theta0"], script.BASE_THETA + float(offset))
        assert entry["objective"].noise.std == script.NOISE_STD


def test_run_sweep_submits_to_slurm_before_running(monkeypatch) -> None:
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
    assert plan.task_count == len(script.OVERRIDE_LIST) * len(script.RUN_SEEDS)
    assert plan.requires_jax is False
    assert args.no_sbatch is False
    assert isinstance(argv, list)
    assert "--no-sbatch" not in argv


def test_run_sweep_no_sbatch_runs_without_jax_preflight(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_run_launch_plan(plan, *, args, argv):
        calls["args"] = args
        calls["argv"] = argv

    monkeypatch.setattr(script, "run_launch_plan", fake_run_launch_plan)

    script.main(["--no-sbatch"])

    assert calls["args"].no_sbatch is True
    assert "--no-sbatch" in calls["argv"]


def test_run_sweep_serial_path_uses_seed_aware_sweep(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_run_sweep(**kwargs):
        calls["sweep"] = kwargs
        return SimpleNamespace(run_results=[object()], summary_rows=[], project_dir="outputs")

    monkeypatch.setattr(script, "run_sweep", fake_run_sweep)

    script._run_sweep_serial(SimpleNamespace())

    assert calls["sweep"] == {
        "base_preset": script.BASE_PRESET,
        "run_seeds": script.RUN_SEEDS,
        "override_list": script.OVERRIDE_LIST,
        "vary": script.VARY,
        "anchor_seed": script.ANCHOR_SEED,
        "fixed": script.FIXED_SEEDS,
        "project_name": script.PROJECT_NAME,
        "display_keys": script.DISPLAY_KEYS,
    }
