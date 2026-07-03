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

    def fake_submit_to_slurm_if_needed(*, requires_jax, no_sbatch, argv):
        calls["submit"] = (requires_jax, no_sbatch, argv)
        return SimpleNamespace(
            profile=SimpleNamespace(name="gpu", output="outputs/slurm/%x-%j.out"),
            job_id="12345",
        )

    def fail_run_sweep(**kwargs):
        raise AssertionError("sweep should not run in the parent process")

    monkeypatch.setattr(script, "submit_to_slurm_if_needed", fake_submit_to_slurm_if_needed)
    monkeypatch.setattr(script, "run_sweep", fail_run_sweep)

    script.main([])

    requires_jax, no_sbatch, argv = calls["submit"]
    assert requires_jax is False
    assert no_sbatch is False
    assert isinstance(argv, list)
    assert "--no-sbatch" not in argv


def test_run_sweep_no_sbatch_runs_without_jax_preflight(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_submit_to_slurm_if_needed(*, requires_jax, no_sbatch, argv):
        calls["submit"] = (requires_jax, no_sbatch, argv)
        return None

    def fake_assert_jax_gpu_available(configs):
        calls["preflight"] = configs
        return "JAX backend: gpu; devices: [FakeGpu]"

    def fake_run_sweep(**kwargs):
        calls["sweep"] = kwargs
        return SimpleNamespace(run_results=[object()], summary_rows=[], project_dir="outputs")

    monkeypatch.setattr(script, "submit_to_slurm_if_needed", fake_submit_to_slurm_if_needed)
    monkeypatch.setattr(script, "assert_jax_gpu_available", fake_assert_jax_gpu_available)
    monkeypatch.setattr(script, "run_sweep", fake_run_sweep)

    script.main(["--no-sbatch"])

    requires_jax, no_sbatch, _ = calls["submit"]
    assert requires_jax is False
    assert no_sbatch is True
    assert "preflight" not in calls
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
