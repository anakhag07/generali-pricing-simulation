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
    sweep_calls: list[dict[str, object]] = []

    def fake_submit_to_slurm_if_needed(*, requires_jax, no_sbatch, argv):
        calls["submit"] = (requires_jax, no_sbatch, argv)
        return None

    def fake_assert_jax_gpu_available(configs):
        calls["preflight"] = configs
        return "JAX backend: gpu; devices: [FakeGpu]"

    def fake_run_sweep(**kwargs):
        sweep_calls.append(kwargs)
        return SimpleNamespace(run_results=[object()], summary_rows=[], project_dir="outputs")

    monkeypatch.setattr(script, "submit_to_slurm_if_needed", fake_submit_to_slurm_if_needed)
    monkeypatch.setattr(script, "assert_jax_gpu_available", fake_assert_jax_gpu_available)
    monkeypatch.setattr(script, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(script, "_variant_is_completed", lambda variant_dir, required_estimators: False)
    monkeypatch.setattr(script, "_regenerate_distance_plots", lambda: None)

    script.main(["--no-sbatch"])

    requires_jax, no_sbatch, _ = calls["submit"]
    assert requires_jax is False
    assert no_sbatch is True
    assert "preflight" not in calls
    assert len(sweep_calls) == 2
    theta_call, noise_call = sweep_calls
    assert {key: theta_call[key] for key in theta_call if key != "override_list"} == {
        "base_preset": script.BASE_PRESET,
        "run_seeds": script.RUN_SEEDS,
        "vary": script.VARY,
        "anchor_seed": script.ANCHOR_SEED,
        "fixed": script.FIXED_SEEDS,
        "runs_root": "outputs",
        "project_name": script.THETA_PROJECT_NAME,
        "display_keys": script.DISPLAY_KEYS,
    }
    assert [entry["_run_name"] for entry in theta_call["override_list"]] == [
        entry["_run_name"] for entry in script.THETA_OVERRIDE_LIST
    ]
    assert {key: noise_call[key] for key in noise_call if key != "override_list"} == {
        "base_preset": script.BASE_PRESET,
        "run_seeds": script.RUN_SEEDS,
        "vary": script.VARY,
        "anchor_seed": script.ANCHOR_SEED,
        "fixed": script.FIXED_SEEDS,
        "runs_root": "outputs",
        "project_name": script.NOISE_PROJECT_NAME,
        "display_keys": script.DISPLAY_KEYS,
    }
    assert [entry["_run_name"] for entry in noise_call["override_list"]] == [
        entry["_run_name"] for entry in script.NOISE_OVERRIDE_LIST
    ]


def test_run_sweep_skips_completed_variants(monkeypatch) -> None:
    calls: list[object] = []

    def fake_variant_is_completed(variant_dir, required_estimators):
        return variant_dir.name in {"theta-offset-0", "theta-offset-0.01"}

    def fake_run_sweep(**kwargs):
        calls.append(kwargs["override_list"])
        return SimpleNamespace(run_results=[], summary_rows=[], project_dir="outputs")

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
