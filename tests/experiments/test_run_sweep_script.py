from __future__ import annotations

from types import SimpleNamespace

from scripts import run_sweep as script


def test_run_sweep_defaults_to_jax_backend() -> None:
    assert script.OVERRIDE_GRID["compute_backend"] == ["jax"]


def test_run_sweep_submits_to_slurm_before_running(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_submit_to_slurm_if_needed(*, requires_jax, no_sbatch, argv):
        calls["submit"] = (requires_jax, no_sbatch, argv)
        return SimpleNamespace(
            profile=SimpleNamespace(name="gpu", output="outputs/slurm/%x-%j.out"),
            job_id="12345",
        )

    def fail_run_preset_sweep(**kwargs):
        raise AssertionError("sweep should not run in the parent process")

    monkeypatch.setattr(script, "submit_to_slurm_if_needed", fake_submit_to_slurm_if_needed)
    monkeypatch.setattr(script, "run_preset_sweep", fail_run_preset_sweep)

    script.main([])

    requires_jax, no_sbatch, argv = calls["submit"]
    assert requires_jax is True
    assert no_sbatch is False
    assert isinstance(argv, list)
    assert "--no-sbatch" not in argv


def test_run_sweep_no_sbatch_runs_after_jax_preflight(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_submit_to_slurm_if_needed(*, requires_jax, no_sbatch, argv):
        calls["submit"] = (requires_jax, no_sbatch, argv)
        return None

    def fake_assert_jax_gpu_available(configs):
        calls["preflight"] = configs
        return "JAX backend: gpu; devices: [FakeGpu]"

    def fake_run_preset_sweep(**kwargs):
        calls["sweep"] = kwargs
        return [("run", object())]

    monkeypatch.setattr(script, "submit_to_slurm_if_needed", fake_submit_to_slurm_if_needed)
    monkeypatch.setattr(script, "assert_jax_gpu_available", fake_assert_jax_gpu_available)
    monkeypatch.setattr(script, "run_preset_sweep", fake_run_preset_sweep)

    script.main(["--no-sbatch"])

    requires_jax, no_sbatch, _ = calls["submit"]
    assert requires_jax is True
    assert no_sbatch is True
    assert getattr(calls["preflight"][0], "compute_backend") == "jax"
    assert calls["sweep"] == {
        "base_preset": script.BASE_PRESET,
        "override_grid": script.OVERRIDE_GRID,
        "project_name": script.PROJECT_NAME,
        "display_keys": script.DISPLAY_KEYS,
    }
