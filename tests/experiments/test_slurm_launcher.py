from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from experiments.slurm import (
    CPU_PROFILE,
    GPU_PROFILE,
    SLURM_CHILD_ENV,
    assert_jax_gpu_available,
    build_sbatch_command,
    configs_require_jax,
    in_slurm_allocation,
    run_specs_require_jax,
    submit_to_slurm_if_needed,
)


class _FakeDevice:
    def __init__(self, platform: str) -> None:
        self.platform = platform


def _wrap_arg(command: list[str] | tuple[str, ...]) -> str:
    return next(part for part in command if part.startswith("--wrap="))


def _expected_pythonpath_export(tmp_path) -> str:
    return f"export PYTHONPATH={tmp_path / 'src'}${{PYTHONPATH:+:$PYTHONPATH}}"


def test_run_specs_require_jax_from_overrides() -> None:
    specs = [
        "fixed_regression_base",
        ("real_data_glm_base", {"compute_backend": "jax"}),
    ]

    assert run_specs_require_jax(specs) is True
    assert run_specs_require_jax([("real_data_glm_base", {"compute_backend": "numpy"})]) is False


def test_build_sbatch_command_uses_gpu_profile_for_jax(tmp_path) -> None:
    command = build_sbatch_command(GPU_PROFILE, ["main.py"], cwd=tmp_path)
    wrap = _wrap_arg(command)

    assert "--partition=mit_normal_gpu" in command
    assert "--gres=gpu:l40s:1" in command
    assert "--job-name=generali-jax" in command
    assert "--output=outputs/slurm/%x-%j.out" in command
    assert f"--chdir={tmp_path}" in command
    assert _expected_pythonpath_export(tmp_path) in wrap
    assert "JAX_PLATFORM_NAME=gpu" in wrap
    assert "python main.py --no-sbatch" in wrap


def test_build_sbatch_command_uses_cpu_profile_without_gpu(tmp_path) -> None:
    command = build_sbatch_command(CPU_PROFILE, ["main.py"], cwd=tmp_path)
    wrap = _wrap_arg(command)

    assert "--partition=mit_normal" in command
    assert "--job-name=generali-cpu" in command
    assert _expected_pythonpath_export(tmp_path) in wrap
    assert not any(part.startswith("--gres=") for part in command)
    assert "JAX_PLATFORM_NAME" not in wrap


def test_submit_to_slurm_creates_log_dir_and_returns_job(tmp_path, monkeypatch) -> None:
    captured: dict[str, list[str]] = {}
    results_root = tmp_path / "results"
    monkeypatch.setenv("GENERALI_RESULTS_ROOT", str(results_root))

    def fake_runner(command, *, check, capture_output, text):
        captured["command"] = command
        assert check is True
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(command, 0, stdout="12345\n", stderr="")

    submission = submit_to_slurm_if_needed(
        requires_jax=True,
        no_sbatch=False,
        argv=["main.py"],
        cwd=tmp_path,
        runner=fake_runner,
    )

    assert submission is not None
    assert submission.job_id == "12345"
    assert submission.profile.name == "gpu"
    assert (results_root / "slurm").is_dir()
    assert submission.profile.output == str(results_root / "slurm" / "%x-%j.out")
    assert f"--output={results_root / 'slurm' / '%x-%j.out'}" in captured["command"]
    assert captured["command"] == list(submission.command)


def test_submit_to_slurm_skips_when_disabled_or_already_allocated(tmp_path) -> None:
    def fail_runner(*args, **kwargs):
        raise AssertionError("runner should not be called")

    assert submit_to_slurm_if_needed(
        requires_jax=False,
        no_sbatch=True,
        argv=["main.py"],
        cwd=tmp_path,
        runner=fail_runner,
    ) is None
    assert submit_to_slurm_if_needed(
        requires_jax=False,
        no_sbatch=False,
        argv=["main.py"],
        cwd=tmp_path,
        env={"SLURM_JOB_ID": "99"},
        runner=fail_runner,
    ) is None
    assert in_slurm_allocation({SLURM_CHILD_ENV: "1"}) is True


def test_assert_jax_gpu_available_checks_resolved_configs() -> None:
    jax_config = SimpleNamespace(compute_backend="jax")
    numpy_config = SimpleNamespace(compute_backend="numpy")
    fake_gpu_jax = SimpleNamespace(
        default_backend=lambda: "gpu",
        devices=lambda: [_FakeDevice("gpu")],
    )

    assert configs_require_jax([numpy_config, jax_config]) is True
    status = assert_jax_gpu_available([jax_config], jax_module=fake_gpu_jax)
    assert status is not None
    assert "JAX backend: gpu" in status
    assert assert_jax_gpu_available([numpy_config], jax_module=fake_gpu_jax) is None


def test_assert_jax_gpu_available_rejects_cpu_backend() -> None:
    jax_config = SimpleNamespace(compute_backend="jax")
    fake_cpu_jax = SimpleNamespace(
        default_backend=lambda: "cpu",
        devices=lambda: [_FakeDevice("cpu")],
    )

    with pytest.raises(RuntimeError, match="requires a GPU Slurm allocation"):
        assert_jax_gpu_available([jax_config], jax_module=fake_cpu_jax)
