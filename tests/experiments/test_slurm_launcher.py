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
    override_specs_require_jax,
    submit_to_slurm_if_needed,
)


class _FakeDevice:
    def __init__(self, platform: str) -> None:
        self.platform = platform


def _wrap_arg(command: list[str] | tuple[str, ...]) -> str:
    return next(part for part in command if part.startswith("--wrap="))


def test_override_specs_require_jax_from_grid_or_list() -> None:
    assert override_specs_require_jax(
        override_grid={"seed": [1], "compute_backend": ["numpy", "jax"]}
    ) is True
    assert override_specs_require_jax(
        override_list=[{"seed": 1}, {"compute_backend": "jax"}]
    ) is True
    assert override_specs_require_jax(
        override_grid={"seed": [1], "compute_backend": ["numpy"]}
    ) is False
    assert override_specs_require_jax(override_list=[{"seed": 1}]) is False


def test_build_sbatch_command_uses_gpu_profile_for_jax(tmp_path) -> None:
    command = build_sbatch_command(GPU_PROFILE, ["scripts/run_sweep.py"], cwd=tmp_path)
    wrap = _wrap_arg(command)

    assert "--partition=mit_normal_gpu" in command
    assert "--gres=gpu:l40s:1" in command
    assert "--job-name=generali-jax" in command
    assert "--output=outputs/slurm/%x-%j.out" in command
    assert f"--chdir={tmp_path}" in command
    assert "JAX_PLATFORM_NAME=gpu" in wrap
    assert "python scripts/run_sweep.py --no-sbatch" in wrap


def test_build_sbatch_command_uses_cpu_profile_without_gpu(tmp_path) -> None:
    command = build_sbatch_command(CPU_PROFILE, ["scripts/run_sweep.py"], cwd=tmp_path)
    wrap = _wrap_arg(command)

    assert "--partition=mit_normal" in command
    assert "--job-name=generali-cpu" in command
    assert not any(part.startswith("--gres=") for part in command)
    assert "JAX_PLATFORM_NAME" not in wrap


def test_submit_to_slurm_creates_log_dir_and_returns_job(tmp_path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_runner(command, *, check, capture_output, text):
        captured["command"] = command
        assert check is True
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(command, 0, stdout="12345\n", stderr="")

    submission = submit_to_slurm_if_needed(
        requires_jax=True,
        no_sbatch=False,
        argv=["scripts/run_sweep.py"],
        cwd=tmp_path,
        runner=fake_runner,
    )

    assert submission is not None
    assert submission.job_id == "12345"
    assert submission.profile.name == "gpu"
    assert (tmp_path / "outputs" / "slurm").is_dir()
    assert captured["command"] == list(submission.command)


def test_submit_to_slurm_skips_when_disabled_or_already_allocated(tmp_path) -> None:
    def fail_runner(*args, **kwargs):
        raise AssertionError("runner should not be called")

    assert submit_to_slurm_if_needed(
        requires_jax=False,
        no_sbatch=True,
        argv=["scripts/run_sweep.py"],
        cwd=tmp_path,
        runner=fail_runner,
    ) is None
    assert submit_to_slurm_if_needed(
        requires_jax=False,
        no_sbatch=False,
        argv=["scripts/run_sweep.py"],
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


def test_assert_jax_gpu_available_explains_missing_cuda_jaxlib() -> None:
    jax_config = SimpleNamespace(compute_backend="jax")

    def fail_default_backend() -> str:
        raise RuntimeError("Unknown backend: 'gpu'")

    fake_jax = SimpleNamespace(default_backend=fail_default_backend, devices=lambda: [])

    with pytest.raises(RuntimeError, match="pip install -U -e"):
        assert_jax_gpu_available([jax_config], jax_module=fake_jax)
