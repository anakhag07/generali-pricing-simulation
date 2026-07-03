"""Slurm launch helpers for ORCD experiment entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any


from experiments.paths import results_root


SLURM_CHILD_ENV = "GENERALI_SLURM_CHILD"
DEFAULT_CONDA_ENV = "simulation_env"
DEFAULT_MODULE = "miniforge/24.3.0-0"


@dataclass(frozen=True)
class SlurmProfile:
    """Resource profile for one ORCD Slurm submission."""

    name: str
    partition: str
    time: str
    nodes: int
    ntasks: int
    cpus_per_task: int
    memory: str
    job_name: str
    output: str
    gres: str | None = None


@dataclass(frozen=True)
class SlurmSubmission:
    """Submitted Slurm job metadata returned to the parent launcher."""

    profile: SlurmProfile
    job_id: str
    command: tuple[str, ...]


CPU_PROFILE = SlurmProfile(
    name="cpu",
    partition="mit_normal",
    time="06:00:00",
    nodes=1,
    ntasks=1,
    cpus_per_task=8,
    memory="64G",
    job_name="generali-cpu",
    output="outputs/slurm/%x-%j.out",
)

GPU_PROFILE = SlurmProfile(
    name="gpu",
    partition="mit_normal_gpu",
    time="06:00:00",
    nodes=1,
    ntasks=1,
    cpus_per_task=8,
    memory="64G",
    gres="gpu:l40s:1",
    job_name="generali-jax",
    output="outputs/slurm/%x-%j.out",
)


def run_specs_require_jax(run_specs: Sequence[Any]) -> bool:
    """Return whether lightweight run specs explicitly request the JAX backend."""
    for run_spec in run_specs:
        if not isinstance(run_spec, tuple) or len(run_spec) != 2:
            continue
        _, overrides = run_spec
        if isinstance(overrides, Mapping) and overrides.get("compute_backend") == "jax":
            return True
    return False


def configs_require_jax(configs: Sequence[Any]) -> bool:
    """Return whether any resolved config uses ``compute_backend='jax'``."""
    return any(getattr(config, "compute_backend", "numpy") == "jax" for config in configs)


def profile_for_backend(*, requires_jax: bool) -> SlurmProfile:
    """Select the ORCD Slurm profile for the selected experiment backend."""
    return GPU_PROFILE if requires_jax else CPU_PROFILE


def in_slurm_allocation(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the current process already belongs to a Slurm job."""
    env_map = os.environ if env is None else env
    return bool(env_map.get("SLURM_JOB_ID") or env_map.get(SLURM_CHILD_ENV) == "1")


def child_argv(argv: Sequence[str]) -> list[str]:
    """Return the command the Slurm child should run, with autosubmit disabled."""
    script_argv = list(argv) if argv else ["main.py"]
    if "--no-sbatch" not in script_argv:
        script_argv.append("--no-sbatch")
    return ["python", *script_argv]


def build_sbatch_command(
    profile: SlurmProfile,
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    output: str | None = None,
    conda_env: str = DEFAULT_CONDA_ENV,
    module_name: str = DEFAULT_MODULE,
) -> list[str]:
    """Build the ``sbatch`` command for a parent process to submit."""
    workdir = (Path.cwd() if cwd is None else Path(cwd)).resolve()
    source_path = workdir / "src"
    setup_commands = [
        "set -euo pipefail",
        f"module load {shlex.quote(module_name)}",
        'source "$(conda info --base)/etc/profile.d/conda.sh"',
        f"conda activate {shlex.quote(conda_env)}",
        f"export {SLURM_CHILD_ENV}=1",
        "export PYTHONUNBUFFERED=1",
        f"export PYTHONPATH={shlex.quote(str(source_path))}${{PYTHONPATH:+:$PYTHONPATH}}",
    ]
    if profile.gres is not None:
        setup_commands.append("export JAX_PLATFORM_NAME=gpu")
    setup_commands.append(shlex.join(child_argv(argv)))
    wrapped = "bash -lc " + shlex.quote("; ".join(setup_commands))

    command = [
        "sbatch",
        "--parsable",
        f"--partition={profile.partition}",
        f"--time={profile.time}",
        f"--nodes={profile.nodes}",
        f"--ntasks={profile.ntasks}",
        f"--cpus-per-task={profile.cpus_per_task}",
        f"--mem={profile.memory}",
        f"--job-name={profile.job_name}",
        f"--output={output or profile.output}",
        f"--chdir={workdir}",
    ]
    if profile.gres is not None:
        command.append(f"--gres={profile.gres}")
    command.append(f"--wrap={wrapped}")
    return command


def submit_to_slurm_if_needed(
    *,
    requires_jax: bool,
    no_sbatch: bool,
    argv: Sequence[str],
    cwd: Path | None = None,
    log_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
    runner: Any = subprocess.run,
) -> SlurmSubmission | None:
    """Submit the current entry point to Slurm unless already allocated or disabled."""
    if no_sbatch or in_slurm_allocation(env):
        return None

    profile = profile_for_backend(requires_jax=requires_jax)
    workdir = (Path.cwd() if cwd is None else Path(cwd)).resolve()
    if log_dir is None:
        log_path = results_root() / "slurm"
    else:
        log_path = log_dir if log_dir.is_absolute() else workdir / log_dir
    log_path.mkdir(parents=True, exist_ok=True)

    output = str(log_path / "%x-%j.out")
    command = build_sbatch_command(profile, argv, cwd=workdir, output=output)
    result = runner(command, check=True, capture_output=True, text=True)
    job_id = str(result.stdout).strip() or "unknown"
    return SlurmSubmission(profile=replace(profile, output=output), job_id=job_id, command=tuple(command))


def assert_jax_gpu_available(configs: Sequence[Any], *, jax_module: Any | None = None) -> str | None:
    """Validate that JAX configs are running on a non-CPU backend."""
    if not configs_require_jax(configs):
        return None

    if jax_module is None:
        try:
            import jax as jax_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError("compute_backend='jax' requires JAX to be installed.") from exc

    backend = str(jax_module.default_backend()).lower()
    devices = list(jax_module.devices())
    platforms = {str(getattr(device, "platform", "")).lower() for device in devices}
    if backend == "cpu" or not devices or platforms <= {"cpu"}:
        raise RuntimeError(
            "compute_backend='jax' requires a GPU Slurm allocation. "
            f"JAX reported backend={backend!r}, devices={devices!r}. "
            "Run without --no-sbatch from the entry point, or request a GPU allocation manually."
        )
    return f"JAX backend: {backend}; devices: {devices}"


__all__ = [
    "CPU_PROFILE",
    "GPU_PROFILE",
    "SLURM_CHILD_ENV",
    "SlurmProfile",
    "SlurmSubmission",
    "assert_jax_gpu_available",
    "build_sbatch_command",
    "child_argv",
    "configs_require_jax",
    "in_slurm_allocation",
    "profile_for_backend",
    "run_specs_require_jax",
    "submit_to_slurm_if_needed",
]
