from __future__ import annotations

from argparse import Namespace
import subprocess
from types import SimpleNamespace

import pytest

from experiments.launch import LaunchPlan, read_task_records, run_launch_plan


def _args(**overrides):
    values = {
        "launch": "local",
        "no_sbatch": False,
        "array": False,
        "array_max_parallel": None,
        "task_index": None,
        "sweep_id": "test-sweep",
        "collect": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_local_array_runs_tasks_and_collector(tmp_path) -> None:
    calls: list[object] = []

    def run_task(index, context):
        calls.append(index)
        return {"value": index, "task_dir": context.tasks_dir}

    def collect(context):
        calls.append(("collect", len(read_task_records(context))))

    plan = LaunchPlan(
        name="demo",
        task_count=2,
        requires_jax=False,
        run_task=run_task,
        collect=collect,
        runs_root=str(tmp_path),
    )

    run_launch_plan(plan, args=_args(array=True), argv=["script.py"])

    assert calls == [0, 1, ("collect", 2)]
    records = read_task_records(
        SimpleNamespace(tasks_dir=tmp_path / "demo" / "sweeps" / "test-sweep" / "tasks")
    )
    assert [record["task_index"] for record in records] == [0, 1]


def test_task_index_runs_single_task(tmp_path) -> None:
    calls: list[int] = []
    plan = LaunchPlan(
        name="demo",
        task_count=3,
        requires_jax=False,
        run_task=lambda index, context: calls.append(index) or {},
        runs_root=str(tmp_path),
    )

    run_launch_plan(plan, args=_args(task_index=1), argv=["script.py"])

    assert calls == [1]


def test_slurm_array_parent_submits_array_and_collector(tmp_path) -> None:
    commands: list[list[str]] = []

    def fake_runner(command, *, check, capture_output, text):
        commands.append(command)
        job_id = "111" if len(commands) == 1 else "222"
        return subprocess.CompletedProcess(command, 0, stdout=f"{job_id}\n", stderr="")

    plan = LaunchPlan(
        name="demo",
        task_count=4,
        requires_jax=False,
        run_task=lambda index, context: {},
        collect=lambda context: None,
        runs_root=str(tmp_path),
    )

    run_launch_plan(
        plan,
        args=_args(launch="slurm", array=True, array_max_parallel=2),
        argv=["script.py", "--launch", "slurm", "--array"],
        cwd=tmp_path,
        env={},
        runner=fake_runner,
    )

    assert len(commands) == 2
    assert "--array=0-3%2" in commands[0]
    assert any(part == "--dependency=afterany:111" for part in commands[1])
    assert not any(part.startswith("--array=") for part in commands[1])


def test_slurm_parent_submission_failure_exits_without_traceback(tmp_path) -> None:
    def fake_runner(command, *, check, capture_output, text):
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr="sbatch: error: Batch job submission failed",
        )

    plan = LaunchPlan(
        name="demo",
        task_count=4,
        requires_jax=True,
        run_task=lambda index, context: {},
        collect=lambda context: None,
        runs_root=str(tmp_path),
    )

    with pytest.raises(SystemExit) as excinfo:
        run_launch_plan(
            plan,
            args=_args(launch="slurm", array=True),
            argv=["script.py", "--launch", "slurm", "--array"],
            cwd=tmp_path,
            env={},
            runner=fake_runner,
        )

    message = str(excinfo.value)
    assert "Slurm sbatch submission failed with exit code 1." in message
    assert "sbatch stderr:\nsbatch: error: Batch job submission failed" in message


def test_slurm_array_child_runs_only_array_task(tmp_path) -> None:
    calls: list[int] = []
    plan = LaunchPlan(
        name="demo",
        task_count=3,
        requires_jax=False,
        run_task=lambda index, context: calls.append(index) or {},
        runs_root=str(tmp_path),
    )

    run_launch_plan(
        plan,
        args=_args(launch="slurm", no_sbatch=True, array=True),
        argv=["script.py"],
        env={"SLURM_JOB_ID": "99", "SLURM_ARRAY_TASK_ID": "2"},
    )

    assert calls == [2]
