from __future__ import annotations

import json

from scripts import run_experiment_manifest as script


def _manifest_file(tmp_path, *, launch=None, matrix=None):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "name": "manifest-launch",
                "objective": {"preset": "synthetic_quadratic_base"},
                "objective_modifications": [],
                "optimizer": {
                    "step_rule": "l-bfgs-b",
                    "n_grad_samples": 4,
                    "t_steps": 2,
                    "plot": False,
                    "enabled_estimators": ["first_order"],
                },
                "seeds": {"run_seeds": [7], "anchor_seed": 7, "vary": ["theta"]},
                "truth": {"source": "clean_base_objective"},
                "launch": launch or {"mode": "slurm", "array": "variant", "array_max_parallel": 2},
                "matrix": matrix or {"dimension": [2, 3]},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_parse_args_leaves_launch_defaults_to_manifest(tmp_path) -> None:
    manifest_path = _manifest_file(tmp_path)

    args = script._parse_args([str(manifest_path)])

    assert args.manifest == str(manifest_path)
    assert args.launch is None
    assert args.array is None


def test_build_launch_plan_uses_manifest_array_shape(tmp_path) -> None:
    manifest_path = _manifest_file(tmp_path)
    args = script._parse_args([str(manifest_path), "--runs-root", str(tmp_path / "runs")])
    manifest = script.load_experiment_manifest(manifest_path)
    script._apply_manifest_launch_defaults(args, manifest)

    plan = script._build_launch_plan(args, manifest)

    assert plan.name == "manifest-launch"
    assert plan.task_count == 2
    assert plan.default_launch == "slurm"
    assert plan.default_array is True
    assert plan.runs_root == str(tmp_path / "runs")
    assert args.array_max_parallel == 2


def test_build_launch_plan_serial_manifest_is_single_task(tmp_path) -> None:
    manifest_path = _manifest_file(
        tmp_path,
        launch={"mode": "local", "array": "none"},
        matrix={"dimension": [2, 3]},
    )
    args = script._parse_args([str(manifest_path)])
    manifest = script.load_experiment_manifest(manifest_path)

    plan = script._build_launch_plan(args, manifest)

    assert plan.task_count == 1
    assert plan.default_launch == "local"
    assert plan.default_array is False


def test_main_delegates_to_shared_launcher(monkeypatch, tmp_path) -> None:
    manifest_path = _manifest_file(tmp_path)
    calls: dict[str, object] = {}

    def fake_run_launch_plan(plan, *, args, argv):
        calls["plan"] = plan
        calls["args"] = args
        calls["argv"] = argv

    monkeypatch.setattr(script, "run_launch_plan", fake_run_launch_plan)

    script.main([str(manifest_path), "--force"])

    assert calls["plan"].task_count == 2
    assert calls["args"].force is True
    assert str(manifest_path) in calls["argv"]
