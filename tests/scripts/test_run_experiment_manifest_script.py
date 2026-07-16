from __future__ import annotations

import json
from types import SimpleNamespace

from scripts import run_experiment_manifest as script


def test_run_experiment_manifest_parses_args() -> None:
    args = script._parse_args(["experiments.json", "--dry-run", "--runs-root", "/tmp/results"])

    assert args.manifest == "experiments.json"
    assert args.dry_run is True
    assert args.runs_root == "/tmp/results"
    assert args.launch == "local"


def test_launch_plan_uses_manifest_name_and_jax_requirement(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "name": "manifest-demo",
                "base_preset": "real_data_glm_base",
                "defaults": {"compute_backend": "jax"},
            }
        ),
        encoding="utf-8",
    )
    args = script._parse_args([str(path)])

    plan = script._build_launch_plan(args)

    assert plan.name == "manifest-demo"
    assert plan.task_count == 1
    assert plan.requires_jax is True


def test_run_task_returns_manifest_payload(monkeypatch, tmp_path) -> None:
    result = SimpleNamespace(
        sweeps=[
            SimpleNamespace(
                variants=[object(), object()],
                skipped_variants=["a"],
                executed_runs=3,
                dry_run=False,
                name="demo",
                project_dir=tmp_path,
            )
        ],
        executed_runs=3,
    )
    monkeypatch.setattr(script, "run_experiment_manifest", lambda *args, **kwargs: result)
    args = script._parse_args(["manifest.json", "--runs-root", str(tmp_path)])

    payload = script._run_task(0, SimpleNamespace(), args=args)

    assert payload == {
        "manifest": "manifest.json",
        "dry_run": False,
        "n_sweeps": 1,
        "n_variants": 2,
        "n_skipped_variants": 1,
        "n_executed_runs": 3,
    }


def test_main_delegates_to_launch_plan(monkeypatch, tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"name": "demo", "base_preset": "planted_logistic_base"}),
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    def fake_run_launch_plan(plan, *, args, argv):
        calls["plan"] = plan
        calls["args"] = args
        calls["argv"] = argv

    monkeypatch.setattr(script, "run_launch_plan", fake_run_launch_plan)

    script.main([str(manifest_path), "--dry-run"])

    assert calls["plan"].name == "demo"
    assert calls["args"].dry_run is True
    assert "--dry-run" in calls["argv"]
