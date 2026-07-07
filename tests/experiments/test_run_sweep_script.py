from __future__ import annotations

from types import SimpleNamespace

from scripts import run_sweep as script


def test_run_sweep_parses_minimal_seed_sweep_args() -> None:
    args = script._parse_args(["fixed_regression_base"])

    assert args.base_preset == "fixed_regression_base"
    assert args.run_seeds == [7]
    assert args.vary == ["theta"]
    assert args.project_name is None
    assert args.launch == "local"


def test_run_sweep_parses_override_grid_json() -> None:
    args = script._parse_args(
        [
            "real_data_glm_base",
            "--override-grid-json",
            '{"compute_backend": ["jax"], "n_grad_samples": [4, 8]}',
        ]
    )

    override_grid, override_list = script._override_inputs(args)

    assert override_list is None
    assert override_grid == {"compute_backend": ["jax"], "n_grad_samples": [4, 8]}
    assert script._sweep_requires_jax(args) is True


def test_run_sweep_execute_passes_generic_arguments(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_seed_sweep(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(project_dir="results/proj", run_results=[object(), object()])

    monkeypatch.setattr(script, "run_seed_sweep", fake_run_seed_sweep)
    args = script._parse_args(
        [
            "real_data_glm_base",
            "--project-name",
            "proj",
            "--run-seeds",
            "7",
            "8",
            "--vary",
            "optimizer",
            "noise",
            "--anchor-seed",
            "7",
            "--fixed-json",
            '{"data": 7}',
            "--override-list-json",
            '[{"_run_name": "a", "n_grad_samples": 4}]',
            "--display-keys",
            "n_grad_samples",
            "--per-seed-plots",
        ]
    )

    result = script._execute_sweep(args)

    assert result.project_dir == "results/proj"
    assert calls == [
        {
            "base_preset": "real_data_glm_base",
            "run_seeds": (7, 8),
            "override_grid": None,
            "override_list": [{"_run_name": "a", "n_grad_samples": 4}],
            "vary": ("optimizer", "noise"),
            "anchor_seed": 7,
            "fixed": {"data": 7},
            "per_seed_plots": True,
            "project_name": "proj",
            "display_keys": ["n_grad_samples"],
        }
    ]


def test_run_sweep_launch_plan_is_single_generic_task() -> None:
    args = script._parse_args(
        [
            "real_data_glm_base",
            "--project-name",
            "generic-proj",
            "--requires-jax",
        ]
    )

    plan = script._build_launch_plan(args)

    assert plan.name == "generic-proj"
    assert plan.task_count == 1
    assert plan.requires_jax is True


def test_run_sweep_task_returns_project_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        script,
        "run_seed_sweep",
        lambda **kwargs: SimpleNamespace(project_dir="results/generic", run_results=[1, 2, 3]),
    )
    args = script._parse_args(["fixed_regression_base"])

    payload = script._run_task(0, SimpleNamespace(), args=args)

    assert payload == {"project_dir": "results/generic", "n_runs": 3}


def test_run_sweep_main_delegates_to_launch_plan(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_run_launch_plan(plan, *, args, argv):
        calls["plan"] = plan
        calls["args"] = args
        calls["argv"] = argv

    monkeypatch.setattr(script, "run_launch_plan", fake_run_launch_plan)

    script.main(["fixed_regression_base", "--no-sbatch"])

    assert calls["plan"].task_count == 1
    assert calls["args"].base_preset == "fixed_regression_base"
    assert "--no-sbatch" in calls["argv"]
