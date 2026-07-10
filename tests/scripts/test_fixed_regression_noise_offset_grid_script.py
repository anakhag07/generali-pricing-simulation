from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from experiments.configs import get_config
from experiments.seeds import replicate_seed_setup
from scripts import run_fixed_regression_noise_offset_grid as script


def _reference() -> script.TruthReference:
    objective = get_config("fixed_regression_base").objective
    return script.TruthReference(
        theta=np.asarray([0.4, 0.1, -0.2], dtype=float),
        final_u=0.123,
        final_value=-1.0,
        base_objective=objective,
        anchor_seed=7,
    )


def test_grid_override_lists_cover_noise_by_offset_product() -> None:
    reference = _reference()

    for family in (script.HOMO_FAMILY, script.HETERO_FAMILY):
        override_list = script._build_grid_override_list(family, reference)
        assert len(override_list) == len(family.noise_levels) * len(script.GRID_THETA_OFFSETS)
        expected_names = [
            script._grid_run_name(family, level, offset)
            for level in family.noise_levels
            for offset in script.GRID_THETA_OFFSETS
        ]
        assert [entry["_run_name"] for entry in override_list] == expected_names
        for entry, (level, offset) in zip(
            override_list,
            [(level, offset) for level in family.noise_levels for offset in script.GRID_THETA_OFFSETS],
        ):
            np.testing.assert_allclose(entry["theta0"], reference.theta + float(offset))
            noise = entry["objective"].noise
            if family is script.HOMO_FAMILY:
                assert noise.std == float(level)
            else:
                assert noise.growth == float(level)
                assert noise.base_std == 0.0
                assert noise.u_center == reference.final_u


def test_grid_variant_name_round_trip() -> None:
    for family in (script.HOMO_FAMILY, script.HETERO_FAMILY):
        name = script._grid_run_name(family, 0.25, 0.05)
        assert script._parse_grid_variant(family, name) == (0.25, 0.05)
    assert script._parse_grid_variant(script.HOMO_FAMILY, "theta-offset-0.05") is None
    assert script._parse_grid_variant(script.HOMO_FAMILY, "noise-growth-1__theta-offset-1") is None


def test_task_groups_are_one_per_family_noise_level() -> None:
    groups = script._task_groups(script.FAMILY_GROUPS["all"])

    n_levels = len(script.HOMO_FAMILY.noise_levels) + len(script.HETERO_FAMILY.noise_levels)
    assert len(groups) == n_levels
    projects = {family.project_name for family, _ in groups}
    assert projects == {script.HOMO_PROJECT_NAME, script.HETERO_PROJECT_NAME}
    homo_levels = {level for family, level in groups if family is script.HOMO_FAMILY}
    assert homo_levels == {float(value) for value in script.HOMO_FAMILY.noise_levels}


def test_missing_overrides_require_all_seed_summaries(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GENERALI_RESULTS_ROOT", str(tmp_path))
    project_dir = tmp_path / "proj"
    variant_dir = project_dir / "noise-std-0__theta-offset-0"
    variant_dir.mkdir(parents=True)
    _write_summary(variant_dir / "summary-seed-7.json")
    override = {"_run_name": "noise-std-0__theta-offset-0"}

    assert script._missing_overrides(
        project_name="proj",
        override_list=[override],
        run_seeds=(7, 8),
        required_estimators=script.REQUIRED_ESTIMATORS,
    ) == [override]

    _write_summary(variant_dir / "summary-seed-8.json")
    assert script._missing_overrides(
        project_name="proj",
        override_list=[override],
        run_seeds=(7, 8),
        required_estimators=script.REQUIRED_ESTIMATORS,
    ) == []


def test_run_missing_variant_sweeps_delegates_to_run_sweep(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GENERALI_RESULTS_ROOT", str(tmp_path))
    calls: list[dict[str, object]] = []

    def fake_run_sweep(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run_results=[object(), object()])

    monkeypatch.setattr(script, "run_sweep", fake_run_sweep)

    overrides = [
        {"_run_name": "noise-std-0__theta-offset-0"},
        {"_run_name": "noise-std-0__theta-offset-1"},
    ]

    n_runs = script._run_missing_variant_sweeps(
        project_name="proj",
        override_list=overrides,
        run_seeds=(7, 8),
        anchor_seed=7,
    )

    assert n_runs == 4
    assert len(calls) == 2
    assert calls[0]["base_preset"] == script.BASE_PRESET
    assert calls[0]["run_seeds"] == (7, 8)
    assert calls[0]["override_list"] == [overrides[0]]
    assert calls[0]["vary"] == script.VARY
    assert calls[0]["anchor_seed"] == 7
    assert calls[0]["project_name"] == "proj"


def test_variant_rows_recompute_clean_objective_gap(tmp_path) -> None:
    reference = _reference()
    variant_dir = tmp_path / "noise-std-0__theta-offset-0"
    variant_dir.mkdir()
    seed_setup = replicate_seed_setup(7, 7, vary=(), fixed={})
    config = get_config(
        "fixed_regression_base",
        overrides={
            "seed_setup": seed_setup,
            "n_samples": 6,
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        },
    )
    theta_hat = reference.theta + np.asarray([0.2, -0.1, 0.05], dtype=float)
    summary_path = variant_dir / "summary-seed-7.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "estimators": {
                    "finite_difference": {
                        "theta": theta_hat.tolist(),
                        "final_u": 0.2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rows = script._variant_rows(
        variant_dir,
        script.HOMO_FAMILY,
        noise_level=0.0,
        offset=0.0,
        reference=reference,
        run_seeds=(7,),
    )

    x_train = script._train_x(json.loads(summary_path.read_text(encoding="utf-8")))
    expected_gap = config.objective.value(theta_hat, x_train) - config.objective.value(reference.theta, x_train)
    assert len(rows) == 1
    assert rows[0]["clean_objective_gap"] == expected_gap
    assert rows[0]["theta_distance_to_truth"] == np.linalg.norm(theta_hat - reference.theta)


def test_launch_plan_does_not_compute_truth_reference(monkeypatch) -> None:
    monkeypatch.setattr(
        script,
        "_truth_reference",
        lambda args: (_ for _ in ()).throw(AssertionError("truth should not be computed for plan sizing")),
    )
    args = script._parse_args(["--families", "heteroskedastic"])

    plan = script._build_launch_plan(args, script.FAMILY_GROUPS[args.families])

    assert plan.requires_jax is False
    assert plan.default_launch == "auto"
    assert plan.task_count == len(script.HETERO_FAMILY.noise_levels)


def test_axis_labels_state_offset_and_gap_definitions() -> None:
    assert r"\theta_0 = \theta^{\mathrm{FO}}_{\mathrm{clean}} + \delta\,\mathbf{1}" in script.X_AXIS_LABEL
    assert "every coordinate" in script.X_AXIS_LABEL
    assert r"\|\hat{\theta}_{\mathrm{final}} - \theta^{\mathrm{FO}}_{\mathrm{clean}}\|_2" in script.THETA_DISTANCE_LABEL
    assert "train batch" in script.OBJECTIVE_GAP_LABEL


def _write_summary(path) -> None:
    path.write_text(
        json.dumps(
            {
                "estimators": {
                    "finite_difference": {"theta": [0.0]},
                    "stein_difference": {"theta": [0.0]},
                }
            }
        ),
        encoding="utf-8",
    )
