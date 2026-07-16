from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import experiments.manifest as manifest
from experiments.config import CorrectnessSpec
from objective.noise import HeteroskedasticGaussianNoise, NoisyObjective
from objective.objectives import BiasedObjective, UpperSupportHingeBias


def test_plan_manifest_sweep_expands_axes_and_wraps_objective() -> None:
    payload = {
        "base_preset": "planted_logistic_base",
        "defaults": {
            "n_samples": 8,
            "t_steps": 1,
            "enabled_estimators": ["finite_difference"],
            "correctness": {"gradient_source": "denoised_exact"},
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        },
        "matrix": {
            "step_rule": ["l-bfgs-b", "optax-adam"],
            "noise": [{"label": "hetero", "kind": "heteroskedastic", "growth": 0.25}],
            "bias": [
                {
                    "label": "hinge",
                    "kind": "upper_support_hinge",
                    "lambda_bias": 0.05,
                    "support_radius": 0.1,
                }
            ],
        },
        "run_name_template": "{step_rule}__{noise.label}__{bias.label}",
    }

    variants, _ = manifest.plan_manifest_sweep(payload)

    assert [variant.name for variant in variants] == [
        "l-bfgs-b__hetero__hinge",
        "optax-adam__hetero__hinge",
    ]
    first = variants[0].overrides
    assert first["step_rule"] == "l-bfgs-b"
    assert first["enabled_estimators"] == ("finite_difference",)
    assert isinstance(first["correctness"], CorrectnessSpec)
    assert first["correctness"].gradient_source == "denoised_exact"

    objective = first["objective"]
    assert isinstance(objective, NoisyObjective)
    assert isinstance(objective.noise, HeteroskedasticGaussianNoise)
    assert objective.noise.growth == pytest.approx(0.25)
    assert isinstance(objective.base_objective, BiasedObjective)
    assert isinstance(objective.base_objective.bias, UpperSupportHingeBias)
    assert objective.base_objective.bias.lambda_bias == pytest.approx(0.05)
    assert objective.base_objective.bias.support_radius == pytest.approx(0.1)
    assert objective.noise.u_center == pytest.approx(objective.base_objective.optimal_u())


def test_manifest_axis_override_blocks_support_data_ladders() -> None:
    payload = {
        "base_preset": "planted_logistic_base",
        "defaults": {"plot": False, "verbose": False, "wandb_enabled": False},
        "matrix": {
            "data_ladder": [
                {"label": "tiny", "overrides": {"n_samples": 8}},
                {"label": "small", "overrides": {"n_samples": 16}},
            ],
        },
    }

    variants, _ = manifest.plan_manifest_sweep(payload)

    assert [variant.name for variant in variants] == ["tiny", "small"]
    assert [variant.overrides["n_samples"] for variant in variants] == [8, 16]


def test_completed_variant_names_requires_every_seed_summary(tmp_path) -> None:
    variants = [
        manifest.ManifestVariant(name="variant-a", axes={}, overrides={}),
        manifest.ManifestVariant(name="variant-b", axes={}, overrides={}),
    ]
    _write_summary(tmp_path / "variant-a" / "summary-seed-7.json")
    _write_summary(tmp_path / "variant-b" / "summary-seed-7.json")
    _write_summary(tmp_path / "variant-b" / "summary-seed-8.json")

    completed = manifest.completed_variant_names(
        project_dir=tmp_path,
        variants=variants,
        run_seeds=(7, 8),
        required_estimators=("finite_difference", "stein_difference"),
    )

    assert completed == ["variant-b"]


def test_run_manifest_sweep_skips_completed_variants(tmp_path, monkeypatch) -> None:
    payload = {
        "name": "skip-demo",
        "base_preset": "planted_logistic_base",
        "project_name": "skip-demo",
        "run_seeds": [7, 8],
        "completion": {"required_estimators": ["finite_difference"]},
        "defaults": {
            "n_samples": 8,
            "t_steps": 1,
            "enabled_estimators": ["finite_difference"],
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        },
        "matrix": {"sigma": [0.1]},
    }
    variant_dir = tmp_path / "skip-demo" / "sigma-0.1"
    _write_summary(variant_dir / "summary-seed-7.json", estimators=("finite_difference",))
    _write_summary(variant_dir / "summary-seed-8.json", estimators=("finite_difference",))
    calls: list[dict[str, object]] = []

    def fake_run_sweep(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run_results=[object()])

    monkeypatch.setattr(manifest, "run_sweep", fake_run_sweep)

    result = manifest.run_manifest_sweep(payload, runs_root=tmp_path)

    assert calls == []
    assert result.skipped_variants == ["sigma-0.1"]
    assert result.executed_runs == 0


def test_run_manifest_sweep_runs_only_missing_variants(tmp_path, monkeypatch) -> None:
    payload = {
        "name": "partial-demo",
        "base_preset": "planted_logistic_base",
        "project_name": "partial-demo",
        "run_seeds": [7],
        "completion": {"required_estimators": ["finite_difference"]},
        "defaults": {
            "n_samples": 8,
            "t_steps": 1,
            "enabled_estimators": ["finite_difference"],
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        },
        "matrix": {"sigma": [0.1, 0.2]},
    }
    _write_summary(
        tmp_path / "partial-demo" / "sigma-0.1" / "summary-seed-7.json",
        estimators=("finite_difference",),
    )
    calls: list[dict[str, object]] = []

    def fake_run_sweep(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run_results=[object(), object()])

    monkeypatch.setattr(manifest, "run_sweep", fake_run_sweep)

    result = manifest.run_manifest_sweep(payload, runs_root=tmp_path)

    assert result.skipped_variants == ["sigma-0.1"]
    assert result.executed_runs == 2
    assert len(calls) == 1
    assert calls[0]["project_name"] == "partial-demo"
    assert [entry["_run_name"] for entry in calls[0]["override_list"]] == ["sigma-0.2"]


def test_manifest_requires_jax_detects_default_and_axis_overrides() -> None:
    assert manifest.manifest_requires_jax(
        {"base_preset": "real_data_glm_base", "defaults": {"compute_backend": "jax"}}
    )
    assert manifest.manifest_requires_jax(
        {
            "base_preset": "real_data_glm_base",
            "matrix": {
                "data_ladder": [
                    {"label": "cpu", "overrides": {"compute_backend": "numpy"}},
                    {"label": "gpu", "overrides": {"compute_backend": "jax"}},
                ]
            },
        }
    )


def test_run_experiment_manifest_accepts_json_path(tmp_path, monkeypatch) -> None:
    payload = {
        "name": "path-demo",
        "base_preset": "planted_logistic_base",
        "defaults": {"plot": False, "verbose": False, "wandb_enabled": False},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run_manifest_sweep(sweep, *, dry_run, runs_root):
        calls.append({"sweep": sweep, "dry_run": dry_run, "runs_root": runs_root})
        return manifest.ManifestSweepResult(
            name="path-demo",
            base_preset="planted_logistic_base",
            project_dir=tmp_path,
            variants=[],
            skipped_variants=[],
            sweep_results=[],
            dry_run=dry_run,
        )

    monkeypatch.setattr(manifest, "run_manifest_sweep", fake_run_manifest_sweep)

    result = manifest.run_experiment_manifest(manifest_path, dry_run=True, runs_root=tmp_path)

    assert result.path == manifest_path
    assert len(result.sweeps) == 1
    assert calls[0]["sweep"]["base_preset"] == "planted_logistic_base"
    assert calls[0]["dry_run"] is True


def _write_summary(path, estimators=("finite_difference", "stein_difference")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"estimators": {name: {"theta": [0.0]} for name in estimators}}),
        encoding="utf-8",
    )
