from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.manifest as manifest_mod
from experiments.manifest import (
    collect_derived_metric_rows,
    collect_manifest_final_rows,
    collect_manifest_outputs,
    parse_experiment_manifest,
    run_manifest_variant,
    variant_complete,
)


def _manifest_payload(**overrides):
    payload = {
        "name": "demo-manifest",
        "objective": {"preset": "synthetic_quadratic_base"},
        "objective_modifications": [],
        "optimizer": {
            "step_rule": "l-bfgs-b",
            "n_grad_samples": 4,
            "t_steps": 2,
            "plot": False,
            "enabled_estimators": ["first_order"],
        },
        "seeds": {"run_seeds": [7, 8], "anchor_seed": 7, "vary": ["theta"]},
        "truth": {"source": "clean_base_objective"},
        "launch": {"mode": "local", "array": "variant"},
        "matrix": {"dimension": [2, 3]},
    }
    payload.update(overrides)
    return payload


def _summary(path: Path, *, theta: list[float], final_value: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run": {"run_dir": str(path.parent / "seeds" / "seed-7")},
                "estimators": {
                    "first_order": {
                        "final_u": None,
                        "final_value": final_value,
                        "runtime_sec": 0.5,
                        "theta": theta,
                        "mean_acceptance": None,
                        "train": {
                            "objective_value": final_value,
                            "objective_sum": final_value,
                            "mean_u": None,
                            "mean_acceptance": None,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_parse_manifest_requires_explicit_orchestration_fields() -> None:
    manifest = parse_experiment_manifest(_manifest_payload())

    assert manifest.name == "demo-manifest"
    assert manifest.base_preset == "synthetic_quadratic_base"
    assert manifest.optimizer["step_rule"] == "l-bfgs-b"
    assert manifest.seeds.run_seeds == (7, 8)
    assert manifest.seeds.anchor_seed == 7
    assert manifest.seeds.vary == ("theta",)
    assert manifest.truth.source == "clean_base_objective"
    assert manifest.launch.mode == "local"
    assert manifest.launch.array == "variant"
    assert [variant.name for variant in manifest.variants] == ["dimension-2", "dimension-3"]
    assert manifest.variants[0].overrides["dimension"] == 2
    assert manifest.variants[0].overrides["objective_modifications"] == []


def test_parse_manifest_rejects_missing_required_truth_seed_optimizer_and_launch() -> None:
    for field in ("truth", "seeds", "optimizer", "launch"):
        payload = _manifest_payload()
        payload.pop(field)
        with pytest.raises(ValueError):
            parse_experiment_manifest(payload)

    payload = _manifest_payload()
    payload["optimizer"] = {"t_steps": 2}
    with pytest.raises(ValueError, match="optimizer.step_rule"):
        parse_experiment_manifest(payload)

    payload = _manifest_payload()
    payload.pop("objective_modifications")
    with pytest.raises(ValueError, match="objective_modifications"):
        parse_experiment_manifest(payload)


def test_matrix_axis_can_supply_labeled_nested_overrides() -> None:
    manifest = parse_experiment_manifest(
        _manifest_payload(
            matrix={
                "noise": [
                    {
                        "label": "no-noise",
                        "value": "none",
                        "overrides": {"objective_modifications": []},
                    },
                    {
                        "label": "homo-0.1",
                        "value": 0.1,
                        "overrides": {
                            "objective_modifications": [
                                {
                                    "type": "noise",
                                    "noise": {
                                        "type": "HomoskedasticGaussianNoise",
                                        "std": 0.1,
                                        "seed": 11,
                                    },
                                }
                            ]
                        },
                    },
                ]
            }
        )
    )

    assert [variant.name for variant in manifest.variants] == ["no-noise", "homo-0.1"]
    assert manifest.variants[0].axes == {"noise": "none"}
    assert manifest.variants[1].overrides["objective_modifications"][0]["type"] == "NoiseModification"


def test_manifest_rejects_non_mapping_matrix() -> None:
    with pytest.raises(ValueError, match="matrix"):
        parse_experiment_manifest(_manifest_payload(matrix=[]))


def test_variant_complete_and_run_skip_existing_seed_summaries(monkeypatch, tmp_path) -> None:
    manifest = parse_experiment_manifest(_manifest_payload(matrix={}))
    variant = manifest.variants[0]
    for seed in manifest.seeds.run_seeds:
        _summary(manifest.variant_dir(variant, tmp_path) / f"summary-seed-{seed}.json", theta=[1.0, 2.0])

    calls: list[object] = []
    monkeypatch.setattr(manifest_mod, "run_seed_sweep", lambda **kwargs: calls.append(kwargs))

    assert variant_complete(manifest, variant, runs_root=tmp_path) is True
    payload = run_manifest_variant(manifest, 0, runs_root=tmp_path)

    assert payload["skipped"] is True
    assert calls == []
    assert (manifest.project_dir(tmp_path) / "EXPERIMENT.md").exists()


def test_collect_manifest_rows_and_summary_truth_metrics(monkeypatch, tmp_path) -> None:
    truth_path = tmp_path / "truth-summary.json"
    _summary(truth_path, theta=[0.0, 0.0], final_value=0.0)
    manifest = parse_experiment_manifest(
        _manifest_payload(
            matrix={},
            seeds={"run_seeds": [7], "anchor_seed": 7, "vary": ["theta"]},
            truth={"source": "summary_json", "path": str(truth_path), "estimator": "first_order"},
        )
    )
    variant = manifest.variants[0]
    _summary(manifest.variant_dir(variant, tmp_path) / "summary-seed-7.json", theta=[3.0, 4.0])
    monkeypatch.setattr(manifest_mod, "_write_seed_grid_plots", lambda *args, **kwargs: None)

    final_rows = collect_manifest_final_rows(manifest, runs_root=tmp_path)
    derived_rows = collect_derived_metric_rows(manifest, runs_root=tmp_path)
    payload = collect_manifest_outputs(manifest, runs_root=tmp_path)

    assert len(final_rows) == 1
    assert final_rows[0]["variant"] == "base"
    assert final_rows[0]["estimator"] == "first_order"
    assert derived_rows[0]["truth_source"] == "summary_json"
    assert derived_rows[0]["theta_l2_gap"] == pytest.approx(5.0)
    assert payload["n_final_rows"] == 1
    assert (manifest.project_dir(tmp_path) / "seed_grid_finals.csv").exists()
    assert (manifest.project_dir(tmp_path) / "derived_metrics.csv").exists()
