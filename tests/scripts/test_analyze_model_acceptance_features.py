from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data.dataset_metadata import ACCEPTANCE_STATE_COLS
from data.loader import ModelArtifactBundle
from experiments.launch import LaunchContext
from scripts import analyze_model_acceptance_features as analysis


def _args(**overrides) -> SimpleNamespace:
    defaults = {
        "u_min": 0.0,
        "u_max": 0.16,
        "u_count": 3,
        "chunk_size": 2,
        "histogram_bins": 100,
        "importance_n_rows": 20,
        "sample_seed": 0,
        "permutation_seed": 42,
        "permutation_repeats": 1,
        "spline_importance_group_size": 2,
        "n_jobs": 1,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _context(tmp_path: Path) -> LaunchContext:
    sweep_dir = tmp_path / "results" / "sweeps" / "test"
    return LaunchContext(
        plan_name=analysis.PROJECT_NAME,
        runs_root=tmp_path / "results",
        sweep_id="test",
        sweep_dir=sweep_dir,
        tasks_dir=sweep_dir / "tasks",
        launch_mode="local",
        array=True,
    )


def test_chunk_aggregation_covers_rows_and_reconstructs_statistics(tmp_path: Path) -> None:
    context = _context(tmp_path)
    u_values = np.asarray([0.0, 0.08])
    matrices = [
        np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        np.asarray([[0.5, 0.6]]),
    ]
    payloads = []
    for index, (row_indices, matrix) in enumerate(
        [(np.asarray([1, 3]), matrices[0]), (np.asarray([5]), matrices[1])]
    ):
        path = analysis._chunk_output_path(context, index)
        summaries = {
            model: analysis.summarize_acceptance_matrix(matrix, histogram_bins=100)
            for model in analysis.MODEL_ORDER
        }
        analysis._write_chunk(
            path,
            row_indices=row_indices,
            u_values=u_values,
            summaries=summaries,
        )
        payloads.append(
            {
                "kind": "curve",
                "start": index * 2,
                "output_path": str(path),
                "spline_failures": 0,
            }
        )

    rows, failures = analysis.aggregate_curve_chunks(payloads, np.asarray([1, 3, 5]))

    glm_rows = analysis._model_curve_rows(rows, "glm")
    np.testing.assert_allclose([row["mean"] for row in glm_rows], [0.3, 0.4])
    assert all(row["n_rows"] == 3 for row in glm_rows)
    assert failures == 0


class _DominantAcceptanceModel:
    feature_names_in_ = np.asarray([*ACCEPTANCE_STATE_COLS, "U"])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        logit = 5.0 * frame[ACCEPTANCE_STATE_COLS[0]].to_numpy(float) - frame["U"].to_numpy(float)
        acceptance = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1.0 - acceptance, acceptance])


def test_prediction_sensitivity_ranks_known_dominant_feature_first(monkeypatch) -> None:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {feature: rng.normal(size=500) for feature in ACCEPTANCE_STATE_COLS}
    )
    artifact = ModelArtifactBundle(
        model=_DominantAcceptanceModel(),
        preprocessor=None,
        u_cols=("U",),
        x_feature_cols=tuple(ACCEPTANCE_STATE_COLS),
        probability_target="acceptance",
    )
    monkeypatch.setattr(analysis, "_importance_artifact", lambda model, target: artifact)
    task = analysis.AnalysisTask(
        "importance",
        model="glm",
        target="acceptance",
        repeat=0,
        features=(ACCEPTANCE_STATE_COLS[0], ACCEPTANCE_STATE_COLS[1]),
    )

    rows, failures = analysis._run_standard_importance_task(task, frame, _args())

    scores = {row["feature"]: row["score"] for row in rows}
    assert scores[ACCEPTANCE_STATE_COLS[0]] > scores[ACCEPTANCE_STATE_COLS[1]] * 100
    assert failures == 0


def test_collector_writes_csvs_and_only_acceptance_plots(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.tasks_dir.mkdir(parents=True)
    eligible = np.asarray([1, 3])
    u_values = np.asarray([0.0, 0.08, 0.16])
    path = analysis._chunk_output_path(context, 0)
    matrix = np.asarray([[0.8, 0.7, 0.6], [0.6, 0.5, 0.4]])
    summaries = {
        model: analysis.summarize_acceptance_matrix(
            matrix - 0.02 * index, histogram_bins=100
        )
        for index, model in enumerate(analysis.MODEL_ORDER)
    }
    analysis._write_chunk(
        path,
        row_indices=eligible,
        u_values=u_values,
        summaries=summaries,
    )
    curve_payload = {
        "kind": "curve",
        "start": 0,
        "stop": 2,
        "n_rows": 2,
        "spline_failures": 0,
        "output_path": str(path),
    }
    importance_payload = {
        "kind": "importance",
        "n_rows": 2,
        "spline_failures": 0,
        "rows": [
            {
                "model": model,
                "target": target,
                "feature": ACCEPTANCE_STATE_COLS[0],
                "repeat": 0,
                "score": 0.1,
            }
            for model, target in (
                ("glm", "acceptance"),
                ("xgb", "acceptance"),
                ("spline", "acceptance"),
                ("glm", "loss"),
                ("xgb", "loss"),
            )
        ],
    }
    for index, payload in enumerate((curve_payload, importance_payload)):
        context.task_record_path(index).write_text(
            json.dumps({"task_index": index, "status": "success", "payload": payload}),
            encoding="utf-8",
        )

    analysis._collect(context, args=_args(), eligible=eligible)

    assert (context.sweep_dir / "acceptance_by_u.csv").exists()
    assert (context.sweep_dir / "feature_importance.csv").exists()
    assert (context.sweep_dir / "analysis_config.json").exists()
    assert {path.name for path in context.sweep_dir.glob("*.png")} == {
        "glm_acceptance_by_u.png",
        "xgb_acceptance_by_u.png",
        "spline_acceptance_by_u.png",
        "acceptance_model_comparison.png",
    }


def test_main_builds_launch_plan_from_cli(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(analysis, "_eligible_rows", lambda: np.arange(5))
    monkeypatch.setattr(
        analysis,
        "run_launch_plan",
        lambda plan, **kwargs: captured.update(plan=plan, kwargs=kwargs),
    )

    analysis.main(
        [
            "--launch",
            "local",
            "--chunk-size",
            "2",
            "--importance-n-rows",
            "4",
            "--permutation-repeats",
            "1",
        ]
    )

    assert captured["plan"].task_count == 3 + 4 + 10
    assert captured["plan"].default_array is True
