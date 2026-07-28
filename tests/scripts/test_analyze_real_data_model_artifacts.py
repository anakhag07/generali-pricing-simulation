from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data.feature_processor import FeatureProcessor
from scripts import analyze_real_data_model_artifacts as script


REPO_ROOT = Path(__file__).resolve().parents[2]


def _fitted_preprocessor() -> FeatureProcessor:
    return FeatureProcessor(numeric_cols=["x"], categorical_cols=[]).fit(
        pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    )


def test_dataset_comparison_reports_identity_and_differences() -> None:
    reference = pd.DataFrame(
        {
            "id": [1, 2],
            "dummy_id": [10, 11],
            "U": [0.0, 0.1],
            "is_churn": [0, 1],
            "Y_G_Loss": [10.0, 20.0],
        }
    )
    candidate = reference.copy()
    identical = script.compare_dataset_frames(
        reference,
        candidate,
        reference_sha256="same",
        candidate_sha256="same",
        reference_size=100,
        candidate_size=100,
    )
    assert identical["match"].all()

    candidate.loc[1, "U"] = 0.2
    changed = script.compare_dataset_frames(
        reference,
        candidate,
        reference_sha256="reference",
        candidate_sha256="candidate",
    )
    hash_row = changed[
        (changed["section"] == "file") & (changed["metric"] == "sha256")
    ].iloc[0]
    mean_u = changed[
        (changed["section"] == "numeric_summary")
        & (changed["metric"] == "mean")
        & (changed["column"] == "U")
    ].iloc[0]
    assert not bool(hash_row["match"])
    assert not bool(mean_u["match"])


def test_normalize_artifact_handles_cv_and_selected_best_fold() -> None:
    preprocessor = _fitted_preprocessor()
    model_zero = SimpleNamespace()
    model_one = SimpleNamespace()
    cv = {
        "trained_models": [model_zero, model_one],
        "trained_preprocessors": [
            {"preprocessor": preprocessor, "x_feature_cols": ["x"], "u_cols": ["U"]},
            {"preprocessor": preprocessor, "x_feature_cols": ["x"], "u_cols": ["U"]},
        ],
        "model_features": ["x", "U"],
    }
    selected = {
        "model": model_one,
        "preprocessor": {
            "preprocessor": preprocessor,
            "x_feature_cols": ["x"],
            "u_cols": ["U"],
        },
        "best_fold": 4,
        "model_features": ["x", "U"],
    }

    cv_view = script.normalize_artifact(
        cv,
        name="cv",
        family="xgb",
        role="acceptance",
        path="cv.pkl",
    )
    selected_view = script.normalize_artifact(
        selected,
        name="selected",
        family="xgb",
        role="acceptance",
        path="selected.pkl",
    )

    assert cv_view.source_format == "cv_first_fold"
    assert cv_view.model is model_zero
    assert cv_view.selected_fold == 0
    assert selected_view.source_format == "selected_best_fold"
    assert selected_view.model is model_one
    assert selected_view.selected_fold == 4


def test_pairwise_metrics_and_action_summary_are_exact() -> None:
    artifacts = [
        SimpleNamespace(name="a", role="acceptance"),
        SimpleNamespace(name="b", role="acceptance"),
        SimpleNamespace(name="loss", role="loss"),
    ]
    predictions = {
        "a": np.asarray([0.2, 0.4]),
        "b": np.asarray([0.1, 0.5]),
        "loss": np.asarray([10.0, 20.0]),
    }
    pairwise = script.pairwise_prediction_metrics(artifacts, predictions)
    assert pairwise.shape[0] == 1
    assert pairwise.iloc[0]["mae"] == pytest.approx(0.1)
    assert pairwise.iloc[0]["rmse"] == pytest.approx(0.1)

    summary = script.summarize_action_predictions(
        model_name="smooth",
        cohort="covered",
        action_grid=np.asarray([0.0, 0.1]),
        predictions=np.asarray([[0.9, 0.8], [0.7, 0.6]]),
    )
    np.testing.assert_allclose(summary["mean_acceptance"], [0.8, 0.7])


def test_sigmoid_parameter_extraction_and_portable_evaluation() -> None:
    wrapper = SimpleNamespace(
        _function_name="sigmoid_with_shift",
        _curves={
            "b": SimpleNamespace(params=np.asarray([10.0, 0.1, 0.0])),
            "a": SimpleNamespace(params=np.asarray([20.0, 0.2, 0.1])),
        },
    )
    policy_ids, parameters = script.extract_sigmoid_parameters(wrapper)
    predictions = script.sigmoid_acceptance_matrix(
        parameters, np.asarray([0.0, 0.1, 0.2])
    )

    assert policy_ids.tolist() == ["a", "b"]
    assert predictions.shape == (2, 3)
    assert np.all(np.diff(predictions, axis=1) <= 0.0)


@pytest.mark.parametrize(
    "wrapper",
    [
        SimpleNamespace(_function_name="other", _curves={"a": object()}),
        SimpleNamespace(_function_name="sigmoid_with_shift", _curves={}),
        SimpleNamespace(
            _function_name="sigmoid_with_shift",
            _curves={"a": SimpleNamespace(params=np.asarray([1.0, 2.0]))},
        ),
    ],
)
def test_sigmoid_parameter_extraction_rejects_unsupported_artifacts(wrapper) -> None:
    with pytest.raises(ValueError):
        script.extract_sigmoid_parameters(wrapper)


def test_write_outputs_creates_complete_report_bundle(tmp_path: Path) -> None:
    dataset_summary = pd.DataFrame(
        [
            {
                "section": "file",
                "metric": "sha256",
                "column": "",
                "reference": '"same"',
                "candidate": '"same"',
                "match": True,
            },
            {
                "section": "eligibility",
                "metric": "eligible_rows",
                "column": "",
                "reference": "100",
                "candidate": "100",
                "match": True,
            },
        ]
    )
    inventory = pd.DataFrame(
        [
            {
                "artifact": "candidate_xgb_acceptance",
                "role": "acceptance",
                "family": "xgb",
                "source_format": "selected_best_fold",
                "model_class": "FakeClassifier",
                "selected_fold_zero_based": 0,
                "size_bytes": 1,
            }
        ]
    )
    cv_metrics = pd.DataFrame(
        [
            {
                "artifact": "candidate_xgb_acceptance",
                "fold": 1,
                "roc_auc": 0.7,
                "mae": np.nan,
                "n_train": 80,
                "n_val": 20,
            }
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "artifact": "candidate_xgb_acceptance",
                "family": "xgb",
                "role": "acceptance",
                "mean_prediction": 0.8,
                "roc_auc": 0.7,
                "log_loss": 0.4,
                "brier_score": 0.1,
                "mae": np.nan,
                "rmse": np.nan,
            },
            {
                "artifact": "reference_xgb_loss",
                "family": "xgb",
                "role": "loss",
                "mean_prediction": 100.0,
                "roc_auc": np.nan,
                "log_loss": np.nan,
                "brier_score": np.nan,
                "mae": 10.0,
                "rmse": 12.0,
            },
            {
                "artifact": "candidate_xgb_loss",
                "family": "xgb",
                "role": "loss",
                "mean_prediction": 102.0,
                "roc_auc": np.nan,
                "log_loss": np.nan,
                "brier_score": np.nan,
                "mae": 11.0,
                "rmse": 13.0,
            },
        ]
    )
    pairwise = pd.DataFrame(
        [
            {
                "role": "loss",
                "left_artifact": "reference_xgb_loss",
                "right_artifact": "candidate_xgb_loss",
                "mean_delta": -2.0,
                "mae": 2.0,
                "rmse": 2.0,
                "correlation": 1.0,
            }
        ]
    )
    action_grid = pd.concat(
        [
            script.summarize_action_predictions(
                model_name="candidate_xgb_acceptance",
                cohort="deterministic_common_sample",
                action_grid=np.asarray([0.0, 0.1]),
                predictions=np.asarray([[0.9, 0.8], [0.8, 0.7]]),
            ),
            script.summarize_action_predictions(
                model_name="candidate_xgb_smoothed",
                cohort="candidate_200_covered_ids",
                action_grid=np.asarray([0.0, 0.1]),
                predictions=np.asarray([[0.95, 0.85], [0.85, 0.75]]),
            ),
        ],
        ignore_index=True,
    )
    smoothing_coverage = pd.DataFrame(
        [
            {
                "artifact": "candidate_xgb_smoothed",
                "metric": metric,
                "value": value,
                "detail": "",
            }
            for metric, value in {
                "covered_policy_ids": 200,
                "coverage_fraction": 0.001,
                "current_cohort_id_intersection": 0,
                "smooth_vs_embedded_raw_mae": 0.02,
                "embedded_raw_same_booster_as_saved_candidate": False,
                "legacy_numpy_scalar_conversion_breaks": True,
            }.items()
        ]
    )
    predictions = {
        "candidate_xgb_acceptance": np.asarray([0.8, 0.9]),
        "reference_xgb_loss": np.asarray([90.0, 110.0]),
        "candidate_xgb_loss": np.asarray([92.0, 112.0]),
    }
    smoothing_matrices = {
        "candidate_smooth_minus_embedded_raw": np.asarray(
            [[-0.01, 0.02], [0.01, -0.02]]
        )
    }

    script.write_outputs(
        output_dir=tmp_path,
        dataset_summary=dataset_summary,
        inventory=inventory,
        cv_metrics=cv_metrics,
        metrics=metrics,
        pairwise=pairwise,
        action_grid=action_grid,
        smoothing_coverage=smoothing_coverage,
        predictions=predictions,
        smoothing_matrices=smoothing_matrices,
        sample_size=2,
        seed=7,
    )

    expected = {
        "dataset_summary.csv",
        "artifact_inventory.csv",
        "cv_metrics.csv",
        "prediction_metrics.csv",
        "pairwise_prediction_deltas.csv",
        "acceptance_action_grid.csv",
        "smoothing_coverage.csv",
        "acceptance_model_comparison.png",
        "loss_model_comparison.png",
        "smoothing_model_comparison.png",
        "eda_summary.md",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    report = (tmp_path / "eda_summary.md").read_text(encoding="utf-8")
    assert "Integration handoff" in report
    assert "No runtime integration was performed" in report


def _candidate_artifact_dir() -> Path | None:
    candidates = (
        REPO_ROOT.parent / "model_processing" / "artifacts",
        REPO_ROOT.parents[1] / "model_processing" / "artifacts",
    )
    return next((path for path in candidates if path.is_dir()), None)


@pytest.mark.skipif(
    _candidate_artifact_dir() is None,
    reason="candidate model_processing artifacts are unavailable",
)
def test_real_candidate_artifacts_and_smoothing_ids_smoke() -> None:
    artifact_dir = _candidate_artifact_dir()
    assert artifact_dir is not None
    acceptance = script.load_artifact_view(
        artifact_dir / "acceptance_model_xgb.pkl",
        name="candidate_xgb_acceptance",
        family="xgb",
        role="acceptance",
    )
    loss = script.load_artifact_view(
        artifact_dir / "financial_loss_model_xgb.pkl",
        name="candidate_xgb_loss",
        family="xgb",
        role="loss",
    )
    smoothing = script._load_pickle(
        artifact_dir / "acceptance_smoothing_wrapper.pkl"
    )
    policy_ids, parameters = script.extract_sigmoid_parameters(smoothing)

    assert acceptance.source_format == "selected_best_fold"
    assert loss.source_format == "selected_best_fold"
    assert policy_ids.size == 200
    assert parameters.shape == (200, 3)

    id_frame = pd.read_csv(
        script.DATASET_PATH,
        sep=";",
        usecols=["id"],
        dtype={"id": "string"},
    )
    counts = id_frame["id"].astype(str).value_counts()
    assert all(counts.get(policy_id, 0) == 1 for policy_id in policy_ids)
