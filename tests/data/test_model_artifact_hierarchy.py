"""Contract for the canonical date-free model hierarchy."""

from pathlib import Path

from data.dataset_metadata import ACCEPTANCE_MODEL_ARTIFACTS, DATA_DIR, LOSS_MODEL_ARTIFACTS
from data.loader import _load_pickle
from data.monotone_spline_xgb import load_monotone_spline_artifact


MODEL_DIR = DATA_DIR / "models"


def test_model_tree_contains_only_three_runtime_families() -> None:
    assert {path.name for path in MODEL_DIR.iterdir()} == {
        "linear",
        "xgb",
        "monotone-spline-xgb",
    }
    assert {path.name for path in (MODEL_DIR / "linear").iterdir()} == {
        "acceptance.pkl",
        "loss.pkl",
    }
    assert {path.name for path in (MODEL_DIR / "xgb").iterdir()} == {
        "acceptance.pkl",
        "loss.pkl",
    }
    assert {path.name for path in (MODEL_DIR / "monotone-spline-xgb").iterdir()} == {
        "acceptance-curves.npz"
    }
    assert not (DATA_DIR / "model_sources").exists()


def test_xgb_runtime_models_are_fold_zero_exports() -> None:
    acceptance = _load_pickle(ACCEPTANCE_MODEL_ARTIFACTS["xgb"]["path"])
    loss = _load_pickle(LOSS_MODEL_ARTIFACTS["xgb"]["path"])
    assert acceptance["source_fold"] == 0
    assert loss["source_fold"] == 0
    assert acceptance["source_cv_artifact"] == "acceptance_model_xgb_cv_20260723_101800.pkl"
    assert loss["source_cv_artifact"] == "financial_loss_model_xgb_cv_20260706_160647.pkl"


def test_monotone_wrapper_is_derived_from_canonical_xgb() -> None:
    import hashlib

    base_path = ACCEPTANCE_MODEL_ARTIFACTS["xgb"]["path"]
    curves = load_monotone_spline_artifact(
        ACCEPTANCE_MODEL_ARTIFACTS["monotone_spline_xgb"]["path"]
    )
    assert curves.source_fold == 0
    assert curves.base_artifact_sha256 == hashlib.sha256(base_path.read_bytes()).hexdigest()
    assert curves.policy_ids.size == 200
