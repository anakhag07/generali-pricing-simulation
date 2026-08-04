"""Contracts keeping runtime artifacts distinct from conversion sources."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parents[2] / "src" / "data"
MODEL_DIR = DATA_DIR / "models"
SOURCE_DIR = DATA_DIR / "model_sources" / "acceptance"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_runtime_model_tree_contains_only_registered_families() -> None:
    assert {path.name for path in MODEL_DIR.iterdir() if path.is_dir()} == {
        "linear",
        "xgb",
        "xgb_logit_spline",
        "xgb_monotone_spline",
    }
    assert not list(MODEL_DIR.glob("*.pkl"))
    assert not (MODEL_DIR / "spline_acceptance").exists()
    assert not (MODEL_DIR / "xgb_sigmoid").exists()


def test_conversion_sources_are_isolated_and_named_by_artifact() -> None:
    assert {path.name for path in SOURCE_DIR.iterdir() if path.is_file()} == {
        "xgb_logit_spline_20260706.source.pkl",
        "xgb_monotone_spline_20260728.source.pkl",
    }


def test_registry_paths_are_unique_and_runtime_artifacts_are_not_duplicates() -> None:
    from data.dataset_metadata import ACCEPTANCE_MODEL_ARTIFACTS, LOSS_MODEL_ARTIFACTS

    paths = [
        spec["path"]
        for registry in (ACCEPTANCE_MODEL_ARTIFACTS, LOSS_MODEL_ARTIFACTS)
        for spec in registry.values()
    ]
    assert len(paths) == len(set(paths))
    hashes = [_sha256(path) for path in paths]
    assert len(hashes) == len(set(hashes))


def test_portable_curve_provenance_matches_archived_sources() -> None:
    pairs = {
        MODEL_DIR / "xgb_logit_spline" / "acceptance_xgb_logit_spline_20260706_112929.npz": (
            SOURCE_DIR / "xgb_logit_spline_20260706.source.pkl"
        ),
        MODEL_DIR / "xgb_monotone_spline" / "acceptance_xgb_monotone_spline_20260728.npz": (
            SOURCE_DIR / "xgb_monotone_spline_20260728.source.pkl"
        ),
    }
    for runtime_path, source_path in pairs.items():
        with np.load(runtime_path, allow_pickle=False) as artifact:
            assert str(artifact["source_sha256"]) == _sha256(source_path)
