"""Build the canonical, date-free runtime model hierarchy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import pickle
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset_metadata import (  # noqa: E402
    ACCEPTANCE_MODEL_ARTIFACTS,
    LOSS_MODEL_ARTIFACTS,
    REQUIRED_DATASET_COLUMNS,
    DATASET_PATH,
)
from data.loader import _load_pickle, _normalize_artifact  # noqa: E402
from data.monotone_spline_xgb import (  # noqa: E402
    fit_monotone_spline_artifact,
    save_monotone_spline_artifact,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_fold(source: Path, output: Path, fold: int) -> None:
    cv = _load_pickle(source)
    models = cv.get("trained_models", ())
    preprocessors = cv.get("trained_preprocessors", ())
    if not 0 <= fold < len(models):
        raise ValueError(f"Fold {fold} is unavailable in {source}.")
    if fold < len(preprocessors):
        preprocessor = preprocessors[fold]
    elif cv.get("preprocessor") is not None:
        model_features = tuple(cv.get("model_features", ()))
        preprocessor = {
            "preprocessor": cv["preprocessor"],
            "x_feature_cols": cv.get("x_feature_cols", ()),
            "u_cols": ("U",) if "U" in model_features else (),
        }
    else:
        raise ValueError(f"No preprocessor is available for fold {fold} in {source}.")
    runtime = {
        "model": models[fold],
        "preprocessor": preprocessor,
        "model_features": cv.get("model_features", ()),
        "target": cv.get("target"),
        "source_format": "cv_selected_fold",
        "source_fold": fold,
        "source_cv_artifact": source.name,
        "source_cv_sha256": _sha256(source),
        "cv_results": cv.get("cv_results"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(runtime, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _curve_cache_rows(dataset: pd.DataFrame, n_policies: int) -> list[int]:
    required = ["id", *REQUIRED_DATASET_COLUMNS]
    complete = dataset.loc[:, required].notna().all(axis=1)
    in_range = dataset["U"].between(0.0, 0.16)
    pool = dataset.loc[complete & in_range].drop_duplicates("id", keep="first")
    if len(pool) < n_policies:
        raise ValueError(f"Only {len(pool)} complete unique policy profiles are available.")
    return pool.sample(n=n_policies, random_state=0).index.astype(int).tolist()


def _prune(output_root: Path, keep: set[Path]) -> None:
    for path in output_root.rglob("*"):
        if path.is_file() and path not in keep:
            path.unlink()
    for path in sorted(output_root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-xgb-cv", type=Path, required=True)
    parser.add_argument("--loss-xgb-cv", type=Path, required=True)
    parser.add_argument("--acceptance-linear-cv", type=Path, required=True)
    parser.add_argument("--loss-linear-cv", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--curve-cache-size", type=int, default=200)
    parser.add_argument("--prune", action="store_true")
    args = parser.parse_args()

    outputs = {
        ACCEPTANCE_MODEL_ARTIFACTS["linear"]["path"],
        LOSS_MODEL_ARTIFACTS["linear"]["path"],
        ACCEPTANCE_MODEL_ARTIFACTS["xgb"]["path"],
        LOSS_MODEL_ARTIFACTS["xgb"]["path"],
        ACCEPTANCE_MODEL_ARTIFACTS["monotone_spline_xgb"]["path"],
    }
    _select_fold(args.acceptance_linear_cv, ACCEPTANCE_MODEL_ARTIFACTS["linear"]["path"], 0)
    _select_fold(args.loss_linear_cv, LOSS_MODEL_ARTIFACTS["linear"]["path"], 0)
    _select_fold(args.acceptance_xgb_cv, ACCEPTANCE_MODEL_ARTIFACTS["xgb"]["path"], args.fold)
    _select_fold(args.loss_xgb_cv, LOSS_MODEL_ARTIFACTS["xgb"]["path"], args.fold)

    xgb_acceptance = _normalize_artifact(
        _load_pickle(ACCEPTANCE_MODEL_ARTIFACTS["xgb"]["path"]),
        probability_target="acceptance",
    )
    dataset = pd.read_csv(DATASET_PATH, sep=";", dtype={"id": "string"})
    curve_rows = _curve_cache_rows(dataset, args.curve_cache_size)
    curves = fit_monotone_spline_artifact(
        xgb_acceptance,
        dataset,
        curve_rows,
        base_artifact_path=ACCEPTANCE_MODEL_ARTIFACTS["xgb"]["path"],
        source_fold=args.fold,
    )
    save_monotone_spline_artifact(
        curves,
        ACCEPTANCE_MODEL_ARTIFACTS["monotone_spline_xgb"]["path"],
        overwrite=True,
    )
    if args.prune:
        _prune(ACCEPTANCE_MODEL_ARTIFACTS["linear"]["path"].parents[1], outputs)

    for output in sorted(outputs):
        print(output)


if __name__ == "__main__":
    main()
