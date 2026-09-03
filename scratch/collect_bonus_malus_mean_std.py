"""Collect customer-level bonus-malus means and standard deviations.

The saved bonus-malus partial-dependence tables contain exact mean curves but
not customer-level dispersion.  This scratch task preserves those means and
estimates each pointwise customer standard deviation on one deterministic
sample.  The default sample (20,000 eligible rows, seed 0) matches the sample
used by the saved spline partial-dependence analysis.

Example
-------
MPLCONFIGDIR=/tmp/generali-mpl-cache \
  python scratch/collect_bonus_malus_mean_std.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from data.loader import eligible_csv_row_indices, load_model_artifacts, load_x_frame
from scripts.analyze_model_acceptance_features import (
    _predict_acceptance_matrix,
    _predict_loss,
    _spline_weights,
    exact_spline_acceptance_matrix,
)


DEFAULT_RESULTS_ROOT = Path(
    os.environ.get(
        "GENERALI_RESULTS_ROOT",
        Path.home() / "projects" / "generali-pricing" / "results",
    )
)
DEFAULT_ANALYSIS_DIR = (
    DEFAULT_RESULTS_ROOT
    / "model-acceptance-feature-analysis"
    / "sweeps"
    / "20260809_105106"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--fixed-u", type=float, default=0.08)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV; defaults to ANALYSIS_DIR/bonus_malus_mean_std.csv.",
    )
    args = parser.parse_args(argv)
    if args.sample_size <= 1:
        parser.error("--sample-size must be greater than one")
    if args.n_jobs <= 0:
        parser.error("--n-jobs must be positive")
    return args


def _saved_table(analysis_dir: Path, model: str) -> pd.DataFrame:
    path = analysis_dir / f"{model}_bonus_malus_partial_dependence.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required saved mean curve not found: {path}")
    return pd.read_csv(path)


def _set_thread_count(artifact: object, n_jobs: int) -> None:
    model = getattr(artifact, "model")
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=n_jobs)


def _row(
    *,
    model: str,
    target: str,
    rating: float,
    saved_mean: float,
    sample_values: np.ndarray,
    sample_size: int,
    sample_seed: int,
) -> dict[str, object]:
    values = np.asarray(sample_values, dtype=float)
    if values.shape != (sample_size,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {model}/{target} predictions at rating={rating}")
    return {
        "model": model,
        "target": target,
        "bonus_malus_rating": rating,
        "mean": saved_mean,
        "std": float(np.std(values, ddof=0)),
        "sample_mean": float(np.mean(values)),
        "std_n_customers": sample_size,
        "sample_seed": sample_seed,
        "mean_scope": "saved_original_curve",
        "std_scope": "deterministic_customer_sample",
    }


def collect_bonus_malus_mean_std(
    analysis_dir: Path,
    *,
    sample_size: int,
    sample_seed: int,
    fixed_u: float,
    n_jobs: int,
) -> pd.DataFrame:
    """Return saved means plus pointwise customer-SD estimates."""
    analysis_dir = analysis_dir.expanduser().resolve()
    eligible = eligible_csv_row_indices("linear")
    if sample_size > eligible.size:
        raise ValueError(
            f"sample_size={sample_size} exceeds {eligible.size} eligible customers"
        )
    rng = np.random.default_rng(sample_seed)
    selected = np.sort(rng.choice(eligible, size=sample_size, replace=False))
    base_frame = load_x_frame("linear", row_indices=selected)

    saved = {
        model: _saved_table(analysis_dir, model)
        for model in ("glm", "spline", "xgb")
    }
    rows: list[dict[str, object]] = []
    xgb_acceptance = None

    for model, family in (("glm", "linear"), ("xgb", "xgb")):
        acceptance_artifact, claims_artifact = load_model_artifacts(family)
        if model == "xgb":
            xgb_acceptance = acceptance_artifact
        _set_thread_count(acceptance_artifact, n_jobs)
        _set_thread_count(claims_artifact, n_jobs)
        for index, saved_row in saved[model].iterrows():
            rating = float(saved_row["bonus_malus_rating"])
            frame = base_frame.copy()
            frame["X_bonus_malus_rating"] = rating
            acceptance = _predict_acceptance_matrix(
                acceptance_artifact, frame, [fixed_u]
            )[:, 0]
            claims = _predict_loss(claims_artifact, frame)
            rows.append(
                _row(
                    model=model,
                    target="acceptance",
                    rating=rating,
                    saved_mean=float(
                        saved_row["mean_acceptance_probability_at_u_0_08"]
                    ),
                    sample_values=acceptance,
                    sample_size=sample_size,
                    sample_seed=sample_seed,
                )
            )
            rows.append(
                _row(
                    model=model,
                    target="claims",
                    rating=rating,
                    saved_mean=float(saved_row["mean_predicted_loss"]),
                    sample_values=claims,
                    sample_size=sample_size,
                    sample_seed=sample_seed,
                )
            )
            if index % 10 == 0 or index + 1 == len(saved[model]):
                print(f"{model}: {index + 1}/{len(saved[model])}", flush=True)

    xgb_claims_rows = [
        row
        for row in rows
        if row["model"] == "xgb" and row["target"] == "claims"
    ]
    for xgb_row in xgb_claims_rows:
        spline_claims_row = dict(xgb_row)
        spline_claims_row["model"] = "spline"
        rows.append(spline_claims_row)

    if xgb_acceptance is None:
        raise RuntimeError("XGBoost acceptance artifact was not loaded")
    _set_thread_count(xgb_acceptance, n_jobs)
    weights = _spline_weights(eligible)
    for index, saved_row in saved["spline"].iterrows():
        rating = float(saved_row["bonus_malus_rating"])
        frame = base_frame.copy()
        frame["X_bonus_malus_rating"] = rating
        acceptance, failures = exact_spline_acceptance_matrix(
            xgb_acceptance,
            frame,
            [fixed_u],
            weights,
            n_jobs=n_jobs,
        )
        if failures:
            raise RuntimeError(
                f"Spline fitting failed for {failures} customers at rating={rating}"
            )
        row = _row(
            model="spline",
            target="acceptance",
            rating=rating,
            saved_mean=float(saved_row["mean_acceptance_probability_at_u_0_08"]),
            sample_values=acceptance[:, 0],
            sample_size=sample_size,
            sample_seed=sample_seed,
        )
        rows.append(row)
        print(f"spline: {index + 1}/{len(saved['spline'])}", flush=True)

    result = pd.DataFrame(rows).sort_values(
        ["model", "target", "bonus_malus_rating"], ignore_index=True
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    analysis_dir = args.analysis_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else analysis_dir / "bonus_malus_mean_std.csv"
    )
    result = collect_bonus_malus_mean_std(
        analysis_dir,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        fixed_u=args.fixed_u,
        n_jobs=args.n_jobs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} rows to {output}")


if __name__ == "__main__":
    main()
