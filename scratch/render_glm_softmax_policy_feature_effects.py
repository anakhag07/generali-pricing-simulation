"""Render all-customer feature effects for a saved GLM softmax policy.

For each selected raw customer feature, this task fixes that feature to every
value on the configured grid while retaining every other customer covariate.
It replays the saved policy for every eligible customer, stores the pointwise
population mean and standard deviation, and renders the results as vector PDFs.

The saved policy is linear before its bounded sigmoid.  The task verifies that
each requested raw feature has a constant logit slope through the saved
preprocessing pipeline, then uses that exact slope to evaluate the complete
customer-by-grid collection efficiently.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Sequence
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.special import expit

from experiments.policy_artifacts import PolicyArtifact, load_policy_artifact


DEFAULT_PROJECT_ROOT = Path.home() / "projects" / "generali-pricing"
DEFAULT_POLICY_ARTIFACT = (
    DEFAULT_PROJECT_ROOT
    / "scratch-results"
    / "real_data_glm_base__20260706_124627"
    / "policies"
    / "first_order"
    / "policy.json"
)
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get("GENERALI_RESULTS_ROOT", DEFAULT_PROJECT_ROOT / "results")
)
DEFAULT_ANALYSIS_DIR = (
    DEFAULT_RESULTS_ROOT
    / "policy-feature-analysis"
    / "glm-softmax-first-order-20260706_124627"
)


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    column: str
    label: str
    grid: np.ndarray


FEATURE_SPECS = (
    FeatureSpec(
        key="bonus_malus_rating",
        column="X_bonus_malus_rating",
        label="Bonus-Malus Rating",
        grid=np.linspace(45.0, 100.0, 101),
    ),
    FeatureSpec(
        key="vehicle_age",
        column="X_vehicle_age",
        label="Vehicle Age",
        grid=np.arange(0.0, 38.0, 1.0),
    ),
    FeatureSpec(
        key="customer_age",
        column="X_age",
        label="Customer Age",
        grid=np.arange(18.0, 87.0, 1.0),
    ),
    FeatureSpec(
        key="policy_tenure",
        column="X_policy_tenure",
        label="Policy Tenure",
        grid=np.arange(0.0, 21.0, 1.0),
    ),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-artifact",
        type=Path,
        default=DEFAULT_POLICY_ARTIFACT,
    )
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="PDF destination; defaults to ANALYSIS_DIR/plots.",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute policy effects even when the collected CSV exists.",
    )
    args = parser.parse_args(argv)
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


def _install_numpy_pickle_compatibility() -> None:
    """Expose NumPy 1.x aliases used by the NumPy 2.x artifact pickle."""
    import numpy.core.numeric as numeric

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", numeric)


def _load_verified_artifact(path: Path) -> tuple[PolicyArtifact, object]:
    _install_numpy_pickle_compatibility()
    artifact = load_policy_artifact(path)
    if artifact.estimator != "first_order":
        raise ValueError(
            f"Expected the first_order policy artifact, got {artifact.estimator!r}"
        )
    if artifact.policy_head.type != "SoftmaxPolicy":
        raise ValueError(
            f"Expected SoftmaxPolicy, got {artifact.policy_head.type!r}"
        )
    if not np.isclose(artifact.policy_head.action_low, -0.1) or not np.isclose(
        artifact.policy_head.action_high, 0.2
    ):
        raise ValueError(
            "Expected policy bounds [-0.1, 0.2], got "
            f"[{artifact.policy_head.action_low}, {artifact.policy_head.action_high}]"
        )
    if artifact.objective.model_type not in {"glm", "linear"}:
        raise ValueError(
            f"Expected a GLM/linear objective, got {artifact.objective.model_type!r}"
        )
    if artifact.data_binding.selected_row_indices.size != 715_023:
        raise ValueError(
            "Expected all 715,023 complete eligible rows, got "
            f"{artifact.data_binding.selected_row_indices.size:,}"
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
        objective = artifact.build_objective()
    return artifact, objective


def _policy_logits(
    objective: object,
    theta: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    acceptance_model = getattr(objective, "acceptance_model")
    artifact_preprocessor = getattr(acceptance_model, "preprocessor")
    feature_columns = tuple(getattr(acceptance_model, "x_feature_cols"))
    state = frame.loc[:, list(feature_columns)].copy()
    base_features = np.asarray(artifact_preprocessor.transform(state), dtype=float)
    policy_preprocessor = getattr(objective, "policy_preprocessor")
    policy_features = np.asarray(
        policy_preprocessor.transform(base_features),
        dtype=float,
    )
    logits = float(theta[0]) + policy_features @ np.asarray(theta[1:], dtype=float)
    if logits.shape != (len(frame),) or not np.isfinite(logits).all():
        raise ValueError("Saved policy produced invalid logits")
    return logits


def _verified_raw_slopes(
    objective: object,
    theta: np.ndarray,
    base_frame: pd.DataFrame,
) -> dict[str, float]:
    probe = base_frame.iloc[:256].copy()
    base_logits = _policy_logits(objective, theta, probe)
    slopes: dict[str, float] = {}
    for spec in FEATURE_SPECS:
        low_frame = probe.copy()
        high_frame = probe.copy()
        low_frame[spec.column] = float(spec.grid[0])
        high_frame[spec.column] = float(spec.grid[-1])
        low_logits = _policy_logits(objective, theta, low_frame)
        high_logits = _policy_logits(objective, theta, high_frame)
        row_slopes = (high_logits - low_logits) / float(spec.grid[-1] - spec.grid[0])
        slope = float(np.mean(row_slopes))
        if not np.allclose(row_slopes, slope, rtol=1e-10, atol=1e-10):
            raise ValueError(f"{spec.label} does not have a constant policy-logit slope")

        midpoint = float(spec.grid[len(spec.grid) // 2])
        midpoint_frame = probe.copy()
        midpoint_frame[spec.column] = midpoint
        actual = _policy_logits(objective, theta, midpoint_frame)
        reconstructed = base_logits + slope * (
            midpoint - probe[spec.column].to_numpy(dtype=float)
        )
        if not np.allclose(actual, reconstructed, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Failed exact policy-logit replay for {spec.label}")
        slopes[spec.key] = slope
    return slopes


def _empty_stats() -> dict[str, dict[str, np.ndarray]]:
    return {
        spec.key: {
            "sum": np.zeros(spec.grid.size, dtype=float),
            "sum_sq": np.zeros(spec.grid.size, dtype=float),
        }
        for spec in FEATURE_SPECS
    }


def _collect_effects(
    artifact: PolicyArtifact,
    objective: object,
    *,
    chunk_size: int,
) -> pd.DataFrame:
    base_frame = artifact.load_x("all")
    slopes = _verified_raw_slopes(objective, artifact.theta, base_frame)
    stats = _empty_stats()
    base_u_sum = 0.0
    action_low = float(artifact.policy_head.action_low)
    action_high = float(artifact.policy_head.action_high)
    action_span = action_high - action_low

    n_chunks = int(np.ceil(len(base_frame) / chunk_size))
    for chunk_index, start in enumerate(range(0, len(base_frame), chunk_size), start=1):
        stop = min(start + chunk_size, len(base_frame))
        chunk = base_frame.iloc[start:stop].copy()
        base_logits = _policy_logits(objective, artifact.theta, chunk)
        base_u_sum += float(np.sum(action_low + action_span * expit(base_logits)))
        for spec in FEATURE_SPECS:
            raw_values = chunk[spec.column].to_numpy(dtype=float)
            slope = slopes[spec.key]
            current = stats[spec.key]
            for grid_index, value in enumerate(spec.grid):
                logits = base_logits + slope * (float(value) - raw_values)
                policy_u = action_low + action_span * expit(logits)
                current["sum"][grid_index] += float(np.sum(policy_u, dtype=float))
                current["sum_sq"][grid_index] += float(
                    np.sum(policy_u * policy_u, dtype=float)
                )
        print(
            f"chunk {chunk_index}/{n_chunks} complete ({stop - start:,} customers)",
            flush=True,
        )

    n_customers = len(base_frame)
    replayed_mean_u = base_u_sum / n_customers
    saved_mean_u = float(artifact.train_metrics.mean_u)
    if not np.isclose(replayed_mean_u, saved_mean_u, rtol=1e-9, atol=1e-9):
        raise ValueError(
            "Replayed policy mean does not match the saved metric: "
            f"{replayed_mean_u:.12f} versus {saved_mean_u:.12f}"
        )

    rows: list[dict[str, object]] = []
    for spec in FEATURE_SPECS:
        sums = stats[spec.key]["sum"]
        sums_sq = stats[spec.key]["sum_sq"]
        means = sums / n_customers
        variance = np.maximum(sums_sq / n_customers - means * means, 0.0)
        standard_deviations = np.sqrt(variance)
        for value, mean, std in zip(
            spec.grid,
            means,
            standard_deviations,
            strict=True,
        ):
            rows.append(
                {
                    "estimator": artifact.estimator,
                    "policy_type": artifact.policy_head.type,
                    "feature": spec.key,
                    "feature_column": spec.column,
                    "feature_label": spec.label,
                    "feature_value": float(value),
                    "n_customers": n_customers,
                    "mean_proposed_price_change": float(mean),
                    "std_proposed_price_change": float(std),
                    "std_ddof": 0,
                    "action_low": action_low,
                    "action_high": action_high,
                    "raw_logit_slope": slopes[spec.key],
                }
            )
    return pd.DataFrame(rows)


def _validate_collected(frame: pd.DataFrame) -> None:
    required = {
        "estimator",
        "policy_type",
        "feature",
        "feature_value",
        "n_customers",
        "mean_proposed_price_change",
        "std_proposed_price_change",
        "action_low",
        "action_high",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Collected policy-effect CSV is missing {sorted(missing)}")
    if set(frame["feature"]) != {spec.key for spec in FEATURE_SPECS}:
        raise ValueError("Collected policy-effect CSV has unexpected features")
    if set(frame["estimator"]) != {"first_order"}:
        raise ValueError("Collected policy-effect CSV is not for first_order")
    if set(frame["policy_type"]) != {"SoftmaxPolicy"}:
        raise ValueError("Collected policy-effect CSV is not for SoftmaxPolicy")
    if set(frame["n_customers"]) != {715_023}:
        raise ValueError("Collected policy-effect CSV is not all-customer")
    for spec in FEATURE_SPECS:
        values = frame.loc[
            frame["feature"].eq(spec.key), "feature_value"
        ].to_numpy(dtype=float)
        if not np.allclose(values, spec.grid):
            raise ValueError(f"Collected grid differs for {spec.key}; rerun with --recompute")


def _plot_effect(
    frame: pd.DataFrame,
    *,
    feature_label: str,
    output_path: Path,
) -> None:
    ordered = frame.sort_values("feature_value")
    x = ordered["feature_value"].to_numpy(dtype=float)
    mean = ordered["mean_proposed_price_change"].to_numpy(dtype=float) * 100.0
    std = ordered["std_proposed_price_change"].to_numpy(dtype=float) * 100.0
    action_low = float(ordered["action_low"].iloc[0]) * 100.0
    action_high = float(ordered["action_high"].iloc[0]) * 100.0
    lower = np.clip(mean - std, action_low, action_high)
    upper = np.clip(mean + std, action_low, action_high)

    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.fill_between(x, lower, upper, color=color, alpha=0.20, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=3)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_xlabel(feature_label, fontsize=12)
    ax.set_ylabel("Proposed Price Change (%)", fontsize=12)
    ax.set_title(f"Proposed Price Change by {feature_label}", fontsize=16)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    ax.margins(y=0.12)
    fig.savefig(output_path, format=output_path.suffix.removeprefix("."))
    plt.close(fig)


def _render_effects(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in FEATURE_SPECS:
        selected = frame.loc[frame["feature"].eq(spec.key)]
        output_path = output_dir / f"policy_{spec.key}_vs_proposed_price_change.pdf"
        _plot_effect(
            selected,
            feature_label=spec.label,
            output_path=output_path,
        )
        written.append(output_path)
    return written


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    policy_path = args.policy_artifact.expanduser().resolve()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else analysis_dir / "plots"
    )
    collected_path = analysis_dir / "policy_feature_effect_mean_std.csv"
    metadata_path = analysis_dir / "analysis_config.json"

    artifact, objective = _load_verified_artifact(policy_path)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if args.recompute or not collected_path.is_file():
        collected = _collect_effects(
            artifact,
            objective,
            chunk_size=args.chunk_size,
        )
        collected.to_csv(collected_path, index=False)
        metadata = {
            "policy_artifact": str(policy_path),
            "estimator": artifact.estimator,
            "policy_type": artifact.policy_head.type,
            "action_bounds": [
                artifact.policy_head.action_low,
                artifact.policy_head.action_high,
            ],
            "n_customers": int(artifact.data_binding.selected_row_indices.size),
            "construction": (
                "For each grid value, replace that raw feature for every customer, "
                "retain all other covariates, replay the saved policy, and compute "
                "the pointwise population mean and population standard deviation."
            ),
            "std_ddof": 0,
            "features": {
                spec.key: {
                    "column": spec.column,
                    "label": spec.label,
                    "grid_min": float(spec.grid[0]),
                    "grid_max": float(spec.grid[-1]),
                    "grid_count": int(spec.grid.size),
                }
                for spec in FEATURE_SPECS
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(collected)} rows to {collected_path}")
    else:
        collected = pd.read_csv(collected_path)
        print(f"Reusing {collected_path}")

    _validate_collected(collected)
    written = _render_effects(collected, output_dir)
    print(f"Wrote {len(written)} PDFs to {output_dir}")
    for path in written:
        print(path.name)


if __name__ == "__main__":
    main()
