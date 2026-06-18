"""Diagnose policy scores and GLM acceptance logits on low-sensitivity rows."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.loader import (
    ACCEPTANCE_STATE_COLS,
    extract_glm_acceptance_coefficients,
    load_model_artifacts,
    load_x_frame,
    sample_csv_row_indices,
)
from experiments.configs.real_data_factory import _artifact_policy_features
from experiments.policy_artifacts import load_policy_artifact
from experiments.sensitivity_buckets import (
    SENSITIVITY_BUCKETS,
    SensitivityBucket,
    build_glm_sensitivity_buckets,
    median_observed_u,
)
from objective._math import _sigmoid
from objective.policy_preprocessing import fit_policy_feature_preprocessor


DEFAULT_THETA: tuple[float, ...] = (
    0.463,
    0.342,
    -0.059,
    -0.216,
    -0.376,
    0.508,
    0.230,
    -0.109,
    0.325,
    -0.001,
    -0.147,
    0.002,
    -0.114,
    0.293,
    -0.356,
    -0.005,
    -0.063,
    -0.165,
    0.086,
    -0.003,
)
DEFAULT_OUTPUT_ROOT = Path("outputs") / "low-sensitivity-policy-acceptance-diagnostics"
DEFAULT_PREPROCESSOR_N_SAMPLES = 700000
DEFAULT_PREPROCESSOR_SEED = 42
BUCKET_ROW_SOURCES = ("eligible", "artifact-all", "artifact-train", "artifact-test")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover a GLM sensitivity bucket, evaluate a softmax policy theta, "
            "and write policy-score / acceptance-logit diagnostics."
        )
    )
    parser.add_argument(
        "--bucket",
        choices=(*SENSITIVITY_BUCKETS, "all"),
        nargs="+",
        default=["low"],
        help="Sensitivity bucket(s) to diagnose. Use 'all' for low, medium, and high.",
    )
    parser.add_argument(
        "--policy-artifact",
        type=Path,
        default=None,
        help=(
            "Saved policy artifact policy.json, or its containing directory. "
            "When supplied, theta, policy bounds, feature map, and preprocessing are replayed from the artifact."
        ),
    )
    parser.add_argument(
        "--theta",
        type=float,
        nargs="+",
        default=None,
        help="Manual softmax theta values. Defaults to the supplied full-data theta when --policy-artifact is omitted.",
    )
    parser.add_argument(
        "--bucket-u-ref",
        type=float,
        default=None,
        help="Reference u used to score sensitivity buckets. Defaults to the median observed GLM U.",
    )
    parser.add_argument(
        "--bucket-row-source",
        choices=BUCKET_ROW_SOURCES,
        default="eligible",
        help=(
            "Rows used before splitting sensitivity buckets: all eligible GLM rows, "
            "or rows saved in the policy artifact split. Artifact row sources require --policy-artifact."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV and plots. Defaults to a timestamped outputs directory.",
    )
    parser.add_argument(
        "--preprocessor-n-samples",
        type=int,
        default=DEFAULT_PREPROCESSOR_N_SAMPLES,
        help=(
            "Number of seeded GLM rows used to refit the policy-side preprocessor. "
            "Only used with manual --theta. Default matches the supplied theta's full-data run."
        ),
    )
    parser.add_argument(
        "--preprocessor-seed",
        type=int,
        default=DEFAULT_PREPROCESSOR_SEED,
        help="Seed used for the manual-theta policy-side preprocessor row sample.",
    )
    parser.add_argument(
        "--u-coef",
        type=float,
        default=None,
        help="Optional GLM beta_u override. Defaults to the policy artifact u_coef when present, otherwise the GLM artifact coefficient.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on bucket rows for quick debugging.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Histogram bin count.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=8,
        help="Number of diagnostic rows to print to stdout.",
    )
    return parser.parse_args(argv)


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / timestamp


def _select_bucket(name: str) -> SensitivityBucket:
    for bucket in build_glm_sensitivity_buckets():
        if bucket.name == name:
            return bucket
    raise ValueError(f"Unknown sensitivity bucket: {name}")


def _resolve_bucket_names(bucket_args: Sequence[str]) -> tuple[str, ...]:
    if "all" in bucket_args:
        return tuple(SENSITIVITY_BUCKETS)
    names: list[str] = []
    for name in bucket_args:
        if name not in names:
            names.append(name)
    return tuple(names)


def _bucket_map(
    *,
    u_ref: float,
    row_indices: Sequence[int] | np.ndarray | None,
) -> dict[str, SensitivityBucket]:
    return {
        bucket.name: bucket
        for bucket in build_glm_sensitivity_buckets(u_ref=float(u_ref), row_indices=row_indices)
    }


def _artifact_json_path(path: Path) -> Path:
    return path / "policy.json" if path.is_dir() else path


def _validate_policy_artifact(artifact: object) -> None:
    objective = getattr(artifact, "objective", None)
    model_type = getattr(objective, "model_type", None)
    if model_type != "glm":
        raise ValueError(f"diagnose_low_sensitivity_policy_acceptance only supports GLM artifacts, got {model_type!r}.")
    policy_head = getattr(artifact, "policy_head", None)
    policy_type = getattr(policy_head, "type", None)
    if policy_type != "SoftmaxPolicy":
        raise ValueError(f"diagnose_low_sensitivity_policy_acceptance requires a SoftmaxPolicy artifact, got {policy_type!r}.")


def _bucket_source_row_indices(bucket_row_source: str, artifact: object | None) -> np.ndarray | None:
    if bucket_row_source == "eligible":
        return None
    if artifact is None:
        raise ValueError("Artifact bucket row sources require --policy-artifact.")
    split = bucket_row_source.removeprefix("artifact-")
    row_indices_fn = getattr(artifact, "row_indices")
    return np.asarray(row_indices_fn(split), dtype=int)


def _limit_bucket(bucket: SensitivityBucket, max_rows: int | None) -> SensitivityBucket:
    if max_rows is None:
        return bucket
    if max_rows <= 0:
        raise ValueError("max_rows must be positive when provided.")
    return SensitivityBucket(
        name=bucket.name,
        row_indices=bucket.row_indices[:max_rows].copy(),
        scores=bucket.scores[:max_rows].copy(),
    )


def _fit_full_data_policy_preprocessor(
    acceptance_model: object,
    *,
    n_samples: int,
    seed: int,
):
    if n_samples <= 0:
        raise ValueError("preprocessor_n_samples must be positive.")
    row_indices = sample_csv_row_indices("glm", n_rows=int(n_samples), seed=int(seed))
    x_train = load_x_frame("glm", row_indices=row_indices)
    base_features = _artifact_policy_features(acceptance_model, x_train)
    return fit_policy_feature_preprocessor(
        base_features,
        standardize=True,
        sphere=True,
        pca_dim=None,
    )


def _policy_features(
    acceptance_model: object,
    x_frame: pd.DataFrame,
    policy_preprocessor: object,
) -> np.ndarray:
    base_features = _artifact_policy_features(acceptance_model, x_frame)
    transform = getattr(policy_preprocessor, "transform")
    return np.asarray(transform(base_features), dtype=float)


def _policy_feature_names(policy_preprocessor: object, n_features: int) -> list[str]:
    names = list(getattr(policy_preprocessor, "output_feature_names_", ()))
    if len(names) != int(n_features):
        names = [f"policy_feature_{idx + 1}" for idx in range(int(n_features))]
    return names


def _artifact_policy_feature_names(artifact: object, n_features: int) -> list[str]:
    feature_map = getattr(artifact, "feature_map", None)
    feature_map_type = getattr(feature_map, "type", None)
    policy_input = getattr(artifact, "policy_input", None)
    policy_preprocessor = getattr(policy_input, "policy_preprocessor", None)
    names = list(getattr(policy_preprocessor, "output_feature_names_", ()))
    if feature_map_type == "IdentityFeatureMap" and len(names) == int(n_features):
        return names
    label = str(feature_map_type or "mapped_policy")
    return [f"{label}_feature_{idx + 1}" for idx in range(int(n_features))]


def _policy_outputs(
    theta: Sequence[float],
    policy_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_arr = np.asarray(theta, dtype=float)
    features = np.asarray(policy_features, dtype=float)
    if theta_arr.ndim != 1:
        raise ValueError("theta must be 1D.")
    if features.ndim != 2:
        raise ValueError("policy_features must be 2D.")
    if theta_arr.size != features.shape[1] + 1:
        raise ValueError(
            "theta length must equal one intercept plus policy feature width. "
            f"Got theta length {theta_arr.size} and feature width {features.shape[1]}."
        )
    feature_dot = features @ theta_arr[1:]
    policy_score = theta_arr[0] + feature_dot
    policy_sigmoid = _sigmoid(policy_score)
    policy_u = -0.5 + policy_sigmoid
    return feature_dot, policy_score, policy_sigmoid, policy_u


def _artifact_policy_outputs(
    artifact: object,
    x_frame: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(getattr(artifact, "theta"), dtype=float)
    mapped_features_fn = getattr(artifact, "mapped_features")
    design_matrix_fn = getattr(artifact, "policy_design_matrix")
    predict_u_fn = getattr(artifact, "predict_u")
    mapped_features = np.asarray(mapped_features_fn(x_frame), dtype=float)
    design_matrix = np.asarray(design_matrix_fn(x_frame), dtype=float)
    if design_matrix.ndim != 2 or design_matrix.shape[1] != theta.size:
        raise ValueError(
            "Saved policy design matrix width must match theta length. "
            f"Got design width {design_matrix.shape[1] if design_matrix.ndim == 2 else 'invalid'} "
            f"and theta length {theta.size}."
        )
    if mapped_features.ndim != 2 or mapped_features.shape[1] != theta.size - 1:
        raise ValueError(
            "Saved SoftmaxPolicy mapped feature width must equal theta length minus one. "
            f"Got mapped width {mapped_features.shape[1] if mapped_features.ndim == 2 else 'invalid'} "
            f"and theta length {theta.size}."
        )
    feature_dot = design_matrix[:, 1:] @ theta[1:]
    policy_score = design_matrix @ theta
    policy_sigmoid = _sigmoid(policy_score)
    policy_u = np.asarray(predict_u_fn(x_frame, clip=True), dtype=float).reshape(-1)
    if policy_u.shape != (x_frame.shape[0],):
        raise ValueError("Saved policy artifact must return one policy action per row.")
    names = _artifact_policy_feature_names(artifact, mapped_features.shape[1])
    return names, mapped_features, feature_dot, policy_score, policy_sigmoid, policy_u


def _diagnostic_policy_outputs(
    *,
    theta: Sequence[float],
    acceptance_model: object,
    x_frame: pd.DataFrame,
    policy_preprocessor: object | None,
    artifact: object | None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if artifact is not None:
        return _artifact_policy_outputs(artifact, x_frame)
    if policy_preprocessor is None:
        raise ValueError("Manual theta diagnostics require a fitted policy preprocessor.")
    features = _policy_features(acceptance_model, x_frame, policy_preprocessor)
    policy_names = _policy_feature_names(policy_preprocessor, features.shape[1])
    feature_dot, policy_score, policy_sigmoid, policy_u = _policy_outputs(theta, features)
    return policy_names, features, feature_dot, policy_score, policy_sigmoid, policy_u


def _acceptance_base_terms(
    acceptance_model: object,
    x_frame: pd.DataFrame,
    coeffs: dict[str, object],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    x_feature_cols = tuple(getattr(acceptance_model, "x_feature_cols", tuple(ACCEPTANCE_STATE_COLS)))
    raw_frame = x_frame.loc[:, list(x_feature_cols)].copy()
    raw_frame["U"] = 0.0
    model_frame_fn = getattr(acceptance_model, "model_frame", None)
    model_frame = model_frame_fn(raw_frame) if callable(model_frame_fn) else raw_frame

    feature_names = list(coeffs["x_feature_names"])
    x_matrix = model_frame.loc[:, feature_names].to_numpy(dtype=float)
    beta_x = np.asarray(coeffs["x_coef"], dtype=float)
    base_logit = float(coeffs["intercept"]) + x_matrix @ beta_x
    return base_logit, feature_names, x_matrix, beta_x


def _acceptance_outputs(
    acceptance_base_logit: np.ndarray,
    beta_u: float,
    policy_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    acceptance_logit = np.asarray(acceptance_base_logit, dtype=float) + float(beta_u) * np.asarray(
        policy_u,
        dtype=float,
    )
    return acceptance_logit, _sigmoid(acceptance_logit)


def _safe_feature_suffix(index: int, name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")
    if not cleaned:
        cleaned = "feature"
    return f"{index:02d}_{cleaned}"


def _feature_value_frame(
    prefix: str,
    names: Sequence[str],
    values: np.ndarray,
) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float)
    columns = [f"{prefix}_{_safe_feature_suffix(idx, name)}" for idx, name in enumerate(names)]
    return pd.DataFrame(arr, columns=columns)


def _diagnostic_frame(
    *,
    bucket: SensitivityBucket,
    policy_source: str,
    policy_estimator: str,
    bucket_u_ref: float,
    bucket_row_source: str,
    policy_feature_names: Sequence[str],
    policy_features: np.ndarray,
    policy_theta_coef: np.ndarray,
    feature_dot: np.ndarray,
    policy_score: np.ndarray,
    policy_sigmoid: np.ndarray,
    policy_u: np.ndarray,
    acceptance_feature_names: Sequence[str],
    acceptance_features: np.ndarray,
    acceptance_beta_x: np.ndarray,
    acceptance_base_logit: np.ndarray,
    beta_u: float,
    acceptance_logit: np.ndarray,
    acceptance_probability: np.ndarray,
) -> pd.DataFrame:
    row_indices = np.asarray(bucket.row_indices, dtype=int)
    frame = pd.DataFrame(
        {
            "row_index": row_indices,
            "csv_line_number": row_indices + 2,
            "bucket": bucket.name,
            "bucket_u_ref": float(bucket_u_ref),
            "bucket_row_source": bucket_row_source,
            "policy_source": policy_source,
            "policy_estimator": policy_estimator,
            "bucket_position": np.arange(row_indices.size, dtype=int),
            "sensitivity_score": np.asarray(bucket.scores, dtype=float),
            "policy_feature_dot_without_intercept": feature_dot,
            "policy_score": policy_score,
            "policy_sigmoid": policy_sigmoid,
            "policy_u": policy_u,
            "acceptance_base_logit": acceptance_base_logit,
            "acceptance_beta_u": float(beta_u),
            "acceptance_u_term": float(beta_u) * policy_u,
            "acceptance_logit": acceptance_logit,
            "acceptance_probability": acceptance_probability,
        }
    )
    policy_values = _feature_value_frame("policy_feature", policy_feature_names, policy_features)
    policy_contributions = _feature_value_frame(
        "policy_contribution",
        policy_feature_names,
        np.asarray(policy_features, dtype=float) * np.asarray(policy_theta_coef, dtype=float)[None, :],
    )
    acceptance_values = _feature_value_frame(
        "acceptance_feature",
        acceptance_feature_names,
        acceptance_features,
    )
    acceptance_contributions = _feature_value_frame(
        "acceptance_contribution",
        acceptance_feature_names,
        np.asarray(acceptance_features, dtype=float) * np.asarray(acceptance_beta_x, dtype=float)[None, :],
    )
    return pd.concat(
        [frame, policy_values, policy_contributions, acceptance_values, acceptance_contributions],
        axis=1,
    )


def _plot_histogram(
    values: np.ndarray,
    *,
    xlabel: str,
    title: str,
    output_path: Path,
    bins: int,
) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError(f"No finite values available for {title}.")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.75))
    ax.hist(finite, bins=bins, color="#4c78a8", edgecolor="white", alpha=0.9)
    ax.axvline(float(np.mean(finite)), color="#f58518", linewidth=1.8, label="mean")
    ax.axvline(float(np.median(finite)), color="#54a24b", linewidth=1.8, label="median")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Customer count")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _print_summary(name: str, values: np.ndarray) -> None:
    summary = _summary(values)
    formatted = ", ".join(f"{key}={value:.8g}" for key, value in summary.items())
    print(f"{name}: {formatted}")


def _run_bucket_diagnostics(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    bucket: SensitivityBucket,
    theta: np.ndarray,
    artifact: object | None,
    policy_source: str,
    policy_estimator: str,
    bucket_u_ref: float,
    bucket_row_source: str,
    acceptance_model: object,
    coeffs: dict[str, object],
    beta_u: float,
    policy_preprocessor: object | None,
) -> dict[str, Path]:
    x_frame = load_x_frame("glm", row_indices=bucket.row_indices)
    (
        policy_names,
        features,
        feature_dot,
        policy_score,
        policy_sigmoid,
        policy_u,
    ) = _diagnostic_policy_outputs(
        theta=theta,
        acceptance_model=acceptance_model,
        x_frame=x_frame,
        policy_preprocessor=policy_preprocessor,
        artifact=artifact,
    )

    acceptance_base_logit, acceptance_names, acceptance_features, acceptance_beta_x = _acceptance_base_terms(
        acceptance_model,
        x_frame,
        coeffs,
    )
    acceptance_logit, acceptance_probability = _acceptance_outputs(
        acceptance_base_logit,
        beta_u,
        policy_u,
    )

    diagnostics = _diagnostic_frame(
        bucket=bucket,
        policy_source=policy_source,
        policy_estimator=policy_estimator,
        bucket_u_ref=bucket_u_ref,
        bucket_row_source=bucket_row_source,
        policy_feature_names=policy_names,
        policy_features=features,
        policy_theta_coef=theta[1:],
        feature_dot=feature_dot,
        policy_score=policy_score,
        policy_sigmoid=policy_sigmoid,
        policy_u=policy_u,
        acceptance_feature_names=acceptance_names,
        acceptance_features=acceptance_features,
        acceptance_beta_x=acceptance_beta_x,
        acceptance_base_logit=acceptance_base_logit,
        beta_u=beta_u,
        acceptance_logit=acceptance_logit,
        acceptance_probability=acceptance_probability,
    )

    csv_path = output_dir / f"{bucket.name}_sensitivity_policy_acceptance_diagnostics.csv"
    diagnostics.to_csv(csv_path, index=False)

    policy_hist_path = output_dir / f"{bucket.name}_policy_score_histogram.png"
    acceptance_base_logit_hist_path = output_dir / f"{bucket.name}_acceptance_base_logit_histogram.png"
    acceptance_logit_hist_path = output_dir / f"{bucket.name}_acceptance_logit_histogram.png"
    _plot_histogram(
        policy_score,
        xlabel="policy_score = theta0 + processed_policy_X @ theta[1:]",
        title=f"{bucket.name.capitalize()}-Sensitivity Policy Sigmoid Input",
        output_path=policy_hist_path,
        bins=int(args.bins),
    )
    _plot_histogram(
        acceptance_base_logit,
        xlabel="acceptance_base_logit = beta0 + beta_x^T x",
        title=f"{bucket.name.capitalize()}-Sensitivity Acceptance Base Logit",
        output_path=acceptance_base_logit_hist_path,
        bins=int(args.bins),
    )
    _plot_histogram(
        acceptance_logit,
        xlabel="acceptance_logit = beta0 + beta_x^T x + beta_u * policy_u",
        title=f"{bucket.name.capitalize()}-Sensitivity Acceptance Sigmoid Input",
        output_path=acceptance_logit_hist_path,
        bins=int(args.bins),
    )

    print(
        "bucket rows: "
        f"name={bucket.name}, n={bucket.row_indices.size}, min={int(bucket.row_indices.min())}, "
        f"max={int(bucket.row_indices.max())}, head={bucket.row_indices[:10].tolist()}"
    )
    _print_summary(f"{bucket.name} policy_score", policy_score)
    _print_summary(f"{bucket.name} policy_u", policy_u)
    _print_summary(f"{bucket.name} acceptance_base_logit", acceptance_base_logit)
    _print_summary(f"{bucket.name} acceptance_logit", acceptance_logit)
    _print_summary(f"{bucket.name} acceptance_probability", acceptance_probability)
    if int(args.preview_rows) > 0:
        preview_cols = [
            "row_index",
            "csv_line_number",
            "sensitivity_score",
            "policy_score",
            "policy_u",
            "acceptance_base_logit",
            "acceptance_logit",
            "acceptance_probability",
        ]
        print(diagnostics.loc[:, preview_cols].head(int(args.preview_rows)).to_string(index=False))
    print(f"wrote CSV: {csv_path}")
    print(f"wrote plot: {policy_hist_path}")
    print(f"wrote plot: {acceptance_base_logit_hist_path}")
    print(f"wrote plot: {acceptance_logit_hist_path}")

    return {
        f"{bucket.name}_csv": csv_path,
        f"{bucket.name}_policy_score_histogram": policy_hist_path,
        f"{bucket.name}_acceptance_base_logit_histogram": acceptance_base_logit_hist_path,
        f"{bucket.name}_acceptance_logit_histogram": acceptance_logit_hist_path,
    }


def run_diagnostics(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = None
    policy_source = "manual-theta"
    policy_estimator = "manual"
    if args.policy_artifact is not None:
        if args.theta is not None:
            raise ValueError("Use either --policy-artifact or --theta, not both.")
        artifact_path = _artifact_json_path(args.policy_artifact)
        artifact = load_policy_artifact(artifact_path)
        _validate_policy_artifact(artifact)
        policy_source = str(artifact_path)
        policy_estimator = str(getattr(artifact, "estimator", "artifact"))

    theta_source = getattr(artifact, "theta", None) if artifact is not None else args.theta
    theta = np.asarray(theta_source if theta_source is not None else DEFAULT_THETA, dtype=float)
    acceptance_model, _ = load_model_artifacts("glm")
    coeffs = extract_glm_acceptance_coefficients(acceptance_model)
    artifact_u_coef = getattr(getattr(artifact, "objective", None), "u_coef", None) if artifact is not None else None
    beta_u = (
        float(args.u_coef)
        if args.u_coef is not None
        else float(artifact_u_coef)
        if artifact_u_coef is not None
        else float(coeffs["u_coef"])
    )
    policy_preprocessor = None
    if artifact is None:
        policy_preprocessor = _fit_full_data_policy_preprocessor(
            acceptance_model,
            n_samples=int(args.preprocessor_n_samples),
            seed=int(args.preprocessor_seed),
        )

    bucket_u_ref = median_observed_u("glm") if args.bucket_u_ref is None else float(args.bucket_u_ref)
    bucket_row_indices = _bucket_source_row_indices(args.bucket_row_source, artifact)

    bucket_names = _resolve_bucket_names(args.bucket)
    buckets = _bucket_map(u_ref=bucket_u_ref, row_indices=bucket_row_indices)
    policy_feature_width = int(
        getattr(policy_preprocessor, "output_dim_", theta.size - 1)
        if policy_preprocessor is not None
        else theta.size - 1
    )
    print(f"Recovered sensitivity buckets at u_ref={bucket_u_ref:.12g}")
    print(f"bucket row source: {args.bucket_row_source}")
    print(f"policy source: {policy_source}")
    print(f"policy estimator: {policy_estimator}")
    print(f"theta length: {theta.size}; policy feature width: {policy_feature_width}")
    print(f"acceptance beta_u: {beta_u:.12g}")
    outputs: dict[str, Path] = {}
    for bucket_name in bucket_names:
        bucket = _limit_bucket(buckets[bucket_name], args.max_rows)
        outputs.update(
            _run_bucket_diagnostics(
                args=args,
                output_dir=output_dir,
                bucket=bucket,
                theta=theta,
                artifact=artifact,
                policy_source=policy_source,
                policy_estimator=policy_estimator,
                bucket_u_ref=bucket_u_ref,
                bucket_row_source=str(args.bucket_row_source),
                acceptance_model=acceptance_model,
                coeffs=coeffs,
                beta_u=beta_u,
                policy_preprocessor=policy_preprocessor,
            )
        )
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_diagnostics(args)


if __name__ == "__main__":
    main()
