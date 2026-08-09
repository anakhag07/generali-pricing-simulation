"""Analyze all-customer acceptance curves and raw-X prediction sensitivity."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset_metadata import (
    ACCEPTANCE_MODEL_ARTIFACTS,
    ACCEPTANCE_STATE_COLS,
    DATASET_PATH,
    LOSS_FEATURE_COLS,
    LOSS_MODEL_ARTIFACTS,
)
from data.loader import (
    ModelArtifactBundle,
    eligible_csv_row_indices,
    load_model_artifacts,
    load_observed_u_array,
    load_x_frame,
)
from data.monotone_spline_xgb import fit_monotone_churn_curve
from experiments.launch import (
    LaunchContext,
    LaunchPlan,
    add_launch_args,
    run_launch_plan,
    task_payloads,
)


PROJECT_NAME = "model-acceptance-feature-analysis"
MODEL_ORDER = ("glm", "xgb", "spline")
MODEL_LABELS = {
    "glm": "GLM",
    "xgb": "XGBoost",
    "spline": "Monotone-spline XGBoost",
}
MODEL_COLORS = {"glm": "#2166ac", "xgb": "#b2182b", "spline": "#1b7837"}
ANCHOR_U = np.linspace(0.0, 0.16, 17)
IMPORTANCE_U = np.asarray([0.0, 0.04, 0.08, 0.12, 0.16])
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class AnalysisTask:
    """One resumable curve or feature-importance task."""

    kind: str
    start: int | None = None
    stop: int | None = None
    model: str | None = None
    target: str | None = None
    repeat: int | None = None
    features: tuple[str, ...] = ()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u-min", type=float, default=0.0)
    parser.add_argument("--u-max", type=float, default=0.16)
    parser.add_argument("--u-count", type=int, default=161)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--histogram-bins", type=int, default=1000)
    parser.add_argument("--importance-n-rows", type=int, default=20_000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--permutation-seed", type=int, default=42)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--spline-importance-group-size", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=8)
    add_launch_args(parser, default_launch="auto", default_array=True)
    args = parser.parse_args(argv)
    if args.u_min != 0.0 or args.u_max != 0.16:
        raise ValueError("Exact source-recipe spline analysis requires u in [0, 0.16].")
    for name in (
        "u_count",
        "chunk_size",
        "histogram_bins",
        "importance_n_rows",
        "permutation_repeats",
        "spline_importance_group_size",
        "n_jobs",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    return args


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible_rows() -> np.ndarray:
    return eligible_csv_row_indices("linear")


def _spline_weights(row_indices: np.ndarray) -> np.ndarray:
    observed_u = load_observed_u_array("linear", row_indices=row_indices)
    in_support = observed_u[(observed_u >= 0.0) & (observed_u <= 0.16)]
    frequencies = pd.Series(in_support).round(2).value_counts(normalize=True)
    return frequencies.reindex(ANCHOR_U.round(2), fill_value=0.0).to_numpy(float)


def _prepared_acceptance_frame(
    artifact: ModelArtifactBundle,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    raw = frame.copy()
    raw["U"] = 0.0
    return artifact.model_frame(raw)


def _predict_acceptance_matrix(
    artifact: ModelArtifactBundle,
    frame: pd.DataFrame,
    u_values: Sequence[float],
) -> np.ndarray:
    model_frame = _prepared_acceptance_frame(artifact, frame)
    matrix = np.empty((len(frame), len(u_values)), dtype=float)
    for index, u_value in enumerate(u_values):
        model_frame["U"] = float(u_value)
        probability = np.asarray(artifact.model.predict_proba(model_frame), dtype=float)
        matrix[:, index] = probability[:, 1]
    return np.clip(matrix, 0.0, 1.0)


def _predict_loss(artifact: ModelArtifactBundle, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(artifact.model.predict(artifact.model_frame(frame)), dtype=float)


def _fit_spline_row(
    acceptance_at_anchors: np.ndarray,
    u_values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    fitted = fit_monotone_churn_curve(
        ANCHOR_U,
        1.0 - acceptance_at_anchors,
        weights=weights,
        dense_grid_size=500,
    )
    return np.clip(1.0 - fitted.curve(u_values), 0.0, 1.0)


def exact_spline_acceptance_matrix(
    xgb_acceptance: ModelArtifactBundle,
    frame: pd.DataFrame,
    u_values: Sequence[float],
    weights: np.ndarray,
    *,
    n_jobs: int,
    raw_fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Fit and evaluate one exact source-recipe spline per input row."""
    u_array = np.asarray(u_values, dtype=float)
    anchors = _predict_acceptance_matrix(xgb_acceptance, frame, ANCHOR_U)

    def fit_one(row: np.ndarray) -> tuple[np.ndarray | None, bool]:
        try:
            return _fit_spline_row(row, u_array, weights), True
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None, False

    fitted = Parallel(n_jobs=int(n_jobs), prefer="threads")(
        delayed(fit_one)(row) for row in anchors
    )
    output = np.empty((len(frame), u_array.size), dtype=float)
    failures = 0
    for index, (values, success) in enumerate(fitted):
        if success and values is not None:
            output[index] = values
            continue
        failures += 1
        if raw_fallback is None:
            fallback_frame = frame.iloc[[index]]
            output[index] = _predict_acceptance_matrix(
                xgb_acceptance, fallback_frame, u_array
            )[0]
        else:
            output[index] = raw_fallback[index]
    return output, failures


def summarize_acceptance_matrix(
    matrix: np.ndarray,
    *,
    histogram_bins: int,
) -> dict[str, np.ndarray]:
    """Return additive sufficient statistics for an acceptance matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("matrix must be a non-empty 2D array.")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("acceptance values must be finite and in [0, 1].")
    edges = np.linspace(0.0, 1.0, int(histogram_bins) + 1)
    hist = np.stack(
        [np.histogram(values[:, column], bins=edges)[0] for column in range(values.shape[1])]
    )
    return {
        "sum": np.sum(values, axis=0),
        "sum_sq": np.sum(values * values, axis=0),
        "hist": hist.astype(np.int64),
    }


def _chunk_output_path(context: LaunchContext, task_index: int) -> Path:
    return context.sweep_dir / "chunks" / f"acceptance_chunk_{task_index:03d}.npz"


def _write_chunk(
    path: Path,
    *,
    row_indices: np.ndarray,
    u_values: np.ndarray,
    summaries: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "row_indices": np.asarray(row_indices, dtype=int),
        "u_values": np.asarray(u_values, dtype=float),
    }
    for model, summary in summaries.items():
        for metric, values in summary.items():
            payload[f"{model}_{metric}"] = np.asarray(values)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _resume_payload(context: LaunchContext, task_index: int) -> dict[str, Any] | None:
    path = context.task_record_path(task_index)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("status") != "success" or not isinstance(record.get("payload"), dict):
        return None
    payload = dict(record["payload"])
    output_path = payload.get("output_path")
    if output_path is not None and not Path(str(output_path)).exists():
        return None
    print(f"[resume] task {task_index} already complete", flush=True)
    return payload


def _run_curve_task(
    task_index: int,
    task: AnalysisTask,
    context: LaunchContext,
    args: argparse.Namespace,
    eligible: np.ndarray,
) -> dict[str, Any]:
    assert task.start is not None and task.stop is not None
    selected = eligible[task.start : task.stop]
    frame = load_x_frame("xgb", row_indices=selected)
    glm_acceptance, _ = load_model_artifacts("linear")
    xgb_acceptance, _ = load_model_artifacts("xgb")
    u_values = np.linspace(args.u_min, args.u_max, args.u_count)
    weights = _spline_weights(eligible)
    glm_matrix = _predict_acceptance_matrix(glm_acceptance, frame, u_values)
    xgb_matrix = _predict_acceptance_matrix(xgb_acceptance, frame, u_values)
    spline_matrix, failures = exact_spline_acceptance_matrix(
        xgb_acceptance,
        frame,
        u_values,
        weights,
        n_jobs=args.n_jobs,
        raw_fallback=xgb_matrix,
    )
    summaries = {
        "glm": summarize_acceptance_matrix(glm_matrix, histogram_bins=args.histogram_bins),
        "xgb": summarize_acceptance_matrix(xgb_matrix, histogram_bins=args.histogram_bins),
        "spline": summarize_acceptance_matrix(spline_matrix, histogram_bins=args.histogram_bins),
    }
    output_path = _chunk_output_path(context, task_index)
    _write_chunk(
        output_path,
        row_indices=selected,
        u_values=u_values,
        summaries=summaries,
    )
    return {
        "kind": "curve",
        "start": task.start,
        "stop": task.stop,
        "n_rows": int(selected.size),
        "spline_failures": int(failures),
        "output_path": str(output_path),
    }


def _importance_sample(eligible: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    n_rows = min(int(args.importance_n_rows), int(eligible.size))
    rng = np.random.default_rng(int(args.sample_seed))
    return np.sort(rng.choice(eligible, size=n_rows, replace=False).astype(int))


def _permuted_frame(
    frame: pd.DataFrame,
    feature: str,
    *,
    repeat: int,
    permutation_seed: int,
) -> pd.DataFrame:
    feature_index = ACCEPTANCE_STATE_COLS.index(feature)
    seed = np.random.SeedSequence([int(permutation_seed), int(repeat), feature_index])
    rng = np.random.default_rng(seed)
    output = frame.copy()
    values = output[feature].to_numpy(copy=True)
    output[feature] = values[rng.permutation(values.size)]
    return output


def _importance_score(baseline: np.ndarray, perturbed: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(baseline) - np.asarray(perturbed))))


def _importance_artifact(model: str, target: str) -> ModelArtifactBundle:
    family = "linear" if model == "glm" else "xgb"
    acceptance, loss = load_model_artifacts(family)
    return acceptance if target == "acceptance" else loss


def _run_standard_importance_task(
    task: AnalysisTask,
    frame: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int]:
    assert task.model is not None and task.target is not None and task.repeat is not None
    artifact = _importance_artifact(task.model, task.target)
    if task.target == "acceptance":
        baseline = _predict_acceptance_matrix(artifact, frame, IMPORTANCE_U)
    else:
        baseline = _predict_loss(artifact, frame)
    rows: list[dict[str, Any]] = []
    for feature in task.features:
        permuted = _permuted_frame(
            frame,
            feature,
            repeat=task.repeat,
            permutation_seed=args.permutation_seed,
        )
        if task.target == "acceptance":
            prediction = _predict_acceptance_matrix(artifact, permuted, IMPORTANCE_U)
        else:
            prediction = _predict_loss(artifact, permuted)
        rows.append(
            {
                "model": task.model,
                "target": task.target,
                "feature": feature,
                "repeat": task.repeat,
                "score": _importance_score(baseline, prediction),
            }
        )
    return rows, 0


def _run_spline_importance_task(
    task: AnalysisTask,
    frame: pd.DataFrame,
    args: argparse.Namespace,
    weights: np.ndarray,
) -> tuple[list[dict[str, Any]], int]:
    assert task.repeat is not None
    xgb_acceptance = _importance_artifact("xgb", "acceptance")
    raw_baseline = _predict_acceptance_matrix(xgb_acceptance, frame, IMPORTANCE_U)
    baseline, failures = exact_spline_acceptance_matrix(
        xgb_acceptance,
        frame,
        IMPORTANCE_U,
        weights,
        n_jobs=args.n_jobs,
        raw_fallback=raw_baseline,
    )
    rows: list[dict[str, Any]] = []
    for feature in task.features:
        permuted = _permuted_frame(
            frame,
            feature,
            repeat=task.repeat,
            permutation_seed=args.permutation_seed,
        )
        raw_permuted = _predict_acceptance_matrix(xgb_acceptance, permuted, IMPORTANCE_U)
        prediction, feature_failures = exact_spline_acceptance_matrix(
            xgb_acceptance,
            permuted,
            IMPORTANCE_U,
            weights,
            n_jobs=args.n_jobs,
            raw_fallback=raw_permuted,
        )
        failures += feature_failures
        rows.append(
            {
                "model": "spline",
                "target": "acceptance",
                "feature": feature,
                "repeat": task.repeat,
                "score": _importance_score(baseline, prediction),
            }
        )
    return rows, failures


def _run_importance_task(
    task: AnalysisTask,
    args: argparse.Namespace,
    eligible: np.ndarray,
) -> dict[str, Any]:
    selected = _importance_sample(eligible, args)
    frame = load_x_frame("xgb", row_indices=selected)
    if task.model == "spline":
        rows, failures = _run_spline_importance_task(
            task, frame, args, _spline_weights(eligible)
        )
    else:
        rows, failures = _run_standard_importance_task(task, frame, args)
    return {
        "kind": "importance",
        "n_rows": int(selected.size),
        "spline_failures": int(failures),
        "rows": rows,
    }


def _run_task(
    task_index: int,
    context: LaunchContext,
    *,
    tasks: Sequence[AnalysisTask],
    args: argparse.Namespace,
    eligible: np.ndarray,
) -> dict[str, Any]:
    resumed = _resume_payload(context, task_index)
    if resumed is not None:
        return resumed
    task = tasks[task_index]
    print(f"[analysis] task={task_index} kind={task.kind}", flush=True)
    if task.kind == "curve":
        return _run_curve_task(task_index, task, context, args, eligible)
    if task.kind == "importance":
        return _run_importance_task(task, args, eligible)
    raise ValueError(f"Unknown task kind {task.kind!r}.")


def build_tasks(eligible: np.ndarray, args: argparse.Namespace) -> list[AnalysisTask]:
    tasks = [
        AnalysisTask("curve", start=start, stop=min(start + args.chunk_size, eligible.size))
        for start in range(0, eligible.size, args.chunk_size)
    ]
    for repeat in range(args.permutation_repeats):
        tasks.extend(
            [
                AnalysisTask(
                    "importance",
                    model=model,
                    target=target,
                    repeat=repeat,
                    features=tuple(
                        ACCEPTANCE_STATE_COLS if target == "acceptance" else LOSS_FEATURE_COLS
                    ),
                )
                for model, target in (
                    ("glm", "acceptance"),
                    ("xgb", "acceptance"),
                    ("glm", "loss"),
                    ("xgb", "loss"),
                )
            ]
        )
        spline_features = tuple(ACCEPTANCE_STATE_COLS)
        for start in range(0, len(spline_features), args.spline_importance_group_size):
            tasks.append(
                AnalysisTask(
                    "importance",
                    model="spline",
                    target="acceptance",
                    repeat=repeat,
                    features=spline_features[start : start + args.spline_importance_group_size],
                )
            )
    return tasks


def _histogram_quantile(hist: np.ndarray, q: float) -> np.ndarray:
    cumulative = np.cumsum(hist, axis=1)
    totals = cumulative[:, -1]
    thresholds = q * totals
    indices = np.asarray(
        [np.searchsorted(row, threshold, side="left") for row, threshold in zip(cumulative, thresholds, strict=True)]
    )
    return (indices.astype(float) + 0.5) / hist.shape[1]


def aggregate_curve_chunks(
    payloads: Sequence[Mapping[str, Any]],
    eligible: np.ndarray,
) -> tuple[list[dict[str, Any]], int]:
    curve_payloads = sorted(
        (payload for payload in payloads if payload.get("kind") == "curve"),
        key=lambda payload: int(payload["start"]),
    )
    if not curve_payloads:
        raise ValueError("No curve task outputs were produced.")
    covered: list[np.ndarray] = []
    totals: dict[str, dict[str, np.ndarray]] = {}
    u_values: np.ndarray | None = None
    spline_failures = 0
    for payload in curve_payloads:
        with np.load(str(payload["output_path"]), allow_pickle=False) as chunk:
            covered.append(chunk["row_indices"].copy())
            chunk_u = chunk["u_values"].copy()
            if u_values is None:
                u_values = chunk_u
            else:
                np.testing.assert_allclose(u_values, chunk_u)
            for model in MODEL_ORDER:
                model_totals = totals.setdefault(model, {})
                for metric in ("sum", "sum_sq", "hist"):
                    values = chunk[f"{model}_{metric}"].astype(np.float64 if metric != "hist" else np.int64)
                    if metric not in model_totals:
                        model_totals[metric] = values
                    else:
                        model_totals[metric] += values
        spline_failures += int(payload.get("spline_failures", 0))
    concatenated = np.concatenate(covered)
    if concatenated.size != eligible.size or not np.array_equal(np.sort(concatenated), eligible):
        raise ValueError("Curve chunks do not cover every eligible row exactly once.")
    assert u_values is not None
    n_rows = int(eligible.size)
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        summed = totals[model]["sum"]
        summed_sq = totals[model]["sum_sq"]
        hist = totals[model]["hist"]
        mean = summed / n_rows
        variance = np.maximum(summed_sq / n_rows - mean * mean, 0.0)
        quantile_values = [_histogram_quantile(hist, q) for q in QUANTILES]
        for index, u_value in enumerate(u_values):
            rows.append(
                {
                    "model": model,
                    "u": float(u_value),
                    "n_rows": n_rows,
                    "mean": float(mean[index]),
                    "std": float(np.sqrt(variance[index])),
                    "q05": float(quantile_values[0][index]),
                    "q25": float(quantile_values[1][index]),
                    "q50": float(quantile_values[2][index]),
                    "q75": float(quantile_values[3][index]),
                    "q95": float(quantile_values[4][index]),
                }
            )
    return rows, spline_failures


def aggregate_importance_rows(
    payloads: Sequence[Mapping[str, Any]],
    *,
    expected_repeats: int,
) -> tuple[list[dict[str, Any]], int]:
    raw_rows = [
        row
        for payload in payloads
        if payload.get("kind") == "importance"
        for row in payload.get("rows", [])
    ]
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in raw_rows:
        key = (str(row["model"]), str(row["target"]), str(row["feature"]))
        grouped.setdefault(key, []).append(float(row["score"]))
    rows: list[dict[str, Any]] = []
    for (model, target, feature), scores in grouped.items():
        if len(scores) != expected_repeats:
            raise ValueError(
                f"Expected {expected_repeats} importance repeats for {model}/{target}/{feature}, "
                f"found {len(scores)}."
            )
        score_array = np.asarray(scores, dtype=float)
        rows.append(
            {
                "model": model,
                "target": target,
                "feature": feature,
                "importance_mean": float(np.mean(score_array)),
                "importance_std": float(np.std(score_array, ddof=1)) if score_array.size > 1 else 0.0,
                "metric": "mean_absolute_prediction_change",
                "provenance": f"{model}_{target}_artifact",
            }
        )
    xgb_loss = [row for row in rows if row["model"] == "xgb" and row["target"] == "loss"]
    for row in xgb_loss:
        duplicate = dict(row)
        duplicate["model"] = "spline"
        duplicate["provenance"] = "shared_xgb_loss_artifact"
        rows.append(duplicate)
    for model in MODEL_ORDER:
        for target in ("acceptance", "loss"):
            subset = sorted(
                (row for row in rows if row["model"] == model and row["target"] == target),
                key=lambda row: (-float(row["importance_mean"]), str(row["feature"])),
            )
            for rank, row in enumerate(subset, start=1):
                row["rank"] = rank
    rows.sort(key=lambda row: (MODEL_ORDER.index(str(row["model"])), str(row["target"]), int(row["rank"])))
    failures = sum(
        int(payload.get("spline_failures", 0))
        for payload in payloads
        if payload.get("kind") == "importance"
    )
    return rows, failures


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _model_curve_rows(rows: Sequence[Mapping[str, Any]], model: str) -> list[Mapping[str, Any]]:
    return sorted((row for row in rows if row["model"] == model), key=lambda row: float(row["u"]))


def _plot_one_model(rows: Sequence[Mapping[str, Any]], model: str, output_dir: Path) -> None:
    selected = _model_curve_rows(rows, model)
    u = np.asarray([row["u"] for row in selected], dtype=float)
    mean = np.asarray([row["mean"] for row in selected], dtype=float)
    q05 = np.asarray([row["q05"] for row in selected], dtype=float)
    q25 = np.asarray([row["q25"] for row in selected], dtype=float)
    q75 = np.asarray([row["q75"] for row in selected], dtype=float)
    q95 = np.asarray([row["q95"] for row in selected], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(u, q05, q95, color=MODEL_COLORS[model], alpha=0.12, label="5–95%")
    ax.fill_between(u, q25, q75, color=MODEL_COLORS[model], alpha=0.25, label="25–75%")
    ax.plot(u, mean, color=MODEL_COLORS[model], linewidth=2.0, label="Mean")
    ax.set(title=MODEL_LABELS[model], xlabel="Proposed price change u", ylabel="Acceptance probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{model}_acceptance_by_u.png", dpi=200)
    plt.close(fig)


def _plot_comparison(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    fig, (ax, delta_ax) = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    means: dict[str, np.ndarray] = {}
    u_values: np.ndarray | None = None
    for model in MODEL_ORDER:
        selected = _model_curve_rows(rows, model)
        current_u = np.asarray([row["u"] for row in selected], dtype=float)
        means[model] = np.asarray([row["mean"] for row in selected], dtype=float)
        u_values = current_u if u_values is None else u_values
        ax.plot(current_u, means[model], color=MODEL_COLORS[model], linewidth=2.0, label=MODEL_LABELS[model])
    assert u_values is not None
    delta_ax.plot(u_values, means["spline"] - means["xgb"], color=MODEL_COLORS["spline"], linewidth=1.8)
    delta_ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Mean acceptance")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    delta_ax.set_xlabel("Proposed price change u")
    delta_ax.set_ylabel("Spline − XGB")
    delta_ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "acceptance_model_comparison.png", dpi=200)
    plt.close(fig)


def _write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    eligible: np.ndarray,
    *,
    curve_failures: int,
    importance_failures: int,
) -> None:
    artifacts = {
        "dataset": DATASET_PATH,
        "glm_acceptance": ACCEPTANCE_MODEL_ARTIFACTS["linear"]["path"],
        "glm_loss": LOSS_MODEL_ARTIFACTS["linear"]["path"],
        "xgb_acceptance": ACCEPTANCE_MODEL_ARTIFACTS["xgb"]["path"],
        "xgb_loss": LOSS_MODEL_ARTIFACTS["xgb"]["path"],
    }
    metadata = {
        "project": PROJECT_NAME,
        "eligible_rows": int(eligible.size),
        "row_index_min": int(eligible.min()),
        "row_index_max": int(eligible.max()),
        "curve_spline_failures": int(curve_failures),
        "importance_spline_failures": int(importance_failures),
        "artifact_sha256": {name: _sha256(path) for name, path in artifacts.items()},
        "config": {
            name: getattr(args, name)
            for name in (
                "u_min", "u_max", "u_count", "chunk_size", "histogram_bins",
                "importance_n_rows", "sample_seed", "permutation_seed",
                "permutation_repeats", "spline_importance_group_size", "n_jobs",
            )
        },
        "spline_recipe": {
            "anchor_u": ANCHOR_U.tolist(),
            "dense_grid_size": 500,
            "steps": ["weighted_smoothing_spline", "clip", "isotonic", "pchip"],
        },
    }
    (output_dir / "analysis_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def _print_top_features(rows: Sequence[Mapping[str, Any]], n: int = 10) -> None:
    for model in MODEL_ORDER:
        for target in ("acceptance", "loss"):
            selected = [
                row for row in rows
                if row["model"] == model and row["target"] == target and int(row["rank"]) <= n
            ]
            print(f"\n{MODEL_LABELS[model]} — {target}")
            for row in selected:
                print(
                    f"  {int(row['rank']):2d}. {str(row['feature']):32s} "
                    f"{float(row['importance_mean']):.6g} ± {float(row['importance_std']):.3g}"
                )


def _collect(
    context: LaunchContext,
    *,
    args: argparse.Namespace,
    eligible: np.ndarray,
) -> None:
    payloads = task_payloads(context)
    curve_rows, curve_failures = aggregate_curve_chunks(payloads, eligible)
    importance_rows, importance_failures = aggregate_importance_rows(
        payloads, expected_repeats=args.permutation_repeats
    )
    output_dir = context.sweep_dir
    _write_csv(output_dir / "acceptance_by_u.csv", curve_rows)
    _write_csv(output_dir / "feature_importance.csv", importance_rows)
    for model in MODEL_ORDER:
        _plot_one_model(curve_rows, model, output_dir)
    _plot_comparison(curve_rows, output_dir)
    _write_metadata(
        output_dir,
        args,
        eligible,
        curve_failures=curve_failures,
        importance_failures=importance_failures,
    )
    _print_top_features(importance_rows)
    print(f"\nWrote all-customer analysis to {output_dir}.")


def _build_launch_plan(
    args: argparse.Namespace,
    eligible: np.ndarray,
    tasks: Sequence[AnalysisTask],
) -> LaunchPlan:
    return LaunchPlan(
        name=PROJECT_NAME,
        task_count=len(tasks),
        requires_jax=False,
        run_task=lambda index, context: _run_task(
            index, context, tasks=tasks, args=args, eligible=eligible
        ),
        collect=lambda context: _collect(context, args=args, eligible=eligible),
        default_launch="auto",
        default_array=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    eligible = _eligible_rows()
    tasks = build_tasks(eligible, args)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    print(
        f"Prepared {len(tasks)} tasks for {eligible.size:,} eligible rows "
        f"({sum(task.kind == 'curve' for task in tasks)} curve tasks).",
        flush=True,
    )
    run_launch_plan(
        _build_launch_plan(args, eligible, tasks),
        args=args,
        argv=original_argv,
    )


if __name__ == "__main__":
    main()
