"""Plot sampled client-level acceptance curves for a saved policy artifact."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from experiments.paths import results_root
from experiments.policy_artifacts import load_policy_artifact


BUCKET_NAMES: tuple[str, str, str] = ("low", "medium", "high")


@dataclass(frozen=True)
class BucketSamples:
    """Sampled rows and acceptance curves for one diagnostic bucket."""

    name: str
    positions: np.ndarray
    row_indices: np.ndarray
    score_values: np.ndarray
    sensitivity_scores: np.ndarray
    predicted_loss: np.ndarray
    policy_u: np.ndarray
    policy_acceptance: np.ndarray
    acceptance_curves: np.ndarray


def _artifact_json_path(path: Path) -> Path:
    return path / "policy.json" if path.is_dir() else path


def _resolve_u_grid(u_min: float, u_max: float, u_count: int) -> np.ndarray:
    if not np.isfinite([u_min, u_max]).all():
        raise ValueError("u_min and u_max must be finite.")
    if u_min > u_max:
        raise ValueError("u_min must be <= u_max.")
    if u_count <= 0:
        raise ValueError("u_count must be positive.")
    return np.linspace(float(u_min), float(u_max), int(u_count), dtype=float)


def _row_count(x_batch: object) -> int:
    return int(getattr(x_batch, "shape")[0])


def _slice_rows(x_batch: object, positions: Sequence[int] | np.ndarray) -> object:
    indices = np.asarray(positions, dtype=int)
    iloc = getattr(x_batch, "iloc", None)
    if iloc is not None:
        return iloc[indices].reset_index(drop=True)
    return np.asarray(x_batch)[indices]


def _tertile_positions(scores: Sequence[float] | np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size < len(BUCKET_NAMES):
        raise ValueError("Need at least three rows to form low/medium/high buckets.")
    if not np.isfinite(values).all():
        raise ValueError("Bucket scores must be finite.")

    order = np.argsort(values, kind="mergesort")
    groups = np.array_split(order, len(BUCKET_NAMES))
    return {
        name: np.asarray(group, dtype=int)
        for name, group in zip(BUCKET_NAMES, groups)
    }


def _sample_bucket_positions(
    buckets: Mapping[str, np.ndarray],
    *,
    n_clients: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    if n_clients <= 0:
        raise ValueError("n_clients must be positive.")
    sampled: dict[str, np.ndarray] = {}
    for name in BUCKET_NAMES:
        positions = np.asarray(buckets[name], dtype=int)
        if positions.size == 0:
            sampled[name] = positions.copy()
            continue
        n_take = min(int(n_clients), int(positions.size))
        sampled[name] = rng.choice(positions, size=n_take, replace=False).astype(int)
    return sampled


def _mean_abs_sensitivity_scores(objective: object, x_batch: object, u_grid: np.ndarray) -> np.ndarray:
    derivative_fn = getattr(objective, "_d_acceptance_du_batch", None)
    if not callable(derivative_fn):
        raise ValueError("Policy acceptance grid requires objective._d_acceptance_du_batch.")

    n_rows = _row_count(x_batch)
    scores = np.zeros(n_rows, dtype=float)
    for u_value in np.asarray(u_grid, dtype=float).reshape(-1):
        u_arr = np.full(n_rows, float(u_value), dtype=float)
        derivative = np.asarray(derivative_fn(x_batch, u_arr), dtype=float).reshape(-1)
        if derivative.shape != (n_rows,):
            raise ValueError("objective._d_acceptance_du_batch must return one value per row.")
        scores += np.abs(derivative)
    return scores / float(u_grid.size)


def _predicted_loss_scores(objective: object, x_batch: object) -> np.ndarray:
    loss_fn = getattr(objective, "_loss_prediction", None)
    if not callable(loss_fn):
        raise ValueError("Policy acceptance grid requires objective._loss_prediction.")
    loss = np.asarray(loss_fn(x_batch), dtype=float).reshape(-1)
    if loss.shape != (_row_count(x_batch),):
        raise ValueError("objective._loss_prediction must return one value per row.")
    if not np.isfinite(loss).all():
        raise ValueError("Predicted loss scores must be finite.")
    return loss


def _policy_u_values(artifact: object, x_batch: object) -> np.ndarray:
    predict_fn = getattr(artifact, "predict_u", None)
    if not callable(predict_fn):
        raise ValueError("Policy artifact must provide predict_u().")
    u_values = np.asarray(predict_fn(x_batch, clip=True), dtype=float).reshape(-1)
    if u_values.shape != (_row_count(x_batch),):
        raise ValueError("artifact.predict_u must return one value per row.")
    if not np.isfinite(u_values).all():
        raise ValueError("Policy u values must be finite.")
    return u_values


def _acceptance_values(objective: object, x_batch: object, u_values: np.ndarray) -> np.ndarray:
    acceptance_fn = getattr(objective, "_acceptance_proba", None)
    if not callable(acceptance_fn):
        raise ValueError("Policy acceptance grid requires objective._acceptance_proba.")
    u_arr = np.asarray(u_values, dtype=float).reshape(-1)
    values = np.asarray(acceptance_fn(x_batch, u_arr), dtype=float).reshape(-1)
    if values.shape != (_row_count(x_batch),):
        raise ValueError("objective._acceptance_proba must return one value per row.")
    if not np.isfinite(values).all():
        raise ValueError("Acceptance probabilities must be finite.")
    return values


def _acceptance_curve_matrix(objective: object, x_batch: object, u_grid: np.ndarray) -> np.ndarray:
    n_rows = _row_count(x_batch)
    curves = np.empty((n_rows, u_grid.size), dtype=float)
    for idx, u_value in enumerate(np.asarray(u_grid, dtype=float).reshape(-1)):
        curves[:, idx] = _acceptance_values(
            objective,
            x_batch,
            np.full(n_rows, float(u_value), dtype=float),
        )
    return curves


def _build_bucket_samples(
    *,
    objective: object,
    artifact: object,
    x_batch: object,
    row_indices: np.ndarray,
    u_grid: np.ndarray,
    score_values: np.ndarray,
    sensitivity_scores: np.ndarray,
    predicted_loss: np.ndarray,
    sampled_positions: Mapping[str, np.ndarray],
) -> list[BucketSamples]:
    samples: list[BucketSamples] = []
    for name in BUCKET_NAMES:
        positions = np.asarray(sampled_positions[name], dtype=int)
        x_sample = _slice_rows(x_batch, positions)
        policy_u = _policy_u_values(artifact, x_sample)
        policy_acceptance = _acceptance_values(objective, x_sample, policy_u)
        curves = _acceptance_curve_matrix(objective, x_sample, u_grid)
        samples.append(
            BucketSamples(
                name=name,
                positions=positions,
                row_indices=np.asarray(row_indices, dtype=int)[positions],
                score_values=np.asarray(score_values, dtype=float)[positions],
                sensitivity_scores=np.asarray(sensitivity_scores, dtype=float)[positions],
                predicted_loss=np.asarray(predicted_loss, dtype=float)[positions],
                policy_u=policy_u,
                policy_acceptance=policy_acceptance,
                acceptance_curves=curves,
            )
        )
    return samples


def _x_limits(u_grid: np.ndarray, samples: Sequence[BucketSamples]) -> tuple[float, float]:
    values = [np.asarray(u_grid, dtype=float).reshape(-1)]
    values.extend(sample.policy_u for sample in samples if sample.policy_u.size > 0)
    combined = np.concatenate(values)
    finite = combined[np.isfinite(combined)]
    x_min = float(np.min(finite))
    x_max = float(np.max(finite))
    if x_min == x_max:
        return x_min - 0.01, x_max + 0.01
    margin = 0.04 * (x_max - x_min)
    return x_min - margin, x_max + margin


def _plot_acceptance_curves(
    *,
    samples: Sequence[BucketSamples],
    u_grid: np.ndarray,
    score_label: str,
    output_path: Path,
    dpi: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(BUCKET_NAMES), figsize=(15.0, 4.7), sharey=True)
    axes_arr = np.asarray(axes, dtype=object).reshape(-1)
    x_min, x_max = _x_limits(u_grid, samples)

    for ax, sample in zip(axes_arr, samples):
        colors = plt.cm.viridis(np.linspace(0.10, 0.90, max(sample.positions.size, 1)))
        for idx in range(sample.positions.size):
            color = colors[idx]
            ax.plot(
                u_grid,
                sample.acceptance_curves[idx],
                color=color,
                linewidth=1.4,
                alpha=0.72,
            )
            ax.scatter(
                [sample.policy_u[idx]],
                [sample.policy_acceptance[idx]],
                color=color,
                edgecolor="#111111",
                linewidth=0.6,
                s=38.0,
                zorder=4,
            )

        if sample.score_values.size > 0:
            score_min = float(np.min(sample.score_values))
            score_max = float(np.max(sample.score_values))
            score_text = f"{score_label}: {score_min:.4g}-{score_max:.4g}"
        else:
            score_text = f"{score_label}: n/a"
        ax.set_title(f"{sample.name.capitalize()} bucket\n{score_text}")
        ax.set_xlabel("Proposed increase u")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(True, alpha=0.28)
        ax.axvline(float(u_grid[0]), color="#969696", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.axvline(float(u_grid[-1]), color="#969696", linewidth=0.8, linestyle=":", alpha=0.6)

    axes_arr[0].set_ylabel("Predicted acceptance probability")
    handles = [
        Line2D([0], [0], color="#4c78a8", linewidth=1.5, label="sampled client curve"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor="#4c78a8",
            markeredgecolor="#111111",
            markersize=6,
            linewidth=0.0,
            label="artifact policy u",
        ),
        Line2D([0], [0], color="#969696", linestyle=":", linewidth=0.8, label="simulated range"),
    ]
    axes_arr[-1].legend(handles=handles, loc="lower left", fontsize="small")
    fig.suptitle(f"Client-level acceptance curves by {score_label}", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_sample_csv(
    rows_by_plot: Mapping[str, Sequence[BucketSamples]],
    output_path: Path,
    *,
    u_grid: np.ndarray,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "plot",
        "bucket",
        "bucket_rank",
        "selection_order",
        "split_position",
        "csv_row_index",
        "representative_score",
        "sensitivity_score",
        "predicted_loss",
        "policy_u",
        "policy_acceptance",
        "policy_u_in_simulated_range",
        "acceptance_at_u_min",
        "acceptance_at_u_max",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for plot_name, samples in rows_by_plot.items():
            for bucket_rank, sample in enumerate(samples):
                for idx in range(sample.positions.size):
                    writer.writerow(
                        {
                            "plot": plot_name,
                            "bucket": sample.name,
                            "bucket_rank": bucket_rank,
                            "selection_order": idx,
                            "split_position": int(sample.positions[idx]),
                            "csv_row_index": int(sample.row_indices[idx]),
                            "representative_score": float(sample.score_values[idx]),
                            "sensitivity_score": float(sample.sensitivity_scores[idx]),
                            "predicted_loss": float(sample.predicted_loss[idx]),
                            "policy_u": float(sample.policy_u[idx]),
                            "policy_acceptance": float(sample.policy_acceptance[idx]),
                            "policy_u_in_simulated_range": bool(
                                float(u_grid[0]) <= float(sample.policy_u[idx]) <= float(u_grid[-1])
                            ),
                            "acceptance_at_u_min": float(sample.acceptance_curves[idx, 0]),
                            "acceptance_at_u_max": float(sample.acceptance_curves[idx, -1]),
                        }
                    )
    return output_path


def _summary_values(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _write_summary_json(
    output_path: Path,
    *,
    artifact_path: Path,
    artifact: object,
    split: str,
    n_rows: int,
    n_clients: int,
    seed: int | None,
    u_grid: np.ndarray,
    sensitivity_scores: np.ndarray,
    predicted_loss: np.ndarray,
    plot_paths: Mapping[str, Path],
    sample_csv: Path,
) -> Path:
    payload = {
        "artifact_path": str(artifact_path),
        "estimator": str(getattr(artifact, "estimator", "artifact")),
        "split": split,
        "n_rows_scored": int(n_rows),
        "n_clients_per_bucket": int(n_clients),
        "seed": seed,
        "u_min": float(u_grid[0]),
        "u_max": float(u_grid[-1]),
        "u_count": int(u_grid.size),
        "sensitivity_score": _summary_values(sensitivity_scores),
        "predicted_loss": _summary_values(predicted_loss),
        "outputs": {name: str(path) for name, path in plot_paths.items()},
        "sample_csv": str(sample_csv),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_root() / "policy-acceptance-grid" / f"acceptance_grid_{timestamp}"


def run_acceptance_grid(args: argparse.Namespace) -> dict[str, Path]:
    artifact_path = _artifact_json_path(Path(args.policy_artifact))
    artifact = load_policy_artifact(artifact_path)
    objective = artifact.build_objective()
    x_batch = artifact.load_x(split=args.split)
    row_indices = np.asarray(artifact.row_indices(args.split), dtype=int)
    n_rows = _row_count(x_batch)
    if row_indices.shape != (n_rows,):
        raise ValueError("Artifact row_indices must match loaded split row count.")

    u_grid = _resolve_u_grid(args.u_min, args.u_max, args.u_count)
    sensitivity_scores = _mean_abs_sensitivity_scores(objective, x_batch, u_grid)
    predicted_loss = _predicted_loss_scores(objective, x_batch)
    rng = np.random.default_rng(args.seed)

    sensitivity_samples = _build_bucket_samples(
        objective=objective,
        artifact=artifact,
        x_batch=x_batch,
        row_indices=row_indices,
        u_grid=u_grid,
        score_values=sensitivity_scores,
        sensitivity_scores=sensitivity_scores,
        predicted_loss=predicted_loss,
        sampled_positions=_sample_bucket_positions(
            _tertile_positions(sensitivity_scores),
            n_clients=int(args.n_clients),
            rng=rng,
        ),
    )
    loss_samples = _build_bucket_samples(
        objective=objective,
        artifact=artifact,
        x_batch=x_batch,
        row_indices=row_indices,
        u_grid=u_grid,
        score_values=predicted_loss,
        sensitivity_scores=sensitivity_scores,
        predicted_loss=predicted_loss,
        sampled_positions=_sample_bucket_positions(
            _tertile_positions(predicted_loss),
            n_clients=int(args.n_clients),
            rng=rng,
        ),
    )

    output_dir = Path(args.output_dir) if args.output_dir is not None else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_plot = _plot_acceptance_curves(
        samples=sensitivity_samples,
        u_grid=u_grid,
        score_label="mean absolute sensitivity",
        output_path=output_dir / "acceptance_curves_by_sensitivity.png",
        dpi=int(args.dpi),
    )
    loss_plot = _plot_acceptance_curves(
        samples=loss_samples,
        u_grid=u_grid,
        score_label="predicted loss",
        output_path=output_dir / "acceptance_curves_by_predicted_loss.png",
        dpi=int(args.dpi),
    )
    sample_csv = _write_sample_csv(
        {
            "sensitivity": sensitivity_samples,
            "predicted_loss": loss_samples,
        },
        output_dir / "sampled_clients.csv",
        u_grid=u_grid,
    )
    summary_json = _write_summary_json(
        output_dir / "acceptance_grid_summary.json",
        artifact_path=artifact_path,
        artifact=artifact,
        split=str(args.split),
        n_rows=n_rows,
        n_clients=int(args.n_clients),
        seed=args.seed,
        u_grid=u_grid,
        sensitivity_scores=sensitivity_scores,
        predicted_loss=predicted_loss,
        plot_paths={
            "sensitivity": sensitivity_plot,
            "predicted_loss": loss_plot,
        },
        sample_csv=sample_csv,
    )
    return {
        "sensitivity_plot": sensitivity_plot,
        "predicted_loss_plot": loss_plot,
        "sample_csv": sample_csv,
        "summary_json": summary_json,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot sampled client-level acceptance curves over simulated price increases "
            "and overlay a saved policy artifact's optimized action."
        )
    )
    parser.add_argument(
        "--policy-artifact",
        type=Path,
        required=True,
        help="Saved policies/<estimator>/policy.json artifact, or its containing directory.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="all",
        help="Artifact row split to score and sample. Defaults to all.",
    )
    parser.add_argument("--u-min", type=float, default=0.0, help="Minimum simulated increase u.")
    parser.add_argument("--u-max", type=float, default=0.15, help="Maximum simulated increase u.")
    parser.add_argument("--u-count", type=int, default=61, help="Number of simulated u grid points.")
    parser.add_argument(
        "--n-clients",
        type=int,
        default=10,
        help="Number of clients sampled per low/medium/high bucket. Defaults to 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible bucket sampling. Omit to sample new clients each run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to timestamped results_root()/policy-acceptance-grid/ directory.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Saved plot DPI.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    outputs = run_acceptance_grid(args)
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
