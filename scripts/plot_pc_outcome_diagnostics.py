"""Plot processed policy component diagnostics against acceptance and loss."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.loader import FEATURE_COLS_GLM, FEATURE_COLS_XGB, load_x_array, sample_csv_row_indices
from experiments.configs import get_config


@dataclass(frozen=True)
class DiagnosticData:
    """Per-row quantities used by the diagnostic plots."""

    components: np.ndarray
    component_names: tuple[str, ...]
    u: np.ndarray
    acceptance: np.ndarray
    loss: np.ndarray
    premium: np.ndarray
    per_sample_objective: np.ndarray


def _safe_corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return Pearson r, or nan when either vector is constant/invalid."""
    x_arr = np.asarray(x_values, dtype=float).reshape(-1)
    y_arr = np.asarray(y_values, dtype=float).reshape(-1)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.sum(valid)) < 2:
        return float("nan")
    x_valid = x_arr[valid]
    y_valid = y_arr[valid]
    if np.allclose(x_valid, x_valid[0]) or np.allclose(y_valid, y_valid[0]):
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def _load_summary_theta(summary_json: Path, estimator: str) -> np.ndarray:
    with summary_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        theta = payload["estimators"][estimator]["theta"]
    except KeyError as exc:
        available = sorted(payload.get("estimators", {}).keys())
        raise ValueError(
            f"Estimator '{estimator}' not found in {summary_json}. Available: {available}."
        ) from exc
    return np.asarray(theta, dtype=float)


def _infer_model_type(config: object) -> Literal["glm", "xgb"]:
    state_dim = int(getattr(config, "state_dim"))
    if state_dim == len(FEATURE_COLS_GLM):
        return "glm"
    if state_dim == len(FEATURE_COLS_XGB):
        return "xgb"
    raise ValueError("Could not infer model type from config.state_dim.")


def _resolve_x_array(
    config: object,
    *,
    n_rows: int | None,
    row_seed: int | None,
    model_type: Literal["glm", "xgb"] | None,
) -> np.ndarray:
    if n_rows is not None and n_rows <= 0:
        raise ValueError("n_rows must be positive when provided.")
    if row_seed is not None:
        resolved_model_type = model_type or _infer_model_type(config)
        rows = int(n_rows if n_rows is not None else getattr(config, "n_samples"))
        row_indices = sample_csv_row_indices(resolved_model_type, n_rows=rows, seed=int(row_seed))
        return load_x_array(resolved_model_type, row_indices=row_indices)

    x_fixed = getattr(config, "x_fixed", None)
    if x_fixed is None:
        raise ValueError(
            "Config does not provide x_fixed. Pass --row-seed/--model-type to sample real-data rows."
        )
    x_arr = np.asarray(x_fixed, dtype=float)
    if n_rows is not None:
        x_arr = x_arr[: int(n_rows)]
    if x_arr.shape[0] == 0:
        raise ValueError("No rows available for diagnostics.")
    return x_arr


def _component_names(objective: object, n_components: int) -> tuple[str, ...]:
    artifact_preprocessor = getattr(objective, "_artifact_preprocessor", None)
    acceptance_model = getattr(objective, "acceptance_model", None)
    names: Sequence[str] | None = None
    if callable(artifact_preprocessor) and acceptance_model is not None:
        preprocessor = artifact_preprocessor(acceptance_model)
        names = getattr(preprocessor, "output_feature_names_", None)
    if names is None:
        names = [f"component_{idx + 1}" for idx in range(n_components)]
    cleaned = tuple(str(name) for name in names[:n_components])
    if len(cleaned) < n_components:
        cleaned = (*cleaned, *(f"component_{idx + 1}" for idx in range(len(cleaned), n_components)))
    return cleaned


def build_diagnostic_data(config: object, theta: np.ndarray, x_batch: np.ndarray) -> DiagnosticData:
    """Compute per-row processed components, actions, acceptance, and loss."""
    objective = getattr(config, "objective")
    policy_theta_dim = getattr(objective, "policy_theta_dim", None)
    if callable(policy_theta_dim):
        expected_dim = int(policy_theta_dim())
        if int(theta.size) != expected_dim:
            raise ValueError(f"theta has {theta.size} entries; expected {expected_dim} for this preset.")

    policy_features = getattr(objective, "_policy_features", None)
    acceptance_proba = getattr(objective, "_acceptance_proba", None)
    loss_prediction = getattr(objective, "_loss_prediction", None)
    clip_u = getattr(objective, "_clip_u", None)
    if not callable(policy_features) or not callable(acceptance_proba) or not callable(loss_prediction):
        raise ValueError("Diagnostic script requires a ModelBasedObjective-like objective.")

    x_arr = np.asarray(x_batch, dtype=float)
    theta_arr = np.asarray(theta, dtype=float)
    components = np.asarray(policy_features(x_arr), dtype=float)
    u_raw = np.asarray(objective.policy_value(theta_arr, x_arr), dtype=float)
    u = np.asarray(clip_u(u_raw), dtype=float) if callable(clip_u) else u_raw
    acceptance = np.asarray(acceptance_proba(x_arr, u), dtype=float)
    loss = np.asarray(loss_prediction(x_arr), dtype=float)
    premium_col = int(getattr(objective, "premium_col"))
    premium = x_arr[:, premium_col]
    revenue = (u + 1.0) * premium
    per_sample_objective = acceptance * (loss - revenue)
    return DiagnosticData(
        components=components,
        component_names=_component_names(objective, components.shape[1]),
        u=u,
        acceptance=acceptance,
        loss=loss,
        premium=premium,
        per_sample_objective=per_sample_objective,
    )


def _sample_indices(n_rows: int, max_points: int | None, seed: int) -> np.ndarray:
    if max_points is None or max_points >= n_rows:
        return np.arange(n_rows, dtype=int)
    if max_points <= 0:
        raise ValueError("max_points must be positive when provided.")
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_rows, size=int(max_points), replace=False))


def _plot_component_scatter_grid(
    components: np.ndarray,
    component_names: Sequence[str],
    y_values: np.ndarray,
    *,
    y_label: str,
    title: str,
    output_path: Path,
    max_components: int,
    point_indices: np.ndarray,
) -> Path:
    n_components = min(int(max_components), components.shape[1])
    if n_components <= 0:
        raise ValueError("At least one component is required for plotting.")
    n_cols = 2
    n_rows = int(np.ceil(n_components / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.0, max(3.4, 3.0 * n_rows)))
    axes_arr = np.asarray(axes).reshape(-1)
    y_sample = np.asarray(y_values, dtype=float)[point_indices]
    for idx in range(n_components):
        ax = axes_arr[idx]
        x_sample = components[point_indices, idx]
        corr = _safe_corr(components[:, idx], y_values)
        ax.scatter(x_sample, y_sample, s=8.0, alpha=0.25, linewidths=0.0, rasterized=True)
        corr_text = "nan" if not np.isfinite(corr) else f"{corr:.3f}"
        ax.set_title(f"{component_names[idx]} vs {y_label} (r={corr_text})", fontsize=9)
        ax.set_xlabel(component_names[idx])
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
    for ax in axes_arr[n_components:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_u_vs_acceptance(
    data: DiagnosticData,
    *,
    output_path: Path,
    point_indices: np.ndarray,
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.2))
    scatter = ax.scatter(
        data.u[point_indices],
        data.acceptance[point_indices],
        c=data.loss[point_indices],
        cmap="viridis",
        s=10.0,
        alpha=0.45,
        linewidths=0.0,
        rasterized=True,
    )
    corr = _safe_corr(data.u, data.acceptance)
    corr_text = "constant action" if not np.isfinite(corr) else f"r={corr:.3f}"
    ax.set_title(f"final u vs acceptance ({corr_text})")
    ax.set_xlabel("final policy u")
    ax.set_ylabel("acceptance probability f_acc")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="predicted loss")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write_correlations(data: DiagnosticData, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = {
        "u": data.u,
        "f_acc": data.acceptance,
        "loss": data.loss,
        "premium": data.premium,
        "per_sample_objective": data.per_sample_objective,
    }
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["component_index", "component_name", *targets.keys()],
        )
        writer.writeheader()
        for idx, component_name in enumerate(data.component_names):
            row: dict[str, object] = {
                "component_index": idx + 1,
                "component_name": component_name,
            }
            for target_name, target_values in targets.items():
                corr = _safe_corr(data.components[:, idx], target_values)
                row[target_name] = "" if not np.isfinite(corr) else corr
            writer.writerow(row)
    return output_path


def write_diagnostic_plots(
    data: DiagnosticData,
    output_dir: Path,
    *,
    max_components: int,
    max_points: int | None,
    sample_seed: int,
) -> list[Path]:
    """Write scatter plots and a per-component correlation CSV."""
    point_indices = _sample_indices(data.components.shape[0], max_points, sample_seed)
    outputs = [
        _plot_component_scatter_grid(
            data.components,
            data.component_names,
            data.acceptance,
            y_label="f_acc",
            title="Processed policy components vs final acceptance",
            output_path=output_dir / "pc_vs_acceptance.png",
            max_components=max_components,
            point_indices=point_indices,
        ),
        _plot_component_scatter_grid(
            data.components,
            data.component_names,
            data.loss,
            y_label="loss",
            title="Processed policy components vs predicted loss",
            output_path=output_dir / "pc_vs_loss.png",
            max_components=max_components,
            point_indices=point_indices,
        ),
        _plot_component_scatter_grid(
            data.components,
            data.component_names,
            data.u,
            y_label="u",
            title="Processed policy components vs final policy action",
            output_path=output_dir / "pc_vs_u.png",
            max_components=max_components,
            point_indices=point_indices,
        ),
        _plot_u_vs_acceptance(data, output_path=output_dir / "u_vs_acceptance.png", point_indices=point_indices),
        _write_correlations(data, output_dir / "pc_diagnostic_correlations.csv"),
    ]
    return outputs


def _default_output_dir(summary_json: Path, estimator: str) -> Path:
    return summary_json.parent / "pc_outcome_diagnostics" / estimator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot final-policy processed-component diagnostics against f_acc, loss, and u."
    )
    parser.add_argument("--preset", required=True, help="Config preset used to rebuild the objective.")
    parser.add_argument("--policy-kind", default=None, help="Optional real-data config policy_kind override.")
    parser.add_argument("--feature-order", default=None, help="Optional real-data config feature_order override.")
    parser.add_argument(
        "--policy-preprocessing",
        default=None,
        help="Optional real-data config policy_preprocessing override.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Path to a run summary.json containing final estimator theta.",
    )
    parser.add_argument("--estimator", default="first_order", help="Estimator theta to read from summary.json.")
    parser.add_argument("--max-components", type=int, default=8, help="Number of processed components to plot.")
    parser.add_argument("--max-points", type=int, default=5000, help="Maximum scatter points to draw per plot.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for scatter downsampling.")
    parser.add_argument("--n-rows", type=int, default=None, help="Limit rows, or row count when --row-seed is set.")
    parser.add_argument(
        "--row-seed",
        type=int,
        default=None,
        help="Regenerate a real-data row sample with this seed instead of using config.x_fixed.",
    )
    parser.add_argument(
        "--model-type",
        choices=("glm", "xgb"),
        default=None,
        help="Model type for --row-seed sampling; inferred from preset when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots. Defaults beside summary.json under pc_outcome_diagnostics/<estimator>/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "policy_kind": args.policy_kind,
            "feature_order": args.feature_order,
            "policy_preprocessing": args.policy_preprocessing,
        }.items()
        if value is not None
    }
    config = get_config(args.preset, overrides=overrides)
    theta = _load_summary_theta(args.summary_json, args.estimator)
    x_batch = _resolve_x_array(
        config,
        n_rows=args.n_rows,
        row_seed=args.row_seed,
        model_type=args.model_type,
    )
    output_dir = args.output_dir if args.output_dir is not None else _default_output_dir(args.summary_json, args.estimator)
    data = build_diagnostic_data(config, theta, x_batch)
    outputs = write_diagnostic_plots(
        data,
        output_dir,
        max_components=args.max_components,
        max_points=args.max_points,
        sample_seed=args.sample_seed,
    )
    print(f"Read theta for estimator '{args.estimator}' from {args.summary_json}.")
    print(f"Computed diagnostics on {x_batch.shape[0]} rows and {data.components.shape[1]} processed components.")
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
