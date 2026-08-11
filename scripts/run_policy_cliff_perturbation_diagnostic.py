#!/usr/bin/env python3
"""Run the 200-customer policy-output and local-price perturbation diagnostic.

The cohort is defined by the curves stored in model_processing's monotone
smoothing wrapper.  The raw and spline acceptance arms share the wrapper's
embedded XGBoost model, and every arm shares the same XGBoost loss artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PPoly


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.dataset_metadata import (  # noqa: E402
    ACCEPTANCE_STATE_COLS,
    LOSS_FEATURE_COLS,
    OBSERVED_CHURN_COL,
    OBSERVED_U_COL,
    PREMIUM_COL,
)
from data.feature_processor import FeatureProcessor  # noqa: E402
from data.loader import ModelArtifactBundle, dataset_csv_path  # noqa: E402
from experiments.config import (  # noqa: E402
    CorrectnessSpec,
    build_experiment_config,
    canonical_runtime_block,
    canonical_training_block,
    make_model_based_objective,
)
from experiments.execution import execute_experiment_run  # noqa: E402
from experiments.paths import results_root  # noqa: E402
from experiments.reporting.base import ReporterStack  # noqa: E402
from experiments.reporting.console import ConsoleReporter  # noqa: E402
from experiments.reporting.context import create_run_context  # noqa: E402
from experiments.reporting.json_summary import JsonReporter  # noqa: E402
from experiments.reporting.step_logger import FileStepLogger  # noqa: E402
from objective.policy import SoftmaxPolicy  # noqa: E402


DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT.parent.parent / "model_processing" / "artifacts"
)
DEFAULT_WRAPPER_NAME = "acceptance_smoothing_wrapper_monotone_smoothing_spline.pkl"
DEFAULT_LOSS_NAME = "financial_loss_model_xgb.pkl"
DEFAULT_DELTAS = (-0.0025, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.0025)
MODEL_COLORS = {"xgboost": "#d95f02", "spline": "#1b9e77"}
MODE_STYLES = {"unconstrained": "-", "acceptance_penalty": "--"}


class _PickledState:
    """State-only compatibility target for legacy wrapper classes."""

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(state)


class _ModelProcessingUnpickler(pickle.Unpickler):
    """Load model_processing artifacts without importing that repository."""

    def find_class(self, module: str, name: str) -> Any:
        if name == "FeatureProcessor" and module in {"preprocessing", "__main__"}:
            return FeatureProcessor
        if module == "black_box_objective" and name == "SmoothedXGBoostWrapper":
            return _PickledState
        if module == "src.smoothing.curve_specifications" and name.startswith("_"):
            return _PickledState
        return super().find_class(module, name)


@dataclass(frozen=True)
class StoredMonotoneCurve:
    """Portable evaluation of one stored model_processing churn curve."""

    polynomial: PPoly
    x_min: float
    p_min: float
    x_max: float
    p_max: float
    slope_p: float

    def value(self, u: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=float)
        result = np.empty_like(u_arr)
        below = u_arr < self.x_min
        above = u_arr > self.x_max
        inside = ~(below | above)
        result[below] = self.p_min
        result[above] = np.clip(
            self.p_max + self.slope_p * (u_arr[above] - self.x_max), 0.0, 1.0
        )
        result[inside] = np.clip(self.polynomial(u_arr[inside]), 0.0, 1.0)
        return result

    def derivative(self, u: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=float)
        result = np.zeros_like(u_arr)
        above = u_arr > self.x_max
        inside = (u_arr >= self.x_min) & (u_arr <= self.x_max)
        if np.any(inside):
            raw = np.asarray(self.polynomial.derivative()(u_arr[inside]), dtype=float)
            values = np.asarray(self.polynomial(u_arr[inside]), dtype=float)
            result[inside] = np.where((values > 0.0) & (values < 1.0), raw, 0.0)
        if np.any(above):
            extrapolated = self.p_max + self.slope_p * (u_arr[above] - self.x_max)
            result[above] = np.where(
                (extrapolated > 0.0) & (extrapolated < 1.0), self.slope_p, 0.0
            )
        return result


@dataclass(frozen=True)
class StoredMonotoneAcceptance:
    """Acceptance hook over the exact policies in a stored spline wrapper."""

    raw_artifact: ModelArtifactBundle
    curves: Mapping[str, StoredMonotoneCurve]
    artifact_path: str
    probability_target: str = "acceptance"
    artifact_id: str = "model_processing_monotone_spline_xgb"
    model_type: str = "monotone_spline_xgb"
    role: str = "acceptance"
    auxiliary_state_cols: tuple[str, ...] = ("id",)

    @property
    def preprocessor(self) -> Any:
        return self.raw_artifact.preprocessor

    @property
    def x_feature_cols(self) -> tuple[str, ...]:
        return self.raw_artifact.x_feature_cols

    def policy_feature_dim(self) -> int:
        return self.raw_artifact.policy_feature_dim()

    def _curve_rows(self, frame: pd.DataFrame) -> list[StoredMonotoneCurve]:
        if "id" not in frame.columns:
            raise ValueError("Stored spline evaluation requires an 'id' column.")
        ids = [_normalize_id(value) for value in frame["id"]]
        missing = sorted({policy_id for policy_id in ids if policy_id not in self.curves})
        if missing:
            raise ValueError(
                "The diagnostic cohort must contain only stored spline policies; "
                f"missing {len(missing)} curve IDs."
            )
        return [self.curves[policy_id] for policy_id in ids]

    def predict_acceptance(self, frame: pd.DataFrame, u: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=float).reshape(-1)
        if u_arr.shape != (len(frame),):
            raise ValueError("u must contain one action per customer.")
        curves = self._curve_rows(frame)
        churn = np.asarray(
            [curve.value(np.asarray([u_value]))[0] for curve, u_value in zip(curves, u_arr)],
            dtype=float,
        )
        return np.clip(1.0 - churn, 0.0, 1.0)

    def d_acceptance_du(self, frame: pd.DataFrame, u: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=float).reshape(-1)
        curves = self._curve_rows(frame)
        return -np.asarray(
            [curve.derivative(np.asarray([u_value]))[0] for curve, u_value in zip(curves, u_arr)],
            dtype=float,
        )


@dataclass(frozen=True)
class DiagnosticArtifacts:
    raw_acceptance: ModelArtifactBundle
    spline_acceptance: StoredMonotoneAcceptance
    loss: ModelArtifactBundle
    curve_ids: tuple[str, ...]
    hashes: Mapping[str, str]


@dataclass(frozen=True)
class Cohort:
    frame: pd.DataFrame
    row_indices: np.ndarray
    observed_u: np.ndarray
    observed_acceptance: float
    duplicate_match_ids: tuple[str, ...]
    imputed_cells: tuple[str, ...]


def _normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as stream:
        return _ModelProcessingUnpickler(stream).load()


def _rehydrate_curve(curve: Any) -> StoredMonotoneCurve:
    state = curve.interp.__getstate__()[1]
    polynomial = PPoly(
        np.asarray(state["_c"], dtype=float),
        np.asarray(state["_x"], dtype=float),
        extrapolate=bool(state.get("extrapolate", True)),
        axis=int(state.get("axis", 0)),
    )
    return StoredMonotoneCurve(
        polynomial=polynomial,
        x_min=float(curve.x_min),
        p_min=float(curve.p_min),
        x_max=float(curve.x_max),
        p_max=float(curve.p_max),
        slope_p=float(curve.slope_p),
    )


def load_diagnostic_artifacts(
    artifact_dir: Path,
    *,
    wrapper_name: str = DEFAULT_WRAPPER_NAME,
    loss_name: str = DEFAULT_LOSS_NAME,
    xgb_n_jobs: int = 1,
) -> DiagnosticArtifacts:
    """Load and normalize the external wrapper and shared loss model."""
    wrapper_path = artifact_dir / wrapper_name
    loss_path = artifact_dir / loss_name
    for path in (wrapper_path, loss_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required model_processing artifact not found: {path}")

    wrapper = _load_pickle(wrapper_path)
    curves_state = getattr(wrapper, "_curves", None)
    prep = getattr(wrapper, "_prep", None)
    model = getattr(wrapper, "_model", None)
    if not isinstance(curves_state, Mapping) or len(curves_state) != 200:
        raise ValueError("Expected the monotone smoothing wrapper to contain exactly 200 curves.")
    if not isinstance(prep, Mapping) or model is None:
        raise ValueError("Unexpected monotone smoothing wrapper layout.")
    if callable(getattr(model, "set_params", None)):
        model.set_params(n_jobs=int(xgb_n_jobs))

    x_feature_cols = tuple(prep.get("x_feature_cols", ()))
    if not x_feature_cols:
        raise ValueError("Wrapper preprocessor does not record x_feature_cols.")
    raw_acceptance = ModelArtifactBundle(
        model=model,
        preprocessor=prep.get("preprocessor"),
        u_cols=("U",),
        x_feature_cols=x_feature_cols,
        probability_target="acceptance",
        source_format="model_processing_embedded_best_fold",
        model_type="xgb",
        artifact_id="model_processing_embedded_xgb",
        role="acceptance",
        artifact_path=str(wrapper_path),
    )
    object.__setattr__(raw_acceptance, "auxiliary_state_cols", ("id",))
    curves = {
        _normalize_id(policy_id): _rehydrate_curve(curve)
        for policy_id, curve in curves_state.items()
    }
    spline_acceptance = StoredMonotoneAcceptance(
        raw_artifact=raw_acceptance,
        curves=curves,
        artifact_path=str(wrapper_path),
    )

    loss_payload = _load_pickle(loss_path)
    if not isinstance(loss_payload, Mapping) or "model" not in loss_payload:
        raise ValueError("Unexpected XGBoost loss artifact layout.")
    if callable(getattr(loss_payload["model"], "set_params", None)):
        loss_payload["model"].set_params(n_jobs=int(xgb_n_jobs))
    loss_prep = loss_payload.get("preprocessor")
    x_loss_cols: Sequence[str] = ()
    preprocessor = loss_prep
    if isinstance(loss_prep, Mapping):
        preprocessor = loss_prep.get("preprocessor")
        x_loss_cols = tuple(
            loss_prep.get("x_feature_cols", loss_prep.get("feature_cols", ()))
        )
    if not x_loss_cols:
        x_loss_cols = tuple(loss_payload.get("model_features", ()))
    if not x_loss_cols:
        raise ValueError("Loss artifact does not record model features.")
    loss = ModelArtifactBundle(
        model=loss_payload["model"],
        preprocessor=preprocessor,
        u_cols=(),
        x_feature_cols=tuple(x_loss_cols),
        probability_target="none",
        source_format="model_processing_selected_best_fold",
        model_type="xgb",
        artifact_id="model_processing_financial_loss_xgb",
        role="loss",
        artifact_path=str(loss_path),
    )
    return DiagnosticArtifacts(
        raw_acceptance=raw_acceptance,
        spline_acceptance=spline_acceptance,
        loss=loss,
        curve_ids=tuple(curves),
        hashes={wrapper_path.name: _sha256(wrapper_path), loss_path.name: _sha256(loss_path)},
    )


def load_curve_cohort(
    csv_path: Path,
    curve_ids: Sequence[str],
    *,
    numeric_imputation_values: Mapping[str, float] | None = None,
) -> Cohort:
    """Resolve one canonical dataset row for every stored curve ID."""
    required = list(
        dict.fromkeys(
            [
                "id",
                OBSERVED_U_COL,
                OBSERVED_CHURN_COL,
                *ACCEPTANCE_STATE_COLS,
                *LOSS_FEATURE_COLS,
            ]
        )
    )
    source = pd.read_csv(csv_path, sep=";", usecols=required)
    source["_curve_id"] = source["id"].map(_normalize_id)
    source["_csv_row_index"] = np.arange(len(source), dtype=int)
    groups = source.groupby("_curve_id", sort=False).indices
    missing = [policy_id for policy_id in curve_ids if policy_id not in groups]
    if missing:
        raise ValueError(f"Canonical dataset is missing {len(missing)} stored curve IDs.")
    duplicate_ids = tuple(
        policy_id for policy_id in curve_ids if len(groups[policy_id]) > 1
    )
    selected_positions = [int(groups[policy_id][0]) for policy_id in curve_ids]
    selected = source.iloc[selected_positions].reset_index(drop=True)
    model_columns = list(dict.fromkeys([*ACCEPTANCE_STATE_COLS, *LOSS_FEATURE_COLS]))
    imputed_cells: list[str] = []
    if numeric_imputation_values is not None:
        for column in model_columns:
            missing = selected[column].isna()
            if not np.any(missing):
                continue
            if column not in numeric_imputation_values:
                continue
            selected.loc[missing, column] = float(numeric_imputation_values[column])
            imputed_cells.extend(
                f"{_normalize_id(selected.loc[index, 'id'])}:{column}"
                for index in selected.index[missing]
            )
    if selected[model_columns].isna().any().any():
        raise ValueError("Stored curve cohort contains missing model features.")
    return Cohort(
        frame=selected.loc[:, ["id", *ACCEPTANCE_STATE_COLS]].copy(),
        row_indices=selected["_csv_row_index"].to_numpy(dtype=int),
        observed_u=selected[OBSERVED_U_COL].to_numpy(dtype=float),
        observed_acceptance=float(
            np.mean(1.0 - selected[OBSERVED_CHURN_COL].to_numpy(dtype=float))
        ),
        duplicate_match_ids=duplicate_ids,
        imputed_cells=tuple(imputed_cells),
    )


def _initial_theta(policy: SoftmaxPolicy, input_dim: int, initial_u: float) -> np.ndarray:
    if not policy.action_low < initial_u < policy.action_high:
        raise ValueError("initial_u must lie strictly within the policy action bounds.")
    theta = np.zeros(policy.theta_dim(input_dim), dtype=float)
    fraction = (initial_u - policy.action_low) / policy.action_span
    theta[0] = np.log(fraction / (1.0 - fraction))
    return theta


def build_diagnostic_config(
    *,
    acceptance_model: Any,
    loss_model: ModelArtifactBundle,
    cohort: Cohort,
    constrained: bool,
    acceptance_floor: float,
    penalty_weight: float,
    penalty_temperature: float,
    u_bounds: tuple[float, float],
    initial_u: float,
    fd_eps: float,
    t_steps: int,
    seed: int,
    verbose: bool,
):
    """Build one run while preserving the repository's standard objective/optimizer path."""
    policy = SoftmaxPolicy(action_low=u_bounds[0], action_high=u_bounds[1])
    objective = make_model_based_objective(
        policy=policy,
        acceptance_model=acceptance_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=PREMIUM_COL,
        u_bounds=u_bounds,
    )
    object.__setattr__(objective, "_fd_eps", float(fd_eps))
    theta0 = _initial_theta(policy, objective.policy_input_dim(), initial_u)
    floor = acceptance_floor if constrained else None
    training = canonical_training_block(
        n_samples=len(cohort.frame),
        step_rule="l-bfgs-b",
        t_steps=t_steps,
        step_size=0.01,
        sigma=fd_eps,
        n_grad_samples=1,
        enabled_estimators=("first_order",),
        perturbation_space="u",
        grad_norm_tol=1e-6,
        acceptance_floor=floor,
        acceptance_penalty_weight=penalty_weight if constrained else None,
        acceptance_penalty_temperature=penalty_temperature,
    )
    runtime = canonical_runtime_block(
        plot=False,
        verbose=verbose,
        wandb_enabled=False,
        wandb_mode="disabled",
    )
    return build_experiment_config(
        seed=seed,
        state_dim=len(ACCEPTANCE_STATE_COLS),
        x_fixed=cohort.frame,
        x_fixed_row_indices=cohort.row_indices,
        objective=objective,
        theta0=theta0,
        training=training,
        runtime=runtime,
        correctness=CorrectnessSpec(gradient_source="none"),
    )


def _reporter_stack(config: Any) -> ReporterStack:
    # External model_processing artifacts are deliberately not written as canonical
    # replayable policy artifacts; the aggregate provenance file records them instead.
    return ReporterStack(
        [
            ConsoleReporter(verbose=config.verbose),
            FileStepLogger(),
            JsonReporter(),
        ]
    )


def evaluate_actions(
    objective: Any,
    frame: pd.DataFrame,
    actions: np.ndarray,
) -> pd.DataFrame:
    """Return row-level predictions and objective contributions for actions."""
    u = np.asarray(actions, dtype=float).reshape(-1)
    acceptance = np.asarray(objective._acceptance_proba(frame, u), dtype=float)
    loss = np.asarray(objective._loss_prediction(frame), dtype=float)
    premium = np.asarray(objective._premium_values(frame), dtype=float)
    revenue = (1.0 + u) * premium
    objective_contribution = acceptance * (loss - revenue)
    return pd.DataFrame(
        {
            "id": frame["id"].map(_normalize_id),
            "u": u,
            "acceptance": acceptance,
            "predicted_loss": loss,
            "predicted_revenue": revenue,
            "objective_contribution": objective_contribution,
        }
    )


def perturb_policy_actions(
    objective: Any,
    frame: pd.DataFrame,
    optimized_u: np.ndarray,
    deltas: Iterable[float],
    *,
    u_bounds: tuple[float, float],
    acceptance_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay clipped additive price perturbations and summarize their effects."""
    row_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    base_u = np.asarray(optimized_u, dtype=float)
    for delta in deltas:
        perturbed_u = np.clip(base_u + float(delta), *u_bounds)
        rows = evaluate_actions(objective, frame, perturbed_u)
        rows.insert(1, "delta_u", float(delta))
        rows["realized_delta_u"] = perturbed_u - base_u
        row_tables.append(rows)
        mean_acceptance = float(rows["acceptance"].mean())
        mean_objective = float(rows["objective_contribution"].mean())
        summary_rows.append(
            {
                "delta_u": float(delta),
                "mean_realized_delta_u": float(np.mean(perturbed_u - base_u)),
                "mean_u": float(np.mean(perturbed_u)),
                "mean_acceptance": mean_acceptance,
                "mean_objective": mean_objective,
                "objective_sum": float(rows["objective_contribution"].sum()),
                "acceptance_slack": mean_acceptance - float(acceptance_floor),
                "violates_acceptance_floor": bool(mean_acceptance < acceptance_floor),
                "clipped_lower_count": int(np.count_nonzero(perturbed_u <= u_bounds[0] + 1e-12)),
                "clipped_upper_count": int(np.count_nonzero(perturbed_u >= u_bounds[1] - 1e-12)),
            }
        )
    row_table = pd.concat(row_tables, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("delta_u").reset_index(drop=True)
    baseline = summary.loc[np.isclose(summary["delta_u"], 0.0)]
    if len(baseline) != 1:
        raise ValueError("Perturbation deltas must contain exactly one zero baseline.")
    baseline_acceptance = float(baseline.iloc[0]["mean_acceptance"])
    baseline_objective = float(baseline.iloc[0]["mean_objective"])
    summary["change_mean_acceptance"] = summary["mean_acceptance"] - baseline_acceptance
    summary["change_mean_objective"] = summary["mean_objective"] - baseline_objective
    return row_table, summary


def _histogram_peak(actions: np.ndarray, u_bounds: tuple[float, float]) -> tuple[float, float]:
    counts, edges = np.histogram(np.asarray(actions, dtype=float), bins=32, range=u_bounds)
    index = int(np.argmax(counts))
    return float(0.5 * (edges[index] + edges[index + 1])), float(counts[index] / len(actions))


def _arm_summary(
    *,
    model: str,
    mode: str,
    actions: np.ndarray,
    perturbation_summary: pd.DataFrame,
    u_bounds: tuple[float, float],
    headline_delta: float,
    optimizer_success: bool | None,
    optimizer_status: int | None,
    optimizer_message: str | None,
    optimizer_steps: int,
) -> dict[str, Any]:
    baseline = perturbation_summary.loc[np.isclose(perturbation_summary["delta_u"], 0.0)].iloc[0]
    plus = perturbation_summary.loc[
        np.isclose(perturbation_summary["delta_u"], headline_delta)
    ]
    minus = perturbation_summary.loc[
        np.isclose(perturbation_summary["delta_u"], -headline_delta)
    ]
    if len(plus) != 1 or len(minus) != 1:
        raise ValueError("Perturbation deltas must contain both headline +/- delta values.")
    peak_center, peak_share = _histogram_peak(actions, u_bounds)
    return {
        "model": model,
        "constraint_mode": mode,
        "n_customers": int(len(actions)),
        "optimizer_success": optimizer_success,
        "optimizer_status": optimizer_status,
        "optimizer_message": optimizer_message,
        "optimizer_steps": int(optimizer_steps),
        "mean_u": float(np.mean(actions)),
        "u_q05": float(np.quantile(actions, 0.05)),
        "u_q50": float(np.quantile(actions, 0.50)),
        "u_q95": float(np.quantile(actions, 0.95)),
        "share_at_lower_bound_1bp": float(np.mean(actions <= u_bounds[0] + 0.0001)),
        "share_at_upper_bound_1bp": float(np.mean(actions >= u_bounds[1] - 0.0001)),
        "most_populated_u_bin_center": peak_center,
        "most_populated_u_bin_share": peak_share,
        "mean_acceptance": float(baseline["mean_acceptance"]),
        "mean_objective": float(baseline["mean_objective"]),
        "acceptance_slack": float(baseline["acceptance_slack"]),
        "plus_delta_acceptance_change": float(plus.iloc[0]["change_mean_acceptance"]),
        "plus_delta_objective_change": float(plus.iloc[0]["change_mean_objective"]),
        "minus_delta_acceptance_change": float(minus.iloc[0]["change_mean_acceptance"]),
        "minus_delta_objective_change": float(minus.iloc[0]["change_mean_objective"]),
    }


def _arm_label(model: str, mode: str) -> str:
    suffix = "penalized" if mode == "acceptance_penalty" else "unconstrained"
    return f"{model.title()} — {suffix}"


def plot_combined_policy_outputs(
    arm_rows: pd.DataFrame,
    observed_u: np.ndarray,
    output_dir: Path,
    *,
    acceptance_floor: float,
    u_bounds: tuple[float, float],
) -> None:
    """Write direct policy price and predicted-acceptance histograms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bins_u = np.linspace(u_bounds[0], u_bounds[1], 33)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.hist(observed_u, bins=bins_u, density=True, color="#bdbdbd", alpha=0.35, label="Historical U")
    for (model, mode), group in arm_rows.groupby(["model", "constraint_mode"], sort=False):
        ax.hist(
            group["u"],
            bins=bins_u,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=MODEL_COLORS[model],
            linestyle=MODE_STYLES[mode],
            label=_arm_label(model, mode),
        )
    ax.set_title("Optimized Price Changes for the 200 Stored-Spline Customers")
    ax.set_xlabel("Proposed Price Change")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "policy_u_histograms.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bins_acceptance = np.linspace(0.0, 1.0, 41)
    for (model, mode), group in arm_rows.groupby(["model", "constraint_mode"], sort=False):
        ax.hist(
            group["acceptance"],
            bins=bins_acceptance,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=MODEL_COLORS[model],
            linestyle=MODE_STYLES[mode],
            label=f"{_arm_label(model, mode)} (mean={group['acceptance'].mean():.3f})",
        )
    ax.set_title("Predicted Acceptance Under the Optimized Policies")
    ax.set_xlabel("Customer Predicted Acceptance Probability")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.text(
        0.99,
        0.02,
        f"Cohort mean floor = {acceptance_floor:.3f}\n(not a per-customer threshold)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "policy_acceptance_histograms.png", dpi=200)
    plt.close(fig)


def plot_perturbation_effects(
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    output_dir: Path,
    *,
    acceptance_floor: float,
    headline_delta: float,
) -> None:
    """Write aggregate perturbation curves plus per-arm distribution overlays."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    for (model, mode), group in summary.groupby(["model", "constraint_mode"], sort=False):
        ordered = group.sort_values("delta_u")
        style = dict(
            color=MODEL_COLORS[model],
            linestyle=MODE_STYLES[mode],
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=_arm_label(model, mode),
        )
        axes[0].plot(100.0 * ordered["delta_u"], ordered["mean_acceptance"], **style)
        axes[1].plot(100.0 * ordered["delta_u"], ordered["mean_objective"], **style)
    axes[0].axhline(acceptance_floor, color="#333333", linestyle=":", linewidth=1.4, label="Mean-acceptance floor")
    axes[0].set_title("Mean Acceptance After a Uniform Price Perturbation")
    axes[0].set_ylabel("Mean Predicted Acceptance")
    axes[1].set_title("Mean Objective After a Uniform Price Perturbation")
    axes[1].set_ylabel("Mean Objective (lower is better)")
    for ax in axes:
        ax.axvline(0.0, color="#777777", linewidth=1.0, alpha=0.65)
        ax.set_xlabel("Additive Price Perturbation (percentage points)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Local Stability of the Optimized Policies, Averaged Over 200 Customers")
    fig.tight_layout()
    fig.savefig(output_dir / "perturbation_mean_effects.png", dpi=200)
    plt.close(fig)

    for (model, mode), group in rows.groupby(["model", "constraint_mode"], sort=False):
        selected = group[
            np.isclose(group["delta_u"], 0.0)
            | np.isclose(group["delta_u"], headline_delta)
            | np.isclose(group["delta_u"], -headline_delta)
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
        baseline = selected.loc[np.isclose(selected["delta_u"], 0.0)].sort_values("id")
        for delta, delta_group in selected.groupby("delta_u", sort=True):
            label = "Optimized" if np.isclose(delta, 0.0) else f"u {delta:+.4f}"
            color = "#333333" if np.isclose(delta, 0.0) else ("#7570b3" if delta < 0 else "#e7298a")
            axes[0, 0].hist(delta_group["acceptance"], bins=35, density=True, histtype="step", linewidth=1.8, color=color, label=label)
            axes[0, 1].hist(delta_group["objective_contribution"], bins=35, density=True, histtype="step", linewidth=1.8, color=color, label=label)
            if np.isclose(delta, 0.0):
                continue
            ordered = delta_group.sort_values("id")
            axes[1, 0].hist(
                ordered["acceptance"].to_numpy() - baseline["acceptance"].to_numpy(),
                bins=35,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=color,
                label=label,
            )
            axes[1, 1].hist(
                ordered["objective_contribution"].to_numpy()
                - baseline["objective_contribution"].to_numpy(),
                bins=35,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=color,
                label=label,
            )
        axes[0, 0].set_xlabel("Customer Predicted Acceptance")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_title("Acceptance Distribution")
        axes[0, 0].text(
            0.99,
            0.02,
            f"Mean floor = {acceptance_floor:.3f}\n(not per customer)",
            transform=axes[0, 0].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        axes[0, 1].set_xlabel("Customer Objective Contribution (lower is better)")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].set_title("Objective Distribution")
        axes[1, 0].axvline(0.0, color="#777777", linewidth=1.0)
        axes[1, 0].set_xlabel("Change in Customer Predicted Acceptance")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title("Per-Customer Acceptance Change")
        axes[1, 1].axvline(0.0, color="#777777", linewidth=1.0)
        axes[1, 1].set_xlabel("Change in Customer Objective Contribution")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Per-Customer Objective Change")
        for ax in axes.reshape(-1):
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"{_arm_label(model, mode)}: Effect of +/-{headline_delta:.4f} in u")
        fig.tight_layout()
        fig.savefig(output_dir / f"perturbation_distributions__{model}__{mode}.png", dpi=200)
        plt.close(fig)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--dataset", type=Path, default=dataset_csv_path())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--u-min", type=float, default=0.0)
    parser.add_argument("--u-max", type=float, default=0.16)
    parser.add_argument("--initial-u", type=float)
    parser.add_argument("--acceptance-floor", type=float)
    parser.add_argument("--penalty-weight", type=float, default=1e4)
    parser.add_argument("--penalty-temperature", type=float, default=0.05)
    parser.add_argument("--fd-eps", type=float, default=0.001)
    parser.add_argument("--t-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--headline-delta", type=float, default=0.001)
    parser.add_argument("--perturbation-deltas", type=float, nargs="+", default=list(DEFAULT_DELTAS))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    u_bounds = (float(args.u_min), float(args.u_max))
    if not u_bounds[0] < u_bounds[1]:
        raise ValueError("u-min must be less than u-max.")
    deltas = tuple(float(value) for value in args.perturbation_deltas)
    if sum(np.isclose(deltas, 0.0)) != 1:
        raise ValueError("perturbation-deltas must contain zero exactly once.")
    if not any(np.isclose(deltas, args.headline_delta)) or not any(
        np.isclose(deltas, -args.headline_delta)
    ):
        raise ValueError("perturbation-deltas must contain +/- headline-delta.")

    artifacts = load_diagnostic_artifacts(
        args.artifact_dir.resolve(), xgb_n_jobs=int(args.xgb_n_jobs)
    )
    acceptance_preprocessor = artifacts.raw_acceptance.preprocessor
    numeric_means = getattr(acceptance_preprocessor, "numeric_means_", pd.Series(dtype=float))
    cohort = load_curve_cohort(
        args.dataset.resolve(),
        artifacts.curve_ids,
        numeric_imputation_values={str(key): float(value) for key, value in numeric_means.items()},
    )
    floor = float(args.acceptance_floor) if args.acceptance_floor is not None else cohort.observed_acceptance
    initial_u = (
        float(args.initial_u)
        if args.initial_u is not None
        else float(np.clip(np.mean(cohort.observed_u), u_bounds[0] + 1e-6, u_bounds[1] - 1e-6))
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_root() / "policy-cliff-perturbation-diagnostic" / stamp
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    arm_specs = (
        ("xgboost", "unconstrained", artifacts.raw_acceptance, False),
        ("xgboost", "acceptance_penalty", artifacts.raw_acceptance, True),
        ("spline", "unconstrained", artifacts.spline_acceptance, False),
        ("spline", "acceptance_penalty", artifacts.spline_acceptance, True),
    )
    all_rows: list[pd.DataFrame] = []
    all_perturbation_rows: list[pd.DataFrame] = []
    all_perturbation_summary: list[pd.DataFrame] = []
    arm_summaries: list[dict[str, Any]] = []
    for model, mode, acceptance_model, constrained in arm_specs:
        print(f"\n=== {model} / {mode} ===", flush=True)
        config = build_diagnostic_config(
            acceptance_model=acceptance_model,
            loss_model=artifacts.loss,
            cohort=cohort,
            constrained=constrained,
            acceptance_floor=floor,
            penalty_weight=float(args.penalty_weight),
            penalty_temperature=float(args.penalty_temperature),
            u_bounds=u_bounds,
            initial_u=initial_u,
            fd_eps=float(args.fd_eps),
            t_steps=int(args.t_steps),
            seed=int(args.seed),
            verbose=not args.quiet,
        )
        arm_dir = output_dir / "runs" / f"{model}__{mode}"
        context = create_run_context(
            f"policy-cliff-{model}-{mode}",
            run_dir=arm_dir,
            run_metadata={
                "diagnostic": "policy_cliff_perturbation",
                "model": model,
                "constraint_mode": mode,
                "external_artifact_hashes": dict(artifacts.hashes),
            },
        )
        executed = execute_experiment_run(
            f"policy-cliff-{model}-{mode}",
            config,
            run_context=context,
            reporter_stack_factory=_reporter_stack,
        )
        result = executed.result
        theta = result.results["first_order"].theta
        trace = result.traces["first_order"]
        objective = result.config.objective
        optimized_u = np.asarray(objective.policy_value(theta, cohort.frame), dtype=float)
        optimized_rows = evaluate_actions(objective, cohort.frame, optimized_u)
        optimized_rows.insert(0, "constraint_mode", mode)
        optimized_rows.insert(0, "model", model)
        all_rows.append(optimized_rows)

        perturbation_rows, perturbation_summary = perturb_policy_actions(
            objective,
            cohort.frame,
            optimized_u,
            deltas,
            u_bounds=u_bounds,
            acceptance_floor=floor,
        )
        perturbation_rows.insert(0, "constraint_mode", mode)
        perturbation_rows.insert(0, "model", model)
        perturbation_summary.insert(0, "constraint_mode", mode)
        perturbation_summary.insert(0, "model", model)
        all_perturbation_rows.append(perturbation_rows)
        all_perturbation_summary.append(perturbation_summary)
        arm_summaries.append(
            _arm_summary(
                model=model,
                mode=mode,
                actions=optimized_u,
                perturbation_summary=perturbation_summary,
                u_bounds=u_bounds,
                headline_delta=float(args.headline_delta),
                optimizer_success=trace.optimizer_success,
                optimizer_status=trace.optimizer_status,
                optimizer_message=trace.optimizer_message,
                optimizer_steps=len(trace.steps),
            )
        )

    policy_rows = pd.concat(all_rows, ignore_index=True)
    perturbation_rows = pd.concat(all_perturbation_rows, ignore_index=True)
    perturbation_summary = pd.concat(all_perturbation_summary, ignore_index=True)
    arm_summary = pd.DataFrame(arm_summaries)
    policy_rows.to_csv(output_dir / "policy_outputs.csv", index=False)
    perturbation_rows.to_csv(output_dir / "perturbation_rows.csv", index=False)
    perturbation_summary.to_csv(output_dir / "perturbation_summary.csv", index=False)
    arm_summary.to_csv(output_dir / "arm_summary.csv", index=False)

    plot_dir = output_dir / "plots"
    plot_combined_policy_outputs(
        policy_rows,
        cohort.observed_u,
        plot_dir,
        acceptance_floor=floor,
        u_bounds=u_bounds,
    )
    plot_perturbation_effects(
        perturbation_summary,
        perturbation_rows,
        plot_dir,
        acceptance_floor=floor,
        headline_delta=float(args.headline_delta),
    )
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset.resolve()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "artifact_sha256": dict(artifacts.hashes),
        "n_stored_curves": len(artifacts.curve_ids),
        "n_matched_customers": len(cohort.frame),
        "duplicate_curve_ids_in_dataset": list(cohort.duplicate_match_ids),
        "mean_imputed_cells": list(cohort.imputed_cells),
        "cohort_rule": "first canonical CSV row for each stored curve ID, in stored-wrapper order",
        "historical_mean_u": float(np.mean(cohort.observed_u)),
        "historical_observed_acceptance": cohort.observed_acceptance,
        "acceptance_floor": floor,
        "u_bounds": list(u_bounds),
        "initial_u": initial_u,
        "optimizer": {
            "step_rule": "l-bfgs-b",
            "gradient": "action derivative chained through the existing policy",
            "raw_xgboost_acceptance_derivative": "central finite difference in one-dimensional u",
            "raw_xgboost_fd_eps": float(args.fd_eps),
            "spline_acceptance_derivative": "stored PPoly analytical derivative",
            "t_steps": int(args.t_steps),
            "seed": int(args.seed),
            "xgb_n_jobs": int(args.xgb_n_jobs),
        },
        "constraint": {
            "scope": "mean predicted acceptance over the 200-customer cohort",
            "mode": "existing smooth squared-softplus penalty",
            "weight": float(args.penalty_weight),
            "temperature": float(args.penalty_temperature),
        },
        "perturbation_deltas": list(deltas),
        "headline_delta": float(args.headline_delta),
        "shared_loss_model_for_all_arms": True,
    }
    _write_json(output_dir / "provenance.json", provenance)
    print("\nFinal arm summary:")
    print(arm_summary.to_string(index=False))
    print(f"\nWrote diagnostic to {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
