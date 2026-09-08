"""Manifest-ready real-data model comparison and plotting.

The public seam is deliberately small: resolve a named model, then ask it for a
customer-by-action profit matrix. Artifact preprocessing and exact-spline
construction remain behind that seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.dataset_metadata import (
    ACCEPTANCE_MODEL_ARTIFACTS,
    DATASET_PATH,
    LOSS_MODEL_ARTIFACTS,
    PREMIUM_COL,
)
from data.loader import (
    AcceptanceModelType,
    LossModelType,
    ModelArtifactBundle,
    eligible_csv_row_indices,
    load_model_artifact_pair,
    load_x_frame,
)
from experiments.provenance import array_sha256, file_record, write_provenance
from reporting.profit_dispersion import (
    ProfitDispersion,
    exact_spline_acceptance_matrix,
    predict_acceptance_matrix,
    predict_loss,
    spline_anchor_weights,
    summarize_profit,
)

AcceptanceStrategy = Literal["artifact", "exact_spline"]


@dataclass(frozen=True)
class RealDataModelSpec:
    """Serializable selection of acceptance and loss model behavior."""

    name: str
    acceptance_model: AcceptanceModelType
    loss_model: LossModelType
    acceptance_strategy: AcceptanceStrategy = "artifact"


_MODEL_ALIASES: dict[str, RealDataModelSpec] = {
    "glm": RealDataModelSpec("glm", "linear", "linear"),
    "xgb": RealDataModelSpec("xgb", "xgb", "xgb"),
    "monotone_spline_xgb": RealDataModelSpec(
        "monotone_spline_xgb", "monotone_spline_xgb", "xgb"
    ),
    "exact_spline_xgb": RealDataModelSpec(
        "exact_spline_xgb", "xgb", "xgb", "exact_spline"
    ),
}


@dataclass
class ResolvedRealDataModel:
    """Loaded model pair with one stable profit-evaluation interface."""

    spec: RealDataModelSpec
    acceptance_artifact: ModelArtifactBundle
    loss_artifact: ModelArtifactBundle
    exact_spline_weights: np.ndarray | None = None

    def profit_matrix(
        self,
        frame: pd.DataFrame,
        u_values: Sequence[float],
        *,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Return predicted customer profit for every proposed action."""
        u = _action_grid(u_values)
        if self.spec.acceptance_strategy == "exact_spline":
            if self.exact_spline_weights is None:
                raise ValueError("Exact-spline evaluation requires anchor weights.")
            acceptance = exact_spline_acceptance_matrix(
                self.acceptance_artifact,
                frame,
                u,
                self.exact_spline_weights,
                n_jobs=int(n_jobs),
            )
        else:
            acceptance = predict_acceptance_matrix(
                self.acceptance_artifact,
                frame,
                u,
            )
        loss = predict_loss(self.loss_artifact, frame)
        premium = frame[PREMIUM_COL].to_numpy(dtype=float)
        profit = acceptance * (
            premium[:, None] * (1.0 + u[None, :]) - loss[:, None]
        )
        if profit.shape != (len(frame), len(u)) or not np.isfinite(profit).all():
            raise ValueError("Model evaluation produced an invalid profit matrix.")
        return profit

    def provenance(self) -> dict[str, Any]:
        """Return model selections and immutable artifact identities."""
        acceptance_path = ACCEPTANCE_MODEL_ARTIFACTS[self.spec.acceptance_model]["path"]
        loss_path = LOSS_MODEL_ARTIFACTS[self.spec.loss_model]["path"]
        payload: dict[str, Any] = {
            "name": self.spec.name,
            "acceptance_model": self.spec.acceptance_model,
            "loss_model": self.spec.loss_model,
            "acceptance_strategy": self.spec.acceptance_strategy,
            "acceptance_artifact": file_record(acceptance_path),
            "loss_artifact": file_record(loss_path),
        }
        if self.exact_spline_weights is not None:
            payload["spline_anchor_weights"] = self.exact_spline_weights.tolist()
        return payload


def model_spec(value: str | Mapping[str, Any]) -> RealDataModelSpec:
    """Resolve a model alias or explicit manifest mapping."""
    if isinstance(value, str):
        try:
            return _MODEL_ALIASES[value]
        except KeyError as error:
            raise ValueError(
                f"Unknown real-data model {value!r}; choose from {sorted(_MODEL_ALIASES)}."
            ) from error
    if not isinstance(value, Mapping):
        raise TypeError("A real-data model must be an alias or JSON object.")
    strategy = str(value.get("acceptance_strategy", "artifact"))
    if strategy not in {"artifact", "exact_spline"}:
        raise ValueError("acceptance_strategy must be 'artifact' or 'exact_spline'.")
    acceptance = str(value["acceptance_model"])
    loss = str(value["loss_model"])
    if acceptance not in ACCEPTANCE_MODEL_ARTIFACTS:
        raise ValueError(f"Unknown acceptance model {acceptance!r}.")
    if loss not in LOSS_MODEL_ARTIFACTS:
        raise ValueError(f"Unknown loss model {loss!r}.")
    if strategy == "exact_spline" and acceptance != "xgb":
        raise ValueError("exact_spline requires acceptance_model='xgb'.")
    return RealDataModelSpec(
        name=str(value.get("name") or f"{acceptance}-{loss}"),
        acceptance_model=acceptance,  # type: ignore[arg-type]
        loss_model=loss,  # type: ignore[arg-type]
        acceptance_strategy=strategy,  # type: ignore[arg-type]
    )


def resolve_real_data_model(
    value: str | Mapping[str, Any],
    *,
    eligible_rows: Sequence[int] | None = None,
) -> ResolvedRealDataModel:
    """Load one model pair; exact spline is selected with one alias or field."""
    spec = model_spec(value)
    acceptance, loss = load_model_artifact_pair(
        spec.acceptance_model,
        spec.loss_model,
    )
    weights = None
    if spec.acceptance_strategy == "exact_spline":
        rows = (
            eligible_csv_row_indices("xgb")
            if eligible_rows is None
            else np.asarray(eligible_rows, dtype=int)
        )
        weights = spline_anchor_weights(rows)
    return ResolvedRealDataModel(spec, acceptance, loss, weights)


def run_profit_dispersion_report(
    *,
    output_dir: str | Path,
    models: Sequence[str | Mapping[str, Any]],
    sample_size: int,
    seed: int,
    u_min: float,
    u_max: float,
    u_count: int,
    n_jobs: int = 1,
) -> list[Path]:
    """Evaluate named models on one deterministic cohort and write PDF/CSV outputs."""
    if int(sample_size) <= 0:
        raise ValueError("sample_size must be positive.")
    if int(n_jobs) == 0:
        raise ValueError("n_jobs cannot be zero.")
    if not models:
        raise ValueError("models must contain at least one model selection.")
    u = _action_grid(np.linspace(float(u_min), float(u_max), int(u_count)))
    eligible = eligible_csv_row_indices("xgb")
    if sample_size > len(eligible):
        raise ValueError("sample_size cannot exceed the eligible population.")
    rows = np.sort(
        np.random.default_rng(int(seed)).choice(
            eligible,
            size=int(sample_size),
            replace=False,
        )
    )
    frame = load_x_frame("xgb", row_indices=rows)
    resolved = [resolve_real_data_model(item, eligible_rows=eligible) for item in models]
    summaries: dict[str, ProfitDispersion] = {}
    for model in resolved:
        if model.spec.name in summaries:
            raise ValueError(f"Duplicate report model name {model.spec.name!r}.")
        summaries[model.spec.name] = summarize_profit(
            model.profit_matrix(frame, u, n_jobs=n_jobs)
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "profit_dispersion.csv"
    pd.DataFrame(_dispersion_rows(summaries, u, sample_size)).to_csv(
        csv_path, index=False
    )
    mean_pdf = destination / "profit_mean_std.pdf"
    median_pdf = destination / "profit_median_mad.pdf"
    _plot_dispersion(summaries, u, "mean", "std", "Mean predicted profit", mean_pdf)
    _plot_dispersion(
        summaries, u, "median", "mad", "Median predicted profit", median_pdf
    )
    provenance_path = destination / "provenance.json"
    write_provenance(
        provenance_path,
        {
            "recipe": "real_data_profit_dispersion",
            "dataset": file_record(DATASET_PATH),
            "cohort": {
                "sample_size": int(sample_size),
                "seed": int(seed),
                "selection": "sorted choice without replacement from eligible rows",
                "row_indices_sha256": array_sha256(rows.astype("<i8")),
            },
            "action_grid": {"u_min": float(u[0]), "u_max": float(u[-1]), "count": len(u)},
            "models": [model.provenance() for model in resolved],
            "outputs": [file_record(path) for path in (csv_path, mean_pdf, median_pdf)],
            "optimizer": "not used; this report evaluates no optimum",
        },
    )
    return [csv_path, mean_pdf, median_pdf, provenance_path]


def _action_grid(values: Sequence[float]) -> np.ndarray:
    u = np.asarray(values, dtype=float)
    if u.ndim != 1 or u.size < 2 or not np.isfinite(u).all():
        raise ValueError("Action grid must contain at least two finite values.")
    if not np.all(np.diff(u) > 0.0):
        raise ValueError("Action grid must be strictly increasing.")
    return u


def _dispersion_rows(
    summaries: Mapping[str, ProfitDispersion],
    u: np.ndarray,
    n_customers: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, summary in summaries.items():
        for center_name, spread_name in (("mean", "std"), ("median", "mad")):
            center = np.asarray(getattr(summary, center_name), dtype=float)
            spread = np.asarray(getattr(summary, spread_name), dtype=float)
            for index, proposed_u in enumerate(u):
                rows.append(
                    {
                        "model": model_name,
                        "u": float(proposed_u),
                        "center_statistic": center_name,
                        "center": float(center[index]),
                        "dispersion_statistic": spread_name,
                        "dispersion": float(spread[index]),
                        "lower": float(center[index] - spread[index]),
                        "upper": float(center[index] + spread[index]),
                        "n_customers": int(n_customers),
                    }
                )
    return rows


def _plot_dispersion(
    summaries: Mapping[str, ProfitDispersion],
    u: np.ndarray,
    center_name: str,
    spread_name: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(constrained_layout=True)
    for model_name, summary in summaries.items():
        center = np.asarray(getattr(summary, center_name), dtype=float)
        spread = np.asarray(getattr(summary, spread_name), dtype=float)
        line = ax.plot(100.0 * u, center, label=model_name)[0]
        ax.fill_between(
            100.0 * u,
            center - spread,
            center + spread,
            color=line.get_color(),
            alpha=0.2,
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Proposed price change (%)", fontsize=12)
    ax.set_ylabel("Predicted profit per customer", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


__all__ = [
    "RealDataModelSpec",
    "ResolvedRealDataModel",
    "model_spec",
    "resolve_real_data_model",
    "run_profit_dispersion_report",
]
