"""Regenerate the acceptance-floor frontier with current linear artifacts.

The configuration intentionally follows the historical first-order sweep where
possible: 5,000 deterministic rows, the artifact policy features, the default
``(-0.5, 0.5)`` softmax action range, and SciPy ``trust-constr``.  The plotted
profit is the negative of the minimized repository objective.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data.dataset_metadata import DATASET_PATH, MODEL_ARTIFACTS
from experiments.configs import get_config
from experiments.execution import execute_experiment_run

matplotlib.use("Agg")


ACCEPTANCE_FLOORS = (
    0.500,
    0.550,
    0.600,
    0.650,
    0.700,
    0.750,
    0.800,
    0.840,
    0.870,
    0.890,
    0.910,
    0.925,
    0.940,
    0.950,
    0.960,
    0.970,
    0.978,
    0.985,
    0.990,
    0.993,
    0.995,
)

DEFAULT_OUTPUT_DIR = Path("outputs/current-glm-acceptance-floor-frontier")
CSV_FIELDS = (
    "acceptance_floor",
    "mean_acceptance",
    "objective_value",
    "expected_profit_per_customer",
    "mean_u",
    "constraint_violation",
    "optimizer_success",
    "optimizer_status",
    "optimizer_message",
    "runtime_seconds",
    "run_dir",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_profit_frontier(path: Path, rows: Sequence[dict[str, object]]) -> None:
    values = np.asarray(
        [
            (
                row["acceptance_floor"],
                row["mean_acceptance"],
                row["expected_profit_per_customer"],
            )
            for row in rows
        ],
        dtype=float,
    )
    floors, mean_acceptance, expected_profit = values.T
    norm = matplotlib.colors.Normalize(vmin=float(floors.min()), vmax=float(floors.max()))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(mean_acceptance, expected_profit, alpha=0.25, linewidth=1.0)
    scatter = ax.scatter(
        mean_acceptance,
        expected_profit,
        c=floors,
        cmap="viridis",
        norm=norm,
        marker="X",
        s=28.0,
        linewidths=0.6,
        alpha=0.9,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Acceptance Floor", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    ax.set_xlabel("Mean Acceptance", fontsize=12)
    ax.set_ylabel("Expected Profit per Customer", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _artifact_provenance() -> dict[str, object]:
    provenance: dict[str, object] = {}
    for role in ("acceptance", "loss"):
        artifact_path = Path(MODEL_ARTIFACTS["linear"][role]["path"])
        provenance[role] = {
            "path": str(artifact_path),
            "sha256": _sha256(artifact_path),
            "description": MODEL_ARTIFACTS["linear"][role]["description"],
            "probability_target": MODEL_ARTIFACTS["linear"][role]["probability_target"],
        }
    return provenance


def run_frontier(output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir / "runs"
    base_config = get_config(
        "real_data_glm_base",
        overrides={
            "policy_kind": "softmax",
            "feature_order": "linear",
            "policy_preprocessing": "artifact",
            "constraint_mode": "trust_constr",
            "acceptance_floor": ACCEPTANCE_FLOORS[0],
            "n_samples": 5000,
            "t_steps": 1000,
            "enabled_estimators": ("first_order",),
            "compute_backend": "numpy",
            "seed": 42,
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        },
    )

    rows: list[dict[str, object]] = []
    for index, floor in enumerate(ACCEPTANCE_FLOORS, start=1):
        print(f"[{index:02d}/{len(ACCEPTANCE_FLOORS)}] acceptance_floor={floor:.3f}", flush=True)
        config = replace(base_config, acceptance_floor=float(floor))
        executed = execute_experiment_run(
            f"current_glm_acceptance_floor_{floor:.3f}",
            config,
            runs_root=runs_root,
            run_metadata={
                "purpose": "current_glm_acceptance_floor_frontier",
                "acceptance_floor": float(floor),
            },
        )
        result = executed.result.results["first_order"]
        trace = executed.result.traces["first_order"]
        objective_value = float(result.value)
        rows.append(
            {
                "acceptance_floor": float(floor),
                "mean_acceptance": float(result.mean_acceptance),
                "objective_value": objective_value,
                "expected_profit_per_customer": -objective_value,
                "mean_u": float(result.u),
                "constraint_violation": float(result.constraint_violation or 0.0),
                "optimizer_success": trace.optimizer_success,
                "optimizer_status": trace.optimizer_status,
                "optimizer_message": trace.optimizer_message,
                "runtime_seconds": float(result.time),
                "run_dir": str(executed.run_context.run_dir),
            }
        )
        _write_csv(output_dir / "current_glm_acceptance_floor_frontier.csv", rows)

    pdf_path = output_dir / "current_glm_expected_profit_acceptance_frontier.pdf"
    _plot_profit_frontier(pdf_path, rows)
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective_formula": "objective = mean(acceptance * (loss - (1 + u) * premium))",
        "plotted_quantity": "expected_profit_per_customer = -objective_value",
        "preset": "real_data_glm_base (runtime model family: linear)",
        "configuration": {
            "acceptance_floors": list(ACCEPTANCE_FLOORS),
            "policy_kind": "softmax",
            "softmax_action_bounds": [-0.5, 0.5],
            "feature_order": "linear",
            "policy_preprocessing": "artifact",
            "constraint_mode": "trust_constr",
            "n_samples": 5000,
            "t_steps": 1000,
            "enabled_estimators": ["first_order"],
            "compute_backend": "numpy",
            "seed": 42,
        },
        "dataset": {
            "path": str(DATASET_PATH),
            "sha256": _sha256(DATASET_PATH),
        },
        "artifacts": _artifact_provenance(),
        "outputs": {
            "csv": str(output_dir / "current_glm_acceptance_floor_frontier.csv"),
            "pdf": str(pdf_path),
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = run_frontier(args.output_dir)
    print(f"Wrote {len(rows)} frontier rows and the vector PDF to {args.output_dir}.")


if __name__ == "__main__":
    main()
