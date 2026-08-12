from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, PPoly

from data.dataset_metadata import (
    ACCEPTANCE_STATE_COLS,
    LOSS_FEATURE_COLS,
    OBSERVED_CHURN_COL,
    OBSERVED_U_COL,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_policy_cliff_perturbation_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("policy_cliff_diagnostic", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stored_curve_preserves_values_derivatives_and_boundary_rules() -> None:
    x = np.asarray([0.0, 0.08, 0.16])
    churn = np.asarray([0.05, 0.10, 0.20])
    interp = PchipInterpolator(x, churn)
    curve = MODULE.StoredMonotoneCurve(
        polynomial=PPoly(interp.c, interp.x),
        x_min=0.0,
        p_min=0.05,
        x_max=0.16,
        p_max=0.20,
        slope_p=1.25,
    )

    inside = np.asarray([0.02, 0.08, 0.14])
    np.testing.assert_allclose(curve.value(inside), interp(inside))
    np.testing.assert_allclose(curve.derivative(inside), interp.derivative()(inside))
    np.testing.assert_allclose(curve.value(np.asarray([-0.1])), [0.05])
    np.testing.assert_allclose(curve.derivative(np.asarray([-0.1])), [0.0])
    np.testing.assert_allclose(curve.value(np.asarray([0.20])), [0.25])
    np.testing.assert_allclose(curve.derivative(np.asarray([0.20])), [1.25])


def test_curve_cohort_keeps_stored_order_and_records_mean_imputation(tmp_path: Path) -> None:
    columns = list(
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
    rows = []
    for policy_id, observed_u, churn in (("10", 0.04, 0), ("20", 0.08, 1)):
        row = {column: 1.0 for column in columns}
        row.update({"id": policy_id, OBSERVED_U_COL: observed_u, OBSERVED_CHURN_COL: churn})
        rows.append(row)
    rows[1]["X_age"] = np.nan
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, sep=";", index=False)

    cohort = MODULE.load_curve_cohort(
        csv_path,
        ("20", "10"),
        numeric_imputation_values={"X_age": 47.0},
    )

    assert cohort.frame["id"].map(MODULE._normalize_id).tolist() == ["20", "10"]
    assert cohort.row_indices.tolist() == [1, 0]
    assert cohort.observed_u.tolist() == [0.08, 0.04]
    assert cohort.observed_acceptance == 0.5
    assert cohort.frame.loc[0, "X_age"] == 47.0
    assert cohort.imputed_cells == ("20:X_age",)


class _FakeObjective:
    def _acceptance_proba(self, frame: pd.DataFrame, u: np.ndarray) -> np.ndarray:
        del frame
        return 1.0 - np.asarray(u, dtype=float)

    def _loss_prediction(self, frame: pd.DataFrame) -> np.ndarray:
        return np.ones(len(frame), dtype=float)

    def _premium_values(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), 2.0, dtype=float)


def test_perturbation_replay_reports_changes_slack_and_clipping() -> None:
    frame = pd.DataFrame({"id": [1, 2]})
    rows, summary = MODULE.perturb_policy_actions(
        _FakeObjective(),
        frame,
        np.asarray([0.0, 0.1595]),
        (-0.001, 0.0, 0.001),
        u_bounds=(0.0, 0.16),
        acceptance_floor=0.95,
    )

    assert len(rows) == 6
    baseline = summary.loc[np.isclose(summary["delta_u"], 0.0)].iloc[0]
    plus = summary.loc[np.isclose(summary["delta_u"], 0.001)].iloc[0]
    minus = summary.loc[np.isclose(summary["delta_u"], -0.001)].iloc[0]
    assert baseline["mean_acceptance"] == np.mean([1.0, 1.0 - 0.1595])
    assert plus["clipped_upper_count"] == 1
    assert minus["clipped_lower_count"] == 1
    assert plus["change_mean_acceptance"] < 0.0
    assert minus["change_mean_acceptance"] > 0.0
    assert bool(baseline["violates_acceptance_floor"])


def test_dense_cliff_step_summary_preserves_adjacent_jump_location() -> None:
    rows = pd.DataFrame(
        {
            "model": ["xgboost"] * 6,
            "constraint_mode": ["trust_constr"] * 6,
            "id": ["1", "2", "1", "2", "1", "2"],
            "delta_u": [-0.001, -0.001, 0.0, 0.0, 0.001, 0.001],
            "acceptance": [0.95, 0.90, 0.95, 0.90, 0.75, 0.90],
            "objective_contribution": [-2.0, -1.0, -2.1, -1.1, -1.0, -1.2],
        }
    )

    summary = MODULE.summarize_dense_cliff_steps(rows)

    assert len(summary) == 2
    right_step = summary.loc[np.isclose(summary["delta_right"], 0.001)].iloc[0]
    assert right_step["fraction_acceptance_changed"] == 0.5
    assert np.isclose(right_step["max_abs_acceptance_step"], 0.20)
    assert np.isclose(right_step["max_abs_objective_step"], 1.10)


def test_cli_defaults_to_wider_hard_constraint_diagnostic() -> None:
    args = MODULE.parse_args([])

    assert (args.u_min, args.u_max) == (-0.1, 0.2)
    assert args.initial_u == -0.05
    assert args.t_steps == 500
    assert args.perturbation_count == 161
    assert args.models == ("xgboost", "spline")
    assert not hasattr(args, "penalty_weight")
