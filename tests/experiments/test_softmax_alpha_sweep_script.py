from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import run_glm_softmax_alpha_sweep as script


class _FakeObjective:
    def policy_value(self, theta: np.ndarray, x_batch: pd.DataFrame) -> np.ndarray:
        del theta
        return x_batch["u"].to_numpy(dtype=float)

    def _clip_u(self, u_values: np.ndarray) -> np.ndarray:
        return np.asarray(u_values, dtype=float)

    def _acceptance_proba(self, x_batch: pd.DataFrame, u_values: np.ndarray) -> np.ndarray:
        del u_values
        return x_batch["acceptance"].to_numpy(dtype=float)

    def _loss_prediction(self, x_batch: pd.DataFrame) -> np.ndarray:
        return x_batch["loss"].to_numpy(dtype=float)

    def _premium_values(self, x_batch: pd.DataFrame) -> np.ndarray:
        return x_batch["premium"].to_numpy(dtype=float)

    def _value_batch(self, x_batch: pd.DataFrame, u_values: np.ndarray) -> np.ndarray:
        del u_values
        return x_batch["objective"].to_numpy(dtype=float)


class _FakeArtifact:
    estimator = "first_order"
    theta = np.asarray([0.0], dtype=float)

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def load_x(self, split: str = "train") -> pd.DataFrame:
        if split == "test":
            raise ValueError("Saved policy artifact has no test split rows.")
        return self._frame.copy()

    def build_objective(self) -> _FakeObjective:
        return _FakeObjective()


def test_alpha_sweep_uses_confirmed_trust_constrained_grid() -> None:
    assert script.ALPHA_VALUES == (0.5, 0.4, 0.3, 0.2, 0.15, 0.125, 0.1, 0.075)
    assert script.RUN_OVERRIDES["constraint_mode"] == "trust_constr"
    assert script.RUN_OVERRIDES["enabled_estimators"] == ("first_order",)
    assert script.RUN_OVERRIDES["initial_u"] == 0.0
    assert script._alpha_label(0.075) == "alpha_0p075"


def test_u_bin_edges_span_symmetric_alpha_bounds() -> None:
    np.testing.assert_allclose(script._u_bin_edges(0.1, 4), [-0.1, -0.05, 0.0, 0.05, 0.1])


def test_artifact_bin_rows_summarize_acceptance_and_u_bins() -> None:
    rows = script._collect_artifact_bin_rows(
        alpha=0.1,
        run_name="alpha_0p100",
        artifact=_FakeArtifact(_frame()),
        policy_artifact_path="policy.json",
        splits=("train",),
        u_bin_count=2,
        acceptance_threshold=0.5,
    )

    acceptance_rows = {
        row["bin_name"]: row for row in rows if row["bin_type"] == "acceptance_threshold"
    }
    assert acceptance_rows["acceptance_le_0p500"]["n_rows"] == 2
    assert acceptance_rows["acceptance_le_0p500"]["mean_expected_profit"] == -5.0
    assert acceptance_rows["acceptance_gt_0p500"]["n_rows"] == 2
    assert acceptance_rows["acceptance_gt_0p500"]["mean_expected_profit"] == 9.0

    u_rows = {row["bin_name"]: row for row in rows if row["bin_type"] == "u"}
    assert u_rows["u_bin_00"]["n_rows"] == 2
    assert u_rows["u_bin_00"]["mean_expected_profit"] == -5.0
    assert u_rows["u_bin_01"]["n_rows"] == 2
    assert u_rows["u_bin_01"]["mean_expected_profit"] == 9.0


def test_artifact_bin_rows_skip_missing_test_split() -> None:
    rows = script._collect_artifact_bin_rows(
        alpha=0.1,
        run_name="alpha_0p100",
        artifact=_FakeArtifact(_frame()),
        policy_artifact_path="policy.json",
        splits=("test",),
        u_bin_count=2,
    )

    assert rows == []


def test_write_plots_creates_alpha_and_per_alpha_u_outputs(tmp_path) -> None:
    final_rows = [
        _final_row(alpha=0.1, value=-10.0),
        _final_row(alpha=0.2, value=-12.0),
    ]
    bin_rows = []
    for alpha in (0.1, 0.2):
        bin_rows.extend(
            script._collect_artifact_bin_rows(
                alpha=alpha,
                run_name=script._alpha_label(alpha),
                artifact=_FakeArtifact(_frame()),
                policy_artifact_path=f"policy_{alpha}.json",
                splits=("train",),
                u_bin_count=2,
            )
        )

    script._write_rows(final_rows, tmp_path / "softmax_alpha_sweep.csv", script._FINAL_FIELDNAMES)
    script._write_rows(bin_rows, tmp_path / "softmax_alpha_bin_summary.csv", script._BIN_FIELDNAMES)
    script._write_plots(final_rows, bin_rows, tmp_path)

    assert (tmp_path / "softmax_alpha_sweep.csv").exists()
    assert (tmp_path / "softmax_alpha_bin_summary.csv").exists()
    assert (tmp_path / "alpha_vs_objective.png").exists()
    assert (tmp_path / "alpha_vs_expected_profit.png").exists()
    assert (tmp_path / "alpha_profit_by_acceptance_bin.png").exists()
    assert (tmp_path / "profit_by_u_bins_alpha_0p100.png").exists()
    assert (tmp_path / "profit_by_u_bins_alpha_0p200.png").exists()


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "u": [-0.08, -0.03, 0.02, 0.09],
            "acceptance": [0.4, 0.5, 0.6, 0.9],
            "loss": [100.0, 100.0, 100.0, 100.0],
            "premium": [10.0, 10.0, 10.0, 10.0],
            "objective": [4.0, 6.0, -8.0, -10.0],
        }
    )


def _final_row(alpha: float, value: float) -> dict[str, object]:
    row = {field: "" for field in script._FINAL_FIELDNAMES}
    row.update(
        {
            "run_name": script._alpha_label(alpha),
            "alpha": alpha,
            "action_low": -alpha,
            "action_high": alpha,
            "estimator": "first_order",
            "u": 0.0,
            "mean_acceptance": 0.75,
            "value": value,
            "expected_profit": -value,
            "runtime_sec": 1.0,
            "train_n_samples": 4,
            "train_objective_value": value,
            "train_expected_profit": -value,
            "policy_artifact": f"policy_{alpha}.json",
            "run_dir": "run",
        }
    )
    return row
