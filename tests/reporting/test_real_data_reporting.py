from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reporting.real_data import (
    ResolvedRealDataModel,
    model_spec,
)


class _AcceptanceArtifact:
    probability_target = "acceptance"

    def __init__(self) -> None:
        self.model = self

    def model_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        acceptance = 0.8 - 0.5 * frame["U"].to_numpy(dtype=float)
        return np.column_stack([1.0 - acceptance, acceptance])


class _LossArtifact:
    def __init__(self) -> None:
        self.model = self

    def model_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["loss"].to_numpy(dtype=float)


def test_model_aliases_are_one_line_swaps() -> None:
    assert model_spec("glm").acceptance_model == "linear"
    assert model_spec("xgb").acceptance_model == "xgb"
    assert model_spec("exact_spline_xgb").acceptance_strategy == "exact_spline"

    explicit = model_spec(
        {
            "name": "mixed",
            "acceptance_model": "linear",
            "loss_model": "xgb",
        }
    )
    assert explicit.name == "mixed"
    assert explicit.loss_model == "xgb"


def test_resolved_model_profit_matrix_uses_shared_pricing_formula() -> None:
    model = ResolvedRealDataModel(
        model_spec("glm"),
        _AcceptanceArtifact(),  # type: ignore[arg-type]
        _LossArtifact(),  # type: ignore[arg-type]
    )
    frame = pd.DataFrame(
        {
            "X_policy_premium": [100.0, 200.0],
            "loss": [20.0, 50.0],
        }
    )

    profit = model.profit_matrix(frame, [0.0, 0.1])

    expected_acceptance = np.asarray([[0.8, 0.75], [0.8, 0.75]])
    expected_margin = np.asarray([[80.0, 90.0], [150.0, 170.0]])
    np.testing.assert_allclose(profit, expected_acceptance * expected_margin)


def test_model_spec_rejects_invalid_exact_spline_source() -> None:
    with pytest.raises(ValueError, match="requires acceptance_model='xgb'"):
        model_spec(
            {
                "acceptance_model": "linear",
                "loss_model": "linear",
                "acceptance_strategy": "exact_spline",
            }
        )
