"""Check observed-profit semantics and paired historical-action evaluation."""

import numpy as np
import pandas as pd
import pytest

from scratch import plot_generali_profit_residuals as report
from reporting.real_data import ResolvedRealDataModel, model_spec


def _history():
    return pd.DataFrame({"U": [.1, -.2, 0.], "X_policy_premium": [100., 200., 80.],
                         "X_upcoming_premium": [110., 160., 80.],
                         "Y_G_Loss": [20., 50., 100.], "is_churn": [0, 1, 0]})


def test_observed_profit_retains_churn_zero_and_negative_margin():
    profit, error = report._observed_profit(_history())
    np.testing.assert_allclose(profit, [90., 0., -20.])
    assert error < 1e-10


def test_observed_profit_rejects_misaligned_revenue_and_invalid_churn():
    frame = _history()
    frame.loc[0, "X_upcoming_premium"] += 1
    with pytest.raises(ValueError, match="identity"):
        report._observed_profit(frame)
    frame = _history()
    frame.loc[0, "is_churn"] = 2
    with pytest.raises(ValueError, match="binary"):
        report._observed_profit(frame)


class _Acceptance:
    def __init__(self):
        self.model = self

    def model_frame(self, frame):
        return frame

    def predict_proba(self, frame):
        p = .8 - .5*frame["U"].to_numpy()
        return np.column_stack([1-p, p])


class _Loss(_Acceptance):
    def predict(self, frame):
        return frame["Y_G_Loss"].to_numpy()


def test_prediction_uses_each_rows_own_action_and_keeps_residual_sign():
    frame = _history()
    model = ResolvedRealDataModel(model_spec("glm"), _Acceptance(), _Loss())
    acceptance, loss, profit = report._predicted_profit(frame, frame.U.to_numpy(), model)
    np.testing.assert_allclose(acceptance, [.75, .9, .8])
    np.testing.assert_allclose(profit, [67.5, 99., -16.])
    np.testing.assert_allclose(report._observed_profit(frame)[0]-profit, [22.5, -99., -4.])
    cached = report._predicted_profit(frame, frame.U.to_numpy(), model, np.array([.7, .6, .5]))
    np.testing.assert_allclose(cached[2], [63., 66., -10.])
