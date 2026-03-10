import numpy as np

from objective.fixed_objective import FixedRegressionObjective
from experiments.reporters import _u_star_for_plot


class DummyObjective:
    pass


def test_u_star_for_plot_fixed_regression_returns_none() -> None:
    objective = FixedRegressionObjective.from_parameters(
        beta_1=np.array([1.0]),
        beta_2=-0.1,
        beta_3=np.array([0.2]),
        beta_4=0.3,
    )
    assert _u_star_for_plot(objective, 1.1) is None


def test_u_star_for_plot_non_fixed_regression_returns_value() -> None:
    assert _u_star_for_plot(DummyObjective(), 1.1) == 1.1
