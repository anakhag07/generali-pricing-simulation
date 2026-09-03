from __future__ import annotations

import numpy as np

from scripts.plot_customer_coverage_envelope_slides import _optimize_display_curve


def test_display_curve_solution_comes_from_continuous_optimizer() -> None:
    u = np.linspace(0.0, 0.16, 17)
    values = -(u - 0.0734) ** 2

    solution = _optimize_display_curve(u, values)

    assert solution["success"] is True
    assert abs(float(solution["u"]) - 0.0734) < 0.002
    assert not np.any(np.isclose(float(solution["u"]), u, atol=1e-7))


def test_uncertainty_penalty_moves_optimizer_toward_supported_region() -> None:
    u = np.linspace(0.0, 0.16, 161)
    profit = 160.0 - 1_200.0 * (u - 0.13) ** 2
    width = 10.0 * (1.0 - np.exp(-0.5 * ((u - 0.09) / 0.025) ** 2))

    profit_solution = _optimize_display_curve(u, profit)
    adjusted_solution = _optimize_display_curve(u, profit - width)

    assert float(adjusted_solution["u"]) < float(profit_solution["u"]) - 0.01
    assert abs(float(adjusted_solution["u"]) - 0.09) < abs(
        float(profit_solution["u"]) - 0.09
    )
