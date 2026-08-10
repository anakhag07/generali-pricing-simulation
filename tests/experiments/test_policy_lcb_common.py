from __future__ import annotations

import pytest

import experiments.finite_policy_lcb as legacy_finite
from experiments.policy_lcb import common
from experiments.policy_lcb import finite
from experiments.policy_lcb.continuous import continuous_lcb_quantile


def test_legacy_finite_module_is_the_finite_adapter() -> None:
    assert legacy_finite is finite
    assert legacy_finite.FinitePolicyLCBSpec is finite.FinitePolicyLCBSpec


def test_finite_quantile_and_coverage_delegate_to_shared_math() -> None:
    for delta in (0.2, 0.05):
        assert finite.lcb_quantile(delta, 11) == pytest.approx(
            common.gaussian_lcb_quantile(delta, 11)
        )
        assert finite.analytic_joint_coverage(delta, 11) == pytest.approx(
            common.independent_joint_coverage(delta, 11)
        )


def test_shared_gaussian_coverage_is_exact_nominal_coverage() -> None:
    assert common.shared_gaussian_coverage(0.05) == pytest.approx(0.95)


def test_shared_continuum_uses_one_gaussian_coordinate_not_finite_multiplicity() -> None:
    delta = 0.05

    assert continuous_lcb_quantile(delta) == pytest.approx(
        common.gaussian_lcb_quantile(delta, multiplicity=1)
    )
    assert continuous_lcb_quantile(delta) < finite.lcb_quantile(delta, policy_count=11)
