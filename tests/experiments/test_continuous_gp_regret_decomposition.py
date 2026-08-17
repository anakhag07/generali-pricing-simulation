from __future__ import annotations

import numpy as np
import pytest

from experiments.policy_lcb.continuous_gp import ContinuousGPLandscape
from experiments.policy_lcb.continuous_gp_core import (
    DecomposedGPLandscape,
    FiniteFourierGPSpec,
    FourierGPDraw,
    GlobalReferenceSpec,
    SmoothClippedUncertaintySpec,
    UniformConfidenceSpec,
    analytic_uniform_certificate,
    certified_coverage_probability,
    certified_shape_ratio,
    certified_weighted_gp_supremum,
)


def _draw() -> FourierGPDraw:
    gp = FiniteFourierGPSpec(rank=4, lengthscale=0.2)
    return FourierGPDraw(
        gp,
        (0.2, -0.5, 1.1, 0.7),
        (-0.4, 0.3, 0.8, -0.2),
    )


def _uncertainty() -> SmoothClippedUncertaintySpec:
    return SmoothClippedUncertaintySpec((0.0, 0.25, 0.5, 0.75, 1.0), 0.1, 1.0, 0.5)


def test_decomposed_landscape_derivatives_and_matched_legacy_parity() -> None:
    draw = _draw()
    uncertainty = _uncertainty()
    confidence = UniformConfidenceSpec(delta=0.05, net_count=33)
    quantile = analytic_uniform_certificate(draw.spec, confidence).quantile
    landscape = DecomposedGPLandscape(
        draw,
        uncertainty,
        surrogate_center=0.25,
        surrogate_scale=0.5,
        envelope_center=0.25,
        envelope_scale=0.5,
        quantile=quantile,
    )
    legacy = ContinuousGPLandscape(
        draw,
        uncertainty,
        center=0.25,
        noise_scale=0.5,
        quantile=quantile,
        target="variable_lcb",
    )
    x = 0.37
    step = 1e-5
    first = (landscape.evaluate(x + step) - landscape.evaluate(x - step)) / (2.0 * step)
    second = (
        landscape.evaluate(x + step)
        - 2.0 * landscape.evaluate(x)
        + landscape.evaluate(x - step)
    ) / step**2
    assert landscape.evaluate(x, 1) == pytest.approx(first, rel=1e-7, abs=1e-7)
    assert landscape.evaluate(x, 2) == pytest.approx(second, rel=2e-5, abs=2e-5)
    grid = np.linspace(0.0, 1.0, 101)
    assert landscape.evaluate(grid) == pytest.approx(legacy.evaluate(grid))


def test_certificate_inversion_and_shape_ratio_are_locked() -> None:
    gp = FiniteFourierGPSpec(rank=32, lengthscale=0.2)
    confidence = UniformConfidenceSpec(delta=0.05, net_count=129)
    quantile = analytic_uniform_certificate(gp, confidence).quantile
    assert certified_coverage_probability(gp, confidence, quantile) == pytest.approx(
        0.95, abs=1e-10
    )
    reference = GlobalReferenceSpec(1e-6, 200000, 65)
    matched = certified_shape_ratio(_uncertainty(), 0.5, 0.5, reference)
    mismatched = certified_shape_ratio(_uncertainty(), 0.5, 0.25, reference)
    assert matched.lower_bound == matched.upper_bound == 1.0
    assert mismatched.certified
    assert 0.0 < mismatched.lower_bound <= mismatched.value <= mismatched.upper_bound < 1.0


def test_weighted_surrogate_error_supremum_brackets_dense_evaluation() -> None:
    draw = _draw()
    uncertainty = _uncertainty()
    reference = GlobalReferenceSpec(1e-6, 200000, 65)
    result = certified_weighted_gp_supremum(draw, uncertainty, 0.25, reference)
    grid = np.linspace(0.0, 1.0, 10001)
    dense = np.max(
        np.abs(
            np.asarray(draw.evaluate(grid))
            * np.asarray(
                [
                    0.1
                    + 0.9
                    * (
                        6 * min(abs(x - 0.25) / 0.5, 1.0) ** 5
                        - 15 * min(abs(x - 0.25) / 0.5, 1.0) ** 4
                        + 10 * min(abs(x - 0.25) / 0.5, 1.0) ** 3
                    )
                    for x in grid
                ]
            )
        )
    )
    assert result.certified
    assert result.lower_bound <= dense + 1e-8
    assert dense <= result.upper_bound + 1e-8
