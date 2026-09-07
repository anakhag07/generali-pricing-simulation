from __future__ import annotations

import numpy as np
import pytest

from scripts.run_full_population_support_softmax_policy import (
    _SupportBandActionBias,
    _load_constrained_reference_policy,
    _load_replayed_profit_policy,
)


def test_support_band_action_bias_interpolates_displayed_half_width() -> None:
    bias = _SupportBandActionBias(
        u_grid=np.asarray([0.0, 0.08, 0.16]),
        half_width=np.asarray([5.0, 1.0, 10.0]),
    )

    values = bias.values(None, np.asarray([0.0, 0.08, 0.16]))

    assert np.allclose(values, [5.0, 1.0, 10.0])
    assert np.isfinite(bias.grad_u(None, np.asarray([0.04, 0.12]))).all()


def test_support_band_action_bias_scales_values_and_gradient() -> None:
    u_grid = np.asarray([0.0, 0.08, 0.16])
    half_width = np.asarray([5.0, 1.0, 10.0])
    query = np.asarray([0.04, 0.12])
    unit_bias = _SupportBandActionBias(u_grid, half_width)
    scaled_bias = _SupportBandActionBias(u_grid, half_width, lambda_bias=2.5)

    np.testing.assert_allclose(
        scaled_bias.values(None, query),
        2.5 * unit_bias.values(None, query),
    )
    np.testing.assert_allclose(
        scaled_bias.grad_u(None, query),
        2.5 * unit_bias.grad_u(None, query),
    )
    epsilon = 1e-6
    finite_difference = (
        scaled_bias.values(None, query + epsilon)
        - scaled_bias.values(None, query - epsilon)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        scaled_bias.grad_u(None, query), finite_difference, rtol=1e-6, atol=1e-6
    )


def test_load_replayed_profit_policy_requires_matching_theta_and_summary(
    tmp_path,
) -> None:
    artifact_path = tmp_path / "softmax_policy_actions.npz"
    np.savez_compressed(artifact_path, profit_theta=np.asarray([1.0, 2.0]))
    (tmp_path / "summary.json").write_text(
        '{"profit_policy": {"optimizer_success": true}}\n',
        encoding="utf-8",
    )

    theta, summary = _load_replayed_profit_policy(artifact_path, theta_dim=2)

    assert np.array_equal(theta, [1.0, 2.0])
    assert summary == {"optimizer_success": True}

    with pytest.raises(ValueError, match="expected"):
        _load_replayed_profit_policy(artifact_path, theta_dim=3)


def test_load_constrained_reference_policy_preserves_saved_actions(tmp_path) -> None:
    artifact_path = tmp_path / "reference.npz"
    rows = np.asarray([1, 4, 8])
    actions = np.asarray([0.02, 0.08, 0.14])
    np.savez_compressed(
        artifact_path,
        row_indices=rows,
        actions=actions,
        theta=np.asarray([0.5, -0.2]),
        u_bounds=np.asarray([0.0, 0.16]),
        acceptance_floor=np.asarray(0.88),
        optimizer_success=np.asarray(False),
        optimizer_message=np.asarray("maximum evaluations"),
    )

    reference = _load_constrained_reference_policy(
        artifact_path,
        expected_rows=rows,
        theta_dim=2,
    )

    assert np.array_equal(reference["actions"], actions)
    assert reference["acceptance_floor"] == pytest.approx(0.88)
    assert reference["optimizer_success"] is False


def test_load_constrained_reference_policy_selects_requested_rows(tmp_path) -> None:
    artifact_path = tmp_path / "reference.npz"
    rows = np.asarray([1, 4, 8, 12])
    actions = np.asarray([0.02, 0.08, 0.14, 0.15])
    np.savez_compressed(
        artifact_path,
        row_indices=rows,
        actions=actions,
        theta=np.asarray([0.5, -0.2]),
        u_bounds=np.asarray([0.0, 0.16]),
        acceptance_floor=np.asarray(0.88),
        optimizer_success=np.asarray(False),
        optimizer_message=np.asarray("maximum evaluations"),
    )

    reference = _load_constrained_reference_policy(
        artifact_path,
        expected_rows=np.asarray([4, 12]),
        theta_dim=2,
    )

    assert np.array_equal(reference["row_indices"], [4, 12])
    assert np.array_equal(reference["actions"], [0.08, 0.15])
