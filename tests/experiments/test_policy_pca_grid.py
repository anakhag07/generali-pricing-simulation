"""Tests for policy PCA grid utilities."""

from pathlib import Path

import numpy as np

from data.loader import ACCEPTANCE_STATE_COLS
from experiments.policy_pca_grid import (
    PolicyPcaGridSpec,
    build_policy_pca_condition,
    write_policy_pca_outputs,
    _grid_output_dir,
)
from objective.policy_preprocessing import fit_policy_feature_preprocessor


class _AcceptanceModel:
    pass


class _LossModel:
    pass


def test_policy_pca_grid_verbose_defaults_to_true() -> None:
    assert PolicyPcaGridSpec().verbose is True


def test_policy_pca_grid_default_seed_is_single_42() -> None:
    assert PolicyPcaGridSpec().seeds == (42,)


def test_policy_pca_grid_default_output_root_uses_results_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GENERALI_RESULTS_ROOT", str(tmp_path / "results"))

    output_dir = _grid_output_dir(PolicyPcaGridSpec())

    assert output_dir.parent == tmp_path / "results" / "policy-pca-grid"


def test_build_condition_uses_policy_preprocessor_dimension() -> None:
    rng = np.random.default_rng(123)
    x_fixed = rng.normal(size=(8, len(ACCEPTANCE_STATE_COLS)))
    x_policy = x_fixed[:, : len(ACCEPTANCE_STATE_COLS)]
    preprocessor = fit_policy_feature_preprocessor(x_policy, pca_dim=4)
    spec = PolicyPcaGridSpec(n_samples=x_fixed.shape[0], seeds=(1,), t_steps=2)

    condition = build_policy_pca_condition(
        spec=spec,
        policy_class="quadratic",
        pca_dim=4,
        seed=1,
        x_fixed=x_fixed,
        row_indices=np.arange(x_fixed.shape[0]),
        acceptance_model=_AcceptanceModel(),
        loss_model=_LossModel(),
        u_coef=0.0,
        policy_preprocessor=preprocessor,
    )

    assert condition.config.objective.policy_input_dim() == 4
    assert condition.config.objective.policy_feature_cols is None
    assert condition.config.theta0 is None
    assert condition.config.objective.policy_theta_dim() == 15


def test_build_condition_supports_softmax_feature_policy() -> None:
    from objective.policy import QuadraticFeatureMap, SoftmaxPolicy

    rng = np.random.default_rng(123)
    x_fixed = rng.normal(size=(8, len(ACCEPTANCE_STATE_COLS)))
    x_policy = x_fixed[:, : len(ACCEPTANCE_STATE_COLS)]
    preprocessor = fit_policy_feature_preprocessor(x_policy, pca_dim=3)
    spec = PolicyPcaGridSpec(n_samples=x_fixed.shape[0], seeds=(1,), t_steps=2)

    condition = build_policy_pca_condition(
        spec=spec,
        policy_class="softmax_quadratic",
        pca_dim=3,
        seed=1,
        x_fixed=x_fixed,
        row_indices=np.arange(x_fixed.shape[0]),
        acceptance_model=_AcceptanceModel(),
        loss_model=_LossModel(),
        u_coef=0.0,
        policy_preprocessor=preprocessor,
    )

    policy = condition.config.objective.policy
    assert isinstance(policy, SoftmaxPolicy)
    assert isinstance(policy.feature_map, QuadraticFeatureMap)
    assert condition.config.theta0 is not None
    assert condition.config.theta0.size == condition.config.objective.policy_theta_dim()
    assert np.all(condition.config.theta0 == 0.0)


def test_build_condition_supports_constrained_grid() -> None:
    rng = np.random.default_rng(123)
    x_fixed = rng.normal(size=(8, len(ACCEPTANCE_STATE_COLS)))
    x_policy = x_fixed[:, : len(ACCEPTANCE_STATE_COLS)]
    preprocessor = fit_policy_feature_preprocessor(x_policy, pca_dim=2)
    spec = PolicyPcaGridSpec(
        n_samples=x_fixed.shape[0],
        seeds=(1,),
        step_rule="trust-constr",
        acceptance_floor=0.8,
        initial_constr_penalty=1.0,
        t_steps=500,
    )

    condition = build_policy_pca_condition(
        spec=spec,
        policy_class="constant",
        pca_dim=2,
        seed=1,
        x_fixed=x_fixed,
        row_indices=np.arange(x_fixed.shape[0]),
        acceptance_model=_AcceptanceModel(),
        loss_model=_LossModel(),
        u_coef=0.0,
        policy_preprocessor=preprocessor,
    )

    assert condition.config.step_rule == "trust-constr"
    assert condition.config.acceptance_floor == 0.8
    assert condition.config.initial_constr_penalty == 1.0
    assert condition.config.t_steps == 500


def test_write_policy_pca_outputs_creates_csvs_and_plots(tmp_path: Path) -> None:
    final_rows = [
        _final_row("constant", 2, 1, -1.0),
        _final_row("linear", 2, 1, -1.5),
        _final_row("constant", 4, 1, -1.1),
        _final_row("linear", 4, 1, -1.8),
    ]
    trace_rows = [
        {
            "pca_dim": 2,
            "standardize": True,
            "sphere": True,
            "seed": 1,
            "policy_class": "linear",
            "estimator": "first_order",
            "n_samples": 10,
            "dim_policy_input": 2,
            "dim_theta": 3,
            "step": 0,
            "u": 0.0,
            "objective": -1.5,
            "theta_grad_norm": 0.1,
            "mean_acceptance": 0.8,
            "step_size": "",
        }
    ]

    write_policy_pca_outputs(final_rows, trace_rows, tmp_path)

    assert (tmp_path / "policy_pca_finals.csv").exists()
    assert (tmp_path / "policy_pca_traces.csv").exists()
    assert (tmp_path / "policy_pca_summary.md").exists()
    assert (tmp_path / "policy_pca_final_objective.png").exists()
    assert (tmp_path / "policy_pca_richness_gap.png").exists()
    assert (tmp_path / "policy_pca_u_spread.png").exists()
    assert (tmp_path / "policy_pca_acceptance_spread.png").exists()


def _final_row(policy_class: str, pca_dim: int, seed: int, final_value: float) -> dict[str, object]:
    return {
        "pca_dim": pca_dim,
        "standardize": True,
        "sphere": True,
        "seed": seed,
        "policy_class": policy_class,
        "estimator": "first_order",
        "step_rule": "l-bfgs-b",
        "acceptance_floor": "",
        "n_samples": 10,
        "dim_policy_input": pca_dim,
        "dim_theta": pca_dim + 1,
        "final_u": 0.0,
        "final_value": final_value,
        "final_objective_sum": 10 * final_value,
        "runtime_sec": 0.1,
        "mean_acceptance": 0.8,
        "final_u_std": 0.05,
        "final_u_p05": -0.1,
        "final_u_p50": 0.0,
        "final_u_p95": 0.1,
        "final_u_iqr90": 0.2,
        "final_acceptance_std": 0.04,
        "final_acceptance_p05": 0.7,
        "final_acceptance_p50": 0.8,
        "final_acceptance_p95": 0.9,
        "final_acceptance_iqr90": 0.2,
        "constraint_violation": "",
        "acceptance_multiplier": "",
        "constraint_penalty": "",
        "theta_l2_norm": 1.0,
        "theta_delta_l2_norm": 0.5,
        "objective_value_calls": 2,
        "estimated_m_evals": 20,
        "optimizer_success": True,
        "optimizer_status": 0,
        "optimizer_message": "ok",
        "converged": True,
        "error": "",
    }
