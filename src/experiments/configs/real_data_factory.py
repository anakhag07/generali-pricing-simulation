"""Factory for real-data model-based experiment configs."""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

import numpy as np

from data.loader import (
    AcceptanceModelType,
    ACCEPTANCE_STATE_COLS,
    eligible_csv_row_indices,
    FEATURE_COLS_GLM,
    FEATURE_COLS_XGB,
    LossModelType,
    LOSS_TARGET_COL,
    LOSS_FEATURE_COLS,
    PREMIUM_COL,
    extract_glm_u_coef,
    load_mean_observed_acceptance,
    load_model_artifact_pair,
    load_observed_loss_array,
    load_x_frame,
    sample_csv_row_indices,
    resolve_model_artifact_ids,
)
from experiments.config import (
    CorrectnessSpec,
    ExperimentConfig,
    build_experiment_config,
    canonical_runtime_block,
    canonical_training_block,
    make_model_based_objective,
)
from objective.policy import (
    ConstantPolicy,
    CubicFeatureMap,
    FeatureMap,
    IdentityFeatureMap,
    LinearPolicy,
    MLPPolicy,
    QuadraticFeatureMap,
    QuarticFeatureMap,
    SoftmaxPolicy,
    mlp_init_theta,
)
from objective.policy_preprocessing import fit_policy_feature_preprocessor
from experiments.seeds import SeedSetup, resolve_seed_setup

ModelType = Literal["glm", "xgb", "xgb_logit_spline"]
PolicyKind = Literal["constant", "linear", "softmax", "mlp"]
FeatureOrder = Literal["linear", "identity", "quadratic", "cubic", "third_order", "quartic", "fourth_order"]
PolicyPreprocessing = Literal["artifact", "no_pca"]
ConstraintMode = Literal["none", "trust_constr", "trust-constr", "penalty", "lagrangian"]
LossSource = Literal["predicted", "observed"]


def build_real_data_config(
    *,
    model_type: ModelType | None = None,
    acceptance_model_type: AcceptanceModelType | None = None,
    loss_model_type: LossModelType | None = None,
    row_cohort_model_type: AcceptanceModelType | None = None,
    policy_kind: PolicyKind = "softmax",
    feature_order: FeatureOrder = "linear",
    policy_preprocessing: PolicyPreprocessing = "artifact",
    constraint_mode: ConstraintMode = "none",
    loss_source: LossSource = "predicted",
    softmax_action_bounds: tuple[float, float] | None = None,
    seed: int = 42,
    seed_setup: SeedSetup | Mapping[str, int | None] | None = None,
    n_samples: int | None = None,
    train_fraction: float = 1.0,
    test_fraction: float = 0.0,
    row_indices: np.ndarray | None = None,
    x_fixed: object | None = None,
    x_fixed_row_indices: np.ndarray | None = None,
    t_steps: int | None = None,
    step_rule: str | None = None,
    compute_backend: Literal["numpy", "jax"] = "numpy",
    step_size: float = 0.01,
    sigma: float | None = None,
    n_grad_samples: int | None = None,
    enabled_estimators: tuple[str, ...] | None = None,
    perturbation_space: Literal["theta", "u"] = "u",
    batch_size: int | None = None,
    grad_norm_tol: float | None = 1e-6,
    ftol: float | None = None,
    initial_constr_penalty: float | None = None,
    acceptance_floor: float | None = None,
    acceptance_penalty_weight: float | None = None,
    acceptance_penalty_temperature: float | None = None,
    lagrangian_lambda: float | None = None,
    u_bounds: tuple[float, float] | None = None,
    u_coef: float | None = None,
    objective_modifications: Sequence[object] = (),
    initial_u: float | None = None,
    theta0: np.ndarray | None | Literal["auto"] = "auto",
    constant_u_baselines: Sequence[float] = (),
    plot: bool = True,
    verbose: bool = True,
    wandb_enabled: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = "generali-pricing",
    wandb_group: str | None = None,
    wandb_job_type: str = "experiment",
    wandb_tags: tuple[str, ...] = (),
    wandb_mode: Literal["online", "offline", "disabled"] = "online",
    wandb_log_plots: bool = True,
    wandb_estimator_allowlist: tuple[str, ...] | None = None,
) -> ExperimentConfig:
    """Build a real-data config; omitted ``n_samples`` uses all complete rows."""
    if model_type is not None:
        model_type = _normalize_model_type(model_type)
    acceptance_model_type, loss_model_type = resolve_model_artifact_ids(
        model_type=model_type,
        acceptance_model_type=acceptance_model_type,
        loss_model_type=loss_model_type,
    )
    data_model_type = row_cohort_model_type or acceptance_model_type
    constraint_mode = _normalize_constraint_mode(constraint_mode)
    loss_source = _normalize_loss_source(loss_source)
    resolved_seeds = resolve_seed_setup(seed_setup, seed)
    requested_n_samples = _normalize_n_samples(n_samples)
    state_dim = len(_feature_cols(acceptance_model_type))
    if _is_curve_acceptance(acceptance_model_type) and compute_backend != "numpy":
        raise ValueError("Per-policy acceptance curves support only compute_backend='numpy'.")
    if u_coef is not None and not _is_glm_acceptance(acceptance_model_type):
        raise ValueError("u_coef override is supported only for GLM acceptance artifacts.")
    if _is_curve_acceptance(acceptance_model_type) and policy_kind == "softmax" and softmax_action_bounds is None:
        softmax_action_bounds = (0.0, 0.16)
    if softmax_action_bounds is not None and policy_kind != "softmax":
        raise ValueError("softmax_action_bounds is supported only when policy_kind='softmax'.")
    acceptance_model, loss_model = load_model_artifact_pair(
        acceptance_model_type, loss_model_type
    )
    artifact_u_coef = (
        extract_glm_u_coef(acceptance_model)
        if _is_glm_acceptance(acceptance_model_type)
        else None
    )
    effective_u_coef = float(u_coef) if u_coef is not None else artifact_u_coef

    if x_fixed is None:
        if row_indices is None:
            if requested_n_samples is None:
                row_indices = eligible_csv_row_indices(data_model_type)
            else:
                row_indices = sample_csv_row_indices(
                    data_model_type,
                    n_rows=requested_n_samples,
                    seed=resolved_seeds.data_seed,
                )
        x_fixed_arr = load_x_frame(acceptance_model_type, row_indices=row_indices)
        x_fixed_row_indices_arr = np.asarray(row_indices, dtype=int)
        if loss_source == "observed":
            x_fixed_arr = x_fixed_arr.copy()
            x_fixed_arr[LOSS_TARGET_COL] = load_observed_loss_array(
                data_model_type,
                row_indices=x_fixed_row_indices_arr,
            )
    else:
        x_fixed_arr = x_fixed.reset_index(drop=True).copy() if hasattr(x_fixed, "iloc") else np.asarray(x_fixed, dtype=object)
        x_fixed_row_indices_arr = (
            np.asarray(x_fixed_row_indices, dtype=int)
            if x_fixed_row_indices is not None
            else None
        )
        if loss_source == "observed":
            if not hasattr(x_fixed_arr, "columns"):
                raise ValueError("loss_source='observed' requires DataFrame x_fixed or generated real-data rows.")
            if LOSS_TARGET_COL not in x_fixed_arr.columns:
                if x_fixed_row_indices_arr is None:
                    raise ValueError(
                        "loss_source='observed' requires Y_G_Loss in x_fixed or x_fixed_row_indices to load it."
                    )
                x_fixed_arr = x_fixed_arr.copy()
                x_fixed_arr[LOSS_TARGET_COL] = load_observed_loss_array(
                    data_model_type,
                    row_indices=x_fixed_row_indices_arr,
                )

    resolved_n_samples = (
        requested_n_samples
        if requested_n_samples is not None
        else int(x_fixed_arr.shape[0])
    )

    policy_preprocessor = None
    policy_feature_cols = None
    if policy_preprocessing == "no_pca":
        x_policy = _artifact_policy_features(acceptance_model, x_fixed_arr)
        policy_preprocessor = fit_policy_feature_preprocessor(
            x_policy,
            standardize=True,
            sphere=True,
            pca_dim=None,
        )
        policy_input_dim = int(policy_preprocessor.output_dim_)
    elif policy_preprocessing == "artifact":
        policy_input_dim = _artifact_policy_input_dim(acceptance_model)
    else:
        raise ValueError("policy_preprocessing must be 'artifact' or 'no_pca'.")

    policy = _make_policy(policy_kind, feature_order, softmax_action_bounds)
    theta0_arr = _resolve_theta0(
        theta0=theta0,
        policy=policy,
        input_dim=policy_input_dim,
        seed=resolved_seeds.theta_seed,
        initial_u=initial_u,
    )

    floor = acceptance_floor
    if constraint_mode != "none" and floor is None:
        floor = load_mean_observed_acceptance(data_model_type)

    resolved_step_rule = step_rule or _default_step_rule(constraint_mode)
    resolved_t_steps = int(t_steps if t_steps is not None else (500 if constraint_mode == "trust_constr" else 1000))
    resolved_sigma = float(sigma if sigma is not None else _default_sigma(policy_kind))
    resolved_n_grad_samples = int(n_grad_samples if n_grad_samples is not None else _default_n_grad_samples(acceptance_model_type, constraint_mode))
    resolved_enabled_estimators = enabled_estimators or _default_estimators(acceptance_model_type, policy_kind)
    resolved_initial_constr_penalty = initial_constr_penalty
    resolved_penalty_weight = acceptance_penalty_weight
    resolved_penalty_temperature = acceptance_penalty_temperature
    resolved_lagrangian_lambda = lagrangian_lambda
    resolved_u_bounds = (
        tuple(float(value) for value in u_bounds)
        if u_bounds is not None
        else None
    )
    if _is_curve_acceptance(acceptance_model_type) and resolved_u_bounds is None:
        resolved_u_bounds = (0.0, 0.16)

    if constraint_mode == "trust_constr" and resolved_initial_constr_penalty is None:
        resolved_initial_constr_penalty = 1.0
    if constraint_mode == "penalty":
        if resolved_penalty_weight is None:
            resolved_penalty_weight = 1e4
        if resolved_penalty_temperature is None:
            resolved_penalty_temperature = 0.05
        if _is_raw_xgb_acceptance(acceptance_model_type) and resolved_u_bounds is None:
            resolved_u_bounds = (-0.05, 0.5)
    elif resolved_penalty_temperature is None:
        resolved_penalty_temperature = 0.01
    if constraint_mode == "lagrangian" and resolved_lagrangian_lambda is None:
        resolved_lagrangian_lambda = 250.0

    objective = make_model_based_objective(
        policy=policy,
        acceptance_model=acceptance_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=PREMIUM_COL,
        loss_source=loss_source,
        observed_loss_col=LOSS_TARGET_COL,
        u_coef=effective_u_coef,
        u_bounds=resolved_u_bounds,
        policy_preprocessor=policy_preprocessor,
        policy_feature_cols=policy_feature_cols,
    )

    training = canonical_training_block(
        n_samples=resolved_n_samples,
        train_fraction=float(train_fraction),
        test_fraction=float(test_fraction),
        step_rule=resolved_step_rule,
        compute_backend=compute_backend,
        t_steps=resolved_t_steps,
        step_size=float(step_size),
        sigma=resolved_sigma,
        n_grad_samples=resolved_n_grad_samples,
        enabled_estimators=tuple(resolved_enabled_estimators),
        constant_u_baselines=tuple(float(u) for u in constant_u_baselines),
        perturbation_space=perturbation_space,
        batch_size=batch_size,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        initial_constr_penalty=resolved_initial_constr_penalty,
        acceptance_floor=floor,
        acceptance_penalty_weight=resolved_penalty_weight,
        acceptance_penalty_temperature=float(resolved_penalty_temperature),
        lagrangian_lambda=resolved_lagrangian_lambda,
    )
    runtime = canonical_runtime_block(
        plot=plot,
        verbose=verbose,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project or _default_project_name(
            acceptance_model_type,
            loss_model_type,
            policy_kind,
            feature_order,
            policy_preprocessing,
            constraint_mode,
        ),
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_job_type=wandb_job_type,
        wandb_tags=wandb_tags,
        wandb_mode=wandb_mode,
        wandb_log_plots=wandb_log_plots,
        wandb_estimator_allowlist=wandb_estimator_allowlist,
    )
    return build_experiment_config(
        seed=resolved_seeds.run_seed,
        seed_setup=seed_setup,
        state_dim=state_dim,
        x_fixed=x_fixed_arr,
        x_fixed_row_indices=x_fixed_row_indices_arr,
        objective=objective,
        objective_modifications=objective_modifications,
        theta0=theta0_arr,
        training=training,
        runtime=runtime,
        correctness=CorrectnessSpec(gradient_source="none"),
    )


def _normalize_model_type(model_type: str) -> ModelType:
    if model_type not in {"glm", "xgb", "xgb_logit_spline"}:
        raise ValueError("model_type must be 'glm', 'xgb', or 'xgb_logit_spline'.")
    return model_type  # type: ignore[return-value]


def _normalize_constraint_mode(mode: str) -> Literal["none", "trust_constr", "penalty", "lagrangian"]:
    if mode == "trust-constr":
        mode = "trust_constr"
    if mode not in {"none", "trust_constr", "penalty", "lagrangian"}:
        raise ValueError("constraint_mode must be none, trust_constr, penalty, or lagrangian.")
    return mode  # type: ignore[return-value]


def _normalize_loss_source(loss_source: str) -> LossSource:
    if loss_source not in {"predicted", "observed"}:
        raise ValueError("loss_source must be 'predicted' or 'observed'.")
    return loss_source  # type: ignore[return-value]


def _normalize_n_samples(n_samples: int | None) -> int | None:
    if n_samples is None:
        return None
    n_samples_int = int(n_samples)
    if n_samples_int <= 0:
        raise ValueError("n_samples must be positive when provided.")
    return n_samples_int


def _feature_cols(model_type: AcceptanceModelType) -> tuple[str, ...]:
    return tuple(
        FEATURE_COLS_GLM
        if _is_glm_acceptance(model_type)
        else FEATURE_COLS_XGB
    )


def _is_glm_acceptance(model_type: AcceptanceModelType) -> bool:
    return model_type == "glm_20260527"


def _is_raw_xgb_acceptance(model_type: AcceptanceModelType) -> bool:
    return model_type in {"xgb_20260527", "xgb_20260728"}


def _is_curve_acceptance(model_type: AcceptanceModelType) -> bool:
    return model_type in {
        "xgb_logit_spline_20260706",
        "xgb_monotone_spline_20260728",
    }


def _feature_map(feature_order: FeatureOrder) -> FeatureMap:
    if feature_order in {"linear", "identity"}:
        return IdentityFeatureMap()
    if feature_order == "quadratic":
        return QuadraticFeatureMap()
    if feature_order in {"cubic", "third_order"}:
        return CubicFeatureMap()
    if feature_order in {"quartic", "fourth_order"}:
        return QuarticFeatureMap()
    raise ValueError(f"Unknown feature_order '{feature_order}'.")


def _make_policy(
    policy_kind: PolicyKind,
    feature_order: FeatureOrder,
    softmax_action_bounds: tuple[float, float] | None = None,
) -> object:
    feature_map = _feature_map(feature_order)
    if policy_kind == "constant":
        return ConstantPolicy()
    if policy_kind == "linear":
        return LinearPolicy(feature_map=feature_map)
    if policy_kind == "softmax":
        if softmax_action_bounds is None:
            return SoftmaxPolicy(feature_map=feature_map)
        low, high = softmax_action_bounds
        return SoftmaxPolicy(
            feature_map=feature_map,
            action_low=float(low),
            action_high=float(high),
        )
    if policy_kind == "mlp":
        return MLPPolicy(feature_map=feature_map)
    raise ValueError(f"Unknown policy_kind '{policy_kind}'.")


def _artifact_policy_input_dim(acceptance_model: object) -> int:
    policy_feature_dim = getattr(acceptance_model, "policy_feature_dim", None)
    if callable(policy_feature_dim):
        return int(policy_feature_dim())
    return len(ACCEPTANCE_STATE_COLS)


def _artifact_policy_features(acceptance_model: object, x_fixed: object) -> np.ndarray:
    """Return numeric acceptance-preprocessed features for policy-side preprocessing."""
    if hasattr(x_fixed, "iloc"):
        x_frame = x_fixed.reset_index(drop=True)
    else:
        x_frame = np.asarray(x_fixed, dtype=object)
    x_feature_cols = tuple(getattr(acceptance_model, "x_feature_cols", ACCEPTANCE_STATE_COLS))
    preprocessor = getattr(acceptance_model, "preprocessor", None)
    if hasattr(x_frame, "loc"):
        raw_features = x_frame.loc[:, list(x_feature_cols)].copy()
        if preprocessor is None:
            return raw_features.to_numpy(dtype=float)
        return np.asarray(preprocessor.transform(raw_features), dtype=float)
    x_arr = np.asarray(x_frame, dtype=object)
    if x_arr.shape[1] != len(ACCEPTANCE_STATE_COLS):
        raise ValueError("Array x_fixed must use the configured acceptance-state column order.")
    raw_features = x_arr[:, : len(x_feature_cols)]
    if preprocessor is None:
        return raw_features.astype(float)
    # Raw categorical arrays cannot be safely mapped without column names; real-data
    # no-PCA preprocessing should use DataFrame batches.
    raise ValueError("policy_preprocessing='no_pca' requires DataFrame x_fixed for 052726 artifacts.")


def _resolve_theta0(
    *,
    theta0: np.ndarray | None | Literal["auto"],
    policy: object,
    input_dim: int,
    seed: int,
    initial_u: float | None,
) -> np.ndarray | None:
    if not isinstance(theta0, str):
        return np.asarray(theta0, dtype=float) if theta0 is not None else None
    if theta0 != "auto":
        raise ValueError("theta0 must be an array, None, or 'auto'.")
    if isinstance(policy, ConstantPolicy):
        return np.asarray([0.0 if initial_u is None else float(initial_u)], dtype=float)
    if isinstance(policy, SoftmaxPolicy):
        theta = np.zeros(policy.theta_dim(input_dim), dtype=float)
    elif isinstance(policy, MLPPolicy):
        theta = mlp_init_theta(np.random.default_rng(seed), d_in=input_dim, hidden=policy.hidden)
    elif initial_u is not None:
        theta = np.zeros(policy.theta_dim(input_dim), dtype=float)
    else:
        return None
    if initial_u is not None:
        if isinstance(policy, SoftmaxPolicy):
            theta[0] = _softmax_intercept_for_u(policy, float(initial_u))
        else:
            theta[0] = float(initial_u)
    return theta


def _softmax_intercept_for_u(policy: SoftmaxPolicy, initial_u: float) -> float:
    low = float(policy.action_low)
    high = float(policy.action_high)
    if not low < float(initial_u) < high:
        raise ValueError("initial_u must lie strictly inside SoftmaxPolicy action bounds.")
    p = (float(initial_u) - low) / (high - low)
    return float(np.log(p / (1.0 - p)))


def _default_step_rule(constraint_mode: str) -> str:
    return "trust-constr" if constraint_mode == "trust_constr" else "l-bfgs-b"


def _default_sigma(policy_kind: PolicyKind) -> float:
    return 0.05 if policy_kind in {"softmax", "mlp"} else 0.01


def _default_n_grad_samples(
    model_type: AcceptanceModelType, constraint_mode: str
) -> int:
    if _is_raw_xgb_acceptance(model_type) and constraint_mode == "penalty":
        return 10
    return 50


def _default_estimators(
    model_type: AcceptanceModelType, policy_kind: PolicyKind
) -> tuple[str, ...]:
    if _is_raw_xgb_acceptance(model_type):
        return ("finite_difference", "spsa", "stein_difference")
    if policy_kind == "mlp":
        return ("first_order", "spsa", "stein_difference")
    return ("first_order", "finite_difference", "spsa", "stein_difference")


def _default_project_name(
    acceptance_model_type: AcceptanceModelType,
    loss_model_type: LossModelType,
    policy_kind: PolicyKind,
    feature_order: FeatureOrder,
    policy_preprocessing: PolicyPreprocessing,
    constraint_mode: str,
) -> str:
    parts = [
        f"acceptance-{acceptance_model_type}",
        f"loss-{loss_model_type}",
        policy_kind,
    ]
    if policy_kind in {"linear", "softmax", "mlp"}:
        parts.append("linear" if feature_order == "identity" else feature_order.replace("_", "-"))
    if policy_preprocessing != "artifact":
        parts.append(policy_preprocessing.replace("_", "-"))
    if constraint_mode != "none":
        parts.append(constraint_mode.replace("_", "-"))
    return "-".join(parts)
