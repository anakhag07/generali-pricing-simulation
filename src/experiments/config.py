"""Experiment configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np

from objective.action_regularizers import (
    ActionRegularizedObjective,
    CallableSigmaProvider,
    HeteroskedasticNoiseScaleProvider,
    HomoskedasticNoiseScaleProvider,
    SigmaProvider,
)
from objective.base import Objective, Policy
from objective.objectives import (
    ActionBias,
    BiasedObjective,
    FixedRegressionObjective,
    LinearActionBias,
    ModelBasedObjective,
    PlantedLogisticObjective,
    UpperSupportHingeBias,
)
from objective.noise import (
    HeteroskedasticGaussianNoise,
    HomoskedasticGaussianNoise,
    NoisyObjective,
    NoNoise,
    ObjectiveNoise,
)
from objective.policy import ConstantPolicy, LinearPolicy, MLPPolicy, SoftmaxPolicy, policy_theta_dim
from objective.policy_preprocessing import PolicyFeaturePreprocessor
from optimization.steps import OPTAX_STEP_RULES, STEP_RULES, STEP_RULE_TRUST_CONSTR
from experiments.seeds import SeedSetup, resolve_seed_setup, seed_setup_from_mapping

UReferenceSource = Literal["array", "constant", "historical"]


def _policy_theta_dim_for_objective(objective: object, state_dim: int) -> int | None:
    """Return the required policy theta dimension when the objective exposes one."""
    objective_theta_dim = getattr(objective, "policy_theta_dim", None)
    if callable(objective_theta_dim):
        return int(objective_theta_dim(state_dim))
    policy = getattr(objective, "policy", None)
    if policy is None:
        return None
    return policy_theta_dim(policy, state_dim)


def _x_fixed_frame_matches_state_dim(objective: object, x_fixed_frame: Any, state_dim: int) -> bool:
    if getattr(objective, "loss_source", "predicted") != "observed":
        return x_fixed_frame.shape[1] == state_dim
    if x_fixed_frame.shape[1] == state_dim:
        return False
    state_cols = tuple(getattr(objective, "acceptance_state_cols", ()))
    observed_loss_col = getattr(objective, "observed_loss_col", None)
    if not state_cols or observed_loss_col is None:
        return False
    expected_cols = set(state_cols) | {str(observed_loss_col)}
    return (
        len(state_cols) == state_dim
        and set(x_fixed_frame.columns) == expected_cols
        and x_fixed_frame.shape[1] == state_dim + 1
    )


@dataclass(frozen=True)
class CorrectnessSpec:
    """Controls how "true" gradients are computed for diagnostics."""

    gradient_source: Literal["exact", "denoised_exact", "numdiff", "none"] = "exact"
    numdiff_method: Literal["central", "forward", "backward"] = "central"
    numdiff_step: float = 1e-4
    numdiff_aggregate: Literal["per-sample", "batch"] = "batch"
    numdiff_bounds: Optional[tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.gradient_source not in {"exact", "denoised_exact", "numdiff", "none"}:
            raise ValueError(
                "gradient_source must be 'exact', 'denoised_exact', 'numdiff', or 'none'."
            )
        if self.numdiff_method not in {"central", "forward", "backward"}:
            raise ValueError("numdiff_method must be 'central', 'forward', or 'backward'.")
        if self.numdiff_step <= 0.0:
            raise ValueError("numdiff_step must be positive.")
        if self.numdiff_aggregate not in {"per-sample", "batch"}:
            raise ValueError("numdiff_aggregate must be 'per-sample' or 'batch'.")
        if self.numdiff_bounds is not None:
            lower, upper = self.numdiff_bounds
            lower = float(lower)
            upper = float(upper)
            if lower >= upper:
                raise ValueError("numdiff_bounds must be an increasing (lower, upper) tuple.")
            object.__setattr__(self, "numdiff_bounds", (lower, upper))


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen configuration for a single experiment run with validation."""

    state_dim: int
    n_samples: int
    step_rule: str
    objective: Objective
    perturbation_space: Literal["theta", "u"]
    compute_backend: Literal["numpy", "jax"] = "numpy"
    train_fraction: float = 1.0
    test_fraction: float = 0.0
    theta0: np.ndarray | None = None
    batch_size: int | None = None
    seed: int = 7
    seed_setup: SeedSetup | Mapping[str, int | None] | None = None
    t_steps: int = 100
    step_size: float = 0.01
    grad_norm_tol: Optional[float] = None
    ftol: Optional[float] = None
    initial_constr_penalty: float | None = None
    acceptance_floor: float | None = None
    acceptance_penalty_weight: float | None = None
    acceptance_penalty_temperature: float = 0.01
    lagrangian_lambda: float | None = None
    proximal_weight: float | None = None
    u_reference: np.ndarray | None = None
    u_reference_source: UReferenceSource | None = None
    u_reference_value: float | None = None
    support_weight: float | None = None
    sigma_provider: SigmaProvider | None = None
    sigma: float = 0.1
    n_grad_samples: int = 64
    verbose: bool = False
    plot: bool = True
    enabled_estimators: tuple[str, ...] = ("first_order", "finite_difference", "stein_difference")
    constant_u_baselines: tuple[float, ...] = ()
    wandb_enabled: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str = "experiment"
    wandb_tags: tuple[str, ...] = ()
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_log_plots: bool = True
    wandb_estimator_allowlist: tuple[str, ...] | None = None
    correctness: CorrectnessSpec = field(default_factory=CorrectnessSpec)
    x_fixed: Any | None = None  # real data rows; replaces sample_states when set
    x_fixed_row_indices: np.ndarray | None = None  # source CSV row positions for x_fixed

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", int(self.seed))
        if self.seed_setup is not None:
            object.__setattr__(self, "seed_setup", seed_setup_from_mapping(self.seed_setup))

        estimator_aliases = {
            "finite-difference": "finite_difference",
            "stein-difference": "stein_difference",
        }
        enabled_estimators = tuple(estimator_aliases.get(name, name) for name in self.enabled_estimators)
        object.__setattr__(self, "enabled_estimators", enabled_estimators)
        if not enabled_estimators:
            raise ValueError("enabled_estimators must include at least one estimator.")
        if len(set(enabled_estimators)) != len(enabled_estimators):
            raise ValueError("enabled_estimators must not contain duplicates.")
        allowed_estimators = {
            "constant",
            "first_order",
            "finite_difference",
            "gauss_stein",
            "spsa",
            "stein_difference",
        }
        unknown = [name for name in enabled_estimators if name not in allowed_estimators]
        if unknown:
            allowed = ", ".join(sorted(allowed_estimators))
            unknown_list = ", ".join(unknown)
            raise ValueError(f"Unknown estimators: {unknown_list}. Allowed: {allowed}.")

        constant_u_baselines = tuple(float(u) for u in self.constant_u_baselines)
        object.__setattr__(self, "constant_u_baselines", constant_u_baselines)
        if len(set(constant_u_baselines)) != len(constant_u_baselines):
            raise ValueError("constant_u_baselines must not contain duplicates.")
        if not all(np.isfinite(u) for u in constant_u_baselines):
            raise ValueError("constant_u_baselines must contain only finite values.")

        if self.perturbation_space not in {"theta", "u"}:
            raise ValueError("perturbation_space must be 'theta' or 'u'.")

        wandb_tags = tuple(self.wandb_tags)
        object.__setattr__(self, "wandb_tags", wandb_tags)
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be 'online', 'offline', or 'disabled'.")
        if self.wandb_enabled and self.wandb_mode == "disabled":
            raise ValueError("wandb_mode='disabled' is incompatible with wandb_enabled=True.")
        if self.wandb_estimator_allowlist is not None:
            wandb_allowlist = tuple(estimator_aliases.get(name, name) for name in self.wandb_estimator_allowlist)
            object.__setattr__(self, "wandb_estimator_allowlist", wandb_allowlist)
            if len(set(wandb_allowlist)) != len(wandb_allowlist):
                raise ValueError("wandb_estimator_allowlist must not contain duplicates.")
            unknown_wandb = [name for name in wandb_allowlist if name not in allowed_estimators]
            if unknown_wandb:
                allowed = ", ".join(sorted(allowed_estimators))
                unknown_list = ", ".join(unknown_wandb)
                raise ValueError(
                    f"Unknown wandb estimators: {unknown_list}. Allowed: {allowed}."
                )

        if self.perturbation_space == "u":
            policy = getattr(self.objective, "policy", None)
            if policy is None or not callable(getattr(policy, "value", None)) or not callable(getattr(policy, "grad", None)):
                raise ValueError(
                    "perturbation_space='u' requires objective.policy with value() and grad()."
                )

        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        train_fraction = float(self.train_fraction)
        test_fraction = float(self.test_fraction)
        if not np.isfinite(train_fraction) or not np.isfinite(test_fraction):
            raise ValueError("train_fraction and test_fraction must be finite.")
        if not 0.0 <= train_fraction <= 1.0:
            raise ValueError("train_fraction must be in [0, 1].")
        if not 0.0 <= test_fraction <= 1.0:
            raise ValueError("test_fraction must be in [0, 1].")
        if train_fraction <= 0.0:
            raise ValueError("train_fraction must be positive.")
        if not np.isclose(train_fraction + test_fraction, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("train_fraction and test_fraction must sum to 1.0.")
        object.__setattr__(self, "train_fraction", train_fraction)
        object.__setattr__(self, "test_fraction", test_fraction)

        if self.theta0 is not None:
            theta0_arr = np.asarray(self.theta0, dtype=float)
            if theta0_arr.ndim != 1 or theta0_arr.size < 1:
                raise ValueError("theta0 must be a 1D array with at least one element.")
            object.__setattr__(self, "theta0", theta0_arr)

        if self.x_fixed is not None:
            if hasattr(self.x_fixed, "iloc") and hasattr(self.x_fixed, "columns"):
                x_fixed_frame = self.x_fixed.reset_index(drop=True).copy()
                if x_fixed_frame.ndim != 2:
                    raise ValueError("x_fixed must be a 2D array/DataFrame of shape (n_rows, state_dim).")
                if not _x_fixed_frame_matches_state_dim(self.objective, x_fixed_frame, self.state_dim):
                    raise ValueError(
                        f"x_fixed has {x_fixed_frame.shape[1]} columns but state_dim={self.state_dim}."
                    )
                object.__setattr__(self, "x_fixed", x_fixed_frame)
            else:
                x_fixed_arr = np.asarray(self.x_fixed, dtype=float)
                if x_fixed_arr.ndim != 2:
                    raise ValueError("x_fixed must be a 2D array/DataFrame of shape (n_rows, state_dim).")
                if x_fixed_arr.shape[1] != self.state_dim:
                    raise ValueError(
                        f"x_fixed has {x_fixed_arr.shape[1]} columns but state_dim={self.state_dim}."
                    )
                object.__setattr__(self, "x_fixed", x_fixed_arr)

        if self.x_fixed_row_indices is not None:
            row_indices = np.asarray(self.x_fixed_row_indices)
            if row_indices.ndim != 1:
                raise ValueError("x_fixed_row_indices must be a 1D array.")
            if row_indices.size == 0:
                raise ValueError("x_fixed_row_indices must contain at least one row index.")
            if not np.issubdtype(row_indices.dtype, np.integer):
                raise ValueError("x_fixed_row_indices must contain integer row positions.")
            row_indices = row_indices.astype(int, copy=False)
            if np.any(row_indices < 0):
                raise ValueError("x_fixed_row_indices must be nonnegative.")
            if len(set(row_indices.tolist())) != row_indices.size:
                raise ValueError("x_fixed_row_indices must not contain duplicates.")
            if self.x_fixed is not None and row_indices.size != self.x_fixed.shape[0]:
                raise ValueError("x_fixed_row_indices length must match x_fixed rows.")
            object.__setattr__(self, "x_fixed_row_indices", row_indices)

        if self.batch_size is not None:
            if self.batch_size <= 0:
                raise ValueError("batch_size must be positive when provided.")
            if self.batch_size > self.n_samples:
                raise ValueError("batch_size must be <= n_samples when provided.")

        if self.step_rule not in STEP_RULES:
            allowed = ", ".join(sorted(STEP_RULES))
            raise ValueError(f"step_rule must be one of {allowed}.")
        if self.compute_backend not in {"numpy", "jax"}:
            raise ValueError("compute_backend must be 'numpy' or 'jax'.")
        if self.compute_backend == "jax" and (
            self.step_rule != STEP_RULE_TRUST_CONSTR and self.step_rule not in OPTAX_STEP_RULES
        ):
            raise ValueError(
                "compute_backend='jax' is currently supported only with step_rule='trust-constr' "
                "or an optax step rule."
            )
        if self.compute_backend == "jax" and self.batch_size is not None:
            raise ValueError("compute_backend='jax' requires batch_size=None because it uses a fixed full batch.")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        if self.grad_norm_tol is not None and self.grad_norm_tol <= 0.0:
            raise ValueError("grad_norm_tol must be positive when provided.")
        if self.ftol is not None and self.ftol <= 0.0:
            raise ValueError("ftol must be positive when provided.")
        if self.initial_constr_penalty is not None and self.initial_constr_penalty <= 0.0:
            raise ValueError("initial_constr_penalty must be positive when provided.")
        if self.lagrangian_lambda is not None and self.lagrangian_lambda < 0.0:
            raise ValueError("lagrangian_lambda must be nonnegative when provided.")
        if self.proximal_weight is not None:
            proximal_weight = float(self.proximal_weight)
            if not np.isfinite(proximal_weight) or proximal_weight < 0.0:
                raise ValueError("proximal_weight must be finite and nonnegative when provided.")
            object.__setattr__(self, "proximal_weight", proximal_weight)
        if self.support_weight is not None:
            support_weight = float(self.support_weight)
            if not np.isfinite(support_weight) or support_weight < 0.0:
                raise ValueError("support_weight must be finite and nonnegative when provided.")
            object.__setattr__(self, "support_weight", support_weight)
        if self.u_reference_source is not None and self.u_reference_source not in {"array", "constant", "historical"}:
            raise ValueError("u_reference_source must be 'array', 'constant', 'historical', or None.")
        if self.u_reference is not None:
            u_reference = np.asarray(self.u_reference, dtype=float)
            if u_reference.ndim != 1:
                raise ValueError("u_reference must be a 1D array.")
            if u_reference.size == 0:
                raise ValueError("u_reference must contain at least one value.")
            if not np.isfinite(u_reference).all():
                raise ValueError("u_reference must contain finite values.")
            if u_reference.size != self.n_samples:
                raise ValueError("u_reference length must match n_samples before train/test splitting.")
            object.__setattr__(self, "u_reference", u_reference.copy())
            if self.u_reference_source is None:
                object.__setattr__(self, "u_reference_source", "array")
        if self.u_reference_value is not None:
            u_reference_value = float(self.u_reference_value)
            if not np.isfinite(u_reference_value):
                raise ValueError("u_reference_value must be finite when provided.")
            object.__setattr__(self, "u_reference_value", u_reference_value)
        if self.proximal_weight is not None and self.u_reference is None:
            if self.u_reference_source != "constant":
                raise ValueError("proximal_weight requires u_reference or u_reference_source='constant'.")
            if self.u_reference_value is None:
                raise ValueError("u_reference_source='constant' requires u_reference_value.")
        if self.step_rule == STEP_RULE_TRUST_CONSTR and self.ftol is not None:
            raise ValueError("ftol is not supported when step_rule='trust-constr'.")
        if self.step_rule in OPTAX_STEP_RULES and self.ftol is not None:
            raise ValueError("ftol is not supported by optax step rules; use grad_norm_tol.")
        if self.step_rule != STEP_RULE_TRUST_CONSTR and self.initial_constr_penalty is not None:
            raise ValueError(
                "initial_constr_penalty is only used by step_rule='trust-constr'."
            )
        if self.acceptance_floor is not None:
            if not 0.0 < self.acceptance_floor < 1.0:
                raise ValueError("acceptance_floor must be in (0, 1) when provided.")
            mean_acceptance_fn = getattr(self.objective, "mean_acceptance", None)
            if not callable(mean_acceptance_fn):
                raise ValueError(
                    "acceptance_floor requires an objective with mean_acceptance(theta, x_batch)."
                )
            if self.step_rule == STEP_RULE_TRUST_CONSTR:
                mean_acceptance_grad_fn = getattr(self.objective, "mean_acceptance_grad", None)
                if not callable(mean_acceptance_grad_fn):
                    raise ValueError(
                        "step_rule='trust-constr' requires an objective with "
                        "mean_acceptance_grad(theta, x_batch)."
                    )
                if self.batch_size is not None:
                    raise ValueError(
                        "step_rule='trust-constr' requires batch_size=None so the constraint is full-batch."
                    )
                if self.lagrangian_lambda is not None:
                    raise ValueError(
                        "lagrangian_lambda is only supported for unconstrained step rules and must be omitted "
                        "when step_rule='trust-constr'."
                    )
                if self.acceptance_penalty_weight is not None:
                    raise ValueError(
                        "acceptance_penalty_weight is only used by the penalty path and must be omitted "
                        "when step_rule='trust-constr'."
                    )
                if self.acceptance_penalty_temperature != 0.01:
                    raise ValueError(
                        "acceptance_penalty_temperature is only used by the penalty path and must stay "
                        "at the default when step_rule='trust-constr'."
                    )
            else:
                if self.acceptance_penalty_weight is not None and self.lagrangian_lambda is not None:
                    raise ValueError(
                        "acceptance_penalty_weight and lagrangian_lambda are mutually exclusive; choose one "
                        "acceptance-floor path."
                    )
                if self.lagrangian_lambda is not None:
                    mean_acceptance_grad_fn = getattr(self.objective, "mean_acceptance_grad", None)
                    if not callable(mean_acceptance_grad_fn):
                        raise ValueError(
                            "lagrangian_lambda requires an objective with mean_acceptance_grad(theta, x_batch)."
                        )
                    if self.acceptance_penalty_temperature != 0.01:
                        raise ValueError(
                            "acceptance_penalty_temperature is only used by the penalty path and must stay "
                            "at the default when lagrangian_lambda is provided."
                        )
                elif self.acceptance_penalty_weight is None or self.acceptance_penalty_weight <= 0.0:
                    raise ValueError(
                        "acceptance_penalty_weight must be positive when acceptance_floor is provided."
                    )
                if self.acceptance_penalty_weight is not None and self.acceptance_penalty_temperature <= 0.0:
                    raise ValueError(
                        "acceptance_penalty_temperature must be positive when acceptance_floor is provided."
                    )
        elif self.acceptance_penalty_weight is not None:
            raise ValueError(
                "acceptance_penalty_weight requires acceptance_floor to be provided."
            )
        elif self.lagrangian_lambda is not None:
            raise ValueError("lagrangian_lambda requires acceptance_floor to be provided.")
        elif self.step_rule == STEP_RULE_TRUST_CONSTR:
            raise ValueError("step_rule='trust-constr' requires acceptance_floor to be provided.")
        if self.n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")

        objective = self.objective
        value_fn = getattr(objective, "value", None)
        grad_fn = getattr(objective, "grad", None)
        if value_fn is None or not callable(value_fn):
            raise ValueError("objective must implement value(theta, x_batch).")
        if grad_fn is None or not callable(grad_fn):
            raise ValueError("objective must implement grad(theta, x_batch).")
        if self.constant_u_baselines:
            value_at_u_fn = getattr(objective, "value_at_u", None)
            if not callable(value_at_u_fn):
                raise ValueError(
                    "constant_u_baselines require an objective with value_at_u(x_batch, u)."
                )

        policy = getattr(objective, "policy", None)
        if policy is not None:
            expected_theta_dim = _policy_theta_dim_for_objective(objective, self.state_dim)
            if self.theta0 is not None and expected_theta_dim is not None and theta0_arr.size != expected_theta_dim:
                raise ValueError(
                    f"theta0 has dimension {theta0_arr.size}, but policy requires {expected_theta_dim}."
                )
            policy_value = getattr(objective, "policy_value", None)
            policy_grad = getattr(objective, "policy_grad", None)
            if not callable(policy_value):
                policy_value = getattr(policy, "value", None)
            if not callable(policy_grad):
                policy_grad = getattr(policy, "grad", None)
            if not callable(policy_value) or not callable(policy_grad):
                raise ValueError("policy must implement value(theta, x_batch) and grad(theta, x_batch).")
            # Probe with a single-sample batch
            probe_theta = (
                theta0_arr if self.theta0 is not None
                else np.zeros(expected_theta_dim if expected_theta_dim is not None else self.state_dim + 1, dtype=float)
            )
            x_probe = np.zeros((1, self.state_dim), dtype=float)
            u_probe_arr = np.asarray(policy_value(probe_theta, x_probe), dtype=float)
            if not bool(np.isfinite(u_probe_arr).all()):
                raise ValueError("policy.value(theta0, x_batch) must be finite.")
            grad_probe = np.asarray(policy_grad(probe_theta, x_probe), dtype=float)
            if grad_probe.ndim != 2 or grad_probe.shape[1] != probe_theta.size:
                raise ValueError("policy.grad(theta0, x_batch) must return (n_samples, theta_dim).")

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary for JSON output."""
        return {
            "state_dim": int(self.state_dim),
            "n_samples": int(self.n_samples),
            "train_fraction": float(self.train_fraction),
            "test_fraction": float(self.test_fraction),
            "batch_size": int(self.batch_size) if self.batch_size is not None else None,
            "step_rule": self.step_rule,
            "compute_backend": self.compute_backend,
            "seed": int(self.seed),
            "seed_setup": self.seed_setup.to_dict()
            if self.seed_setup is not None
            else None,
            "resolved_seed_setup": resolve_seed_setup(self.seed_setup, self.seed).to_dict(),
            "t_steps": int(self.t_steps),
            "step_size": float(self.step_size),
            "grad_norm_tol": float(self.grad_norm_tol)
            if self.grad_norm_tol is not None
            else None,
            "ftol": float(self.ftol) if self.ftol is not None else None,
            "initial_constr_penalty": float(self.initial_constr_penalty)
            if self.initial_constr_penalty is not None
            else None,
            "acceptance_floor": float(self.acceptance_floor)
            if self.acceptance_floor is not None
            else None,
            "acceptance_penalty_weight": float(self.acceptance_penalty_weight)
            if self.acceptance_penalty_weight is not None
            else None,
            "acceptance_penalty_temperature": float(self.acceptance_penalty_temperature),
            "lagrangian_lambda": float(self.lagrangian_lambda)
            if self.lagrangian_lambda is not None
            else None,
            "proximal_weight": float(self.proximal_weight)
            if self.proximal_weight is not None
            else None,
            "u_reference_source": self.u_reference_source,
            "u_reference_value": float(self.u_reference_value)
            if self.u_reference_value is not None
            else None,
            "u_reference_shape": list(self.u_reference.shape)
            if self.u_reference is not None
            else None,
            "u_reference_min": float(np.min(self.u_reference))
            if self.u_reference is not None
            else None,
            "u_reference_max": float(np.max(self.u_reference))
            if self.u_reference is not None
            else None,
            "u_reference_head": [float(val) for val in self.u_reference[:10]]
            if self.u_reference is not None
            else None,
            "support_weight": float(self.support_weight)
            if self.support_weight is not None
            else None,
            "sigma_provider": _sigma_provider_to_dict(self.sigma_provider)
            if self.sigma_provider is not None
            else None,
            "sigma": float(self.sigma),
            "n_grad_samples": int(self.n_grad_samples),
            "verbose": bool(self.verbose),
            "plot": bool(self.plot),
            "enabled_estimators": list(self.enabled_estimators),
            "constant_u_baselines": [float(u) for u in self.constant_u_baselines],
            "perturbation_space": self.perturbation_space,
            "theta0": _as_list(self.theta0) if self.theta0 is not None else None,
            "objective": _objective_to_dict(self.objective),
            "wandb": {
                "enabled": bool(self.wandb_enabled),
                "project": self.wandb_project,
                "entity": self.wandb_entity,
                "group": self.wandb_group,
                "job_type": self.wandb_job_type,
                "tags": list(self.wandb_tags),
                "mode": self.wandb_mode,
                "log_plots": bool(self.wandb_log_plots),
                "estimator_allowlist": list(self.wandb_estimator_allowlist)
                if self.wandb_estimator_allowlist is not None
                else None,
            },
            "correctness": _correctness_to_dict(self.correctness),
            "x_fixed_shape": list(self.x_fixed.shape) if self.x_fixed is not None else None,
            "x_fixed_row_indices_shape": list(self.x_fixed_row_indices.shape)
            if self.x_fixed_row_indices is not None
            else None,
            "x_fixed_row_indices_min": int(np.min(self.x_fixed_row_indices))
            if self.x_fixed_row_indices is not None
            else None,
            "x_fixed_row_indices_max": int(np.max(self.x_fixed_row_indices))
            if self.x_fixed_row_indices is not None
            else None,
            "x_fixed_row_indices_head": [int(idx) for idx in self.x_fixed_row_indices[:10]]
            if self.x_fixed_row_indices is not None
            else None,
        }


def _objective_to_dict(objective: Objective) -> dict[str, Any]:
    """Serialize objective to dictionary."""
    if isinstance(objective, ActionRegularizedObjective):
        return {
            "type": "ActionRegularizedObjective",
            "base_objective": _objective_to_dict(objective.base_objective),
            "proximal_weight": float(objective.proximal_weight)
            if objective.proximal_weight is not None
            else None,
            "u_reference_shape": list(objective.u_reference.shape)
            if objective.u_reference is not None
            else None,
            "u_reference_min": float(np.min(objective.u_reference))
            if objective.u_reference is not None
            else None,
            "u_reference_max": float(np.max(objective.u_reference))
            if objective.u_reference is not None
            else None,
            "support_weight": float(objective.support_weight)
            if objective.support_weight is not None
            else None,
            "sigma_provider": _sigma_provider_to_dict(objective.sigma_provider)
            if objective.sigma_provider is not None
            else None,
        }
    if isinstance(objective, NoisyObjective):
        return {
            "type": "NoisyObjective",
            "base_objective": _objective_to_dict(objective.base_objective),
            "noise": _noise_to_dict(objective.noise),
        }
    if isinstance(objective, BiasedObjective):
        return {
            "type": "BiasedObjective",
            "base_objective": _objective_to_dict(objective.base_objective),
            "lambda_bias": float(objective.lambda_bias),
            "bias": _action_bias_to_dict(objective.bias),
        }
    if isinstance(objective, FixedRegressionObjective):
        return {
            "type": "FixedRegressionObjective",
            "policy": _policy_to_dict(objective.policy),
            "beta_1": _as_list(objective.beta_1),
            "beta_2": float(objective.beta_2),
            "beta_3": _as_list(objective.beta_3),
            "beta_4": float(objective.beta_4),
        }
    if isinstance(objective, PlantedLogisticObjective):
        return {
            "type": "PlantedLogisticObjective",
            "policy": _policy_to_dict(objective.policy),
            "alpha": float(objective.alpha),
            "beta": _as_list(objective.beta),
            "bias": float(objective.bias),
            "u_star": float(objective.u_star),
        }
    if isinstance(objective, ModelBasedObjective):
        return {
            "type": "ModelBasedObjective",
            "policy": _policy_to_dict(objective.policy),
            "acceptance_state_cols": list(objective.acceptance_state_cols),
            "loss_cols": list(objective.loss_cols),
            "premium_col": objective.premium_col,
            "loss_source": objective.loss_source,
            "observed_loss_col": objective.observed_loss_col,
            "u_coef": float(objective.u_coef) if objective.u_coef is not None else None,
            "u_bounds": list(objective.u_bounds) if objective.u_bounds is not None else None,
            "acceptance_floor": float(objective.acceptance_floor)
            if objective.acceptance_floor is not None
            else None,
            "acceptance_penalty_weight": float(objective.acceptance_penalty_weight)
            if objective.acceptance_penalty_weight is not None
            else None,
            "acceptance_penalty_temperature": float(objective.acceptance_penalty_temperature),
            "lagrangian_lambda": float(objective.lagrangian_lambda)
            if objective.lagrangian_lambda is not None
            else None,
            "policy_preprocessor": (
                objective.policy_preprocessor.to_dict()
                if objective.policy_preprocessor is not None
                else None
            ),
            "policy_feature_cols": list(objective.policy_feature_cols)
            if objective.policy_feature_cols is not None
            else None,
        }
    return {"type": type(objective).__name__}


def _noise_to_dict(noise: ObjectiveNoise) -> dict[str, Any]:
    """Serialize objective noise to dictionary."""
    if isinstance(noise, NoNoise):
        return {"type": "NoNoise"}
    if isinstance(noise, HomoskedasticGaussianNoise):
        return {
            "type": "HomoskedasticGaussianNoise",
            "std": float(noise.std),
            "seed": int(noise.seed) if noise.seed is not None else None,
        }
    if isinstance(noise, HeteroskedasticGaussianNoise):
        return {
            "type": "HeteroskedasticGaussianNoise",
            "base_std": float(noise.base_std),
            "growth": float(noise.growth),
            "u_center": float(noise.u_center),
            "seed": int(noise.seed) if noise.seed is not None else None,
        }
    return {"type": type(noise).__name__}


def _sigma_provider_to_dict(provider: SigmaProvider | None) -> dict[str, Any] | None:
    """Serialize a support-scale provider to dictionary."""
    if provider is None:
        return None
    if isinstance(provider, HomoskedasticNoiseScaleProvider):
        return {"type": "HomoskedasticNoiseScaleProvider", "std": float(provider.std)}
    if isinstance(provider, HeteroskedasticNoiseScaleProvider):
        return {
            "type": "HeteroskedasticNoiseScaleProvider",
            "base_std": float(provider.base_std),
            "growth": float(provider.growth),
            "u_center": float(provider.u_center),
        }
    if isinstance(provider, CallableSigmaProvider):
        return {"type": "CallableSigmaProvider"}
    return {"type": type(provider).__name__}


def _action_bias_to_dict(bias: ActionBias) -> dict[str, Any]:
    """Serialize an action-bias term to dictionary."""
    if isinstance(bias, LinearActionBias):
        return {
            "type": "LinearActionBias",
            "lambda_bias": float(bias.lambda_bias),
        }
    if isinstance(bias, UpperSupportHingeBias):
        return {
            "type": "UpperSupportHingeBias",
            "lambda_bias": float(bias.lambda_bias),
            "support_center": float(bias.support_center),
            "support_radius": float(bias.support_radius),
            "support_upper": float(bias.support_upper),
            "smooth_tau": float(bias.smooth_tau) if bias.smooth_tau is not None else None,
        }
    return {"type": type(bias).__name__, "lambda_bias": float(bias.lambda_bias)}


def _policy_to_dict(policy: object) -> dict[str, Any]:
    """Serialize policy to dictionary."""
    feature_map = getattr(policy, "feature_map", None)
    feature_map_dict = None
    if feature_map is not None:
        feature_map_dict = {
            "type": type(feature_map).__name__,
            "kind": getattr(feature_map, "kind", None),
            "feature_dim": getattr(feature_map, "feature_dim", None),
            "include_interactions": getattr(feature_map, "include_interactions", None),
            "name": getattr(feature_map, "name", None),
        }
    if isinstance(policy, ConstantPolicy):
        return {"type": "ConstantPolicy"}
    if isinstance(policy, LinearPolicy):
        return {"type": "LinearPolicy", "feature_map": feature_map_dict}
    if isinstance(policy, SoftmaxPolicy):
        return {
            "type": "SoftmaxPolicy",
            "feature_map": feature_map_dict,
            "action_low": float(policy.action_low),
            "action_high": float(policy.action_high),
        }
    if isinstance(policy, MLPPolicy):
        return {
            "type": "MLPPolicy",
            "feature_map": feature_map_dict,
            "hidden": int(policy.hidden),
        }
    return {"type": type(policy).__name__}


def _correctness_to_dict(correctness: CorrectnessSpec) -> dict[str, Any]:
    """Serialize correctness spec to dictionary."""
    return {
        "gradient_source": correctness.gradient_source,
        "numdiff_method": correctness.numdiff_method,
        "numdiff_step": float(correctness.numdiff_step),
        "numdiff_aggregate": correctness.numdiff_aggregate,
        "numdiff_bounds": list(correctness.numdiff_bounds)
        if correctness.numdiff_bounds is not None
        else None,
    }


def _as_list(values: object) -> list[float]:
    """Convert array-like to list of floats."""
    arr = np.asarray(values, dtype=float)
    return [float(val) for val in arr.tolist()]


# --- Factory helpers for preset configs ---


def make_fixed_regression_objective(
    *,
    policy: Policy,
    beta_1: np.ndarray | Sequence[float],
    beta_2: float,
    beta_3: np.ndarray | Sequence[float],
    beta_4: float,
) -> FixedRegressionObjective:
    """Create a FixedRegressionObjective with the given parameters."""
    return FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=np.asarray(beta_1, dtype=float),
        beta_2=float(beta_2),
        beta_3=np.asarray(beta_3, dtype=float),
        beta_4=float(beta_4),
    )


def make_planted_logistic_objective(
    *,
    policy: Policy,
    alpha: float,
    beta: np.ndarray | Sequence[float],
    bias: float,
    u_star: float,
) -> PlantedLogisticObjective:
    """Create a PlantedLogisticObjective with the given parameters."""
    return PlantedLogisticObjective.from_parameters(
        policy=policy,
        alpha=float(alpha),
        beta=np.asarray(beta, dtype=float),
        bias=float(bias),
        u_star=float(u_star),
    )


def make_softmax_policy(
    action_low: float = -0.5,
    action_high: float = 0.5,
) -> SoftmaxPolicy:
    """Create a bounded sigmoid SoftmaxPolicy instance."""
    return SoftmaxPolicy(action_low=action_low, action_high=action_high)


def make_model_based_objective(
    *,
    policy: Policy,
    acceptance_model: object,
    loss_model: object,
    acceptance_state_cols: tuple[str, ...],
    loss_cols: tuple[str, ...],
    premium_col: int | str = 9,
    loss_source: Literal["predicted", "observed"] = "predicted",
    observed_loss_col: str = "Y_G_Loss",
    u_coef: float | None = None,
    u_bounds: tuple[float, float] | None = None,
    acceptance_floor: float | None = None,
    acceptance_penalty_weight: float | None = None,
    acceptance_penalty_temperature: float = 0.01,
    lagrangian_lambda: float | None = None,
    policy_preprocessor: PolicyFeaturePreprocessor | None = None,
    policy_feature_cols: tuple[str, ...] | None = None,
) -> ModelBasedObjective:
    """Create a ModelBasedObjective wrapping trained sklearn/XGBoost models."""
    return ModelBasedObjective(
        policy=policy,
        acceptance_model=acceptance_model,
        loss_model=loss_model,
        acceptance_state_cols=acceptance_state_cols,
        loss_cols=loss_cols,
        premium_col=premium_col,
        loss_source=loss_source,
        observed_loss_col=observed_loss_col,
        u_coef=u_coef,
        u_bounds=u_bounds,
        acceptance_floor=acceptance_floor,
        acceptance_penalty_weight=acceptance_penalty_weight,
        acceptance_penalty_temperature=acceptance_penalty_temperature,
        lagrangian_lambda=lagrangian_lambda,
        policy_preprocessor=policy_preprocessor,
        policy_feature_cols=policy_feature_cols,
    )


def canonical_training_block(
    *,
    n_samples: int,
    train_fraction: float = 1.0,
    test_fraction: float = 0.0,
    step_rule: str,
    compute_backend: Literal["numpy", "jax"] = "numpy",
    t_steps: int,
    step_size: float,
    sigma: float,
    n_grad_samples: int,
    enabled_estimators: tuple[str, ...],
    constant_u_baselines: tuple[float, ...] = (),
    perturbation_space: Literal["theta", "u"],
    batch_size: int | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    initial_constr_penalty: float | None = None,
    acceptance_floor: float | None = None,
    acceptance_penalty_weight: float | None = None,
    acceptance_penalty_temperature: float = 0.01,
    lagrangian_lambda: float | None = None,
    proximal_weight: float | None = None,
    u_reference: np.ndarray | Sequence[float] | None = None,
    u_reference_source: UReferenceSource | None = None,
    u_reference_value: float | None = None,
    support_weight: float | None = None,
    sigma_provider: SigmaProvider | None = None,
) -> dict[str, Any]:
    """Create a canonical training configuration block."""
    return {
        "n_samples": int(n_samples),
        "train_fraction": float(train_fraction),
        "test_fraction": float(test_fraction),
        "step_rule": step_rule,
        "compute_backend": compute_backend,
        "t_steps": int(t_steps),
        "step_size": float(step_size),
        "sigma": float(sigma),
        "n_grad_samples": int(n_grad_samples),
        "enabled_estimators": tuple(enabled_estimators),
        "constant_u_baselines": tuple(float(u) for u in constant_u_baselines),
        "perturbation_space": perturbation_space,
        "batch_size": int(batch_size) if batch_size is not None else None,
        "grad_norm_tol": float(grad_norm_tol) if grad_norm_tol is not None else None,
        "ftol": float(ftol) if ftol is not None else None,
        "initial_constr_penalty": float(initial_constr_penalty)
        if initial_constr_penalty is not None
        else None,
        "acceptance_floor": float(acceptance_floor) if acceptance_floor is not None else None,
        "acceptance_penalty_weight": float(acceptance_penalty_weight)
        if acceptance_penalty_weight is not None
        else None,
        "acceptance_penalty_temperature": float(acceptance_penalty_temperature),
        "lagrangian_lambda": float(lagrangian_lambda) if lagrangian_lambda is not None else None,
        "proximal_weight": float(proximal_weight) if proximal_weight is not None else None,
        "u_reference": np.asarray(u_reference, dtype=float) if u_reference is not None else None,
        "u_reference_source": u_reference_source,
        "u_reference_value": float(u_reference_value) if u_reference_value is not None else None,
        "support_weight": float(support_weight) if support_weight is not None else None,
        "sigma_provider": sigma_provider,
    }


def canonical_runtime_block(
    *,
    plot: bool,
    verbose: bool,
    wandb_enabled: bool,
    wandb_project: str | None = None,
    wandb_entity: str | None = "generali-pricing",
    wandb_group: str | None = None,
    wandb_job_type: str = "experiment",
    wandb_tags: tuple[str, ...] = (),
    wandb_mode: Literal["online", "offline", "disabled"] = "online",
    wandb_log_plots: bool = True,
    wandb_estimator_allowlist: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Create a canonical runtime configuration block."""
    return {
        "plot": bool(plot),
        "verbose": bool(verbose),
        "wandb_enabled": bool(wandb_enabled),
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_group": wandb_group,
        "wandb_job_type": wandb_job_type,
        "wandb_tags": tuple(wandb_tags),
        "wandb_mode": wandb_mode,
        "wandb_log_plots": bool(wandb_log_plots),
        "wandb_estimator_allowlist": tuple(wandb_estimator_allowlist)
        if wandb_estimator_allowlist is not None
        else None,
    }


def build_experiment_config(
    *,
    seed: int,
    seed_setup: SeedSetup | Mapping[str, int | None] | None = None,
    state_dim: int,
    objective: Objective,
    theta0: np.ndarray | None = None,
    training: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
    correctness: CorrectnessSpec | None = None,
    x_fixed: Any | None = None,
    x_fixed_row_indices: np.ndarray | None = None,
) -> ExperimentConfig:
    """Build an ExperimentConfig from component blocks.

    If ``x_fixed`` is provided, the runner uses it directly as the state array
    instead of sampling from N(0, I).
    """
    payload: dict[str, Any] = {
        "seed": int(seed),
        "seed_setup": seed_setup,
        "state_dim": int(state_dim),
        "objective": objective,
        "theta0": np.asarray(theta0, dtype=float) if theta0 is not None else None,
    }
    payload.update(dict(training))
    if runtime is not None:
        payload.update(dict(runtime))
    if correctness is not None:
        payload["correctness"] = correctness
    if x_fixed is not None:
        payload["x_fixed"] = x_fixed.reset_index(drop=True).copy() if hasattr(x_fixed, "iloc") else np.asarray(x_fixed, dtype=float)
    if x_fixed_row_indices is not None:
        payload["x_fixed_row_indices"] = np.asarray(x_fixed_row_indices, dtype=int)
    return ExperimentConfig(**payload)
