"""Diagnose Stein-difference divergence between NumPy and JAX GLM backends."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from data.loader import load_x_frame
from experiments.configs import get_config
from experiments.seeding import optimizer_rngs, resolve_seed_setup
from objective.objectives import prepare_jax_glm_objective
from objective.utils import _policy_value, _theta_grad_from_u_grad
from optimization.base import Optimization
from optimization.gradients.methods import SteinDifferenceGradient, _action_objective_values_many


DEFAULT_OUTPUT_ROOT = Path("outputs") / "backend-divergence"


@dataclass(frozen=True)
class BackendPair:
    """Reconstructed CPU/JAX objectives over the exact saved train rows."""

    config: Any
    cpu_objective: Any
    cpu_x_train: Any
    jax_objective: Any
    jax_x_train: np.ndarray
    theta0: np.ndarray
    theta_numpy_final: np.ndarray
    theta_jax_final: np.ndarray
    selected_row_indices: np.ndarray
    train_row_indices: np.ndarray


@dataclass(frozen=True)
class SteinProbe:
    """Stein-difference quantities for one backend at fixed theta/samples."""

    grad: np.ndarray
    grad_u: np.ndarray
    u: np.ndarray
    values_plus: np.ndarray
    values_minus: np.ndarray


@dataclass
class EventRecorder:
    """Mutable collector for optimizer callback diagnostics."""

    backend: str
    events: list[dict[str, Any]] = field(default_factory=list)
    gradient_source: str = "record"
    gradient_call_index: int = 0
    value_call_index: int = 0
    callback_call_index: int = 0

    @contextmanager
    def source(self, name: str) -> Iterator[None]:
        old = self.gradient_source
        self.gradient_source = name
        try:
            yield
        finally:
            self.gradient_source = old

    def append(self, event: Mapping[str, Any]) -> None:
        self.events.append(dict(event))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Stein-difference CPU/JAX backend behavior using the saved run's "
            "existing optimizer_seed."
        )
    )
    parser.add_argument("--numpy-summary", type=Path, required=True)
    parser.add_argument("--jax-summary", type=Path, required=True)
    parser.add_argument("--estimator", default="stein_difference")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--probe-blocks",
        type=int,
        default=4,
        help="Number of consecutive optimizer RNG blocks to compare in frozen probes.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Optional trust-constr maxiter for instrumented reruns. Defaults to the saved config t_steps.",
    )
    parser.add_argument(
        "--skip-optimizer-trace",
        action="store_true",
        help="Only run fixed-theta frozen perturbation probes.",
    )
    parser.add_argument("--theta-tol", type=float, default=1e-8)
    parser.add_argument("--grad-tol", type=float, default=1e-8)
    parser.add_argument("--value-tol", type=float, default=1e-8)
    return parser.parse_args(argv)


def load_summary(path: Path) -> dict[str, Any]:
    """Load a run summary JSON payload."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "config" not in payload or "estimators" not in payload:
        raise ValueError("summary must contain config and estimators sections.")
    return payload


def reconstruct_backend_pair(
    numpy_summary_path: Path,
    jax_summary_path: Path,
    estimator: str,
) -> BackendPair:
    """Rebuild CPU/JAX objectives from saved summaries and policy artifacts."""
    numpy_payload = load_summary(numpy_summary_path)
    jax_payload = load_summary(jax_summary_path)
    _validate_matching_configs(numpy_payload["config"], jax_payload["config"])
    if estimator not in numpy_payload["estimators"] or estimator not in jax_payload["estimators"]:
        raise ValueError(f"Estimator '{estimator}' must be present in both summaries.")

    artifact_path = _policy_artifact_path(numpy_summary_path, numpy_payload, estimator)
    with np.load(artifact_path.parent / "arrays.npz") as arrays:
        selected_row_indices = np.asarray(arrays["selected_row_indices"], dtype=int)
        train_row_indices = np.asarray(arrays["train_row_indices"], dtype=int)

    cfg = _config_from_summary(numpy_payload, selected_row_indices, estimator)
    x_train = load_x_frame("glm", row_indices=train_row_indices)
    cpu_objective = _objective_with_acceptance_controls(cfg)
    jax_objective, jax_batch = prepare_jax_glm_objective(
        cpu_objective,
        x_train,
        row_indices=train_row_indices,
    )
    theta0 = np.asarray(cfg.theta0, dtype=float)
    jax_objective.warmup(theta0)
    return BackendPair(
        config=cfg,
        cpu_objective=cpu_objective,
        cpu_x_train=x_train,
        jax_objective=jax_objective,
        jax_x_train=jax_batch.x_array,
        theta0=theta0,
        theta_numpy_final=np.asarray(numpy_payload["estimators"][estimator]["theta"], dtype=float),
        theta_jax_final=np.asarray(jax_payload["estimators"][estimator]["theta"], dtype=float),
        selected_row_indices=selected_row_indices,
        train_row_indices=train_row_indices,
    )


def _objective_with_acceptance_controls(config: Any) -> Any:
    objective = config.objective
    if config.acceptance_floor is None and config.lagrangian_lambda is None:
        return objective
    if not hasattr(objective, "acceptance_floor"):
        return objective
    return replace(
        objective,
        acceptance_floor=float(config.acceptance_floor),
        acceptance_penalty_weight=(
            float(config.acceptance_penalty_weight)
            if config.acceptance_penalty_weight is not None
            else None
        ),
        acceptance_penalty_temperature=float(config.acceptance_penalty_temperature),
        lagrangian_lambda=(
            float(config.lagrangian_lambda)
            if config.lagrangian_lambda is not None
            else None
        ),
    )


def run_fixed_probes(pair: BackendPair, estimator: str, probe_blocks: int) -> list[dict[str, Any]]:
    """Compare CPU/JAX Stein gradients at fixed theta with shared perturbations."""
    if probe_blocks <= 0:
        raise ValueError("probe_blocks must be positive.")
    seeds = resolve_seed_setup(pair.config.seed_setup, pair.config.seed)
    _, gradient_rng = optimizer_rngs(seeds, estimator)
    w_blocks = [
        gradient_rng.normal(0.0, 1.0, size=pair.config.n_grad_samples).astype(float)
        for _ in range(probe_blocks)
    ]
    theta_cases = [
        ("theta0", pair.theta0),
        ("numpy_final", pair.theta_numpy_final),
        ("jax_final", pair.theta_jax_final),
    ]
    rows: list[dict[str, Any]] = []
    for theta_name, theta in theta_cases:
        for block_index, w_samples in enumerate(w_blocks):
            cpu_probe = stein_gradient_with_samples(
                pair.cpu_objective,
                pair.cpu_x_train,
                theta,
                w_samples,
                pair.config.sigma,
            )
            jax_probe = stein_gradient_with_samples(
                pair.jax_objective,
                pair.jax_x_train,
                theta,
                w_samples,
                pair.config.sigma,
            )
            rows.append(
                {
                    "theta_name": theta_name,
                    "block_index": block_index,
                    "w_hash": _array_hash(w_samples),
                    **probe_difference_metrics(cpu_probe, jax_probe),
                }
            )
    return rows


def stein_gradient_with_samples(
    objective: Any,
    x_batch: Any,
    theta: np.ndarray,
    w_samples: np.ndarray,
    sigma: float,
) -> SteinProbe:
    """Compute the u-space Stein-difference gradient with fixed samples."""
    theta_arr = np.asarray(theta, dtype=float)
    w_arr = np.asarray(w_samples, dtype=float).reshape(-1)
    if w_arr.size == 0:
        raise ValueError("w_samples must not be empty.")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    u_arr = _policy_value(objective, theta_arr, x_batch).reshape(-1)
    values_plus = _action_objective_values_many(
        objective,
        x_batch,
        u_arr[None, :] + float(sigma) * w_arr[:, None],
    )
    values_minus = _action_objective_values_many(
        objective,
        x_batch,
        u_arr[None, :] - float(sigma) * w_arr[:, None],
    )
    grad_u = np.mean(((values_plus - values_minus) / (2.0 * float(sigma))) * w_arr[:, None], axis=0)
    grad = _theta_grad_from_u_grad(objective, theta_arr, x_batch, grad_u)
    return SteinProbe(
        grad=np.asarray(grad, dtype=float),
        grad_u=np.asarray(grad_u, dtype=float),
        u=np.asarray(u_arr, dtype=float),
        values_plus=np.asarray(values_plus, dtype=float),
        values_minus=np.asarray(values_minus, dtype=float),
    )


def probe_difference_metrics(cpu_probe: SteinProbe, jax_probe: SteinProbe) -> dict[str, float]:
    """Return scalar parity metrics between two fixed-sample Stein probes."""
    return {
        "grad_linf_diff": _linf(cpu_probe.grad, jax_probe.grad),
        "grad_l2_diff": _l2(cpu_probe.grad, jax_probe.grad),
        "grad_cosine": _cosine(cpu_probe.grad, jax_probe.grad),
        "grad_u_linf_diff": _linf(cpu_probe.grad_u, jax_probe.grad_u),
        "grad_u_l2_diff": _l2(cpu_probe.grad_u, jax_probe.grad_u),
        "u_linf_diff": _linf(cpu_probe.u, jax_probe.u),
        "u_l2_diff": _l2(cpu_probe.u, jax_probe.u),
        "values_plus_linf_diff": _linf(cpu_probe.values_plus, jax_probe.values_plus),
        "values_minus_linf_diff": _linf(cpu_probe.values_minus, jax_probe.values_minus),
        "values_plus_mean_diff": float(np.mean(cpu_probe.values_plus) - np.mean(jax_probe.values_plus)),
        "values_minus_mean_diff": float(np.mean(cpu_probe.values_minus) - np.mean(jax_probe.values_minus)),
    }


def run_instrumented_optimizer(
    pair: BackendPair,
    estimator: str,
    backend: str,
    maxiter: int,
) -> tuple[np.ndarray, Any, EventRecorder]:
    """Run one backend with callback, RNG, and cross-backend gradient logging."""
    seeds = resolve_seed_setup(pair.config.seed_setup, pair.config.seed)
    batch_rng, gradient_rng = optimizer_rngs(seeds, estimator)
    if backend == "numpy":
        objective = pair.cpu_objective
        x_samples = pair.cpu_x_train
        peer_objective = pair.jax_objective
        peer_x = pair.jax_x_train
    elif backend == "jax":
        objective = pair.jax_objective
        x_samples = pair.jax_x_train
        peer_objective = pair.cpu_objective
        peer_x = pair.cpu_x_train
    else:
        raise ValueError("backend must be 'numpy' or 'jax'.")

    recorder = EventRecorder(backend=backend)
    gradient = RecordingSteinDifferenceGradient(
        recorder=recorder,
        peer_objective=peer_objective,
        peer_x=peer_x,
    )
    tracing_minimize = TracingMinimize(
        recorder=recorder,
        peer_objective=peer_objective,
        peer_x=peer_x,
    )
    optimizer = Optimization(
        objective,
        x_samples,
        gradient,
        algorithm=pair.config.step_rule,
        t_steps=int(maxiter),
        n_grad_samples=pair.config.n_grad_samples,
        sigma=pair.config.sigma,
        perturbation_space=pair.config.perturbation_space,
        step_size=pair.config.step_size,
        batch_size=pair.config.batch_size,
        grad_norm_tol=pair.config.grad_norm_tol,
        ftol=pair.config.ftol,
        initial_constr_penalty=pair.config.initial_constr_penalty,
        batch_rng=batch_rng,
        gradient_rng=gradient_rng,
        minimize_fn=tracing_minimize,
    )
    theta_final, trace = optimizer.solve(pair.theta0)
    return theta_final, trace, recorder


class RecordingSteinDifferenceGradient(SteinDifferenceGradient):
    """Stein-difference gradient that records fixed samples and peer parity."""

    def __init__(self, *, recorder: EventRecorder, peer_objective: Any, peer_x: Any) -> None:
        self.recorder = recorder
        self.peer_objective = peer_objective
        self.peer_x = peer_x

    def _u_grad(self, optimizer: Optimization, theta: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if indices.size != optimizer.n_total:
            raise ValueError("backend divergence diagnostics require batch_size=None/full-batch indices.")
        theta_arr = np.asarray(theta, dtype=float)
        state_before = _rng_state_hash(optimizer.rng)
        w_samples = optimizer.rng.normal(0.0, 1.0, size=optimizer.n_grad_samples).astype(float)
        state_after = _rng_state_hash(optimizer.rng)
        probe = stein_gradient_with_samples(
            optimizer.objective,
            optimizer.x_array,
            theta_arr,
            w_samples,
            optimizer.sigma,
        )
        peer_probe = stein_gradient_with_samples(
            self.peer_objective,
            self.peer_x,
            theta_arr,
            w_samples,
            optimizer.sigma,
        )
        event = {
            "event": "gradient",
            "backend": self.recorder.backend,
            "source": self.recorder.gradient_source,
            "gradient_call_index": self.recorder.gradient_call_index,
            "theta": theta_arr.tolist(),
            "theta_hash": _array_hash(theta_arr),
            "theta_l2": _norm(theta_arr),
            "rng_state_before": state_before,
            "rng_state_after": state_after,
            "w_hash": _array_hash(w_samples),
            "w_mean": float(np.mean(w_samples)),
            "w_std": float(np.std(w_samples)),
            "grad_l2": _norm(probe.grad),
            "grad_linf": float(np.max(np.abs(probe.grad))),
            "peer_grad_linf_diff": _linf(probe.grad, peer_probe.grad),
            "peer_grad_l2_diff": _l2(probe.grad, peer_probe.grad),
            "peer_grad_cosine": _cosine(probe.grad, peer_probe.grad),
            "peer_u_linf_diff": _linf(probe.u, peer_probe.u),
            "peer_values_plus_linf_diff": _linf(probe.values_plus, peer_probe.values_plus),
            "peer_values_minus_linf_diff": _linf(probe.values_minus, peer_probe.values_minus),
        }
        self.recorder.append(event)
        self.recorder.gradient_call_index += 1
        return probe.grad


class TracingMinimize:
    """SciPy minimize wrapper that logs value calls and callback points."""

    def __init__(self, *, recorder: EventRecorder, peer_objective: Any, peer_x: Any) -> None:
        self.recorder = recorder
        self.peer_objective = peer_objective
        self.peer_x = peer_x

    def __call__(self, fun: Any, **kwargs: Any) -> Any:
        jac = kwargs.get("jac")
        callback = kwargs.get("callback")

        def wrapped_fun(theta: np.ndarray) -> float:
            theta_arr = np.asarray(theta, dtype=float)
            value = float(fun(theta_arr))
            peer_value = float(self.peer_objective.value(theta_arr, self.peer_x))
            self.recorder.append(
                {
                    "event": "value",
                    "backend": self.recorder.backend,
                    "value_call_index": self.recorder.value_call_index,
                    "theta": theta_arr.tolist(),
                    "theta_hash": _array_hash(theta_arr),
                    "theta_l2": _norm(theta_arr),
                    "value": value,
                    "peer_value": peer_value,
                    "peer_value_diff": value - peer_value,
                }
            )
            self.recorder.value_call_index += 1
            return value

        def wrapped_jac(theta: np.ndarray) -> np.ndarray:
            if jac is None:
                raise ValueError("Stein diagnostics require SciPy jac callback.")
            with self.recorder.source("jac"):
                return np.asarray(jac(theta), dtype=float)

        def wrapped_callback(theta: np.ndarray, state: Any | None = None) -> bool:
            theta_arr = np.asarray(theta, dtype=float)
            self.recorder.append(
                {
                    "event": "callback",
                    "backend": self.recorder.backend,
                    "callback_call_index": self.recorder.callback_call_index,
                    "theta": theta_arr.tolist(),
                    "theta_hash": _array_hash(theta_arr),
                    "theta_l2": _norm(theta_arr),
                    "state_niter": _state_attr(state, "niter"),
                    "state_fun": _state_attr(state, "fun"),
                    "state_optimality": _state_attr(state, "optimality"),
                    "state_constr_violation": _state_attr(state, "constr_violation"),
                    "state_tr_radius": _state_attr(state, "tr_radius"),
                    "state_constr_penalty": _state_attr(state, "constr_penalty"),
                }
            )
            self.recorder.callback_call_index += 1
            if callback is None:
                return False
            with self.recorder.source("callback_record"):
                return bool(callback(theta_arr, state))

        kwargs["jac"] = wrapped_jac
        if callback is not None:
            kwargs["callback"] = wrapped_callback
        return minimize(wrapped_fun, **kwargs)


def compare_event_traces(
    numpy_events: Sequence[Mapping[str, Any]],
    jax_events: Sequence[Mapping[str, Any]],
    *,
    theta_tol: float,
    grad_tol: float,
    value_tol: float,
) -> dict[str, Any]:
    """Find the first material CPU/JAX trace differences by event type."""
    return {
        "gradient": _compare_events_by_type(
            numpy_events,
            jax_events,
            event_type="gradient",
            theta_tol=theta_tol,
            metric_key="peer_grad_linf_diff",
            metric_tol=grad_tol,
            extra_keys=("source", "w_hash", "rng_state_before"),
        ),
        "value": _compare_events_by_type(
            numpy_events,
            jax_events,
            event_type="value",
            theta_tol=theta_tol,
            metric_key="peer_value_diff",
            metric_tol=value_tol,
            extra_keys=(),
        ),
        "callback": _compare_events_by_type(
            numpy_events,
            jax_events,
            event_type="callback",
            theta_tol=theta_tol,
            metric_key="state_fun",
            metric_tol=value_tol,
            extra_keys=("state_niter",),
        ),
    }


def _compare_events_by_type(
    numpy_events: Sequence[Mapping[str, Any]],
    jax_events: Sequence[Mapping[str, Any]],
    *,
    event_type: str,
    theta_tol: float,
    metric_key: str,
    metric_tol: float,
    extra_keys: Sequence[str],
) -> dict[str, Any]:
    np_events = [event for event in numpy_events if event.get("event") == event_type]
    jx_events = [event for event in jax_events if event.get("event") == event_type]
    limit = min(len(np_events), len(jx_events))
    result: dict[str, Any] = {
        "numpy_count": len(np_events),
        "jax_count": len(jx_events),
        "first_difference_index": None,
        "first_difference_reason": None,
    }
    for idx in range(limit):
        np_event = np_events[idx]
        jx_event = jx_events[idx]
        theta_diff = _event_theta_linf(np_event, jx_event)
        theta_mismatch = theta_diff > theta_tol if np.isfinite(theta_diff) else np_event.get("theta_hash") != jx_event.get("theta_hash")
        if theta_mismatch:
            result.update(
                {
                    "first_difference_index": idx,
                    "first_difference_reason": "theta",
                    "theta_linf_diff": theta_diff,
                    "numpy_event": dict(np_event),
                    "jax_event": dict(jx_event),
                }
            )
            return result
        for key in extra_keys:
            if np_event.get(key) != jx_event.get(key):
                result.update(
                    {
                        "first_difference_index": idx,
                        "first_difference_reason": key,
                        "numpy_event": dict(np_event),
                        "jax_event": dict(jx_event),
                    }
                )
                return result
        if event_type == "gradient":
            metric = max(abs(float(np_event.get(metric_key, 0.0))), abs(float(jx_event.get(metric_key, 0.0))))
        else:
            metric = abs(float(np_event.get(metric_key, 0.0)) - float(jx_event.get(metric_key, 0.0)))
        if metric > metric_tol:
            result.update(
                {
                    "first_difference_index": idx,
                    "first_difference_reason": metric_key,
                    "metric": metric,
                    "numpy_event": dict(np_event),
                    "jax_event": dict(jx_event),
                }
            )
            return result
    if len(np_events) != len(jx_events):
        result.update(
            {
                "first_difference_index": limit,
                "first_difference_reason": "event_count",
            }
        )
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write mappings to CSV, preserving all observed keys."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    pair = reconstruct_backend_pair(args.numpy_summary, args.jax_summary, args.estimator)
    maxiter = int(args.maxiter if args.maxiter is not None else pair.config.t_steps)
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{args.estimator}_{args.numpy_summary.parent.name}_vs_{args.jax_summary.parent.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_rows = run_fixed_probes(pair, args.estimator, args.probe_blocks)
    write_csv(output_dir / "fixed_stein_probes.csv", fixed_rows)
    summary: dict[str, Any] = {
        "estimator": args.estimator,
        "n_train_rows": int(pair.jax_x_train.shape[0]),
        "n_grad_samples": int(pair.config.n_grad_samples),
        "sigma": float(pair.config.sigma),
        "fixed_probe_max": _max_metrics(fixed_rows),
    }

    if not args.skip_optimizer_trace:
        theta_np, trace_np, recorder_np = run_instrumented_optimizer(pair, args.estimator, "numpy", maxiter)
        theta_jx, trace_jx, recorder_jx = run_instrumented_optimizer(pair, args.estimator, "jax", maxiter)
        write_csv(output_dir / "numpy_optimizer_events.csv", recorder_np.events)
        write_csv(output_dir / "jax_optimizer_events.csv", recorder_jx.events)
        trace_diff = compare_event_traces(
            recorder_np.events,
            recorder_jx.events,
            theta_tol=float(args.theta_tol),
            grad_tol=float(args.grad_tol),
            value_tol=float(args.value_tol),
        )
        summary["optimizer_trace"] = {
            "maxiter": maxiter,
            "numpy_theta_final_head": theta_np[:5].tolist(),
            "jax_theta_final_head": theta_jx[:5].tolist(),
            "theta_final_linf_diff": _linf(theta_np, theta_jx),
            "numpy_steps": len(trace_np.steps),
            "jax_steps": len(trace_jx.steps),
            "differences": trace_diff,
        }

    with (output_dir / "diagnosis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    _write_markdown_summary(output_dir / "summary.md", summary)
    print(f"Wrote Stein backend divergence diagnostics to {output_dir}")


def _config_from_summary(payload: Mapping[str, Any], row_indices: np.ndarray, estimator: str) -> Any:
    config = payload["config"]
    objective = config["objective"]
    policy = objective["policy"]
    policy_type = policy["type"]
    if policy_type != "SoftmaxPolicy":
        raise ValueError(f"This diagnostic currently supports SoftmaxPolicy, got {policy_type}.")
    if objective["type"] != "ModelBasedObjective" or config["state_dim"] != 19:
        raise ValueError("This diagnostic currently supports GLM ModelBasedObjective summaries.")
    constraint_mode = "trust_constr" if config.get("step_rule") == "trust-constr" else "none"
    preprocessing = "no_pca" if objective.get("policy_preprocessor") is not None else "artifact"
    return get_config(
        "real_data_glm_base",
        overrides={
            "policy_kind": "softmax",
            "softmax_action_bounds": (float(policy["action_low"]), float(policy["action_high"])),
            "feature_order": "linear",
            "policy_preprocessing": preprocessing,
            "constraint_mode": constraint_mode,
            "row_indices": np.asarray(row_indices, dtype=int),
            "train_fraction": float(config.get("train_fraction", 1.0)),
            "test_fraction": float(config.get("test_fraction", 0.0)),
            "theta0": np.asarray(config["theta0"], dtype=float),
            "seed": int(config["seed"]),
            "t_steps": int(config["t_steps"]),
            "step_rule": str(config["step_rule"]),
            "compute_backend": "numpy",
            "sigma": float(config["sigma"]),
            "n_grad_samples": int(config["n_grad_samples"]),
            "enabled_estimators": (estimator,),
            "perturbation_space": str(config["perturbation_space"]),
            "batch_size": config.get("batch_size"),
            "grad_norm_tol": config.get("grad_norm_tol"),
            "ftol": config.get("ftol"),
            "initial_constr_penalty": config.get("initial_constr_penalty"),
            "acceptance_floor": config.get("acceptance_floor"),
            "acceptance_penalty_weight": config.get("acceptance_penalty_weight"),
            "acceptance_penalty_temperature": float(config.get("acceptance_penalty_temperature", 0.01)),
            "lagrangian_lambda": config.get("lagrangian_lambda"),
            "u_coef": objective.get("u_coef"),
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        },
    )


def _policy_artifact_path(summary_path: Path, payload: Mapping[str, Any], estimator: str) -> Path:
    artifacts = payload.get("policy_artifacts", {})
    if estimator not in artifacts:
        raise ValueError(f"Summary does not contain a policy artifact for estimator '{estimator}'.")
    return summary_path.parent / str(artifacts[estimator])


def _validate_matching_configs(numpy_config: Mapping[str, Any], jax_config: Mapping[str, Any]) -> None:
    diffs = []
    for key in sorted(set(numpy_config) | set(jax_config)):
        if key == "compute_backend":
            continue
        if numpy_config.get(key) != jax_config.get(key):
            diffs.append(key)
    if diffs:
        raise ValueError(f"Summaries differ in config fields beyond compute_backend: {diffs[:10]}")


def _max_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if key.endswith("_diff") or key == "grad_cosine":
                try:
                    numeric = abs(float(value))
                except (TypeError, ValueError):
                    continue
                result[key] = max(result.get(key, 0.0), numeric)
    return result


def _event_theta_linf(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if "theta" not in left or "theta" not in right:
        return float("nan")
    return _linf(np.asarray(left["theta"], dtype=float), np.asarray(right["theta"], dtype=float))


def _write_markdown_summary(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Stein Backend Divergence Diagnostic",
        "",
        f"- Estimator: `{summary['estimator']}`",
        f"- Train rows: `{summary['n_train_rows']}`",
        f"- n_grad_samples: `{summary['n_grad_samples']}`",
        f"- sigma: `{summary['sigma']}`",
        "",
        "## Fixed Probe Maxima",
    ]
    for key, value in sorted(summary.get("fixed_probe_max", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    trace = summary.get("optimizer_trace")
    if isinstance(trace, Mapping):
        lines.extend(
            [
                "",
                "## Optimizer Trace",
                f"- maxiter: `{trace['maxiter']}`",
                f"- theta_final_linf_diff: `{trace['theta_final_linf_diff']}`",
                f"- numpy_steps: `{trace['numpy_steps']}`",
                f"- jax_steps: `{trace['jax_steps']}`",
            ]
        )
        differences = trace.get("differences", {})
        if isinstance(differences, Mapping):
            for event_type, payload in sorted(differences.items()):
                if isinstance(payload, Mapping):
                    lines.append(
                        f"- `{event_type}` first difference: index `{payload.get('first_difference_index')}`, "
                        f"reason `{payload.get('first_difference_reason')}`"
                    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _state_attr(state: Any | None, name: str) -> float | int | None:
    if state is None or not hasattr(state, name):
        return None
    value = getattr(state, name)
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size != 1:
        return None
    scalar = arr.reshape(-1)[0]
    if np.issubdtype(arr.dtype, np.integer):
        return int(scalar)
    return float(scalar)


def _rng_state_hash(rng: np.random.Generator) -> str:
    return hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array, dtype=float))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()[:16]


def _linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.linalg.norm(diff))


def _norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=float).reshape(-1)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a_arr, b_arr) / denom)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


if __name__ == "__main__":
    main()
