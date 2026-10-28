"""Residual-basis study adapter for the existing Generali comparison entry point."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import expit
from sklearn.preprocessing import PolynomialFeatures

from data.dataset_metadata import DATASET_PATH
from data.full_monotone_spline_cache import ShardedMonotoneSplineCache
from data.loader import load_x_frame
from experiments.envelopes.feature_wild import WildFeatureDesign, feature_band_components
from experiments.envelopes.wild import residual_radii
from experiments.launch import LaunchPlan, run_launch_plan
from experiments.paths import results_root
from experiments.policy_utils import artifact_policy_features, constant_softmax_theta, optimization_trace_summary
from experiments.provenance import file_record, file_sha256, save_arrays_atomic, write_json_atomic
from experiments.seeds import derive_seed
from experiments.slurm import CPU_PROFILE
from objective.policy import SoftmaxPolicy
from objective.policy_preprocessing import PolicyFeaturePreprocessor
from optimization.solvers import run_first_order_minimize
from reporting.real_data import resolve_real_data_model
from scratch.generali_residual_wild import _metrics
from scratch.generali_wild_policy import _Responses, _AcceptanceBandObjective, _policy_band

ROOT = Path(__file__).resolve().parents[1]
DEGREES = {"affine": 1, "quadratic": 2, "cubic": 3}


def _load(path):
    with np.load(path, allow_pickle=False) as arrays:
        return dict(arrays)


def _verify(record, path=None):
    path = Path(record["path"]) if path is None else Path(path)
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return path


def _split(customer_ids, old_train, holdout, seed):
    ids = np.unique(customer_ids[old_train])
    shuffled = np.random.default_rng(seed).permutation(ids)
    chosen = shuffled[:len(shuffled)//2]
    mask = np.isin(customer_ids[old_train], chosen)
    representation, beta = old_train[mask], old_train[~mask]
    pools = (representation, beta, holdout)
    if not np.array_equal(np.sort(np.concatenate(pools)), np.arange(len(customer_ids))):
        raise ValueError("Study split does not partition all saved rows.")
    for i in range(3):
        for j in range(i):
            if np.intersect1d(customer_ids[pools[i]], customer_ids[pools[j]]).size:
                raise ValueError("Customer identity crosses study partitions.")
    return {"representation": representation, "beta": beta, "holdout": holdout}


class _Basis:
    def __init__(self, kind, input_mean, input_scale, *, hidden=16, weights=None,
                 feature_mean=None, feature_scale=None):
        self.kind, self.input_mean, self.input_scale = kind, input_mean, input_scale
        self.hidden, self.weights = hidden, weights
        self.feature_mean, self.feature_scale = feature_mean, feature_scale
        self.poly = PolynomialFeatures(DEGREES[kind], include_bias=True).fit(
            np.zeros((1, len(input_mean)))) if kind in DEGREES else None

    def raw(self, inputs, derivative=False):
        z = (inputs-self.input_mean)/self.input_scale
        if self.poly is not None:
            out = self.poly.transform(z)
            if not derivative:
                return out
            powers = self.poly.powers_
            d = np.zeros_like(out)
            for j in np.flatnonzero(powers[:, -1]):
                exponent = powers[j].copy()
                exponent[-1] -= 1
                d[:, j] = powers[j, -1]*np.prod(z**exponent, axis=1)/self.input_scale[-1]
            return out, d
        w1, b1, w2, b2 = self.weights
        h1 = np.tanh(z@w1+b1)
        h2 = np.tanh(h1@w2+b2)
        out = np.column_stack((np.ones(len(inputs)), h2))
        if not derivative:
            return out
        d1 = (1-h1*h1)*w1[-1]/self.input_scale[-1]
        d2 = (1-h2*h2)*(d1@w2)
        return out, np.column_stack((np.zeros(len(inputs)), d2))

    def evaluate(self, inputs, derivative=False):
        if derivative:
            phi, d = self.raw(inputs, True)
            return (phi-self.feature_mean)/self.feature_scale, d/self.feature_scale
        return (self.raw(inputs)-self.feature_mean)/self.feature_scale

    def state(self):
        arrays = {"basis_kind": np.array(self.kind), "input_mean": self.input_mean,
                  "input_scale": self.input_scale, "feature_mean": self.feature_mean,
                  "feature_scale": self.feature_scale, "hidden": np.array(self.hidden)}
        if self.weights is not None:
            arrays.update(dict(zip(("w1", "b1", "w2", "b2"), self.weights)))
        return arrays

    @classmethod
    def restore(cls, state):
        kind = str(state["basis_kind"])
        return cls(kind, state["input_mean"], state["input_scale"], hidden=int(state["hidden"]),
                   weights=tuple(state[k] for k in ("w1", "b1", "w2", "b2")) if kind == "mlp" else None,
                   feature_mean=state["feature_mean"], feature_scale=state["feature_scale"])


class _Deadline(Exception):
    pass


class _TrackedObjective:
    def __init__(self, objective):
        self.objective = objective
        self.last_theta = None

    def __getattr__(self, name):
        return getattr(self.objective, name)

    def value(self, theta, inputs):
        self.last_theta = np.array(theta, copy=True)
        return self.objective.value(theta, inputs)

    def grad(self, theta, inputs):
        return self.objective.grad(theta, inputs)


class _Checkpoint:
    def __init__(self, objective, path, wall_seconds):
        self.objective, self.path = objective, path
        self.started, self.wall_seconds = time.monotonic(), wall_seconds
        self.rows, self.theta = [], None

    def log_step(self, method, step, action, value, gradient, **kwargs):
        self.theta = self.objective.last_theta.copy()
        self.rows.append([step, value, gradient, time.monotonic()-self.started])
        save_arrays_atomic(self.path, theta=self.theta, progress=np.array(self.rows))
        if step % 100 == 0:
            print(f"{self.path.parent.name}/{self.path.stem}: step {step}, value {value:.7g}, "
                  f"elapsed {self.rows[-1][-1]:.1f}s", flush=True)
        if time.monotonic()-self.started >= self.wall_seconds:
            raise _Deadline("Study wall-time limit reached at a saved accepted iterate.")


def _solve(objective, inputs, theta0, settings, output, *, algorithm, resume=True):
    result_path = output.with_suffix(".json")
    checkpoint_path = output.with_suffix(".npz")
    if result_path.exists():
        result = json.loads(result_path.read_text())
        saved = _load(_verify(result["checkpoint"], checkpoint_path))
        return saved["theta"], result
    resumed = resume and checkpoint_path.exists()
    if resumed:
        theta0 = _load(checkpoint_path)["theta"]
    tracked = _TrackedObjective(objective)
    reporter = _Checkpoint(tracked, checkpoint_path, settings["wall_seconds"])
    started = time.monotonic()
    try:
        theta, trace = run_first_order_minimize(theta0, inputs, tracked,
            t_steps=settings["max_steps"], n_grad_samples=1, sigma=.001,
            algorithm=algorithm, grad_norm_tol=settings["grad_norm_tol"], step_reporter=reporter)
        info = optimization_trace_summary(trace)
        save_arrays_atomic(checkpoint_path, theta=theta, progress=np.array(reporter.rows),
                           theta_trace=np.asarray(trace.theta_values))
    except _Deadline as error:
        theta = reporter.theta
        info = {"success": False, "status": "wall_time_limit", "message": str(error),
                "steps": len(reporter.rows)-1}
    info.update(runtime_seconds=time.monotonic()-started, algorithm=algorithm,
                entry_point="optimization.solvers.run_first_order_minimize",
                restarted_from_saved_iterate=resumed, restart_preserves_solver_memory=False,
                final_objective=objective.value(theta, inputs), checkpoint=file_record(checkpoint_path))
    if hasattr(objective, "mean_acceptance"):
        info["mean_acceptance"] = objective.mean_acceptance(theta, inputs)
    write_json_atomic(result_path, info)
    return theta, info


class _MLPLoss:
    def __init__(self, inputs, targets, hidden):
        self.d, self.h = inputs.shape[1], hidden
        self.target_mean, self.target_scale = float(targets.mean()), float(targets.std())
        if not self.target_scale > 0:
            raise ValueError("Zero neural training target variance.")
        self.inputs = inputs
        self.targets = (targets-self.target_mean)/self.target_scale
        self.last, self.cached = None, None

    def unpack(self, t):
        d, h = self.d, self.h
        k = d*h
        w1, b1 = t[:k].reshape(d, h), t[k:k+h]
        k += h
        w2, b2 = t[k:k+h*h].reshape(h, h), t[k+h*h:k+h*h+h]
        k += h*h+h
        return w1, b1, w2, b2, t[k:k+h], t[-1]

    def _evaluate(self, theta):
        if self.last is None or not np.array_equal(theta, self.last):
            # Same two-tanh-layer chain rule as MLPPolicy, with an unbounded
            # regression head. BLAS evaluates the full fixed training batch.
            w1, b1, w2, b2, w3, b3 = self.unpack(theta)
            h1 = np.tanh(self.inputs@w1+b1)
            h2 = np.tanh(h1@w2+b2)
            residual = h2@w3+b3-self.targets
            delta = residual/len(residual)
            second = (delta[:, None]*w3)*(1-h2*h2)
            first = (second@w2.T)*(1-h1*h1)
            grad = np.r_[(self.inputs.T@first).ravel(), first.sum(0),
                         (h1.T@second).ravel(), second.sum(0), h2.T@delta, delta.sum()]
            self.cached = (float(.5*np.mean(residual**2)), grad)
            self.last = np.array(theta, copy=True)
        return self.cached

    def value(self, theta, inputs):
        return self._evaluate(theta)[0]

    def grad(self, theta, inputs):
        return self._evaluate(theta)[1]


def _learn_basis(kind, inputs, residual, split, config, output, name):
    representation = split["representation"]
    mean, scale = inputs[representation].mean(axis=0), inputs[representation].std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Constant representation input; do not silently remove it.")
    weights, network_info = None, None
    if kind == "mlp":
        hidden = config["mlp"]["hidden"]
        x = (inputs[representation]-mean)/scale
        loss = _MLPLoss(x, residual[representation], hidden)
        seed = derive_seed(config["seed_root"], f"generali-residual:mlp-initialization:{name}")
        rng = np.random.default_rng(seed)
        d, h = inputs.shape[1], hidden
        theta0 = np.r_[rng.uniform(-np.sqrt(6/(d+h)), np.sqrt(6/(d+h)), d*h), np.zeros(h),
                       rng.uniform(-np.sqrt(3/h), np.sqrt(3/h), h*h), np.zeros(h),
                       rng.uniform(-np.sqrt(6/(h+1)), np.sqrt(6/(h+1)), h), 0.]
        theta, network_info = _solve(loss, x, theta0, config["mlp"], output/"network", algorithm="l-bfgs-b")
        weights = tuple(np.asarray(w) for w in loss.unpack(theta)[:4])
        network_info.update(seed=seed, target_mean=loss.target_mean, target_scale=loss.target_scale,
                            hidden_widths=[hidden, hidden], trained_parameters=len(theta))
        del loss
    basis = _Basis(kind, mean, scale, hidden=config["mlp"]["hidden"], weights=weights)
    phi = basis.raw(inputs[split["beta"]])
    center, scaling = phi.mean(axis=0), phi.std(axis=0)
    center[0], scaling[0] = 0., 1.
    if np.any(scaling <= 0):
        raise ValueError("Constant non-intercept feature; no implicit dropped terms.")
    basis.feature_mean, basis.feature_scale = center, scaling
    phi -= center
    phi /= scaling
    return basis, phi, network_info


class _FeatureFibers:
    def __init__(self, inputs, basis, beta, triangular, critical):
        self.inputs, self.basis, self.beta = inputs, basis, beta
        self.triangular, self.critical = triangular, float(critical)
        self.prepared = None

    def values(self, positions, actions):
        x = self.inputs[positions].copy()
        x[:, -1] = actions
        return feature_band_components(self.basis.evaluate(x), self.beta, self.triangular, self.critical)

    def direct(self, positions, actions):
        x = self.inputs[positions].copy()
        x[:, -1] = actions
        phi, dphi = self.basis.evaluate(x, True)
        v = solve_triangular(self.triangular.T, phi.T, lower=True).T
        dv = solve_triangular(self.triangular.T, dphi.T, lower=True).T
        norms = np.linalg.norm(v, axis=1)
        if np.any(norms <= 0):
            raise ValueError("Nonpositive frozen-feature norm.")
        return self._parts(phi@self.beta, self.critical*norms, dphi@self.beta,
                           self.critical*np.einsum("ij,ij->i", v, dv)/norms)

    @staticmethod
    def _parts(g, w, dg, dw):
        lower, upper = residual_radii(g, w)
        return {"residual_fit": g, "width": w, "radius_lower": lower, "radius_upper": upper,
                "lower_adjustment_derivative": (lower > 0)*(dg-dw),
                "upper_adjustment_derivative": (upper > 0)*(dg+dw)}

    def prepare(self, positions):
        if self.basis.poly is None:
            return
        # Exact coefficients in standardized action t; no sampled interpolation.
        powers = self.basis.poly.powers_
        degree = DEGREES[self.basis.kind]
        z = (self.inputs[positions]-self.basis.input_mean)/self.basis.input_scale
        z[:, -1] = 1.
        base = self.basis.poly.transform(z)/self.basis.feature_scale
        vectors, g = [], []
        for k in range(degree+1):
            phi = base*(powers[:, -1] == k)
            if k == 0:
                phi -= self.basis.feature_mean/self.basis.feature_scale
            g.append(phi@self.beta)
            vectors.append(solve_triangular(self.triangular.T, phi.T, lower=True).T)
        q = np.zeros((len(positions), 2*degree+1))
        for i in range(degree+1):
            for j in range(degree+1):
                q[:, i+j] += np.einsum("ij,ij->i", vectors[i], vectors[j])
        self.prepared = (positions.copy(), np.column_stack(g), q)
        probe = np.linspace(-.1, .2, len(positions))
        actual, expected = self.evaluate(positions, probe), self.direct(positions, probe)
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-7, atol=1e-6)

    def evaluate(self, positions, actions):
        if self.prepared is None or not np.array_equal(positions, self.prepared[0]):
            return self.direct(positions, actions)
        _, g, q = self.prepared
        t = (np.asarray(actions)-self.basis.input_mean[-1])/self.basis.input_scale[-1]
        def evaluate(c):
            return np.polynomial.polynomial.polyval(t, c.T, tensor=False)
        qvalue = evaluate(q)
        if np.any(qvalue <= 0):
            raise ValueError("Nonpositive cached polynomial norm.")
        root = np.sqrt(qvalue)
        dg = evaluate(g[:, 1:]*np.arange(1, g.shape[1]))/self.basis.input_scale[-1]
        dq = evaluate(q[:, 1:]*np.arange(1, q.shape[1]))/self.basis.input_scale[-1]
        return self._parts(evaluate(g), self.critical*root, dg, self.critical*dq/(2*root))


def _inputs(config, source):
    baseline = results_root()/config["baseline_fit"]
    provenance = json.loads((baseline/"provenance.json").read_text())
    _verify(provenance["input_residuals"], source/"residuals.npz")
    _verify(provenance["dataset"], DATASET_PATH)
    record = next(r for r in provenance["outputs"] if Path(r["path"]).name == "fit_state.npz")
    state = _load(_verify(record, baseline/"fit_state.npz"))
    arrays = _load(source/"residuals.npz")
    np.testing.assert_array_equal(state["row_indices"], arrays["row_indices"])
    split_seed = derive_seed(config["seed_root"], "generali-residual:representation-beta-split")
    split = _split(arrays["customer_id"], state["train_positions"], state["test_positions"], split_seed)
    inputs = np.column_stack((state["X"], arrays["a"]))
    return arrays, inputs, split, provenance, state["bootstrap_draw_seeds"]


def _fit(config, kind, name, arrays, inputs, split, draw_seeds, output):
    fit_path = output/"band.npz"
    if (output/"fit.json").exists():
        info = json.loads((output/"fit.json").read_text())
        state = _load(_verify(info["artifact"], fit_path))
        return _Basis.restore(state), state, info
    started = time.monotonic()
    residual = arrays[f"{name}_residual"]
    basis, matrix, network = _learn_basis(kind, inputs, residual, split, config, output, name)
    print(f"{name}/{kind}: QR of {matrix.shape}", flush=True)
    design = WildFeatureDesign(matrix)
    beta = design.fit(residual[split["beta"]])
    errors = residual[split["beta"]]-matrix@beta
    setup_seconds = time.monotonic()-started
    count, batch_size = config["bootstrap"]["count"], config["bootstrap"]["batch_size"]
    if len(draw_seeds) < count:
        raise ValueError("Requested draws exceed the original saved paired seed sequence.")
    partial = output/"bootstrap_checkpoint.npz"
    statistics = np.full(count, np.nan)
    coefficients = np.full((count, len(beta)), np.nan)
    start, digest = 0, hashlib.sha256()
    if partial.exists():
        old = _load(partial)
        np.testing.assert_allclose(old["beta"], beta, rtol=0, atol=0)
        statistics, coefficients, start = old["statistics"], old["errors"], int(old["completed"])
    batch_times = []
    for first in range(0, count, batch_size):
        stop = min(first+batch_size, count)
        tick = time.monotonic()
        signs = np.stack([2*np.random.default_rng(int(seed)).integers(
            0, 2, size=len(inputs), dtype=np.int8)-1 for seed in draw_seeds[first:stop]])
        digest.update(signs.tobytes())
        if stop <= start:
            continue
        if first < start:
            raise ValueError("Bootstrap checkpoint is not at a configured batch boundary.")
        coefficients[first:stop], statistics[first:stop] = design.bootstrap_errors(errors, signs[:, split["beta"]])
        save_arrays_atomic(partial, beta=beta, statistics=statistics, errors=coefficients, completed=np.array(stop))
        batch_times.append(time.monotonic()-tick)
        if first == 0 or stop == count or first % (batch_size*16) == 0:
            estimate = np.mean(batch_times)*(count-stop)/batch_size
            print(f"{name}/{kind}: bootstrap {stop}/{count}; estimated remaining {estimate:.0f}s", flush=True)
    critical = float(np.partition(statistics, math.ceil(count*(1-config["bootstrap"]["delta"]))-1)[
        math.ceil(count*(1-config["bootstrap"]["delta"]))-1])
    # Validate on untouched residual outcomes, before any policy optimization.
    fit_metrics = []
    for stage, indices in split.items():
        prediction = np.empty(len(indices))
        for first in range(0, len(indices), config["evaluation_batch_size"]):
            stop = min(first+config["evaluation_batch_size"], len(indices))
            prediction[first:stop] = basis.evaluate(inputs[indices[first:stop]])@beta
        fit_metrics.extend(_metrics(residual[indices], prediction, arrays["is_churn"][indices],
                                    model=name, stage=stage))
    state = {**basis.state(), "beta": beta, "triangular": design.triangular,
             "critical": np.array(critical), "statistics": statistics,
             "bootstrap_errors": coefficients, "draw_seeds": draw_seeds[:count],
             "beta_row_indices": arrays["row_indices"][split["beta"]],
             **{f"{k}_positions": v for k, v in split.items()}}
    save_arrays_atomic(fit_path, **state)
    info = {"model": name, "basis": kind, "beta_features": len(beta),
            "n": {key: len(value) for key, value in split.items()}, "network": network,
            "condition_number": design.condition_number, "max_leverage": float(design.leverage.max()),
            "leverage_quantiles": np.quantile(design.leverage, [.5,.95,.99,.999]).tolist(),
            "critical": critical, "bootstrap_draws": count,
            "full_row_multiplier_sha256": digest.hexdigest(), "fit_metrics": fit_metrics,
            "setup_seconds": setup_seconds, "fit_bootstrap_seconds": time.monotonic()-started,
            "artifact": file_record(fit_path), "calibration": "coefficient_ellipsoid",
            "coverage": None, "coverage_status": "unknown; conditional frozen-feature bootstrap approximation"}
    write_json_atomic(output/"fit.json", info)
    return basis, state, info


def _responses(config, name, arrays, provenance):
    model = resolve_real_data_model("glm" if name == "glm" else "xgb")
    for key in ("acceptance_artifact", "loss_artifact"):
        if model.provenance()[key]["sha256"] != provenance["models"][name][key]["sha256"]:
            raise ValueError(f"Changed frozen {name} predictor.")
    frame = load_x_frame("xgb", row_indices=arrays["row_indices"])
    cache, logit, slope = None, None, None
    if name == "glm":
        raw = frame.copy()
        raw["U"] = 0.
        artifact = model.acceptance_artifact
        transformed = artifact.model_frame(raw)
        logit = artifact.model.decision_function(transformed)
        slope = float(artifact.model.coef_[0, list(transformed.columns).index("U")])
        np.testing.assert_allclose(expit(logit+slope*arrays["a"]), arrays["glm_acceptance"], rtol=1e-11, atol=1e-12)
    else:
        from scratch.plot_generali_profit_residuals import _validate_cache
        cache = ShardedMonotoneSplineCache(results_root()/config["spline_cache"])
        _validate_cache(cache, arrays["row_indices"], provenance["dataset"], provenance["models"][name],
                        provenance["models"][name]["spline_anchor_weights"])
    return model, frame, _Responses(name, arrays, cache=cache, glm_logit=logit, glm_slope=slope)


def _scores(response, fibers, actions, *, positions, cohort, policy, batch_size):
    totals = {}
    for first in range(0, len(positions), batch_size):
        selected = positions[first:first+batch_size]
        p, b = response.evaluate(selected, actions[selected]), fibers.values(selected, actions[selected])
        values = {"predicted_profit": p["center"], "corrected_profit": p["center"]+b["residual_fit"],
                  "lower_bound": p["center"]-b["radius_lower"], "upper_bound": p["center"]+b["radius_upper"],
                  "acceptance": p["acceptance"], "residual_fit": b["residual_fit"],
                  "half_width": b["width"], "lower_penalty": b["radius_lower"], "upper_penalty": b["radius_upper"]}
        for key, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite {key} in full-row policy evaluation.")
            totals[key] = totals.get(key, 0.)+float(np.sum(value))
    return {"model": response.name, "basis": fibers.basis.kind, "policy": policy, "cohort": cohort,
            "n": len(positions), "mean_action": float(np.mean(actions[positions])),
            **{f"mean_{key}": value/len(positions) for key, value in totals.items()}}


def _policies(config, kind, name, arrays, inputs, basis, state, fit_info, provenance, output):
    model, frame, response = _responses(config, name, arrays, provenance)
    baseline_dir = results_root()/config["baseline_policy"]
    historical = json.loads((baseline_dir/"summary.json").read_text())
    historical_provenance = json.loads((baseline_dir/"provenance.json").read_text())
    artifact_name = f"{name}_nominal_policy.npz"
    record = next(r for r in historical_provenance["outputs"] if Path(r["path"]).name == artifact_name)
    nominal = _load(_verify(record, baseline_dir/artifact_name))
    np.testing.assert_array_equal(nominal["row_indices"], arrays["row_indices"])
    sample_rows = nominal["optimization_row_indices"]
    positions = np.searchsorted(arrays["row_indices"], sample_rows)
    np.testing.assert_array_equal(arrays["row_indices"][positions], sample_rows)
    if len(positions) != 20000:
        raise ValueError("Preserve the exact saved 20k optimization cohort.")
    reference = historical["references"][name]
    processor = PolicyFeaturePreprocessor.from_state(reference["policy_preprocessor"],
        {k.removeprefix("preprocessor_"): v for k, v in nominal.items() if k.startswith("preprocessor_")})
    features = processor.transform(artifact_policy_features(model.acceptance_artifact, frame))
    sample = features[positions]
    bounds = config["policy"]["action_bounds"]
    policy = SoftmaxPolicy(action_low=bounds[0], action_high=bounds[1])
    np.testing.assert_allclose(policy.value(nominal["theta"], features), nominal["actions"], atol=1e-13, rtol=1e-12)
    response.prepare_policy_queries(positions)
    fibers = _FeatureFibers(inputs, basis, state["beta"], state["triangular"], state["critical"])
    prepare_start = time.monotonic()
    fibers.prepare(positions)
    prepare_seconds = time.monotonic()-prepare_start
    band = _policy_band(response, fibers, positions, sample, bounds)
    band.metadata.update(construction="frozen_feature_coefficient_ellipsoid_transfer")
    objective = _AcceptanceBandObjective(band, policy, endpoint="lower", sense="maximize",
        artifact=file_record(output/"band.npz"), acceptance_floor=float(reference["acceptance_floor"]))
    theta0 = constant_softmax_theta(policy, sample.shape[1], config["policy"]["initial_action"])
    theta, solver = _solve(objective, sample, theta0, config["policy"], output/"policy_optimizer", algorithm="trust-constr")
    actions = np.clip(policy.value(theta, features), *bounds)
    action_path = output/"policy.npz"
    save_arrays_atomic(action_path, theta=theta, actions=actions, row_indices=arrays["row_indices"],
                       optimization_row_indices=sample_rows,
                       **{k:v for k,v in nominal.items() if k.startswith("preprocessor_")})
    metrics = []
    full = np.arange(len(inputs))
    for label, values in (("wild_lower", actions), ("nominal", nominal["actions"]), ("historical", arrays["a"])):
        for cohort, selected in (("full", full), ("optimization", positions)):
            metrics.append(_scores(response, fibers, values, positions=selected, cohort=cohort,
                                   policy=label, batch_size=config["evaluation_batch_size"]))
    return {"model": name, "basis": kind, "fit": fit_info, "optimizer": solver,
            "policy": file_record(action_path), "nominal_reference": file_record(baseline_dir/artifact_name),
            "nominal_optimizer": historical["optimizers"][f"{name}_nominal"],
            "metrics": metrics, "polynomial_cache_seconds": prepare_seconds,
            "mean_absolute_action_change_from_nominal": float(np.mean(np.abs(actions-nominal["actions"]))),
            "acceptance_floor": reference["acceptance_floor"], "global_optimality": "not certified",
            "realized_counterfactual_profit": None}


def _contract(config, source, config_path):
    baseline = results_root()/config["baseline_fit"]
    return {"config": config, "config_file": file_record(config_path),
            "residuals": file_record(source/"residuals.npz"),
            "baseline_fit": file_record(baseline/"fit_state.npz"),
            "baseline_provenance": file_record(baseline/"provenance.json"),
            "sources": [file_record(p) for p in [Path(__file__), ROOT/"scratch/generali_wild_policy.py",
                ROOT/"src/experiments/envelopes/feature_wild.py", ROOT/"src/experiments/envelopes/wild.py",
                ROOT/"src/optimization/solvers.py", ROOT/"src/optimization/base.py",
                ROOT/"src/objective/policy.py", ROOT/"src/objective/banded.py",
                ROOT/"src/data/full_monotone_spline_cache.py"]]}


def _run_task(config, source, config_path, output, index):
    tasks = [(kind, name) for kind in config["bases"] for name in config["models"]]
    kind, name = tasks[index]
    folder = output/kind/name
    folder.mkdir(parents=True, exist_ok=True)
    contract_path = folder/"contract.json"
    contract = _contract(config, source, config_path)
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Study inputs or source changed; preserve the completed contract.")
    write_json_atomic(contract_path, contract)
    if (folder/"summary.json").exists():
        saved = json.loads((folder/"summary.json").read_text())
        _verify(saved["policy"])
        _verify(saved["fit"]["artifact"])
        return {"status": "replayed_completed_task", "summary": file_record(folder/"summary.json")}
    started = time.monotonic()
    try:
        arrays, inputs, split, provenance, seeds = _inputs(config, source)
        basis, state, fit_info = _fit(config, kind, name, arrays, inputs, split, seeds, folder)
        summary = _policies(config, kind, name, arrays, inputs, basis, state, fit_info, provenance, folder)
        summary["total_seconds"] = time.monotonic()-started
        write_json_atomic(folder/"summary.json", summary)
        return {"status": "complete", "summary": file_record(folder/"summary.json")}
    except Exception as error:
        write_json_atomic(folder/"failure.json", {"type": type(error).__name__, "message": str(error),
            "elapsed_seconds": time.monotonic()-started, "completed_fit": (folder/"fit.json").exists()})
        raise


def _collect(config, output):
    metrics, fits, outcomes, missing = [], [], [], []
    for kind in config["bases"]:
        for name in config["models"]:
            path = output/kind/name/"summary.json"
            if not path.exists():
                missing.append(f"{kind}/{name}")
                continue
            saved = json.loads(path.read_text())
            metrics.extend(saved["metrics"])
            fits.extend({**row, "basis": kind} for row in saved["fit"]["fit_metrics"])
            outcomes.append({"basis": kind, "model": name, "beta_features": saved["fit"]["beta_features"],
                "max_leverage": saved["fit"]["max_leverage"], "condition_number": saved["fit"]["condition_number"],
                "critical": saved["fit"]["critical"], "total_seconds": saved["total_seconds"],
                "optimizer_success": saved["optimizer"]["success"], "optimizer_message": saved["optimizer"]["message"]})
    pd.DataFrame(metrics).to_csv(output/"policy_metrics.csv", index=False)
    pd.DataFrame(fits).to_csv(output/"residual_fit_metrics.csv", index=False)
    pd.DataFrame(outcomes).to_csv(output/"solver_and_geometry.csv", index=False)
    write_json_atomic(output/"summary.json", {"family": config["family"], "config": config,
        "completed": len(outcomes), "missing": missing, "outcomes": outcomes, "metrics": metrics,
        "coverage": None, "evaluation": "equal-weight all-row scores at personalized actions",
        "split": "40% representation / 40% beta and bootstrap / original 20% residual validation",
        "historical_baseline": config["baseline_policy"]})
    (output/"EXPERIMENT.md").write_text((ROOT/"docs/generali_residual_basis_study.md").read_text())
    print(f"Collected {len(outcomes)}/8 residual-basis tasks; missing={missing}", flush=True)


def run_basis_study(args, argv):
    config_path = args.fit_wild_bases.resolve()
    config = json.loads(config_path.read_text())
    if (config["family"] != "glm-spline-profit-dispersion"
            or config["representation_fraction_of_saved_train"] != .5
            or config["bootstrap"]["calibration"] != "coefficient_ellipsoid"):
        raise ValueError("Use the approved family, 40/40/20 split and common calibration.")
    output, source = args.output_dir.resolve(), args.replay.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = LaunchPlan(name="generali-residual-bases", task_count=len(config["bases"])*len(config["models"]),
        requires_jax=False, runs_root=str(output), default_array=True,
        slurm_profile=replace(CPU_PROFILE, job_name="generali-residual-bases"),
        run_task=lambda index, context: _run_task(config, source, config_path, output, index),
        collect=lambda context: _collect(config, output))
    run_launch_plan(plan, args=args, argv=[str(ROOT/"scratch/plot_generali_profit_residuals.py"), *argv], cwd=ROOT)
