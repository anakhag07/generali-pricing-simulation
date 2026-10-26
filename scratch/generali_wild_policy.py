"""Replay the Generali affine wild band into presentation-style policy analysis.

The existing comparison driver invokes this study adapter. Band coefficients
and QR geometry stay frozen; all policy endpoints come from repository solvers.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import expit

from data.dataset_metadata import DATASET_PATH
from data.full_monotone_spline_cache import ShardedMonotoneSplineCache
from data.loader import load_x_frame
from experiments.envelopes.core import FittedEnvelope
from experiments.envelopes.fitting import band_from_state
from experiments.envelopes.plotting import _STYLE
from experiments.envelopes.wild import residual_radii
from experiments.policy_utils import artifact_policy_features, constant_softmax_theta, optimization_trace_summary
from experiments.provenance import file_record, file_sha256, write_provenance
from objective.banded import BandObjective
from objective.policy import IdentityFeatureMap, SoftmaxPolicy
from objective.policy_preprocessing import PolicyFeaturePreprocessor
from optimization.solvers import run_first_order_minimize
from reporting.real_data import resolve_real_data_model
from reporting.profit_dispersion import row_index_sha256

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"glm": "GLM", "spline_xgb": "Spline + XGBoost"}


class _FrozenFibers:
    """Cache the exact action-polynomial QR norm from an existing affine band."""

    def __init__(self, z, triangular, beta, critical):
        phi0 = np.column_stack((np.ones(len(z)), z, np.zeros(len(z))))
        v = solve_triangular(triangular.T, phi0.T, lower=True).T
        direction = np.zeros(len(beta))
        direction[-1] = 1.
        t = solve_triangular(triangular.T, direction, lower=True)
        self.q0, self.q1, self.q2 = np.sum(v*v, axis=1), v@t, float(t@t)
        self.g0, self.slope = phi0@beta, float(beta[-1])
        self.critical = float(critical)

    def evaluate(self, positions, actions):
        a = np.asarray(actions, dtype=float)
        q = self.q0[positions] + 2*a*self.q1[positions] + a*a*self.q2
        if np.any(q <= 0):
            raise ValueError("Nonpositive saved QR width geometry.")
        root = np.sqrt(q)
        width = self.critical*root
        fitted = self.g0[positions]+self.slope*a
        lower, upper = residual_radii(fitted, width)
        dwidth = self.critical*(self.q1[positions]+a*self.q2)/root
        return {"residual_fit": fitted, "width": width, "radius_lower": lower, "radius_upper": upper,
                "lower_adjustment_derivative": -((lower > 0)*(dwidth-self.slope)),
                "upper_adjustment_derivative": (upper > 0)*(dwidth+self.slope)}


class _Responses:
    """Evaluate unchanged GLM logits or cached exact-spline acceptance pairwise."""

    def __init__(self, name, arrays, *, cache=None, glm_logit=None, glm_slope=None):
        self.name, self.arrays, self.cache = name, arrays, cache
        self.logit, self.slope = glm_logit, glm_slope

    def evaluate(self, positions, actions, *, common_action=False):
        a = np.asarray(actions, dtype=float)
        rows = self.arrays["row_indices"][positions]
        if self.name == "glm":
            acceptance = expit(self.logit[positions] + self.slope*a)
            derivative = self.slope*acceptance*(1-acceptance)
        elif common_action:
            acceptance = self.cache.acceptance(rows, float(a))
            derivative = np.zeros_like(acceptance)  # Not used for common-action reporting.
        else:
            acceptance, derivative = self.cache.pairwise_acceptance_and_derivative(rows, a)
        premium = self.arrays["premium"][positions]
        margin = premium*(1+a)-self.arrays[f"{self.name}_loss"][positions]
        return {"center": acceptance*margin, "center_derivative": derivative*margin+acceptance*premium,
                "acceptance": acceptance, "acceptance_derivative": derivative}


class _AcceptanceBandObjective(BandObjective):
    """Use the existing policy/band objective with exact acceptance callbacks."""

    def mean_acceptance(self, theta, x_batch):
        return self._acceptance(theta, x_batch)

    def mean_acceptance_grad(self, theta, x_batch):
        return self._acceptance_grad(theta, x_batch)

    def mean_acceptance_at_u(self, x_batch, u):
        return self._acceptance_at_u(x_batch, u)


def _policy_band(response, fibers, positions, policy_features, bounds):
    last_actions, last_parts = None, None

    def parts(X, actions):
        nonlocal last_actions, last_parts
        if not np.array_equal(X, policy_features):
            raise ValueError("Policy objective requires its original ordered full optimization cohort.")
        if np.any((actions < bounds[0]-1e-12) | (actions > bounds[1]+1e-12)):
            raise ValueError("Policy query outside the inherited action interval.")
        # Saturated softmax arithmetic can exceed an endpoint by one ulp.
        actions = np.clip(actions, *bounds)
        if last_actions is None or not np.array_equal(actions, last_actions):
            last_parts = {**response.evaluate(positions, actions), **fibers.evaluate(positions, actions)}
            last_actions = actions.copy()
        return last_parts

    def derivatives(X, actions):
        p = parts(X, actions)
        return {"center": p["center_derivative"],
                "lower": p["center_derivative"]+p["lower_adjustment_derivative"],
                "upper": p["center_derivative"]+p["upper_adjustment_derivative"]}

    return FittedEnvelope(
        center=lambda X, a: parts(X, a)["center"],
        radii=lambda X, a: (parts(X, a)["radius_lower"], parts(X, a)["radius_upper"]),
        components=parts, action_derivatives=derivatives, state={"X": policy_features},
        metadata={"construction": "unchanged_saved_affine_wild_residual_transfer",
                  "query_domain": "fixed_policy_cohort_cross_inherited_action_bounds"})


def _verify_fibers(state, name, fibers, examples):
    selected = np.unique(np.r_[np.linspace(0, len(state["X"])-1, 64, dtype=int),
                                examples["selection_positions"]])
    # This is a query-only view of full saved geometry; no recalibration or fit.
    view = {"X": state["X"][selected], "actions": state["actions"][selected],
            "y": state[f"{name}_observed_residual"][selected],
            "beta": state[f"{name}_beta"], "covariance": state["covariance"],
            "triangular": state["triangular"], "critical": state[f"{name}_critical"]}
    reference = band_from_state(view, {"method": "residual_wild", "action_degree": 1,
        "predictor": {"type": "linear", "coefficients": np.zeros(len(view["beta"])).tolist()}})
    maximum = 0.
    for value in [-.1, -.0371, 0., .08, .1234, .16, .2]:
        actions = np.full(len(selected), value)
        actual = fibers.evaluate(selected, actions)
        expected = reference.components(view["X"], actions)
        for key in ["residual_fit", "width", "radius_lower", "radius_upper"]:
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-11, atol=1e-8)
            maximum = max(maximum, float(np.max(np.abs(actual[key]-expected[key]))))
        np.testing.assert_allclose(actual["lower_adjustment_derivative"],
                                   reference.derivative(view["X"], actions, "lower"), rtol=2e-10, atol=1e-8)
    return maximum


def _mean_curves(response, fibers, grid):
    rows = []
    for j, action in enumerate(grid):
        p = response.evaluate(slice(None), action, common_action=True)
        b = fibers.evaluate(slice(None), action)
        center = p["center"]
        rows.append({"model": response.name, "action": action, "mean_prediction": np.mean(center),
                     "mean_corrected_prediction": np.mean(center+b["residual_fit"]),
                     "mean_lower": np.mean(center-b["radius_lower"]),
                     "mean_upper": np.mean(center+b["radius_upper"]),
                     "mean_residual_fit": np.mean(b["residual_fit"]), "mean_residual_half_width": np.mean(b["width"]),
                     "mean_acceptance": np.mean(p["acceptance"])})
        if j % 100 == 0:
            print(f"{response.name}: averaged exact endpoints at {j+1}/{len(grid)} actions", flush=True)
    return rows


def _policy_metrics(response, fibers, positions, actions, *, label, cohort):
    p, b = response.evaluate(positions, actions), fibers.evaluate(positions, actions)
    return {"model": response.name, "policy": label, "cohort": cohort, "n": len(actions),
            "mean_action": np.mean(actions), "action_std": np.std(actions),
            "mean_predicted_profit": np.mean(p["center"]),
            "mean_corrected_profit": np.mean(p["center"]+b["residual_fit"]),
            "mean_lower_bound": np.mean(p["center"]-b["radius_lower"]),
            "mean_upper_bound": np.mean(p["center"]+b["radius_upper"]),
            "mean_acceptance": np.mean(p["acceptance"]),
            **{f"action_q{q:g}": np.quantile(actions, q) for q in [.01, .1, .5, .9, .99]}}


def _plots(output, curves, actions, arrays, config):
    bounds = config["action_bounds"]
    bins = np.linspace(*bounds, round((bounds[1]-bounds[0])/config["histogram_bin_width"])+1)
    histogram_rows = []
    historic = arrays["a"]
    visible_history = historic[(historic >= bounds[0]) & (historic <= bounds[1])]
    with plt.rc_context(_STYLE):
        for kind in ["context_averaged_wild_bands", "policy_actions", "mean_bands_and_policy_actions"]:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
            legend_handles, legend_labels = [], []
            for ax, (name, label) in zip(axes, MODELS.items()):
                frame = curves[curves.model.eq(name)]
                if kind != "policy_actions":
                    ax.fill_between(frame.action, frame.mean_lower, frame.mean_upper, color="C0", alpha=.2,
                                    label="Mean transferred envelope (nominal 95%)")
                    ax.plot(frame.action, frame.mean_prediction, color="C0", label="Mean frozen prediction")
                    ax.plot(frame.action, frame.mean_corrected_prediction, "--", color="C1", label="Mean prediction + fitted residual")
                    ax.plot(frame.action, frame.mean_lower, color="C2", label="Mean lower endpoint")
                    ax.set_ylabel("Mean profit per row")
                if kind == "mean_bands_and_policy_actions":
                    density_ax = ax.twinx()
                    density_ax.hist(actions[name, "wild_lower"], bins=bins, density=True, histtype="step",
                                    color="C3", label="Wild-LCB policy actions")
                    density_ax.set_ylabel("Personalized action density")
                    if name == "glm":
                        handles, labels = density_ax.get_legend_handles_labels()
                        legend_handles.extend(handles)
                        legend_labels.extend(labels)
                if kind == "policy_actions":
                    for color, (series, values) in enumerate([
                        ("Nominal-profit policy", actions[name, "nominal"]),
                        ("Wild-LCB policy", actions[name, "wild_lower"]),
                        ("Historical actions in display range", visible_history)]):
                        count = np.histogram(values, bins=bins)[0]
                        if count.sum() != len(values):
                            raise ValueError("Action histogram omitted policy actions.")
                        density = count/(len(values)*np.diff(bins))
                        ax.stairs(density, bins, color=f"C{color}", label=series)
                        histogram_rows.extend({"model": name, "series": series, "left": low, "right": high,
                                               "count": n, "density": d} for low, high, n, d in
                                              zip(bins[:-1], bins[1:], count, density))
                    ax.set_ylabel("Price-change density")
                ax.set(title=label, xlabel="Price change, a")
                if name == "glm":
                    handles, labels = ax.get_legend_handles_labels()
                    legend_handles.extend(handles)
                    legend_labels.extend(labels)
            fig.legend(legend_handles, legend_labels, loc="outside lower center", ncol=2)
            fig.savefig(output / f"{kind}.pdf", format="pdf")
            plt.close(fig)
    pd.DataFrame(histogram_rows).to_csv(output / "policy_action_histograms.csv", index=False)
    return {"historical_rows_in_range": len(visible_history), "historical_rows_outside_range": len(historic)-len(visible_history)}


def run_wild_policy(arrays, output, source, config_path):
    config = json.loads(config_path.read_text())
    if config["family"] != "glm-spline-profit-dispersion":
        raise ValueError("Keep the existing Generali comparison family.")
    resolve = lambda value: (config_path.parent / value).resolve()
    fit_run = resolve(config["fit_run"])
    fit_provenance = json.loads((fit_run / "provenance.json").read_text())
    if file_sha256(source / "residuals.npz") != fit_provenance["input_residuals"]["sha256"]:
        raise ValueError("Residual cohort differs from fitted band.")
    fit_path = fit_run / "fit_state.npz"
    expected = next(item for item in fit_provenance["outputs"] if Path(item["path"]).name == fit_path.name)
    if file_sha256(fit_path) != expected["sha256"]:
        raise ValueError("Saved wild fit state changed.")
    if file_sha256(DATASET_PATH) != fit_provenance["dataset"]["sha256"]:
        raise ValueError("Generali dataset changed.")
    with np.load(fit_path, allow_pickle=False) as data:
        state = dict(data)
    np.testing.assert_array_equal(state["row_indices"], arrays["row_indices"])
    examples_path = Path(fit_provenance["customer_curves"]["path"])
    if file_sha256(examples_path) != fit_provenance["customer_curves"]["sha256"]:
        raise ValueError("Representative customer inputs changed.")
    with np.load(examples_path, allow_pickle=False) as data:
        examples = dict(data)
    frame = load_x_frame("xgb", row_indices=arrays["row_indices"])
    models = {"glm": resolve_real_data_model("glm"), "spline_xgb": resolve_real_data_model("xgb")}
    for name, model in models.items():
        for key in ["acceptance_artifact", "loss_artifact"]:
            if model.provenance()[key]["sha256"] != fit_provenance["models"][name][key]["sha256"]:
                raise ValueError(f"Changed frozen {name} {key}.")
    raw = frame.copy()
    raw["U"] = 0.
    artifact = models["glm"].acceptance_artifact
    model_frame = artifact.model_frame(raw)
    logit = artifact.model.decision_function(model_frame)
    action_column = list(model_frame.columns).index("U")
    slope = float(artifact.model.coef_[0, action_column])
    np.testing.assert_allclose(expit(logit+slope*arrays["a"]), arrays["glm_acceptance"], atol=1e-12, rtol=1e-11)
    del raw, model_frame
    cache = ShardedMonotoneSplineCache(resolve(config["spline_cache"]))
    from scratch.plot_generali_profit_residuals import _validate_cache
    _validate_cache(cache, arrays["row_indices"], fit_provenance["dataset"],
                    fit_provenance["models"]["spline_xgb"],
                    fit_provenance["models"]["spline_xgb"]["spline_anchor_weights"])
    responses = {name: _Responses(name, arrays, cache=cache, glm_logit=logit, glm_slope=slope) for name in MODELS}
    grid = np.linspace(*config["action_bounds"], config["action_grid_count"])
    curves, metrics, policy_results, all_actions, references, validation = [], [], {}, {}, {}, {}
    shared_rows = None
    for name, model in models.items():
        print(f"Preparing {name} policy objective from saved beta and QR geometry.", flush=True)
        fibers = _FrozenFibers(state["X"], state["triangular"], state[f"{name}_beta"], state[f"{name}_critical"])
        validation[name] = {"band_parity_max_abs_error": _verify_fibers(state, name, fibers, examples)}
        curves.extend(_mean_curves(responses[name], fibers, grid))
        reference_dir = resolve(config["reference_policies"][name])
        reference_summary = json.loads((reference_dir / "summary.json").read_text())
        reference_path = reference_dir / "support_lower_bound_policy_20k.npz"
        expected = reference_summary["outputs"][reference_path.name]
        if file_sha256(reference_path) != expected:
            raise ValueError("Presentation policy input checksum changed.")
        with np.load(reference_path, allow_pickle=False) as data:
            reference = dict(data)
        sample_rows = reference["row_indices"]
        if row_index_sha256(sample_rows) != reference_summary["sample"]["row_indices_sha256"]:
            raise ValueError("Presentation row order changed.")
        if shared_rows is not None:
            np.testing.assert_array_equal(sample_rows, shared_rows)
        shared_rows = sample_rows
        positions = np.searchsorted(arrays["row_indices"], sample_rows)
        np.testing.assert_array_equal(arrays["row_indices"][positions], sample_rows)
        np.testing.assert_allclose(reference["action_bounds"], config["action_bounds"], rtol=0, atol=0)
        processor = PolicyFeaturePreprocessor.from_state(reference_summary["policy"]["preprocessor"],
            {key: value for key, value in ((key.removeprefix("preprocessor_"), value)
             for key, value in reference.items() if key.startswith("preprocessor_"))})
        policy_features = processor.transform(artifact_policy_features(model.acceptance_artifact, frame))
        sample_features = policy_features[positions]
        policy = SoftmaxPolicy(feature_map=IdentityFeatureMap(), action_low=config["action_bounds"][0], action_high=config["action_bounds"][1])
        band = _policy_band(responses[name], fibers, positions, sample_features, config["action_bounds"])
        theta0 = constant_softmax_theta(policy, sample_features.shape[1], config["initial_action"])
        floor = float(reference["acceptance_floor"])
        references[name] = {"policy": file_record(reference_path), "summary": file_record(reference_dir / "summary.json"),
                            "acceptance_floor": floor, "optimizer_rows": len(positions),
                            "policy_preprocessor": reference_summary["policy"]["preprocessor"]}
        for label, action in [("historical", arrays["a"]), ("initial_0.08", np.full(len(frame), config["initial_action"]))]:
            metrics.append(_policy_metrics(responses[name], fibers, slice(None), action, label=label, cohort="full"))
        for label, endpoint in [("nominal", "center"), ("wild_lower", "lower")]:
            objective = _AcceptanceBandObjective(band, policy, endpoint=endpoint, sense="maximize",
                artifact=file_record(fit_path), acceptance_floor=floor)
            if objective.mean_acceptance(theta0, sample_features) < floor:
                raise ValueError("Inherited initial action violates the inherited acceptance floor.")
            print(f"{name}/{label}: repository first-order trust-constr on {len(positions):,} rows", flush=True)
            start = time.perf_counter()
            theta, trace = run_first_order_minimize(theta0.copy(), sample_features, objective,
                t_steps=config["optimizer"]["max_steps"], n_grad_samples=1, sigma=.001,
                algorithm=config["optimizer"]["algorithm"], grad_norm_tol=config["optimizer"]["grad_norm_tol"],
                initial_constr_penalty=config["optimizer"]["initial_constr_penalty"])
            elapsed = time.perf_counter()-start
            result = {**optimization_trace_summary(trace), "runtime_seconds": elapsed,
                      "entry_point": "optimization.solvers.run_first_order_minimize",
                      "endpoint": endpoint, "initial_objective": objective.value(theta0, sample_features),
                      "final_objective": objective.value(theta, sample_features)}
            actions = np.clip(policy.value(theta, policy_features), *config["action_bounds"])
            np.savez_compressed(output / f"{name}_{label}_policy.npz", theta=theta,
                row_indices=arrays["row_indices"], actions=actions, optimization_row_indices=sample_rows,
                theta_trace=np.asarray(trace.theta_values), objective_trace=trace.objective_values,
                gradient_norm_trace=trace.theta_grad_norms,
                **{key:value for key,value in reference.items() if key.startswith("preprocessor_")})
            policy_results[f"{name}_{label}"] = result
            all_actions[name, label] = actions
            for cohort, selected in [("optimization", positions), ("full", np.arange(len(frame)))]:
                metrics.append(_policy_metrics(responses[name], fibers, selected, actions[selected], label=label, cohort=cohort))
            print(f"{name}/{label}: success={result['success']}, {result['message']}; "
                  f"objective={result['final_objective']:.6f}", flush=True)
    curve_frame = pd.DataFrame(curves)
    curve_frame.to_csv(output / "context_averaged_band.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "policy_metrics.csv", index=False)
    histogram = _plots(output, curve_frame, all_actions, arrays, config)
    summary = {"family": config["family"], "config": config, "n_contexts_averaged": len(frame),
               "metrics": metrics, "optimizers": policy_results, "validation": validation,
               "references": references, "histogram": histogram,
               "target": "mean of actual customer-specific lower endpoints at personalized policy actions",
               "global_optimality": "not certified", "realized_counterfactual_profit": None,
               "coverage": "unknown; inherited nominal wild-bootstrap calibration",
               "randomness": "none added; saved cohort and preprocessing, frozen band, deterministic full-batch optimizer"}
    write_provenance(output / "summary.json", summary)
    (output / "EXPERIMENT.md").write_text(_documentation(config, source, config_path), encoding="utf-8")
    record = {"family": config["family"], "revision": "affine-wild-personalized-policy",
              "config": file_record(config_path), "fit_state": file_record(fit_path),
              "fit_provenance": file_record(fit_run / "provenance.json"),
              "residuals": file_record(source / "residuals.npz"), "references": references,
              "spline_cache": file_record(resolve(config["spline_cache"]) / "manifest.json"),
              "sources": [file_record(p) for p in [Path(__file__), ROOT / "MATH.md",
                  ROOT / "scratch/plot_generali_profit_residuals.py", ROOT / "src/objective/banded.py",
                  ROOT / "src/optimization/solvers.py", ROOT / "src/optimization/base.py",
                  ROOT / "src/experiments/envelopes/fitting.py", ROOT / "src/data/full_monotone_spline_cache.py"]],
              "outputs": [file_record(p) for p in sorted(output.iterdir())]}
    write_provenance(output / "provenance.json", record)
    print(f"Saved averaged wild bands and personalized policy results to {output}", flush=True)


def _documentation(config, source, config_path):
    return (
        "# Generali affine wild band: context averages and downstream policies\n\n"
        "The existing wild-linear fit is replayed exactly: no beta fits, bootstrap draws or "
        "calibration changes. Context curves average every row's original, corrected, lower and "
        "upper profit endpoints over all 715,023 rows at a common action. Transfer occurs before "
        "averaging; the average-context feature vector is never substituted.\n\n"
        "Optimization maximizes the mean customer-specific lower endpoint at each customer's "
        "personalized policy action. It does not optimize the averaged curve or a common price. "
        "The repository BandObjective composes each endpoint with SoftmaxPolicy, and "
        "optimization.solvers.run_first_order_minimize supplies the first-order trust-constr "
        "solver. The matched nominal-policy comparison maximizes the original predictor. "
        "No grid maximum, direct SciPy optimizer or analytical policy optimum is substituted.\n\n"
        "Inherited presentation setup: the exact saved 20,000 optimization rows, model-specific "
        "19-feature policy preprocessing, action interval [-0.1,0.2], constant initial action .08, "
        "and mean predicted acceptance floor 0.8787745289312372. The optimizer budget is 1,000 "
        "steps with gradient tolerance 1e-6. Sample constraints do not guarantee full-population "
        "constraints; both are reported. Optimizer messages/optimality are retained and there is "
        "no global-optimality claim. No new stochastic process or sampling is introduced.\n\n"
        "Every action query uses the exact saved affine residual fit and QR half-width, "
        "precomputed as an action-quadratic norm. This cache and its derivatives are checked "
        "against the existing saved-band evaluator. Positive-part kinks use its zero-derivative "
        "convention. GLM uses the exact artifact's affine logits, verified at all logged actions; "
        "spline acceptance uses the validated full-population cache with analytical derivatives, "
        "constant-left and clipped-tangent-right behavior. Rendering grids are visualization "
        "only; neither response nor band is linearly interpolated for optimization.\n\n"
        "The GLM historical reference folder has a synthetic-tail name, but only its recorded "
        "cohort, policy preprocessing and constraint are reused. No support or synthetic-tail "
        "penalty enters the wild objective. All policy actions originate from the newly saved "
        "repository optimizer endpoints. Full-population summaries evaluate each of those "
        "actions directly, not the mean-action curve. Historical density is conditioned on "
        "the displayed interval; outside counts are recorded.\n\n"
        "Reported profits at new prices are model-based predictions, residual-adjusted means "
        "and nominal lower-bound values; actual counterfactual profit and band coverage remain "
        "unknown. These are expected-profit bands, not individual-outcome prediction intervals. "
        "The linear basis, high leverage and repeated-ID limitations from the fitted band remain.\n\n"
        "Regenerate with `python scratch/plot_generali_profit_residuals.py "
        f"--optimize-wild {config_path.resolve()} --replay {source.resolve()} "
        "--output-dir <fresh-directory>`.\n")
