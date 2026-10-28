"""Saved Generali wild bands by feature norm and historical-action cohort.

Study adapter for the existing residual driver: no fitting, draws or optimization.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.dataset_metadata import DATASET_PATH
from data.full_monotone_spline_cache import ShardedMonotoneSplineCache
from data.loader import load_x_frame
from experiments.envelopes.plotting import _STYLE, _slice_indices
from experiments.provenance import file_record, file_sha256, write_provenance
from reporting.real_data import resolve_real_data_model
from scratch.generali_residual_wild import _render_band_curves
from scratch.generali_wild_policy import _FrozenFibers, _Responses, _verify_fibers

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"glm": "GLM", "spline_xgb": "Spline + XGBoost"}


def _action_groups(actions, z, specification):
    """Select inclusive historical-action cohorts and their stable median norms."""
    low, middle = specification["centers"]
    half = specification["half_width"]
    threshold = float(np.quantile(actions, specification["high_quantile"]))
    cuts = [("near_zero", "Historical a near 0", low-half, low+half),
            ("near_0p1", "Historical a near 0.1", middle-half, middle+half),
            ("high_action", f"Highest {100*(1-specification['high_quantile']):g}% historical a", threshold, float(actions.max()))]
    groups = []
    for key, label, left, right in cuts:
        positions = np.flatnonzero((actions >= left) & (actions <= right))
        if len(positions) < 3:
            raise ValueError(f"Too few rows in {key}.")
        representative = int(positions[_slice_indices(z[positions])[1]])
        groups.append({"key": key, "label": label, "lower": left, "upper": right,
                       "positions": positions, "representative": representative})
    return groups


def _endpoints(response, fibers, positions, actions):
    parts = fibers.evaluate(positions, actions)
    prediction = response.evaluate(positions, actions)["center"]
    return {**parts, "prediction": prediction,
            "corrected_prediction": prediction+parts["residual_fit"],
            "lower": prediction-parts["radius_lower"],
            "upper": prediction+parts["radius_upper"]}


def _group_plots(output, curves, groups, *, action_bounds=None):
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        fig.suptitle("Mean customer-specific envelopes by historical action · nominal 95%")
        for row, (name, label) in enumerate(MODELS.items()):
            for column, group in enumerate(groups):
                ax = axes[row, column]
                data = curves[(curves.model == name) & (curves.group == group["key"])]
                ax.fill_between(data.action, data.lower, data.upper, alpha=.25, color=f"C{row}", label="Mean transferred envelope")
                ax.plot(data.action, data.prediction, color=f"C{row}", label="Mean frozen profit prediction")
                ax.plot(data.action, data.corrected_prediction, "--", color="C3", label="Mean prediction + fitted residual")
                ax.set(title=f"{label} · {group['label']}\nn = {len(group['positions']):,}",
                       xlabel="Queried price change, a", ylabel="Mean profit per row")
                if action_bounds is not None:
                    ax.set_xlim(*action_bounds)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3)
        fig.savefig(output / "historical_action_mean_bands.pdf", format="pdf")
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for row, (name, label) in enumerate(MODELS.items()):
            for column, (key, noun) in enumerate([("width", "Residual half-width"), ("radius_lower", "Lower penalty: prediction − lower endpoint")]):
                ax = axes[row, column]
                for group in groups:
                    data = curves[(curves.model == name) & (curves.group == group["key"])]
                    ax.plot(data.action, data[key], label=group["label"])
                ax.set(title=f"{label}\n{noun}", xlabel="Queried price change, a", ylabel="Mean profit units")
                if action_bounds is not None:
                    ax.set_xlim(*action_bounds)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3)
        fig.savefig(output / "historical_action_widths_and_penalties.pdf", format="pdf")
        plt.close(fig)


def render_saved_group_figures(source, output):
    """Render verified saved cohort means without reevaluating any customer."""
    provenance = json.loads((source / "provenance.json").read_text())
    inputs = [file_record(source / "provenance.json")]

    def checked(name):
        record = next(item for item in provenance["outputs"] if Path(item["path"]).name == name)
        path = Path(record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Changed group report input: {path}")
        inputs.append(file_record(path))
        return path

    curves = pd.read_csv(checked("historical_action_mean_curves.csv"))
    summary = json.loads(checked("summary.json").read_text())
    high = 100*(1-summary["config"]["action_groups"]["high_quantile"])
    labels = {"near_zero": "Historical a near 0", "near_0p1": "Historical a near 0.1",
              "high_action": f"Highest {high:g}% historical a"}
    with np.load(checked("cohort_rows.npz"), allow_pickle=False) as rows:
        groups = [{"key": key, "label": label, "positions": rows[key]} for key, label in labels.items()]
    output.mkdir(parents=True, exist_ok=False)
    _group_plots(output, curves, groups, action_bounds=summary["config"].get("display_action_bounds_override"))
    write_provenance(output / "provenance.json", {
        "operation": "render saved group means with wrapped titles; no evaluation, fits, draws or optimization",
        "inputs": inputs, "sources": [file_record(Path(__file__)), file_record(ROOT / "src/experiments/envelopes/plotting.py")],
        "outputs": [file_record(path) for path in sorted(output.iterdir())]})


def _support_diagnostics(arrays, state, frame, selections, optimization_rows):
    """Describe observed numeric-feature support, excluding the query's customer ID."""
    z, norms = state["X"], np.linalg.norm(state["X"], axis=1)
    records, features = [], []
    raw_names = [str(name).removeprefix("z:") for name in state["feature_names"][1:-1]]
    for selection, position in selections:
        difference = z-z[position]
        distances = np.einsum("ij,ij->i", difference, difference)
        distances[arrays["customer_id"] == arrays["customer_id"][position]] = np.inf
        nearest = int(np.argmin(distances))  # Neighbour lookup, not policy/action optimization.
        record = {"selection": selection, "source_row": int(arrays["row_indices"][position]),
                  "feature_norm": norms[position], "norm_percentile": 100*np.mean(norms <= norms[position]),
                  "historical_action": arrays["a"][position], "logged_leverage": state["leverage"][position],
                  "nearest_other_customer_row": int(arrays["row_indices"][nearest]),
                  "nearest_other_customer_distance": float(np.sqrt(distances[nearest])),
                  "in_calibration_cohort": True,
                  "in_optimization_cohort": bool(arrays["row_indices"][position] in optimization_rows)}
        records.append(record)
        for name in raw_names:
            values, value = frame[name].to_numpy(float), float(frame.iloc[position][name])
            features.append({"selection": selection, "source_row": record["source_row"], "feature": name,
                             "value": value, "empirical_midrank_percentile": 100*(np.mean(values < value)+.5*np.mean(values == value)),
                             "population_min": values.min(), "population_median": np.median(values),
                             "population_q99": np.quantile(values, .99), "population_max": values.max()})
    return records, features


def run_wild_slices(arrays, output, source, config_path, *, action_bounds=None):
    config = json.loads(config_path.read_text())
    if config["family"] != "glm-spline-profit-dispersion":
        raise ValueError("Keep the existing comparison family.")
    if action_bounds is not None:
        if not np.isfinite(action_bounds).all() or action_bounds[0] >= action_bounds[1]:
            raise ValueError("Display bounds must be finite and increasing.")
        config["action_bounds"] = list(action_bounds)
        config["display_action_bounds_override"] = list(action_bounds)
    resolve = lambda value: (config_path.parent / value).resolve()
    fit_run = resolve(config["fit_run"])
    provenance = json.loads((fit_run / "provenance.json").read_text())
    inputs = [file_record(fit_run / "provenance.json"), file_record(config_path)]

    def checked(record, *, current_path=None):
        path = Path(record["path"]) if current_path is None else current_path
        actual = file_record(path)
        if actual["sha256"] != record["sha256"]:
            raise ValueError(f"Changed saved input: {path}")
        inputs.append(actual)
        return path

    def fitted(name):
        return checked(next(item for item in provenance["outputs"] if Path(item["path"]).name == name))

    if checked(provenance["input_residuals"]) != (source / "residuals.npz").resolve():
        raise ValueError("Use the fitted band's original residual cohort.")
    if file_sha256(DATASET_PATH) != provenance["dataset"]["sha256"]:
        raise ValueError("Generali dataset changed.")
    with np.load(fitted("fit_state.npz"), allow_pickle=False) as data:
        state = dict(data)
    with np.load(fitted("band_curves.npz"), allow_pickle=False) as data:
        original_curves = dict(data)
    with np.load(checked(provenance["customer_curves"]), allow_pickle=False) as data:
        norm_examples = dict(data)
    np.testing.assert_array_equal(state["row_indices"], arrays["row_indices"])
    np.testing.assert_array_equal(state["actions"], arrays["a"])
    np.testing.assert_array_equal(norm_examples["selection_positions"], _slice_indices(state["X"]))
    fit_config = json.loads(fitted("resolved_config.json").read_text())
    if action_bounds is None:
        _render_band_curves(output, arrays, norm_examples, original_curves,
                           fit_config["bootstrap"]["delta"], quantities=("profit",),
                           filename_prefix="customer_norm")

    frame = load_x_frame("xgb", row_indices=arrays["row_indices"])
    models = {"glm": resolve_real_data_model("glm"), "spline_xgb": resolve_real_data_model("xgb")}
    for name, model in models.items():
        for key in ["acceptance_artifact", "loss_artifact"]:
            if model.provenance()[key]["sha256"] != provenance["models"][name][key]["sha256"]:
                raise ValueError(f"Changed frozen predictor: {name}/{key}")
    artifact = models["glm"].acceptance_artifact
    query = frame.copy()
    query["U"] = 0.
    model_frame = artifact.model_frame(query)
    logit = artifact.model.decision_function(model_frame)
    slope = float(artifact.model.coef_[0, list(model_frame.columns).index("U")])
    del query, model_frame
    cache = ShardedMonotoneSplineCache(resolve(config["spline_cache"]))
    from scratch.plot_generali_profit_residuals import _validate_cache
    _validate_cache(cache, arrays["row_indices"], provenance["dataset"], provenance["models"]["spline_xgb"],
                    provenance["models"]["spline_xgb"]["spline_anchor_weights"])
    inputs.append(file_record(cache.cache_dir / "manifest.json"))
    groups = _action_groups(arrays["a"], state["X"], config["action_groups"])
    selected = np.array([group["representative"] for group in groups])
    logged = arrays["a"][selected]
    bounds = [min(config["action_bounds"][0], float(logged.min())),
              max(config["action_bounds"][1], float(logged.max()))]
    if action_bounds is not None:
        bounds = list(action_bounds)
    grid = np.unique(np.r_[np.linspace(*bounds, config["action_grid_count"]), logged, config["comparison_action"], 0., .2])
    examples = {"selection_positions": selected, "row_indices": arrays["row_indices"][selected], "action_grid": grid}
    selections = list(zip(["Smallest norm", "Median norm", "Largest norm"], norm_examples["selection_positions"]))
    selections += [(group["label"], group["representative"]) for group in groups]
    policy_run = resolve(config["policy_run"])
    policy_provenance = json.loads((policy_run / "provenance.json").read_text())
    inputs.append(file_record(policy_run / "provenance.json"))
    record = next(p for p in policy_provenance["outputs"] if Path(p["path"]).name == "glm_wild_lower_policy.npz")
    with np.load(checked(record, current_path=policy_run / "glm_wild_lower_policy.npz"), allow_pickle=False) as data:
        optimization_rows = data["optimization_row_indices"]
    support, features = _support_diagnostics(arrays, state, frame, selections, optimization_rows)
    pd.DataFrame(support).to_csv(output / "selected_customer_support.csv", index=False)
    pd.DataFrame(features).to_csv(output / "selected_customer_features.csv", index=False)
    all_positions = np.array([int(position) for _, position in selections])
    selected_frame = frame.iloc[all_positions].copy()
    selected_frame.insert(0, "source_row", arrays["row_indices"][all_positions])
    selected_frame.insert(1, "selection", [name for name, _ in selections])
    selected_frame.to_csv(output / "selected_customers.csv", index=False)
    norms = np.linalg.norm(state["X"], axis=1)
    coordinate_rows, group_metrics, selected_metrics, slice_values, validations = [], [], [], {}, {}
    norm_values = {}
    norm_grid = np.unique(np.r_[grid, norm_examples["action_grid"]])
    norm_grid = norm_grid[(norm_grid >= bounds[0]) & (norm_grid <= bounds[1])]
    for name in MODELS:
        print(f"Evaluating saved {name} band slices and historical-action cohorts.", flush=True)
        response = _Responses(name, arrays, cache=cache, glm_logit=logit, glm_slope=slope)
        fibers = _FrozenFibers(state["X"], state["triangular"], state[f"{name}_beta"], state[f"{name}_critical"])
        validations[name] = _verify_fibers(state, name, fibers, norm_examples)
        if action_bounds is not None:
            norm_positions = norm_examples["selection_positions"]
            response.prepare_policy_queries(norm_positions)
            norm_parts = [_endpoints(response, fibers, norm_positions, np.full(len(norm_positions), a)) for a in norm_grid]
            norm_values.update({f"{name}_{key}": np.stack([p[key] for p in norm_parts], axis=1) for key in norm_parts[0]})
            # Check the overlap against the original saved curves; the spline
            # cache has the original report's documented storage tolerance.
            retained = (norm_examples["action_grid"] >= bounds[0]) & (norm_examples["action_grid"] <= bounds[1])
            indices = np.searchsorted(norm_grid, norm_examples["action_grid"][retained])
            for key in ("residual_fit", "width", "radius_lower", "radius_upper"):
                np.testing.assert_allclose(norm_values[f"{name}_{key}"][:, indices],
                                           original_curves[f"{name}_{key}"][:, retained], rtol=2e-11, atol=1e-8)
            np.testing.assert_allclose(norm_values[f"{name}_prediction"][:, indices],
                                       original_curves[f"{name}_prediction"][:, retained], rtol=2e-6, atol=1e-5)
        response.prepare_policy_queries(selected)
        plotted = [_endpoints(response, fibers, selected, np.full(len(selected), a)) for a in grid]
        for key in plotted[0]:
            slice_values[f"{name}_{key}"] = np.stack([parts[key] for parts in plotted], axis=1)
        for label, position in selections:
            for where, action in [("common_action", config["comparison_action"]), ("historical_action", arrays["a"][position])]:
                parts = _endpoints(response, fibers, np.array([position]), np.array([action]))
                selected_metrics.append({"model": name, "selection": label, "source_row": int(arrays["row_indices"][position]),
                                         "evaluation": where, "action": action,
                                         **{key: float(value[0]) for key, value in parts.items()}})
                if where == "historical_action":
                    np.testing.assert_allclose(parts["prediction"][0], arrays[f"{name}_profit"][position], atol=1e-7, rtol=2e-9)
        for group in groups:
            positions = group["positions"]
            response.prepare_policy_queries(positions)
            fixed_actions = np.full(len(positions), config["comparison_action"])
            fixed = _endpoints(response, fibers, positions, fixed_actions)
            group_metrics.append({"model": name, "group": group["key"], "n": len(positions),
                "historical_lower": group["lower"], "historical_upper": group["upper"],
                "mean_historical_action": np.mean(arrays["a"][positions]),
                "median_feature_norm": np.median(norms[positions]), "norm_q95": np.quantile(norms[positions], .95),
                "fraction_in_population_top_1pct_norm": np.mean(norms[positions] >= np.quantile(norms, .99)),
                "optimization_rows": int(np.isin(arrays["row_indices"][positions], optimization_rows).sum()),
                "comparison_action": config["comparison_action"], "mean_width": np.mean(fixed["width"]),
                "median_width": np.median(fixed["width"]), "width_q95": np.quantile(fixed["width"], .95),
                "mean_lower_penalty": np.mean(fixed["radius_lower"]),
                "mean_prediction": np.mean(fixed["prediction"]), "mean_lower_endpoint": np.mean(fixed["lower"])})
            for action in grid:
                parts = _endpoints(response, fibers, positions, np.full(len(positions), action))
                coordinate_rows.append({"model": name, "group": group["key"], "action": action,
                                        **{key: np.mean(value) for key, value in parts.items()}})
            print(f"{name}/{group['key']}: all {len(positions):,} rows evaluated.", flush=True)
    _render_band_curves(output, arrays, examples, slice_values, fit_config["bootstrap"]["delta"],
                        labels=[group["label"] for group in groups], quantities=("profit",),
                        action_bounds=action_bounds, filename_prefix="historical_action_customer")
    if action_bounds is not None:
        expanded_examples = {**norm_examples, "action_grid": norm_grid}
        _render_band_curves(output, arrays, expanded_examples, norm_values, fit_config["bootstrap"]["delta"],
                            quantities=("profit",), action_bounds=action_bounds, filename_prefix="customer_norm")
        np.savez_compressed(output / "norm_slice_curves.npz", action_grid=norm_grid,
                            row_indices=norm_examples["row_indices"], selection_positions=norm_examples["selection_positions"], **norm_values)
    np.savez_compressed(output / "historical_action_slice_curves.npz", **examples, **slice_values)
    np.savez_compressed(output / "cohort_rows.npz", **{group["key"]: arrays["row_indices"][group["positions"]] for group in groups})
    curves = pd.DataFrame(coordinate_rows)
    curves.to_csv(output / "historical_action_mean_curves.csv", index=False)
    pd.DataFrame(group_metrics).to_csv(output / "historical_action_summary.csv", index=False)
    pd.DataFrame(selected_metrics).to_csv(output / "selected_customer_band_values.csv", index=False)
    _group_plots(output, curves, groups, action_bounds=action_bounds)
    summary = {"family": config["family"], "config": config, "calibration_rows": len(frame),
               "norm_quantiles": dict(zip(["median", "q95", "q99", "q999", "maximum"], np.quantile(norms, [.5,.95,.99,.999,1]))),
               "groups": group_metrics, "selected_customers": support, "selected_band_values": selected_metrics,
               "band_parity_max_abs_error": validations, "display_action_bounds": bounds,
               "ood_status": "no validated OOD labels; all selected rows are calibration observations; frozen predictor training membership unknown",
               "optimizer": None, "randomness": None, "new_fits_or_bootstrap_draws": False}
    write_provenance(output / "summary.json", summary)
    (output / "EXPERIMENT.md").write_text(_documentation(config_path, source, action_bounds), encoding="utf-8")
    write_provenance(output / "provenance.json", {
        "family": config["family"], "revision": "saved-wild-norm-and-historical-action-slices",
        "inputs": inputs, "dataset": provenance["dataset"], "models": provenance["models"],
        "operation": "saved-band evaluation only; no fitting, bootstrap, sampling or optimization",
        "sources": [file_record(p) for p in [Path(__file__), ROOT / "scratch/generali_residual_wild.py",
            ROOT / "scratch/generali_wild_policy.py", ROOT / "scratch/plot_generali_profit_residuals.py",
            ROOT / "src/data/full_monotone_spline_cache.py", ROOT / "src/experiments/envelopes/wild.py",
            ROOT / "src/experiments/envelopes/plotting.py"]],
        "outputs": [file_record(path) for path in sorted(output.rglob("*")) if path.is_file()]})
    print(f"Saved wild-band cohort exploration to {output}", flush=True)


def _documentation(config_path, source, action_bounds=None):
    override = "" if action_bounds is None else f" --wild-action-bounds {action_bounds[0]:g} {action_bounds[1]:g}"
    norm_mode = ("Norm slices replay their original saved coordinates and plotting source exactly."
                 if action_bounds is None else
                 f"Both norm and historical-action plots evaluate the existing functions on [{action_bounds[0]:g},{action_bounds[1]:g}]. "
                 "Original customers, coefficients, critical values, observed points and styles are preserved. "
                 "The original norm-grid coordinates are retained inside this range and checked against the saved curves. "
                 "Only displayed queries change; policy bounds and optimizer outputs remain unchanged.")
    return (
        "# Saved Generali wild bands: norm and historical-action slices\n\n"
        "This replays the existing full-data affine beta/QR/critical values and frozen predictors. "
        "No new fit, bootstrap draw, random sampling, optimizer action or optimum is produced. "
        f"{norm_mode}\n\n"
        "Historical action groups use inclusive [-0.005,0.005], [0.095,0.105], and the empirical "
        "95th percentile through the maximum logged action. All qualifying rows contribute to each "
        "mean curve; transfer each customer's endpoints before averaging. A group is defined by "
        "its logged action, while the horizontal axis varies the queried counterfactual action. "
        "These descriptive cohorts are not randomized and differ in feature composition.\n\n"
        "The individual action-group panels select each group's stable median feature norm using "
        "the original wild-report helper. They are illustrative individuals, not group-mean contexts. "
        "Only a customer's single logged outcome is marked. Group-mean plots have no purported "
        "observed counterfactual outcome. All panels retain independent profit axes.\n\n"
        "Norms and nearest-neighbour distances use the original 11 GLM-whitened numeric features, "
        "excluding action, outcomes and categorical encodings. Neighbours exclude all records with "
        "the query customer's ID. Empirical raw-feature ranks use midranks for ties. These describe "
        "extremeness and isolation in the observed numeric feature representation, not a calibrated "
        "OOD detector or full mixed-feature support. All selected customers were included in beta "
        "calibration. Frozen predictor training/holdout membership is unknown. Saved optimizer cohort "
        "membership is reported by replaying its verified row IDs, without optimizing.\n\n"
        "Residual half-width is w=c*sqrt(phi.T (Phi.T Phi)^(-1) phi); the transferred lower penalty "
        "is max(w-phi.T beta,0), and the upper radius is max(w+phi.T beta,0). They are distinct. "
        "Historical-action leverage is evaluated at the logged action; width comparisons also use "
        "one common action 0.1 to remove query-action differences. A large norm alone does not "
        "determine leverage because the full design covariance and action matter.\n\n"
        "GLM logits and saved exact spline coefficients are evaluated directly. Fixed-cohort spline "
        "knots are cached and validated against the original evaluator. Below action 0 the spline "
        "acceptance is constant; above 0.16 it uses a clipped tangent. Display points only render "
        "curves, never select actions. Nominal 95% mean-function coverage remains unverified; these "
        "are not individual-outcome prediction intervals. Existing runs remain unchanged.\n\n"
        f"Regenerate: `python scratch/plot_generali_profit_residuals.py --explore-wild {config_path.resolve()} "
        f"--replay {source.resolve()}{override} --output-dir <fresh-directory>`.\n")
