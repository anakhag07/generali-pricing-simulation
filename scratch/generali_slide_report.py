"""Saved Generali/synthetic evidence for the existing residual driver's slide mode."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.full_monotone_spline_cache import ShardedMonotoneSplineCache
from experiments.envelopes.plotting import _STYLE
from experiments.paths import results_root
from experiments.provenance import file_record, file_sha256, write_provenance
from scratch.generali_residual_basis_study import _Basis, _FeatureFibers
from scratch.generali_residual_wild import _render_band_curves
from scratch.generali_wild_policy import _FrozenFibers, _Responses

ROOT = Path(__file__).resolve().parents[1]
MODEL = "spline_xgb"
BASIS_LABELS = {"affine": "Affine", "action_a2": r"+ $a^2$",
                "action_a2_a4": r"+ $a^2+a^4$", "mlp": "MLP"}


class _Evidence:
    def __init__(self):
        self.records = {}

    def read(self, path, expected=None):
        path = Path(path)
        record = file_record(path)
        if expected is not None and record["sha256"] != expected["sha256"]:
            raise ValueError(f"Saved evidence changed: {path}")
        self.records[str(path.resolve())] = record
        return path

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected).read_text())

    def npz(self, path, expected=None):
        with np.load(self.read(path, expected), allow_pickle=False) as data:
            return dict(data)

    def reported(self, folder, name, provenance="provenance.json"):
        record = self.json(folder/provenance)
        expected = next(r for r in record["outputs"] if Path(r["path"]).name == name)
        return self.read(folder/name, expected)


def _cached_csv(path, contract, evaluate):
    metadata = path.with_suffix(".json")
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved["contract"] != contract or file_sha256(path) != saved["artifact"]["sha256"]:
            raise ValueError(f"Cached report query changed: {path}")
        return pd.read_csv(path, float_precision="round_trip")
    frame = evaluate()
    frame.to_csv(path, index=False)
    write_provenance(metadata, {"contract": contract, "artifact": file_record(path)})
    return frame


def _save(fig, path):
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _residual_figures(evidence, folders, figures, data):
    histogram = pd.read_csv(evidence.reported(folders["central95"], "histogram_counts.csv"))
    selection = evidence.json(evidence.reported(folders["central95"], "plot_selection.json"))
    bounds = selection["models"][MODEL]["residual_bounds"]
    edges = np.r_[histogram.left, histogram.right.iloc[-1]]
    assert histogram[MODEL].sum() == selection["models"][MODEL]["retained_rows"]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.stairs(histogram[MODEL], edges, fill=True, color="C1")
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(xlabel="Observed profit − predicted profit", ylabel="Number of records",
           xlim=bounds)
    _save(fig, figures/"01_residual_errors_central95.pdf")
    bins = pd.read_csv(evidence.reported(folders["affine_fit"], "holdout_calibration_bins.csv"))
    bins = bins[bins.model.eq(MODEL)]
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.plot(bins.mean_fitted_residual, bins.mean_observed_residual, "o",
            label="Holdout means (20 equal-count bins)")
    limits = [min(bins.mean_fitted_residual.min(), bins.mean_observed_residual.min()),
              max(bins.mean_fitted_residual.max(), bins.mean_observed_residual.max())]
    ax.plot(limits, limits, "--", color="0.5", label="Agreement")
    ax.set(xlabel="Mean fitted residual", ylabel="Mean observed residual")
    ax.legend()
    _save(fig, figures/"02_affine_residual_fit_holdout.pdf")
    metrics = pd.read_csv(evidence.reported(folders["affine_fit"], "fit_metrics.csv"))
    metrics = metrics[metrics.model.eq(MODEL) & metrics.group.eq("All")].copy()
    metrics.to_csv(data/"affine_residual_rmse.csv", index=False)
    return metrics, selection["models"][MODEL]


def _capture(arrays, state, ranking, customers, fibers, bounds, data):
    np.testing.assert_array_equal(ranking["row_indices"], arrays["row_indices"])
    actions, y, prediction = arrays["a"], arrays["observed_profit"], arrays[f"{MODEL}_profit"]
    parts = fibers.evaluate(np.arange(len(actions)), actions)
    lower, upper = prediction-parts["radius_lower"], prediction+parts["radius_upper"]
    captured = (y >= lower) & (y <= upper)
    residual_captured = np.abs(arrays[f"{MODEL}_residual"]-parts["residual_fit"]) <= parts["width"]
    assert np.all(~residual_captured | captured)
    masks = {"All eligible records": np.ones(len(y), bool),
        "Logged action in [-0.1, 0.2]": (actions >= bounds[0]) & (actions <= bounds[1])}
    for key in ("low", "medium", "high"):
        masks[f"{key.capitalize()} elasticity"] = np.isin(arrays["row_indices"], ranking[key])
    masks.update({"Renewed": arrays["is_churn"] == 0, "Churned": arrays["is_churn"] == 1})
    masks["Three displayed customers"] = np.isin(arrays["row_indices"], customers["row_indices"])
    table = pd.DataFrame([{"group": name, "n": int(mask.sum()),
        "captured": int(captured[mask].sum()), "capture_percent": 100*float(captured[mask].mean()),
        "residual_band_capture_percent": 100*float(residual_captured[mask].mean())}
        for name, mask in masks.items()])
    table.to_csv(data/"observed_profit_capture.csv", index=False)
    positions = customers["selection_positions"]
    pd.DataFrame({"source_row": arrays["row_indices"][positions], "action": actions[positions],
        "observed_profit": y[positions], "lower": lower[positions], "upper": upper[positions],
        "captured": captured[positions]}).to_csv(data/"displayed_customer_capture.csv", index=False)
    np.savez_compressed(data/"observed_profit_capture.npz", row_indices=arrays["row_indices"],
        actions=actions, observed_profit=y, lower=lower, upper=upper, captured=captured,
        residual_band_captured=residual_captured)
    return table


def _wide_means(evidence, folders, arrays, state, fibers, config, data, narrow):
    cache = ShardedMonotoneSplineCache(folders["spline_cache"])
    manifest = evidence.json(folders["spline_cache"]/"manifest.json")
    origin = evidence.json(folders["affine_fit"]/"provenance.json")
    from scratch.plot_generali_profit_residuals import _validate_cache
    _validate_cache(cache, arrays["row_indices"], origin["dataset"], origin["models"][MODEL],
                    origin["models"][MODEL]["spline_anchor_weights"])
    grid = np.unique(np.r_[np.linspace(*config["wide_bounds"], config["wide_grid_count"]),
                           config["main_bounds"], 0., .1])
    contract = {"fit": file_record(folders["affine_fit"]/"fit_state.npz"),
        "residuals": file_record(folders["residuals"]/"residuals.npz"),
        "cache_manifest": file_record(folders["spline_cache"]/"manifest.json"),
        "grid": grid.tolist(), "n": len(arrays["a"]), "model": MODEL}

    def evaluate():
        response = _Responses(MODEL, arrays, cache=cache)
        keys = ("mean_prediction", "mean_corrected_prediction", "mean_lower", "mean_upper")
        totals = np.zeros((len(grid), len(keys)))
        batch = config["batch_size"]
        for first in range(0, len(arrays["a"]), batch):
            positions = np.arange(first, min(first+batch, len(arrays["a"])))
            response.prepare_policy_queries(positions)
            logged = response.evaluate(positions, arrays["a"][positions])["center"]
            np.testing.assert_allclose(logged, arrays[f"{MODEL}_profit"][positions], atol=1e-7, rtol=2e-9)
            for j, action in enumerate(grid):
                actions = np.full(len(positions), action)
                prediction = response.evaluate(positions, actions)["center"]
                parts = fibers.evaluate(positions, actions)
                totals[j] += [prediction.sum(), (prediction+parts["residual_fit"]).sum(),
                             (prediction-parts["radius_lower"]).sum(),
                             (prediction+parts["radius_upper"]).sum()]
            if first % (12*batch) == 0:
                print(f"Mean spline bands: {min(first+batch, len(arrays['a'])):,} contexts", flush=True)
        return pd.DataFrame({"action": grid, **dict(zip(keys, (totals/len(arrays["a"])).T))})

    curves = _cached_csv(data/"mean_band_wide.csv", contract, evaluate)
    for action in (-.1, 0., .1, .2):
        a = curves[np.isclose(curves.action, action, atol=1e-12, rtol=0)].iloc[0]
        b = narrow[np.isclose(narrow.action, action, atol=1e-12, rtol=0)].iloc[0]
        for key in ("mean_prediction", "mean_corrected_prediction", "mean_lower", "mean_upper"):
            np.testing.assert_allclose(a[key], b[key], rtol=2e-11, atol=1e-8)
    return curves


def _mean_plot(curves, bounds, path):
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.fill_between(curves.action, curves.mean_lower, curves.mean_upper, color="C0", alpha=.2,
                    label="Mean transferred envelope (nominal 95%)")
    ax.plot(curves.action, curves.mean_prediction, color="C0", label="Mean frozen prediction")
    ax.plot(curves.action, curves.mean_corrected_prediction, "--", color="C1",
            label="Mean prediction + fitted residual")
    ax.plot(curves.action, curves.mean_lower, color="C2", label="Mean lower endpoint")
    ax.set(xlabel="Common price change, a", ylabel="Mean profit per record", xlim=bounds)
    ax.legend()
    _save(fig, path)


def _original_policies(evidence, folders, config, figures, data):
    summary = evidence.json(evidence.reported(folders["affine_policy"], "summary.json"))
    for name in ("nominal", "wild_lower"):
        evidence.reported(folders["affine_policy"], f"{MODEL}_{name}_policy.npz")
        assert summary["optimizers"][f"{MODEL}_{name}"]["entry_point"] == "optimization.solvers.run_first_order_minimize"
    hist = pd.read_csv(evidence.reported(folders["affine_policy"], "policy_action_histograms.csv"))
    hist = hist[hist.model.eq(MODEL)]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for color, (series, label) in enumerate([
        ("Nominal-profit policy", "Nominal policy"), ("Wild-LCB policy", "Affine lower-band policy"),
        ("Historical actions in display range", "Historical actions (within displayed range)")]):
        selected = hist[hist.series.eq(series)]
        edges = np.r_[selected.left, selected.right.iloc[-1]]
        np.testing.assert_allclose(np.sum(selected.density*np.diff(edges)), 1.)
        if color < 2:
            ax.stairs(selected.density, edges, fill=True, alpha=config["policy_fill_alpha"], color=f"C{color}")
        ax.stairs(selected.density, edges, color=f"C{color}", label=label)
    ax.set(xlabel="Personalized price change, a", ylabel="Action density", xlim=config["main_bounds"])
    ax.legend()
    _save(fig, figures/"06_nominal_and_affine_policy_actions.pdf")
    metrics = pd.read_csv(evidence.reported(folders["affine_policy"], "policy_metrics.csv"))
    metrics[metrics.model.eq(MODEL)].to_csv(data/"original_policy_metrics.csv", index=False)
    write_provenance(data/"original_policy_status.json", {
        key: value for key, value in summary["optimizers"].items() if key.startswith(MODEL)})
    return summary


def _basis_comparison(evidence, folders, arrays, state, config, figures, data):
    curves = pd.read_csv(evidence.reported(folders["action_powers"], "width_by_action.csv", "report_provenance.json"),
                         float_precision="round_trip")
    curves = curves[curves.model.eq(MODEL)].copy()
    studies, policy_arrays, bands = {}, {}, {}
    for kind in BASIS_LABELS:
        folder = folders["trimmed_bases" if kind == "mlp" else "action_powers"]/kind/MODEL
        summary = evidence.json(folder/"summary.json")
        band = evidence.npz(folder/"band.npz", summary["fit"]["artifact"])
        policy = evidence.npz(folder/"policy.npz", summary["policy"])
        assert summary["optimizer"]["entry_point"] == "optimization.solvers.run_first_order_minimize"
        assert len(band["statistics"]) == 1999
        np.testing.assert_array_equal(band["draw_seeds"], state["bootstrap_draw_seeds"])
        np.testing.assert_array_equal(band["row_indices"], policy["row_indices"])
        if bands:
            reference = bands["affine"]
            for key in ("row_indices", "beta_row_indices", "representation_positions", "beta_positions", "holdout_positions"):
                np.testing.assert_array_equal(band[key], reference[key])
            np.testing.assert_array_equal(policy["optimization_row_indices"], policy_arrays["affine"]["optimization_row_indices"])
        studies[kind], policy_arrays[kind], bands[kind] = summary, policy, band
    saved = bands["mlp"]
    positions = saved["original_positions"]
    np.testing.assert_array_equal(arrays["row_indices"][positions], saved["row_indices"])
    inputs = np.column_stack((state["X"][positions], arrays["a"][positions]))
    basis = _Basis.restore(saved)
    fibers = _FeatureFibers(inputs, basis, saved["beta"], saved["triangular"], saved["critical"])
    grid = np.unique(np.r_[np.linspace(*config["main_bounds"], config["mlp_grid_count_per_window"]),
                           np.linspace(*config["wide_bounds"], config["mlp_grid_count_per_window"])])
    contract = {"band": studies["mlp"]["fit"]["artifact"], "grid": grid.tolist(),
                "n": len(inputs), "source": file_record(ROOT/"src/experiments/envelopes/feature_wild.py")}

    def evaluate():
        records = []
        batch = config["batch_size"]
        for j, action in enumerate(grid):
            total = 0.
            for first in range(0, len(inputs), batch):
                ids = np.arange(first, min(first+batch, len(inputs)))
                parts = fibers.values(ids, np.full(len(ids), action))
                total += parts["width"].sum()
                if j == 0 and first == 0:
                    direct = fibers.direct(ids[:32], np.full(32, action))
                    for key in ("width", "residual_fit", "radius_lower", "radius_upper"):
                        np.testing.assert_allclose(parts[key][:32], direct[key], rtol=1e-12, atol=1e-10)
            records.append({"basis": "mlp", "model": MODEL, "action": action,
                            "mean_half_width": total/len(inputs)})
            if j % 20 == 0:
                print(f"Frozen MLP mean width: {j+1}/{len(grid)} action queries over all {len(inputs):,} contexts", flush=True)
        return pd.DataFrame(records)

    mlp = _cached_csv(data/"mlp_mean_width.csv", contract, evaluate)
    merged = pd.concat([curves, mlp], ignore_index=True)
    merged.to_csv(data/"basis_width_curves.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, bounds, label in zip(axes, [config["main_bounds"], config["wide_bounds"]], ["Policy range", "Wider range"]):
        for kind, name in BASIS_LABELS.items():
            selected = merged[merged.basis.eq(kind) & merged.action.between(*bounds)]
            ax.plot(selected.action, selected.mean_half_width, label=name)
        ax.set(xlabel="Common price change, a", ylabel="Mean residual half-width", xlim=bounds, title=label)
        ax.legend()
    _save(fig, figures/"07_action_power_mlp_band_widths.pdf")
    nominal = evidence.npz(Path(studies["affine"]["nominal_reference"]["path"]), studies["affine"]["nominal_reference"])
    indices = np.searchsorted(nominal["row_indices"], saved["row_indices"])
    np.testing.assert_array_equal(nominal["row_indices"][indices], saved["row_indices"])
    bins = np.linspace(*config["main_bounds"], 61)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    distributions = []
    for color, (kind, actions) in enumerate([("nominal", nominal["actions"][indices]),
                                           *[(k, p["actions"]) for k, p in policy_arrays.items()]]):
        counts = np.histogram(actions, bins=bins)[0]
        assert counts.sum() == len(saved["row_indices"])
        density = counts/(len(actions)*np.diff(bins))
        label = "Nominal" if kind == "nominal" else BASIS_LABELS[kind]
        ax.stairs(density, bins, color=f"C{color}", label=label)
        distributions.extend({"policy": kind, "left": low, "right": high, "count": int(count), "density": d}
            for low, high, count, d in zip(bins[:-1], bins[1:], counts, density))
    ax.set(xlabel="Personalized price change, a", ylabel="Action density", xlim=config["main_bounds"])
    ax.legend()
    _save(fig, figures/"08_action_power_mlp_policy_actions.pdf")
    pd.DataFrame(distributions).to_csv(data/"basis_policy_distributions.csv", index=False)
    metrics = pd.DataFrame([metric for study in studies.values() for metric in study["metrics"]
                            if metric["cohort"] == "full"])
    metrics.to_csv(data/"basis_policy_metrics.csv", index=False)
    rmse = pd.DataFrame([{**metric, "basis": kind} for kind, study in studies.items()
        for metric in study["fit"]["fit_metrics"] if metric["stage"] == "holdout" and metric["group"] == "All"])
    rmse.to_csv(data/"basis_holdout_rmse.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    y = np.arange(len(BASIS_LABELS))
    for ax, column, label in zip(axes, ["mean_lower_bound", "mean_predicted_profit"],
                                ["Own-band lower score", "Frozen-model predicted profit"]):
        for offset, policy, legend in [(-.18, "nominal", "Saved nominal policy"),
                                      (.18, "wild_lower", "Policy optimized for this band")]:
            values = [float(metrics[metrics.basis.eq(k) & metrics.policy.eq(policy)][column].iloc[0]) for k in BASIS_LABELS]
            bars = ax.barh(y+offset, values, height=.34, alpha=.35, label=legend)
            ax.bar_label(bars, fmt="%.2f", label_type="center", fontsize=10)
        ax.set(yticks=y, yticklabels=list(BASIS_LABELS.values()), xlabel="Mean profit per record",
               ylabel="Residual basis", title=label)
        ax.invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    _save(fig, figures/"09_action_power_mlp_policy_values.pdf")
    write_provenance(data/"basis_solver_status.json", {kind: {"policy": study["optimizer"],
        "network": study["fit"].get("network"), "nominal": study["nominal_optimizer"]} for kind, study in studies.items()})
    return studies, rmse, metrics


def _synthetic(evidence, folders, config, figures, data):
    from scratch import run_wild_data_location as source
    folder = folders["synthetic_sweep"]
    contract = evidence.json(folder/"run-contract.json")
    summary = evidence.json(folder/"repetitions/rep_000/summary.json")
    assert summary["contract"] == contract
    for name, digest in summary["artifact_sha256"].items():
        evidence.read(folder/"repetitions/rep_000"/name, {"sha256": digest})
    with zipfile.ZipFile(evidence.read(folder/"source_snapshot.zip")) as archive:
        for name, digest in contract["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        original = ast.parse(archive.read("scratch/run_wild_data_location.py").decode())
    current = ast.parse(Path(source.__file__).read_text())
    function = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "band_values")
    assert ast.dump(function(original)) == ast.dump(function(current))
    spec = evidence.json(folder/"sweep-config.json")
    evidence.read(folder/"plot-overrides.json")
    selected_rows = [row for row in summary["rows"] if row["mean"] == config["synthetic_data_center"]]
    pd.DataFrame(selected_rows).to_csv(data/"synthetic_representative_results.csv", index=False)
    for axis, filename in [("n", "10_synthetic_sample_size.pdf"), ("steepness", "11_synthetic_steepness.pdf")]:
        source._plot_sweep_examples(spec, folder, figures, summary["rows"], axis,
                                   means=[config["synthetic_data_center"]])
        (figures/f"{axis}_bands_data_center_{config['synthetic_data_center']:g}.pdf").rename(figures/filename)
    write_provenance(data/"synthetic_solver_provenance.json", {"source_summary": file_record(folder/"repetitions/rep_000/summary.json"),
        "saved_repository_solver_details": {key: value for key, value in summary["details"].items() if key.startswith("mean_10_")},
        "operation": "Replay saved repository optimizer endpoints; no optimizer or grid selection"})
    return contract, selected_rows


def run_mlp_action_report(config_path, output):
    """Replay saved MLP-band policy actions with historical actions and elasticity groups."""
    config = json.loads(config_path.read_text())
    if config["family"] != "glm-spline-profit-dispersion" or config["model"] != MODEL:
        raise ValueError("This report uses the saved Generali spline/XGBoost study.")
    if (output/"provenance.json").exists():
        raise FileExistsError("Preserve the completed action report before regenerating it.")
    evidence = _Evidence()
    evidence.read(config_path)
    folder = results_root()/config["trimmed_bases"]/"mlp"/MODEL
    summary = evidence.json(folder/"summary.json")
    contract = evidence.json(folder/"contract.json")
    assert summary["optimizer"]["entry_point"] == "optimization.solvers.run_first_order_minimize"
    policy = evidence.npz(folder/"policy.npz", summary["policy"])
    band = evidence.npz(folder/"band.npz", summary["fit"]["artifact"])
    evidence.npz(folder/"policy_optimizer.npz", summary["optimizer"]["checkpoint"])
    evidence.npz(folder/"network.npz", summary["fit"]["network"]["checkpoint"])
    # Original absolute source paths and contracts remain immutable.
    for source in contract["sources"]:
        evidence.read(source["path"], source)
    residuals = evidence.npz(results_root()/config["residuals"]/"residuals.npz", contract["residuals"])
    state = evidence.npz(results_root()/config["affine_fit"]/"fit_state.npz", contract["baseline_fit"])
    origin = evidence.json(results_root()/config["affine_fit"]/"provenance.json", contract["baseline_provenance"])
    ranking = evidence.npz(evidence.reported(results_root()/config["elasticity"], "elasticity_ranking.npz"))
    np.testing.assert_array_equal(residuals["row_indices"], ranking["row_indices"])
    np.testing.assert_array_equal(residuals["row_indices"], state["row_indices"])
    positions, rows = band["original_positions"], band["row_indices"]
    np.testing.assert_array_equal(rows, residuals["row_indices"][positions])
    np.testing.assert_array_equal(rows, policy["row_indices"])
    arrays = {k: v[positions] for k, v in residuals.items()}
    history, actions = arrays["a"], policy["actions"]
    bounds = config["main_bounds"]
    assert np.all((actions >= bounds[0]) & (actions <= bounds[1]))
    groups = {"All": np.ones(len(rows), bool), **{
        name.capitalize(): np.isin(rows, ranking[name]) for name in ("low", "medium", "high")}}
    np.testing.assert_array_equal(sum(groups[name] for name in ("Low", "Medium", "High")), 1)
    buckets = np.select([groups[name] for name in ("Low", "Medium", "High")], [0, 1, 2])
    edges = np.linspace(*bounds, 61)
    histograms, statistics = [], []
    for group, mask in groups.items():
        for label, values in (("Historical", history), ("MLP-band policy", actions)):
            selected = values[mask]
            counts = np.histogram(selected, edges)[0]
            histograms.extend({"group": group, "policy": label, "left": low, "right": high,
                "count": int(count), "density": count/(len(selected)*(high-low))}
                for low, high, count in zip(edges[:-1], edges[1:], counts))
            statistics.append({"group": group, "policy": label, "n": len(selected),
                "mean": float(selected.mean()), "minimum": float(selected.min()),
                "q01": float(np.quantile(selected, .01)), "median": float(np.median(selected)),
                "q99": float(np.quantile(selected, .99)), "maximum": float(selected.max()),
                "below_window": int(np.sum(selected < bounds[0])),
                "above_window": int(np.sum(selected > bounds[1])),
                **{f"fraction_ge_{a:g}": float(np.mean(selected >= a)) for a in (.18, .19, .195, .199, .2)},
                "mean_absolute_glm_derivative": float(np.abs(ranking["derivative"][positions][mask]).mean())})
    histogram = pd.DataFrame(histograms)
    statistics = pd.DataFrame(statistics)
    output.mkdir(parents=True, exist_ok=True)
    histogram.to_csv(output/"action_histograms.csv", index=False)
    statistics.to_csv(output/"action_statistics.csv", index=False)
    np.savez_compressed(output/"paired_actions.npz", row_indices=rows, historical=history,
                        mlp_policy=actions, elasticity_bucket=buckets,
                        elasticity_magnitude=np.abs(ranking["derivative"][positions]))

    def draw(ax, group):
        for label in ("Historical", "MLP-band policy"):
            selected = histogram[histogram.group.eq(group) & histogram.policy.eq(label)]
            stats = statistics[statistics.group.eq(group) & statistics.policy.eq(label)].iloc[0]
            shown = 100*(1-(stats.below_window+stats.above_window)/stats.n)
            ax.stairs(selected.density, edges, label=f"{label} ({shown:.1f}% in window)")
        ax.set(xlabel="Price change, a", ylabel="Action density", xlim=bounds)
        ax.legend()

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        draw(ax, "All")
        ax.set_title("Historical and MLP-band policy actions")
        _save(fig, output/"mlp_vs_historical_actions.pdf")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, constrained_layout=True)
        for ax, name in zip(axes, ("Low", "Medium", "High")):
            draw(ax, name)
            ax.set_title(f"{name} elasticity (n={groups[name].sum():,})")
        _save(fig, output/"mlp_vs_historical_actions_by_elasticity.pdf")

    # MATH sections 6 and 7.20: evaluate exact frozen predictions and transferred
    # endpoints at specified actions; these queries never select a new optimum.
    from scratch.plot_generali_profit_residuals import _validate_cache
    cache = ShardedMonotoneSplineCache(results_root()/config["spline_cache"])
    evidence.read(cache.cache_dir/"manifest.json")
    _validate_cache(cache, residuals["row_indices"], origin["dataset"], origin["models"][MODEL],
                    origin["models"][MODEL]["spline_anchor_weights"])
    inputs = np.column_stack((state["X"][positions], history))
    fibers = _FeatureFibers(inputs, _Basis.restore(band), band["beta"], band["triangular"], band["critical"])
    response = _Responses(MODEL, arrays, cache=cache)
    score_groups = {**groups, "Optimization": np.isin(rows, policy["optimization_row_indices"])}
    assert score_groups["Optimization"].sum() == 20000
    scores = []
    choices = [("Historical", history), ("MLP-band policy", actions),
               *[(f"Fixed {a:.2f}", np.full(len(rows), a)) for a in (.10, .15, .18, .20)]]
    for label, values in choices:
        totals = {name: {} for name in score_groups}
        for first in range(0, len(rows), config["batch_size"]):
            ids = np.arange(first, min(first+config["batch_size"], len(rows)))
            p, b = response.evaluate(ids, values[ids]), fibers.values(ids, values[ids])
            metrics = {"action": values[ids], "predicted_profit": p["center"],
                "acceptance": p["acceptance"], "residual_fit": b["residual_fit"],
                "half_width": b["width"], "lower_penalty": b["radius_lower"],
                "lower_bound": p["center"]-b["radius_lower"]}
            if label == "Historical":
                np.testing.assert_allclose(p["center"], arrays[f"{MODEL}_profit"][ids], rtol=2e-9, atol=1e-7)
            for name, mask in score_groups.items():
                for key, value in metrics.items():
                    assert np.isfinite(value).all()
                    totals[name][key] = totals[name].get(key, 0.)+float(value[mask[ids]].sum())
        scores.extend({"group": name, "policy": label, "n": int(score_groups[name].sum()),
                       **{f"mean_{k}": v/score_groups[name].sum() for k, v in total.items()}}
                      for name, total in totals.items())
        print(f"Evaluated {label} over all {len(rows):,} retained records", flush=True)
    scores = pd.DataFrame(scores)
    for label, saved_label in (("Historical", "historical"), ("MLP-band policy", "wild_lower")):
        for group, cohort in (("All", "full"), ("Optimization", "optimization")):
            expected = next(m for m in summary["metrics"] if m["policy"] == saved_label and m["cohort"] == cohort)
            actual = scores[scores.group.eq(group) & scores.policy.eq(label)].iloc[0]
            for key in ("mean_action", "mean_acceptance", "mean_predicted_profit", "mean_lower_bound",
                        "mean_half_width", "mean_residual_fit", "mean_lower_penalty"):
                np.testing.assert_allclose(actual[key], expected[key], rtol=2e-9, atol=1e-7)
    scores.to_csv(output/"fixed_action_diagnostics.csv", index=False)
    write_provenance(output/"summary.json", {"model": MODEL, "n": len(rows),
        "elasticity": "Existing GLM absolute acceptance derivative tertiles; not dimensionless elasticity",
        "elasticity_reference_action": float(ranking["reference_action"]),
        "display_bounds": bounds, "density_denominator": "All retained rows in each group; no tail renormalization",
        "acceptance_floor": summary["acceptance_floor"], "network": summary["fit"]["network"],
        "fit_counts": summary["fit"]["n"], "policy_optimizer": summary["optimizer"],
        "training_metrics": summary["fit"]["fit_metrics"],
        "action_statistics": statistics.to_dict("records"), "scores": scores.to_dict("records"),
        "validation": "Saved full/20k action, profit, acceptance and band means replayed within 1e-7 absolute + 2e-9 relative",
        "new_fits": False, "new_bootstrap_draws": False, "new_optimization": False, "new_randomness": False,
        "fixed_actions": "Prespecified counterfactual diagnostics; no per-customer or policy optimum claimed"})
    shutil.copyfile(config_path, output/"resolved_config.json")
    sources = [Path(__file__), ROOT/"scratch/plot_generali_profit_residuals.py",
               ROOT/"scratch/generali_residual_basis_study.py", ROOT/"scratch/generali_wild_policy.py",
               ROOT/"src/experiments/envelopes/feature_wild.py", ROOT/"src/experiments/envelopes/wild.py",
               ROOT/"src/experiments/envelopes/plotting.py", ROOT/"src/data/full_monotone_spline_cache.py"]
    write_provenance(output/"provenance.json", {"family": config["family"],
        "operation": "Exact saved MLP-band policy replay and fixed-action evaluation",
        "inputs": list(evidence.records.values()), "sources": [file_record(p) for p in sources],
        "outputs": [file_record(p) for p in sorted(output.iterdir()) if p.is_file()]})


def run_slide_report(config_path, output):
    """Generate PDF slide assets and Beamer text from verified saved experiments."""
    config = json.loads(config_path.read_text())
    if config["family"] != "glm-spline-profit-dispersion" or config["model"] != MODEL:
        raise ValueError("This recipe retains the Generali family and spline/XGBoost predictor.")
    if (output/"provenance.json").exists():
        raise FileExistsError("Archive the completed slide report before regenerating it.")
    output.mkdir(parents=True, exist_ok=True)
    figures, data = output/"figures", output/"data"
    figures.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    keys = ("residuals", "central95", "affine_fit", "affine_policy", "elasticity", "action_powers",
            "trimmed_bases", "spline_cache", "synthetic_sweep")
    folders = {key: results_root()/config[key] for key in keys}
    evidence = _Evidence()
    evidence.read(config_path)
    arrays = evidence.npz(evidence.reported(folders["residuals"], "residuals.npz"))
    state = evidence.npz(evidence.reported(folders["affine_fit"], "fit_state.npz"))
    np.testing.assert_array_equal(arrays["row_indices"], state["row_indices"])
    fibers = _FrozenFibers(state["X"], state["triangular"], state[f"{MODEL}_beta"], state[f"{MODEL}_critical"])
    customers = evidence.npz(evidence.reported(folders["elasticity"], "customer_elasticity_curves.npz"))
    ranking = evidence.npz(evidence.reported(folders["elasticity"], "elasticity_ranking.npz"))
    narrow = pd.read_csv(evidence.reported(folders["affine_policy"], "context_averaged_band.csv"))
    narrow = narrow[narrow.model.eq(MODEL)].copy()
    narrow.to_csv(data/"mean_band_policy_range.csv", index=False)
    with plt.rc_context(_STYLE):
        rmse, central = _residual_figures(evidence, folders, figures, data)
        capture = _capture(arrays, state, ranking, customers, fibers, config["main_bounds"], data)
        print(capture.to_string(index=False), flush=True)
        _mean_plot(narrow, config["main_bounds"], figures/"03_mean_wild_band_policy_range.pdf")
        _render_band_curves(figures, arrays, customers, customers, .05,
            labels=["Low elasticity", "Medium elasticity", "High elasticity"], quantities=("profit",),
            action_bounds=config["main_bounds"], filename_prefix="05_customers_by_elasticity",
            models={MODEL: "Spline + XGBoost"}, include_figure_title=False)
        (figures/"05_customers_by_elasticity_profit_bands.pdf").rename(figures/"05_customers_by_elasticity.pdf")
        policies = _original_policies(evidence, folders, config, figures, data)
        wide = _wide_means(evidence, folders, arrays, state, fibers, config, data, narrow)
        _mean_plot(wide, config["wide_bounds"], figures/"04_mean_wild_band_wide_range.pdf")
        studies, basis_rmse, basis_metrics = _basis_comparison(evidence, folders, arrays, state, config, figures, data)
        synthetic, synthetic_rows = _synthetic(evidence, folders, config, figures, data)
    from scratch.generali_slide_text import write_slide_text
    write_slide_text(output, rmse, capture, studies, basis_rmse, basis_metrics, central)
    shutil.copyfile(config_path, output/"resolved_config.json")
    write_provenance(output/"summary.json", {"model": MODEL, "calibration_rows": len(arrays["a"]),
        "capture": capture.to_dict("records"), "original_affine_rmse": rmse.to_dict("records"),
        "basis_holdout_rmse": basis_rmse.to_dict("records"), "original_policy_status": policies["optimizers"],
        "synthetic_data_center": config["synthetic_data_center"], "new_fits": False,
        "new_bootstrap_draws": False, "new_optimization": False, "new_randomness": False})
    source_names = ("generali_slide_report.py", "generali_slide_text.py", "generali_residual_wild.py",
                    "generali_wild_policy.py", "generali_residual_basis_study.py", "run_wild_data_location.py",
                    "plot_generali_profit_residuals.py")
    write_provenance(output/"provenance.json", {"family": config["family"], "operation": "Saved-run slide assembly and deterministic band queries",
        "inputs": list(evidence.records.values()), "sources": [file_record(ROOT/"scratch"/name) for name in source_names],
        "math_sources": [file_record(ROOT/"src/experiments/envelopes"/name) for name in ("wild.py", "feature_wild.py", "plotting.py")],
        "outputs": [file_record(path) for path in sorted(output.rglob("*")) if path.is_file()]})
    print(f"Slide figures and Beamer text saved to {output}", flush=True)
