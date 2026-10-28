"""Replay action-power bands and repository policies into paired diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.paths import results_root
from experiments.provenance import file_record, write_json_atomic


LABELS = {"affine": "Affine", "action_a2": "+ a²", "action_a2_a4": "+ a² + a⁴"}


def _common_action_sums(fibers, actions):
    """Sum exact per-context widths and derivatives for a common action array."""
    _, g, q = fibers.prepared
    coordinate, scale = fibers.action_coordinate(actions)
    values = np.polynomial.polynomial.polyval(coordinate, q.T)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Nonpositive or nonfinite cached query geometry.")
    geometry = np.sqrt(values)
    derivative = np.polynomial.polynomial.polyval(
        coordinate, (q[:, 1:]*np.arange(1, q.shape[1])).T)/scale
    width = fibers.critical*geometry
    fitted = np.polynomial.polynomial.polyval(coordinate, g.T)
    lower, upper = np.maximum(width-fitted, 0), np.maximum(width+fitted, 0)
    return {
        "mean_half_width": width.sum(axis=0),
        "mean_geometry": geometry.sum(axis=0),
        "half_width_slope_per_percentage_point":
            (.01*fibers.critical*derivative/(2*geometry)).sum(axis=0),
        "mean_residual_fit": fitted.sum(axis=0),
        "mean_lower_penalty": lower.sum(axis=0),
        "mean_upper_penalty": upper.sum(axis=0),
        "mean_profit_envelope_width": (lower+upper).sum(axis=0),
    }


def _width_curves(inputs, basis, state, grid, batch_size):
    from scratch.generali_residual_basis_study import _FeatureFibers
    fibers = _FeatureFibers(inputs, basis, state["beta"], state["triangular"], state["critical"])
    totals = {}
    for first in range(0, len(inputs), batch_size):
        positions = np.arange(first, min(first+batch_size, len(inputs)))
        fibers.prepare(positions)
        # Independent exact QR checks across the full displayed action domain.
        if first == 0:
            probe = np.linspace(grid[0], grid[-1], len(positions))
            actual, direct = fibers.evaluate(positions, probe), fibers.direct(positions, probe)
            for key in direct:
                np.testing.assert_allclose(actual[key], direct[key], rtol=2e-7, atol=1e-6)
        for key, values in _common_action_sums(fibers, grid).items():
            totals[key] = totals.get(key, np.zeros(len(grid)))+values
        if first == 0 or first+batch_size >= len(inputs):
            print(f"{basis.kind}: width contexts {min(first+batch_size, len(inputs))}/{len(inputs)}", flush=True)
    frame = pd.DataFrame({"action": grid, **{k:v/len(inputs) for k,v in totals.items()}})
    frame["mean_residual_full_width"] = 2*frame["mean_half_width"]
    frame["critical"] = float(state["critical"])
    return frame


def _action_metrics(actions, reference, calibration):
    mean, median = float(np.mean(calibration)), float(np.median(calibration))
    low, high = np.quantile(calibration, [.05, .95])
    result = {"n": len(actions), "mean_action": float(np.mean(actions)),
              "std_action": float(np.std(actions)),
              "mean_shift_from_affine_percentage_points": float(100*np.mean(actions-reference)),
              "fraction_outside_calibration_central90": float(np.mean((actions < low) | (actions > high)))}
    for quantile in [.01, .05, .25, .5, .75, .95, .99]:
        result[f"action_q{quantile:g}"] = float(np.quantile(actions, quantile))
    for label, center in (("mean", mean), ("median", median)):
        distance, original = np.abs(actions-center), np.abs(reference-center)
        result[f"mean_absolute_distance_to_calibration_{label}"] = float(np.mean(distance))
        result[f"rms_distance_to_calibration_{label}"] = float(np.sqrt(np.mean(distance**2)))
        result[f"fraction_closer_to_{label}_than_affine"] = float(np.mean(distance < original-1e-10))
        result[f"fraction_farther_from_{label}_than_affine"] = float(np.mean(distance > original+1e-10))
        result[f"mean_distance_change_to_{label}_percentage_points"] = float(100*np.mean(distance-original))
    return result


def _plot_report(config, output, curves, distributions, calibration):
    report = config["action_power_report"]
    models, bases = config["models"], config["bases"]
    bounds = [report["main_bounds"], report["extended_bounds"]]
    names = {"glm": "GLM", "spline_xgb": "Spline/XGBoost"}
    with plt.rc_context({"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 12,
                         "figure.titlesize": 16, "legend.fontsize": 10,
                         "font.weight": "normal", "axes.titleweight": "normal",
                         "axes.labelweight": "normal"}):
        fig, axes = plt.subplots(2, len(models), figsize=(12, 8), constrained_layout=True, squeeze=False)
        for row, interval in enumerate(bounds):
            for col, model in enumerate(models):
                ax = axes[row, col]
                for kind in bases:
                    data = curves[(curves.model == model) & (curves.basis == kind)
                                  & curves.action.between(*interval)]
                    ax.plot(data.action, data.mean_half_width, label=LABELS[kind])
                ax.set(xlabel="Common price-change action a", ylabel="Mean residual half-width (profit units)",
                       title=f"{names[model]}: {'policy range' if row == 0 else 'wider action range'}")
                ax.legend()
        fig.savefig(output/"average_band_half_width_by_action.pdf", format="pdf")
        plt.close(fig)

        fig, axes = plt.subplots(2, len(models), figsize=(12, 8), constrained_layout=True, squeeze=False)
        for col, model in enumerate(models):
            for kind in bases:
                data = curves[(curves.model == model) & (curves.basis == kind)
                              & curves.action.between(*bounds[0])]
                axes[0, col].plot(data.action, data.half_width_ratio_to_affine, label=LABELS[kind])
                axes[1, col].plot(data.action, data.half_width_slope_per_percentage_point, label=LABELS[kind])
            axes[0, col].set(xlabel="Common price-change action a", ylabel="Mean half-width / affine half-width",
                             title=f"{names[model]}: relative width")
            axes[1, col].set(xlabel="Common price-change action a", ylabel="Half-width change per percentage point",
                             title=f"{names[model]}: signed width slope")
            axes[0, col].legend()
            axes[1, col].legend()
        fig.savefig(output/"band_width_growth_by_action.pdf", format="pdf")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for ax, interval, title in zip(axes, bounds, ["Policy action range", "Wider action range"]):
            for kind in bases:
                data = curves[(curves.model == models[0]) & (curves.basis == kind)
                              & curves.action.between(*interval)]
                ax.plot(data.action, data.mean_geometry, label=LABELS[kind])
            ax.set(xlabel="Common price-change action a", ylabel="Mean geometric factor (half-width / c)", title=title)
            ax.legend()
        fig.savefig(output/"band_geometry_by_action.pdf", format="pdf")
        plt.close(fig)

        fig, axes = plt.subplots(2, len(models), figsize=(12, 8), constrained_layout=True, squeeze=False)
        for col, model in enumerate(models):
            for j, kind in enumerate([*bases, "calibration", "nominal"]):
                data = distributions[(distributions.model == model) & (distributions.series == kind)]
                label = LABELS.get(kind, {"calibration": "Historical calibration actions", "nominal": "Nominal policy"}.get(kind))
                edges = np.r_[data.bin_low.to_numpy(), data.bin_high.iloc[-1]]
                axes[0, col].stairs(data.density, edges, label=label, linestyle="-" if j < len(bases) else "--")
                axes[1, col].plot(data.bin_high, data.cdf_at_high, label=label, linestyle="-" if j < len(bases) else "--")
            axes[0, col].set(xlabel="Personalized price-change action a", ylabel="Density (full distribution normalization)",
                             title=f"{names[model]}: policy action distributions")
            axes[1, col].set(xlabel="Personalized price-change action a", ylabel="Cumulative fraction", title=names[model])
            axes[0, col].legend()
            axes[1, col].legend()
        fig.savefig(output/"optimized_policy_action_distributions.pdf", format="pdf")
        plt.close(fig)


def write_action_power_report(config, output):
    """Replay saved bands/policies; never fit, bootstrap, or optimize in reporting."""
    from scratch.generali_residual_basis_study import _Basis, _inputs, _load, _verify, _subset_positions
    output = Path(output)
    first = output/config["bases"][0]/config["models"][0]
    contract = json.loads((first/"contract.json").read_text())
    source = _verify(contract["residuals"]).parent
    arrays, inputs, split, _, _ = _inputs(config, source)
    calibration = arrays["a"][split["beta"]]
    report = config["action_power_report"]
    grid = np.unique(np.r_[np.linspace(*report["main_bounds"], report["grid_count"]),
                           np.linspace(*report["extended_bounds"], report["grid_count"]),
                           [-.1, 0., .05, .1, .15, .2, .5, 1., 2.],
                           np.mean(calibration), np.median(calibration)])
    all_curves, sources, actions, outcomes, checks = [], [], {}, {}, []
    paired_hash = None
    for kind in config["bases"]:
        for name in config["models"]:
            folder = output/kind/name
            summary = json.loads((folder/"summary.json").read_text())
            saved_contract = json.loads((folder/"contract.json").read_text())
            for record in saved_contract["sources"]:
                _verify(record)
            _verify(saved_contract["config_file"])
            state = _load(_verify(summary["fit"]["artifact"]))
            policy = _load(_verify(summary["policy"]))
            np.testing.assert_array_equal(state["row_indices"], arrays["row_indices"])
            np.testing.assert_array_equal(policy["row_indices"], arrays["row_indices"])
            np.testing.assert_array_equal(state["beta_row_indices"], arrays["row_indices"][split["beta"]])
            for stage, positions in split.items():
                np.testing.assert_array_equal(state[f"{stage}_positions"], positions)
            multiplier_hash = summary["fit"]["full_row_multiplier_sha256"]
            if paired_hash is not None and paired_hash != multiplier_hash:
                raise ValueError("Multiplier draws are not paired across all variants.")
            paired_hash = multiplier_hash
            if len(state["statistics"]) != config["bootstrap"]["count"]:
                raise ValueError("Incomplete bootstrap calibration.")
            if np.any((policy["actions"] < config["policy"]["action_bounds"][0])
                      | (policy["actions"] > config["policy"]["action_bounds"][1])):
                raise ValueError("Saved policy action violates its declared bounds.")
            check = {"basis": kind, "model": name, "optimizer_success": summary["optimizer"]["success"],
                     "optimizer_message": summary["optimizer"]["message"],
                     "full_row_multiplier_sha256": multiplier_hash}
            for metric in summary["metrics"]:
                if metric["policy"] == "wild_lower":
                    check[f"{metric['cohort']}_acceptance_minus_floor"] = metric["mean_acceptance"]-summary["acceptance_floor"]
            if kind == "affine" and config.get("affine_reference"):
                old_folder = results_root()/config["affine_reference"]/kind/name
                old_summary = json.loads((old_folder/"summary.json").read_text())
                old_state = _load(_verify(old_summary["fit"]["artifact"]))
                old_policy = _load(_verify(old_summary["policy"]))
                for field in ["row_indices", "beta_row_indices", "draw_seeds"]:
                    np.testing.assert_array_equal(state[field], old_state[field])
                for field in ["beta", "triangular", "critical", "statistics"]:
                    np.testing.assert_allclose(state[field], old_state[field], rtol=2e-10, atol=1e-9)
                np.testing.assert_array_equal(policy["optimization_row_indices"], old_policy["optimization_row_indices"])
                check["saved_affine_fit_reproduced"] = True
                check["max_action_difference_from_saved_affine"] = float(np.max(np.abs(policy["actions"]-old_policy["actions"])))
                sources.extend([old_summary["fit"]["artifact"], old_summary["policy"]])
            checks.append(check)
            basis = _Basis.restore(state)
            cache_path, metadata_path = folder/"width_by_action.csv", folder/"width_report.json"
            cache_contract = {"band": summary["fit"]["artifact"], "report": report,
                              "source": file_record(Path(__file__))}
            if metadata_path.exists():
                saved = json.loads(metadata_path.read_text())
                if saved["contract"] != cache_contract:
                    raise ValueError("Saved width report inputs or source changed.")
                frame = pd.read_csv(_verify(saved["curves"]))
                np.testing.assert_allclose(frame.action, grid, rtol=0, atol=1e-14)
            else:
                frame = _width_curves(inputs, basis, state, grid, config["evaluation_batch_size"])
                frame["basis"], frame["model"] = kind, name
                frame.to_csv(cache_path, index=False)
                write_json_atomic(metadata_path, {"contract": cache_contract, "curves": file_record(cache_path)})
            all_curves.append(frame)
            sources.extend([summary["fit"]["artifact"], summary["policy"], file_record(folder/"summary.json")])
            actions[name, kind], outcomes[name, kind] = policy["actions"], summary
    curves = pd.concat(all_curves, ignore_index=True)
    baseline = curves[curves.basis == "affine"][["model", "action", "mean_half_width"]].rename(
        columns={"mean_half_width": "affine_half_width"})
    curves = curves.merge(baseline, on=["model", "action"], validate="many_to_one")
    curves["half_width_ratio_to_affine"] = curves.mean_half_width/curves.affine_half_width
    curves.to_csv(output/"width_by_action.csv", index=False)
    # Identical design matrices across predictors must give identical geometry.
    for kind in config["bases"]:
        groups = [curves[(curves.basis == kind) & (curves.model == name)].mean_geometry.to_numpy()
                  for name in config["models"]]
        for values in groups[1:]:
            np.testing.assert_allclose(groups[0], values, rtol=1e-12, atol=1e-14)

    rows, histogram = [], []
    bins = np.linspace(*report["main_bounds"], report["histogram_bins"]+1)
    for name in config["models"]:
        nominal_record = outcomes[name, "affine"]["nominal_reference"]
        nominal = _load(_verify(nominal_record))
        nominal_actions = nominal["actions"][_subset_positions(nominal["row_indices"], arrays["row_indices"])]
        sources.append(nominal_record)
        series = {kind: actions[name, kind] for kind in config["bases"]}
        series.update(calibration=calibration, nominal=nominal_actions)
        for kind, values in series.items():
            counts, _ = np.histogram(values, bins=bins)
            cdf = np.searchsorted(np.sort(values), bins[1:], side="right")/len(values)
            histogram.extend({"model": name, "series": kind, "bin_low": low, "bin_high": high,
                              "count": int(count), "density": count/(len(values)*(high-low)),
                              "cdf_at_high": cumulative, "total_count": len(values)}
                             for low, high, count, cumulative in zip(bins[:-1], bins[1:], counts, cdf))
            if kind == "calibration":
                continue
            rows.append({"model": name, "basis": kind,
                         **_action_metrics(values, actions[name, "affine"], calibration)})
    pd.DataFrame(rows).to_csv(output/"policy_action_comparison.csv", index=False)
    distributions = pd.DataFrame(histogram)
    distributions.to_csv(output/"policy_action_distributions.csv", index=False)
    support = {"n": len(calibration), "mean": float(np.mean(calibration)),
               "median": float(np.median(calibration)), "q05": float(np.quantile(calibration, .05)),
               "q95": float(np.quantile(calibration, .95)),
               "fraction_within_policy_bounds": float(np.mean((calibration >= bins[0]) & (calibration <= bins[-1]))),
               "reference": "Historical actions on the unchanged beta-fitting rows"}
    write_json_atomic(output/"calibration_action_distribution.json", support)
    _plot_report(config, output, curves, distributions, support)
    artifacts = [*output.glob("*.pdf"), output/"width_by_action.csv", output/"policy_action_comparison.csv",
                 output/"policy_action_distributions.csv", output/"calibration_action_distribution.json"]
    write_json_atomic(output/"report_provenance.json", {
        "sources": sources, "report_source": file_record(Path(__file__)),
        "outputs": [file_record(path) for path in artifacts], "n_contexts": len(inputs),
        "coverage": "Unknown; frozen-feature bootstrap approximation, no bias certificate",
        "action_queries": "Exact polynomial evaluations; grid used only for display",
        "policy_actions": "Exact saved repository trust-constr outputs; no reporting optimization",
        "centering_diagnostics": "Marginal descriptive distances, not conditional support certificates"})
    write_json_atomic(output/"validation.json", {
        "status": "complete", "tasks": checks, "completed_tasks": len(checks),
        "n_contexts": len(inputs), "split_counts": {k:len(v) for k,v in split.items()},
        "checks": ["source and saved artifact checksums", "identical retained split memberships",
                   "original paired full-coordinate multipliers", "all-row personalized policy action bounds",
                   "saved affine fit and 20k cohort reproduction", "direct/cache query parity on extended domain",
                   "identical geometry across response models"],
        "outputs": [file_record(path) for path in artifacts]})
