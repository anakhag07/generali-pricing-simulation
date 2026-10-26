"""Descriptive GLM/spline profit residuals at logged actions (MATH section 6).

Replays the validated full-population spline cache; no model training, sampling,
band fitting, or optimization. Use --replay to render saved residuals afresh.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.dataset_metadata import DATASET_PATH, REQUIRED_DATASET_COLUMNS
from data.full_monotone_spline_cache import ShardedMonotoneSplineCache
from data.loader import eligible_csv_row_indices, load_x_frame
from experiments.paths import results_root
from experiments.provenance import file_record, file_sha256, write_provenance
from reporting.profit_dispersion import exact_spline_acceptance_matrix, predict_loss
from reporting.real_data import resolve_real_data_model

FAMILY = "glm-spline-profit-dispersion"
MODELS = {"glm": "GLM", "spline_xgb": "Spline + XGBoost"}
DEFAULT_CACHE = results_root() / "cache/monotone-spline-xgb-full-v1/sweeps/full-715023-v1-20260812"


def _observed_profit(frame):
    revenue = (1.0 + frame["U"].to_numpy(float)) * frame["X_policy_premium"].to_numpy(float)
    upcoming = frame["X_upcoming_premium"].to_numpy(float)
    churn = frame["is_churn"].to_numpy(float)
    loss = frame["Y_G_Loss"].to_numpy(float)
    if not np.isfinite(np.column_stack((revenue, upcoming, churn, loss))).all():
        raise ValueError("Historical-profit inputs must be finite.")
    if not np.isin(churn, [0, 1]).all():
        raise ValueError("is_churn must be binary.")
    error = float(np.max(np.abs(upcoming - revenue)))
    if not np.allclose(upcoming, revenue, rtol=0, atol=1e-8):
        raise ValueError("Historical premium/action identity failed.")
    return (1 - churn) * (upcoming - loss), error


def _predicted_profit(frame, actions, model, acceptance=None):
    if acceptance is None:
        raw = frame.copy()
        raw["U"] = actions
        artifact = model.acceptance_artifact
        acceptance = np.asarray(artifact.model.predict_proba(artifact.model_frame(raw)))[:, 1]
    acceptance = np.asarray(acceptance, dtype=float)
    loss = predict_loss(model.loss_artifact, frame)
    if acceptance.shape != (len(frame),) or not np.isfinite(acceptance).all():
        raise ValueError("Require one finite acceptance probability per row.")
    if np.any((acceptance < 0) | (acceptance > 1)):
        raise ValueError("Acceptance probability is outside [0, 1].")
    profit = acceptance * ((1 + actions) * frame["X_policy_premium"].to_numpy(float) - loss)
    if not np.isfinite(profit).all():
        raise ValueError("Predicted profit must be finite.")
    return acceptance, loss, profit


def _validate_cache(cache, eligible, dataset, spline_provenance, weights):
    manifest = cache.manifest
    if manifest["source"]["dataset_sha256"] != dataset["sha256"]:
        raise ValueError("Spline cache dataset hash mismatch.")
    if manifest["source"]["xgb_acceptance_sha256"] != spline_provenance["acceptance_artifact"]["sha256"]:
        raise ValueError("Spline cache acceptance-model hash mismatch.")
    if manifest["validation"]["status"] != "passed" or manifest["failure_count"] != 0:
        raise ValueError("Spline cache has not passed validation.")
    if not np.array_equal(cache.eligible_row_indices, eligible):
        raise ValueError("Spline cache population does not match canonical eligible rows.")
    if not np.allclose(manifest["weighting"]["weights_before_zero_floor"], weights, rtol=0, atol=1e-15):
        raise ValueError("Spline cache anchor weights mismatch.")
    index = manifest["eligible_row_indices"]
    if file_sha256(cache.cache_dir / index["path"]) != index["sha256"]:
        raise ValueError("Spline cache row-index checksum mismatch.")
    for shard in cache.shards:
        root = cache.cache_dir / shard["path"]
        for name, record in shard["files"].items():
            if file_sha256(root / name) != record["sha256"]:
                raise ValueError(f"Spline cache checksum mismatch: {root / name}")
        rows = np.load(root / "row_indices.npy", allow_pickle=False)
        if not np.array_equal(rows, eligible[shard["start_position"]:shard["stop_position"]]):
            raise ValueError("Spline cache shard row order mismatch.")


def _evaluate(cache_path):
    raw = pd.read_csv(DATASET_PATH, sep=";", dtype={"id": "string"})
    eligible = np.intersect1d(eligible_csv_row_indices("linear"), eligible_csv_row_indices("xgb"))
    diagnostics = ["U", "X_policy_premium", "X_upcoming_premium", "Y_G_Loss", "is_churn"]
    finite = np.isfinite(raw.iloc[eligible][diagnostics].to_numpy(float)).all(axis=1)
    rows = eligible[finite]
    observed = raw.iloc[rows]
    actual, identity_error = _observed_profit(observed)
    actions = observed["U"].to_numpy(float)
    frame = load_x_frame("xgb", row_indices=rows)
    print(f"Evaluating {len(rows):,} common rows at their historical actions.", flush=True)
    glm = resolve_real_data_model("glm")
    spline = resolve_real_data_model("exact_spline_xgb", eligible_rows=eligible)
    models = {"glm": glm, "spline_xgb": spline}
    model_records = {name: model.provenance() for name, model in models.items()}
    dataset = file_record(DATASET_PATH)
    cache = ShardedMonotoneSplineCache(cache_path)
    print("Verifying spline cache source identities and all payload checksums.", flush=True)
    _validate_cache(cache, eligible, dataset, model_records["spline_xgb"], spline.exact_spline_weights)
    ids = observed["id"].to_numpy(dtype=str)
    if not np.array_equal(cache.customer_ids(rows), ids):
        raise ValueError("Spline cache customer IDs are not aligned.")
    acceptance, _ = cache.pairwise_acceptance_and_derivative(rows, actions)
    arrays = {
        "row_indices": rows, "customer_id": ids, "a": actions,
        "premium": observed["X_policy_premium"].to_numpy(float),
        "upcoming_premium": observed["X_upcoming_premium"].to_numpy(float),
        "is_churn": observed["is_churn"].to_numpy(int),
        "observed_loss": observed["Y_G_Loss"].to_numpy(float), "observed_profit": actual,
    }
    for name, model in models.items():
        p, loss, profit = _predicted_profit(frame, actions, model, acceptance if name == "spline_xgb" else None)
        arrays.update({f"{name}_acceptance": p, f"{name}_loss": loss,
                       f"{name}_profit": profit, f"{name}_residual": actual - profit})
        print(f"{MODELS[name]}: mean residual {np.mean(actual-profit):.6f}", flush=True)

    # Deterministic validation locations cover population order and both tails.
    positions = list(np.linspace(0, len(rows)-1, 24, dtype=int))
    for mask in (actions < 0, (actions >= 0) & (actions <= .16), actions > .16):
        matches = np.flatnonzero(mask)
        if len(matches):
            positions.extend(matches[np.linspace(0, len(matches)-1, 6, dtype=int)])
    check = np.unique(positions)
    grid = np.unique(actions[check])
    selected = np.searchsorted(grid, actions[check])
    direct_acceptance = exact_spline_acceptance_matrix(
        spline.acceptance_artifact, frame.iloc[check], grid, spline.exact_spline_weights, n_jobs=1,
    )[np.arange(len(check)), selected]
    np.testing.assert_allclose(acceptance[check], direct_acceptance, atol=2e-6, rtol=0)
    profit_checks = {}
    for name, model in models.items():
        direct = model.profit_matrix(frame.iloc[check], grid, n_jobs=1)[np.arange(len(check)), selected]
        # The serialized spline has an acceptance tolerance of 2e-6; propagate
        # that to profit units using each customer's absolute predicted margin.
        margin = np.abs((1+actions[check])*arrays["premium"][check]-arrays[f"{name}_loss"][check])
        error = np.abs(arrays[f"{name}_profit"][check]-direct)
        if np.any(error > 2e-6*margin + 1e-8):
            raise ValueError(f"{name} pairwise profit disagrees with the existing evaluator.")
        profit_checks[name] = float(error.max())
    return arrays, {
        "dataset": dataset, "models": model_records,
        "spline_cache": file_record(cache.cache_dir / "manifest.json"),
        "cache_checksums": "all payload files verified against cache manifest",
        "cohort": {"source_rows": len(raw), "common_eligible_rows": len(eligible),
                   "analyzed_rows": len(rows), "excluded_incomplete_rows": len(raw)-len(eligible),
                   "excluded_nonfinite_diagnostics": int((~finite).sum()),
                   "missing_counts_by_required_column": raw[list(REQUIRED_DATASET_COLUMNS)].isna().sum().to_dict()},
        "validation": {"premium_identity_max_abs_error": identity_error,
                       "direct_model_check_source_rows": rows[check].tolist(),
                       "spline_acceptance_max_abs_error": float(np.max(np.abs(acceptance[check]-direct_acceptance))),
                       "profit_max_abs_error": profit_checks},
    }


def _plots_and_summaries(arrays, output, *, central_fraction=1., trim_actions=False, action_renderer="scatter"):
    tail = (1-central_fraction)/2
    action_bounds = np.quantile(arrays["a"], [tail, 1-tail]) if trim_actions else [None, None]
    action_mask = ((arrays["a"] >= action_bounds[0]) & (arrays["a"] <= action_bounds[1])
                   if trim_actions else np.ones(len(arrays["a"]), dtype=bool))
    masks, selections = {}, {}
    for name in MODELS:
        residual = arrays[f"{name}_residual"]
        lower, upper = np.quantile(residual, [tail, 1-tail])
        masks[name] = action_mask & (residual >= lower) & (residual <= upper)
        if not np.any(masks[name]):
            raise ValueError("Central selection contains no observations.")
        selections[name] = {"residual_bounds": [float(lower), float(upper)],
                            "retained_rows": int(masks[name].sum()),
                            "retained_fraction": float(masks[name].mean())}
    selection = {"central_fraction_per_variable": central_fraction,
                 "trim_actions": trim_actions, "action_bounds": list(action_bounds),
                 "rule": "inclusive marginal quantile bounds computed on the full original cohort; intersection when action trimming is enabled",
                 "action_renderer": action_renderer, "models": selections}
    write_provenance(output / "plot_selection.json", selection)
    np.savez_compressed(output / "plot_rows.npz", **{m: arrays["row_indices"][mask] for m, mask in masks.items()})
    all_residuals = np.concatenate([arrays[f"{m}_residual"][masks[m]] for m in MODELS])
    bins = np.histogram_bin_edges(all_residuals, bins=100)
    counts = {m: np.histogram(arrays[f"{m}_residual"][masks[m]], bins=bins)[0] for m in MODELS}
    for m, count in counts.items():
        if count.sum() != masks[m].sum():
            raise ValueError("Histogram lost observations.")
    pd.DataFrame({"left": bins[:-1], "right": bins[1:], **counts}).to_csv(output / "histogram_counts.csv", index=False)
    summaries = []
    for name, label in MODELS.items():
        residual = arrays[f"{name}_residual"]
        summaries.append({"model": label, "n": len(residual),
            "observed_profit_mean": np.mean(arrays["observed_profit"]),
            "predicted_profit_mean": np.mean(arrays[f"{name}_profit"]),
            "residual_mean": np.mean(residual), "residual_std_ddof0": np.std(residual),
            "mae": np.mean(np.abs(residual)), "rmse": np.sqrt(np.mean(residual**2)),
            **{f"residual_q{q:g}": np.quantile(residual, q) for q in (0, .01, .05, .5, .95, .99, 1)}})
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    with plt.rc_context({"axes.titlesize": 14, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)
        densities = []
        displayed_actions = arrays["a"][np.logical_or.reduce(list(masks.values()))]
        extent = (float(displayed_actions.min()), float(displayed_actions.max()),
                  float(all_residuals.min()), float(all_residuals.max()))
        for index, (name, label) in enumerate(MODELS.items()):
            ax = axes[index]
            selected = masks[name]
            if action_renderer == "density":
                count, xedges, yedges = np.histogram2d(
                    arrays["a"][selected], arrays[f"{name}_residual"][selected], bins=120,
                    range=[extent[:2], extent[2:]])
                density = ax.pcolormesh(xedges, yedges, np.ma.masked_equal(count.T, 0),
                                       cmap="viridis", norm=LogNorm(), rasterized=False)
                if int(density.get_array().sum()) != int(selected.sum()):
                    raise ValueError("Density plot lost observations.")
                densities.append(density)
            else:
                ax.scatter(arrays["a"][selected], arrays[f"{name}_residual"][selected], s=1, alpha=.2, linewidths=0, color=f"C{index}")
            ax.axhline(0, color="black", linewidth=.8)
            ax.set(title=label, xlabel="Historical price change, a (fraction)", ylabel="Observed profit − predicted profit")
            ax.tick_params(labelleft=True)
        if densities:
            norm = LogNorm(vmin=1, vmax=max(float(d.get_array().max()) for d in densities))
            for density in densities:
                density.set_norm(norm)
            colorbar = fig.colorbar(densities[0], ax=axes)
            # Matplotlib otherwise rasterizes a continuous colorbar in PDF.
            colorbar.solids.set_rasterized(False)
            colorbar.set_label("Customers per cell", fontsize=12)
            colorbar.ax.tick_params(labelsize=10)
        fig.savefig(output / "residual_vs_action.pdf", format="pdf")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)
        for index, (name, label) in enumerate(MODELS.items()):
            ax = axes[index]
            ax.stairs(counts[name], bins, fill=True, color=f"C{index}")
            ax.axvline(0, color="black", linewidth=.8)
            ax.set(title=label, xlabel="Observed profit − predicted profit", ylabel="Number of customers")
            ax.tick_params(labelleft=True)
        fig.savefig(output / "residual_histograms.pdf", format="pdf")
        plt.close(fig)
    return summaries


def _shape_and_action_diagnostics(arrays, output):
    from scipy.stats import kurtosis, norm, skew

    shape_rows, qq_rows = [], []
    probabilities = np.unique(np.r_[.0001, .0005, np.linspace(.001, .999, 301), .9995, .9999])
    gaussian = norm.ppf(probabilities)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for column, (name, label) in enumerate(MODELS.items()):
        residual = arrays[f"{name}_residual"]
        lo, hi = np.quantile(residual, [.025, .975])
        views = {"Full": np.ones(len(residual), dtype=bool),
                 "Central 95%": (residual >= lo) & (residual <= hi),
                 "Renewed": arrays["is_churn"] == 0, "Churned": arrays["is_churn"] == 1}
        for row, (view, mask) in enumerate(views.items()):
            values = residual[mask]
            shape_rows.append({"model": name, "view": view, "n": len(values),
                               "mean": np.mean(values), "median": np.median(values),
                               "std_ddof0": np.std(values), "skewness": skew(values),
                               "excess_kurtosis": kurtosis(values)})
            if row >= 2:
                continue
            observed = (np.quantile(values, probabilities)-np.mean(values))/np.std(values)
            qq_rows.extend({"model": name, "view": view, "probability": p,
                            "normal_quantile": q, "standardized_residual_quantile": y}
                           for p, q, y in zip(probabilities, gaussian, observed))
            ax = axes[row, column]
            ax.plot(gaussian, observed, ".", color=f"C{column}")
            ax.plot(gaussian, gaussian, "--", color="0.5", label="Normal reference")
            ax.set(title=f"{label}: {view.lower()}", xlabel="Standard normal quantile",
                   ylabel="Standardized residual quantile")
            ax.legend()
    fig.savefig(output / "residual_normal_qq.pdf", format="pdf")
    plt.close(fig)
    pd.DataFrame(shape_rows).to_csv(output / "residual_shape.csv", index=False)
    pd.DataFrame(qq_rows).to_csv(output / "qq_coordinates.csv", index=False)

    # A declared zoom resolves the crowded action region without changing any
    # existing residual plot or excluding tail observations from shape metrics.
    edges = np.linspace(-.1, .3, 41)
    actions = arrays["a"]
    masks = [(actions >= low) & ((actions < high) if j < 39 else (actions <= high))
             for j, (low, high) in enumerate(zip(edges[:-1], edges[1:]))]
    counts = np.array([mask.sum() for mask in masks])
    centers = (edges[:-1]+edges[1:])/2
    in_view = (actions >= edges[0]) & (actions <= edges[-1])
    if counts.sum() != in_view.sum():
        raise ValueError("Action histogram omitted or duplicated observations.")
    rows = []
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
    fig.suptitle("Logged-action diagnostics: −10% to +30% zoom", fontsize=16)
    axes[0].bar(centers, counts, width=np.diff(edges))
    axes[0].set(title=f"{counts.sum():,} of {len(actions):,} customers in view",
                ylabel="Customers per 1% action bin", xlabel="Historical price change, a")
    for ax, (name, label) in zip(axes[1:], MODELS.items()):
        residual = arrays[f"{name}_residual"]
        means, medians, low_quantiles, high_quantiles = [], [], [], []
        for low, high, mask in zip(edges[:-1], edges[1:], masks):
            values = residual[mask]
            q10, median, q90 = np.quantile(values, [.1, .5, .9]) if len(values) else (np.nan,)*3
            mean = np.mean(values) if len(values) else np.nan
            rows.append({"model": name, "action_low": low, "action_high": high,
                         "n": len(values), "mean": mean, "median": median,
                         "q10": q10, "q90": q90,
                         "churn_fraction": np.mean(arrays["is_churn"][mask]) if len(values) else np.nan})
            means.append(mean if len(values) >= 50 else np.nan)
            medians.append(median if len(values) >= 50 else np.nan)
            low_quantiles.append(q10 if len(values) >= 50 else np.nan)
            high_quantiles.append(q90 if len(values) >= 50 else np.nan)
        ax.fill_between(centers, low_quantiles, high_quantiles, alpha=.2, color="C0", label="10th–90th percentiles")
        ax.plot(centers, means, color="C0", label="Mean residual")
        ax.plot(centers, medians, "--", color="C1", label="Median residual")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=label, xlabel="Historical price change, a", ylabel="Observed − predicted profit")
        ax.legend()
    fig.savefig(output / "residual_action_bin_summary.pdf", format="pdf")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "action_bin_summary.csv", index=False)
    return {"action_zoom": [float(edges[0]), float(edges[-1])],
            "rows_in_zoom": int(in_view.sum()), "rows_outside_zoom": int((~in_view).sum()),
            "fraction_actions_0_to_016": float(np.mean((actions >= 0) & (actions <= .16))),
            "binned_curve_min_count": 50, "shape": shape_rows}


def _explore(arrays, output, source):
    from experiments.envelopes.plotting import _slice_indices, _STYLE

    origin = json.loads((source / "provenance.json").read_text())
    while "models" not in origin:
        origin = origin["original_analysis"]
    if file_sha256(DATASET_PATH) != origin["dataset"]["sha256"]:
        raise ValueError("Dataset changed since residual calculation.")
    rows = arrays["row_indices"]
    frame = load_x_frame("xgb", row_indices=rows)
    glm = resolve_real_data_model("glm")
    spline = resolve_real_data_model("exact_spline_xgb", eligible_rows=rows)
    models = {"glm": glm, "spline_xgb": spline}
    model_provenance = {name: model.provenance() for name, model in models.items()}
    for name, record in model_provenance.items():
        for key in ("acceptance_artifact", "loss_artifact"):
            if record[key]["sha256"] != origin["models"][name][key]["sha256"]:
                raise ValueError(f"Changed {name} {key}.")
    processor = glm.acceptance_artifact.preprocessor
    z = processor.transform(frame[list(glm.acceptance_artifact.x_feature_cols)])[
        list(processor.numeric_feature_names_)].to_numpy(float)
    norms = np.linalg.norm(z, axis=1)
    positions = _slice_indices(z)
    labels = ["Smallest norm", "Median norm", "Largest norm"]
    selected = frame.iloc[positions].copy()
    selected.insert(0, "source_row", rows[positions])
    selected.insert(1, "selection", labels)
    selected["feature_norm"] = norms[positions]
    for name in ["a", "observed_profit", "observed_loss", "is_churn",
                 "glm_profit", "glm_residual", "spline_xgb_profit", "spline_xgb_residual"]:
        selected[name] = arrays[name][positions]
    selected.to_csv(output / "representative_customers.csv", index=False)

    # Extend the established slide interval only to include the three logged
    # actions. Predictions are queried directly, never selected as optima.
    actions = arrays["a"][positions]
    grid = np.unique(np.r_[np.linspace(min(-.1, float(actions.min())),
                                      max(.2, float(actions.max())), 401), actions])
    curves = {name: model.profit_matrix(frame.iloc[positions], grid, n_jobs=1)
              for name, model in models.items()}
    check_columns = np.searchsorted(grid, actions)
    for name, curve in curves.items():
        at_logged = curve[np.arange(len(positions)), check_columns]
        margin = np.abs((1+actions)*arrays["premium"][positions]-arrays[f"{name}_loss"][positions])
        if np.any(np.abs(at_logged-arrays[f"{name}_profit"][positions]) > 2e-6*margin+1e-8):
            raise ValueError(f"{name} slice predictions disagree with saved residual inputs.")
    np.savez_compressed(output / "customer_curves.npz", row_indices=rows[positions],
                        selection_positions=positions, action_grid=grid, observed_actions=actions,
                        observed_profit=arrays["observed_profit"][positions],
                        numeric_feature_norms=norms, numeric_features=z[positions],
                        numeric_feature_names=np.array(processor.numeric_feature_names_),
                        **{f"{name}_profit": curve for name, curve in curves.items()})
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, constrained_layout=True)
        for col, (ax, position, label) in enumerate(zip(axes, positions, labels)):
            for j, (name, model_label) in enumerate(MODELS.items()):
                ax.plot(grid, curves[name][col], color=f"C{j}", label=model_label)
                predicted = arrays[f"{name}_profit"][position]
                ax.plot(actions[col], predicted, "o", color=f"C{j}")
                ax.vlines(actions[col], predicted, arrays["observed_profit"][position],
                          color=f"C{j}", linestyles=":")
            ax.plot(actions[col], arrays["observed_profit"][position], "D", color="C2",
                    label="Observed profit at logged a")
            ax.set(title=f"{label}: ‖z‖ = {norms[position]:.2f}\nSource row {rows[position]}",
                   xlabel="Price change, a", ylabel="Profit per customer")
            ax.legend()
        fig.savefig(output / "representative_customer_profit_curves.pdf", format="pdf")
        plt.close(fig)
        diagnostics = _shape_and_action_diagnostics(arrays, output)
    record = {"family": FAMILY, "revision": "customer-slices-and-residual-shape",
              "input_residuals": file_record(source / "residuals.npz"),
              "input_provenance": file_record(source / "provenance.json"),
              "models": model_provenance, "diagnostics": diagnostics,
              "selection": {"helper": "experiments.envelopes.plotting._slice_indices",
                            "representation": "unclipped whitened numeric block of saved GLM acceptance preprocessor",
                            "excluded_from_distance": ["categorical label codes", "action", "outcomes"],
                            "rows": rows[positions].tolist(), "norms": norms[positions].tolist()},
              "action_domain": [float(grid[0]), float(grid[-1])],
              "truth": "one observed profit per customer at the historical action; no true counterfactual curve",
              "optimizer": None, "fitted_residual_basis": None, "randomness": None,
              "sources": [file_record(Path(__file__)), file_record(ROOT / "MATH.md"),
                          file_record(ROOT / "src/experiments/envelopes/plotting.py"),
                          file_record(ROOT / "src/reporting/real_data.py"),
                          file_record(ROOT / "src/reporting/profit_dispersion.py"),
                          file_record(ROOT / "src/data/feature_processor.py")],
              "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "xgboost")}}
    (output / "EXPERIMENT.md").write_text(
        "# Representative Generali customers and residual diagnostics\n\n"
        "The same smallest, median and largest norm selection helper as the wild-bootstrap "
        "plots is applied to the saved GLM preprocessor's whitened numeric features. "
        "No clipping or new preprocessing fit is applied. Categorical label codes, action "
        "and outcomes do not define distance; both predictors still use all model features. "
        "The same three customers are used for both models.\n\n"
        "The diamond is the single observed historical profit. Curves are predictions; "
        "dotted vertical segments show residuals only at the historical action. No observed "
        "or true profit curve is available at unlogged actions. Each panel has its own "
        "profit scale. The initial [-0.1, 0.2] action interval is extended to include "
        "selected historical actions. The existing spline is constant below 0 and uses "
        "its clipped tangent above 0.16; this extrapolation is not validated by the slice.\n\n"
        "Q-Q plots compare standardized empirical residual quantiles (0.01%–99.99%) "
        "with normal quantiles. Full, central-95%, renewed and churned shape metrics are "
        "saved separately. Trimming changes distributional shape. These are pooled "
        "saved-model residuals, not identified conditional innovations.\n\n"
        "The action diagnostic explicitly zooms to [-0.1, 0.3], using width-0.01 bins "
        "and plotting conditional summaries only where n >= 50. Counts and exclusions "
        "are recorded. The 10th–90th percentile ribbon is observed spread, not a confidence "
        "band. Connected bin statistics are visualization only. Existing original and "
        "central95 reports are unchanged. No residual basis or optimizer is fitted.\n\n"
        "Regenerate with `python scratch/plot_generali_profit_residuals.py --explore "
        f"--replay {source.resolve()} --output-dir <fresh-directory>`.\n", encoding="utf-8")
    record["outputs"] = [file_record(p) for p in sorted(output.iterdir())]
    write_provenance(output / "provenance.json", record)
    print(f"Customer exploration saved to {output}", flush=True)


def _churn_split(arrays, output, source):
    """Describe frozen-model residuals by observed renewal outcome, without refits."""
    from scipy.stats import kurtosis, skew
    from experiments.envelopes.plotting import _STYLE

    origin = json.loads((source / "provenance.json").read_text())
    while "dataset" not in origin:
        origin = origin["original_analysis"]
    if file_sha256(DATASET_PATH) != origin["dataset"]["sha256"]:
        raise ValueError("Dataset changed since residual calculation.")
    labels = pd.read_csv(DATASET_PATH, sep=";", usecols=["is_churn"])["is_churn"].to_numpy()
    churn = arrays["is_churn"]
    if not np.isin(churn, [0, 1]).all() or not np.array_equal(labels[arrays["row_indices"]], churn):
        raise ValueError("Saved churn labels disagree with the actual dataset outcomes.")
    if np.any(arrays["observed_profit"][churn == 1] != 0):
        raise ValueError("Churned observed profit must be zero under this report's convention.")
    groups = {0: "Renewed", 1: "Churned"}
    summaries, histogram_rows, selections = [], [], []
    for name in MODELS:
        np.testing.assert_allclose(arrays[f"{name}_residual"],
                                   arrays["observed_profit"] - arrays[f"{name}_profit"], rtol=0, atol=1e-10)
        for value, label in groups.items():
            residual = arrays[f"{name}_residual"][churn == value]
            if not len(residual):
                raise ValueError(f"No {label.lower()} customers in saved residuals.")
            summaries.append({"model": name, "group": label, "is_churn": value, "n": len(residual),
                              "mean": np.mean(residual), "median": np.median(residual),
                              "std_ddof0": np.std(residual), "skewness": skew(residual),
                              "excess_kurtosis": kurtosis(residual),
                              "fraction_negative": np.mean(residual < 0),
                              **{f"q{q:g}": np.quantile(residual, q) for q in (0, .025, .975, 1)}})
    pd.DataFrame(summaries).to_csv(output / "churn_split_summary.csv", index=False)
    with plt.rc_context(_STYLE):
        for view, fraction in (("full", 1.), ("central95", .95)):
            displayed = {}
            for name in MODELS:
                for value, label in groups.items():
                    residual = arrays[f"{name}_residual"][churn == value]
                    tail = (1-fraction)/2
                    low, high = np.quantile(residual, [tail, 1-tail])
                    values = residual[(residual >= low) & (residual <= high)]
                    displayed[name, value] = values
                    selections.append({"view": view, "model": name, "is_churn": value,
                                       "group": label, "residual_low": float(low), "residual_high": float(high),
                                       "group_n": len(residual), "displayed_n": len(values)})
            bins = np.histogram_bin_edges(np.concatenate(list(displayed.values())), bins=100)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
            title = "All residuals" if view == "full" else "Central 95% within each group and model"
            fig.suptitle(f"Observed renewal outcome: {title}", fontsize=16)
            for row, (name, model_label) in enumerate(MODELS.items()):
                for col, (value, group_label) in enumerate(groups.items()):
                    values = displayed[name, value]
                    count = np.histogram(values, bins=bins)[0]
                    if count.sum() != len(values):
                        raise ValueError("Churn-split histogram lost observations.")
                    density = count / (len(values)*np.diff(bins))
                    np.testing.assert_allclose(np.sum(density*np.diff(bins)), 1.)
                    histogram_rows.extend({"view": view, "model": name, "is_churn": value,
                                           "left": low, "right": high, "n": n, "density": d}
                                          for low, high, n, d in zip(bins[:-1], bins[1:], count, density))
                    summary = next(s for s in summaries if s["model"] == name and s["is_churn"] == value)
                    ax = axes[row, col]
                    ax.stairs(density, bins, fill=True, color=f"C{row}", alpha=.6)
                    ax.axvline(0, color="0.5", linewidth=.8)
                    ax.axvline(summary["mean"], color="black", linestyle="--",
                               label=f"Full-group mean: {summary['mean']:.2f}")
                    ax.set(title=f"{model_label} · {group_label} (is_churn={value})\n"
                                 f"{len(values):,} of {summary['n']:,} customers",
                           xlabel="Observed profit − predicted profit", ylabel="Residual probability density")
                    ax.tick_params(labelbottom=True, labelleft=True)
                    ax.legend()
            fig.savefig(output / f"residual_histograms_by_churn_{view}.pdf", format="pdf")
            plt.close(fig)
    pd.DataFrame(histogram_rows).to_csv(output / "histogram_counts.csv", index=False)
    write_provenance(output / "plot_selection.json", {"selections": selections,
        "rule": "inclusive residual quantile bounds within each observed group and model; no action trimming",
        "normalization": "each displayed group-model histogram integrates to one; common bins and axes per PDF"})
    (output / "EXPERIMENT.md").write_text(
        "# Generali residuals split by observed churn\n\n"
        "Uses the saved residuals and original source-row alignment. `is_churn=0` means renewed; "
        "`is_churn=1` means churned. Labels are verified against the checksummed canonical dataset, "
        "not inferred from model acceptance probabilities. No models are re-evaluated or fitted.\n\n"
        "Residual = observed profit minus model expected profit. Observed profit is "
        "(1-is_churn)*(X_upcoming_premium-Y_G_Loss); consequently churned residuals equal "
        "minus predicted profit. Renewal and churn subsets need not separately have zero mean "
        "even for a calibrated expected-profit model. This is an outcome-conditioned diagnostic; "
        "the realized churn flag is not a pre-decision feature for phi.\n\n"
        "Both PDFs use 100 shared equal-width bins, shared axes, and group-normalized densities. "
        "The full PDF retains all residuals; extreme tails compress the center. The central95 PDF "
        "retains inclusive 2.5th–97.5th residual percentiles separately within each group/model, "
        "with no action trimming or winsorization. Dashed means and churn_split_summary.csv statistics always "
        "use full groups. Exact bounds/counts and histogram coordinates are saved.\n\n"
        "The existing wild bootstrap fits mean residuals and sign-flips the post-fit errors. "
        "Both observed groups contribute when fitting on the pooled sample, but this targets "
        "uncertainty in expected profit, not a prediction interval spanning both realized outcomes. "
        "Rademacher pseudo-errors are symmetric and do not recreate the observed renewal/churn "
        "mixture. These diagnostic plots alone establish no bootstrap coverage. No bootstrap, "
        "basis fitting, optimization, sampling or changes to previous reports occur here.\n\n"
        "Regenerate with `python scratch/plot_generali_profit_residuals.py --split-churn "
        f"--replay {source.resolve()} --output-dir <fresh-directory>`.\n", encoding="utf-8")
    record = {"family": FAMILY, "revision": "observed-churn-split",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "input_residuals": file_record(source / "residuals.npz"),
              "input_provenance": file_record(source / "provenance.json"),
              "dataset": origin["dataset"], "formula": origin["formula"],
              "group_counts": {label: int(np.sum(churn == value)) for value, label in groups.items()},
              "validation": "dataset checksum, true churn labels, residual identity, zero churned profit and histogram mass verified",
              "optimizer": None, "band": None, "randomness": None,
              "sources": [file_record(Path(__file__)), file_record(ROOT / "MATH.md"),
                          file_record(ROOT / "src/experiments/envelopes/wild.py")],
              "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "matplotlib")}}
    record["outputs"] = [file_record(p) for p in sorted(output.iterdir())]
    write_provenance(output / "provenance.json", record)
    print(f"Observed-churn split saved to {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh report directory within the comparison family.")
    parser.add_argument("--spline-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--replay", type=Path, help="Saved residual input; only --fit-wild-linear enables residual fitting/bootstrap.")
    parser.add_argument("--central-fraction", type=float, default=1., help="Displayed central residual fraction per model; full saved data and summaries are retained.")
    parser.add_argument("--trim-actions", action="store_true", help="Also intersect with the same central historical-action quantiles.")
    parser.add_argument("--action-renderer", choices=("scatter", "density"), default="scatter", help="Density counts every selected row in compact vector cells with a common log color scale.")
    parser.add_argument("--explore", action="store_true", help="Use saved residuals for representative customer curves, Q-Q plots and action-bin diagnostics.")
    parser.add_argument("--split-churn", action="store_true", help="Split saved residuals by true is_churn, showing full and within-group central-95% histograms.")
    parser.add_argument("--fit-wild-linear", type=Path, help="Explicit JSON setup for affine residual fitting and paired wild-bootstrap bands.")
    parser.add_argument("--render-wild", type=Path, help="Replay a completed wild run's saved band coordinates without fitting.")
    parser.add_argument("--optimize-wild", type=Path, help="JSON setup for full-context band averages and repository optimization of personalized lower endpoints.")
    parser.add_argument("--explore-wild", type=Path, help="JSON setup for saved wild-band slices, historical-action groups and empirical feature support.")
    args = parser.parse_args(argv)
    if args.explore_wild and (args.replay is None or args.optimize_wild or args.fit_wild_linear
                             or args.render_wild or args.explore or args.split_churn
                             or args.trim_actions or args.central_fraction != 1):
        raise ValueError("Saved wild-band exploration requires --replay and no other report mode.")
    if args.optimize_wild and (args.replay is None or args.fit_wild_linear or args.render_wild
                              or args.explore or args.split_churn or args.trim_actions or args.central_fraction != 1):
        raise ValueError("Wild policy analysis requires --replay and no other report mode.")
    if args.render_wild:
        if args.replay or args.fit_wild_linear or args.explore or args.split_churn or args.trim_actions or args.central_fraction != 1:
            raise ValueError("Saved wild-coordinate replay cannot be combined with other report modes.")
        from scratch.generali_residual_wild import render_saved_wild
        render_saved_wild(args.render_wild, args.output_dir.resolve())
        return
    if args.fit_wild_linear and (args.replay is None or args.explore or args.split_churn
                                or args.central_fraction != 1 or args.trim_actions):
        raise ValueError("Wild fitting requires --replay and no other diagnostic or trimming mode.")
    if args.split_churn and (args.replay is None or args.explore or args.central_fraction != 1 or args.trim_actions):
        raise ValueError("Churn splitting requires --replay without exploration or display trimming options.")
    if args.explore and (args.replay is None or args.central_fraction != 1 or args.trim_actions):
        raise ValueError("Exploration requires --replay of the full saved residuals, without display trimming options.")
    if not 0 < args.central_fraction <= 1:
        raise ValueError("central-fraction must be in (0, 1].")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Preserve existing outputs; choose a fresh directory: {output}")
    if args.replay:
        saved = json.loads((args.replay / "provenance.json").read_text())
        record = next(item for item in saved["outputs"] if Path(item["path"]).name == "residuals.npz")
        artifact = args.replay / "residuals.npz"
        if file_sha256(artifact) != record["sha256"]:
            raise ValueError("Saved residuals checksum mismatch.")
        with np.load(artifact, allow_pickle=False) as data:
            arrays = dict(data)
        provenance = {"replayed_from": file_record(args.replay / "provenance.json"),
                      "original_analysis": saved}
    else:
        arrays, provenance = _evaluate(args.spline_cache)
    output.mkdir(parents=True, exist_ok=False)
    if args.explore_wild:
        from scratch.generali_wild_slices import run_wild_slices
        run_wild_slices(arrays, output, args.replay, args.explore_wild)
        return
    if args.optimize_wild:
        from scratch.generali_wild_policy import run_wild_policy
        run_wild_policy(arrays, output, args.replay, args.optimize_wild)
        return
    if args.fit_wild_linear:
        from scratch.generali_residual_wild import run_linear_wild
        run_linear_wild(arrays, output, args.replay, args.fit_wild_linear)
        return
    if args.split_churn:
        _churn_split(arrays, output, args.replay)
        return
    if args.explore:
        _explore(arrays, output, args.replay)
        return
    np.savez_compressed(output / "residuals.npz", **arrays)
    summaries = _plots_and_summaries(arrays, output, central_fraction=args.central_fraction,
                                    trim_actions=args.trim_actions, action_renderer=args.action_renderer)
    source_paths = [Path(__file__), ROOT / "MATH.md", *(ROOT / "src" / p for p in (
        "data/dataset_metadata.py", "data/loader.py", "data/full_monotone_spline_cache.py",
        "data/monotone_spline_xgb.py", "reporting/real_data.py", "reporting/profit_dispersion.py"))]
    provenance.update({"family": FAMILY, "revision": "historical-profit-residuals",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        "seed": None, "randomness": "none; all rows, fixed saved models, serial evaluation",
        "optimizer": "none; no optimum computed or marked", "band": None,
        "formula": {"actual": "(1-is_churn)*(X_upcoming_premium-Y_G_Loss)",
                    "predicted": "p_model(X,a)*((1+a)*X_policy_premium-L_model(X))", "residual": "actual-predicted", "a": "U"},
        "interpretation": "descriptive saved-model residuals; not held-out performance or conditional-mean truth",
        "plotting": {"all_rows": args.central_fraction == 1., "clipping": False,
                     "central_fraction_per_variable": args.central_fraction, "trim_actions": args.trim_actions,
                     "action_renderer": args.action_renderer, "smoothing": False, "histogram_bins": 100, "format": "vector PDF"},
        "sources": [file_record(p) for p in source_paths],
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib")},
        "summary": summaries})
    (output / "EXPERIMENT.md").write_text(
        "# Generali historical profit residuals\n\n"
        "Family: `glm-spline-profit-dispersion`; revision: historical-profit-residuals.\n\n"
        "All common eligible rows are evaluated at their own historical action a = U. "
        "Actual profit = (1-is_churn)*(X_upcoming_premium-Y_G_Loss). "
        "Predicted profit = acceptance_model(X,a)*((1+a)*X_policy_premium-loss_model(X)). "
        "Residual = actual minus predicted. The premium identity is checked before evaluation.\n\n"
        "GLM uses saved logistic acceptance and Ridge loss; spline + XGBoost uses the "
        "validated full-population monotone-spline acceptance cache and saved XGBoost loss. "
        "Cache source hashes, payload checksums, row IDs and anchor weights are verified. "
        "Direct spline/profit comparisons use deterministic rows. Cache storage rounding "
        "is checked at acceptance tolerance 2e-6, propagated to profit units by the margin. "
        "Spline behavior is constant below a=0 and clipped tangent above a=0.16.\n\n"
        "The saved residuals and summary.csv retain the entire original cohort. "
        f"Plots display the central {args.central_fraction:.0%} of residuals per model"
        + (" intersected with the same central action quantiles. " if args.trim_actions else ". ") +
        "No values are winsorized. Exact inclusive bounds and retained counts are in "
        "plot_selection.json; plot_rows.npz saves each model's displayed source row IDs. "
        f"The action plot uses {args.action_renderer}; density counts all selected observations "
        "on a common grid with a common logarithmic color scale. Histograms use 100 "
        "shared equal-width bins over the displayed range. No smoothing, "
        "basis fitting, bootstrap, optimizer, random sampling, or model training is performed. "
        "These are descriptive residuals; predictor holdout membership is not established. "
        "Observed profit follows the dataset loss-target convention, not a known conditional mean.\n\n"
        "`residuals.npz` retains source row IDs, customer IDs, a, observed inputs, each model's "
        "acceptance/loss/profit, and residuals. `summary.csv` and `histogram_counts.csv` "
        "provide numerical summaries; `provenance.json` records exact inputs, source hashes, "
        "model artifacts, validation, environment and executable command.\n\n"
        "Replay without model evaluation: `python scratch/plot_generali_profit_residuals.py "
        f"--replay {output} --output-dir <fresh-directory>`.\n", encoding="utf-8")
    provenance["outputs"] = [file_record(p) for p in sorted(output.iterdir())]
    write_provenance(output / "provenance.json", provenance)
    print(f"Report saved to {output}", flush=True)


if __name__ == "__main__":
    main()
