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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh report directory within the comparison family.")
    parser.add_argument("--spline-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--replay", type=Path, help="Existing report to replot without evaluating any model.")
    parser.add_argument("--central-fraction", type=float, default=1., help="Displayed central residual fraction per model; full saved data and summaries are retained.")
    parser.add_argument("--trim-actions", action="store_true", help="Also intersect with the same central historical-action quantiles.")
    parser.add_argument("--action-renderer", choices=("scatter", "density"), default="scatter", help="Density counts every selected row in compact vector cells with a common log color scale.")
    args = parser.parse_args(argv)
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
