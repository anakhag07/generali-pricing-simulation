"""Generali residual-fit orchestration for the existing comparison driver.

All fitting, wild refits, all-real calibration statistics and transfer radii
come from experiments.envelopes.wild; no new band mathematics lives here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.dataset_metadata import DATASET_PATH
from data.loader import load_x_frame
from experiments.envelopes.plotting import _STYLE, _slice_indices
from experiments.envelopes.wild import WildLinearDesign
from experiments.provenance import file_record, file_sha256, write_provenance
from experiments.seeds import derive_seed
from reporting.real_data import resolve_real_data_model

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"glm": "GLM", "spline_xgb": "Spline + XGBoost"}


def _inputs(arrays, source, customer_report):
    origin = json.loads((source / "provenance.json").read_text())
    while "dataset" not in origin:
        origin = origin["original_analysis"]
    if file_sha256(DATASET_PATH) != origin["dataset"]["sha256"]:
        raise ValueError("Dataset changed since the saved residuals.")
    glm = resolve_real_data_model("glm")
    if glm.provenance() != origin["models"]["glm"]:
        raise ValueError("GLM model/preprocessor differs from the saved residuals.")
    frame = load_x_frame("xgb", row_indices=arrays["row_indices"])
    processor = glm.acceptance_artifact.preprocessor
    names = list(processor.numeric_feature_names_)
    z = processor.transform(frame[list(glm.acceptance_artifact.x_feature_cols)])[names].to_numpy(float)
    report = json.loads((customer_report / "provenance.json").read_text())
    if report["input_residuals"]["sha256"] != file_sha256(source / "residuals.npz"):
        raise ValueError("Customer curves and residuals use different observations.")
    path = customer_report / "customer_curves.npz"
    saved = next(r for r in report["outputs"] if Path(r["path"]).name == path.name)
    if file_sha256(path) != saved["sha256"]:
        raise ValueError("Saved customer-curve checksum mismatch.")
    with np.load(path, allow_pickle=False) as data:
        examples = dict(data)
    positions = _slice_indices(z)
    np.testing.assert_array_equal(examples["selection_positions"], positions)
    np.testing.assert_array_equal(examples["row_indices"], arrays["row_indices"][positions])
    np.testing.assert_allclose(examples["numeric_features"], z[positions], rtol=0, atol=1e-12)
    for name in MODELS:
        np.testing.assert_allclose(arrays[f"{name}_residual"],
                                   arrays["observed_profit"]-arrays[f"{name}_profit"], rtol=0, atol=1e-10)
    return z, names, examples, origin


def _metrics(residual, fitted, churn, *, model, stage):
    rows = []
    for group, mask in (("All", np.ones(len(residual), dtype=bool)),
                        ("Renewed", churn == 0), ("Churned", churn == 1)):
        y, predicted = residual[mask], fitted[mask]
        error = y-predicted
        sst = np.sum((y-y.mean())**2)
        rows.append({"model": model, "stage": stage, "group": group, "n": len(y),
                     "raw_residual_mean": np.mean(y), "fitted_residual_mean": np.mean(predicted),
                     "postfit_error_mean": np.mean(error),
                     "raw_rmse": np.sqrt(np.mean(y**2)), "fit_rmse": np.sqrt(np.mean(error**2)),
                     "raw_mae": np.mean(np.abs(y)), "fit_mae": np.mean(np.abs(error)),
                     "residual_r2": 1-np.sum(error**2)/sst if sst > 0 else np.nan})
    return rows


def _bootstrap_batches(design, residuals, *, count, batch_size, method_seed):
    """Pair both models using draw-specific seeds, independent of batch size."""
    for start in range(0, count, batch_size):
        indices = range(start, min(start+batch_size, count))
        seeds = np.array([derive_seed(method_seed, f"draw:{i}") for i in indices], dtype=np.uint64)
        signs = np.stack([2*np.random.default_rng(int(seed)).integers(
            0, 2, size=len(design.design), dtype=np.int8)-1 for seed in seeds])
        outputs = {}
        for name, residual in residuals.items():
            beta, errors = design.bootstrap_errors(residual, signs)
            outputs[name] = {"beta": beta, "errors": errors, "statistics": design.statistics(errors)}
        yield seeds, signs, outputs


def _plot_bands(output, arrays, examples, design, fits, delta):
    grid = examples["action_grid"]
    positions = examples["selection_positions"]
    contexts = np.repeat(examples["numeric_features"], len(grid), axis=0)
    actions = np.tile(grid, len(positions))
    labels = ["Smallest norm", "Median norm", "Largest norm"]
    coordinates, summaries, stored = [], [], {}
    for name, fit in fits.items():
        components = design.residual_components(contexts, actions, fit["beta"], fit["critical"])
        parts = {key: value.reshape(3, len(grid)) for key, value in components.items()}
        parts["prediction"] = examples[f"{name}_profit"]
        parts["lower"] = parts["prediction"]-parts["radius_lower"]
        parts["upper"] = parts["prediction"]+parts["radius_upper"]
        parts["corrected_prediction"] = parts["prediction"]+parts["residual_fit"]
        stored.update({f"{name}_{key}": value for key, value in parts.items()})
        for j, position in enumerate(positions):
            logged = int(np.searchsorted(grid, arrays["a"][position]))
            summaries.append({"model": name, "selection": labels[j],
                              "source_row": int(arrays["row_indices"][position]),
                              "historical_action": arrays["a"][position],
                              "observed_profit": arrays["observed_profit"][position],
                              "observed_residual": arrays[f"{name}_residual"][position],
                              "leverage": design.leverage[position],
                              **{key: value[j, logged] for key, value in parts.items()}})
            coordinates.extend({"model": name, "selection": labels[j], "action": action,
                                **{key: value[j, k] for key, value in parts.items()}}
                               for k, action in enumerate(grid))
    np.savez_compressed(output / "band_curves.npz", action_grid=grid,
                        row_indices=examples["row_indices"], **stored)
    pd.DataFrame(coordinates).to_csv(output / "band_curves.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "band_slice_summary.csv", index=False)
    _render_band_curves(output, arrays, examples, stored, delta)


def _render_band_curves(output, arrays, examples, stored, delta, *, labels=None,
                        quantities=("residual", "profit"), action_bounds=None,
                        filename_prefix="wild"):
    grid = examples["action_grid"]
    positions = examples["selection_positions"]
    if labels is None:
        labels = ["Smallest norm", "Median norm", "Largest norm"]
    with plt.rc_context(_STYLE):
        for quantity in quantities:
            fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, constrained_layout=True)
            noun = "Residual-mean bands" if quantity == "residual" else "Transferred expected-profit envelopes"
            fig.suptitle(f"{noun} · nominal {1-delta:.0%} wild bootstrap", fontsize=16)
            legend_handles, legend_labels = [], []
            for row, (name, model_label) in enumerate(MODELS.items()):
                for col, (position, label) in enumerate(zip(positions, labels)):
                    ax = axes[row, col]
                    g = stored[f"{name}_residual_fit"][col]
                    width = stored[f"{name}_width"][col]
                    if quantity == "residual":
                        ax.fill_between(grid, g-width, g+width, alpha=.25, color=f"C{row}", label="Residual-mean band")
                        ax.plot(grid, g, color=f"C{row}", label=r"Fitted residual $\phi^\top\widehat\beta$")
                        ax.axhline(0, color="0.5", linewidth=.8)
                        observed = arrays[f"{name}_residual"][position]
                        ylabel = "Profit residual"
                    else:
                        ax.fill_between(grid, stored[f"{name}_lower"][col], stored[f"{name}_upper"][col],
                                        alpha=.25, color=f"C{row}", label="Transferred envelope")
                        ax.plot(grid, stored[f"{name}_prediction"][col], color=f"C{row}", label="Frozen profit prediction")
                        ax.plot(grid, stored[f"{name}_corrected_prediction"][col], "--", color="C3",
                                label="Prediction + fitted residual")
                        observed = arrays["observed_profit"][position]
                        ylabel = "Profit per customer"
                    ax.plot(arrays["a"][position], observed, "D", color="C2", label="Observed outcome at logged a")
                    ax.set(title=f"{model_label} · {label}\nSource row {arrays['row_indices'][position]}",
                           xlabel="Price change, a", ylabel=ylabel)
                    if action_bounds is not None:
                        ax.set_xlim(*action_bounds)
                    ax.tick_params(labelbottom=True)
                    if col == 0:
                        handles, entries = ax.get_legend_handles_labels()
                        legend_handles.extend(handles[:2])
                        legend_labels.extend(f"{model_label}: {entry}" for entry in entries[:2])
                        if row == 1:
                            legend_handles.extend(handles[2:])
                            legend_labels.extend(entries[2:])
            # Automatic outside layout keeps the single observed points visible.
            fig.legend(legend_handles, legend_labels, loc="outside lower center", ncol=3)
            fig.savefig(output / f"{filename_prefix}_{quantity}_bands.pdf", format="pdf")
            plt.close(fig)


def render_saved_wild(source, output):
    """Replay exact saved band coordinates without fitting or generating draws."""
    provenance = json.loads((source / "provenance.json").read_text())
    inputs = []

    def checked(path, expected):
        record = file_record(path)
        if record["sha256"] != expected["sha256"]:
            raise ValueError(f"Saved wild-report input changed: {path}")
        inputs.append(record)
        return path

    def local(name):
        record = next(r for r in provenance["outputs"] if Path(r["path"]).name == name)
        return checked(source / name, record)

    with np.load(local("band_curves.npz"), allow_pickle=False) as data:
        stored = dict(data)
    config = json.loads(local("resolved_config.json").read_text())
    record = provenance["input_residuals"]
    with np.load(checked(Path(record["path"]), record), allow_pickle=False) as data:
        arrays = dict(data)
    record = provenance["customer_curves"]
    with np.load(checked(Path(record["path"]), record), allow_pickle=False) as data:
        examples = dict(data)
    np.testing.assert_array_equal(stored["action_grid"], examples["action_grid"])
    np.testing.assert_array_equal(stored["row_indices"], examples["row_indices"])
    output.mkdir(parents=True, exist_ok=False)
    _render_band_curves(output, arrays, examples, stored, config["bootstrap"]["delta"])
    write_provenance(output / "provenance.json", {
        "revision": "automatic-outside-legends", "source_run": file_record(source / "provenance.json"),
        "inputs": inputs, "sources": [file_record(Path(__file__)),
                                         file_record(ROOT / "src/experiments/envelopes/plotting.py")],
        "operation": "render exact saved coordinates; no fits, bootstrap draws or changed endpoints",
        "outputs": [file_record(path) for path in sorted(output.iterdir())]})
    print(f"Saved-coordinate wild figures replayed to {output}", flush=True)


def _plot_holdout(output, residuals, predictions):
    rows = []
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        for ax, (name, label) in zip(axes, MODELS.items()):
            y, predicted = residuals[name], predictions[name]
            groups = np.array_split(np.argsort(predicted, kind="stable"), 20)
            x = np.array([np.mean(predicted[g]) for g in groups])
            observed = np.array([np.mean(y[g]) for g in groups])
            rows.extend({"model": name, "bin": j, "n": len(g), "mean_fitted_residual": x[j],
                         "mean_observed_residual": observed[j]} for j, g in enumerate(groups))
            ax.plot(x, observed, "o", label="Holdout means (20 equal-count bins)")
            bounds = [min(x.min(), observed.min()), max(x.max(), observed.max())]
            ax.plot(bounds, bounds, "--", color="0.5", label="Agreement")
            ax.set(title=label, xlabel="Mean fitted residual", ylabel="Mean observed residual")
            ax.legend()
        fig.savefig(output / "beta_fit_holdout.pdf", format="pdf")
        plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "holdout_calibration_bins.csv", index=False)


def run_linear_wild(arrays, output, source, config_path):
    """Fit the approved affine baseline and save paired wild-bootstrap artifacts."""
    config = json.loads(config_path.read_text())
    if config["family"] != "glm-spline-profit-dispersion" or config["basis"] != "affine_whitened_numeric":
        raise ValueError("This revision supports only the existing family's affine numeric basis.")
    bootstrap = config["bootstrap"]
    if (bootstrap["multipliers"], bootstrap["residuals"], bootstrap["quantile"]) != (
            "rademacher", "raw", "order_statistic"):
        raise ValueError("Use the existing raw-residual Rademacher construction.")
    count, batch_size, delta = bootstrap["count"], bootstrap["batch_size"], bootstrap["delta"]
    fraction = config["holdout_fraction"]
    if not (isinstance(count, int) and count >= 2 and isinstance(batch_size, int) and batch_size >= 1
            and 0 < delta < 1 and 0 < fraction < 1):
        raise ValueError("Invalid bootstrap or holdout settings.")
    customer_report = (config_path.parent / config["customer_report"]).resolve()
    z, names, examples, origin = _inputs(arrays, source, customer_report)
    n = len(z)
    customer_ids, customer_index, customer_counts = np.unique(
        arrays["customer_id"], return_inverse=True, return_counts=True)
    split_seed = derive_seed(config["seed_root"], "generali-residual:validation-split")
    method_seed = derive_seed(config["seed_root"], "generali-residual:wild-multipliers")
    # Keep repeated customer IDs on one side of the validation split. The
    # baseline bootstrap itself retains the repository's row-wise multipliers.
    permutation = np.random.default_rng(split_seed).permutation(len(customer_ids))
    test_customers = permutation[:math.ceil(len(customer_ids)*fraction)]
    test_mask = np.isin(customer_index, test_customers)
    test, train = np.flatnonzero(test_mask), np.flatnonzero(~test_mask)
    if np.intersect1d(customer_index[test], customer_index[train]).size:
        raise ValueError("Customer identity leaked across the residual-fit holdout split.")
    design = WildLinearDesign(z, arrays["a"])
    training_design = WildLinearDesign(z[train], arrays["a"][train])
    residuals = {name: arrays[f"{name}_residual"] for name in MODELS}
    fits, metrics, coefficients, heldout_predictions = {}, [], [], {}
    saved = {"row_indices": arrays["row_indices"], "X": z, "actions": arrays["a"],
             "train_positions": train, "test_positions": test,
             "feature_names": np.array(["intercept", *[f"z:{name}" for name in names], "action"]),
             "triangular": design._triangular, "covariance": design.covariance, "leverage": design.leverage}
    for name, residual in residuals.items():
        beta_train = training_design.fit(residual[train])
        train_fit = training_design.design @ beta_train
        test_fit = design.design[test] @ beta_train
        beta_full = design.fit(residual)
        full_fit = design.design @ beta_full
        for stage, indices, prediction in (("train", train, train_fit), ("holdout", test, test_fit),
                                            ("full_refit", np.arange(n), full_fit)):
            metrics.extend(_metrics(residual[indices], prediction, arrays["is_churn"][indices], model=name, stage=stage))
        heldout_predictions[name] = test_fit
        fits[name] = {"beta": beta_full, "errors": [], "statistics": []}
        saved.update({f"{name}_beta_train": beta_train, f"{name}_beta": beta_full,
                      f"{name}_observed_residual": residual, f"{name}_full_fit": full_fit,
                      f"{name}_holdout_fit": test_fit})
        coefficients.extend({"model": name, "feature": feature, "beta_train": btrain, "beta_full": bfull}
                            for feature, btrain, bfull in zip(saved["feature_names"], beta_train, beta_full))
    del training_design
    pd.DataFrame(metrics).to_csv(output / "fit_metrics.csv", index=False)
    pd.DataFrame(coefficients).to_csv(output / "beta_coefficients.csv", index=False)
    _plot_holdout(output, {name: r[test] for name, r in residuals.items()}, heldout_predictions)
    draw_seeds, digest = [], hashlib.sha256()
    print(f"Fitted {n:,} rows, {design.design.shape[1]} coefficients; bootstrapping {count} paired draws.", flush=True)
    for seeds, signs, outputs in _bootstrap_batches(design, residuals, count=count,
            batch_size=batch_size, method_seed=method_seed):
        draw_seeds.extend(seeds.tolist())
        digest.update(signs.tobytes())
        for name, result in outputs.items():
            np.testing.assert_allclose(result["beta"], fits[name]["beta"], rtol=0, atol=1e-12)
            fits[name]["errors"].append(result["errors"])
            fits[name]["statistics"].append(result["statistics"])
        if len(draw_seeds) % (batch_size*10) == 0 or len(draw_seeds) == count:
            print(f"Completed {len(draw_seeds)}/{count} paired bootstrap draws.", flush=True)
    saved["bootstrap_draw_seeds"] = np.array(draw_seeds, dtype=np.uint64)
    for name, fit in fits.items():
        fit["errors"] = np.concatenate(fit["errors"])
        fit["statistics"] = np.concatenate(fit["statistics"])
        order = math.ceil(count*(1-delta))-1
        fit["critical"] = float(np.partition(fit["statistics"], order)[order])
        saved.update({f"{name}_bootstrap_errors": fit["errors"], f"{name}_bootstrap_statistics": fit["statistics"],
                      f"{name}_critical": np.array(fit["critical"])})
    np.savez_compressed(output / "fit_state.npz", **saved)
    _plot_bands(output, arrays, examples, design, fits, delta)
    top = np.argsort(design.leverage, kind="stable")[-10:][::-1]
    pd.DataFrame({"source_row": arrays["row_indices"][top], "leverage": design.leverage[top],
                  "feature_norm": np.linalg.norm(z[top], axis=1), "is_churn": arrays["is_churn"][top],
                  "in_holdout": np.isin(top, test)}).to_csv(output / "highest_leverage_customers.csv", index=False)
    summary = {"n": n, "n_train": len(train), "n_holdout": len(test), "p": design.design.shape[1],
               "n_unique_customer_ids": len(customer_ids),
               "repeated_customer_ids": int(np.sum(customer_counts > 1)),
               "rows_in_repeated_customer_groups": int(customer_counts[customer_counts > 1].sum()),
               "validation_split_unit": "customer_id", "bootstrap_multiplier_unit": "row",
               "metrics": metrics, "critical_values": {name: fit["critical"] for name, fit in fits.items()},
               "max_leverage": float(design.leverage.max()), "mean_leverage": float(design.leverage.mean()),
               "leverage_quantiles": dict(zip(["median", "q95", "q99", "q999"], np.quantile(design.leverage, [.5, .95, .99, .999]))),
               "design_condition_number": float(np.linalg.cond(design._triangular)),
               "seeds": {"root": config["seed_root"], "split": split_seed, "method": method_seed},
               "multiplier_sha256_raw_int8_row_major": digest.hexdigest(),
               "coverage": None, "coverage_status": "unknown; no conditional-mean truth in observed Generali data"}
    write_provenance(output / "summary.json", summary)
    resolved = {**config, "customer_report": str(customer_report), "seeds": summary["seeds"],
                "source_residuals": str(source.resolve()), "target": "observed contexts crossed with all real actions",
                "plot_action_domain": [float(examples["action_grid"][0]), float(examples["action_grid"][-1])],
                "pairing": "same holdout and draw-specific Rademacher seeds across predictors and future bases"}
    write_provenance(output / "resolved_config.json", resolved)
    (output / "EXPERIMENT.md").write_text(_experiment_text(source, config_path, summary, resolved), encoding="utf-8")
    record = {"family": config["family"], "revision": "affine-generali-residual-wild",
              "config": file_record(config_path), "input_residuals": file_record(source / "residuals.npz"),
              "input_provenance": file_record(source / "provenance.json"),
              "customer_curves": file_record(customer_report / "customer_curves.npz"),
              "customer_provenance": file_record(customer_report / "provenance.json"),
              "dataset": origin["dataset"], "models": origin["models"], "summary": summary,
              "optimizer": None, "frozen_model_training_holdout": "unknown",
              "sources": [file_record(path) for path in (Path(__file__), ROOT / "MATH.md",
                  ROOT / "scratch/plot_generali_profit_residuals.py", ROOT / "src/data/feature_processor.py",
                  ROOT / "src/experiments/envelopes/wild.py", ROOT / "src/experiments/envelopes/linear.py",
                  ROOT / "src/experiments/seeds/streams.py", ROOT / "src/experiments/envelopes/plotting.py")],
              "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "matplotlib")}}
    record["outputs"] = [file_record(path) for path in sorted(output.iterdir())]
    write_provenance(output / "provenance.json", record)
    print(f"Linear residual wild-bootstrap report saved to {output}", flush=True)


def _experiment_text(source, config_path, summary, resolved):
    return (
        "# Affine Generali residual fit and wild-bootstrap baseline\n\n"
        "Family: glm-spline-profit-dispersion. Uses both unchanged frozen profit predictors and "
        f"all {summary['n']:,} eligible saved residuals R=Y-fhat at historical actions. Phi=(1,z',a), "
        "where z is the same saved GLM-whitened numeric block used for customer slices. "
        "There are no powers, interactions, categorical codes, churn flags, new preprocessing "
        "fits or tail exclusions. Each predictor has a separate OLS beta.\n\n"
        f"The user approved a shared random {1-resolved['holdout_fraction']:.0%}/{resolved['holdout_fraction']:.0%} "
        "residual-fit train/holdout split, followed by "
        "a full-data refit for band construction. This validates the residual correction only: "
        "all rows with the same customer ID stay on the same side of this split. "
        "holdout membership for the original predictors/preprocessor is unknown. Metrics include "
        "R2 relative to the subset residual mean, RMSE/MAE before and after correction, and signed "
        "errors, separately pooled/renewed/churned. Full-data statistics are in-sample. The holdout "
        "calibration plot groups all holdout rows into 20 equal-count bins by fitted residual.\n\n"
        f"Bootstrap: {resolved['bootstrap']['count']:,} paired Rademacher draws and nominal "
        f"{1-resolved['bootstrap']['delta']:.0%} calibration, using the existing "
        "WildLinearDesign raw post-fit residuals, refits, analytic all-real action statistic, and "
        "order statistic ceil(B*(1-delta))-1. Split and method seeds are separately derived from "
        "the declared root; each bootstrap row has its own saved derived seed, so batching does "
        "not alter draws. Both models and future bases reuse the same ordered cohort, split and "
        "draw seeds. A hash of the complete row-major int8 sign array verifies exact replay.\n\n"
        "The calibration domain is the fixed observed z contexts crossed with all real actions. "
        "At any real action, the residual fit is exactly affine and its half-width is "
        "critical*norm(R^{-T}phi), with R the full-design QR factor. Queries share the same beta "
        "and QR geometry; bootstrap coefficient draws induce dependence across all queries. "
        "The displayed action grid only renders these functions and selects no optimizer action. "
        "There is no stochastic objective, optimizer, numerical action-grid calibration or "
        "global optimum. The existing affine statistic evaluates its supremum analytically, "
        "including the action-tail limit, without a numerical search tolerance.\n\n"
        "Plots retain the same smallest/median/largest-norm customers and saved exact predictor "
        f"curves over {resolved['plot_action_domain']}. The residual plot shows gtilde +/- w. The profit "
        "plot transfers this using the shared lower=max(w-gtilde,0), upper=max(w+gtilde,0) radii "
        "around fhat, and separately marks fhat+gtilde. Original GLM and spline boundary behavior "
        "is unchanged; saved curve lines are rendering samples. Spline acceptance is constant "
        "below 0 and uses the clipped tangent above 0.16. Physical price decisions and empirical "
        "support are much narrower than the mathematical all-real calibration domain.\n\n"
        "Coverage is unknown. Approximate wild mean-function calibration requires an adequate "
        "conditional-mean basis, suitable independent errors and regularity/leverage conditions. "
        f"There are {summary['repeated_customer_ids']} repeated customer IDs covering "
        f"{summary['rows_in_repeated_customer_groups']} rows; the inherited bootstrap uses "
        "row-wise signs and does not correct potential within-customer dependence. "
        "These are not established from this dataset. We do not transfer the synthetic Gaussian "
        "finite-sample theorem, assert 95% empirical coverage, or use individual observed profits "
        "as conditional-mean truth. Churn/renewal outcome variation contributes to pooled errors; "
        "these are not prediction intervals covering both realized outcomes. Larger bases can "
        "reduce approximation bias while increasing leverage and variance; a better fit alone "
        "does not establish narrower or better-calibrated bands.\n\n"
        "Saved fit_state.npz contains split positions, contexts/actions, coefficients, QR geometry, "
        "residual fits, leverage, draw seeds, bootstrap coefficient errors/statistics and critical "
        "values. band_curves.npz/CSV contain all rendered endpoints. Previous reports are unchanged.\n\n"
        "Regenerate with `python scratch/plot_generali_profit_residuals.py "
        f"--fit-wild-linear {config_path.resolve()} --replay {source.resolve()} "
        "--output-dir <fresh-directory>`.\n")
