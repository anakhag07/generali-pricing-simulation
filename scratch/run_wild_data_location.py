"""Reproduce the three 10/21 quadratic wild-bootstrap figures (MATH.md §7.17)."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import norm, t as student_t
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiments.envelopes.quadratic import features
from experiments.envelopes.quadratic import QuadraticErrorCertificate
from experiments.offline_bootstrap_math import (
    TransferredLCBObjective, _branches, _candidate_lower_value,
    transferred_below_level,
)
from experiments.seeds import SeedSetup, derive_seed, resolve_seed_setup
from experiments.launch import LaunchPlan, add_launch_args, run_launch_plan
from experiments.paths import results_root
from experiments.slurm import CPU_PROFILE
from objective.base import Objective
from optimization.solvers import run_first_order_minimize, run_stein_difference_minimize

CONFIG = {
    "run_seed": 20261021, "n": 300, "variance": 0.45, "means": [0.5, 10.0],
    "noise_sd": 1.0, "bootstrap_draws": 1999, "delta": 0.05,
    "starts": [-0.5, 1.5, 4.0], "step_rule": "constant", "step_size": 0.05,
    "t_steps": 1000, "n_grad_samples": 64, "smoothing_sd": 0.01,
    "gradient_tolerance": 1e-5, "statistic_tolerance": 1e-8,
    "global_gap_tolerance": 1e-6, "display_window": [-0.25, 1.25],
    "logging_windows": [[-1.5, 4.5], [-1.5, 12.0]],
    "truth_coefficients": [0.0, 5.0, -5.0],
}


def fit_band(actions, responses, multipliers):
    """Fit raw-residual wild draws and certify calibration on all real actions."""
    design = features(actions)
    orthogonal, triangular = np.linalg.qr(design, mode="reduced")
    beta = solve_triangular(triangular, orthogonal.T @ responses)
    residuals = responses - design @ beta
    errors = solve_triangular(triangular, orthogonal.T @ (multipliers * residuals).T).T
    inverse = solve_triangular(triangular, np.eye(3))
    covariance = inverse @ inverse.T
    certificate = QuadraticErrorCertificate(covariance, 1.0)
    bounds = np.array([
        certificate.supremum_interval(
            certificate.difference(error, np.zeros(3)), CONFIG["statistic_tolerance"]
        ) for error in errors
    ])
    order = int(np.ceil(len(errors) * (1 - CONFIG["delta"]))) - 1
    # This is a bootstrap quantile, never a choice of optimizer action.
    critical = float(np.partition(bounds[:, 1], order)[order])
    critical_low = float(np.partition(bounds[:, 0], order)[order])
    return {
        "actions": actions, "responses": responses, "beta": beta,
        "residuals": residuals, "bootstrap_errors": errors,
        "bootstrap_statistic_bounds": bounds, "covariance": covariance,
        "triangular": triangular, "critical": critical, "critical_low": critical_low,
        "max_leverage": float(np.max(np.sum(orthogonal**2, axis=1))),
    }


def band_values(band, actions):
    design = features(np.atleast_1d(actions))
    fitted = design @ band["beta"]
    radius = band["critical"] * np.linalg.norm(
        solve_triangular(band["triangular"].T, design.T, lower=True), axis=0
    )
    return fitted, radius


class QueryOnlyLowerBound(Objective):
    """Frozen negative LCB oracle; keep every actual solver query for replay."""

    def __init__(self, band):
        self.band = band
        self.queries = []

    def value(self, theta, x_batch):
        action = float(theta[0])
        fitted, radius = band_values(self.band, [action])
        loss = float(radius[0] - fitted[0])
        self.queries.append((action, loss))
        return loss

    def grad(self, theta, x_batch):
        raise AssertionError("The lower-bound optimizer must use function queries only.")


def certify_action(beta, covariance, critical, action):
    certificate = QuadraticErrorCertificate(covariance, 1.0)
    q = certificate.q.mul_ground(sp.Rational(float(critical)) ** 2)
    lower = _candidate_lower_value(_branches(beta, beta), q, action)
    exact_upper = sp.Rational(lower) + sp.Rational(CONFIG["global_gap_tolerance"])
    upper = float(exact_upper)
    while sp.Rational(upper) > exact_upper:
        upper = float(np.nextafter(upper, -np.inf))
    certified = transferred_below_level(beta, beta, q, upper)
    return {"certified": bool(certified), "value_lower": lower,
            "global_value_upper": upper if certified else None,
            "global_gap_tolerance": CONFIG["global_gap_tolerance"]}


def optimize_band(band, optimizer_seed, output, label, *, step_size=None):
    attempts = []
    for index, start in enumerate(CONFIG["starts"]):
        seed = derive_seed(optimizer_seed, f"stein-difference:start:{index}")
        batch_seed = derive_seed(optimizer_seed, f"batch:start:{index}")
        objective = QueryOnlyLowerBound(band)
        theta, trace = run_stein_difference_minimize(
            np.array([start]), np.zeros((1, 1)), objective,
            np.random.default_rng(seed), t_steps=CONFIG["t_steps"],
            n_grad_samples=CONFIG["n_grad_samples"], sigma=CONFIG["smoothing_sd"],
            algorithm=CONFIG["step_rule"],
            step_size=CONFIG["step_size"] if step_size is None else step_size,
            grad_norm_tol=CONFIG["gradient_tolerance"],
            gradient_rng=np.random.default_rng(seed),
            batch_rng=np.random.default_rng(batch_seed),
        )
        action = float(theta[0])
        fitted, radius = band_values(band, [action])
        certificate = certify_action(band["beta"], band["covariance"], band["critical"], action)
        attempt = {
            "start": start, "action": action, "lower_value": float(fitted[0]-radius[0]),
            "seed": seed, "batch_seed": batch_seed,
            "success": bool(trace.optimizer_success), "status": trace.optimizer_status,
            "message": trace.optimizer_message, "steps": int(trace.steps[-1]),
            "query_count": len(objective.queries), **certificate,
        }
        attempts.append(attempt)
        np.savez_compressed(
            output / f"{label}_start_{index}.npz", queries=np.asarray(objective.queries),
            theta_values=np.asarray(trace.theta_values), losses=np.asarray(trace.objective_values),
            gradient_norms=np.asarray(trace.theta_grad_norms),
        )
    # Compare only the three actual repository optimizer endpoints.
    best = max(attempts, key=lambda attempt: attempt["lower_value"])
    return attempts, best


def _paired_sweep_draws(spec, repetition):
    run_seed = (CONFIG["run_seed"] if repetition == 0 else
                derive_seed(CONFIG["run_seed"], f"repetition:{repetition}"))
    seeds = resolve_seed_setup(SeedSetup(run_seed=run_seed), run_seed)
    bootstrap_seed = derive_seed(run_seed, "wild-bootstrap:multipliers")
    n_max = max(spec["n_values"] + [spec["baseline_n"], CONFIG["n"]])
    rng = np.random.default_rng(bootstrap_seed)
    # Retain the original baseline's row-major draw shape before extending n.
    multipliers = rng.choice([-1., 1.], size=(CONFIG["bootstrap_draws"], CONFIG["n"]))
    if n_max > CONFIG["n"]:
        multipliers = np.column_stack((multipliers, rng.choice(
            [-1., 1.], size=(CONFIG["bootstrap_draws"], n_max-CONFIG["n"]))))
    draws = {
        "design_innovations": np.random.default_rng(seeds.data_seed).normal(size=n_max),
        "noise_innovations": np.random.default_rng(seeds.noise_seed).normal(size=n_max),
        "multipliers": multipliers,
    }
    return seeds, bootstrap_seed, draws


def _shift_signal(band, scale):
    # OLS is equivariant to adding a signal in its design column space.
    shifted = dict(band)
    delta = (scale-1.) * np.asarray(CONFIG["truth_coefficients"])
    shifted["beta"] = band["beta"] + delta
    shifted["responses"] = band["responses"] + features(band["actions"]) @ delta
    return shifted


def _quadratic_reference(beta, output, label):
    if beta[2] >= 0:
        return {"action": None, "value": None, "certified": False,
                "success": False, "message": "No finite maximum of fitted quadratic"}
    objective = TransferredLCBObjective(beta, beta, np.eye(3), 0.)
    theta, trace = run_first_order_minimize(
        np.array([0.]), np.zeros((1, 1)), objective,
        t_steps=100, n_grad_samples=1, sigma=0., grad_norm_tol=1e-10, ftol=1e-14)
    np.savez_compressed(output / f"{label}.npz", theta_values=trace.theta_values,
                        losses=trace.objective_values)
    return {"action": float(theta[0]), "value": -objective.value(theta, None),
            "solver": "optimization.solvers.run_first_order_minimize",
            "success": bool(trace.optimizer_success),
            **certify_action(beta, np.eye(3), 0., float(theta[0]))}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _sweep_contract(spec):
    sources = [Path(__file__), ROOT / "MATH.md"] + sorted((ROOT / "src").rglob("*.py"))
    return {"config": CONFIG, "sweep": spec,
            "source_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in sources},
            "repository_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}


def _run_sweep_repetition(spec, output, repetition, contract):
    folder = output / "repetitions" / f"rep_{repetition:03d}"
    summary_path = folder / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if saved["contract"] != contract:
            raise ValueError("Saved repetition belongs to a different source/configuration.")
        for name, digest in saved["artifact_sha256"].items():
            if _sha256(folder / name) != digest:
                raise ValueError(f"Saved artifact changed: {folder / name}")
        return {"summary": str(summary_path), "reused": True}
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Preserve interrupted outputs before retrying: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    seeds, bootstrap_seed, draws = _paired_sweep_draws(spec, repetition)
    np.savez_compressed(folder / "paired_draws.npz", **draws)
    grid = np.linspace(*spec["width_window"], spec["width_grid_points"])
    base_beta = np.asarray(CONFIG["truth_coefficients"])
    truth = {str(s): _quadratic_reference(s*base_beta, folder, f"truth_scale_{s:g}")
             for s in spec["steepness_values"]}
    rows, details = [], {}
    for mean in CONFIG["means"]:
        for n in sorted(set(spec["n_values"] + [spec["baseline_n"]])):
            actions = mean + np.sqrt(CONFIG["variance"])*draws["design_innovations"][:n]
            responses = features(actions) @ base_beta + CONFIG["noise_sd"]*draws["noise_innovations"][:n]
            print(f"rep={repetition} mean={mean:g} n={n}: calibrating", flush=True)
            base_band = fit_band(actions, responses, draws["multipliers"][:, :n])
            scales = spec["steepness_values"] if n == spec["baseline_n"] else [1.0]
            for scale in scales:
                key = f"mean_{mean:g}_n_{n}_scale_{scale:g}"
                band = _shift_signal(base_band, scale)
                np.savez_compressed(folder / f"{key}_band.npz", **band)
                step_size = CONFIG["step_size"] / max(1., scale)
                attempts, selected = optimize_band(
                    band, seeds.optimizer_seed, folder, key, step_size=step_size)
                fitted_reference = _quadratic_reference(band["beta"], folder, f"{key}_fit_reference")
                reference = truth[str(scale)]
                theta, theta_star = selected["action"], reference["action"]
                theta_fit = fitted_reference["action"]
                _, radius = band_values(band, grid)
                _, data_radius = band_values(band, actions)
                _, true_radius = band_values(band, [theta_star])
                cert = QuadraticErrorCertificate(band["covariance"], 1.)
                row = {
                    "repetition": repetition, "mean": mean, "n": n, "steepness": scale,
                    "theta_hat": theta, "theta_star": theta_star, "theta_fit": theta_fit,
                    "theta_difference": theta-theta_star,
                    "absolute_theta_difference": abs(theta-theta_star),
                    "band_shift": None if theta_fit is None else theta-theta_fit,
                    "absolute_band_shift": None if theta_fit is None else abs(theta-theta_fit),
                    "fit_error": None if theta_fit is None else theta_fit-theta_star,
                    "mean_width": float(np.trapezoid(2*radius, grid)/(grid[-1]-grid[0])),
                    "observed_mean_width": float(np.mean(2*data_radius)),
                    "width_at_truth": float(2*true_radius[0]),
                    "extrapolation_sd": abs(theta-mean)/np.sqrt(CONFIG["variance"]),
                    "realized_all_real_containment": bool(cert.contains(
                        cert.difference(band["beta"], scale*base_beta), band["critical"])),
                    "critical": band["critical"], "max_leverage": band["max_leverage"],
                    "optimizer_success": selected["success"], "certified": selected["certified"],
                    "all_starts_certified": all(a["certified"] for a in attempts),
                    "truth_certified": reference["certified"],
                    "fit_certified": fitted_reference["certified"], "step_size": step_size,
                }
                rows.append(row)
                details[key] = {"attempts": attempts, "selected": selected,
                                "truth_reference": reference, "fitted_reference": fitted_reference}
                print(f"  scale={scale:g}: shift={row['theta_difference']:.6g}, "
                      f"width={row['mean_width']:.6g}, certified={row['all_starts_certified']}", flush=True)
    _write_json(summary_path, {
        "contract": contract, "seeds": {**seeds.to_dict(), "bootstrap_seed": bootstrap_seed},
        "rows": rows, "details": details,
        "artifact_sha256": {p.name: _sha256(p) for p in sorted(folder.glob("*.npz"))}})
    return {"summary": str(summary_path), "conditions": len(rows)}


SWEEP_METRICS = ["theta_difference", "absolute_theta_difference", "band_shift",
                 "absolute_band_shift", "fit_error", "mean_width", "observed_mean_width",
                 "width_at_truth", "extrapolation_sd"]


def _aggregate_sweep(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["mean"], row["n"], row["steepness"]), []).append(row)
    summaries = []
    for (mean, n, scale), group in sorted(groups.items()):
        summary = {"mean": mean, "n": n, "steepness": scale, "repetitions": len(group)}
        for metric in SWEEP_METRICS:
            values = np.asarray([r[metric] for r in group if r[metric] is not None])
            count = len(values)
            std = float(np.std(values, ddof=1)) if count > 1 else 0.
            summary.update({f"{metric}_count": count,
                            f"{metric}_mean": float(np.mean(values)) if count else None,
                            f"{metric}_std": std if count else None,
                            f"{metric}_ci95": float(student_t.ppf(.975, count-1)*std/np.sqrt(count))
                            if count > 1 else None})
        summaries.append(summary)
    return summaries


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_sweep_summary(spec, summary, figure_dir, axis):
    is_scale = axis == "steepness"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = [("theta_difference", r"Mean $\hat\theta-\theta^\star$", "Signed displacement"),
              ("absolute_theta_difference", r"Mean $|\hat\theta-\theta^\star|$", "Displacement magnitude"),
              ("mean_width", "Mean full band width", "Width on the fixed action window"),
              ("extrapolation_sd", r"Mean $|\hat\theta-\mu|/\sqrt{0.45}$", "Distance from the data center")]
    for ax, (metric, ylabel, title) in zip(axes.flat, panels):
        for mean in CONFIG["means"]:
            subset = [r for r in summary if r["mean"] == mean and (
                r["n"] == spec["baseline_n"] if is_scale else r["steepness"] == 1.)]
            subset.sort(key=lambda r: r[axis])
            ax.errorbar([r[axis] for r in subset], [r[f"{metric}_mean"] for r in subset],
                        yerr=[r[f"{metric}_ci95"] for r in subset], marker="o", capsize=3,
                        label=rf"Data center $\mu={mean:g}$")
        ax.set(xscale="log", xlabel="Quadratic steepness factor $s$" if is_scale else "Number of observations $n$",
               ylabel=ylabel, title=title)
        if is_scale:
            ax.set_xticks(spec["steepness_values"], labels=[f"{s:g}" for s in spec["steepness_values"]])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend()
    fig.suptitle(("Steepness sweep" if is_scale else "Sample-size sweep") +
                 f"; {spec['repetitions']} paired repetitions, 95% Monte Carlo intervals")
    fig.savefig(figure_dir / f"{axis}_sweep_metrics.pdf", format="pdf")
    plt.close(fig)


def _plot_shift_decomposition(spec, summary, figure_dir, axis):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, mean in zip(axes, CONFIG["means"]):
        subset = [r for r in summary if r["mean"] == mean and (
            r["n"] == spec["baseline_n"] if axis == "steepness" else r["steepness"] == 1.)]
        subset.sort(key=lambda r: r[axis])
        for metric, label in [("theta_difference", r"Total: $\hat\theta-\theta^\star$"),
                              ("band_shift", r"Band: $\hat\theta-\theta_{\rm fit}$"),
                              ("fit_error", r"Fit: $\theta_{\rm fit}-\theta^\star$")]:
            ax.errorbar([r[axis] for r in subset], [r[f"{metric}_mean"] for r in subset],
                        yerr=[r[f"{metric}_ci95"] for r in subset], marker="o", capsize=3, label=label)
        ax.axhline(0, color="0.5", linestyle=":")
        ax.set(xscale="log", xlabel="Quadratic steepness factor $s$" if axis == "steepness" else "Number of observations $n$",
               ylabel="Mean signed action displacement", title=rf"Data center $\mu={mean:g}$")
        if axis == "steepness":
            ax.set_xticks(spec["steepness_values"], labels=[f"{s:g}" for s in spec["steepness_values"]])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend()
    fig.suptitle("Displacement decomposition; 95% Monte Carlo intervals")
    fig.savefig(figure_dir / f"{axis}_shift_decomposition.pdf", format="pdf")
    plt.close(fig)


def _plot_sweep_examples(spec, output, figure_dir, rows, axis, *, means=None):
    repetition = spec["illustration_repetition"]
    folder = output / "repetitions" / f"rep_{repetition:03d}"
    values = spec["illustration_steepness"] if axis == "steepness" else spec["illustration_n"]
    overrides_path = output / "plot-overrides.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    for mean in CONFIG["means"] if means is None else means:
        filename = f"{axis}_bands_data_center_{mean:g}.pdf"
        window = overrides.get(filename, {}).get("action_window")
        if window is not None and (len(window) != 2 or not np.all(np.isfinite(window))
                                   or window[0] >= window[1]):
            raise ValueError("A plot action window must have two finite, increasing bounds.")
        fig, axes = plt.subplots(2, len(values), figsize=(15, 8), constrained_layout=True)
        for column, value in enumerate(values):
            scale = float(value) if axis == "steepness" else 1.
            n = spec["baseline_n"] if axis == "steepness" else int(value)
            key = f"mean_{mean:g}_n_{n}_scale_{scale:g}"
            band = dict(np.load(folder / f"{key}_band.npz"))
            row = next(r for r in rows if r["repetition"] == repetition and r["mean"] == mean
                       and r["n"] == n and r["steepness"] == scale)
            # Keep the original zoom, expanding only if an endpoint falls outside it.
            lo = min(CONFIG["display_window"][0], row["theta_hat"]-.1)
            hi = max(CONFIG["display_window"][1], row["theta_hat"]+.1)
            if window is not None:
                lo, hi = window
            grid = np.linspace(lo, hi, 1201)
            truth = features(grid) @ (scale*np.asarray(CONFIG["truth_coefficients"]))
            fitted, radius = band_values(band, grid)
            ax, data_ax = axes[:, column]
            ax.plot(grid, truth, color="C0", label="True objective")
            ax.fill_between(grid, fitted-radius, fitted+radius, color="C1", alpha=.25,
                            label="Nominal 95% wild band")
            ax.plot(grid, fitted, "--", color="C1", label=r"OLS fit $\widehat f$")
            ax.plot(grid, fitted-radius, color="C2", label="Lower bound")
            star = row["theta_star"]
            ax.scatter([star], features(np.array([star])) @ (scale*np.asarray(CONFIG["truth_coefficients"])),
                       marker="*", s=110, color="C3", zorder=5, label=r"True optimum $\theta^\star$")
            fit_point, rad_point = band_values(band, [row["theta_hat"]])
            ax.scatter([row["theta_hat"]], fit_point-rad_point, marker="X", s=70,
                       color="C4", zorder=6, label=r"Stein endpoint $\hat\theta$")
            ax.set(xlabel="Action $a$", ylabel="Objective value",
                   title=rf"$s={scale:g}$, $n={n}$; $\hat\theta-\theta^\star={row['theta_difference']:.3f}$")
            density_window = CONFIG["logging_windows"][CONFIG["means"].index(mean)] if window is None else window
            density_grid = np.linspace(*density_window, 1201)
            data_ax.fill_between(density_grid, norm.pdf(density_grid, mean, np.sqrt(CONFIG["variance"])),
                                 color="C0", alpha=.25, label="Logging density")
            data_ax.hist(band["actions"], bins=24, density=True, histtype="step", color="C0",
                         label=f"{n} observed actions")
            data_ax.axvline(star, color="C3", linestyle=":", label=r"$\theta^\star$")
            data_ax.axvline(row["theta_hat"], color="C4", linestyle="--", label=r"$\hat\theta$")
            data_ax.set(xlabel="Action $a$", ylabel="Action density")
            if window is not None:
                ax.set_xlim(*window)
                data_ax.set_xlim(*window)
            data_ax.legend()
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
        fig.savefig(figure_dir / filename, format="pdf")
        plt.close(fig)


def _collect_sweep(spec, output, contract):
    rows = []
    for repetition in range(spec["repetitions"]):
        folder = output / "repetitions" / f"rep_{repetition:03d}"
        saved = json.loads((folder / "summary.json").read_text())
        if saved["contract"] != contract:
            raise ValueError("Cannot combine incompatible repetition contracts.")
        for name, digest in saved["artifact_sha256"].items():
            if _sha256(folder / name) != digest:
                raise ValueError(f"Changed artifact: {folder / name}")
        rows.extend(saved["rows"])
    summary = _aggregate_sweep(rows)
    _write_csv(output / "sweep_finals.csv", rows)
    _write_csv(output / "sweep_summary.csv", summary)
    failures = [r for r in rows if not (r["all_starts_certified"] and r["truth_certified"]
                                      and (r["theta_fit"] is None or r["fit_certified"]))]
    _write_json(output / "validation.json", {"conditions": len(rows), "failed_certificates": failures,
        "optimizer_success_count": sum(r["optimizer_success"] for r in rows),
        "missing_fitted_reference_count": sum(r["theta_fit"] is None for r in rows)})
    if failures:
        raise RuntimeError("Some endpoints failed certification; inspect validation.json.")
    figure_dir = output / "plots"
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "figure.titlesize": 16})
    for axis in ("steepness", "n"):
        _plot_sweep_summary(spec, summary, figure_dir, axis)
        _plot_shift_decomposition(spec, summary, figure_dir, axis)
        _plot_sweep_examples(spec, output, figure_dir, rows, axis)
    _write_json(output / "deliverables_sha256.json", {
        str(p.relative_to(output)): _sha256(p) for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "deliverables_sha256.json" and "sweeps" not in p.parts})
    print(f"Collected {len(rows)} certified conditions in {output}", flush=True)


def _launch_sweep(args):
    spec = json.loads(args.sweep_config.read_text())
    if spec["baseline_n"] not in spec["n_values"] or 1.0 not in spec["steepness_values"]:
        raise ValueError("Both sweeps must include the shared baseline.")
    output = (args.output_dir or results_root()/spec["family"]/spec["revision"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = _sweep_contract(spec)
    contract_path = output / "run-contract.json"
    if contract_path.exists():
        saved_contract = json.loads(contract_path.read_text())
        if args.collect:
            # Reporting may evolve; preserve the runtime contract and verify its
            # mathematical evaluator, source snapshot and all saved endpoints.
            if saved_contract["config"] != CONFIG or saved_contract["sweep"] != spec:
                raise ValueError("Report configuration differs from the saved run.")
            driver = str(Path(__file__).relative_to(ROOT))
            with zipfile.ZipFile(output / "source_snapshot.zip") as archive:
                for name, digest in saved_contract["source_sha256"].items():
                    if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                        raise ValueError(f"Changed runtime source snapshot: {name}")
                    if name != driver and _sha256(ROOT/name) != digest:
                        raise ValueError(f"Use the original mathematical sources for replay: {name}")
                original_tree = ast.parse(archive.read(driver).decode())
            current_tree = ast.parse(Path(__file__).read_text())
            original_evaluator = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == "band_values")
            current_evaluator = next(n for n in current_tree.body if isinstance(n, ast.FunctionDef) and n.name == "band_values")
            if ast.dump(original_evaluator) != ast.dump(current_evaluator):
                raise ValueError("Saved-band evaluation changed; cannot silently replot.")
            (output / "report-source.py").write_bytes(Path(__file__).read_bytes())
            _write_json(output / "report-provenance.json", {
                "runtime_contract_sha256": _sha256(contract_path),
                "report_source_sha256": _sha256(Path(__file__)),
                "report_repository_head": contract["repository_head"],
                "mode": "Saved fits and repository optimizer endpoints only; no fitting or optimization"})
            _collect_sweep(spec, output, saved_contract)
            return
        if saved_contract != contract:
            raise ValueError("Completed runs are immutable; choose a fresh revision/output directory.")
    else:
        _write_json(contract_path, contract)
        _write_json(output / "sweep-config.json", spec)
        math_text = (ROOT / "MATH.md").read_text()
        setup = math_text.split("### 7.17 ", 1)[1].split("### 7.18 ", 1)[0]
        (output / "EXPERIMENT.md").write_text(
            "# " + setup + "\n## Reproduction\n\n"
            "Run from the repository with `conda activate simulation_env`, then\n"
            "`python scratch/run_wild_data_location.py --sweep-config "
            "scratch/wild_data_location_sweeps.json --launch slurm --array "
            "--array-max-parallel 8`.\n\n"
            "For report-only replay use the same command with `--launch local --collect` "
            "instead of Slurm flags. The source snapshot, run contract, paired draws, "
            "fitted bands, all solver queries and endpoints are retained here. "
            "Original illustration outputs remain in the parent folder.\n")
        with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in contract["source_sha256"]:
                archive.write(ROOT / name, name)
    profile = replace(CPU_PROFILE, cpus_per_task=1, memory="4G", time="01:00:00",
                      job_name="wild-location-sweeps", output=str(output / "slurm/%x-%A_%a.out"))
    plan = LaunchPlan(
        name=output.name, task_count=spec["repetitions"], requires_jax=False,
        run_task=lambda index, context: _run_sweep_repetition(spec, output, index, contract),
        collect=lambda context: _collect_sweep(spec, output, contract),
        runs_root=str(output.parent), default_launch="local", default_array=True,
        slurm_profile=profile)
    run_launch_plan(plan, args=args, argv=sys.argv, cwd=ROOT)


def plot_figures(output, figure_dir, bands, summary):
    plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "figure.titlesize": 16})
    grid = np.linspace(*CONFIG["display_window"], 1201)
    truth = features(grid) @ CONFIG["truth_coefficients"]
    reference = summary["truth_reference"]
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    ax.plot(grid, truth, label=r"True objective $f(a)=5a-5a^2$")
    ax.scatter([reference["action"]], [reference["value"]], marker="*", s=110,
               color="C3", zorder=5, label=r"True optimum $a^\star=0.5$")
    ax.set(xlabel="Action $a$", ylabel="Objective value", title="True objective")
    ax.legend()
    fig.savefig(figure_dir / "01_true_objective.pdf", format="pdf")
    plt.close(fig)
    for index, (label, band) in enumerate(bands.items(), start=2):
        density_grid = np.linspace(*CONFIG["logging_windows"][index-2], 1201)
        result = summary["settings"][label]
        fitted, radius = band_values(band, grid)
        fig, (ax, data_ax) = plt.subplots(2, 1, figsize=(7, 8),
                                        constrained_layout=True)
        ax.plot(grid, truth, color="C0", label="True objective")
        ax.fill_between(grid, fitted-radius, fitted+radius, color="C1", alpha=0.25,
                        label="Nominal 95% wild-bootstrap band")
        ax.plot(grid, fitted, "--", color="C1", label=r"OLS fit $\widehat f$")
        ax.plot(grid, fitted-radius, color="C2", label=r"Lower bound $\bar f_{0.05}$")
        ax.scatter([reference["action"]], [reference["value"]], marker="*", s=110,
                   color="C3", zorder=5, label=r"True optimum $a^\star$")
        best = result["selected"]
        ax.scatter([best["action"]], [best["lower_value"]], marker="X", s=70,
                   color="C4", zorder=6,
                   label=rf"Stein endpoint $\hat\theta={best['action']:.3f}$")
        ax.set(xlabel="Action $a$", ylabel="Objective value",
               title=f"Setting {index-1}: data {'near' if index == 2 else 'far from'} the optimum")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2)
        density = norm.pdf(density_grid, loc=result["mean"], scale=np.sqrt(CONFIG["variance"]))
        data_ax.fill_between(density_grid, density, color="C0", alpha=0.25,
                             label=rf"Logging density $\mathcal{{N}}({result['mean']},\,0.45)$")
        data_ax.hist(band["actions"], bins=24, density=True, histtype="step", color="C0",
                     label="300 observed actions")
        data_ax.axvline(reference["action"], color="C3", linestyle=":", label=r"$a^\star$")
        data_ax.axvline(best["action"], color="C4", linestyle="--", label=r"$\hat\theta$")
        data_ax.set(xlabel="Action $a$", ylabel="Action density")
        data_ax.legend()
        fig.savefig(figure_dir / f"{index:02d}_{label}.pdf", format="pdf")
        plt.close(fig)
        np.savetxt(output / f"{label}_plot.csv",
                   np.column_stack((grid, truth, fitted, radius, density_grid, density)), delimiter=",",
                   header="action,truth,fit,half_width,density_action,logging_density", comments="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=None)
    parser.add_argument("--figure-dir", type=Path,
                        default=ROOT.parent / "MEng-Thesis/figures/10-21-update-mean10")
    parser.add_argument("--sweep-config", type=Path)
    add_launch_args(parser, default_launch="local", default_array=True)
    args = parser.parse_args()
    if args.sweep_config is not None:
        _launch_sweep(args)
        return
    if args.output_dir is None:
        args.output_dir = ROOT.parent / "results/wild-data-location-20261021-mean10"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    seeds = resolve_seed_setup(SeedSetup(run_seed=CONFIG["run_seed"]), CONFIG["run_seed"])
    bootstrap_seed = derive_seed(seeds.run_seed, "wild-bootstrap:multipliers")
    z = np.random.default_rng(seeds.data_seed).normal(size=CONFIG["n"])
    noise = np.random.default_rng(seeds.noise_seed).normal(size=CONFIG["n"])
    multipliers = np.random.default_rng(bootstrap_seed).choice(
        [-1., 1.], size=(CONFIG["bootstrap_draws"], CONFIG["n"])
    )
    np.savez_compressed(args.output_dir / "paired_draws.npz", design_innovations=z,
                        noise_innovations=noise, multipliers=multipliers)
    beta0 = np.array(CONFIG["truth_coefficients"])
    truth_objective = TransferredLCBObjective(beta0, beta0, np.eye(3), 0.)
    theta, trace = run_first_order_minimize(
        np.array([0.]), np.zeros((1, 1)), truth_objective,
        t_steps=100, n_grad_samples=1, sigma=0., grad_norm_tol=1e-10, ftol=1e-14,
    )
    reference = {"action": float(theta[0]), "value": -truth_objective.value(theta, None),
                 "solver": "optimization.solvers.run_first_order_minimize",
                 "success": bool(trace.optimizer_success),
                 **certify_action(beta0, np.eye(3), 0., float(theta[0]))}
    np.savez_compressed(args.output_dir / "truth_trace.npz",
                        theta_values=trace.theta_values, losses=trace.objective_values)
    summary = {"config": CONFIG, "seeds": {**seeds.to_dict(), "bootstrap_seed": bootstrap_seed},
               "truth_reference": reference, "settings": {}}
    bands = {}
    for label, mean in zip(("near_optimum", "far_from_optimum"), CONFIG["means"]):
        actions = mean + np.sqrt(CONFIG["variance"]) * z
        responses = features(actions) @ beta0 + CONFIG["noise_sd"] * noise
        print(f"Calibrating {label} on all real actions...", flush=True)
        band = fit_band(actions, responses, multipliers)
        bands[label] = band
        np.savez_compressed(args.output_dir / f"{label}_band.npz", **band)
        attempts, selected = optimize_band(band, seeds.optimizer_seed, args.output_dir, label)
        fit_at_true, radius_at_true = band_values(band, [reference["action"]])
        certificate = QuadraticErrorCertificate(band["covariance"], 1.)
        covered = certificate.contains(certificate.difference(band["beta"], beta0), band["critical"])
        summary["settings"][label] = {
            "mean": mean, "beta": band["beta"].tolist(), "critical": band["critical"],
            "critical_low": band["critical_low"], "max_leverage": band["max_leverage"],
            "half_width_at_true_optimum": float(radius_at_true[0]),
            "fit_at_true_optimum": float(fit_at_true[0]),
            "band_at_true_optimum": [float(fit_at_true[0]-radius_at_true[0]),
                                     float(fit_at_true[0]+radius_at_true[0])],
            "observations_within_0p25_of_true_optimum": int(np.sum(np.abs(actions-reference["action"]) <= .25)),
            "realized_all_real_containment": bool(covered), "attempts": attempts,
            "selected": selected,
            "true_regret": float(reference["value"] - features([selected["action"]])[0] @ beta0),
        }
        print(json.dumps(summary["settings"][label], indent=2), flush=True)
    plot_figures(args.output_dir, args.figure_dir, bands, summary)
    source_files = [Path(__file__), ROOT / "MATH.md"]
    source_files += list((ROOT / "src/optimization").rglob("*.py"))
    source_files += [ROOT / "src/experiments" / name for name in
                     ("envelopes/quadratic.py", "offline_bootstrap_math.py",
                      "seeds/streams.py")]
    summary["provenance"] = {
        "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.figure_dir.glob("*.pdf")},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert reference["certified"]
    assert all(a["certified"] for s in summary["settings"].values() for a in s["attempts"]), \
        "A solver endpoint failed the declared global gap certificate; inspect saved results."
    print(f"Saved results to {args.output_dir} and figures to {args.figure_dir}", flush=True)


if __name__ == "__main__":
    main()
