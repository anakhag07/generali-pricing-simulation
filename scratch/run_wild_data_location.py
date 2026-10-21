"""Reproduce the three 10/21 quadratic wild-bootstrap figures (MATH.md §7.8)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import norm
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiments.bootstrap_band import features
from experiments.bootstrap_band_continuous import QuadraticErrorCertificate
from experiments.offline_bootstrap_math import (
    TransferredLCBObjective, _branches, _candidate_lower_value,
    transferred_below_level,
)
from experiments.seeds import SeedSetup, derive_seed, resolve_seed_setup
from objective.base import Objective
from optimization.solvers import run_first_order_minimize, run_stein_difference_minimize

CONFIG = {
    "run_seed": 20261021, "n": 300, "variance": 0.45, "means": [0.5, 2.5],
    "noise_sd": 1.0, "bootstrap_draws": 1999, "delta": 0.05,
    "starts": [-0.5, 1.5, 4.0], "step_rule": "constant", "step_size": 0.05,
    "t_steps": 1000, "n_grad_samples": 64, "smoothing_sd": 0.01,
    "gradient_tolerance": 1e-5, "statistic_tolerance": 1e-8,
    "global_gap_tolerance": 1e-6, "display_window": [-0.25, 1.25],
    "logging_window": [-1.5, 4.5],
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


def optimize_band(band, optimizer_seed, output, label):
    attempts = []
    for index, start in enumerate(CONFIG["starts"]):
        seed = derive_seed(optimizer_seed, f"stein-difference:start:{index}")
        batch_seed = derive_seed(optimizer_seed, f"batch:start:{index}")
        objective = QueryOnlyLowerBound(band)
        theta, trace = run_stein_difference_minimize(
            np.array([start]), np.zeros((1, 1)), objective,
            np.random.default_rng(seed), t_steps=CONFIG["t_steps"],
            n_grad_samples=CONFIG["n_grad_samples"], sigma=CONFIG["smoothing_sd"],
            algorithm=CONFIG["step_rule"], step_size=CONFIG["step_size"],
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


def plot_figures(output, figure_dir, bands, summary):
    plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "figure.titlesize": 16})
    grid = np.linspace(*CONFIG["display_window"], 1201)
    density_grid = np.linspace(*CONFIG["logging_window"], 1201)
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
                        default=ROOT.parent / "results/wild-data-location-20261021")
    parser.add_argument("--figure-dir", type=Path,
                        default=ROOT.parent / "MEng-Thesis/figures/10-21-update")
    args = parser.parse_args()
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
        _, radius_at_true = band_values(band, [reference["action"]])
        certificate = QuadraticErrorCertificate(band["covariance"], 1.)
        covered = certificate.contains(certificate.difference(band["beta"], beta0), band["critical"])
        summary["settings"][label] = {
            "mean": mean, "beta": band["beta"].tolist(), "critical": band["critical"],
            "critical_low": band["critical_low"], "max_leverage": band["max_leverage"],
            "half_width_at_true_optimum": float(radius_at_true[0]),
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
                     ("bootstrap_band.py", "bootstrap_band_continuous.py", "offline_bootstrap_math.py",
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
