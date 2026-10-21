"""Flat contextual wild-bootstrap illustration with personalized Stein actions (§7.9)."""

from __future__ import annotations

import argparse
import csv
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from experiments.bootstrap_band import features as action_features
from experiments.bootstrap_band_continuous import QuadraticErrorCertificate
from experiments.seeds import SeedSetup, derive_seed, resolve_seed_setup
from objective.base import Objective, Policy
from optimization.solvers import run_first_order_minimize, run_stein_difference_minimize
from scratch.run_wild_data_location import CONFIG as SCALAR_CONFIG, certify_action

CONFIG = {
    **SCALAR_CONFIG, "context_dim": 3, "truth_coefficients": [0., 1., -.5, .25, .5, -.5],
    "display_window": [-1.5, 12.0], "calibration": "wild_coefficient_ellipsoid",
    "policy": "one_free_action_per_observed_context", "perturbation_space": "u",
}
CONFIG.pop("logging_windows")
CONFIG.pop("statistic_tolerance")


def features(contexts, actions):
    contexts = np.asarray(contexts, dtype=float)
    actions = np.asarray(actions, dtype=float)
    return np.column_stack((np.ones(len(contexts)), contexts, actions, actions**2))


def fibers(contexts, beta, inverse):
    """Collapse each fixed context to its quadratic coefficients and QR basis."""
    n, d = contexts.shape
    maps = np.zeros((n, d+3, 3))
    maps[:, 0, 0] = 1.
    maps[:, 1:d+1, 0] = contexts
    maps[:, -2, 1] = 1.
    maps[:, -1, 2] = 1.
    coefficients = np.einsum("npk,p->nk", maps, beta)
    basis = np.einsum("pq,nqk->npk", inverse.T, maps)
    covariance = np.einsum("npi,npj->nij", basis, basis)
    return coefficients, basis, covariance


def fit_band(contexts, actions, responses, multipliers):
    design = features(contexts, actions)
    if len(design) <= design.shape[1] or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Require full column rank and more rows than coefficients.")
    orthogonal, triangular = np.linalg.qr(design, mode="reduced")
    beta = solve_triangular(triangular, orthogonal.T @ responses)
    residuals = responses - design @ beta
    whitened_errors = (multipliers * residuals) @ orthogonal
    errors = solve_triangular(triangular, whitened_errors.T).T
    # ||R d_b|| bounds every standardized feature error by Cauchy--Schwarz.
    statistics = np.linalg.norm(whitened_errors, axis=1)
    order = int(np.ceil(len(statistics)*(1-CONFIG["delta"]))) - 1
    critical = float(np.partition(statistics, order)[order])
    inverse = solve_triangular(triangular, np.eye(design.shape[1]))
    coefficients, basis, covariance = fibers(contexts, beta, inverse)
    return {
        "contexts": contexts, "actions": actions, "responses": responses, "beta": beta,
        "residuals": residuals, "bootstrap_errors": errors, "bootstrap_statistics": statistics,
        "triangular": triangular, "inverse": inverse, "critical": critical,
        "fiber_beta": coefficients, "fiber_basis": basis, "fiber_covariance": covariance,
        "max_leverage": float(np.max(np.sum(orthogonal**2, axis=1))),
    }


def band_at_actions(band, actions):
    """Evaluate one or more personalized action vectors, shape (..., n)."""
    actions = np.asarray(actions, dtype=float)
    p = np.stack((np.ones_like(actions), actions, actions**2), axis=-1)
    fitted = np.einsum("...nj,nj->...n", p, band["fiber_beta"])
    whitened = np.einsum("npj,...nj->...np", band["fiber_basis"], p)
    radius = band["critical"] * np.linalg.norm(whitened, axis=-1)
    return fitted, radius


class ContextActions(Policy):
    """Identity policy on the fixed full batch, with one parameter per context."""

    def value(self, theta, x_batch):
        if np.shape(theta) != (len(x_batch),):
            raise ValueError("The identity policy requires the full ordered context batch.")
        return np.asarray(theta, dtype=float)

    def weighted_grad(self, theta, x_batch, weights):
        self.value(theta, x_batch)
        return np.asarray(weights, dtype=float)


class QueryOnlyContextLCB(Objective):
    """Frozen per-context LCB values, exposed through the existing u-space API."""

    def __init__(self, band):
        self.band = band
        self.policy = ContextActions()
        self.query_count = 0
        self.query_calls = 0
        self.query_min = np.full(len(band["contexts"]), np.inf)
        self.query_max = np.full(len(band["contexts"]), -np.inf)

    def _value_batch_many(self, x_batch, u_matrix):
        if len(x_batch) != len(self.band["contexts"]):
            raise ValueError("Full-batch contextual queries are required.")
        u_matrix = np.asarray(u_matrix, dtype=float)
        self.query_count += u_matrix.size
        self.query_calls += 1
        self.query_min = np.minimum(self.query_min, np.min(u_matrix, axis=0))
        self.query_max = np.maximum(self.query_max, np.max(u_matrix, axis=0))
        fitted, radius = band_at_actions(self.band, u_matrix)
        return radius - fitted

    def _value_batch(self, x_batch, actions):
        return self._value_batch_many(x_batch, np.asarray(actions)[None, :])[0]

    def value(self, theta, x_batch):
        return float(np.mean(self._value_batch(x_batch, self.policy.value(theta, x_batch))))

    def grad(self, theta, x_batch):
        raise AssertionError("The fitted objective must be queried through function values only.")


class TruthObjective(Objective):
    policy = ContextActions()

    def value(self, theta, x_batch):
        return -float(np.mean(features(x_batch, theta) @ CONFIG["truth_coefficients"]))

    def grad(self, theta, x_batch):
        beta = CONFIG["truth_coefficients"]
        return -(beta[-2] + 2*beta[-1]*np.asarray(theta)) / len(x_batch)


def optimize_band(band, optimizer_seed, output, label):
    n = len(band["contexts"])
    attempts, action_vectors, values = [], [], []
    for index, start in enumerate(CONFIG["starts"]):
        seed = derive_seed(optimizer_seed, f"stein-difference:start:{index}")
        batch_seed = derive_seed(optimizer_seed, f"batch:start:{index}")
        objective = QueryOnlyContextLCB(band)
        theta, trace = run_stein_difference_minimize(
            np.full(n, start), band["contexts"], objective, np.random.default_rng(seed),
            t_steps=CONFIG["t_steps"], n_grad_samples=CONFIG["n_grad_samples"],
            sigma=CONFIG["smoothing_sd"], perturbation_space="u", algorithm="constant",
            step_size=n*CONFIG["step_size"],
            grad_norm_tol=CONFIG["gradient_tolerance"]/np.sqrt(n),
            gradient_rng=np.random.default_rng(seed), batch_rng=np.random.default_rng(batch_seed),
        )
        fitted, radius = band_at_actions(band, theta)
        certificates = [certify_action(beta, covariance, band["critical"], float(action))
                        for beta, covariance, action in zip(
                            band["fiber_beta"], band["fiber_covariance"], theta)]
        attempt = {
            "start": start, "seed": seed, "batch_seed": batch_seed,
            "mean_action": float(np.mean(theta)), "action_min": float(np.min(theta)),
            "action_max": float(np.max(theta)), "mean_personalized_lcb": float(np.mean(fitted-radius)),
            "success": bool(trace.optimizer_success), "message": trace.optimizer_message,
            "steps": int(trace.steps[-1]), "scalar_query_count": objective.query_count,
            "query_calls": objective.query_calls,
            "certified_contexts": sum(c["certified"] for c in certificates),
            "certificates": certificates,
        }
        attempts.append(attempt)
        action_vectors.append(theta)
        values.append(fitted-radius)
        np.savez_compressed(output / f"{label}_start_{index}.npz",
                            theta_values=np.asarray(trace.theta_values), losses=trace.objective_values,
                            gradient_norms=trace.theta_grad_norms, final_actions=theta,
                            query_min=objective.query_min, query_max=objective.query_max)
        print(f"{label}, start {start}: mean action {np.mean(theta):.6f}; "
              f"{attempt['certified_contexts']}/{n} certified", flush=True)
    # Each eligible action is an actual repository endpoint for that context.
    selected_runs = np.array([max(range(len(attempts)), key=lambda k: values[k][i])
                              for i in range(n)])
    selected = np.array([action_vectors[k][i] for i, k in enumerate(selected_runs)])
    return attempts, np.asarray(action_vectors), selected, selected_runs


def plot_figures(output, figure_dir, bands, summary, contexts):
    plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10, "figure.titlesize": 16})
    grid = np.linspace(*CONFIG["display_window"], 1501)
    context_mean = float(np.mean(contexts @ np.asarray(CONFIG["truth_coefficients"])[1:4]))
    truth = context_mean + .5*grid - .5*grid**2
    reference = summary["truth_reference"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(grid, truth, label=r"Mean truth: context term $+0.5a-0.5a^2$")
    ax.scatter([reference["mean_action"]], [reference["mean_value"]], marker="*", s=110,
               color="C3", zorder=5, label=r"Mean true optimum $\overline{a^\star}=0.5$")
    ax.set(xlabel="Action $a$", ylabel="Mean objective value", title="Flat objective, three covariates")
    ax.legend()
    fig.savefig(figure_dir / "01_true_objective.pdf", format="pdf")
    plt.close(fig)
    for index, (label, band) in enumerate(bands.items(), start=2):
        result = summary["settings"][label]
        actions = np.broadcast_to(grid[:, None], (len(grid), len(contexts)))
        fitted, radius = band_at_actions(band, actions)
        fitted, radius = fitted.mean(axis=1), radius.mean(axis=1)
        density = norm.pdf(grid, loc=result["mean"], scale=np.sqrt(CONFIG["variance"]))
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        density_ax = ax.twinx()
        density_ax.fill_between(grid, density, color="C5", alpha=.15,
                                label=rf"Action density $\mathcal{{N}}({result['mean']:g},0.45)$")
        density_ax.hist(band["actions"], bins=24, density=True, histtype="step", color="C5",
                        label="300 historical actions")
        ax.plot(grid, truth, color="C0", label="Mean true objective")
        ax.fill_between(grid, fitted-radius, fitted+radius, color="C1", alpha=.25,
                        label="Mean nominal 95% band")
        ax.plot(grid, fitted, "--", color="C1", label=r"Mean fit $\widehat f$")
        ax.plot(grid, fitted-radius, color="C2", label=r"Mean lower bound $\bar f_{0.05}$")
        ax.scatter([reference["mean_action"]], [reference["mean_value"]], marker="*", s=110,
                   color="C3", zorder=5, label=r"Mean true optimum $\overline{a^\star}$")
        ax.scatter([result["mean_selected_action"]], [result["mean_personalized_lcb"]],
                   marker="X", s=70, color="C4", zorder=6,
                   label=rf"Personalized mean $\overline{{\hat a}}={result['mean_selected_action']:.3f}$")
        ax.set(xlabel="Action $a$", ylabel="Mean objective value",
               title=f"Setting {index-1}: historical action mean {result['mean']:g}")
        density_ax.set_ylabel("Action density")
        handles, labels = ax.get_legend_handles_labels()
        dh, dl = density_ax.get_legend_handles_labels()
        fig.legend(handles+dh, labels+dl, loc="outside upper center", ncol=2)
        fig.savefig(figure_dir / f"{index:02d}_{label}.pdf", format="pdf")
        plt.close(fig)
        np.savetxt(output / f"{label}_plot.csv", np.column_stack((grid, truth, fitted, radius, density)),
                   delimiter=",", header="action,mean_truth,mean_fit,mean_half_width,action_density", comments="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT.parent / "results/wild-contextual-flat-20261021")
    parser.add_argument("--figure-dir", type=Path,
                        default=ROOT.parent / "MEng-Thesis/figures/10-21-update-contextual-flat")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    seeds = resolve_seed_setup(SeedSetup(run_seed=CONFIG["run_seed"]), CONFIG["run_seed"])
    context_seed = derive_seed(seeds.run_seed, "data:contexts")
    bootstrap_seed = derive_seed(seeds.run_seed, "wild-bootstrap:multipliers")
    contexts = np.random.default_rng(context_seed).normal(size=(CONFIG["n"], 3))
    z = np.random.default_rng(seeds.data_seed).normal(size=CONFIG["n"])
    noise = np.random.default_rng(seeds.noise_seed).normal(size=CONFIG["n"])
    multipliers = np.random.default_rng(bootstrap_seed).choice(
        [-1., 1.], size=(CONFIG["bootstrap_draws"], CONFIG["n"]))
    np.savez_compressed(args.output_dir / "paired_draws.npz", contexts=contexts,
                        design_innovations=z, noise_innovations=noise, multipliers=multipliers)
    truth_objective = TruthObjective()
    theta, trace = run_first_order_minimize(
        np.zeros(len(contexts)), contexts, truth_objective, t_steps=100,
        n_grad_samples=1, sigma=0., grad_norm_tol=1e-12, ftol=1e-14)
    truth_coefficients = np.column_stack((contexts @ np.asarray(CONFIG["truth_coefficients"])[1:4],
                                         np.full(len(contexts), .5), np.full(len(contexts), -.5)))
    truth_certificates = [certify_action(beta, np.eye(3), 0., float(a))
                          for beta, a in zip(truth_coefficients, theta)]
    reference = {"mean_action": float(np.mean(theta)), "mean_value": -truth_objective.value(theta, contexts),
                 "solver": "optimization.solvers.run_first_order_minimize",
                 "success": bool(trace.optimizer_success), "actions": theta.tolist(),
                 "certified_contexts": sum(c["certified"] for c in truth_certificates)}
    np.savez_compressed(args.output_dir / "truth_trace.npz", theta_values=trace.theta_values,
                        losses=trace.objective_values)
    summary = {"config": CONFIG, "seeds": {**seeds.to_dict(), "context_seed": context_seed,
                                            "bootstrap_seed": bootstrap_seed},
               "truth_reference": reference, "settings": {}}
    bands = {}
    for label, mean in zip(("near_optimum", "far_from_optimum"), CONFIG["means"]):
        actions = mean + np.sqrt(CONFIG["variance"])*z
        responses = features(contexts, actions) @ CONFIG["truth_coefficients"] + CONFIG["noise_sd"]*noise
        band = fit_band(contexts, actions, responses, multipliers)
        bands[label] = band
        np.savez_compressed(args.output_dir / f"{label}_band.npz", **band)
        attempts, vectors, selected, selected_runs = optimize_band(
            band, seeds.optimizer_seed, args.output_dir, label)
        fitted, radius = band_at_actions(band, selected)
        at_true, width_at_true = band_at_actions(band, theta)
        certificates = [QuadraticErrorCertificate(cov, 1.)
                        for cov in band["fiber_covariance"]]
        containment = [c.contains(c.difference(beta, truth), band["critical"])
                       for c, beta, truth in zip(certificates, band["fiber_beta"], truth_coefficients)]
        coefficient_statistic = float(np.linalg.norm(
            band["triangular"] @ (band["beta"]-CONFIG["truth_coefficients"])))
        true_values = features(contexts, selected) @ CONFIG["truth_coefficients"]
        summary["settings"][label] = {
            "mean": mean, "critical": band["critical"], "beta": band["beta"].tolist(),
            "max_leverage": band["max_leverage"], "mean_selected_action": float(np.mean(selected)),
            "selected_action_min": float(np.min(selected)), "selected_action_max": float(np.max(selected)),
            "mean_personalized_lcb": float(np.mean(fitted-radius)),
            "mean_fit_at_true_optimum": float(np.mean(at_true)),
            "mean_half_width_at_true_optimum": float(np.mean(width_at_true)),
            "mean_band_at_true_optimum": [float(np.mean(at_true-width_at_true)),
                                          float(np.mean(at_true+width_at_true))],
            "half_width_range_at_true_optimum": [float(np.min(width_at_true)),float(np.max(width_at_true))],
            "mean_true_regret": float(reference["mean_value"]-np.mean(true_values)),
            "coefficient_event_covered": coefficient_statistic <= band["critical"],
            "all_contexts_all_real_containment": bool(all(containment)),
            "selected_certified_contexts": sum(attempts[k]["certificates"][i]["certified"]
                                               for i,k in enumerate(selected_runs)),
            "attempts": attempts,
        }
        with (args.output_dir / f"{label}_contexts.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["context_id", "x1", "x2", "x3", "true_action", "selected_action",
                             "selected_start", "selected_lcb", "fit_at_true", "half_width_at_true",
                             "covered_all_real_actions", "certified", "start0_action", "start1_action", "start2_action"])
            for i, k in enumerate(selected_runs):
                writer.writerow([i, *contexts[i], theta[i], selected[i], CONFIG["starts"][k],
                                 fitted[i]-radius[i], at_true[i], width_at_true[i], containment[i],
                                 attempts[k]["certificates"][i]["certified"], *vectors[:, i]])
        print(json.dumps({k:v for k,v in summary["settings"][label].items() if k != "attempts"}, indent=2), flush=True)
    plot_figures(args.output_dir, args.figure_dir, bands, summary, contexts)
    source_files = [Path(__file__), ROOT / "scratch/run_wild_data_location.py", ROOT / "MATH.md"]
    source_files += list((ROOT / "src/optimization").rglob("*.py"))
    source_files += [ROOT / "src" / name for name in (
        "objective/utils.py", "experiments/bootstrap_band.py", "experiments/bootstrap_band_continuous.py",
        "experiments/offline_bootstrap_math.py", "experiments/seeds/streams.py")]
    summary["provenance"] = {
        "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.figure_dir.glob("*.pdf")},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    assert reference["certified_contexts"] == len(contexts)
    assert all(a["certified_contexts"] == len(contexts)
               for s in summary["settings"].values() for a in s["attempts"]), \
        "Some solver endpoints are uncertified; inspect the saved artifacts."
    print(f"Saved {args.output_dir} and {args.figure_dir}", flush=True)


if __name__ == "__main__":
    main()
