"""Paired all-real OLS bootstrap sweeps and repository-optimized LCBs (§7.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import sympy as sp
from scipy.linalg import solve_triangular

from experiments.bootstrap_band import features
from experiments.bootstrap_band_continuous import (
    VARIABLE, QuadraticErrorCertificate, nonnegative_on_real_line,
)
from experiments.launch import LaunchPlan
from experiments.policy_lcb.common import PolicyLCBLaunchSpec, read_json, write_json_atomic, wilson_interval
from experiments.provenance import file_record
from experiments.seeds import derive_seed
from experiments.sweep_reporting import write_rows_csv
from objective.base import Objective
from optimization.solvers import run_first_order_minimize

MANIFEST_KIND = "bootstrap_ols_controlled_sweep"


def _poly(coefficients):
    return sp.Poly.from_list([sp.Rational(float(c)) for c in coefficients[::-1]], VARIABLE, domain=sp.QQ)


def lcb_below_level(beta, radius_squared, level):
    """Certify an LCB upper level on R; this verifies, never selects, actions."""
    g = _poly(beta) - sp.Rational(float(level))
    if g.is_zero or nonnegative_on_real_line(-g):
        return True
    s = radius_squared - g**2
    if s.is_zero or nonnegative_on_real_line(s):
        return True
    # Both strict inequalities define an open set. Sample each exact sign cell
    # of g*s, not an evaluation grid. Isolating intervals contain one root each.
    product = (g*s).sqf_part()
    precision = None
    for _ in range(100):
        intervals = [interval for interval, _ in product.intervals(eps=precision)]
        if not intervals:
            samples = [sp.Rational(0)]
        else:
            samples = [intervals[0][0]-1, intervals[-1][1]+1]
            samples += [(left[1]+right[0])/2 for left, right in zip(intervals, intervals[1:])]
        # SymPy may return (-1,0) next to an exact root (0,0). Their midpoint
        # is then the root, not a sample of the intervening open sign cell.
        if all(product.eval(a) != 0 for a in samples):
            break
        precision = sp.Rational(1, 10**8) if precision is None else precision/100
    else:
        raise RuntimeError("Could not separate adjacent algebraic sign cells.")
    return not any(g.eval(a) > 0 and s.eval(a) < 0 for a in samples)


class QuadraticLCBObjective(Objective):
    """Policy-free negative LCB, with analytical gradient and no action bounds."""

    def __init__(self, beta, covariance, scale):
        self.beta = np.asarray(beta, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self.scale = float(scale)

    def value(self, theta, x_batch):
        a = float(theta[0])
        p = np.array([1., a, a*a])
        return float(-p @ self.beta + self.scale*np.sqrt(p @ self.covariance @ p))

    def grad(self, theta, x_batch):
        a = float(theta[0])
        p, dp = np.array([1., a, a*a]), np.array([0., 1., 2*a])
        return np.array([-dp @ self.beta + self.scale*(dp @ self.covariance @ p)/np.sqrt(p @ self.covariance @ p)])


def _candidate_lower_value(beta, radius_squared, action):
    """Outward-round a candidate value before certifying its objective gap."""
    a = sp.Rational(float(action))
    q = radius_squared.eval(a)
    root_upper = float(np.nextafter(np.sqrt(float(q)), np.inf)) if q else 0.
    while sp.Rational(root_upper)**2 < q:
        root_upper = float(np.nextafter(root_upper, np.inf))
    exact_lower = _poly(beta).eval(a) - sp.Rational(root_upper)
    return float(np.nextafter(float(exact_lower), -np.inf))


def optimize_lcb(beta, covariance, sigma_hat, critical, config):
    """Use repository multistart optimization and certify its all-real gap."""
    cert = QuadraticErrorCertificate(covariance, sigma_hat)
    radius_squared = cert.q.mul_ground(sp.Rational(float(critical))**2)
    leading = sp.Rational(float(beta[2]))
    tail_square = radius_squared.nth(4)
    # Comparing squares is exact when beta_2 >= 0; negative beta_2 is coercive.
    if leading >= 0 and leading**2 >= tail_square:
        return {"optimization_state": "unbounded" if leading**2 > tail_square else "degenerate_tail",
                "attempts": [], "action": None, "global_gap_upper": None}
    objective = QuadraticLCBObjective(beta, covariance, critical*sigma_hat)
    attempts = []
    for start in config["starts"]:
        theta, trace = run_first_order_minimize(
            np.array([start], dtype=float), np.zeros((1, 1)), objective,
            t_steps=config["t_steps"], n_grad_samples=1, sigma=0.,
            algorithm=config["step_rule"], grad_norm_tol=config["gradient_tolerance"],
            ftol=config["ftol"],
        )
        a = float(theta[0])
        attempts.append({"start": start, "action": a,
                         "value": float(-objective.value(theta, None)),
                         "success": bool(trace.optimizer_success),
                         "status": trace.optimizer_status, "message": trace.optimizer_message,
                         "gradient_norm": float(np.linalg.norm(objective.grad(theta, None)))})
    finite = [attempt for attempt in attempts if np.isfinite([attempt["action"], attempt["value"]]).all()]
    if not finite:
        return {"optimization_state": "optimizer_failed", "attempts": attempts,
                "action": None, "global_gap_upper": None}
    # Select only among actual repository solver outputs, never grid/root actions.
    best = max(finite, key=lambda attempt: attempt["value"])
    lower = _candidate_lower_value(beta, radius_squared, best["action"])
    upper = float(np.nextafter(lower+config["global_gap_tolerance"], np.inf))
    certified = lcb_below_level(beta, radius_squared, upper)
    return {"optimization_state": "certified" if certified else "uncertified",
            "action": best["action"], "attempts": attempts,
            "solver_success": best["success"], "candidate_value_lower": lower,
            "global_value_upper": upper if certified else None,
            "global_gap_upper": upper-lower if certified else None}


@dataclass(frozen=True)
class SweepManifest:
    name: str
    payload: dict
    source_path: Path
    launch: PolicyLCBLaunchSpec


def load_manifest(path):
    """Validate the controlled sweep's fixed domains, axes, and seed streams."""
    path = Path(path).resolve()
    p = read_json(path)
    if p.get("kind") != MANIFEST_KIND or p.get("domain") != "real_line":
        raise ValueError("Require the controlled all-real bootstrap manifest.")
    if not p["name"] or Path(p["name"]).name != p["name"] or p["name"] in {".", ".."}:
        raise ValueError("Require a safe project name.")
    if p["design"] != {"type": "iid_normal", "mean": 0.0, "std": 1.0}:
        raise ValueError("This controlled experiment fixes standard normal training inputs.")
    if p["truth"] != {"coefficients": [0., 5., -5.]}:
        raise ValueError("This experiment fixes f(a)=5a-5a^2.")
    if not 0 < p["delta"] < 1 or p["datasets"] < 2:
        raise ValueError("Require 0<delta<1 and at least two datasets.")
    for key, lower in (("N", 4), ("B", 2), ("sigma", 0)):
        values = p["axes"][key]
        if not values or values != sorted(set(values)) or p["baseline"][key] not in values:
            raise ValueError("Axes must be sorted, unique, and contain the baseline.")
        if any(not np.isfinite(v) or (v <= 0 if key == "sigma" else v < lower or int(v) != v) for v in values):
            raise ValueError("Invalid sweep values.")
    if set(p["seeds"]) != {"master", "design", "observation", "bootstrap"}:
        raise ValueError("Separate named seed streams are required.")
    if any(not isinstance(seed, int) or seed < 0 for seed in p["seeds"].values()):
        raise ValueError("Seed roots must be nonnegative integers.")
    if not 0 < p["supremum_tolerance"] < 1e-3:
        raise ValueError("Invalid supremum tolerance.")
    opt = p["optimizer"]
    if opt["step_rule"] != "l-bfgs-b" or not opt["starts"] or any(not 0 <= a <= 1 for a in opt["starts"]):
        raise ValueError("Require repository L-BFGS-B with starts in [0,1], without bounds.")
    if any(not np.isfinite(opt[k]) or opt[k] <= 0 for k in ("t_steps", "gradient_tolerance", "ftol", "global_gap_tolerance")):
        raise ValueError("Invalid optimizer tolerances.")
    if p["launch"] != {"mode": "local", "array": "none"}:
        raise ValueError("Use a local non-array launch.")
    return SweepManifest(p["name"], p, path, PolicyLCBLaunchSpec(**p["launch"]))


def settings(payload):
    """Yield each axis setting; the shared baseline is intentionally repeated."""
    for axis, values in payload["axes"].items():
        for value in values:
            yield axis, value, {**payload["baseline"], axis: value}


def dataset_streams(payload, index):
    """Generate nested observations and bootstrap columns with independent seeds."""
    seeds = {key: derive_seed(payload["seeds"][key], f"{payload['seeds']['master']}:dataset:{index}")
             for key in ("design", "observation", "bootstrap")}
    nmax, bmax = max(payload["axes"]["N"]), max(payload["axes"]["B"])
    x = np.random.default_rng(seeds["design"]).normal(size=nmax)
    epsilon = np.random.default_rng(seeds["observation"]).normal(size=nmax)
    z = np.random.default_rng(seeds["bootstrap"]).normal(size=(bmax, nmax))
    return seeds, x, epsilon, z


def calibrate_design(x, z, tolerance):
    """Calibrate standardized bootstrap refits using design and Gaussian draws only."""
    p = features(x)
    if len(x) <= 3 or np.linalg.matrix_rank(p) != 3:
        raise ValueError("Require full-rank quadratic OLS with N>3.")
    q, triangular = np.linalg.qr(p, mode="reduced")
    inverse = solve_triangular(triangular, np.eye(3))
    covariance = inverse @ inverse.T
    d = solve_triangular(triangular, q.T @ z.T).T
    cert = QuadraticErrorCertificate(covariance, 1.)
    bounds = np.array([cert.supremum_interval(cert.difference(draw, np.zeros(3)), tolerance) for draw in d])
    return p, q, triangular, covariance, d, bounds


def _source_contract():
    return [file_record(Path(__file__).with_name(name)) for name in
            ("bootstrap_band_sweep.py", "bootstrap_band_continuous.py")]+[
        file_record(Path(__file__).parents[1]/"optimization"/name)
        for name in ("base.py", "solvers.py", "gradients/methods.py")]


def run_dataset(manifest, index, runs_root, force=False):
    """Compute a paired dataset block, saving sufficient statistics and provenance."""
    started = time.monotonic()
    payload = manifest.payload
    project = Path(runs_root)/manifest.name
    destination = project/"datasets"/f"dataset-{index:03d}"
    contract = {"manifest": payload, "dataset": index, "sources": _source_contract()}
    summary_path = destination/"summary.json"
    if summary_path.exists() and not force:
        previous = read_json(summary_path)
        if previous["contract"] != contract or previous["arrays"] != file_record(destination/"draws.npz"):
            raise ValueError("Cached sweep contract/artifacts changed; use a new name or explicit --force.")
        return previous
    seeds, x, epsilon, z = dataset_streams(payload, index)
    beta0 = np.array(payload["truth"]["coefficients"])
    reference = optimize_lcb(beta0, np.eye(3), 1., 0., payload["optimizer"])
    if reference["optimization_state"] != "certified":
        raise RuntimeError("Repository true-reference optimizer did not certify.")
    a_star = reference["action"]
    arrays = {"training_actions": x, "standardized_observation_errors": epsilon}
    fits, solved, rows = {}, {}, []
    for n in payload["axes"]["N"]:
        bmax = max(payload["axes"]["B"]) if n == payload["baseline"]["N"] else payload["baseline"]["B"]
        p, q, triangular, covariance, d, bounds = calibrate_design(x[:n], z[:bmax, :n], payload["supremum_tolerance"])
        fits[n] = p, q, triangular, covariance, d, bounds
        arrays.update({f"N{n}_covariance": covariance, f"N{n}_bootstrap_unit_perturbations": d,
                       f"N{n}_bootstrap_supremum_bounds": bounds})
    for axis, axis_value, setting in settings(payload):
        n, sigma, b = setting["N"], setting["sigma"], setting["B"]
        key = (n, sigma, b)
        if key not in solved:
            p, q, triangular, covariance, d, bounds = fits[n]
            y = p @ beta0 + sigma*epsilon[:n]
            beta = solve_triangular(triangular, q.T @ y)
            residual = y-p @ beta
            sigma_hat = float(np.linalg.norm(residual)/np.sqrt(n-3))
            critical_low, critical = np.quantile(bounds[:b], 1-payload["delta"], axis=0, method="higher")
            cert = QuadraticErrorCertificate(covariance, sigma_hat)
            error = cert.difference(beta, beta0)
            covered = cert.contains(error, float(critical))
            observed_bounds = cert.supremum_interval(error, payload["supremum_tolerance"])
            identity = solve_triangular(triangular, q.T @ (sigma*epsilon[:n]))
            identity_error = float(np.max(np.abs(identity-(beta-beta0))))
            if identity_error > 1e-9*max(1., np.max(np.abs(identity))):
                raise RuntimeError("OLS error identity failed.")
            outcome = optimize_lcb(beta, covariance, sigma_hat, float(critical), payload["optimizer"])
            p_star = features(np.array([a_star]))[0]
            width = float(critical*sigma_hat*np.sqrt(p_star @ covariance @ p_star))
            valid = outcome["optimization_state"] == "certified"
            regret = float(p_star @ beta0-features(np.array([outcome["action"]]))[0] @ beta0) if valid else None
            regret_bound = 2*width
            if valid and covered and regret > regret_bound+outcome["global_gap_upper"]+reference["global_gap_upper"]:
                raise RuntimeError("Covered certified optimizer violated the regret inequality.")
            prefix = f"N{n}_sd{sigma:g}_B{b}"
            h = cert.threshold_polynomial(error**2, float(critical))
            arrays.update({f"{prefix}_y": y, f"{prefix}_beta_hat": beta,
                           f"{prefix}_error_coefficients": beta-beta0,
                           f"{prefix}_bootstrap_beta": beta+sigma_hat*d[:b],
                           f"{prefix}_containment_exact_coefficients": np.array([str(h.nth(i)) for i in range(5)])})
            solved[key] = {
                "dataset": index, "N": n, "sigma": sigma, "B": b,
                "covered": covered, "width_at_true_optimum": width,
                "regret": regret, "regret_bound": regret_bound,
                "critical_value": float(critical), "critical_value_lower": float(critical_low),
                "sigma_hat": sigma_hat, "sigma2_hat": sigma_hat**2,
                "observed_statistic_lower": observed_bounds[0], "observed_statistic_upper": observed_bounds[1],
                "identity_error": identity_error, "true_action": a_star,
                "lcb_action": outcome["action"], "optimization_state": outcome["optimization_state"],
                "solver_success": outcome.get("solver_success"), "global_gap_upper": outcome["global_gap_upper"],
                "optimizer_attempts": outcome["attempts"], "array_prefix": prefix,
            }
        rows.append({**solved[key], "axis": axis, "axis_value": axis_value})
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination/"draws.npz", **arrays)
    summary = {"contract": contract, "seeds": seeds, "reference": reference, "rows": rows,
               "arrays": file_record(destination/"draws.npz"), "elapsed_seconds": time.monotonic()-started}
    write_json_atomic(summary_path, summary)
    print(f"Dataset {index+1}/{payload['datasets']}: {summary['elapsed_seconds']:.1f}s", flush=True)
    return summary


def _mean_se(values):
    a = np.asarray(values, dtype=float)
    return (float(np.mean(a)), float(np.std(a, ddof=1)/np.sqrt(len(a))) if len(a)>1 else None) if len(a) else (None, None)


def aggregate(rows):
    """Aggregate independent datasets; never hide missing regret denominators."""
    result = []
    for axis, value in dict.fromkeys((r["axis"], r["axis_value"]) for r in rows):
        group = [r for r in rows if r["axis"] == axis and r["axis_value"] == value]
        count, covered = len(group), sum(r["covered"] for r in group)
        lower, upper = wilson_interval(covered, count)
        row = {"axis": axis, "axis_value": value, "datasets": count,
               "N": group[0]["N"], "sigma": group[0]["sigma"], "B": group[0]["B"],
               "covered_count": covered, "coverage": covered/count,
               "coverage_lower": lower, "coverage_upper": upper}
        for metric in ("width_at_true_optimum", "regret", "regret_bound", "critical_value"):
            row[f"{metric}_mean"], row[f"{metric}_se"] = _mean_se([r[metric] for r in group if r[metric] is not None])
        for state in ("certified", "unbounded", "degenerate_tail", "optimizer_failed", "uncertified"):
            row[f"{state}_count"] = sum(r["optimization_state"] == state for r in group)
        row["solver_warning_count"] = sum(r["solver_success"] is False for r in group)
        result.append(row)
    return result


def collect(manifest, runs_root):
    """Validate completed blocks and render three metric-by-axis PDF figures."""
    project = Path(runs_root)/manifest.name
    records = [project/"datasets"/f"dataset-{i:03d}"/"summary.json" for i in range(manifest.payload["datasets"])]
    summaries = [read_json(path) for path in records]
    for i, saved in enumerate(summaries):
        if saved["contract"] != {"manifest": manifest.payload, "dataset": i, "sources": _source_contract()}:
            raise ValueError("Incomplete or stale dataset collection.")
        if saved["arrays"] != file_record(records[i].parent/"draws.npz"):
            raise ValueError("Saved arrays changed.")
    rows = [row for saved in summaries for row in saved["rows"]]
    scalar_rows = [{k: v for k, v in r.items() if k != "optimizer_attempts"} for r in rows]
    table = aggregate(rows)
    write_rows_csv(project/"dataset_metrics.csv", scalar_rows, tuple(scalar_rows[0]))
    write_rows_csv(project/"sweep_summary.csv", table, tuple(table[0]))
    from experiments.bootstrap_band_sweep_reporting import plot_sweep
    plots = plot_sweep(table, manifest.payload, project/"plots")
    write_json_atomic(project/"summary.json", {
        "manifest": manifest.payload, "metrics": table, "datasets": [file_record(path) for path in records],
        "plots": [file_record(path) for path in plots],
        "plotting_source": file_record(Path(__file__).with_name("bootstrap_band_sweep_reporting.py")),
        "coverage_label": "Exact polynomial containment of represented bands on R; empirical approximate bootstrap sampling coverage.",
    })
    (project/"EXPERIMENT.md").write_text(experiment_text(manifest), encoding="utf-8")
    print(f"Completed controlled sweep: {project}", flush=True)


def experiment_text(manifest):
    """Document domain, pairing, optimization provenance, and metric semantics."""
    return f"""# Controlled all-real bootstrap OLS sweeps

The training actions are iid N(0,1), independent of iid Gaussian observation
errors. Truth, fitted quadratic, envelope, calibration, containment, and LCB
optimization are all defined on the entire real line. There is no clipping,
interpolation, action bound, or evaluation grid. Values at arbitrary actions
are direct evaluations of the quadratic basis and its fitted-mean standard
error. Errors at different actions are dependent through the same three OLS
coefficient errors, with covariance sigma^2 p(a)' V p(a').

Three independent named random streams generate training actions, standardized
observation errors, and Gaussian bootstrap arrays per dataset index. Training
observations use prefixes for N, bootstrap arrays use prefixes for B, and sigma
settings reuse all standardized draws. Bootstrap columns also pair across N.
Exact RNG seeds and sufficient arrays are saved for replay; larger bootstrap
counts change calibration, not the observed OLS coefficients. No optimizer
randomness is used: all initialization points are explicitly in the manifest.

The bootstrap refit identity beta_b*=beta_hat+sigma_hat R^(-1)Q'Z_b avoids
materializing response matrices. Gaussian draws and P alone calibrate the
standardized supremum; truth is used only to generate observations and evaluate
coverage/regret. Original sigma_hat stays fixed in bootstrap denominators.
Certified supremum brackets and exact rational polynomial containment tests
are inherited from MATH.md section 7.2. Their tolerance is in the manifest.
These certify represented floating-point polynomials, not exact-arithmetic OLS.
Coverage across datasets is empirical, not an exact finite-sample 95% theorem.

The repository Optimization/first-order L-BFGS-B entry point computes BOTH
the true reference action and the LCB action, without bounds. Starts in [0,1]
are initialization only. Multistart comparisons use only returned repository
actions. Exact rational sign-cell tests verify global objective gaps over R;
they never supply substitute optimizer actions. The reported numerical gap
tolerance is added to the regret inequality when testing it. Unbounded tails,
exactly degenerate tails, solver failures, and uncertified solutions are counted
separately. A certified result may retain a solver warning, explicitly recorded.

Figures (standard Matplotlib, vector PDF) each have three independent-axis
panels: sigma, N, B. Coverage is sum(C_r)/R with 95% Wilson intervals. Width is
W_r=r_delta,r(a_star), NOT sup_R r_delta (which is infinite). Regret is
f(a_star)-f(a_LCB). Width and regret show means with +/- one standard error
across independent datasets. Regret uses certified cases only, with explicit
denominators and failure counts in the CSV; a plot warns if any are excluded.
The 2W line is a bound benchmark that holds pointwise on the coverage event
for exact optimization, not an unconditional bound on mean regret.

This is a newly paired controlled experiment, not a relabeling of old independent
noise cases. Historical grid and all-real replay outputs remain unchanged.

Run: scripts/run_experiment_manifest.py {manifest.source_path} --launch local

```json
{json.dumps(manifest.payload, indent=2)}
```
"""


def build_launch_plan(manifest, *, runs_root=None, force=False):
    """Integrate the paired sweep with the existing local manifest launcher."""
    def run_all(context):
        for index in range(manifest.payload["datasets"]):
            run_dataset(manifest, index, context.runs_root, force)
        collect(manifest, context.runs_root)
    def run_task(index, context):
        if index != 0:
            raise IndexError(index)
        run_all(context)
    return LaunchPlan(name=manifest.name, task_count=1, requires_jax=False,
                      run_task=run_task, run_all=run_all,
                      collect=lambda context: collect(manifest, context.runs_root),
                      runs_root=runs_root, default_launch="local", default_array=False)
