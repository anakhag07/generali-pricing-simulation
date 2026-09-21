"""Paired all-real quadratic OLS sweep with a Gaussian-support envelope (§7.4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.stats import norm

from experiments.bootstrap_band import features
from experiments.bootstrap_band_sweep import optimize_lcb
from experiments.launch import LaunchPlan
from experiments.policy_lcb.common import PolicyLCBLaunchSpec, read_json, wilson_interval, write_json_atomic
from experiments.provenance import file_record
from experiments.seeds import derive_seed
from experiments.sweep_reporting import write_rows_csv
from objective.base import Objective
from optimization.solvers import run_first_order_minimize

MANIFEST_KIND = "coverage_kernel_n_sweep"


class GaussianSupportEnvelope(Objective):
    """Negative Gaussian-support lower envelope on all R."""

    def __init__(self, beta, actions, bandwidth, scale):
        self.beta = np.asarray(beta, dtype=float)
        self.actions = np.asarray(actions, dtype=float)
        self.bandwidth = float(bandwidth)
        self.scale = float(scale)
        if self.beta.shape != (3,) or self.actions.ndim != 1 or not self.actions.size:
            raise ValueError("Require three coefficients and nonempty training actions.")
        if not np.isfinite(self.beta).all() or not np.isfinite(self.actions).all():
            raise ValueError("Coefficients and actions must be finite.")
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0 or not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Bandwidth and envelope scale must be positive and finite.")

    def log_support(self, action):
        z = -0.5*((self.actions-float(action))/self.bandwidth)**2
        return float(logsumexp(z))

    def support(self, action):
        return float(np.exp(self.log_support(action)))

    def radius(self, action):
        return float(self.scale*np.exp(-0.5*self.log_support(action)))

    def lower(self, action):
        a = float(action)
        return float(self.beta[0]+a*self.beta[1]+a*a*self.beta[2]-self.radius(a))

    def value(self, theta, x_batch):
        return -self.lower(theta[0])

    def grad(self, theta, x_batch):
        a = float(theta[0])
        z = -0.5*((self.actions-a)/self.bandwidth)**2
        log_c = float(logsumexp(z))
        weights = np.exp(z-log_c)
        dlog_c = float(weights @ (self.actions-a))/self.bandwidth**2
        radius = self.scale*np.exp(-0.5*log_c)
        # d[-f_hat+radius]/da = -f_hat' - radius*d(log C)/2.
        return np.array([-self.beta[1]-2*self.beta[2]*a-0.5*radius*dlog_c])


def _quadratic_extrema(beta, low, high):
    """Return exact-in-form min/max of a quadratic on a closed interval."""
    b0, b1, b2 = map(float, beta)
    candidates = [low, high]
    if b2:
        vertex = -b1/(2*b2)
        if low < vertex < high:
            candidates.append(vertex)
    values = [b0+b1*a+b2*a*a for a in candidates]
    return min(values), max(values)


def _support_upper(actions, bandwidth, low, high):
    """Bound the Gaussian sum above throughout an interval."""
    distances = np.maximum(np.maximum(low-actions, actions-high), 0.)
    return float(np.sum(np.exp(-0.5*(distances/bandwidth)**2)))


def _tail_radius(actions, bandwidth, error, threshold):
    """Find a finite boundary with an analytical bound for both error tails."""
    maximum = float(np.max(np.abs(actions)))
    radius = max(1., 2*maximum, np.sqrt(8)*bandwidth)
    magnitude = float(np.sum(np.abs(error)))
    if magnitude == 0:
        return radius, 0.
    for _ in range(64):
        log_upper = (2*np.log(magnitude)+np.log(len(actions))+
                     2*np.log1p(radius*radius)-0.5*((radius-maximum)/bandwidth)**2)
        if log_upper < np.log(threshold)-1e-10:
            return radius, float(np.exp(log_upper))
        radius *= 2
    raise RuntimeError("Could not bound both Gaussian coverage tails.")


def check_two_sided_coverage(actions, bandwidth, error, scale, *, max_intervals=50000):
    """Check all-real containment by interval bounds, never an evaluation grid."""
    actions = np.asarray(actions, dtype=float)
    error = np.asarray(error, dtype=float)
    threshold = float(scale)**2
    radius, tail_upper = _tail_radius(actions, bandwidth, error, threshold)
    stack = [(-radius, radius)]
    checked = 0
    while stack and checked < max_intervals:
        low, high = stack.pop()
        checked += 1
        midpoint = (low+high)/2
        value = error[0]+midpoint*error[1]+midpoint*midpoint*error[2]
        statistic = value*value*float(np.exp(logsumexp(-0.5*((actions-midpoint)/bandwidth)**2)))
        if statistic > threshold*(1+1e-12):
            return {"state": "uncovered", "covered": False, "intervals_checked": checked,
                    "witness_action": midpoint, "witness_statistic": statistic,
                    "tail_upper": tail_upper}
        error_min, error_max = _quadratic_extrema(error, low, high)
        abs_error_upper = max(abs(error_min), abs(error_max))
        upper = abs_error_upper**2*_support_upper(actions, bandwidth, low, high)
        if upper*(1+1e-12) <= threshold:
            continue
        if midpoint == low or midpoint == high:
            return {"state": "indeterminate", "covered": None, "intervals_checked": checked,
                    "witness_action": None, "witness_statistic": None, "tail_upper": tail_upper}
        stack.extend(((low, midpoint), (midpoint, high)))
    if stack:
        return {"state": "indeterminate", "covered": None, "intervals_checked": checked,
                "witness_action": None, "witness_statistic": None, "tail_upper": tail_upper}
    return {"state": "covered", "covered": True, "intervals_checked": checked,
            "witness_action": None, "witness_statistic": None, "tail_upper": tail_upper}


def _lcb_tail_radius(objective, candidate_level):
    """Bound both tails above by the fitted quadratic when it is concave."""
    b0, b1, b2 = objective.beta
    if b2 >= 0:
        return None
    vertex = -b1/(2*b2)
    radius = max(1., float(np.max(np.abs(objective.actions)))+1, abs(vertex)+1)
    for _ in range(64):
        if max(b0+b1*a+b2*a*a for a in (-radius, radius)) < candidate_level:
            return radius
        radius *= 2
    return None


def check_lcb_global_gap(objective, candidate_value, tolerance, *, max_intervals=50000):
    """Conservatively bound the all-real LCB above; never propose an action."""
    level = candidate_value+tolerance
    radius = _lcb_tail_radius(objective, level)
    if radius is None:
        return {"state": "indeterminate", "intervals_checked": 0}
    stack = [(-radius, radius)]
    checked = 0
    while stack and checked < max_intervals:
        low, high = stack.pop()
        checked += 1
        midpoint = (low+high)/2
        if objective.lower(midpoint) > level+1e-12:
            return {"state": "uncertified", "intervals_checked": checked}
        _, fitted_upper = _quadratic_extrema(objective.beta, low, high)
        support_upper = _support_upper(objective.actions, objective.bandwidth, low, high)
        upper = fitted_upper-objective.scale/np.sqrt(support_upper) if support_upper else -np.inf
        if upper+1e-12 <= level:
            continue
        if midpoint == low or midpoint == high:
            return {"state": "indeterminate", "intervals_checked": checked}
        stack.extend(((low, midpoint), (midpoint, high)))
    return {"state": "certified" if not stack else "indeterminate", "intervals_checked": checked}


def optimize_kernel_lcb(beta, actions, bandwidth, scale, config):
    """Select only repository-optimizer actions and check their global gap."""
    objective = GaussianSupportEnvelope(beta, actions, bandwidth, scale)
    attempts = []
    for start in config["starts"]:
        theta, trace = run_first_order_minimize(
            np.array([start], dtype=float), np.zeros((1, 1)), objective,
            t_steps=config["t_steps"], n_grad_samples=1, sigma=0.,
            algorithm=config["step_rule"], grad_norm_tol=config["gradient_tolerance"],
            ftol=config["ftol"],
        )
        a = float(theta[0])
        value = objective.lower(a) if np.isfinite(a) else np.nan
        attempts.append({"start": start, "action": a, "value": value,
                         "success": bool(trace.optimizer_success),
                         "status": trace.optimizer_status, "message": trace.optimizer_message})
    finite = [attempt for attempt in attempts if np.isfinite([attempt["action"], attempt["value"]]).all()]
    if not finite:
        return {"optimization_state": "optimizer_failed", "action": None, "attempts": attempts,
                "global_gap_upper": None, "certificate_intervals": 0}
    best = max(finite, key=lambda attempt: attempt["value"])
    certificate = check_lcb_global_gap(objective, best["value"], config["global_gap_tolerance"],
                                       max_intervals=config["max_certificate_intervals"])
    certified = certificate["state"] == "certified"
    return {"optimization_state": certificate["state"], "action": best["action"],
            "value": best["value"], "attempts": attempts, "solver_success": best["success"],
            "global_gap_upper": config["global_gap_tolerance"] if certified else None,
            "certificate_intervals": certificate["intervals_checked"]}


@dataclass(frozen=True)
class CoverageKernelManifest:
    name: str
    payload: dict
    source_path: Path
    launch: PolicyLCBLaunchSpec


def load_manifest(path):
    """Validate the fixed-bandwidth, N-only Gaussian-support experiment."""
    path = Path(path).resolve()
    p = read_json(path)
    if p.get("kind") != MANIFEST_KIND or p.get("domain") != "real_line":
        raise ValueError("Require the all-real Gaussian-support manifest.")
    if not p["name"] or Path(p["name"]).name != p["name"] or p["name"] in {".", ".."}:
        raise ValueError("Require a safe project name.")
    if p["design"] != {"type": "iid_normal", "mean": 0., "std": 1.}:
        raise ValueError("Require standard normal training actions.")
    if p["truth"] != {"coefficients": [0., 5., -5.]}:
        raise ValueError("Require f(a)=5a-5a^2.")
    if p["sigma_epsilon"] != 1 or not 0 < p["alpha"] < 1 or p["datasets"] < 2:
        raise ValueError("Require sigma_epsilon=1, valid alpha, and at least two datasets.")
    n_values = p["N_values"]
    if not n_values or n_values != sorted(set(n_values)) or any(not isinstance(n, int) or n <= 3 for n in n_values):
        raise ValueError("N values must be distinct ascending integers greater than three.")
    if p["bandwidth"] != 0.1:
        raise ValueError("This first comparison fixes Gaussian bandwidth b=0.1.")
    if set(p["seeds"]) != {"master", "design", "observation"} or any(
        not isinstance(seed, int) or seed < 0 for seed in p["seeds"].values()
    ):
        raise ValueError("Require independent nonnegative design and observation seed roots.")
    opt = p["optimizer"]
    if opt["step_rule"] != "l-bfgs-b" or not opt["starts"] or any(not np.isfinite(a) for a in opt["starts"]):
        raise ValueError("Require finite repository L-BFGS-B starts.")
    if any(not np.isfinite(opt[key]) or opt[key] <= 0 for key in
           ("t_steps", "gradient_tolerance", "ftol", "global_gap_tolerance", "max_certificate_intervals")):
        raise ValueError("Invalid optimizer settings.")
    if p["max_coverage_intervals"] < 1 or p["launch"] != {"mode": "local", "array": "none"}:
        raise ValueError("Require positive coverage budget and local non-array launch.")
    return CoverageKernelManifest(p["name"], p, path, PolicyLCBLaunchSpec(**p["launch"]))


def dataset_streams(payload, index):
    """Reproduce the dense bootstrap sweep's nested design and noise streams."""
    seeds = {key: derive_seed(payload["seeds"][key], f"{payload['seeds']['master']}:dataset:{index}")
             for key in ("design", "observation")}
    nmax = max(payload["N_values"])
    x = np.random.default_rng(seeds["design"]).normal(size=nmax)
    epsilon = np.random.default_rng(seeds["observation"]).normal(size=nmax)
    return seeds, x, epsilon


def _source_contract():
    return [file_record(Path(__file__)),
            file_record(Path(__file__).with_name("coverage_kernel_sweep_reporting.py")),
            file_record(Path(__file__).parents[1]/"optimization"/"base.py"),
            file_record(Path(__file__).parents[1]/"optimization"/"solvers.py")]


def run_dataset(manifest, index, runs_root, force=False):
    """Compute one paired dataset block with auditable coverage and actions."""
    started = time.monotonic()
    p = manifest.payload
    destination = Path(runs_root)/manifest.name/"datasets"/f"dataset-{index:03d}"
    contract = {"manifest": p, "dataset": index, "sources": _source_contract()}
    summary_path = destination/"summary.json"
    if summary_path.exists() and not force:
        saved = read_json(summary_path)
        if saved["contract"] != contract or saved["arrays"] != file_record(destination/"draws.npz"):
            raise ValueError("Cached kernel sweep inputs changed; use a new name or --force.")
        return saved
    seeds, x, epsilon = dataset_streams(p, index)
    beta0 = np.array(p["truth"]["coefficients"], dtype=float)
    reference = optimize_lcb(beta0, np.eye(3), 1., 0., p["optimizer"])
    if reference["optimization_state"] != "certified":
        raise RuntimeError("Repository true-reference optimizer did not certify.")
    a_star = reference["action"]
    q = float(norm.ppf(1-p["alpha"]/2))
    arrays = {"training_actions": x, "standardized_observation_errors": epsilon}
    rows = []
    for n in p["N_values"]:
        design = features(x[:n])
        q_matrix, triangular = np.linalg.qr(design, mode="reduced")
        y = design @ beta0+p["sigma_epsilon"]*epsilon[:n]
        beta = solve_triangular(triangular, q_matrix.T @ y)
        sigma_hat = float(np.linalg.norm(y-design @ beta)/np.sqrt(n-3))
        if sigma_hat <= 0:
            raise ValueError("Positive residual scale is required.")
        scale = q*sigma_hat
        envelope = GaussianSupportEnvelope(beta, x[:n], p["bandwidth"], scale)
        coverage = check_two_sided_coverage(
            x[:n], p["bandwidth"], beta-beta0, scale,
            max_intervals=p["max_coverage_intervals"],
        )
        outcome = optimize_kernel_lcb(beta, x[:n], p["bandwidth"], scale, p["optimizer"])
        action = outcome["action"]
        valid = outcome["optimization_state"] == "certified"
        regret = (float(features(np.array([a_star]))[0] @ beta0-
                        features(np.array([action]))[0] @ beta0) if valid else None)
        width = envelope.radius(a_star)
        prefix = f"N{n}"
        arrays.update({f"{prefix}_y": y, f"{prefix}_beta_hat": beta,
                       f"{prefix}_residual_scale": np.array(sigma_hat)})
        rows.append({"dataset": index, "N": n, "sigma_epsilon": p["sigma_epsilon"],
                     "bandwidth": p["bandwidth"], "alpha": p["alpha"], "quantile": q,
                     "sigma_hat": sigma_hat, "covered": coverage["covered"],
                     "coverage_state": coverage["state"],
                     "coverage_intervals": coverage["intervals_checked"],
                     "coverage_witness_action": coverage["witness_action"],
                     "coverage_witness_statistic": coverage["witness_statistic"],
                     "coverage_tail_upper": coverage["tail_upper"],
                     "width_at_true_optimum": width, "true_action": a_star,
                     "lcb_action": action, "lcb_value": outcome.get("value"),
                     "regret": regret, "regret_bound": 2*width,
                     "optimization_state": outcome["optimization_state"],
                     "global_gap_upper": outcome["global_gap_upper"],
                     "certificate_intervals": outcome["certificate_intervals"],
                     "solver_success": outcome.get("solver_success"),
                     "optimizer_attempts": outcome["attempts"], "array_prefix": prefix})
        if valid and coverage["covered"] is True and regret > 2*width+outcome["global_gap_upper"]+reference["global_gap_upper"]:
            raise RuntimeError("Covered certified action violated the regret inequality.")
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination/"draws.npz", **arrays)
    saved = {"contract": contract, "seeds": seeds, "reference": reference,
             "rows": rows, "arrays": file_record(destination/"draws.npz"),
             "elapsed_seconds": time.monotonic()-started}
    write_json_atomic(summary_path, saved)
    print(f"Kernel dataset {index+1}/{p['datasets']}: {saved['elapsed_seconds']:.1f}s", flush=True)
    return saved


def _mean_se(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None, None
    return float(np.mean(a)), float(np.std(a, ddof=1)/np.sqrt(len(a))) if len(a) > 1 else None


def aggregate(rows):
    """Summarize independent datasets with explicit unresolved denominators."""
    result = []
    for n in sorted({r["N"] for r in rows}):
        group = [r for r in rows if r["N"] == n]
        resolved = [r for r in group if r["covered"] is not None]
        covered = sum(r["covered"] is True for r in resolved)
        low, high = wilson_interval(covered, len(resolved)) if resolved else (None, None)
        row = {"N": n, "datasets": len(group), "coverage_resolved_count": len(resolved),
               "covered_count": covered, "coverage_indeterminate_count": len(group)-len(resolved),
               "coverage": covered/len(resolved) if resolved else None,
               "coverage_lower": low, "coverage_upper": high,
               "optimizer_certified_count": sum(r["optimization_state"] == "certified" for r in group),
               "optimizer_uncertified_count": sum(r["optimization_state"] == "uncertified" for r in group),
               "optimizer_indeterminate_count": sum(r["optimization_state"] == "indeterminate" for r in group),
               "optimizer_failed_count": sum(r["optimization_state"] == "optimizer_failed" for r in group),
               "solver_warning_count": sum(r["solver_success"] is False for r in group)}
        for metric in ("width_at_true_optimum", "regret", "regret_bound", "sigma_hat"):
            row[f"{metric}_mean"], row[f"{metric}_se"] = _mean_se(
                [r[metric] for r in group if r[metric] is not None]
            )
        result.append(row)
    return result


def collect(manifest, runs_root):
    """Validate all paired blocks and produce CSV, JSON, and vector PDFs."""
    project = Path(runs_root)/manifest.name
    records = [project/"datasets"/f"dataset-{i:03d}"/"summary.json" for i in range(manifest.payload["datasets"])]
    summaries = [read_json(path) for path in records]
    for i, saved in enumerate(summaries):
        if saved["contract"] != {"manifest": manifest.payload, "dataset": i, "sources": _source_contract()}:
            raise ValueError("Incomplete or stale kernel dataset collection.")
        if saved["arrays"] != file_record(records[i].parent/"draws.npz"):
            raise ValueError("Saved kernel arrays changed.")
    rows = [row for saved in summaries for row in saved["rows"]]
    table = aggregate(rows)
    scalar_rows = [{key: value for key, value in row.items() if key != "optimizer_attempts"} for row in rows]
    write_rows_csv(project/"dataset_metrics.csv", scalar_rows, tuple(scalar_rows[0]))
    write_rows_csv(project/"sweep_summary.csv", table, tuple(table[0]))
    from experiments.coverage_kernel_sweep_reporting import plot_sweep
    plots = plot_sweep(table, manifest.payload, project/"plots")
    write_json_atomic(project/"summary.json", {
        "manifest": manifest.payload, "metrics": table,
        "datasets": [file_record(path) for path in records],
        "plots": [file_record(path) for path in plots],
        "plotting_source": file_record(Path(__file__).with_name("coverage_kernel_sweep_reporting.py")),
        "coverage_label": "Numerical all-real interval-bound containment; pointwise Gaussian scale is not a simultaneous-coverage guarantee.",
    })
    (project/"EXPERIMENT.md").write_text(experiment_text(manifest), encoding="utf-8")
    print(f"Completed Gaussian-support N sweep: {project}", flush=True)


def experiment_text(manifest):
    """Record the data, domain, pairing, certificate, and metric contract."""
    return f"""# Gaussian-support envelope: paired N sweep

Training actions are iid N(0,1), and responses are the fixed quadratic truth
f(a)=5a-5a² plus independent Gaussian noise with SD 1. The same design and
standardized observation streams as the dense bootstrap N sweep are used:
within a dataset, every smaller N uses a prefix of the length-5000 streams.
Dataset indices are independent. There is no held-out calibration set and no
bootstrap resampling in this method.

The action and two-sided simultaneous-coverage domain is the whole real line.
The fitted quadratic and Gaussian-support sum are evaluated directly at every
real optimizer or certificate query; a plotted finite sample is display-only.
The same fitted coefficient error induces dependent errors across all actions.
The fixed bandwidth is b=0.1, with unnormalized support sum and no bandwidth
selection from data. The radius is q_0.975*sigma_hat/sqrt(C_N(a)). This pointwise
Gaussian quantile is not a proof of simultaneous 95% coverage for this support
heuristic; empirical coverage is the experiment's question.

The repository first-order optimizer minimizes the negative lower envelope
without bounds from the manifest starts. The true reference also comes from
the repository optimizer. The Gaussian-support objective is non-polynomial, so
conservative floating-point interval bounds check its all-real objective gap
and the two-sided containment event, with explicit indeterminate/uncertified
states. This is a numerical check, not the exact rational polynomial proof of
the bootstrap experiment. No verification interval directly supplies a
reported action. Both infinite tails are bounded analytically; a plotted
curve or action grid never selects the result.

Coverage is the fraction of resolved independent datasets whose full true
curve lies inside the two-sided envelope; Wilson intervals use only resolved
cases and the number unresolved is reported. Width is the half-width at the
true optimum. Regret is f(a_star)-f(a_LCB) on globally gap-certified optimizer
cases; its denominator and any solver warnings are reported explicitly. The
2W reference is a pointwise coverage-event benchmark, not an unconditional
mean-regret bound. Seeds, source hashes, original observations, OLS fits, and
optimizer attempts are saved for replay.

Run: scripts/run_experiment_manifest.py {manifest.source_path} --launch local

```json
{json.dumps(manifest.payload, indent=2)}
```
"""


def build_launch_plan(manifest, *, runs_root=None, force=False):
    """Integrate the Gaussian-support sweep with the shared manifest runner."""
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
