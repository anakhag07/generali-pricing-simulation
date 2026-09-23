"""Dedicated paired N/B sweep reporting empirical coverage only (MATH.md §7.5)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import math
import time

import numpy as np
from scipy.linalg import solve_triangular

from experiments.bootstrap_band import features
from experiments.bootstrap_band_continuous import QuadraticErrorCertificate
from experiments.bootstrap_band_sweep import SweepManifest, calibrate_design, dataset_streams
from experiments.launch import LaunchPlan
from experiments.policy_lcb.common import PolicyLCBLaunchSpec, read_json, write_json_atomic, wilson_interval
from experiments.provenance import array_sha256, file_record
from experiments.slurm import CPU_PROFILE
from experiments.sweep_reporting import write_rows_csv

MANIFEST_KIND = "bootstrap_ols_joint_coverage"


def load_manifest(path):
    """Validate an explicitly increasing N/B path and the unchanged band rule."""
    path = Path(path).resolve()
    p = read_json(path)
    if p.get("kind") != MANIFEST_KIND or p.get("domain") != "real_line" or p.get("bootstrap_method") != "pairs":
        raise ValueError("Require the joint all-real pairs-bootstrap manifest.")
    if not p["name"] or Path(p["name"]).name != p["name"] or p["name"] in {".", ".."}:
        raise ValueError("Require a safe project name.")
    if p["design"] != {"type": "iid_normal", "mean": 0., "std": 1.} or p["truth"] != {"coefficients": [0., 5., -5.]}:
        raise ValueError("Keep the existing normal-input quadratic data model.")
    if not np.isfinite(p["sigma"]) or p["sigma"] <= 0 or not 0 < p["delta"] < 1:
        raise ValueError("Require positive sigma and 0<delta<1.")
    if not isinstance(p["datasets"], int) or p["datasets"] < 2:
        raise ValueError("Require at least two independent datasets.")
    if not isinstance(p["datasets_per_task"], int) or p["datasets_per_task"] < 1:
        raise ValueError("datasets_per_task must be a positive integer.")
    if not p["settings"]:
        raise ValueError("Require explicit joint settings.")
    previous = {"N": 3, "B": 1}
    for setting in p["settings"]:
        if set(setting) != {"N", "B"} or any(not isinstance(setting[k], int) or setting[k] <= previous[k] for k in previous):
            raise ValueError("N and B must both strictly increase, starting at N>=4 and B>=2.")
        previous = setting
    if not 0 < p["supremum_tolerance"] < 1e-3:
        raise ValueError("Invalid supremum tolerance.")
    if set(p["seeds"]) != {"master", "design", "observation", "bootstrap"} or any(not isinstance(s, int) or s < 0 for s in p["seeds"].values()):
        raise ValueError("Require independent named nonnegative seed roots.")
    launch = p["launch"]
    if launch["mode"] not in {"local", "auto", "slurm"} or launch["array"] not in {"none", "seed"}:
        raise ValueError("Use a supported launch mode and none/seed array.")
    parallel = launch.get("array_max_parallel")
    if parallel is not None and (not isinstance(parallel, int) or parallel <= 0):
        raise ValueError("array_max_parallel must be a positive integer.")
    return SweepManifest(p["name"], p, path, PolicyLCBLaunchSpec(**launch))


def _sources():
    return [file_record(Path(__file__).parent/name) for name in (
        "bootstrap_joint_coverage.py", "bootstrap_band_sweep.py", "bootstrap_band.py",
        "bootstrap_band_continuous.py", "seeds/streams.py", "policy_lcb/common.py",
    )]


def _contract(manifest, index):
    return {"manifest": manifest.payload, "dataset": index, "sources": _sources()}


def run_dataset(manifest, index, runs_root, force=False):
    """Fit actual observed pairs, certify coverage, and checkpoint one dataset."""
    p = manifest.payload
    if not 0 <= index < p["datasets"]:
        raise IndexError(index)
    destination = Path(runs_root)/manifest.name/"datasets"/f"dataset-{index:04d}"
    summary_path, arrays_path = destination/"summary.json", destination/"draws.npz"
    contract = _contract(manifest, index)
    if summary_path.exists() and not force:
        saved = read_json(summary_path)
        if saved["contract"] != contract or saved["arrays"] != file_record(arrays_path):
            raise ValueError("Cached joint sweep contract/artifacts changed; use a new name or --force.")
        return saved
    started = time.monotonic()
    nmax = max(s["N"] for s in p["settings"])
    bmax = max(s["B"] for s in p["settings"])
    stream_spec = {"seeds": p["seeds"], "axes": {}, "baseline": {"N": nmax, "B": bmax}}
    seeds, x, epsilon, uniforms = dataset_streams(stream_spec, index)
    beta0 = np.asarray(p["truth"]["coefficients"])
    arrays = {"training_actions": x, "standardized_observation_errors": epsilon}
    rows = []
    for setting in p["settings"]:
        n, b = setting["N"], setting["B"]
        indices = (n*uniforms[:b, :n]).astype(np.int32)
        y = features(x[:n]) @ beta0+p["sigma"]*epsilon[:n]
        design, q, triangular, covariance, d, bounds, timings = calibrate_design(
            x[:n], y, indices, p["supremum_tolerance"])
        beta = solve_triangular(triangular, q.T @ y)
        sigma_hat = float(np.linalg.norm(y-design @ beta)/np.sqrt(n-3))
        critical_low, critical = np.quantile(bounds, 1-p["delta"], axis=0, method="higher")
        cert = QuadraticErrorCertificate(covariance, sigma_hat)
        error = cert.difference(beta, beta0)
        covered = cert.contains(error, float(critical))
        polynomial = cert.threshold_polynomial(error**2, float(critical))
        prefix = f"N{n}_B{b}"
        arrays.update({f"{prefix}_y": y, f"{prefix}_beta_hat": beta,
                       f"{prefix}_covariance": covariance,
                       f"{prefix}_bootstrap_beta": beta+sigma_hat*d,
                       f"{prefix}_bootstrap_supremum_bounds": bounds,
                       f"{prefix}_containment_exact_coefficients": np.array([str(polynomial.nth(i)) for i in range(5)])})
        rows.append({"dataset": index, **setting, "sigma": p["sigma"], "covered": covered,
                     "critical_value": float(critical), "critical_value_lower": float(critical_low),
                     "sigma_hat": sigma_hat, "array_prefix": prefix,
                     "bootstrap_indices_sha256": array_sha256(indices), **timings})
        print(f"Dataset {index+1}/{p['datasets']}, N={n}, B={b}: covered={covered}", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    saved = {"contract": contract, "seeds": seeds, "uniform_shape": [bmax, nmax],
             "rows": rows, "arrays": file_record(arrays_path),
             "elapsed_seconds": time.monotonic()-started}
    write_json_atomic(summary_path, saved)
    return saved


def collect(manifest, runs_root):
    """Validate all outer datasets and report only empirical coverage and CIs."""
    project = Path(runs_root)/manifest.name
    records, rows = [], []
    for index in range(manifest.payload["datasets"]):
        path = project/"datasets"/f"dataset-{index:04d}"/"summary.json"
        saved = read_json(path)
        if saved["contract"] != _contract(manifest, index) or saved["arrays"] != file_record(path.parent/"draws.npz"):
            raise ValueError("Incomplete or stale joint sweep collection.")
        rows.extend({k: r[k] for k in ("dataset", "N", "B", "covered")} for r in saved["rows"])
        records.append(file_record(path))
    metrics = []
    for setting in manifest.payload["settings"]:
        group = [r for r in rows if r["N"] == setting["N"] and r["B"] == setting["B"]]
        count = len(group)
        if count != manifest.payload["datasets"]:
            raise ValueError("Incomplete setting.")
        successes = sum(r["covered"] for r in group)
        lower, upper = wilson_interval(successes, count)
        metrics.append({**setting, "datasets": count, "covered_count": successes,
                        "coverage": successes/count, "coverage_lower": lower, "coverage_upper": upper})
    write_rows_csv(project/"dataset_coverage.csv", rows, tuple(rows[0]))
    write_rows_csv(project/"coverage_summary.csv", metrics, tuple(metrics[0]))
    from experiments.bootstrap_band_sweep_reporting import plot_joint_coverage
    plot = plot_joint_coverage(metrics, manifest.payload, project/"plots")
    write_json_atomic(project/"summary.json", {
        "manifest": manifest.payload, "datasets": records, "metrics": metrics,
        "plots": [file_record(plot)],
        "plotting_source": file_record(Path(__file__).with_name("bootstrap_band_sweep_reporting.py")),
    })
    (project/"EXPERIMENT.md").write_text(experiment_text(manifest), encoding="utf-8")
    print(f"Completed joint coverage sweep: {project}", flush=True)


def experiment_text(manifest):
    """Record the fixed inference rule, representation and reporting semantics."""
    return f'''# Joint N/B empirical coverage sweep

Original observations: iid x~N(0,1), y=5x-5x^2+sigma*epsilon, epsilon~N(0,1).
Each bootstrap resamples original (x,y) rows with replacement, without new noise.
The band rule is unchanged: the higher 1-delta quantile of bootstrap standardized
whole-line suprema times the original fitted-mean standard error. The shared
quadratic coefficients and covariance determine all off-grid function values.
Analytical supremum brackets at the manifest tolerance and exact rational
polynomial containment test the entire real line, including tails (MATH §7.2).
There is no interpolation, action optimization, Pareto ranking or new objective.

Independent named design, observation and bootstrap seeds are derived per
outer dataset. Original observations use nested prefixes across N. Row indices
are floor(N*U[:B,:N]), with U drawn once at max-B by max-N from the bootstrap
seed. Save that shape, seeds and index hashes to reproduce every row selection.
Rank-deficient resamples fail explicitly. Dataset order and Slurm grouping do
not change seeds. Original observations, coefficients and bootstrap supremum
brackets are saved, with source/artifact hashes for resume and collection.

Report empirical coverage (covered datasets / all datasets) and pointwise 95%
Wilson intervals. Dependence across settings comes from paired observations
and bootstrap streams. Lines only connect evaluated settings. Larger R reduces
coverage measurement error; bootstrap consistency does not imply monotonic
coverage as N or B grow. No estimates are smoothed or forced toward the target.

```json
{json.dumps(manifest.payload, indent=2)}
```
'''


def build_launch_plan(manifest, *, runs_root=None, force=False):
    """Group checkpointed independent datasets into existing ORCD array tasks."""
    p = manifest.payload
    task_count = math.ceil(p["datasets"]/p["datasets_per_task"])
    def run_task(index, context):
        if not 0 <= index < task_count:
            raise IndexError(index)
        indices = range(index*p["datasets_per_task"], min((index+1)*p["datasets_per_task"], p["datasets"]))
        for dataset in indices:
            run_dataset(manifest, dataset, context.runs_root, force)
        return {"datasets": list(indices)}
    def run_all(context):
        for index in range(task_count):
            run_task(index, context)
        collect(manifest, context.runs_root)
    return LaunchPlan(name=manifest.name, task_count=task_count, requires_jax=False,
                      run_task=run_task, run_all=run_all,
                      collect=lambda context: collect(manifest, context.runs_root),
                      runs_root=runs_root, default_launch=manifest.launch.mode,
                      default_array=manifest.launch.array == "seed",
                      slurm_profile=replace(CPU_PROFILE, cpus_per_task=1, memory="4G"))
