"""Paired q or bandwidth sweeps replaying the saved Gaussian-support N run (§7.4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
from scipy.stats import norm

from experiments.bootstrap_band import features
from experiments.coverage_kernel_sweep import (
    GaussianSupportEnvelope, _source_contract as original_source_contract,
    check_two_sided_coverage, optimize_kernel_lcb,
)
from experiments.launch import LaunchPlan
from experiments.policy_lcb.common import PolicyLCBLaunchSpec, read_json, wilson_interval, write_json_atomic
from experiments.provenance import file_record
from experiments.sweep_reporting import write_rows_csv

MANIFEST_KINDS = {"coverage_kernel_q_sweep", "coverage_kernel_b_sweep"}


def _axis(payload):
    return "q" if payload["kind"] == "coverage_kernel_q_sweep" else "b"


def _axis_values(payload):
    return payload["q_values"] if _axis(payload) == "q" else payload["b_values"]


@dataclass(frozen=True)
class KernelParameterSweepManifest:
    name: str
    payload: dict
    source_path: Path
    launch: PolicyLCBLaunchSpec


def load_manifest(path):
    """Validate a positive one-axis sweep with the exact baseline included."""
    path = Path(path).resolve()
    p = read_json(path)
    if p.get("kind") not in MANIFEST_KINDS:
        raise ValueError("Require a Gaussian-support q- or b-sweep manifest.")
    for key in ("name", "source_name"):
        value = p[key]
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError(f"Require a safe {key}.")
    if p["name"] == p["source_name"]:
        raise ValueError("The parameter sweep must have a distinct result tree.")
    if not isinstance(p["datasets"], int) or p["datasets"] < 2:
        raise ValueError("Require at least two saved datasets.")
    n_values = p["N_values"]
    if not n_values or n_values != sorted(set(n_values)) or any(
        not isinstance(n, int) or n <= 3 for n in n_values
    ):
        raise ValueError("Require distinct ascending training sizes greater than three.")
    values = _axis_values(p)
    if not values or values != sorted(set(values)) or any(
        not np.isfinite(value) or value <= 0 for value in values
    ):
        raise ValueError("Require distinct ascending positive finite axis values.")
    if p["report_N_values"] != sorted(set(p["report_N_values"])) or not set(p["report_N_values"]).issubset(n_values):
        raise ValueError("Report N values must be distinct and part of the sweep.")
    if not p["report_N_values"]:
        raise ValueError("Require at least one report N value.")
    opt = p["optimizer"]
    if opt["step_rule"] != "l-bfgs-b" or not opt["starts"] or any(not np.isfinite(a) for a in opt["starts"]):
        raise ValueError("Require finite repository L-BFGS-B starts.")
    if any(not np.isfinite(opt[key]) or opt[key] <= 0 for key in
           ("t_steps", "gradient_tolerance", "ftol", "global_gap_tolerance", "max_certificate_intervals")):
        raise ValueError("Invalid optimizer settings.")
    if p["max_coverage_intervals"] < 1 or p["launch"] != {"mode": "local", "array": "none"}:
        raise ValueError("Require positive coverage budget and local non-array launch.")
    return KernelParameterSweepManifest(p["name"], p, path, PolicyLCBLaunchSpec(**p["launch"]))


def _source_contract():
    here = Path(__file__)
    return [file_record(here), file_record(here.with_name("coverage_kernel_parameter_sweep_reporting.py")),
            *original_source_contract()]


def _source_dataset(manifest, index, runs_root):
    """Return checked saved inputs; do not generate or refit observations."""
    p = manifest.payload
    base = Path(runs_root)/p["source_name"]
    source_summary_path = base/"datasets"/f"dataset-{index:03d}"/"summary.json"
    arrays_path = source_summary_path.parent/"draws.npz"
    source_summary = read_json(source_summary_path)
    original = source_summary["contract"]["manifest"]
    if source_summary["contract"] != {"manifest": original, "dataset": index,
                                      "sources": original_source_contract()}:
        raise ValueError("Original N-sweep source code or dataset contract changed.")
    if source_summary["arrays"] != file_record(arrays_path):
        raise ValueError("Original N-sweep draws changed.")
    if original["kind"] != "coverage_kernel_n_sweep" or original["name"] != p["source_name"]:
        raise ValueError("Require the completed Gaussian-support N sweep as source.")
    if original["datasets"] != p["datasets"] or original["N_values"] != p["N_values"]:
        raise ValueError("Parameter sweep must replay every original dataset and N value.")
    if p["optimizer"] != original["optimizer"] or p["max_coverage_intervals"] != original["max_coverage_intervals"]:
        raise ValueError("Parameter sweep must retain original optimizer and verifier settings.")
    baseline_q = float(norm.ppf(1-original["alpha"]/2))
    if _axis(p) == "q" and baseline_q not in p["q_values"]:
        raise ValueError("q sweep must include the exact original Gaussian quantile.")
    if _axis(p) == "b" and original["bandwidth"] not in p["b_values"]:
        raise ValueError("b sweep must include the exact original bandwidth.")
    if source_summary["reference"]["optimization_state"] != "certified":
        raise ValueError("Original true-reference optimizer was not certified.")
    return source_summary, arrays_path, baseline_q, file_record(source_summary_path)


def _replay_row(index, n, q, bandwidth, sigma_hat, beta, x, beta0, reference, source_row, p, baseline_q):
    """Use the exact baseline optimizer output or run the repository optimizer."""
    axis = _axis(p)
    axis_value = q if axis == "q" else bandwidth
    if q == baseline_q and bandwidth == source_row["bandwidth"]:
        return {**source_row, "q": q, "axis": axis, "axis_value": axis_value,
                "baseline_replay": True}
    scale = q*sigma_hat
    envelope = GaussianSupportEnvelope(beta, x, bandwidth, scale)
    coverage = check_two_sided_coverage(x, bandwidth, beta-beta0, scale,
                                        max_intervals=p["max_coverage_intervals"])
    outcome = optimize_kernel_lcb(beta, x, bandwidth, scale, p["optimizer"])
    a_star = reference["action"]
    action = outcome["action"]
    valid = outcome["optimization_state"] == "certified"
    regret = (float(features(np.array([a_star]))[0] @ beta0-
                    features(np.array([action]))[0] @ beta0) if valid else None)
    width = envelope.radius(a_star)
    if valid and coverage["covered"] is True and regret > 2*width+outcome["global_gap_upper"]+reference["global_gap_upper"]:
        raise RuntimeError("Covered certified action violated the regret inequality.")
    return {"dataset": index, "N": n, "axis": axis, "axis_value": axis_value,
            "q": q, "sigma_epsilon": 1., "bandwidth": bandwidth,
            "alpha": float(2*norm.sf(q)), "quantile": q, "sigma_hat": sigma_hat,
            "covered": coverage["covered"], "coverage_state": coverage["state"],
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
            "optimizer_attempts": outcome["attempts"], "array_prefix": f"N{n}",
            "baseline_replay": False}


def run_dataset(manifest, index, runs_root, force=False):
    """Replay one saved dataset over N×axis, storing provenance and outcomes."""
    started = time.monotonic()
    p = manifest.payload
    source, arrays_path, baseline_q, source_record = _source_dataset(manifest, index, runs_root)
    destination = Path(runs_root)/manifest.name/"datasets"/f"dataset-{index:03d}"
    summary_path = destination/"summary.json"
    contract = {"manifest": p, "dataset": index, "sources": _source_contract(),
                "source_summary": source_record, "source_arrays": file_record(arrays_path)}
    if summary_path.exists() and not force:
        saved = read_json(summary_path)
        if saved["contract"] != contract:
            raise ValueError("Cached parameter-sweep inputs changed; use a new name or --force.")
        return saved
    source_rows = {row["N"]: row for row in source["rows"]}
    beta0 = np.array(source["contract"]["manifest"]["truth"]["coefficients"], dtype=float)
    rows = []
    with np.load(arrays_path, allow_pickle=False) as arrays:
        x = arrays["training_actions"]
        for n in p["N_values"]:
            beta = arrays[f"N{n}_beta_hat"]
            sigma_hat = float(arrays[f"N{n}_residual_scale"])
            if source_rows[n]["sigma_hat"] != sigma_hat:
                raise ValueError("Saved OLS residual scale differs from source result.")
            for value in _axis_values(p):
                q = value if _axis(p) == "q" else baseline_q
                bandwidth = value if _axis(p) == "b" else source_rows[n]["bandwidth"]
                rows.append(_replay_row(index, n, q, bandwidth, sigma_hat, beta, x[:n], beta0,
                                        source["reference"], source_rows[n], p, baseline_q))
    destination.mkdir(parents=True, exist_ok=True)
    saved = {"contract": contract, "rows": rows, "reference": source["reference"],
             "elapsed_seconds": time.monotonic()-started}
    write_json_atomic(summary_path, saved)
    print(f"{_axis(p)} sweep dataset {index+1}/{p['datasets']}: {saved['elapsed_seconds']:.1f}s", flush=True)
    return saved


def _mean_se(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None


def aggregate(rows):
    """Summarize N×axis cells with explicit certification denominators."""
    result = []
    for n, value in sorted({(row["N"], row["axis_value"]) for row in rows}):
        group = [row for row in rows if row["N"] == n and row["axis_value"] == value]
        resolved = [row for row in group if row["covered"] is not None]
        covered = sum(row["covered"] is True for row in resolved)
        low, high = wilson_interval(covered, len(resolved)) if resolved else (None, None)
        row = {"N": n, "axis": group[0]["axis"], "axis_value": value,
               "q": group[0]["q"], "bandwidth": group[0]["bandwidth"],
               "datasets": len(group), "coverage_resolved_count": len(resolved),
               "covered_count": covered, "coverage_indeterminate_count": len(group)-len(resolved),
               "coverage": covered/len(resolved) if resolved else None,
               "coverage_lower": low, "coverage_upper": high,
               "optimizer_certified_count": sum(r["optimization_state"] == "certified" for r in group),
               "optimizer_uncertified_count": sum(r["optimization_state"] == "uncertified" for r in group),
               "optimizer_indeterminate_count": sum(r["optimization_state"] == "indeterminate" for r in group),
               "optimizer_failed_count": sum(r["optimization_state"] == "optimizer_failed" for r in group),
               "solver_warning_count": sum(r["solver_success"] is False for r in group)}
        for metric in ("width_at_true_optimum", "regret", "regret_bound"):
            row[f"{metric}_mean"], row[f"{metric}_se"] = _mean_se(
                [r[metric] for r in group if r[metric] is not None]
            )
        result.append(row)
    return result


def collect(manifest, runs_root):
    """Validate paired blocks and write tabular and PDF reports."""
    project = Path(runs_root)/manifest.name
    records = [project/"datasets"/f"dataset-{i:03d}"/"summary.json" for i in range(manifest.payload["datasets"])]
    summaries = [read_json(path) for path in records]
    for i, saved in enumerate(summaries):
        _, arrays_path, _, source_record = _source_dataset(manifest, i, runs_root)
        if saved["contract"] != {"manifest": manifest.payload, "dataset": i, "sources": _source_contract(),
                                 "source_summary": source_record, "source_arrays": file_record(arrays_path)}:
            raise ValueError("Incomplete or stale parameter-sweep collection.")
    rows = [row for saved in summaries for row in saved["rows"]]
    table = aggregate(rows)
    scalar_rows = [{key: value for key, value in row.items() if key != "optimizer_attempts"} for row in rows]
    write_rows_csv(project/"dataset_metrics.csv", scalar_rows, tuple(scalar_rows[0]))
    write_rows_csv(project/"sweep_summary.csv", table, tuple(table[0]))
    from experiments.coverage_kernel_parameter_sweep_reporting import plot_sweep
    plots = plot_sweep(table, manifest.payload, project/"plots")
    write_json_atomic(project/"summary.json", {
        "manifest": manifest.payload, "metrics": table,
        "datasets": [file_record(path) for path in records],
        "plots": [file_record(path) for path in plots],
        "plotting_source": file_record(Path(__file__).with_name("coverage_kernel_parameter_sweep_reporting.py")),
        "source_project": file_record(Path(runs_root)/manifest.payload["source_name"]/"summary.json"),
        "coverage_label": "Empirical two-sided all-real coverage; no q or b setting is a simultaneous guarantee.",
    })
    (project/"EXPERIMENT.md").write_text(experiment_text(manifest), encoding="utf-8")
    print(f"Completed Gaussian-support {_axis(manifest.payload)} sweep: {project}", flush=True)


def experiment_text(manifest):
    axis = _axis(manifest.payload)
    change = ("Only q changes in lambda=q*sigma_hat; bandwidth remains b=0.1."
              if axis == "q" else
              "Only Gaussian bandwidth b changes; q remains Phi^(-1)(0.975).")
    caveat = ("Coverage is nested upward and width is linear in q for fixed data."
              if axis == "q" else
              "The unnormalized Gaussian sum increases pointwise with b, so width and coverage are nested downward. Bandwidth smoothing and overall support scale change together.")
    return f"""# Gaussian-support envelope: paired {axis} sweep

This experiment replays the exact saved training inputs, responses, OLS fits,
residual scales, and repository-optimizer true reference from
{manifest.payload['source_name']}. For every independent dataset and training
size, {change} Observation-noise SD remains 1 and the action/coverage domain
remains the whole real line. The original q=Phi^(-1)(0.975), b=0.1 row is copied
verbatim from the source output; it is an exact saved repository-optimizer
replay with source artifact hashes, not a new solution from a grid.

All other reported lower-envelope actions come from the repository first-order
optimizer, then receive the same conservative all-real gap check. Two-sided
coverage uses the same all-real interval and analytical-tail verifier.
{caveat} True regret need not be monotone. Indeterminate coverage and
uncertified optimization are explicitly counted, not silently imputed. Wilson
intervals are descriptive across the same 100 datasets. Selecting a parameter
here to hit a target and then reporting its coverage on these same datasets
would be optimistic; independent validation is required for that claim.

Run: scripts/run_experiment_manifest.py {manifest.source_path} --launch local

```json
{json.dumps(manifest.payload, indent=2)}
```
"""


def build_launch_plan(manifest, *, runs_root=None, force=False):
    """Use the shared manifest runner for serial local replay and collection."""
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
