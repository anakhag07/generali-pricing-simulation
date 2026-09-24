"""Extend saved bootstrap prefixes for Cartesian coverage (MATH.md §7.6)."""
from pathlib import Path
import time

import numpy as np

from experiments.bootstrap_band import refit_pairs, features
from experiments.bootstrap_band_continuous import QuadraticErrorCertificate
from experiments.bootstrap_band_sweep import dataset_streams
from experiments.bootstrap_joint_coverage import _contract, experiment_text
from experiments.policy_lcb.common import read_json, write_json_atomic, wilson_interval
from experiments.provenance import array_sha256, file_record
from experiments.sweep_reporting import write_rows_csv


def verified_path(record):
    """Verify saved content across the equivalent ORCD/login home mounts."""
    path = Path(record["path"])
    if not path.exists():
        path = Path(str(path).replace("/orcd/home/002/anakhag/", "/home/anakhag/"))
    current = file_record(path)
    if any(current[k] != record[k] for k in ("sha256", "bytes")):
        raise ValueError(f"Changed reused artifact: {path}")
    return path


def _reuse(manifest, index, runs_root, x, epsilon, uniforms):
    p = manifest.payload
    candidates, inputs, original_outcomes = {}, [], {}
    for project in p["reuse_projects"]:
        paths = [Path(runs_root)/project/"datasets"/f"dataset-{index:04d}"/"summary.json",
                 Path(runs_root)/project/"datasets"/f"dataset-{index:03d}"/"summary.json"]
        path = next((path for path in paths if path.exists()), None)
        if path is None:
            continue
        saved = read_json(path)
        previous = saved["contract"]["manifest"]
        if saved["contract"]["dataset"] != index or any(previous[k] != p[k] for k in (
                "design", "truth", "delta", "seeds", "domain", "bootstrap_method", "supremum_tolerance")):
            raise ValueError("Incompatible reuse model, seeds or calibration contract.")
        # The numerical calibration modules must be unchanged; runner wrappers may differ.
        for record in saved["contract"]["sources"]:
            source = verified_path(record)
            if source.name in {"bootstrap_band_sweep.py", "bootstrap_band.py", "bootstrap_band_continuous.py"}:
                if file_record(Path(__file__).with_name(source.name))["sha256"] != record["sha256"]:
                    raise ValueError("Reused bootstrap calibration implementation changed.")
        inputs.extend([file_record(path), saved["arrays"]])
        with np.load(verified_path(saved["arrays"])) as arrays:
            if not np.array_equal(arrays["training_actions"], x) or not np.array_equal(arrays["standardized_observation_errors"], epsilon):
                raise ValueError("Reused original observations differ.")
            if "uniform_shape" in saved and saved["uniform_shape"][1] != uniforms.shape[1]:
                raise ValueError("Bootstrap uniform row stride changed.")
            for n in p["grid"]["N"]:
                rows = [r for r in saved["rows"] if r["N"] == n and r["sigma"] == p["sigma"]]
                if not rows:
                    continue
                row = max(rows, key=lambda r: r["B"])
                prefix, count = row["array_prefix"], row["B"]
                indices = (n*uniforms[:count, :n]).astype(np.int32)
                if "bootstrap_indices_sha256" in row:
                    if array_sha256(indices) != row["bootstrap_indices_sha256"]:
                        raise ValueError("Reused bootstrap row selections differ.")
                    bounds = arrays[prefix+"_bootstrap_supremum_bounds"]
                    covariance = arrays[prefix+"_covariance"]
                else:
                    if not np.array_equal(indices, arrays[f"N{n}_bootstrap_indices"][:count]):
                        raise ValueError("Reused bootstrap row selections differ.")
                    bounds = arrays[f"N{n}_bootstrap_supremum_bounds"][:count]
                    covariance = arrays[f"N{n}_covariance"]
                candidate = {"count": count, "y": arrays[prefix+"_y"],
                             "beta": arrays[prefix+"_beta_hat"], "sigma_hat": row["sigma_hat"],
                             "covariance": covariance, "bootstrap_beta": arrays[prefix+"_bootstrap_beta"],
                             "bounds": bounds}
                for r in rows:
                    key = (n, r["B"])
                    if key in original_outcomes and original_outcomes[key] != r["covered"]:
                        raise ValueError("Reused coverage outcomes disagree.")
                    original_outcomes[key] = r["covered"]
                if n in candidates:
                    old = candidates[n]
                    overlap = min(count, old["count"])
                    if not np.array_equal(bounds[:overlap], old["bounds"][:overlap]):
                        raise ValueError("Reused bootstrap supremum prefixes disagree.")
                    if not np.array_equal(candidate["beta"], old["beta"]) or candidate["sigma_hat"] != old["sigma_hat"]:
                        raise ValueError("Reused original fits disagree.")
                if n not in candidates or count > candidates[n]["count"]:
                    candidates[n] = candidate
    if set(candidates) != set(p["grid"]["N"]):
        raise ValueError("Missing original fitted data for a requested N; refusing to refit it.")
    return candidates, inputs, original_outcomes


def run_dataset(manifest, index, runs_root, force=False):
    """Reuse each original fit and saved bootstrap prefix; compute missing suffixes."""
    p = manifest.payload
    if not 0 <= index < p["datasets"]:
        raise IndexError(index)
    destination = Path(runs_root)/manifest.name/"datasets"/f"dataset-{index:04d}"
    summary_path = destination/"summary.json"
    contract = _contract(manifest, index)
    if summary_path.exists() and not force:
        saved = read_json(summary_path)
        if saved["contract"] != contract:
            raise ValueError("Cached Cartesian contract changed.")
        for record in saved["reuse_inputs"] + saved["blocks"]:
            verified_path(record)
        for record in saved["blocks"]:
            verified_path(read_json(verified_path(record))["arrays"])
        return saved
    started = time.monotonic()
    nmax, bmax = max(p["grid"]["N"]), max(p["grid"]["B"])
    stream = {"seeds": p["seeds"], "axes": {}, "baseline": {"N": nmax, "B": bmax}}
    seeds, x, epsilon, uniforms = dataset_streams(stream, index)
    candidates, inputs, outcomes = _reuse(manifest, index, runs_root, x, epsilon, uniforms)
    destination.mkdir(parents=True, exist_ok=True)
    rows, blocks, reused, computed = [], [], 0, 0
    for n in p["grid"]["N"]:
        block_summary = destination/f"N{n}.json"
        block_contract = {"dataset_contract": contract, "N": n, "reuse_inputs": inputs}
        if block_summary.exists() and not force:
            block = read_json(block_summary)
            if block["contract"] != block_contract:
                raise ValueError("Cached Cartesian N-block contract changed.")
            verified_path(block["arrays"])
        else:
            original = candidates[n]
            k = min(original["count"], bmax)
            bounds = original["bounds"][:k].copy()
            beta_bootstrap = original["bootstrap_beta"][:k].copy()
            beta, covariance, scale = original["beta"], original["covariance"], original["sigma_hat"]
            if k < bmax:
                indices = (n*uniforms[k:bmax, :n]).astype(np.int32)
                additional = refit_pairs(features(x[:n]), original["y"], indices)
                d = (additional-beta)/scale
                cert = QuadraticErrorCertificate(covariance, 1.)
                suffix_bounds = np.array([cert.supremum_interval(cert.difference(draw, np.zeros(3)),
                                                                p["supremum_tolerance"]) for draw in d])
                bounds = np.concatenate([bounds, suffix_bounds])
                beta_bootstrap = np.concatenate([beta_bootstrap, additional])
            cert = QuadraticErrorCertificate(covariance, scale)
            error = cert.difference(beta, np.asarray(p["truth"]["coefficients"]))
            block_rows, polynomials = [], []
            for b in p["grid"]["B"]:
                c_low, c = np.quantile(bounds[:b], 1-p["delta"], axis=0, method="higher")
                covered = cert.contains(error, float(c))
                if (n,b) in outcomes and covered != outcomes[(n,b)]:
                    raise ValueError("Coverage changed at a previously completed setting.")
                polynomial = cert.threshold_polynomial(error**2, float(c))
                polynomials.append([str(polynomial.nth(i)) for i in range(5)])
                block_rows.append({"dataset": index, "N": n, "B": b, "covered": covered,
                                   "critical_value": float(c), "critical_value_lower": float(c_low)})
            arrays_path = destination/f"N{n}.npz"
            np.savez_compressed(arrays_path, y=original["y"], beta_hat=beta, covariance=covariance,
                                sigma_hat=scale, bootstrap_beta=beta_bootstrap,
                                bootstrap_supremum_bounds=bounds, B=p["grid"]["B"],
                                containment_exact_coefficients=np.array(polynomials))
            block = {"contract": block_contract, "rows": block_rows, "arrays": file_record(arrays_path),
                     "reused_bootstrap_draws": k, "new_bootstrap_draws": bmax-k}
            write_json_atomic(block_summary, block)
        rows.extend(block["rows"])
        blocks.append(file_record(block_summary))
        reused += block["reused_bootstrap_draws"]
        computed += block["new_bootstrap_draws"]
        print(f"Dataset {index+1}/{p['datasets']}, N={n}: "
              f"reuse {block['reused_bootstrap_draws']}, new {block['new_bootstrap_draws']}", flush=True)
    saved = {"contract": contract, "rows": rows, "reuse_inputs": inputs, "blocks": blocks,
             "seeds": seeds, "uniform_shape": [bmax,nmax], "reused_bootstrap_draws": reused,
             "new_bootstrap_draws": computed, "elapsed_seconds": time.monotonic()-started}
    write_json_atomic(summary_path, saved)
    return saved


def collect(manifest, runs_root):
    """Validate all Cartesian cells and generate coverage and uncertainty heatmaps."""
    p = manifest.payload
    project = Path(runs_root)/manifest.name
    rows, records, verified = [], [], set()
    for index in range(p["datasets"]):
        path = project/"datasets"/f"dataset-{index:04d}"/"summary.json"
        saved = read_json(path)
        if saved["contract"] != _contract(manifest, index):
            raise ValueError("Stale Cartesian dataset contract.")
        for record in saved["reuse_inputs"] + saved["blocks"]:
            key = (record["path"], record["sha256"])
            if key not in verified:
                verified_path(record)
                verified.add(key)
        block_rows = []
        for record in saved["blocks"]:
            block = read_json(verified_path(record))
            verified_path(block["arrays"])
            block_rows.extend(block["rows"])
        if block_rows != saved["rows"]:
            raise ValueError("Dataset rows differ from certified blocks.")
        rows.extend(saved["rows"])
        records.append(file_record(path))
    metrics = []
    for n in p["grid"]["N"]:
        for b in p["grid"]["B"]:
            group = [r for r in rows if r["N"] == n and r["B"] == b]
            if len(group) != p["datasets"] or len({r["dataset"] for r in group}) != p["datasets"]:
                raise ValueError("Incomplete Cartesian cell.")
            successes = sum(r["covered"] for r in group)
            lo, hi = wilson_interval(successes, len(group))
            metrics.append({"N": n, "B": b, "datasets": len(group), "covered_count": successes,
                            "coverage": successes/len(group), "coverage_lower": lo, "coverage_upper": hi})
    write_rows_csv(project/"dataset_coverage.csv", rows, tuple(rows[0]))
    write_rows_csv(project/"coverage_summary.csv", metrics, tuple(metrics[0]))
    from experiments.bootstrap_band_sweep_reporting import plot_cartesian_coverage
    plots = plot_cartesian_coverage(metrics, p, project/"plots")
    write_json_atomic(project/"summary.json", {"manifest": p, "datasets": records, "metrics": metrics,
        "plots": [file_record(path) for path in plots],
        "plotting_source": file_record(Path(__file__).with_name("bootstrap_band_sweep_reporting.py"))})
    (project/"EXPERIMENT.md").write_text(experiment_text(manifest)+
        "\nCartesian extension: reuse saved bootstrap prefixes; fit only missing suffixes. "
        "Heatmaps show discrete cells, with empirical coverage and pointwise Wilson uncertainty.\n")
    print(f"Completed Cartesian coverage sweep: {project}", flush=True)
