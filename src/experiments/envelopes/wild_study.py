"""Paired fixed-design wild-bootstrap coverage and width sweeps (MATH §7.9–7.11)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
from scipy.special import expit, log_ndtr, ndtr
from scipy.stats import t, truncnorm

from experiments.launch import LaunchPlan
from experiments.policy_lcb.common import read_json, write_json_atomic, wilson_interval
from experiments.policy_lcb.continuous_gp_core import FiniteFourierGPSpec, FourierGPDraw
from experiments.provenance import array_sha256, file_record
from experiments.seeds import derive_seed
from experiments.slurm import CPU_PROFILE
from experiments.sweep_reporting import write_rows_csv
from .wild import WildLinearDesign, residual_radii

MANIFEST_KIND = "wild_bootstrap_envelopes"


def _save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class WildEnvelopeManifest:
    """Explicit fixed environment, paired sweep axes, and launch settings."""

    payload: dict
    source_path: Path
    launch: SimpleNamespace

    @property
    def name(self):
        """Return the result directory slug."""
        return self.payload["name"]


def load_manifest(path):
    """Validate the fixed-design comparison before generating observations."""
    path = Path(path).resolve()
    p = read_json(path)
    if p.get("kind") != MANIFEST_KIND:
        raise ValueError(f"Require kind={MANIFEST_KIND}.")
    if p.setdefault("band_target", "residual") != "residual":
        raise ValueError("New wild-envelope runs require band_target=residual.")
    name = p["name"]
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name)):
        raise ValueError("Require a safe project name.")
    d = p["design"]["context_dim"]
    if isinstance(d, bool) or not isinstance(d, int) or d < 0:
        raise ValueError("Require a nonnegative context dimension.")
    distribution = p["design"]["distribution"]
    if distribution not in {"independent_standard_normal", "independent_truncated_standard_normal",
                             "independent_standardized_t3"}:
        raise ValueError("Require independent normal, truncated-normal, or standardized-t3 coordinates/actions.")
    if distribution == "independent_truncated_standard_normal":
        if "truncation_bound" not in p["design"]:
            raise ValueError("Require explicit truncation_bound; null means unbounded.")
        bound = p["design"]["truncation_bound"]
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (float, int))
                                  or not np.isfinite(bound) or bound <= 0):
            raise ValueError("Require a finite positive truncation_bound or null for unbounded.")
    degree = p["design"].get("action_degree", 1)
    if isinstance(degree, bool) or degree not in (1, 2) or (degree == 2 and d != 0):
        raise ValueError("Require contextual affine or action-only quadratic features.")
    tolerance = p.get("supremum_tolerance", 1e-8)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (float, int)) or not 0 < tolerance < 1e-3:
        raise ValueError("Require a positive small supremum_tolerance.")
    for model in ("truth", "predictor"):
        coefficients = np.asarray(p[model]["coefficients"], dtype=float)
        if coefficients.shape != (d+degree+1,) or not np.isfinite(coefficients).all():
            raise ValueError("Model and design dimensions disagree.")
    if p["target"] != {"contexts": "calibration", "actions": "real_line"}:
        raise ValueError("Require every calibration context and all real actions.")
    for key, minimum in (("n_values", d+degree+2), ("B_values", 2)):
        vals = p[key]
        if (not vals or any(isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in vals)
                or vals != sorted(set(vals))):
            raise ValueError(f"Require sorted unique valid {key}.")
    if "action_override" in p["design"]:
        override = p["design"]["action_override"]
        if not isinstance(override, dict) or set(override) != {"index", "value"}:
            raise ValueError("Require action_override with exactly index and value.")
        index, value = override["index"], override["value"]
        if (isinstance(index, bool) or not isinstance(index, int)
                or not 0 <= index < min(p["n_values"])):
            raise ValueError("action_override index must occur in every sample-size prefix.")
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not np.isfinite(value)):
            raise ValueError("action_override value must be finite and numeric.")
    if (not p["deltas"] or len(set(p["deltas"])) != len(p["deltas"])
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not 0 < v < 1 for v in p["deltas"])):
        raise ValueError("Require unique deltas strictly in (0,1).")
    for key, axis in (("n", "n_values"), ("B", "B_values"), ("delta", "deltas")):
        if p["baseline"][key] not in p[axis]:
            raise ValueError("The baseline must occur in every sweep axis.")
    mode = p.get("sweep_mode", "grid")
    if mode not in {"grid", "paired"}:
        raise ValueError("Require sweep_mode=grid or paired.")
    if mode == "paired" and (len(p["n_values"]) != len(p["B_values"])
            or (p["baseline"]["n"], p["baseline"]["B"]) not in zip(p["n_values"], p["B_values"])):
        raise ValueError("Paired mode requires equally sized axes and a paired baseline.")
    for key, minimum in (("repetitions", 2), ("repetitions_per_task", 1)):
        value = p[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid {key}.")
    bootstrap = p["bootstrap"]
    if (set(bootstrap) != {"multipliers", "residuals", "quantile"}
            or bootstrap["multipliers"] != "rademacher" or bootstrap["residuals"] != "raw"
            or bootstrap["quantile"] not in {"higher", "order_statistic"}):
        raise ValueError("Require raw residuals, Rademacher multipliers, and the empirical order statistic.")
    # Accept the old manifest spelling, but record the exact current rank rule.
    bootstrap["quantile"] = "order_statistic"
    noise = p["noise"]
    if noise.get("innovation_distribution", "standard_normal") not in {"standard_normal", "centered_exponential"}:
        raise ValueError("Require standard_normal or centered_exponential innovations.")
    if noise["type"] == "constant":
        sigma = noise["sigma_max"]
        if (isinstance(sigma, bool) or not isinstance(sigma, (float, int))
                or not np.isfinite(sigma) or sigma <= 0):
            raise ValueError("Require a finite positive constant noise sigma_max.")
    elif noise["type"] == "bounded_fourier_gp_scale":
        bounds = np.array([noise["sigma_min"], noise["sigma_max"], noise["amplitude"]], dtype=float)
        if not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1] or bounds[2] <= 0:
            raise ValueError("Require positive bounded varying noise scales and GP amplitude.")
        if isinstance(noise["rank"], bool) or not isinstance(noise["rank"], int):
            raise ValueError("GP rank must be an integer.")
        FiniteFourierGPSpec(noise["rank"], noise["lengthscale"])
        projection = np.asarray(noise["context_projection"], dtype=float)
        if projection.shape != (d,) or not np.isfinite(projection).all():
            raise ValueError("GP context projection must match context dimension.")
    else:
        raise ValueError("Require constant noise or the bounded Fourier GP scale profile.")
    seeds = p["seeds"]
    if (set(seeds) != {"contexts", "actions", "observations", "method", "profile"}
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in seeds.values())
            or len(set(seeds.values())) != 5):
        raise ValueError("Require five distinct nonnegative integer seed roots.")
    action = p["diagnostics"]["actions"]
    if (not np.isfinite([action["min"], action["max"]]).all() or action["min"] >= action["max"]
            or isinstance(action["count"], bool) or not isinstance(action["count"], int) or action["count"] < 2):
        raise ValueError("Invalid diagnostic action grid.")
    launch = p["launch"]
    if launch["mode"] not in {"local", "auto", "slurm"} or launch["array"] not in {"none", "repetition"}:
        raise ValueError("Invalid launch policy.")
    parallel = launch.get("array_max_parallel")
    if parallel is not None and (isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < 1):
        raise ValueError("Invalid array concurrency.")
    return WildEnvelopeManifest(p, path, SimpleNamespace(**{**launch, "array_max_parallel": parallel}))


def _sources():
    root = Path(__file__).parents[1]
    return [file_record(root / name) for name in (
        "envelopes/wild.py", "envelopes/wild_study.py", "envelopes/linear.py",
        "envelopes/core.py", "policy_lcb/continuous_gp_core.py", "seeds/streams.py",
        "bootstrap_band_continuous.py", "offline_bootstrap_math.py", "policy_lcb/common.py",
    )]


def _contract(manifest):
    payload = {"manifest": manifest.payload, "sources": _sources(),
               "packages": {name: version(name) for name in ("numpy", "scipy", "sympy")}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest, payload


def noise_scale(X, actions, settings, gp_cos, gp_sin):
    """Evaluate the constant or saved analytic GP scale with a known upper bound."""
    if settings["type"] == "constant":
        return np.full(np.shape(actions), settings["sigma_max"], dtype=float)
    draw = FourierGPDraw(FiniteFourierGPSpec(settings["rank"], settings["lengthscale"]),
                         tuple(gp_cos), tuple(gp_sin))
    t = np.asarray(actions) + np.asarray(X) @ np.asarray(settings["context_projection"])
    return settings["sigma_min"] + (settings["sigma_max"]-settings["sigma_min"]) * expit(
        settings["amplitude"] * draw.evaluate(t))


def _standardized_t3(normal_draws):
    # Symmetry uses the small tail: no upper-CDF rounding, clipping or new draws.
    return np.sign(normal_draws) * t.isf(ndtr(-np.abs(normal_draws)), df=3) / np.sqrt(3.)


@lru_cache(maxsize=4)
def _environment(payload_json):
    p = json.loads(payload_json)
    nmax, d, seeds = max(p["n_values"]), p["design"]["context_dim"], p["seeds"]
    context_rng, action_rng = (np.random.default_rng(seeds[key]) for key in ("contexts", "actions"))
    if p["design"]["distribution"] == "independent_truncated_standard_normal":
        bound = p["design"]["truncation_bound"]
        bound = np.inf if bound is None else bound
        # Shared uniforms pair bounds; samples follow a truncated law, not clipping.
        X = truncnorm.ppf(context_rng.random((nmax, d)), -bound, bound)
        actions = truncnorm.ppf(action_rng.random(nmax), -bound, bound)
    else:
        X = context_rng.normal(size=(nmax, d))
        actions = action_rng.normal(size=nmax)
        if p["design"]["distribution"] == "independent_standardized_t3":
            X, actions = _standardized_t3(X), _standardized_t3(actions)
    if "action_override" in p["design"]:
        override = p["design"]["action_override"]
        # This prescribed design change consumes no draws; noise uses its new location.
        actions[override["index"]] = override["value"]
    coefficients = (np.empty((2, 0)) if p["noise"]["type"] == "constant" else
                    np.random.default_rng(seeds["profile"]).normal(size=(2, p["noise"]["rank"])))
    sigma = noise_scale(X, actions, p["noise"], *coefficients)
    spec = p["diagnostics"]["actions"]
    grid = np.linspace(spec["min"], spec["max"], spec["count"])
    snapshot = {"X": X, "actions": actions, "sigma": sigma,
                "gp_cos": coefficients[0], "gp_sin": coefficients[1], "actions_grid": grid}
    designs, mean_ell = [], []
    for n in p["n_values"]:
        design = WildLinearDesign(X[:n], actions[:n],
                                  action_degree=p["design"].get("action_degree", 1),
                                  supremum_tolerance=p.get("supremum_tolerance", 1e-8))
        contexts = X[:1] if d == 0 else X[:n]
        phi = design.features(np.repeat(contexts, len(grid), axis=0), np.tile(grid, len(contexts)))
        mean_ell.append(float(np.sqrt(np.einsum("ij,jk,ik->i", phi, design.covariance, phi)).mean()))
        designs.append(design)
    return snapshot, designs, np.array(mean_ell)


def _prepare(manifest, runs_root, force=False):
    project = Path(runs_root) / manifest.name
    project.mkdir(parents=True, exist_ok=True)
    digest, contract = _contract(manifest)
    metadata, arrays_path = project / "environment.json", project / "environment.npz"
    snapshot, designs, mean_ell = _environment(json.dumps(manifest.payload, sort_keys=True))
    if metadata.exists() and not force:
        saved = read_json(metadata)
        if saved["contract_sha256"] != digest or saved["arrays"] != file_record(arrays_path):
            raise ValueError("Saved environment or source contract changed; use a new result name or --force.")
        with np.load(arrays_path, allow_pickle=False) as archived:
            for key, value in {**snapshot, "mean_ell": mean_ell}.items():
                if not np.array_equal(value, archived[key]):
                    raise ValueError("Regenerated environment differs from archived inputs.")
    else:
        _save_arrays(arrays_path, **snapshot, mean_ell=mean_ell)
        write_json_atomic(metadata, {"contract_sha256": digest, "contract": contract,
                                    "arrays": file_record(arrays_path)})
    return project, digest, snapshot, designs, mean_ell


def _condition_mask(p):
    shape = (len(p["n_values"]), len(p["B_values"]))
    return np.eye(*shape, dtype=bool) if p.get("sweep_mode") == "paired" else np.ones(shape, dtype=bool)


def _run_repetition(manifest, index, project, digest, snapshot, designs, mean_ell, force=False):
    p = manifest.payload
    output = project / "repetitions" / f"rep-{index:05d}.npz"
    metadata = output.with_suffix(".json")
    if metadata.exists() and not force:
        record = read_json(metadata)
        if record["contract_sha256"] != digest or record["arrays"] != file_record(output):
            raise ValueError(f"Stale repetition {index}; use a new result name or --force.")
        return record
    seeds = {key: derive_seed(p["seeds"][key], f"repetition:{index}") for key in ("observations", "method")}
    nmax, bmax = len(snapshot["X"]), max(p["B_values"])
    innovations = np.random.default_rng(seeds["observations"]).normal(size=nmax)
    if p["noise"].get("innovation_distribution", "standard_normal") == "centered_exponential":
        innovations = -log_ndtr(-innovations)-1.
    multipliers = 2*np.random.default_rng(seeds["method"]).integers(0, 2, size=(bmax, nmax), dtype=np.int8)-1
    truth = np.asarray(p["truth"]["coefficients"])
    predictor = np.asarray(p["predictor"]["coefficients"])
    residual_truth = truth-predictor
    shape = (len(p["n_values"]), len(p["B_values"]), len(p["deltas"]), 2)
    active = _condition_mask(p)
    critical, auxiliary, transferred = np.full(shape, np.nan), np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    betas, statistics, statistic_lower, truth_statistics = [], [], [], []
    diagnostics = {name: np.full(shape, np.nan) for name in
                   ("mean_radius_lower", "mean_radius_upper", "mean_margin_lower", "mean_margin_upper",
                    "moved_radius_lower", "moved_radius_upper")}
    for ni, (n, design) in enumerate(zip(p["n_values"], designs)):
        y = design.design @ truth + snapshot["sigma"][:n]*innovations[:n]
        b_used = max(b for bi, b in enumerate(p["B_values"]) if active[ni, bi])
        beta, errors = design.bootstrap_errors(y, multipliers[:b_used, :n], predictor=design.design @ predictor)
        bounds = design.statistic_bounds(errors)
        boot_statistics = bounds[:, 1]
        true_statistic = float(design.statistics((beta-residual_truth)[None, :])[0])
        betas.append(beta)
        statistics.append(np.pad(boot_statistics, (0, bmax-b_used), constant_values=np.nan))
        statistic_lower.append(np.pad(bounds[:, 0], (0, bmax-b_used), constant_values=np.nan))
        truth_statistics.append(true_statistic)
        grid = snapshot["actions_grid"]
        contexts = snapshot["X"][:1] if snapshot["X"].shape[1] == 0 else snapshot["X"][:n]
        repeated, tiled = np.repeat(contexts, len(grid), axis=0), np.tile(grid, len(contexts))
        phi = design.features(repeated, tiled)
        geometry = design.residual_components(repeated, tiled, beta, 1.)
        fitted_grid, ell = geometry["residual_fit"], geometry["width"]
        truth_grid = phi @ residual_truth
        moved = p["design"].get("action_override", {}).get("index")
        if moved is not None:
            moved_geometry = design.residual_components(snapshot["X"][moved:moved+1],
                             snapshot["actions"][moved:moved+1], beta, 1.)
        decisions = {}
        for bi, b in enumerate(p["B_values"]):
            if not active[ni, bi]:
                continue
            for di, delta in enumerate(p["deltas"]):
                analytical = p["noise"]["sigma_max"]*np.sqrt(len(truth)/delta)
                order = math.ceil(b*(1-delta))-1
                bootstrap = float(np.partition(boot_statistics[:b], order)[order])
                for mi, value in enumerate((analytical, bootstrap)):
                    critical[ni, bi, di, mi] = value
                    covered = design.auxiliary_contains(beta, residual_truth, value)
                    auxiliary[ni, bi, di, mi] = covered
                    if value not in decisions:
                        decisions[value] = bool(covered or design.residual_contains(beta, residual_truth, value))
                    transferred[ni, bi, di, mi] = decisions[value]
                    lower, upper = residual_radii(fitted_grid, value*ell)
                    key = (ni, bi, di, mi)
                    diagnostics["mean_radius_lower"][key] = lower.mean()
                    diagnostics["mean_radius_upper"][key] = upper.mean()
                    diagnostics["mean_margin_lower"][key] = (truth_grid+lower).mean()
                    diagnostics["mean_margin_upper"][key] = (upper-truth_grid).mean()
                    if moved is not None:
                        lo, hi = residual_radii(moved_geometry["residual_fit"], value*moved_geometry["width"])
                        diagnostics["moved_radius_lower"][key] = lo[0]
                        diagnostics["moved_radius_upper"][key] = hi[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_arrays(output, beta=betas, bootstrap_statistics=statistics,
                        bootstrap_statistics_lower=statistic_lower,
                        truth_statistics=truth_statistics, critical=critical,
                        auxiliary_covered=auxiliary, transferred_covered=transferred,
                        innovations=innovations, active_conditions=active, **diagnostics)
    record = {"index": index, "contract_sha256": digest, "seeds": seeds,
              "multiplier_shape": [bmax, nmax], "multiplier_sha256": array_sha256(multipliers),
              "arrays": file_record(output)}
    write_json_atomic(metadata, record)
    return record


def run_repetition(manifest, index, runs_root, force=False):
    """Run or replay one independent response/multiplier repetition."""
    if not 0 <= index < manifest.payload["repetitions"]:
        raise IndexError(index)
    state = _prepare(manifest, runs_root, force=force)
    return _run_repetition(manifest, index, *state, force=force)


def _example(p, snapshot, designs, first, *, baseline=None, context_indices=None):
    baseline = p["baseline"] if baseline is None else baseline
    ni = p["n_values"].index(baseline["n"])
    bi = p["B_values"].index(baseline["B"])
    di = p["deltas"].index(baseline["delta"])
    grid = snapshot["actions_grid"]
    X = snapshot["X"][:baseline["n"]] if context_indices is None else snapshot["X"][context_indices]
    n = len(X)
    repeated, tiled = np.repeat(X, len(grid), axis=0), np.tile(grid, n)
    phi = designs[ni].features(repeated, tiled)
    ell = np.sqrt(np.einsum("ij,jk,ik->i", phi, designs[ni].covariance, phi)).reshape(n, -1)
    example = {"X": X, "actions_grid": grid}
    for name, coefficients in (("truth", p["truth"]["coefficients"]),
                               ("predictor", p["predictor"]["coefficients"]),
                               ("residual_fit", first["beta"][ni])):
        example[name] = (phi @ coefficients).reshape(n, -1)
    example["residual_truth"] = example["truth"]-example["predictor"]
    example["auxiliary"] = example["predictor"]+example["residual_fit"]
    example["analytical_width"] = first["critical"][ni, bi, di, 0]*ell
    example["bootstrap_width"] = first["critical"][ni, bi, di, 1]*ell
    for method in ("analytical", "bootstrap"):
        lower, upper = residual_radii(example["residual_fit"], example[f"{method}_width"])
        example[f"{method}_radius_lower"], example[f"{method}_radius_upper"] = lower, upper
        example[f"{method}_lower"] = example["predictor"]-lower
        example[f"{method}_upper"] = example["predictor"]+upper
    example["sigma_grid"] = noise_scale(repeated, tiled, p["noise"], snapshot["gp_cos"], snapshot["gp_sin"]).reshape(n, -1)
    return example


def collect(manifest, runs_root):
    """Validate complete paired repetitions and report joint coverage and width."""
    project, digest, snapshot, designs, mean_ell = _prepare(manifest, runs_root)
    p = manifest.payload
    active = _condition_mask(p)
    records, aggregate_arrays, first = [], [], None
    for index in range(p["repetitions"]):
        path = project / "repetitions" / f"rep-{index:05d}.json"
        record = read_json(path)
        arrays_path = path.with_suffix(".npz")
        if record["index"] != index or record["contract_sha256"] != digest or record["arrays"] != file_record(arrays_path):
            raise ValueError("Incomplete or stale sweep collection.")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            if not np.array_equal(arrays["active_conditions"], active):
                raise ValueError("Saved conditions disagree with the declared sweep mode.")
            if index == 0:
                first = {key: arrays[key] for key in arrays.files}
            aggregate_arrays.append({key: arrays[key] for key in arrays.files
                                     if key.startswith(("mean_", "moved_")) or key in
                                     ("critical", "auxiliary_covered", "transferred_covered")})
        records.append(file_record(path))
    critical = np.stack([a["critical"] for a in aggregate_arrays])
    auxiliary = np.stack([a["auxiliary_covered"] for a in aggregate_arrays])
    transferred = np.stack([a["transferred_covered"] for a in aggregate_arrays])
    metrics = []
    for ni, n in enumerate(p["n_values"]):
        for bi, b in enumerate(p["B_values"]):
            if not active[ni, bi]:
                continue
            for di, delta in enumerate(p["deltas"]):
                for mi, method in enumerate(("analytical", "wild_bootstrap")):
                    values = critical[:, ni, bi, di, mi]
                    row = {"n": n, "B": b, "delta": delta, "method": method,
                           "repetitions": p["repetitions"], "mean_width": float(values.mean()*mean_ell[ni]),
                           "width_ratio": float((values/critical[:, ni, bi, di, 0]).mean()),
                           "mean_critical": float(values.mean()),
                           "max_leverage": float(np.max(designs[ni].leverage))}
                    for name in aggregate_arrays[0]:
                        if name.startswith("mean_") or (name.startswith("moved_") and "action_override" in p["design"]):
                            row[name] = float(np.mean([a[name][ni, bi, di, mi] for a in aggregate_arrays]))
                    for prefix, decisions in (("auxiliary", auxiliary), ("transferred", transferred)):
                        count = int(decisions[:, ni, bi, di, mi].sum())
                        lo, hi = wilson_interval(count, p["repetitions"])
                        row.update({f"{prefix}_covered_count": count, f"{prefix}_coverage": count/p["repetitions"],
                                    f"{prefix}_lower": lo, f"{prefix}_upper": hi})
                    metrics.append(row)
    csv = project / "coverage_width_summary.csv"
    write_rows_csv(csv, metrics, list(metrics[0]))
    example = _example(p, snapshot, designs, first)
    example_path = project / "example.npz"
    _save_arrays(example_path, **example)
    summary = {"manifest": p, "band_target": "residual", "contract_sha256": digest, "metrics": metrics,
               "environment": file_record(project / "environment.json"), "repetitions": records,
               "csv": file_record(csv), "example": file_record(example_path),
               "coverage_event": "Residual g=f-fhat inside g_tilde +/- w, and f inside the asymmetric predictor envelope; all calibration contexts and all real actions jointly.",
               "bootstrap_claim": "Empirical study; no finite-sample bootstrap coverage guarantee.",
               "analytical_claim": "Finite-sample coverage at least 1-delta conditional on fixed design and noise profile.",
               "numerics": "Affine coefficient statistics; quadratic certified supremum brackets with upper endpoints. All-real containment uses stored floating-point coefficients, not an enclosure of fitting roundoff."}
    from .wild_reporting import write_reports
    if p.get("sweep_mode") == "paired":
        from .diagnostics import _slice_indices
        indices = _slice_indices(snapshot["X"][:min(p["n_values"])])
        slices = [_example(p, snapshot, designs, first,
                           baseline={"n": n, "B": b, "delta": p["baseline"]["delta"]},
                           context_indices=indices) for n, b in zip(p["n_values"], p["B_values"])]
        paired = {"X": slices[0]["X"], "actions_grid": snapshot["actions_grid"],
                  "context_indices": indices, "n_values": p["n_values"], "B_values": p["B_values"]}
        paired.update({key: np.stack([item[key] for item in slices]) for key in slices[0]
                       if key not in {"X", "actions_grid"}})
        paired_path = project / "paired_examples.npz"
        _save_arrays(paired_path, **paired)
        summary["paired_examples"] = file_record(paired_path)
        plots = write_reports(summary, snapshot, example, project / "plots", paired_example=paired)
    else:
        plots = write_reports(summary, snapshot, example, project / "plots")
    summary["plots"] = [file_record(path) for path in plots]
    summary["reporting_source"] = file_record(Path(__file__).with_name("wild_reporting.py"))
    guide = "wild_bootstrap_envelopes.md"
    documentation = Path(__file__).parents[3] / "docs" / guide
    experiment = project / "EXPERIMENT.md"
    experiment.write_text(documentation.read_text()+"\n\n## Saved experiment manifest\n\n```json\n"+
                          json.dumps(p, indent=2)+"\n```\n")
    summary["documentation"] = file_record(experiment)
    write_json_atomic(project / "summary.json", summary)
    print(f"Completed wild-bootstrap envelope comparison: {project}", flush=True)
    return summary


def build_launch_plan(manifest, *, runs_root=None, force=False):
    """Use the existing local/Slurm launcher with checkpointed response repeats."""
    p = manifest.payload
    count = math.ceil(p["repetitions"]/p["repetitions_per_task"])
    def run_task(index, context):
        if not 0 <= index < count:
            raise IndexError(index)
        state = _prepare(manifest, context.runs_root, force=force)
        indices = range(index*p["repetitions_per_task"], min((index+1)*p["repetitions_per_task"], p["repetitions"]))
        for repetition in indices:
            _run_repetition(manifest, repetition, *state, force=force)
            if (repetition+1) % 20 == 0 or repetition == indices.stop-1:
                print(f"Completed repetition {repetition+1}/{p['repetitions']}", flush=True)
        return {"repetitions": list(indices)}
    def run_all(context):
        for index in range(count):
            run_task(index, context)
        return collect(manifest, context.runs_root)
    return LaunchPlan(name=manifest.name, task_count=count, requires_jax=False,
                      run_task=run_task, run_all=run_all,
                      collect=lambda context: collect(manifest, context.runs_root),
                      runs_root=runs_root, default_launch=manifest.launch.mode,
                      default_array=manifest.launch.array == "repetition",
                      slurm_profile=replace(CPU_PROFILE, cpus_per_task=1, memory="4G"))
