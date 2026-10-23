"""Fixed offline pairs-bootstrap envelopes and certified regret (MATH §7.7)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json

import numpy as np
import sympy as sp
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.stats import beta as beta_distribution, binom

from experiments.envelopes.quadratic import features
from experiments.envelopes.quadratic import QuadraticErrorCertificate
from scratch.retired_experiments.studies.bootstrap_band_sweep import calibrate_design, optimize_lcb, settings
from experiments.launch import LaunchPlan
from scratch.retired_experiments.studies.policy_lcb.common import PolicyLCBLaunchSpec, read_json, write_json_atomic
from experiments.synthetic import FiniteFourierGPSpec, FourierGPDraw
from experiments.provenance import array_sha256, file_record
from experiments.seeds import derive_seed
from experiments.slurm import CPU_PROFILE
from experiments.sweep_reporting import write_rows_csv

MANIFEST_KIND = "offline_bootstrap"
INFERENCE = {
    "containment": "Deterministic analytical containment of represented bands over the entire real line.",
    "bootstrap_mc": "Conditional bootstrap quantile mass only; not coverage probability for unknown truth.",
    "sampling_coverage_lower_bound": None,
    "sampling_coverage_requirement": "A finite-N approximation theorem for the rank-conditioned pairs law remains unproved.",
    "regret": "The regret inequality applies on envelope containment with certified optimizer/reference gaps.",
    "outer_datasets": 0,
}


@dataclass(frozen=True)
class OfflineBootstrapManifest:
    """Validated manifest for one saved dataset and one unchanged predictor."""

    name: str
    payload: dict
    source_path: Path
    launch: PolicyLCBLaunchSpec


def _positive(value, name, *, integer=False, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if value <= minimum or (integer and int(value) != value):
        raise ValueError(f"Invalid {name}.")


def quantile_spec(draws, delta, bootstrap):
    """Return the order-statistic rank and its exact conditional Monte Carlo bound."""
    p = 1.0-delta
    eta = bootstrap["mc_failure_probability"]
    if bootstrap["quantile"] == "mc_upper":
        rank = int(binom.ppf(1-eta, draws, p))+1
        if rank > draws:
            raise ValueError("B is too small for the requested bootstrap Monte Carlo confidence.")
    else:
        rank = int(np.ceil((draws-1)*p))+1  # NumPy's higher empirical quantile.
    return {"rank": rank, "draws": draws,
            "mc_confidence": 1-eta,
            "bootstrap_mass_lower": float(beta_distribution.ppf(eta, rank, draws+1-rank)),
            "target_mass": p,
            "target_mass_confidence": float(binom.cdf(rank-1, draws, p))}


def load_manifest(path):
    """Validate offline sources, separate sweep axes, and named seed ownership."""
    path = Path(path).resolve()
    p = read_json(path)
    if p.get("kind") != MANIFEST_KIND or p.get("domain") != "real_line":
        raise ValueError("Require kind=offline_bootstrap and domain=real_line.")
    if not p["name"] or Path(p["name"]).name != p["name"] or p["name"] in {".", ".."}:
        raise ValueError("Require a safe output name.")
    if set(p["baseline"]) != {"N", "B", "sigma"} or not p["axes"] or not set(p["axes"]) <= set(p["baseline"]):
        raise ValueError("Define N, B, sigma baseline and separate axes.")
    for key, value in p["baseline"].items():
        _positive(value, key, integer=key != "sigma", minimum={"N": 3, "B": 1, "sigma": 0}[key])
    for key, values in p["axes"].items():
        if not values or values != sorted(set(values)) or p["baseline"][key] not in values:
            raise ValueError("Each axis must be sorted, unique and contain the baseline.")
        for value in values:
            _positive(value, key, integer=key != "sigma", minimum={"N": 3, "B": 1, "sigma": 0}[key])
    if not 0 < p["delta"] < 1 or not 0 < p["supremum_tolerance"] < 1e-3:
        raise ValueError("Invalid confidence level or supremum tolerance.")
    b = p["bootstrap"]
    if b["method"] != "pairs" or b["rank_deficient"] != "redraw" or b["quantile"] not in {"higher", "mc_upper"}:
        raise ValueError("Require observed pairs, explicit full-rank redraws and a supported quantile.")
    _positive(b["max_redraws"], "max_redraws", integer=True)
    if not 0 < b["mc_failure_probability"] < 1:
        raise ValueError("Invalid bootstrap MC failure probability.")
    for draws in set(p["axes"].get("B", [])) | {p["baseline"]["B"]}:
        quantile_spec(draws, p["delta"], b)
    if set(p["seeds"]) != {"bootstrap", "bootstrap_redraw", "gp_profile"}:
        raise ValueError("Require separate bootstrap, bootstrap_redraw and gp_profile seed roots.")
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in p["seeds"].values()) or len(set(p["seeds"].values())) != 3:
        raise ValueError("Seed roots must be distinct nonnegative integers.")
    for source in (p["offline_data"], p["predictor"]):
        if not source["path"] or len(source["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in source["sha256"]):
            raise ValueError("Saved inputs require paths and pinned SHA256 hashes.")
    _positive(p["offline_data"]["source_sigma"], "source_sigma")
    beta0 = np.asarray(p["truth"]["coefficients"], dtype=float)
    if beta0.shape != (3,) or not np.isfinite(beta0).all() or beta0[2] >= 0:
        raise ValueError("Require a finite strictly concave quadratic synthetic truth.")
    names = []
    for profile in p["profiles"]:
        name = profile["name"]
        if not name or not name.replace("_", "").isalnum() or name in names:
            raise ValueError("Profile names must be unique safe identifiers.")
        names.append(name)
        if profile["type"] not in {"constant", "log_gp"}:
            raise ValueError("Unsupported observation variance profile.")
        if profile["type"] == "log_gp":
            _positive(profile["eta"], "profile eta")
    if not names:
        raise ValueError("At least one profile is required.")
    _positive(p["gp"]["rank"], "GP rank", integer=True)
    _positive(p["gp"]["lengthscale"], "GP lengthscale")
    opt = p["optimizer"]
    if opt["step_rule"] != "l-bfgs-b" or not opt["starts"] or not np.isfinite(opt["starts"]).all():
        raise ValueError("Require repository L-BFGS-B with explicit finite starts.")
    for key in ("t_steps", "gradient_tolerance", "ftol", "global_gap_tolerance"):
        _positive(opt[key], key, integer=key == "t_steps")
    window = p["display_window"]
    if len(window) != 2 or not np.isfinite(window).all() or window[0] >= window[1]:
        raise ValueError("Invalid visualization window.")
    launch = p["launch"]
    if launch["mode"] not in {"local", "slurm", "auto"} or launch["array"] != "none":
        raise ValueError("One offline dataset uses a single non-array task.")
    return OfflineBootstrapManifest(p["name"], p, path, PolicyLCBLaunchSpec(**launch))


def _source_file(manifest, spec):
    path = Path(spec["path"])
    if not path.is_absolute():
        path = manifest.source_path.parent/path
    record = file_record(path)
    if record["sha256"] != spec["sha256"]:
        raise ValueError(f"Offline source checksum changed: {path}")
    return record


def _sources():
    root = Path(__file__).parents[3]/"src"
    files = [Path(__file__), root/"experiments/offline_bootstrap_math.py",
             Path(__file__).with_name("offline_bootstrap_reporting.py"),
             Path(__file__).with_name("bootstrap_band.py"),
             Path(__file__).with_name("bootstrap_band_sweep.py"),
             Path(__file__).with_name("bootstrap_band_continuous.py"),
             Path(__file__).parent/"policy_lcb"/"continuous_gp_core.py",
             root/"experiments/seeds/streams.py", root/"experiments/envelopes/quadratic.py", root/"experiments/synthetic.py"]
    files += [root/"optimization"/name for name in ("base.py", "solvers.py", "helpers.py", "gradients/methods.py")]
    return [file_record(p) for p in files]


def _verify(record):
    if file_record(record["path"]) != record:
        raise ValueError(f"Saved artifact changed: {record['path']}")


def _sigma_key(value):
    return format(float(value), ".17g")


def prepare_snapshot(manifest, project, force=False):
    """Freeze paired synthetic response variants before observation-only calibration."""
    p = manifest.payload
    project = Path(project)
    data_source = _source_file(manifest, p["offline_data"])
    predictor_source = _source_file(manifest, p["predictor"])
    contract = {"manifest": p, "offline_source": data_source,
                "predictor_source": predictor_source, "implementation": _sources()}
    metadata_path = project/"offline_snapshot.json"
    if metadata_path.exists() and not force:
        saved = read_json(metadata_path)
        if saved["contract"] != contract:
            raise ValueError("Offline snapshot contract changed; use a new experiment name.")
        _verify(saved["arrays"])
        with np.load(saved["arrays"]["path"], allow_pickle=False) as a:
            return saved, {key: a[key] for key in a.files}
    with np.load(data_source["path"], allow_pickle=False) as source:
        x = np.asarray(source[p["offline_data"]["x_key"]], dtype=float)
        y = np.asarray(source[p["offline_data"]["y_key"]], dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Offline x and y must be finite matching vectors.")
    if len(x) < max(p["axes"].get("N", [])+[p["baseline"]["N"]]):
        raise ValueError("N exceeds the saved offline reservoir.")
    with np.load(predictor_source["path"], allow_pickle=False) as source:
        candidate = np.asarray(source[p["predictor"]["beta_key"]], dtype=float)
    if candidate.shape != (3,) or not np.isfinite(candidate).all() or candidate[2] >= 0:
        raise ValueError("Saved predictor must be a strictly concave finite quadratic.")
    truth = np.asarray(p["truth"]["coefficients"], dtype=float)
    mean = features(x) @ truth
    z = (y-mean)/p["offline_data"]["source_sigma"]
    arrays = {"x": x, "source_y": y, "standardized_errors": z,
              "candidate_beta": candidate, "truth_beta": truth}
    spec = FiniteFourierGPSpec(**p["gp"])
    rng = np.random.default_rng(p["seeds"]["gp_profile"])
    coefficients = rng.normal(size=(2, spec.rank))
    draw = FourierGPDraw(spec, tuple(coefficients[0]), tuple(coefficients[1]))
    arrays["gp_coefficients"] = coefficients
    profiles = {}
    sigmas = sorted(set(p["axes"].get("sigma", [])) | {p["baseline"]["sigma"]})
    for profile in p["profiles"]:
        name = profile["name"]
        if profile["type"] == "constant":
            h, log_normalizer = np.ones(len(x)), 0.0
        else:
            log_h = profile["eta"]*draw.evaluate(x)
            log_normalizer = float((logsumexp(2*log_h)-np.log(len(x)))/2)
            h = np.exp(log_h-log_normalizer)
        if not np.isfinite(h).all() or np.any(h <= 0):
            raise ValueError("GP profile is not representably positive and finite.")
        arrays[f"{name}_h"] = h
        for sigma in sigmas:
            key = f"{name}_sigma{_sigma_key(sigma)}_y"
            # Preserve the exact original observation bytes for the matching baseline.
            arrays[key] = y.copy() if profile["type"] == "constant" and sigma == p["offline_data"]["source_sigma"] else mean+sigma*h*z
        profiles[name] = {**profile, "log_normalizer": log_normalizer,
                          "normalization": "RMS on the complete saved action reservoir",
                          "rms": float(np.sqrt(np.mean(h*h))),
                          "minimum_on_logged_rows": float(h.min()), "maximum_on_logged_rows": float(h.max())}
    project.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(project/"offline_snapshot.npz", **arrays)
    saved = {"contract": contract, "arrays": file_record(project/"offline_snapshot.npz"),
             "candidate_hash": array_sha256(candidate), "reservoir_size": len(x), "profiles": profiles,
             "inference": INFERENCE}
    write_json_atomic(metadata_path, saved)
    return saved, arrays


def sample_pairs_indices(x, draws, seeds, max_redraws):
    """Resample observed rows, redrawing singular designs with a separate seed stream."""
    design = features(np.asarray(x))
    n = len(design)
    if n <= 3 or np.linalg.matrix_rank(design) != 3:
        raise ValueError("Offline prefix must have full quadratic rank and N>3.")
    rng = np.random.default_rng(derive_seed(seeds["bootstrap"], f"offline-bootstrap:N={n}"))
    indices = rng.integers(0, n, size=(draws, n), dtype=np.int64)
    rejected = np.zeros(draws, dtype=np.int64)
    for b, rows in enumerate(indices):
        retry_rng = None
        while np.linalg.matrix_rank(design[rows]) != 3:
            if rejected[b] >= max_redraws:
                raise RuntimeError(f"Full-rank bootstrap retry limit reached for replicate {b}.")
            if retry_rng is None:
                retry_rng = np.random.default_rng(derive_seed(seeds["bootstrap_redraw"], f"offline-bootstrap:N={n}:b={b}"))
            rows = retry_rng.integers(0, n, size=n, dtype=np.int64)
            rejected[b] += 1
        indices[b] = rows
    return indices, rejected


def _calibration(manifest, project, snapshot, arrays, profile, n, draws, force=False):
    p = manifest.payload
    directory = project/"calibration"/profile/f"N{n}"
    meta_path = directory/"summary.json"
    contract = {"snapshot": snapshot["arrays"], "N": n, "profile": profile, "B": draws}
    if meta_path.exists() and not force:
        saved = read_json(meta_path)
        if saved["contract"] != contract:
            raise ValueError("Calibration checkpoint contract changed.")
        _verify(saved["arrays"])
        with np.load(saved["arrays"]["path"], allow_pickle=False) as a:
            return saved, {key: a[key] for key in a.files}
    x = arrays["x"][:n]
    y = arrays[f"{profile}_sigma{_sigma_key(p['baseline']['sigma'])}_y"][:n]
    indices, rejected = sample_pairs_indices(x, draws, p["seeds"], p["bootstrap"]["max_redraws"])
    design, q, triangular, covariance, d, bounds, timings = calibrate_design(x, y, indices, p["supremum_tolerance"])
    beta = solve_triangular(triangular, q.T @ y)
    residual = y-design @ beta
    sigma_hat = float(np.linalg.norm(residual)/np.sqrt(n-3))
    result = {"x": x, "y": y, "indices": indices, "rejected_draws": rejected,
              "beta": beta, "sigma_hat": np.asarray(sigma_hat), "q": q, "triangular": triangular,
              "covariance": covariance, "standardized_perturbations": d, "supremum_bounds": bounds,
              "standardized_residual": residual/sigma_hat}
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/"draws.npz", **result)
    saved = {"contract": contract, "arrays": file_record(directory/"draws.npz"), "timings": timings,
             "rejected_draws": int(rejected.sum())}
    write_json_atomic(meta_path, saved)
    print(f"Calibrated {profile}, N={n}, B={draws}; rejected {rejected.sum()} singular resamples", flush=True)
    return saved, result


def _directed_float(exact, upper):
    value = float(exact)
    if not np.isfinite(value):
        raise ValueError("Certificate endpoint exceeds finite floating-point range.")
    while (sp.Rational(value) < exact) if upper else (sp.Rational(value) > exact):
        value = float(np.nextafter(value, np.inf if upper else -np.inf))
    return value


def _quadratic_exact(coefficients, action):
    a = sp.Rational(float(action))
    return sum(sp.Rational(float(c))*a**i for i, c in enumerate(coefficients))


def _regret_interval(truth, reference, action):
    value = _quadratic_exact(truth, action)
    return [_directed_float(max(sp.Rational(0), sp.Rational(reference[key])-value), upper)
            for key, upper in (("candidate_value_lower", False), ("global_value_upper", True))]


def _radius_upper(candidate, auxiliary, radius_squared, action):
    q = radius_squared.eval(sp.Rational(float(action)))
    root = float(np.nextafter(np.sqrt(float(q)), np.inf)) if q else 0.0
    while sp.Rational(root)**2 < q:
        root = float(np.nextafter(root, np.inf))
    discrepancy = abs(_quadratic_exact(candidate, action)-_quadratic_exact(auxiliary, action))
    return _directed_float(discrepancy+sp.Rational(root), True)


def run_experiment(manifest, runs_root, force=False):
    """Run separate offline sweeps with analytical containment and optimizer regret."""
    from experiments.offline_bootstrap_math import optimize_transferred_lcb, transferred_contains
    from scratch.retired_experiments.studies.offline_bootstrap_reporting import write_plots

    p = manifest.payload
    project = Path(runs_root)/manifest.name
    snapshot, arrays = prepare_snapshot(manifest, project, force)
    final = project/"summary.json"
    if final.exists() and not force:
        previous = read_json(final)
        if previous["snapshot"] != snapshot["arrays"]:
            raise ValueError("Completed offline run references a different snapshot.")
        for record in previous["artifacts"]:
            _verify(record)
        return previous
    candidate, truth = arrays["candidate_beta"], arrays["truth_beta"]
    config = p["optimizer"]
    reference = optimize_lcb(truth, np.eye(3), 1.0, 0.0, config)
    plugin = optimize_lcb(candidate, np.eye(3), 1.0, 0.0, config)
    if any(o["optimization_state"] != "certified" for o in (reference, plugin)):
        raise RuntimeError("Reference or fixed-predictor repository optimization did not certify.")
    a_ref, a_plugin = reference["action"], plugin["action"]
    plugin_regret = _regret_interval(truth, reference, a_plugin)
    rows, outcomes, records = [], {}, [snapshot["arrays"], file_record(project/"offline_snapshot.json")]
    sweep = list(settings(p))
    for profile_spec in p["profiles"]:
        profile = profile_spec["name"]
        calibrations = {}
        for n in sorted({s["N"] for _, _, s in sweep}):
            draws = max(s["B"] for _, _, s in sweep if s["N"] == n)
            saved, cal = _calibration(manifest, project, snapshot, arrays, profile, n, draws, force)
            calibrations[n] = cal
            records.extend([saved["arrays"], file_record(project/"calibration"/profile/f"N{n}"/"summary.json")])
        evaluated = {}
        for axis, axis_value, setting in sweep:
            n, sigma, draws = setting["N"], setting["sigma"], setting["B"]
            identity = (n, sigma, draws)
            if identity not in evaluated:
                cal = calibrations[n]
                x = arrays["x"][:n]
                y = arrays[f"{profile}_sigma{_sigma_key(sigma)}_y"][:n]
                design = features(x)
                beta = solve_triangular(cal["triangular"], cal["q"].T @ y)
                residual = y-design @ beta
                sigma_hat = float(np.linalg.norm(residual)/np.sqrt(n-3))
                if sigma_hat <= 0 or not np.allclose(residual/sigma_hat, cal["standardized_residual"], rtol=1e-8, atol=1e-9):
                    raise ValueError("Saved response variants do not preserve standardized OLS errors.")
                quantile = quantile_spec(draws, p["delta"], p["bootstrap"])
                bounds = cal["supremum_bounds"][:draws]
                critical_low, critical = np.sort(bounds, axis=0)[quantile["rank"]-1]
                cert = QuadraticErrorCertificate(cal["covariance"], sigma_hat)
                radius_squared = cert.q.mul_ground(sp.Rational(float(critical))**2)
                # Truth appears only after observation-only bootstrap calibration.
                error = cert.difference(beta, truth)
                required = cert.supremum_interval(error, p["supremum_tolerance"])
                auxiliary_covered = cert.contains(error, float(critical))
                envelope_covered = transferred_contains(candidate, beta, truth, radius_squared)
                outcome = optimize_transferred_lcb(candidate, beta, cal["covariance"], sigma_hat, float(critical), config)
                valid = outcome["optimization_state"] == "certified"
                locations = np.array([a_ref, a_plugin] + ([outcome["action"]] if outcome["action"] is not None else []))
                location_features = features(locations)
                width = critical*sigma_hat*np.sqrt(np.einsum("ij,jk,ik->i", location_features, cal["covariance"], location_features))
                discrepancy = np.abs(location_features @ (candidate-beta))
                radius = width+discrepancy
                radius_upper = [_radius_upper(candidate, beta, radius_squared, a) for a in locations]
                regret = _regret_interval(truth, reference, outcome["action"]) if valid else [None, None]
                regret_bound = _directed_float(2*sp.Rational(radius_upper[0])+sp.Rational(outcome["global_gap_upper"])
                                               +sp.Rational(reference["global_gap_upper"]), True) if valid else None
                plugin_bound = _directed_float(sp.Rational(radius_upper[0])+sp.Rational(radius_upper[1])
                                               +sp.Rational(plugin["global_gap_upper"])
                                               +sp.Rational(reference["global_gap_upper"]), True)
                if auxiliary_covered and not envelope_covered:
                    raise RuntimeError("Auxiliary-to-predictor containment transfer failed.")
                if valid and envelope_covered and regret[1] > regret_bound+1e-9:
                    raise RuntimeError("Certified LCB regret inequality failed.")
                h = arrays[f"{profile}_h"][:n]
                oracle = cal["covariance"] @ (design.T @ (((sigma*h)**2)[:, None]*design)) @ cal["covariance"]
                bootstrap_beta = beta+sigma_hat*cal["standardized_perturbations"][:draws]
                empirical_cov = np.cov(bootstrap_beta, rowvar=False, ddof=1)
                p_ref = location_features[0]
                key = f"{profile}_N{n}_sigma{_sigma_key(sigma)}_B{draws}"
                band_path = project/"bands"/f"{key}.npz"
                band_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(band_path, x=x, y=y, auxiliary_beta=beta, candidate_beta=candidate,
                                    covariance=cal["covariance"], oracle_covariance=oracle,
                                    bootstrap_covariance=empirical_cov, bootstrap_beta=bootstrap_beta,
                                    indices=cal["indices"][:draws], rejected_draws=cal["rejected_draws"][:draws],
                                    auxiliary_containment_coefficients=np.array([str(c) for c in cert.threshold_polynomial(error**2, float(critical)).all_coeffs()]))
                records.append(file_record(band_path))
                outcomes[key] = {"lcb": outcome, "quantile": quantile}
                evaluated[identity] = {
                    "profile": profile, **setting, "auxiliary_covered": bool(auxiliary_covered),
                    "envelope_covered": bool(envelope_covered),
                    "auxiliary_required_critical_lower": required[0], "auxiliary_required_critical_upper": required[1],
                    "critical_value": float(critical), "critical_value_lower": float(critical_low),
                    "auxiliary_coverage_ratio": float(required[1]/critical) if critical > 0 else None,
                    "width_at_reference": float(width[0]), "discrepancy_at_reference": float(discrepancy[0]),
                    "radius_at_reference": float(radius[0]),
                    "radius_at_lcb": float(radius[2]) if len(radius)>2 else None,
                    "plugin_regret_lower": plugin_regret[0], "plugin_regret_upper": plugin_regret[1],
                    "lcb_regret_lower": regret[0], "lcb_regret_upper": regret[1],
                    "lcb_regret_bound": regret_bound, "plugin_regret_bound": plugin_bound,
                    "regret_bound_applicable": bool(valid and envelope_covered),
                    "optimization_state": outcome["optimization_state"], "lcb_action": outcome["action"],
                    "plugin_action": a_plugin, "reference_action": a_ref,
                    "global_gap_upper": outcome["global_gap_upper"], "reference_gap_upper": reference["global_gap_upper"],
                    "sigma_hat": sigma_hat, "bootstrap_order_rank": quantile["rank"],
                    "bootstrap_mass_lower": quantile["bootstrap_mass_lower"], "mc_confidence": quantile["mc_confidence"],
                    "oracle_variance_at_reference": float(p_ref @ oracle @ p_ref),
                    "bootstrap_variance_at_reference": float(p_ref @ empirical_cov @ p_ref),
                    "rank_deficient_draws": int(cal["rejected_draws"][:draws].sum()), "artifact_key": key,
                }
            rows.append({**evaluated[identity], "axis": axis, "axis_value": axis_value})
    write_rows_csv(project/"metrics.csv", rows, tuple(rows[0]))
    plots = write_plots(rows, p, project/"plots")
    (project/"EXPERIMENT.md").write_text(experiment_text(manifest, snapshot), encoding="utf-8")
    records += [file_record(project/"metrics.csv"), file_record(project/"EXPERIMENT.md")]+[file_record(path) for path in plots]
    summary = {"manifest": p, "snapshot": snapshot["arrays"], "inference": INFERENCE,
               "reference": reference, "plugin": plugin, "rows": rows, "outcomes": outcomes,
               "artifacts": records, "plots": [file_record(path) for path in plots]}
    write_json_atomic(final, summary)
    print(f"Completed fixed offline experiment: {project}", flush=True)
    return summary


def experiment_text(manifest, snapshot):
    """Record representation, randomness, inference limits, and replay provenance."""
    return f'''# Fixed offline pairs bootstrap and optimization regret

One checksummed offline reservoir; one separately saved fixed quadratic predictor.
Preparation replays observations and standardized errors, freezes paired sigma
and variance-profile response variants, and saves them before calibration.
Calibration and bootstrap refits read only observed rows. N uses nested prefixes;
B uses resample prefixes; sigma/profile settings share indices. No outer datasets.
The GP profile seed is separate from bootstrap and singular-resample retry seeds.
The GP's positive exponential profile uses analytic off-grid Fourier evaluation
and one empirical RMS normalizer on all {snapshot['reservoir_size']} saved actions.
It controls independent observation errors, not a shared additive error curve.

All quadratic fits, envelopes, optimizers and containment tests cover the whole
real line, including tails. Plot windows and connecting lines are display only.
The original OLS denominator is a weight, not a heteroskedastic standard error.
The exact known-noise covariance is diagnostic and never calibrates the band.
Rank-deficient pairs resamples are explicitly redrawn from the same rows with
bounded attempts; indices and rejection counts are saved. This conditions the
specified bootstrap distribution on full rank.

Every reported action comes from the repository optimizer. Rational sign cells
verify global objective gaps (tolerance {manifest.payload['optimizer']['global_gap_tolerance']})
and all-real containment of represented coefficients. Supremum tolerance is
{manifest.payload['supremum_tolerance']}. No grid or certificate root selects an
action. Nonsmooth LCB failures remain explicit. Regret is bracketed using the
certified reference value interval. The bound 2r(a_ref)+gap_LCB+gap_ref applies
only on final-envelope containment with certified solves.

These are deterministic coverage audits of this synthetic dataset, not estimated
sampling coverage percentages. Exact order-statistic Monte Carlo bounds concern
the conditional rank-conditioned bootstrap law only. An explicit finite-N
bootstrap approximation error bound remains an open proof requirement.
See MATH.md section 7.7 and docs/offline_bootstrap.md.

## Manifest

```json
{json.dumps(manifest.payload, indent=2)}
```
'''


def build_launch_plan(manifest, *, runs_root=None, force=False):
    """Use the shared launcher for a single checkpointed offline experiment."""
    def run_all(context):
        run_experiment(manifest, context.runs_root, force)
    def run_task(index, context):
        if index != 0:
            raise IndexError(index)
        result = run_experiment(manifest, context.runs_root, force)
        return {"snapshot": result["snapshot"]}
    return LaunchPlan(name=manifest.name, task_count=1, requires_jax=False,
                      run_task=run_task, run_all=run_all, runs_root=runs_root,
                      default_launch=manifest.launch.mode, default_array=False,
                      slurm_profile=replace(CPU_PROFILE, cpus_per_task=1, memory="4G"))
