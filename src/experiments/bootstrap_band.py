"""Observed-pairs bootstrap quadratic OLS bands; see MATH.md section 7.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

from experiments.launch import LaunchPlan
from experiments.policy_lcb.common import (
    PolicyLCBLaunchSpec, read_json, wilson_interval, write_json_atomic,
)
from experiments.provenance import file_record
from experiments.seeds import derive_seed
from experiments.sweep_reporting import write_rows_csv


MANIFEST_KIND = "bootstrap_ols_grid_band"
COVERAGE_LABEL = (
    "Empirical validation of approximate bootstrap coverage on the specified grid; "
    "not a finite-sample proof or a continuous-domain guarantee."
)


def features(a: np.ndarray) -> np.ndarray:
    """Return the quadratic feature matrix with an intercept."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError("Inputs must be a finite one-dimensional array.")
    return np.column_stack((np.ones_like(a), a, a**2))


def refit_pairs(design: np.ndarray, y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Refit original rows, preserving input/response pairs (MATH.md §7.1).

    Indices are supplied by the dedicated bootstrap stream. Singular resamples
    fail explicitly: silently dropping or redrawing them changes the bootstrap.
    """
    indices = np.asarray(indices)
    if (indices.ndim != 2 or indices.shape[1] != len(design)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(design))):
        raise ValueError("Require an integer (B, N) array of original row indices.")
    fits = np.empty((len(indices), design.shape[1]))
    for b, rows in enumerate(indices):
        fit, _, rank, _ = np.linalg.lstsq(design[rows], y[rows], rcond=None)
        if rank != design.shape[1]:
            raise ValueError(f"Pairs-bootstrap replicate {b} has a rank-deficient design.")
        fits[b] = fit
    return fits


def bootstrap_band(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray, *,
    draws: int, delta: float, bootstrap_seed: int,
) -> dict[str, Any]:
    """Fit and calibrate a grid band using only observations, never truth."""
    design, evaluation = features(x), features(grid)
    y = np.asarray(y, dtype=float)
    n = len(design)
    if n <= 3 or y.shape != (n,) or not np.all(np.isfinite(y)):
        raise ValueError("OLS requires n > 3 and one finite response per input.")
    if draws < 2 or not 0 < delta < 1 or not len(evaluation):
        raise ValueError("Require draws >= 2, 0 < delta < 1, and a nonempty grid.")
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError("Quadratic design is rank deficient.")
    q, triangular = np.linalg.qr(design, mode="reduced")
    beta = solve_triangular(triangular, q.T @ y)
    residual = y - design @ beta
    sigma_hat = float(np.sqrt(residual @ residual / (n - 3)))
    if sigma_hat <= np.finfo(float).eps * max(1.0, float(np.linalg.norm(y))):
        raise ValueError("Residual variance is numerically zero; standardized band is undefined.")
    # P'P = R'R; ||R^{-T}p(a)|| is the prediction SE divided by sigma_hat.
    transformed = solve_triangular(triangular.T, evaluation.T, lower=True)
    se = sigma_hat * np.linalg.norm(transformed, axis=0)
    fitted = evaluation @ beta
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, n, size=(draws, n))
    bootstrap_beta = refit_pairs(design, y, indices)
    # This maximum is the user-specified coverage statistic, not an action selector.
    maxima = np.max(np.abs((bootstrap_beta - beta) @ evaluation.T) / se, axis=1)
    critical = float(np.quantile(maxima, 1.0 - delta, method="higher"))
    radius = critical * se
    return {
        "x": np.asarray(x), "y": y, "grid": np.asarray(grid), "beta_hat": beta,
        "residuals": residual, "sigma_hat": sigma_hat, "sigma2_hat": sigma_hat**2,
        "design_condition_number": float(np.linalg.cond(design)),
        "bootstrap_beta": bootstrap_beta, "bootstrap_indices": indices,
        "bootstrap_maxima": maxima,
        "critical_value": critical, "fitted": fitted, "standard_error": se,
        "radius": radius, "lower": fitted - radius, "upper": fitted + radius,
    }


def evaluate_band(band: dict[str, Any], true_values: np.ndarray) -> dict[str, Any]:
    """Check the fitted band against synthetic truth at every grid point."""
    truth = np.asarray(true_values, dtype=float)
    if truth.shape != band["fitted"].shape or not np.all(np.isfinite(truth)):
        raise ValueError("Truth must provide one finite value per evaluation point.")
    standardized_error = np.abs(band["fitted"] - truth) / band["standard_error"]
    covered = standardized_error <= band["critical_value"]
    return {
        "truth": truth, "standardized_error": standardized_error,
        "observed_max_statistic": float(np.max(standardized_error)),
        "simultaneous_covered": bool(np.all(covered)),
        "fraction_grid_covered": float(np.mean(covered)),
        "mean_radius": float(np.mean(band["radius"])),
    }


def _integer(value: Any, minimum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return float(value)


@dataclass(frozen=True)
class BootstrapBandManifest:
    """Validated, replayable two-stage finite-grid experiment."""

    name: str
    payload: dict[str, Any]
    source_path: Path
    launch: PolicyLCBLaunchSpec

    def cases(self) -> list[tuple[str, int, float, int]]:
        first, repeat = self.payload["stage1"], self.payload["stage2"]
        return [("stage1", first["n"], float(first["noise_std"]), 1)] + [
            ("stage2", n, float(std), repeat["datasets"])
            for n in repeat["sample_sizes"] for std in repeat["noise_stds"]
        ]


def load_bootstrap_band_manifest(path: str | Path) -> BootstrapBandManifest:
    """Validate the explicit model, grid, bootstrap, sweep, and seed contract."""
    path = Path(path).resolve()
    payload = read_json(path)
    if payload.get("kind") != MANIFEST_KIND:
        raise ValueError("Incorrect bootstrap manifest kind.")
    name = payload["name"]
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("name must be a single directory name.")
    if payload["design"] != {"type": "iid_normal", "mean": 0.0, "std": 1.0}:
        raise ValueError("This construction requires independent N(0,1) inputs.")
    if payload["truth"] != {"coefficients": [0.0, 5.0, -5.0]}:
        raise ValueError("This construction requires f(a)=5a-5a^2.")
    grid = payload["grid"]
    if not 0 <= float(grid["lower"]) < float(grid["upper"]) <= 1:
        raise ValueError("Coverage grid must be contained in [0,1].")
    _integer(grid["count"], 2, "grid.count")
    bootstrap = payload["bootstrap"]
    if bootstrap.get("method") != "pairs":
        raise ValueError("Require bootstrap.method='pairs'; historical Gaussian results are separate.")
    _integer(bootstrap["draws"], 2, "bootstrap.draws")
    if not 0 < float(bootstrap["delta"]) < 1 or bootstrap["quantile_method"] != "higher":
        raise ValueError("Require 0 < delta < 1 and quantile_method='higher'.")
    first, repeat = payload["stage1"], payload["stage2"]
    _integer(first["n"], 4, "stage1.n")
    _positive(first["noise_std"], "stage1.noise_std")
    _integer(repeat["datasets"], 2, "stage2.datasets")
    for key in ("sample_sizes", "noise_stds"):
        if not repeat[key] or len(set(repeat[key])) != len(repeat[key]):
            raise ValueError(f"stage2.{key} must be nonempty and unique.")
    for n in repeat["sample_sizes"]:
        _integer(n, 4, "sample size")
    for std in repeat["noise_stds"]:
        _positive(std, "noise SD")
    _integer(payload["seeds"]["master"], 0, "master seed")
    launch = payload["launch"]
    if launch["mode"] not in {"local", "slurm", "auto"} or launch["array"] != "none":
        raise ValueError("Bootstrap manifest uses launch.array='none'.")
    return BootstrapBandManifest(name, payload, path, PolicyLCBLaunchSpec(**launch))


def _case_name(stage: str, n: int, std: float) -> str:
    return f"{stage}-n{n}-sd{std:g}"


def _contract(manifest: BootstrapBandManifest) -> dict[str, Any]:
    return {"manifest": manifest.payload, "implementation": file_record(__file__)}


def _simulate(manifest: BootstrapBandManifest, stage: str, n: int, std: float, index: int):
    payload = manifest.payload
    case = _case_name(stage, n, std)
    streams = {
        process: derive_seed(payload["seeds"]["master"], f"{case}:dataset-{index}:{process}")
        for process in ("design", "observation_noise", "bootstrap")
    }
    x = np.random.default_rng(streams["design"]).normal(size=n)
    noise = std * np.random.default_rng(streams["observation_noise"]).normal(size=n)
    beta0 = np.asarray(payload["truth"]["coefficients"])
    y = features(x) @ beta0 + noise
    grid_spec, bs = payload["grid"], payload["bootstrap"]
    grid = np.linspace(grid_spec["lower"], grid_spec["upper"], grid_spec["count"])
    result = bootstrap_band(x, y, grid, draws=bs["draws"], delta=bs["delta"],
                            bootstrap_seed=streams["bootstrap"])
    result.update(evaluate_band(result, features(grid) @ beta0))
    row = {"stage": stage, "n": n, "noise_std": std, "dataset": index,
           **{f"{key}_seed": value for key, value in streams.items()}}
    row.update({key: value for key, value in result.items() if np.isscalar(value)})
    row.update({f"beta_hat_{j}": float(value) for j, value in enumerate(result["beta_hat"])})
    return result, row


def run_case(manifest: BootstrapBandManifest, index: int, *, runs_root: Path, force: bool):
    """Generate one stage/condition and save all refits and grid evaluations."""
    stage, n, std, count = manifest.cases()[index]
    directory = runs_root / manifest.name / _case_name(stage, n, std)
    summary_path, arrays_path = directory / "summary.json", directory / "draws.npz"
    contract = _contract(manifest)
    if not force and summary_path.exists() and arrays_path.exists():
        saved = read_json(summary_path)
        if saved.get("contract") == contract and saved.get("arrays") == file_record(arrays_path):
            return {"skipped": True, "path": str(summary_path)}
        raise ValueError(f"Existing outputs differ from this contract; use --force: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    results, rows = [], []
    for dataset in range(count):
        result, row = _simulate(manifest, stage, n, std, dataset)
        results.append(result)
        rows.append(row)
    # Stack the exact observations, coefficients, bootstrap draws, and grids for replay.
    np.savez_compressed(arrays_path, **{
        key: np.stack([result[key] for result in results]) for key in results[0]
    })
    write_json_atomic(summary_path, {
        "contract": contract, "arrays": file_record(arrays_path),
        "coverage_label": COVERAGE_LABEL, "rows": rows,
    })
    successes = sum(row["simultaneous_covered"] for row in rows)
    print(f"{directory.name}: {successes}/{count} datasets covered on the grid", flush=True)
    return {"skipped": False, "path": str(summary_path)}


def collect_outputs(manifest: BootstrapBandManifest, *, runs_root: Path):
    """Read completed datasets and report coverage, coefficient spread, and PDFs."""
    from experiments.bootstrap_band_reporting import write_plots

    project = runs_root / manifest.name
    rows, summaries, artifacts = [], [], []
    for stage, n, std, count in manifest.cases():
        directory = project / _case_name(stage, n, std)
        saved = read_json(directory / "summary.json")
        if saved["contract"] != _contract(manifest) or saved["arrays"] != file_record(directory / "draws.npz"):
            raise ValueError(f"Stale or altered artifacts: {directory}")
        group = saved["rows"]
        if len(group) != count:
            raise ValueError(f"Incomplete dataset ensemble: {directory}")
        rows.extend(group)
        artifacts.append(file_record(directory / "summary.json"))
        if stage == "stage1":
            continue
        successes = sum(row["simultaneous_covered"] for row in group)
        low, high = wilson_interval(successes, count)
        summary = {"n": n, "noise_std": std, "datasets": count, "covered": successes,
                   "coverage_rate": successes / count, "wilson_95_low": low, "wilson_95_high": high}
        for key in ("beta_hat_0", "beta_hat_1", "beta_hat_2", "sigma2_hat", "mean_radius"):
            values = np.asarray([row[key] for row in group])
            summary.update({f"{key}_mean": float(np.mean(values)),
                            f"{key}_std": float(np.std(values, ddof=1))})
        summaries.append(summary)
    write_rows_csv(project / "dataset_metrics.csv", rows, tuple(rows[0]))
    write_rows_csv(project / "coverage_summary.csv", summaries, tuple(summaries[0]))
    plots = write_plots(project, manifest.payload, rows, summaries)
    first = rows[0]
    write_json_atomic(project / "summary.json", {
        "coverage_label": COVERAGE_LABEL, "stage1": first, "stage2": summaries,
        "manifest": manifest.payload, "provenance": artifacts,
        "plotting_source": file_record(Path(__file__).with_name("bootstrap_band_reporting.py")),
        "plots": [file_record(path) for path in plots],
    })
    (project / "EXPERIMENT.md").write_text(_experiment_text(manifest), encoding="utf-8")
    print(f"Saved bootstrap band results and {len(plots)} PDFs in {project}", flush=True)
    return {"project_dir": str(project)}


def _experiment_text(manifest: BootstrapBandManifest) -> str:
    import json
    return f"""# Quadratic OLS bootstrap simultaneous band

{COVERAGE_LABEL}

Stage 1 is one Boolean outcome; stage 2 estimates the repeated-dataset probability.
Training inputs are independent N(0,1), independent of Gaussian observation errors.
OLS estimates all coefficients with p(a)=(1,a,a^2), using QR and residual variance
RSS/(n-3). The response is y=5x-5x^2+noise_std*Z. Each bootstrap dataset resamples N original
(x_i,y_i) rows with replacement and refits OLS. No fresh response noise is added.
Prediction standard errors are sigma_hat*sqrt(p(a)'*(P'P)^(-1)*p(a)).
Each bootstrap statistic is max_grid |f_hat_b-f_hat|/s_hat, retaining the original
s_hat in its denominator. The critical value uses np.quantile(method='higher').
The radius is critical_value*s_hat. Truth never enters calibration.

The grid is inclusive and defined below. Fitted curves are analytic polynomials;
their shared random coefficients induce covariance
sigma^2*p(a)'*(P'P)^(-1)*p(b) conditional on the design. Off-grid evaluation uses
the polynomial and standard-error formulas, not interpolation. Lines display
those formulas sampled densely; the confidence statement covers only the grid.
The coverage maximum is a statistic over the entire prescribed finite set.
There is no action selection, optimizer query, or global objective reference.
No coefficient ellipsoid or finite-sample certification is used. The fixed
bootstrap denominator omits residual-scale uncertainty in the real statistic.

Independent named seeds control design, observation noise, and bootstrap.
Seed labels include stage, n, noise SD, and dataset index. No draws are shared
between conditions or stages; each bootstrap uses only that dataset's observed rows.
Stage 2 shows unconditional coverage over both random design and response noise
(and finite-bootstrap randomness). Wilson intervals describe the 100-trial
Monte Carlo uncertainty, not the uncertainty band for f.

draws.npz contains observations, resampled row indices, residuals, fitted and bootstrap coefficients,
bootstrap maxima, grid predictions, SEs, radii, and truth diagnostics. Per-case
JSON files record the contract and hashes. summary.json and CSVs summarize them.
Plots use Matplotlib defaults and vector PDF output only.

Reproduce from {manifest.source_path} via scripts/run_experiment_manifest.py.
Use --force to regenerate; otherwise matching saved cases are reused.

```json
{json.dumps(manifest.payload, indent=2)}
```
"""


def build_bootstrap_band_launch_plan(manifest: BootstrapBandManifest, *, runs_root: str | None, force: bool):
    """Route the two-stage experiment through the repository's shared launcher."""
    def run_all(context):
        for index in range(len(manifest.cases())):
            run_case(manifest, index, runs_root=context.runs_root, force=force)
        collect_outputs(manifest, runs_root=context.runs_root)

    return LaunchPlan(
        name=manifest.name, task_count=len(manifest.cases()), requires_jax=False,
        run_task=lambda index, context: run_case(manifest, index, runs_root=context.runs_root, force=force),
        run_all=run_all, collect=lambda context: collect_outputs(manifest, runs_root=context.runs_root),
        runs_root=runs_root, default_launch=manifest.launch.mode, default_array=False,
    )
