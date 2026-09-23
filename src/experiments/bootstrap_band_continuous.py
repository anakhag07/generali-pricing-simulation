"""Analytical all-real coverage replay from saved OLS/bootstrap fits (§7.2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.linalg import solve_triangular

from experiments.bootstrap_band import features
from experiments.launch import LaunchPlan
from experiments.policy_lcb.common import PolicyLCBLaunchSpec, read_json, wilson_interval, write_json_atomic
from experiments.provenance import file_record
from experiments.sweep_reporting import write_rows_csv

MANIFEST_KIND = "bootstrap_ols_continuous_replay"
VARIABLE = sp.Symbol("a")
COVERAGE_LABEL = (
    "Analytical containment over all real a for each saved dataset; empirical "
    "validation of approximate bootstrap coverage, not an exact 95% sampling theorem."
)


def nonnegative_on_real_line(poly: sp.Poly) -> bool:
    """Decide global polynomial nonnegativity using exact real-root counts."""
    if poly.is_zero:
        return True
    if poly.LC() < 0 or poly.degree() % 2:
        return False
    if poly.degree() == 0 or poly.count_roots(-sp.oo, sp.oo) == 0:
        return True
    # A real root of odd multiplicity changes sign; even roots only touch zero.
    return all(
        multiplicity % 2 == 0 or factor.count_roots(-sp.oo, sp.oo) == 0
        for factor, multiplicity in poly.sqf_list()[1]
    )


class QuadraticErrorCertificate:
    """Exact threshold tests for |p(a)'d| / sqrt(q(a)), with positive quartic q."""

    def __init__(self, covariance: np.ndarray, sigma_hat: float):
        v = [[sp.Rational(float(entry)) for entry in row] for row in covariance]
        matrix = sp.Matrix(v)
        if matrix != matrix.T or any(matrix[:i, :i].det() <= 0 for i in (1, 2, 3)):
            raise ValueError("Represented covariance must be symmetric positive definite.")
        scale = sp.Rational(float(sigma_hat))**2
        if scale <= 0:
            raise ValueError("Positive residual scale is required.")
        self.q_coefficients = [scale * c for c in (
            v[0][0], 2*v[0][1], v[1][1] + 2*v[0][2], 2*v[1][2], v[2][2],
        )]
        self.q = sp.Poly.from_list(self.q_coefficients[::-1], VARIABLE, domain=sp.QQ)
        self.q_float = np.array(self.q_coefficients, dtype=float)

    @staticmethod
    def difference(first: np.ndarray, second: np.ndarray) -> sp.Poly:
        coefficients = [sp.Rational(float(a)) - sp.Rational(float(b)) for a, b in zip(first, second)]
        if len(coefficients) != 3:
            raise ValueError("Three quadratic coefficients are required.")
        return sp.Poly.from_list(coefficients[::-1], VARIABLE, domain=sp.QQ)

    def threshold_polynomial(self, square: sp.Poly, threshold: float) -> sp.Poly:
        return self.q.mul_ground(sp.Rational(float(threshold))**2) - square

    def contains(self, difference: sp.Poly, threshold: float) -> bool:
        """Certify the entire error curve lies inside threshold*sqrt(q(a))."""
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Threshold must be finite and nonnegative.")
        return nonnegative_on_real_line(self.threshold_polynomial(difference**2, threshold))

    def supremum_interval(self, difference: sp.Poly, tolerance: float) -> tuple[float, float]:
        """Certify a tight interval for the all-real statistic, with no grid."""
        if difference.is_zero:
            return 0.0, 0.0
        d = np.array([float(difference.nth(i)) for i in range(3)])
        q = self.q_float
        from numpy.polynomial import polynomial as polynomial
        # Leading degree-five terms cancel identically for quadratic/quartic.
        stationary = polynomial.polysub(
            2 * polynomial.polymul(polynomial.polyder(d), q),
            polynomial.polymul(d, polynomial.polyder(q)),
        )[:5]
        roots = polynomial.polyroots(stationary)
        candidates = [0.0] + [float(r.real) for r in roots if abs(r.imag) <= 1e-7 * max(1, abs(r.real))]
        values = [abs(d[2]) / np.sqrt(q[4])]  # Same limit in both tails.
        for a in candidates:
            # Homogeneous evaluation avoids overflow in far-tail stationary points.
            if abs(a) > 1:
                numerator = polynomial.polyval(1/a, d[::-1])
                denominator = polynomial.polyval(1/a, q[::-1])
            else:
                numerator = polynomial.polyval(a, d)
                denominator = polynomial.polyval(a, q)
            if denominator > 0:
                values.append(abs(numerator) / np.sqrt(denominator))
        proposal = float(max(values))
        if not np.isfinite(proposal):
            raise ValueError("Nonfinite supremum proposal.")
        square = difference**2

        def below(t):
            return nonnegative_on_real_line(self.threshold_polynomial(square, t))

        padding = tolerance * max(1, proposal)
        low, high = max(0.0, proposal - padding), proposal + padding
        for _ in range(100):
            if below(high):
                break
            high = 2*high + padding
        else:
            raise RuntimeError("Could not certify an upper supremum bound.")
        for _ in range(100):
            if low == 0 or not below(low):
                break
            low /= 2
        else:
            raise RuntimeError("Could not certify a lower supremum bound.")
        for _ in range(100):
            if high-low <= 2*tolerance*max(1, high):
                return low, high
            middle = (low+high)/2
            if below(middle):
                high = middle
            else:
                low = middle
        raise RuntimeError("Supremum certificate did not reach its requested tolerance.")


def replay_dataset(saved: dict, beta0: np.ndarray, delta: float, tolerance: float) -> dict:
    """Recalibrate saved bootstrap curves on R, then evaluate known truth."""
    x, y = saved["x"], saved["y"]
    p = features(x)
    q, triangular = np.linalg.qr(p, mode="reduced")
    inverse = solve_triangular(triangular, np.eye(3))
    covariance = inverse @ inverse.T
    certificate = QuadraticErrorCertificate(covariance, float(saved["sigma_hat"]))
    beta = saved["beta_hat"]
    bounds = np.array([
        certificate.supremum_interval(certificate.difference(draw, beta), tolerance)
        for draw in saved["bootstrap_beta"]
    ])
    critical_low, critical = np.quantile(bounds, 1-delta, axis=0, method="higher")
    # The known beta0 is introduced only after bootstrap calibration is finished.
    observed = certificate.difference(beta, beta0)
    observed_bounds = certificate.supremum_interval(observed, tolerance)
    covered = certificate.contains(observed, float(critical))
    old_covered_on_real_line = certificate.contains(observed, float(saved["critical_value"]))
    epsilon = y - p @ beta0
    identity_d = solve_triangular(triangular, q.T @ epsilon)
    identity_error = float(np.linalg.norm(identity_d - (beta-beta0), ord=np.inf))
    if identity_error > 1e-9 * max(1, np.linalg.norm(beta-beta0, ord=np.inf)):
        raise ValueError("OLS observation-error identity failed numerical verification.")
    scaled_radius = float(critical) * saved["standard_error"]
    containment_polynomial = certificate.threshold_polynomial(observed**2, float(critical))
    return {
        "bootstrap_supremum_lower": bounds[:, 0], "bootstrap_supremum_upper": bounds[:, 1],
        "critical_value_lower": float(critical_low), "critical_value": float(critical),
        "observed_statistic_lower": observed_bounds[0], "observed_statistic_upper": observed_bounds[1],
        "simultaneous_covered": covered, "old_band_covered_on_real_line": old_covered_on_real_line,
        "old_grid_covered": bool(saved["simultaneous_covered"]),
        "old_grid_critical_value": float(saved["critical_value"]),
        "mean_radius_display_0_1": float(np.mean(scaled_radius)),
        "error_coefficients": beta-beta0, "observation_errors": epsilon,
        "identity_coefficient_error": identity_error, "covariance": covariance,
        "sigma_hat": float(saved["sigma_hat"]), "sigma2_hat": float(saved["sigma2_hat"]),
        "beta_hat": beta, "bootstrap_beta": saved["bootstrap_beta"],
        "denominator_exact_coefficients": np.array([str(c) for c in certificate.q_coefficients]),
        "error_exact_coefficients": np.array([str(observed.nth(i)) for i in range(3)]),
        "containment_exact_coefficients": np.array([str(containment_polynomial.nth(i)) for i in range(5)]),
    }


@dataclass(frozen=True)
class ContinuousReplayManifest:
    name: str
    payload: dict
    source_path: Path
    launch: PolicyLCBLaunchSpec


def load_manifest(path: str | Path) -> ContinuousReplayManifest:
    path = Path(path).resolve()
    payload = read_json(path)
    if payload.get("kind") != MANIFEST_KIND:
        raise ValueError("Incorrect continuous replay manifest kind.")
    for key in ("name", "source_project"):
        if not payload[key] or Path(payload[key]).name != payload[key] or payload[key] in {".", ".."}:
            raise ValueError(f"{key} must be a single directory name.")
    if payload["name"] == payload["source_project"]:
        raise ValueError("Replay must preserve the source project.")
    if payload["domain"] != "real_line" or not 0 < payload["supremum_tolerance"] < 1e-3:
        raise ValueError("Require real_line and a positive small supremum tolerance.")
    lower, upper = payload["display_window"]
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Invalid display window.")
    if payload["launch"] != {"mode": "local", "array": "none"}:
        raise ValueError("The replay uses a local, non-array launch.")
    return ContinuousReplayManifest(payload["name"], payload, path, PolicyLCBLaunchSpec(**payload["launch"]))


def source_cases(manifest: ContinuousReplayManifest, runs_root: Path) -> list[Path]:
    source = runs_root / manifest.payload["source_project"]
    source_summary = read_json(source / "summary.json")
    return [Path(record["path"]).parent for record in source_summary["provenance"]]


def run_replay_case(manifest: ContinuousReplayManifest, source: Path, runs_root: Path, force: bool):
    saved_summary = read_json(source / "summary.json")
    if saved_summary["arrays"] != file_record(source / "draws.npz"):
        raise ValueError(f"Source arrays changed: {source}")
    destination = runs_root / manifest.name / source.name
    contract = {"manifest": manifest.payload, "source": file_record(source / "summary.json"),
                "source_arrays": saved_summary["arrays"], "implementation": file_record(__file__)}
    final = destination / "summary.json"
    if not force and final.exists():
        previous = read_json(final)
        if previous["contract"] != contract or previous["arrays"] != file_record(destination / "draws.npz"):
            raise ValueError(f"Replay outputs differ; use --force: {destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    specification = saved_summary["contract"]["manifest"]
    beta0 = np.array(specification["truth"]["coefficients"])
    rows, outputs = [], []
    with np.load(source / "draws.npz") as archive:
        for index, original in enumerate(saved_summary["rows"]):
            saved = {key: archive[key][index] for key in archive.files}
            result = replay_dataset(saved, beta0, specification["bootstrap"]["delta"], manifest.payload["supremum_tolerance"])
            outputs.append(result)
            row = {key: original[key] for key in ("stage", "n", "noise_std", "dataset")}
            row.update({key: value for key, value in result.items() if np.isscalar(value)})
            rows.append(row)
            if index % 10 == 0:
                print(f"{source.name}: analytically certified dataset {index+1}/{len(saved_summary['rows'])}", flush=True)
    np.savez_compressed(destination / "draws.npz", **{
        key: np.stack([output[key] for output in outputs]) for key in outputs[0]
    })
    write_json_atomic(final, {"contract": contract, "arrays": file_record(destination / "draws.npz"),
                              "rows": rows, "coverage_label": COVERAGE_LABEL})


def collect(manifest: ContinuousReplayManifest, runs_root: Path):
    from experiments.bootstrap_band_continuous_reporting import write_plots
    project = runs_root / manifest.name
    rows, summaries, records = [], [], []
    for source in source_cases(manifest, runs_root):
        path = project / source.name / "summary.json"
        saved = read_json(path)
        if saved["arrays"] != file_record(path.with_name("draws.npz")):
            raise ValueError(f"Replay arrays changed: {path}")
        rows.extend(saved["rows"])
        records.append(file_record(path))
        group = saved["rows"]
        if group[0]["stage"] != "stage2":
            continue
        count = sum(row["simultaneous_covered"] for row in group)
        low, high = wilson_interval(count, len(group))
        summaries.append({
            "n": group[0]["n"], "noise_std": group[0]["noise_std"], "datasets": len(group),
            "covered": count, "coverage_rate": count/len(group),
            "wilson_95_low": low, "wilson_95_high": high,
            "old_grid_covered": sum(row["old_grid_covered"] for row in group),
            "old_band_real_line_covered": sum(row["old_band_covered_on_real_line"] for row in group),
            "mean_radius_display_0_1": float(np.mean([row["mean_radius_display_0_1"] for row in group])),
            "critical_value_mean": float(np.mean([row["critical_value"] for row in group])),
        })
    write_rows_csv(project / "dataset_metrics.csv", rows, tuple(rows[0]))
    write_rows_csv(project / "coverage_summary.csv", summaries, tuple(summaries[0]))
    plots = write_plots(project, runs_root / manifest.payload["source_project"], manifest.payload, rows, summaries)
    write_json_atomic(project / "summary.json", {
        "stage1": rows[0], "stage2": summaries, "coverage_label": COVERAGE_LABEL,
        "manifest": manifest.payload, "provenance": records,
        "calculation_source": file_record(__file__),
        "plots": [file_record(path) for path in plots],
        "plotting_source": file_record(Path(__file__).with_name("bootstrap_band_continuous_reporting.py")),
    })
    coefficient_rows = []
    for source in source_cases(manifest, runs_root):
        path = project / source.name / "draws.npz"
        with np.load(path) as data:
            for j in range(3):
                coefficient_rows.append({
                    "case": source.name, "dataset": 0, "coefficient": j,
                    "beta_hat": float(data["beta_hat"][0, j]),
                    "beta_star_1": float(data["bootstrap_beta"][0, 0, j]),
                    "bootstrap_q025": float(np.quantile(data["bootstrap_beta"][0, :, j], .025)),
                    "bootstrap_q975": float(np.quantile(data["bootstrap_beta"][0, :, j], .975)),
                })
    write_rows_csv(project / "bootstrap_coefficients.csv", coefficient_rows, tuple(coefficient_rows[0]))
    (project / "EXPERIMENT.md").write_text(experiment_text(manifest), encoding="utf-8")
    print(f"Completed analytical replay: {project}", flush=True)


def experiment_text(manifest):
    import json
    return f"""# Analytical bootstrap containment on the real line

{COVERAGE_LABEL}

This reanalysis preserves all saved observations and bootstrap coefficients from
{manifest.payload['source_project']}. No new random draws or optimizer actions
are introduced. Source JSON/NPZ SHA-256 records are saved in each result.

Every bootstrap calibration statistic is sup over all real a of
|p(a)'(beta_b_star-beta_hat)|/[sigma_hat sqrt(p(a)' V p(a))]. Each supremum is
stored as certified lower/upper bounds. The degree-at-most-four stationary
equation and analytic tail limit propose the bound; exact rational polynomial
root/multiplicity sign checks certify it. The upper-bound empirical quantile
(method='higher') defines c_hat, with its lower bracket also saved. The original
sigma_hat stays fixed in all bootstrap denominators. Truth never calibrates c_hat.

For the observed fit e(a)=p(a)'(beta_hat-beta0), coverage is the exact sign
decision H(a)=c_hat^2 sigma_hat^2 p(a)'V p(a)-e(a)^2 >= 0 for EVERY real a.
Binary floating-point fit and covariance values are interpreted as exact
rationals in this polynomial decision. This certifies the represented band,
not exact arithmetic in the earlier OLS fit. The observation-error OLS identity
is checked numerically for every dataset. Zero/degenerate polynomial cases,
even-multiplicity real roots, and both tails are included.

Analytical extrema here are explicitly requested coverage statistics, not
reported pricing/optimizer solutions. No plotting grid enters calibration or
containment. Display windows merely render the analytic curves. Average width
is labeled on [0,1]; an average over the entire real line is not reported.

One original dataset yields one coverage Boolean. Across R independently drawn
original datasets the rate is sum(C_r)/R, with a 95% Wilson interval. Conditions
retain the original independent seeds, n, noise SD, bootstrap count, and confidence level.
This is a correction to the completed analysis, not the proposed B/noise sweep.
The old across-dataset coefficient figure is omitted from this report.

The fixed-denominator bootstrap remains an approximate coverage
procedure: certifying containment for a realized band does not prove that its
sampling coverage equals 95%. Old grid results remain intact in their source
directory; this report additionally records how those old bands fare on R.

Run scripts/run_experiment_manifest.py {manifest.source_path} --launch local.

```json
{json.dumps(manifest.payload, indent=2)}
```
"""


def build_launch_plan(manifest, *, runs_root: str | None, force: bool):
    def run_all(context):
        for source in source_cases(manifest, context.runs_root):
            run_replay_case(manifest, source, context.runs_root, force)
        collect(manifest, context.runs_root)
    def run_task(index, context):
        if index != 0:
            raise IndexError(index)
        run_all(context)
    return LaunchPlan(name=manifest.name, task_count=1, requires_jax=False,
                      run_task=run_task, run_all=run_all,
                      collect=lambda context: collect(manifest, context.runs_root),
                      runs_root=runs_root, default_launch="local", default_array=False)
