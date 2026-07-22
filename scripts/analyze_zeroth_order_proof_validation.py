"""Analyze completed zeroth-order proof manifests without rerunning optimization."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import t as student_t

from experiments.configs import get_config
from experiments.manifest import ExperimentManifest, ManifestVariant, load_experiment_manifest
from experiments.paths import results_root
from objective import ThetaBias, ThetaBiasedObjective, ZerothOrderProofObjective


MU = ZerothOrderProofObjective.mu
SMOOTHNESS = ZerothOrderProofObjective.smoothness
RHO = ZerothOrderProofObjective.third_derivative_bound
BIAS_LABELS = {
    "LinearThetaBias": "linear",
    "ArctanThetaBias": "arctan",
    "ArctanRemainderThetaBias": "remainder",
}
ESTIMATOR_LABELS = {
    "finite_difference": "Finite difference",
    "stein_difference": "Stein difference",
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--bias-manifest", required=True)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def _base_value(x: float) -> float:
    return float(x * x + 0.5 * (np.sin(x) - x))


def _base_grad(x: np.ndarray | float) -> np.ndarray | float:
    return 2.0 * np.asarray(x) + 0.5 * (np.cos(x) - 1.0)


def _bias_value(bias: ThetaBias | None, x: float) -> float:
    return 0.0 if bias is None else bias.value(np.asarray([x], dtype=float))


def _bias_grad(bias: ThetaBias | None, x: np.ndarray | float) -> np.ndarray | float:
    if bias is None:
        return np.zeros_like(np.asarray(x), dtype=float)
    values = np.asarray(x, dtype=float)
    if values.ndim == 0:
        return float(bias.grad(np.asarray([float(values)]))[0])
    return np.asarray([bias.grad(np.asarray([float(value)]))[0] for value in values])


def _root(fn: Callable[[float], float]) -> float:
    radius = 1.0
    for _ in range(12):
        low, high = -radius, radius
        f_low, f_high = float(fn(low)), float(fn(high))
        if f_low == 0.0:
            return low
        if f_high == 0.0:
            return high
        if f_low < 0.0 < f_high:
            return float(brentq(fn, low, high, xtol=1e-14, rtol=1e-14))
        radius *= 2.0
    raise RuntimeError("Could not bracket the unique strongly-convex root.")


def biased_optimum(bias: ThetaBias | None) -> float:
    r"""Return $$x_b^\star$$, the root of the biased analytical gradient."""
    return _root(lambda x: float(_base_grad(x) + _bias_grad(bias, x)))


def estimator_root(estimator: str, sigma: float, bias: ThetaBias | None) -> float:
    """Return the population fixed point for FD or Stein difference."""
    if estimator == "finite_difference":
        return _root(
            lambda x: (
                _base_value(x + sigma)
                + _bias_value(bias, x + sigma)
                - _base_value(x - sigma)
                - _bias_value(bias, x - sigma)
            )
            / (2.0 * sigma)
        )
    if estimator == "stein_difference":
        nodes, weights = np.polynomial.hermite.hermgauss(80)
        normal_nodes = np.sqrt(2.0) * nodes
        normal_weights = weights / np.sqrt(np.pi)
        return _root(
            lambda x: float(
                np.sum(
                    normal_weights
                    * (
                        _base_grad(x + sigma * normal_nodes)
                        + _bias_grad(bias, x + sigma * normal_nodes)
                    )
                )
            )
        )
    raise ValueError(f"Unsupported estimator {estimator!r}.")


def _bias_metadata(config: object) -> tuple[ThetaBias | None, str, float, float, float, float, float]:
    objective = getattr(config, "objective")
    if not isinstance(objective, ThetaBiasedObjective):
        return None, "none", 0.0, 0.0, 0.0, 0.0, 0.0
    bias = objective.bias
    bounds = bias.derivative_bounds()
    return (
        bias,
        BIAS_LABELS[type(bias).__name__],
        float(bias.alpha),
        bounds.beta,
        bounds.kappa_minus,
        bounds.kappa_plus,
        bounds.rho,
    )


def _theorem_bound(
    estimator: str,
    *,
    x0: float,
    x_b_star: float,
    x_estimator_star: float,
    sigma: float,
    m: int,
    eta: float,
    k_steps: int,
    beta: float,
    kappa_minus: float,
    kappa_plus: float,
    rho_bias: float,
) -> float:
    if estimator == "finite_difference":
        q = max(
            abs(1.0 - eta * (MU - kappa_minus)),
            abs(1.0 - eta * (SMOOTHNESS + kappa_plus)),
        )
        return float(q**k_steps * abs(x0 - x_estimator_star) + beta / MU + RHO * sigma**2 / (6.0 * MU))
    effective_mu = MU - kappa_minus
    effective_l = SMOOTHNESS + kappa_plus
    if eta > effective_mu / (12.0 * effective_l**2) + 1e-15:
        return float("nan")
    effective_rho = RHO + rho_bias
    return float(
        2.0 * (1.0 - eta * effective_mu) ** k_steps * (x0 - x_b_star) ** 2
        + effective_rho**2 * sigma**4 / effective_mu**2
        + 2.0
        * eta
        * effective_rho**2
        * sigma**4
        / effective_mu
        * (0.5 + 35.0 / (6.0 * m))
        + 2.0 * beta**2 / MU**2
    )


def collect_run_rows(
    manifest: ExperimentManifest,
    *,
    runs_root: str | Path | None = None,
) -> list[dict[str, object]]:
    """Read every requested summary and calculate per-run displacement landmarks."""
    rows: list[dict[str, object]] = []
    missing: list[Path] = []
    for variant in manifest.variants:
        config = get_config(manifest.base_preset, overrides=variant.overrides)
        bias, bias_form, alpha, beta, kappa_minus, kappa_plus, rho_bias = _bias_metadata(config)
        x0 = float(np.asarray(config.theta0, dtype=float)[0])
        x_star = 0.0
        x_b_star = biased_optimum(bias)
        roots = {
            estimator: estimator_root(estimator, float(config.sigma), bias)
            for estimator in config.enabled_estimators
        }
        for seed in manifest.seeds.run_seeds:
            path = manifest.variant_dir(variant, runs_root) / f"summary-seed-{seed}.json"
            if not path.exists():
                missing.append(path)
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for estimator, estimator_payload in payload.get("estimators", {}).items():
                if estimator not in roots:
                    continue
                x_k = float(estimator_payload["theta"][0])
                x_estimator_star = roots[estimator]
                bound = _theorem_bound(
                    estimator,
                    x0=x0,
                    x_b_star=x_b_star,
                    x_estimator_star=x_estimator_star,
                    sigma=float(config.sigma),
                    m=int(config.n_grad_samples),
                    eta=float(config.step_size),
                    k_steps=int(config.t_steps),
                    beta=beta,
                    kappa_minus=kappa_minus,
                    kappa_plus=kappa_plus,
                    rho_bias=rho_bias,
                )
                rows.append(
                    {
                        "project": manifest.name,
                        "variant": variant.name,
                        "sweep": variant.axes.get("sweep", "bias"),
                        "run_seed": int(seed),
                        "estimator": estimator,
                        "bias_form": bias_form,
                        "alpha": alpha,
                        "sigma": float(config.sigma),
                        "m": int(config.n_grad_samples),
                        "eta": float(config.step_size),
                        "k_steps": int(config.t_steps),
                        "beta": beta,
                        "kappa_minus": kappa_minus,
                        "kappa_plus": kappa_plus,
                        "rho_bias": rho_bias,
                        "strong_convexity_retained": kappa_minus < MU,
                        "stein_step_valid": (
                            True
                            if estimator != "stein_difference"
                            else float(config.step_size)
                            <= (MU - kappa_minus) / (12.0 * (SMOOTHNESS + kappa_plus) ** 2)
                        ),
                        "x0": x0,
                        "x_k": x_k,
                        "x_star": x_star,
                        "x_b_star": x_b_star,
                        "x_estimator_star": x_estimator_star,
                        "functional_bias_signed": x_b_star - x_star,
                        "functional_bias_abs": abs(x_b_star - x_star),
                        "smoothing_signed": x_estimator_star - x_b_star,
                        "smoothing_abs": abs(x_estimator_star - x_b_star),
                        "finite_run_signed": x_k - x_estimator_star,
                        "finite_run_abs": abs(x_k - x_estimator_star),
                        "truth_error_signed": x_k - x_star,
                        "truth_error_abs": abs(x_k - x_star),
                        "truth_error_squared": (x_k - x_star) ** 2,
                        "theorem_bound": bound,
                        "theorem_metric": "absolute_error" if estimator == "finite_difference" else "squared_error",
                        "summary_path": str(path),
                    }
                )
    if missing:
        preview = "\n".join(str(path) for path in missing[:8])
        raise FileNotFoundError(f"Missing {len(missing)} requested seed summaries; first paths:\n{preview}")
    _add_paired_bias_deltas(rows)
    return rows


def _add_paired_bias_deltas(rows: list[dict[str, object]]) -> None:
    baselines: dict[tuple[str, str, int], float] = {}
    for row in rows:
        if row["bias_form"] != "none" and float(row["alpha"]) == 0.0:
            baselines[(str(row["bias_form"]), str(row["estimator"]), int(row["run_seed"]))] = float(row["x_k"])
    for row in rows:
        key = (str(row["bias_form"]), str(row["estimator"]), int(row["run_seed"]))
        baseline = baselines.get(key)
        row["paired_xk_delta"] = "" if baseline is None else float(row["x_k"]) - baseline


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["project"]), str(row["variant"]), str(row["estimator"]))].append(row)
    output: list[dict[str, object]] = []
    for group in groups.values():
        first = group[0]
        x_values = np.asarray([float(row["x_k"]) for row in group])
        x_star = float(first["x_star"])
        x_estimator_star = float(first["x_estimator_star"])
        n = x_values.size
        mean_x = float(np.mean(x_values))
        variance_population = float(np.var(x_values, ddof=0))
        variance_unbiased = float(np.var(x_values, ddof=1)) if n > 1 else 0.0
        standard_error = float(np.sqrt(variance_unbiased / n)) if n > 1 else 0.0
        t_critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else 0.0
        total_mse = float(np.mean((x_values - x_star) ** 2))
        theorem_observed = abs(mean_x - x_star) if first["estimator"] == "finite_difference" else total_mse
        paired = [float(row["paired_xk_delta"]) for row in group if row["paired_xk_delta"] != ""]
        output.append(
            {
                **{
                    key: first[key]
                    for key in (
                        "project",
                        "variant",
                        "sweep",
                        "estimator",
                        "bias_form",
                        "alpha",
                        "sigma",
                        "m",
                        "eta",
                        "k_steps",
                        "beta",
                        "kappa_minus",
                        "kappa_plus",
                        "rho_bias",
                        "x0",
                        "x_star",
                        "x_b_star",
                        "x_estimator_star",
                        "functional_bias_signed",
                        "functional_bias_abs",
                        "smoothing_signed",
                        "smoothing_abs",
                        "theorem_bound",
                        "theorem_metric",
                        "strong_convexity_retained",
                        "stein_step_valid",
                    )
                },
                "n_seeds": n,
                "mean_x_k": mean_x,
                "std_x_k": float(np.sqrt(variance_unbiased)),
                "ci95_low": mean_x - t_critical * standard_error,
                "ci95_high": mean_x + t_critical * standard_error,
                "variance_x_k": variance_unbiased,
                "variance_population": variance_population,
                "mean_finite_run_signed": mean_x - x_estimator_star,
                "finite_run_rmse": float(np.sqrt(np.mean((x_values - x_estimator_star) ** 2))),
                "mean_truth_error_signed": mean_x - x_star,
                "mean_truth_error_abs": abs(mean_x - x_star),
                "squared_mean_truth_error": (mean_x - x_star) ** 2,
                "total_mse": total_mse,
                "mse_decomposition_residual": total_mse - ((mean_x - x_star) ** 2 + variance_population),
                "paired_delta_mean": float(np.mean(paired)) if paired else "",
                "paired_delta_mse": float(np.mean(np.square(paired))) if paired else "",
                "theorem_observed": theorem_observed,
                "theorem_pass": bool(np.isfinite(float(first["theorem_bound"])) and theorem_observed <= float(first["theorem_bound"])),
            }
        )
    return sorted(output, key=lambda row: (str(row["project"]), str(row["sweep"]), str(row["bias_form"]), float(row["alpha"]), float(row["sigma"]), int(row["m"]), str(row["estimator"])))


def loglog_fit(name: str, x: Iterable[float], y: Iterable[float], expected: float, tolerance: float) -> dict[str, object]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0.0) & (y_arr > 0.0)
    if np.count_nonzero(valid) < 3:
        return {"name": name, "slope": "", "r2": "", "expected": expected, "tolerance": tolerance, "pass": False}
    log_x, log_y = np.log(x_arr[valid]), np.log(y_arr[valid])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    predicted = intercept + slope * log_x
    denominator = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 if denominator == 0.0 else 1.0 - float(np.sum((log_y - predicted) ** 2)) / denominator
    return {
        "name": name,
        "slope": float(slope),
        "r2": r2,
        "expected": expected,
        "tolerance": tolerance,
        "pass": abs(float(slope) - expected) <= tolerance,
    }


def scaling_fits(aggregates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    fits: list[dict[str, object]] = []
    sigma_rows = [row for row in aggregates if row["sweep"] == "sigma"]
    for estimator in ("finite_difference", "stein_difference"):
        subset = sorted((row for row in sigma_rows if row["estimator"] == estimator), key=lambda row: float(row["sigma"]))
        fits.append(loglog_fit(f"{estimator}: population displacement vs sigma", (float(r["sigma"]) for r in subset), (float(r["smoothing_abs"]) for r in subset), 2.0, 0.25))
    fd = [row for row in sigma_rows if row["estimator"] == "finite_difference"]
    fits.append(loglog_fit("finite_difference: final error vs sigma", (float(r["sigma"]) for r in fd), (float(r["mean_truth_error_abs"]) for r in fd), 2.0, 0.25))
    stein = [row for row in sigma_rows if row["estimator"] == "stein_difference"]
    fits.append(loglog_fit("stein_difference: MSE vs sigma", (float(r["sigma"]) for r in stein), (float(r["total_mse"]) for r in stein), 4.0, 0.5))
    m_rows = sorted((row for row in aggregates if row["sweep"] == "m"), key=lambda row: int(row["m"]))
    fits.append(loglog_fit("stein_difference: variance vs m", (float(r["m"]) for r in m_rows), (float(r["variance_x_k"]) for r in m_rows), -1.0, 0.5))

    for bias_form in ("linear", "arctan"):
        subset = [row for row in aggregates if row["bias_form"] == bias_form and row["estimator"] == "finite_difference" and float(row["alpha"]) > 0.0]
        fits.append(loglog_fit(f"{bias_form}: functional displacement vs alpha", (float(r["alpha"]) for r in subset), (float(r["functional_bias_abs"]) for r in subset), 1.0, 0.25))
    for estimator in ("finite_difference", "stein_difference"):
        subset = sorted((row for row in aggregates if row["bias_form"] == "remainder" and row["estimator"] == estimator), key=lambda row: float(row["alpha"]))
        if subset:
            baseline = float(subset[0]["x_estimator_star"])
            positive = [row for row in subset if float(row["alpha"]) > 0.0]
            fits.append(loglog_fit(f"remainder/{estimator}: estimator-root change vs alpha", (float(r["alpha"]) for r in positive), (abs(float(r["x_estimator_star"]) - baseline) for r in positive), 1.0, 0.25))
    return fits


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _estimator_rows(rows: Sequence[Mapping[str, object]], sweep: str, estimator: str) -> list[Mapping[str, object]]:
    key = "m" if sweep == "m" else "sigma"
    return sorted((row for row in rows if row["sweep"] == sweep and row["estimator"] == estimator), key=lambda row: float(row[key]))


def _plot_sigma_landmarks(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for column, estimator in enumerate(("finite_difference", "stein_difference")):
        subset = _estimator_rows(rows, "sigma", estimator)
        x = np.asarray([float(row["sigma"]) for row in subset])
        for axis in axes[:, column]:
            axis.axhline(float(subset[0]["x0"]), color="0.65", linestyle=":", label="$x_0$")
            axis.axhline(0.0, color="black", linewidth=1.0, label="$x^*$")
            axis.plot(x, [float(row["x_estimator_star"]) for row in subset], "--", label="$x^*_{est}$")
            mean = np.asarray([float(row["mean_x_k"]) for row in subset])
            low = np.asarray([float(row["ci95_low"]) for row in subset])
            high = np.asarray([float(row["ci95_high"]) for row in subset])
            axis.plot(x, mean, "o-", label="$x_K$")
            axis.fill_between(x, low, high, alpha=0.2)
            axis.grid(alpha=0.25)
        axes[0, column].set_title(ESTIMATOR_LABELS[estimator])
        zoom_values = [0.0, *[float(row["x_estimator_star"]) for row in subset], *[float(row["ci95_low"]) for row in subset], *[float(row["ci95_high"]) for row in subset]]
        margin = max(1e-4, 0.15 * (max(zoom_values) - min(zoom_values)))
        axes[1, column].set_ylim(min(zoom_values) - margin, max(zoom_values) + margin)
        axes[1, column].set_xlabel("Perturbation radius $\\sigma$")
    axes[0, 0].set_ylabel("Position (full range)")
    axes[1, 0].set_ylabel("Position (near optimum)")
    axes[0, 0].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "sigma_landmarks.png", dpi=180)
    plt.close(fig)


def _plot_sigma_decomposition(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for column, estimator in enumerate(("finite_difference", "stein_difference")):
        subset = _estimator_rows(rows, "sigma", estimator)
        sigma = np.asarray([float(row["sigma"]) for row in subset])
        axes[0, column].loglog(sigma, [float(row["smoothing_abs"]) for row in subset], "o-", label="smoothing")
        axes[0, column].loglog(sigma, [float(row["finite_run_rmse"]) for row in subset], "o-", label="finite-run RMS")
        axes[0, column].loglog(sigma, [np.sqrt(float(row["total_mse"])) for row in subset], "o-", label="truth RMSE")
        reference = float(subset[0]["smoothing_abs"]) * (sigma / sigma[0]) ** 2
        axes[0, column].loglog(sigma, reference, "k:", label="$\\sigma^2$")
        axes[1, column].loglog(sigma, [float(row["smoothing_abs"]) ** 2 for row in subset], "o-", label="smoothing squared")
        axes[1, column].loglog(sigma, [float(row["total_mse"]) for row in subset], "o-", label="total MSE")
        axes[1, column].loglog(sigma, reference**2, "k:", label="$\\sigma^4$")
        for axis in axes[:, column]:
            axis.grid(alpha=0.25, which="both")
            axis.legend(fontsize=8)
        axes[0, column].set_title(ESTIMATOR_LABELS[estimator])
        axes[1, column].set_xlabel("Perturbation radius $\\sigma$")
    axes[0, 0].set_ylabel("Absolute/RMS displacement")
    axes[1, 0].set_ylabel("Squared displacement")
    fig.tight_layout()
    fig.savefig(output / "sigma_displacement_decomposition.png", dpi=180)
    plt.close(fig)


def _plot_m(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    subset = _estimator_rows(rows, "m", "stein_difference")
    m = np.asarray([int(row["m"]) for row in subset])
    mean = np.asarray([float(row["mean_x_k"]) for row in subset])
    low = np.asarray([float(row["ci95_low"]) for row in subset])
    high = np.asarray([float(row["ci95_high"]) for row in subset])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis in axes:
        axis.axhline(0.0, color="black", label="$x^*$")
        axis.axhline(float(subset[0]["x_estimator_star"]), linestyle="--", label="$x^*_{est}$")
        axis.plot(m, mean, "o-", label="$x_K$")
        axis.fill_between(m, low, high, alpha=0.2)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Stein samples $m$")
        axis.grid(alpha=0.25)
    axes[0].axhline(float(subset[0]["x0"]), color="0.65", linestyle=":", label="$x_0$")
    axes[0].set_title("Full range")
    zoom = [0.0, float(subset[0]["x_estimator_star"]), *low, *high]
    margin = 0.15 * (max(zoom) - min(zoom))
    axes[1].set_ylim(min(zoom) - margin, max(zoom) + margin)
    axes[1].set_title("Near optimum")
    axes[0].set_ylabel("Position")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "m_landmarks.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    variance = np.asarray([float(row["variance_x_k"]) for row in subset])
    axis.loglog(m, [float(row["smoothing_abs"]) ** 2 for row in subset], "o-", label="population smoothing squared")
    axis.loglog(m, variance, "o-", label="across-seed variance")
    axis.loglog(m, [float(row["total_mse"]) for row in subset], "o-", label="total MSE")
    axis.loglog(m, variance[0] * m[0] / m, "k:", label="$1/m$")
    axis.set_xlabel("Stein samples $m$")
    axis.set_ylabel("Squared error")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "m_mse_decomposition.png", dpi=180)
    plt.close(fig)


def _bias_rows(rows: Sequence[Mapping[str, object]], form: str, estimator: str) -> list[Mapping[str, object]]:
    return sorted((row for row in rows if row["bias_form"] == form and row["estimator"] == estimator), key=lambda row: float(row["alpha"]))


def _plot_bias_landmarks(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    forms = ("linear", "arctan", "remainder")
    estimators = ("finite_difference", "stein_difference")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    for row_idx, estimator in enumerate(estimators):
        for col_idx, form in enumerate(forms):
            subset = _bias_rows(rows, form, estimator)
            alpha = np.asarray([float(row["alpha"]) for row in subset])
            axis = axes[row_idx, col_idx]
            axis.axhline(float(subset[0]["x0"]), color="0.7", linestyle=":", label="$x_0$")
            axis.axhline(0.0, color="black", linewidth=1.0, label="$x^*$")
            axis.plot(alpha, [float(row["x_b_star"]) for row in subset], "s--", label="$x_b^*$")
            axis.plot(alpha, [float(row["x_estimator_star"]) for row in subset], "^--", label="$x_{est}^*$")
            mean = np.asarray([float(row["mean_x_k"]) for row in subset])
            low = np.asarray([float(row["ci95_low"]) for row in subset])
            high = np.asarray([float(row["ci95_high"]) for row in subset])
            axis.plot(alpha, mean, "o-", label="$x_K$")
            axis.fill_between(alpha, low, high, alpha=0.2)
            axis.grid(alpha=0.25)
            if row_idx == 0:
                axis.set_title(form)
            if col_idx == 0:
                axis.set_ylabel(f"{ESTIMATOR_LABELS[estimator]}\nposition")
            if row_idx == 1:
                axis.set_xlabel("Bias strength $\\alpha$")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "bias_landmarks.png", dpi=180)
    plt.close(fig)


def _plot_bias_decomposition(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    forms = ("linear", "arctan", "remainder")
    estimators = ("finite_difference", "stein_difference")
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=False)
    for estimator_idx, estimator in enumerate(estimators):
        for col_idx, form in enumerate(forms):
            subset = _bias_rows(rows, form, estimator)
            alpha = [float(row["alpha"]) for row in subset]
            signed_axis = axes[2 * estimator_idx, col_idx]
            absolute_axis = axes[2 * estimator_idx + 1, col_idx]
            signed_axis.axhline(0.0, color="black", linewidth=0.8)
            signed_axis.plot(alpha, [float(row["functional_bias_signed"]) for row in subset], "o-", label="functional bias")
            signed_axis.plot(alpha, [float(row["smoothing_signed"]) for row in subset], "o-", label="smoothing")
            signed_axis.plot(alpha, [float(row["mean_finite_run_signed"]) for row in subset], "o-", label="finite-run mean")
            signed_axis.plot(alpha, [float(row["mean_truth_error_signed"]) for row in subset], "o-", label="total truth error")
            signed_axis.grid(alpha=0.25)

            positive = [row for row in subset if float(row["alpha"]) > 0.0]
            positive_alpha = [float(row["alpha"]) for row in positive]
            absolute_axis.loglog(positive_alpha, [float(row["functional_bias_abs"]) for row in positive], "o-", label="functional bias")
            absolute_axis.loglog(positive_alpha, [float(row["smoothing_abs"]) for row in positive], "o-", label="smoothing")
            absolute_axis.loglog(positive_alpha, [float(row["finite_run_rmse"]) for row in positive], "o-", label="finite-run RMS")
            absolute_axis.loglog(positive_alpha, [np.sqrt(float(row["total_mse"])) for row in positive], "o-", label="truth RMSE")
            absolute_axis.grid(alpha=0.25, which="both")

            if estimator_idx == 0:
                signed_axis.set_title(form)
            if col_idx == 0:
                signed_axis.set_ylabel(f"{ESTIMATOR_LABELS[estimator]}\nsigned")
                absolute_axis.set_ylabel("absolute/RMS")
            absolute_axis.set_xlabel("Bias strength $\\alpha$")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "bias_displacement_decomposition.png", dpi=180)
    plt.close(fig)


def _plot_bias_bounds(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    for row_idx, estimator in enumerate(("finite_difference", "stein_difference")):
        for col_idx, form in enumerate(("linear", "arctan", "remainder")):
            subset = _bias_rows(rows, form, estimator)
            alpha = [float(row["alpha"]) for row in subset]
            observed = [float(row["theorem_observed"]) for row in subset]
            bound = [float(row["theorem_bound"]) for row in subset]
            axis = axes[row_idx, col_idx]
            axis.plot(alpha, observed, "o-", label="observed")
            axis.plot(alpha, bound, "--", label="theorem bound")
            axis.set_yscale("log")
            axis.grid(alpha=0.25, which="both")
            if row_idx == 0:
                axis.set_title(form)
            if col_idx == 0:
                metric = "absolute error" if estimator == "finite_difference" else "MSE"
                axis.set_ylabel(f"{ESTIMATOR_LABELS[estimator]}\n{metric}")
            if row_idx == 1:
                axis.set_xlabel("Bias strength $\\alpha$")
    axes[0, 0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "bias_proof_bounds.png", dpi=180)
    plt.close(fig)


def _plot_scaling(fits: Sequence[Mapping[str, object]], output: Path) -> None:
    valid = [fit for fit in fits if fit["slope"] != ""]
    y = np.arange(len(valid))
    slopes = [float(fit["slope"]) for fit in valid]
    expected = [float(fit["expected"]) for fit in valid]
    fig, axis = plt.subplots(figsize=(10, max(5, 0.55 * len(valid))))
    axis.barh(y - 0.18, slopes, height=0.36, label="fitted")
    axis.barh(y + 0.18, expected, height=0.36, alpha=0.45, label="expected")
    axis.set_yticks(y, [str(fit["name"]) for fit in valid])
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Log-log exponent")
    axis.grid(alpha=0.25, axis="x")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "scaling_summary.png", dpi=180)
    plt.close(fig)


def write_outputs(
    run_rows: Sequence[Mapping[str, object]],
    aggregates: Sequence[Mapping[str, object]],
    fits: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_run_metrics.csv", run_rows)
    _write_csv(output_dir / "aggregate_metrics.csv", aggregates)
    _write_csv(output_dir / "scaling_fits.csv", fits)
    theorem_rows = [
        {key: row[key] for key in ("project", "variant", "estimator", "bias_form", "alpha", "sigma", "m", "theorem_metric", "theorem_observed", "theorem_bound", "theorem_pass", "strong_convexity_retained", "stein_step_valid")}
        for row in aggregates
    ]
    _write_csv(output_dir / "theorem_checks.csv", theorem_rows)
    baseline = [row for row in aggregates if row["bias_form"] == "none"]
    bias = [row for row in aggregates if row["bias_form"] != "none"]
    if baseline:
        _plot_sigma_landmarks(baseline, output_dir)
        _plot_sigma_decomposition(baseline, output_dir)
        _plot_m(baseline, output_dir)
    if bias:
        _plot_bias_landmarks(bias, output_dir)
        _plot_bias_decomposition(bias, output_dir)
        _plot_bias_bounds(bias, output_dir)
    _plot_scaling(fits, output_dir)
    theorem_passes = sum(bool(row["theorem_pass"]) for row in aggregates)
    fit_passes = sum(bool(row["pass"]) for row in fits)
    lines = [
        "# Zeroth-Order Proof Validation",
        "",
        f"- Aggregate theorem checks passed: {theorem_passes}/{len(aggregates)}",
        f"- Scaling checks passed: {fit_passes}/{len(fits)}",
        "",
        "## Scaling fits",
        "",
        "| Check | Slope | Expected | R² | Pass |",
        "|---|---:|---:|---:|:---:|",
    ]
    for fit in fits:
        slope = fit["slope"] if fit["slope"] == "" else f"{float(fit['slope']):.4f}"
        r2 = fit["r2"] if fit["r2"] == "" else f"{float(fit['r2']):.4f}"
        lines.append(f"| {fit['name']} | {slope} | {float(fit['expected']):.2f} | {r2} | {'yes' if fit['pass'] else 'no'} |")
    (output_dir / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    baseline_manifest = load_experiment_manifest(args.baseline_manifest)
    bias_manifest = load_experiment_manifest(args.bias_manifest)
    root = results_root() if args.runs_root is None else Path(args.runs_root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "zeroth-order-proof-validation-analysis"
    run_rows = [
        *collect_run_rows(baseline_manifest, runs_root=root),
        *collect_run_rows(bias_manifest, runs_root=root),
    ]
    aggregates = aggregate_rows(run_rows)
    fits = scaling_fits(aggregates)
    write_outputs(run_rows, aggregates, fits, output_dir)
    print(f"Wrote zeroth-order proof validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
