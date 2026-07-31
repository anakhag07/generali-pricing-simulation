"""Analyze completed support-envelope sweeps without rerunning optimization."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import t as student_t

from experiments.configs import get_config
from experiments.manifest import ExperimentManifest, ManifestVariant, load_experiment_manifest
from experiments.paths import results_root
from objective import (
    ConstantThetaRegularizer,
    IntervalDistanceThetaRegularizer,
    RegularizedObjective,
    SmoothSaturatingIntervalThetaRegularizer,
    ThetaRegularizer,
)
from scripts.zeroth_order_landscape import (
    StationaryPoint,
    find_stationary_points,
    global_minimum,
)


X_DUMMY = np.zeros((1, 1), dtype=float)
DOMAIN = (-1.5, 2.25)
FORM_LABELS = {
    "constant": "Constant",
    "linear": "Constant derivative",
    "smooth_nonconvex": r"Smooth nonconvex ($C^\infty$)",
}
FORM_ORDER = ("constant", "linear", "smooth_nonconvex")
ESTIMATOR_LABELS = {
    "finite_difference": "Finite difference",
    "stein_difference": "Stein difference",
}
INIT_STYLES = {
    0.0: ("tab:blue", "o"),
    1.0: ("tab:orange", "^"),
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def _regularizer_metadata(
    config: object,
) -> tuple[ThetaRegularizer, str, float, float, float, float, float]:
    objective = getattr(config, "objective")
    if not isinstance(objective, RegularizedObjective) or len(objective.regularizers) != 1:
        raise ValueError("Envelope variants must contain exactly one theta regularizer.")
    regularizer = objective.regularizers[0]
    if isinstance(regularizer, ConstantThetaRegularizer):
        return regularizer, "constant", regularizer.height, 0.0, np.nan, np.nan, 0.0
    if isinstance(regularizer, IntervalDistanceThetaRegularizer):
        return (
            regularizer,
            "linear",
            float(getattr(config, "_envelope_amplitude", np.nan)),
            regularizer.slope,
            regularizer.lower,
            regularizer.upper,
            0.0,
        )
    if isinstance(regularizer, SmoothSaturatingIntervalThetaRegularizer):
        return (
            regularizer,
            "smooth_nonconvex",
            regularizer.amplitude,
            _smooth_peak_slope(regularizer.amplitude, regularizer.transition_width),
            regularizer.lower,
            regularizer.upper,
            regularizer.transition_width,
        )
    raise TypeError(f"Unsupported envelope regularizer {type(regularizer).__name__}.")


def _variant_metadata(
    variant: ManifestVariant,
    config: object,
) -> dict[str, object]:
    regularizer, form, amplitude, slope, lower, upper, width = _regularizer_metadata(config)
    axes = variant.axes
    amplitude = float(axes.get("amplitude", amplitude))
    return {
        "variant": variant.name,
        "form": str(axes.get("form", form)),
        "amplitude": amplitude,
        "slope": float(axes.get("slope", slope)),
        "lower": float(axes.get("lower", lower)),
        "upper": float(axes.get("upper", upper)),
        "transition_width": float(axes.get("transition_width", width)),
        "theta0": float(np.asarray(getattr(config, "theta0"), dtype=float)[0]),
        "sigma": float(getattr(config, "sigma")),
        "regularizer": regularizer,
    }


def _smooth_peak_slope(amplitude: float, transition_width: float) -> float:
    factor = 2.0 * (1.5**1.5) * np.exp(-1.5)
    return float(factor * float(amplitude) / float(transition_width))


def _value_fn(objective: object):
    return lambda x: float(
        getattr(objective, "value")(np.asarray([float(x)], dtype=float), X_DUMMY)
    )


def _grad_fn(objective: object):
    return lambda x: float(
        np.asarray(
            getattr(objective, "grad")(np.asarray([float(x)], dtype=float), X_DUMMY),
            dtype=float,
        )[0]
    )


def _kinks(metadata: Mapping[str, object]) -> tuple[float, ...]:
    if metadata["form"] != "linear":
        return ()
    return (float(metadata["lower"]), float(metadata["upper"]))


def exact_landscape(
    objective: object,
    metadata: Mapping[str, object],
) -> list[StationaryPoint]:
    """Return every exact stationary point of an envelope objective."""
    return find_stationary_points(
        _value_fn(objective),
        _grad_fn(objective),
        domain=DOMAIN,
        kinks=_kinks(metadata),
    )


def population_landscape(
    metadata: Mapping[str, object],
    estimator: str,
    sigma: float,
    *,
    quadrature_order: int = 80,
    grid_size: int = 2401,
) -> list[StationaryPoint]:
    """Return stationary points of an estimator's population-smoothed objective."""
    population_grad_array, population_value_array = _population_functions(
        metadata,
        estimator,
        sigma,
        quadrature_order=quadrature_order,
    )
    population_grad = lambda x: float(population_grad_array(np.asarray(x)))
    population_value = lambda x: float(population_value_array(np.asarray(x)))
    return find_stationary_points(
        population_value,
        population_grad,
        domain=DOMAIN,
        grid_size=grid_size,
        vectorized_grad_fn=population_grad_array,
    )


def _population_functions(
    metadata: Mapping[str, object],
    estimator: str,
    sigma: float,
    *,
    quadrature_order: int,
):
    sigma_float = float(sigma)
    if not np.isfinite(sigma_float) or sigma_float <= 0.0:
        raise ValueError("sigma must be finite and positive.")
    if int(quadrature_order) < 2:
        raise ValueError("quadrature_order must be at least 2.")
    if estimator == "finite_difference":
        nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))

        def gradient(x):
            x_arr = np.asarray(x, dtype=float)
            return (
                _landscape_value(metadata, x_arr + sigma_float)
                - _landscape_value(metadata, x_arr - sigma_float)
            ) / (2.0 * sigma_float)

        def value(x):
            x_arr = np.asarray(x, dtype=float)
            samples = x_arr[..., None] + sigma_float * nodes
            return 0.5 * np.sum(
                _landscape_value(metadata, samples) * weights,
                axis=-1,
            )

        return gradient, value
    if estimator == "stein_difference":
        nodes, weights = np.polynomial.hermite.hermgauss(int(quadrature_order))
        normal_nodes = np.sqrt(2.0) * nodes
        normal_weights = weights / np.sqrt(np.pi)

        def gradient(x):
            x_arr = np.asarray(x, dtype=float)
            samples = x_arr[..., None] + sigma_float * normal_nodes
            return np.sum(
                _landscape_gradient(metadata, samples) * normal_weights,
                axis=-1,
            )

        def value(x):
            x_arr = np.asarray(x, dtype=float)
            samples = x_arr[..., None] + sigma_float * normal_nodes
            return np.sum(
                _landscape_value(metadata, samples) * normal_weights,
                axis=-1,
            )

        return gradient, value
    raise ValueError(f"Unsupported estimator {estimator!r}.")


def _landscape_value(
    metadata: Mapping[str, object],
    x: np.ndarray,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    clean = x_arr**2 + 0.5 * (np.sin(x_arr) - x_arr)
    form = str(metadata["form"])
    if form == "constant":
        return clean + float(metadata["amplitude"])
    distance = np.maximum.reduce(
        (
            float(metadata["lower"]) - x_arr,
            np.zeros_like(x_arr),
            x_arr - float(metadata["upper"]),
        )
    )
    if form == "linear":
        return clean + float(metadata["slope"]) * distance
    if form == "smooth_nonconvex":
        envelope = np.zeros_like(x_arr)
        outside = distance > 0.0
        if np.any(outside) and float(metadata["amplitude"]) > 0.0:
            z = float(metadata["transition_width"]) / distance[outside]
            envelope[outside] = float(metadata["amplitude"]) * np.exp(-(z**2))
        return clean + envelope
    raise ValueError(f"Unsupported envelope form {form!r}.")


def _landscape_gradient(
    metadata: Mapping[str, object],
    x: np.ndarray,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    clean_grad = 2.0 * x_arr + 0.5 * (np.cos(x_arr) - 1.0)
    form = str(metadata["form"])
    if form == "constant":
        return clean_grad
    lower = float(metadata["lower"])
    upper = float(metadata["upper"])
    direction = np.where(x_arr < lower, -1.0, np.where(x_arr > upper, 1.0, 0.0))
    if form == "linear":
        return clean_grad + float(metadata["slope"]) * direction
    if form == "smooth_nonconvex":
        distance = np.maximum.reduce(
            (lower - x_arr, np.zeros_like(x_arr), x_arr - upper)
        )
        envelope_grad = np.zeros_like(x_arr)
        outside = distance > 0.0
        if np.any(outside) and float(metadata["amplitude"]) > 0.0:
            z = float(metadata["transition_width"]) / distance[outside]
            active = z < 40.0
            slope = np.zeros_like(z)
            active_z = z[active]
            slope[active] = (
                2.0
                * float(metadata["amplitude"])
                / float(metadata["transition_width"])
                * active_z**3
                * np.exp(-(active_z**2))
            )
            envelope_grad[outside] = direction[outside] * slope
        return clean_grad + envelope_grad
    raise ValueError(f"Unsupported envelope form {form!r}.")


def collect_rows(
    manifest: ExperimentManifest,
    *,
    runs_root: str | Path | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collect final iterates and exact/population stationary-point tables."""
    run_rows: list[dict[str, object]] = []
    stationary_rows: list[dict[str, object]] = []
    missing: list[Path] = []
    exact_cache: dict[tuple[object, ...], list[StationaryPoint]] = {}
    population_cache: dict[
        tuple[object, ...], list[StationaryPoint]
    ] = {}
    emitted_exact: set[tuple[object, ...]] = set()
    emitted_population: set[tuple[object, ...]] = set()
    for variant in manifest.variants:
        config = get_config(manifest.base_preset, overrides=variant.overrides)
        metadata = _variant_metadata(variant, config)
        landscape_key = _landscape_key(metadata)
        if landscape_key not in exact_cache:
            exact_cache[landscape_key] = exact_landscape(config.objective, metadata)
        exact = exact_cache[landscape_key]
        exact_global = global_minimum(exact)
        exact_minima = [point for point in exact if point.kind == "minimum"]
        if landscape_key not in emitted_exact:
            for point in exact:
                stationary_rows.append(
                    {
                        **_public_metadata(metadata),
                        "landscape": "exact",
                        "estimator": "",
                        "x": point.x,
                        "kind": point.kind,
                        "value": point.value,
                        "is_global": point == exact_global,
                    }
                )
            emitted_exact.add(landscape_key)

        populations: dict[str, list[StationaryPoint]] = {}
        population_globals: dict[str, StationaryPoint] = {}
        for estimator in config.enabled_estimators:
            population_key = (*landscape_key, estimator, float(config.sigma))
            if population_key not in population_cache:
                population_cache[population_key] = population_landscape(
                    metadata, estimator, float(config.sigma)
                )
            population = population_cache[population_key]
            populations[estimator] = population
            population_global = global_minimum(population)
            population_globals[estimator] = population_global
            if population_key not in emitted_population:
                for point in population:
                    stationary_rows.append(
                        {
                            **_public_metadata(metadata),
                            "landscape": "population",
                            "estimator": estimator,
                            "x": point.x,
                            "kind": point.kind,
                            "value": point.value,
                            "is_global": point == population_global,
                        }
                    )
                emitted_population.add(population_key)

        for seed in manifest.seeds.run_seeds:
            path = manifest.variant_dir(variant, runs_root) / f"summary-seed-{seed}.json"
            if not path.exists():
                missing.append(path)
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for estimator, estimator_payload in payload.get("estimators", {}).items():
                if estimator not in populations:
                    continue
                x_k = float(estimator_payload["theta"][0])
                population_minima = [
                    point for point in populations[estimator] if point.kind == "minimum"
                ]
                assigned_exact = min(exact_minima, key=lambda point: abs(point.x - x_k))
                assigned_population = min(
                    population_minima, key=lambda point: abs(point.x - x_k)
                )
                population_global = population_globals[estimator]
                clean_value = _clean_value(x_k)
                run_rows.append(
                    {
                        **_public_metadata(metadata),
                        "run_seed": int(seed),
                        "estimator": estimator,
                        "x_k": x_k,
                        "x_star": 0.0,
                        "exact_global_x": exact_global.x,
                        "exact_global_value": exact_global.value,
                        "assigned_exact_x": assigned_exact.x,
                        "assigned_population_x": assigned_population.x,
                        "population_global_x": population_global.x,
                        "population_global_value": population_global.value,
                        "reached_exact_global_basin": assigned_exact == exact_global,
                        "reached_population_global_basin": (
                            assigned_population == population_global
                        ),
                        "distance_to_exact_global": abs(x_k - exact_global.x),
                        "distance_to_population_global": abs(
                            x_k - population_global.x
                        ),
                        "distance_to_assigned_population": abs(
                            x_k - assigned_population.x
                        ),
                        "distance_to_truth": abs(x_k),
                        "distance_to_support": _distance_to_interval(
                            x_k, float(metadata["lower"]), float(metadata["upper"])
                        ),
                        "clean_regret": clean_value - _clean_value(0.0),
                        "upper_regret": _value_fn(config.objective)(x_k)
                        - exact_global.value,
                        "summary_path": str(path),
                    }
                )
    if missing:
        preview = "\n".join(str(path) for path in missing[:8])
        raise FileNotFoundError(
            f"Missing {len(missing)} requested seed summaries; first paths:\n{preview}"
        )
    return run_rows, stationary_rows


def _landscape_key(metadata: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        metadata[key]
        for key in (
            "form",
            "amplitude",
            "slope",
            "lower",
            "upper",
            "transition_width",
        )
    )


def aggregate_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate final envelope outcomes across optimizer seeds."""
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["variant"]), str(row["estimator"]))].append(row)
    output: list[dict[str, object]] = []
    for group in groups.values():
        first = group[0]
        x_values = np.asarray([float(row["x_k"]) for row in group], dtype=float)
        n = x_values.size
        mean = float(np.mean(x_values))
        std = float(np.std(x_values, ddof=1)) if n > 1 else 0.0
        standard_error = std / np.sqrt(n) if n > 1 else 0.0
        critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else 0.0
        output.append(
            {
                **{
                    key: first[key]
                    for key in (
                        "variant",
                        "form",
                        "amplitude",
                        "slope",
                        "lower",
                        "upper",
                        "transition_width",
                        "theta0",
                        "sigma",
                        "estimator",
                        "exact_global_x",
                        "exact_global_value",
                        "population_global_x",
                        "population_global_value",
                    )
                },
                "n_seeds": n,
                "mean_x_k": mean,
                "std_x_k": std,
                "ci95_low": mean - critical * standard_error,
                "ci95_high": mean + critical * standard_error,
                "global_basin_rate": float(
                    np.mean([bool(row["reached_exact_global_basin"]) for row in group])
                ),
                "population_global_basin_rate": float(
                    np.mean(
                        [
                            bool(row["reached_population_global_basin"])
                            for row in group
                        ]
                    )
                ),
                "mean_distance_to_exact_global": float(
                    np.mean([float(row["distance_to_exact_global"]) for row in group])
                ),
                "mean_distance_to_population_global": float(
                    np.mean(
                        [float(row["distance_to_population_global"]) for row in group]
                    )
                ),
                "mean_distance_to_assigned_population": float(
                    np.mean(
                        [
                            float(row["distance_to_assigned_population"])
                            for row in group
                        ]
                    )
                ),
                "mean_distance_to_truth": float(
                    np.mean([float(row["distance_to_truth"]) for row in group])
                ),
                "mean_distance_to_support": float(
                    np.mean([float(row["distance_to_support"]) for row in group])
                ),
                "mean_clean_regret": float(
                    np.mean([float(row["clean_regret"]) for row in group])
                ),
                "mean_upper_regret": float(
                    np.mean([float(row["upper_regret"]) for row in group])
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            FORM_ORDER.index(str(row["form"])),
            float(row["amplitude"]),
            float(row["theta0"]),
            float(row["sigma"]),
            str(row["estimator"]),
        ),
    )


def _public_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        key: metadata[key]
        for key in (
            "variant",
            "form",
            "amplitude",
            "slope",
            "lower",
            "upper",
            "transition_width",
            "theta0",
            "sigma",
        )
    }


def _clean_value(x: float) -> float:
    return float(x * x + 0.5 * (np.sin(x) - x))


def _distance_to_interval(x: float, lower: float, upper: float) -> float:
    return float(max(lower - x, 0.0, x - upper))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _representative_variants(
    manifest: ExperimentManifest,
    amplitude: float = 0.42,
) -> dict[str, tuple[ManifestVariant, object, Mapping[str, object]]]:
    selected: dict[str, tuple[ManifestVariant, object, Mapping[str, object]]] = {}
    for variant in manifest.variants:
        config = get_config(manifest.base_preset, overrides=variant.overrides)
        metadata = _variant_metadata(variant, config)
        form = str(metadata["form"])
        if (
            form not in selected
            and np.isclose(float(metadata["amplitude"]), amplitude)
            and np.isclose(float(metadata["theta0"]), 0.0)
            and np.isclose(float(metadata["sigma"]), 0.15)
        ):
            selected[form] = (variant, config, metadata)
    if set(selected) != set(FORM_ORDER):
        missing = sorted(set(FORM_ORDER) - set(selected))
        raise ValueError(f"Manifest is missing representative envelope variants: {missing}.")
    return selected


def _plot_envelope_diagnostics(
    manifest: ExperimentManifest,
    output: Path,
) -> None:
    selected = _representative_variants(manifest)
    x = np.linspace(-0.5, 1.6, 1200)
    clean = np.asarray([_clean_value(value) for value in x])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for column, form in enumerate(FORM_ORDER):
        _, config, metadata = selected[form]
        regularizer = metadata["regularizer"]
        envelope = np.asarray(
            [
                regularizer.value(np.asarray([value], dtype=float))
                for value in x
            ]
        )
        upper = clean + envelope
        exact = exact_landscape(config.objective, metadata)
        exact_global = global_minimum(exact)

        axes[0, column].plot(x, envelope, color="tab:orange", linewidth=2)
        axes[0, column].axvspan(
            float(metadata["lower"]),
            float(metadata["upper"]),
            color="tab:green",
            alpha=0.12,
            label="covered interval",
        )
        axes[0, column].set_title(FORM_LABELS[form])
        axes[0, column].set_ylabel(r"Envelope $\phi(u)$")
        axes[0, column].grid(alpha=0.25)

        axes[1, column].plot(x, clean, color="black", linewidth=1.8, label="true $f(u)$")
        axes[1, column].plot(
            x, upper, color="tab:blue", linewidth=2, label=r"upper $f(u)+\phi(u)$"
        )
        axes[1, column].axvspan(
            float(metadata["lower"]),
            float(metadata["upper"]),
            color="tab:green",
            alpha=0.12,
        )
        axes[1, column].axvline(0.0, color="black", linestyle=":", linewidth=1)
        axes[1, column].axvline(
            exact_global.x, color="tab:red", linestyle="--", linewidth=1.3
        )
        axes[1, column].scatter(
            [exact_global.x],
            [exact_global.value],
            color="tab:red",
            marker="x",
            s=55,
            zorder=5,
            label="global upper minimum",
        )
        axes[1, column].set_xlabel("$u$")
        axes[1, column].set_ylabel("Objective value")
        axes[1, column].grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[1, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Envelope geometry at matched amplitude $A=0.42$")
    fig.tight_layout()
    fig.savefig(output / "envelope_diagnostics.png", dpi=180)
    plt.close(fig)


def _plot_nonconvex_landscapes(
    manifest: ExperimentManifest,
    output: Path,
) -> None:
    candidates: list[tuple[float, object, Mapping[str, object]]] = []
    seen: set[float] = set()
    for variant in manifest.variants:
        config = get_config(manifest.base_preset, overrides=variant.overrides)
        metadata = _variant_metadata(variant, config)
        amplitude = float(metadata["amplitude"])
        if (
            metadata["form"] == "smooth_nonconvex"
            and amplitude not in seen
            and np.isclose(float(metadata["theta0"]), 0.0)
            and np.isclose(float(metadata["sigma"]), 0.15)
        ):
            seen.add(amplitude)
            candidates.append((amplitude, config, metadata))
    candidates.sort(key=lambda item: item[0])
    x = np.linspace(-0.25, 1.1, 1000)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for axis, (amplitude, config, metadata) in zip(axes.flat, candidates, strict=True):
        value_fn = _value_fn(config.objective)
        y = np.asarray([value_fn(value) for value in x])
        points = exact_landscape(config.objective, metadata)
        axis.plot(x, y, color="tab:blue")
        for point in points:
            marker = "o" if point.kind == "minimum" else "x"
            color = "tab:red" if point.kind == "minimum" else "tab:purple"
            axis.scatter([point.x], [point.value], color=color, marker=marker, zorder=4)
        axis.axvspan(0.75, 1.25, color="tab:green", alpha=0.1)
        axis.set_title(f"$A={amplitude:g}$")
        axis.grid(alpha=0.25)
    fig.supxlabel("$u$")
    fig.supylabel(r"Upper objective $f(u)+\phi_{\mathrm{nc}}(u)$")
    fig.suptitle("Exact smooth-nonconvex landscapes")
    fig.tight_layout()
    fig.savefig(output / "nonconvex_landscapes.png", dpi=180)
    plt.close(fig)


def _plot_bifurcation(
    stationary_rows: Sequence[Mapping[str, object]],
    aggregates: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    exact_seen: set[tuple[float, float, str]] = set()
    for row in stationary_rows:
        if row["form"] != "smooth_nonconvex" or row["landscape"] != "exact":
            continue
        key = (float(row["amplitude"]), float(row["x"]), str(row["kind"]))
        if key in exact_seen:
            continue
        exact_seen.add(key)
        marker = "o" if row["kind"] == "minimum" else "x"
        color = "black" if bool(row["is_global"]) else "0.55"
        for axis in axes:
            axis.scatter(
                [float(row["amplitude"])],
                [float(row["x"])],
                marker=marker,
                color=color,
                s=32,
                zorder=3,
            )
    for axis, estimator in zip(
        axes, ("finite_difference", "stein_difference"), strict=True
    ):
        subset = [
            row
            for row in aggregates
            if row["form"] == "smooth_nonconvex" and row["estimator"] == estimator
        ]
        for (theta0, sigma), color in {
            (0.0, 0.05): "tab:blue",
            (0.0, 0.15): "tab:cyan",
            (0.0, 0.30): "tab:green",
            (1.0, 0.05): "tab:red",
            (1.0, 0.15): "tab:orange",
            (1.0, 0.30): "tab:pink",
        }.items():
            rows = sorted(
                (
                    row
                    for row in subset
                    if np.isclose(float(row["theta0"]), theta0)
                    and np.isclose(float(row["sigma"]), sigma)
                ),
                key=lambda row: float(row["amplitude"]),
            )
            if not rows:
                continue
            axis.plot(
                [float(row["amplitude"]) for row in rows],
                [float(row["mean_x_k"]) for row in rows],
                "o-",
                color=color,
                label=f"$u_0={theta0:g}$, $\\sigma={sigma:g}$",
                markersize=4,
            )
        axis.set_title(ESTIMATOR_LABELS[estimator])
        axis.set_xlabel("Envelope amplitude $A$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Stationary/final $u$")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Nonconvex envelope bifurcation and final convergence points")
    fig.tight_layout()
    fig.savefig(output / "nonconvex_bifurcation.png", dpi=180)
    plt.close(fig)


def _plot_basin_rates(
    aggregates: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, estimator in zip(
        axes, ("finite_difference", "stein_difference"), strict=True
    ):
        subset = [
            row
            for row in aggregates
            if row["form"] == "smooth_nonconvex" and row["estimator"] == estimator
        ]
        for theta0, linestyle in ((0.0, "-"), (1.0, "--")):
            for sigma, color in ((0.05, "tab:blue"), (0.15, "tab:orange"), (0.30, "tab:green")):
                rows = sorted(
                    (
                        row
                        for row in subset
                        if np.isclose(float(row["theta0"]), theta0)
                        and np.isclose(float(row["sigma"]), sigma)
                    ),
                    key=lambda row: float(row["amplitude"]),
                )
                axis.plot(
                    [float(row["amplitude"]) for row in rows],
                    [float(row["global_basin_rate"]) for row in rows],
                    marker="o",
                    linestyle=linestyle,
                    color=color,
                    label=f"$u_0={theta0:g}$, $\\sigma={sigma:g}$",
                )
        axis.set_title(ESTIMATOR_LABELS[estimator])
        axis.set_xlabel("Envelope amplitude $A$")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Fraction reaching exact global basin")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Initialization and smoothing control basin selection")
    fig.tight_layout()
    fig.savefig(output / "basin_success_rates.png", dpi=180)
    plt.close(fig)


def _plot_regret(
    aggregates: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, estimator in zip(
        axes, ("finite_difference", "stein_difference"), strict=True
    ):
        for form, color in zip(
            FORM_ORDER, ("black", "tab:blue", "tab:red"), strict=True
        ):
            rows = sorted(
                (
                    row
                    for row in aggregates
                    if row["form"] == form
                    and row["estimator"] == estimator
                    and np.isclose(float(row["theta0"]), 0.0)
                    and np.isclose(float(row["sigma"]), 0.15)
                ),
                key=lambda row: float(row["amplitude"]),
            )
            axis.plot(
                [float(row["amplitude"]) for row in rows],
                [float(row["mean_clean_regret"]) for row in rows],
                "o-",
                color=color,
                label=FORM_LABELS[form],
            )
        axis.set_title(ESTIMATOR_LABELS[estimator])
        axis.set_xlabel("Matched amplitude $A$")
        axis.set_yscale("symlog", linthresh=1e-8)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"True regret $f(u_K)-f(u^\star)$")
    axes[1].legend(fontsize=8)
    fig.suptitle("True-objective cost of conservative envelopes")
    fig.tight_layout()
    fig.savefig(output / "true_regret.png", dpi=180)
    plt.close(fig)


def _plot_population_target_error(
    aggregates: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, estimator in zip(
        axes, ("finite_difference", "stein_difference"), strict=True
    ):
        subset = [
            row
            for row in aggregates
            if row["form"] == "smooth_nonconvex" and row["estimator"] == estimator
        ]
        for (theta0, sigma), color in {
            (0.0, 0.05): "tab:blue",
            (0.0, 0.15): "tab:cyan",
            (0.0, 0.30): "tab:green",
            (1.0, 0.05): "tab:red",
            (1.0, 0.15): "tab:orange",
            (1.0, 0.30): "tab:pink",
        }.items():
            rows = sorted(
                (
                    row
                    for row in subset
                    if np.isclose(float(row["theta0"]), theta0)
                    and np.isclose(float(row["sigma"]), sigma)
                ),
                key=lambda row: float(row["amplitude"]),
            )
            axis.plot(
                [float(row["amplitude"]) for row in rows],
                [
                    float(row["mean_distance_to_assigned_population"])
                    for row in rows
                ],
                "o-",
                color=color,
                label=f"$u_0={theta0:g}$, $\\sigma={sigma:g}$",
                markersize=4,
            )
        axis.set_title(ESTIMATOR_LABELS[estimator])
        axis.set_xlabel("Envelope amplitude $A$")
        axis.set_yscale("symlog", linthresh=1e-5)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Distance to assigned population stationary point")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Finite-run convergence error after population basin selection")
    fig.tight_layout()
    fig.savefig(output / "population_target_error.png", dpi=180)
    plt.close(fig)


def _seed_jitter_map(
    rows: Sequence[Mapping[str, object]],
) -> dict[int, float]:
    seeds = sorted({int(row["run_seed"]) for row in rows})
    if not seeds:
        raise ValueError("Seed convergence plots require at least one run row.")
    offsets = np.linspace(-1.0, 1.0, len(seeds)) if len(seeds) > 1 else np.zeros(1)
    return dict(zip(seeds, offsets, strict=True))


def _reference_series(
    rows: Sequence[Mapping[str, object]],
    key: str,
) -> tuple[list[float], list[float]]:
    values: dict[float, set[float]] = defaultdict(set)
    for row in rows:
        values[float(row["amplitude"])].add(float(row[key]))
    inconsistent = {
        amplitude: sorted(points)
        for amplitude, points in values.items()
        if len(points) != 1
    }
    if inconsistent:
        raise ValueError(f"Reference {key!r} is inconsistent by amplitude: {inconsistent}.")
    amplitudes = sorted(values)
    return amplitudes, [next(iter(values[amplitude])) for amplitude in amplitudes]


def _plot_seed_panel(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    seed_offsets: Mapping[int, float],
) -> None:
    if not rows:
        raise ValueError("Seed convergence panel received no run rows.")
    amplitudes = sorted({float(row["amplitude"]) for row in rows})
    gaps = np.diff(amplitudes)
    spacing = float(np.min(gaps)) if gaps.size else 0.25
    init_offset = 0.08 * spacing
    jitter_scale = 0.045 * spacing

    true_values = {float(row["x_star"]) for row in rows}
    if len(true_values) != 1:
        raise ValueError(f"True minima are inconsistent within a panel: {true_values}.")
    true_minimum = next(iter(true_values))
    axis.axhline(
        true_minimum,
        color="black",
        linestyle="--",
        linewidth=1.4,
        zorder=1,
    )

    exact_amplitudes, exact_points = _reference_series(rows, "exact_global_x")
    axis.plot(
        exact_amplitudes,
        exact_points,
        color="tab:red",
        linestyle="--",
        marker="x",
        linewidth=1.3,
        markersize=6,
        zorder=3,
    )
    population_amplitudes, population_points = _reference_series(
        rows, "population_global_x"
    )
    axis.plot(
        population_amplitudes,
        population_points,
        color="tab:purple",
        linestyle=":",
        marker="D",
        markerfacecolor="none",
        linewidth=1.4,
        markersize=5,
        zorder=3,
    )

    for theta0, (color, marker) in INIT_STYLES.items():
        init_rows = [
            row for row in rows if np.isclose(float(row["theta0"]), theta0)
        ]
        if not init_rows:
            continue
        mean_x: list[float] = []
        means: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []
        raw_x_all: list[float] = []
        raw_finals_all: list[float] = []
        direction = -1.0 if theta0 == 0.0 else 1.0
        for amplitude in amplitudes:
            condition_rows = sorted(
                (
                    row
                    for row in init_rows
                    if np.isclose(float(row["amplitude"]), amplitude)
                ),
                key=lambda row: int(row["run_seed"]),
            )
            if not condition_rows:
                continue
            finals = np.asarray(
                [float(row["x_k"]) for row in condition_rows], dtype=float
            )
            center = amplitude + direction * init_offset
            raw_x_all.extend(
                center + jitter_scale * seed_offsets[int(row["run_seed"])]
                for row in condition_rows
            )
            raw_finals_all.extend(float(value) for value in finals)
            mean = float(np.mean(finals))
            mean_x.append(center)
            means.append(mean)
            lower_errors.append(mean - float(np.min(finals)))
            upper_errors.append(float(np.max(finals)) - mean)
        axis.scatter(
            raw_x_all,
            raw_finals_all,
            color=color,
            marker=marker,
            s=23,
            alpha=0.50,
            linewidths=0.4,
            zorder=4,
        )
        axis.errorbar(
            mean_x,
            means,
            yerr=np.asarray([lower_errors, upper_errors]),
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.5,
            markersize=6,
            capsize=3,
            zorder=5,
        )

    axis.set_xlabel("Matched envelope amplitude $A$")
    axis.grid(alpha=0.25)


def _seed_plot_legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle="-",
            label=f"Final $u_K$, $u_0={theta0:g}$",
        )
        for theta0, (color, marker) in INIT_STYLES.items()
    ] + [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label=r"True minimum $u^\star$",
        ),
        Line2D(
            [0],
            [0],
            color="tab:red",
            marker="x",
            linestyle="--",
            label="Exact upper global minimum",
        ),
        Line2D(
            [0],
            [0],
            color="tab:purple",
            marker="D",
            markerfacecolor="none",
            linestyle=":",
            label="Population global minimum",
        ),
    ]


def _seed_plot_caption(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.012,
        "Small jittered marks are individual optimizer seeds; large marks and "
        "whiskers are mean and min–max. The CSV's assigned population stationary "
        "point is the closest population local minimum to each final mark; the "
        "purple diamonds show the population global minimum.",
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
    )


def _plot_seed_convergence(
    run_rows: Sequence[Mapping[str, object]],
    output: Path,
    *,
    dpi: int = 180,
) -> None:
    seed_offsets = _seed_jitter_map(run_rows)
    legend = _seed_plot_legend()
    for form in ("constant", "linear"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
        for axis, estimator in zip(
            axes, ("finite_difference", "stein_difference"), strict=True
        ):
            panel_rows = [
                row
                for row in run_rows
                if row["form"] == form and row["estimator"] == estimator
            ]
            _plot_seed_panel(axis, panel_rows, seed_offsets)
            axis.set_title(ESTIMATOR_LABELS[estimator])
        axes[0].set_ylabel("Final parameter $u_K$")
        axes[1].legend(handles=legend, fontsize=7, loc="best")
        fig.suptitle(f"Seed-level convergence: {FORM_LABELS[form]} envelope")
        _seed_plot_caption(fig)
        fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.96))
        fig.savefig(output / f"seed_convergence_{form}.png", dpi=dpi)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    estimators = ("finite_difference", "stein_difference")
    sigmas = (0.05, 0.15, 0.30)
    for row_index, estimator in enumerate(estimators):
        for column, sigma in enumerate(sigmas):
            axis = axes[row_index, column]
            panel_rows = [
                row
                for row in run_rows
                if row["form"] == "smooth_nonconvex"
                and row["estimator"] == estimator
                and np.isclose(float(row["sigma"]), sigma)
            ]
            _plot_seed_panel(axis, panel_rows, seed_offsets)
            axis.set_title(f"{ESTIMATOR_LABELS[estimator]}, $\\sigma={sigma:g}$")
    for axis in axes[:, 0]:
        axis.set_ylabel("Final parameter $u_K$")
    axes[0, 2].legend(handles=legend, fontsize=7, loc="best")
    fig.suptitle("Seed-level convergence: smooth nonconvex envelope")
    _seed_plot_caption(fig)
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.96))
    fig.savefig(output / "seed_convergence_smooth_nonconvex.png", dpi=dpi)
    plt.close(fig)


def write_outputs(
    manifest: ExperimentManifest,
    run_rows: Sequence[Mapping[str, object]],
    stationary_rows: Sequence[Mapping[str, object]],
    aggregates: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    """Write all envelope analysis tables, plots, and summary."""
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_run_metrics.csv", run_rows)
    _write_csv(output / "stationary_points.csv", stationary_rows)
    _write_csv(output / "aggregate_metrics.csv", aggregates)
    _plot_envelope_diagnostics(manifest, output)
    _plot_nonconvex_landscapes(manifest, output)
    _plot_bifurcation(stationary_rows, aggregates, output)
    _plot_basin_rates(aggregates, output)
    _plot_regret(aggregates, output)
    _plot_population_target_error(aggregates, output)
    _plot_seed_convergence(run_rows, output)

    nonconvex = [row for row in aggregates if row["form"] == "smooth_nonconvex"]
    exact_success = float(
        np.mean([float(row["global_basin_rate"]) for row in nonconvex])
    )
    population_success = float(
        np.mean([float(row["population_global_basin_rate"]) for row in nonconvex])
    )
    summary = "\n".join(
        [
            "# Zeroth-Order Support Envelope Analysis",
            "",
            f"- Completed final-iterate rows: {len(run_rows)}",
            f"- Aggregate conditions: {len(aggregates)}",
            f"- Classified stationary points: {len(stationary_rows)}",
            f"- Mean nonconvex exact-global-basin rate: {exact_success:.3f}",
            "- Mean nonconvex population-global-basin rate: "
            f"{population_success:.3f}",
            "",
            "The constant envelope is the trajectory-invariance control. "
            "The linear envelope tests the coverage-boundary threshold. "
            "The smooth nonconvex envelope tests basin selection and whether "
            "zeroth-order smoothing removes competing stationary points.",
            "Seeds 101–108 vary only the optimizer RNG; finite difference is "
            "therefore coincident across seeds, while Stein uses different "
            "Gaussian perturbation streams. Initializations are separate "
            "conditions. The true target (`x_star`) is the clean-objective "
            "minimum; `exact_global_x` is the exact upper-objective global "
            "minimum; `population_global_x` is the estimator-smoothed global "
            "minimum; and `assigned_population_x` is the closest local "
            "population minimum to the final iterate.",
            "",
        ]
    )
    (output / "validation_summary.md").write_text(summary, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = load_experiment_manifest(args.manifest)
    root = results_root() if args.runs_root is None else Path(args.runs_root)
    output = (
        Path(args.output_dir)
        if args.output_dir
        else root / "zeroth-order-envelope-analysis"
    )
    run_rows, stationary_rows = collect_rows(manifest, runs_root=root)
    aggregates = aggregate_rows(run_rows)
    write_outputs(manifest, run_rows, stationary_rows, aggregates, output)
    print(f"Wrote envelope analysis to {output}")


if __name__ == "__main__":
    main()
