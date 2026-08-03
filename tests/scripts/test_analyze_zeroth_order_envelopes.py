from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.configs import get_config
from experiments.manifest import load_experiment_manifest
from scripts import analyze_zeroth_order_envelopes as script
from scripts.zeroth_order_landscape import (
    finite_difference_population_gradient,
    find_stationary_points,
    global_minimum,
    stein_population_gradient,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "manifests" / "zeroth_order_envelopes.json"


def test_population_gradient_helpers_match_quadratic_identity() -> None:
    value = lambda x: (x - 0.3) ** 2
    grad = lambda x: 2.0 * (x - 0.3)

    for x in (-0.4, 0.0, 0.8):
        assert finite_difference_population_gradient(value, x, 0.2) == pytest.approx(
            grad(x)
        )
        assert stein_population_gradient(grad, x, 0.2) == pytest.approx(grad(x))


def test_stationary_finder_classifies_smooth_and_kinked_minima() -> None:
    smooth = find_stationary_points(
        lambda x: (x - 0.2) ** 2,
        lambda x: 2.0 * (x - 0.2),
        domain=(-1.0, 1.0),
    )
    kinked = find_stationary_points(
        lambda x: abs(x - 0.4),
        lambda x: -1.0 if x < 0.4 else (1.0 if x > 0.4 else 0.0),
        domain=(-1.0, 1.0),
        kinks=(0.4,),
    )

    assert len(smooth) == 1
    assert smooth[0].x == pytest.approx(0.2)
    assert smooth[0].kind == "minimum"
    assert len(kinked) == 1
    assert kinked[0].x == pytest.approx(0.4)
    assert kinked[0].kind == "minimum"


def test_exact_nonconvex_sweep_crosses_bifurcation_and_global_switch() -> None:
    manifest = load_experiment_manifest(MANIFEST_PATH)
    representatives = [
        variant
        for variant in manifest.variants
        if variant.axes["form"] == "smooth_nonconvex"
        and variant.axes["theta0"] == 0.0
        and variant.axes["sigma"] == 0.15
    ]

    root_counts = []
    global_locations = {}
    for variant in representatives:
        config = get_config(manifest.base_preset, overrides=variant.overrides)
        metadata = script._variant_metadata(variant, config)
        points = script.exact_landscape(config.objective, metadata)
        root_counts.append(len(points))
        global_locations[float(metadata["amplitude"])] = global_minimum(points).x

    assert root_counts == [1, 1, 3, 3, 3, 1]
    assert global_locations[0.35] < 0.1
    assert 0.55 < global_locations[0.42] < 0.75


def test_vectorized_landscape_matches_objective_values_and_gradients() -> None:
    manifest = load_experiment_manifest(MANIFEST_PATH)
    x = np.asarray([-0.4, 0.0, 0.74, 0.75, 0.9, 1.25, 1.4])
    for form in ("constant", "linear", "smooth_nonconvex"):
        variant = next(
            variant
            for variant in manifest.variants
            if variant.axes["form"] == form
            and variant.axes["amplitude"] == 0.42
            and variant.axes["theta0"] == 0.0
            and variant.axes["sigma"] == 0.15
        )
        config = get_config(manifest.base_preset, overrides=variant.overrides)
        metadata = script._variant_metadata(variant, config)
        expected_value = np.asarray(
            [config.objective.value(np.asarray([value]), script.X_DUMMY) for value in x]
        )
        expected_grad = np.asarray(
            [config.objective.grad(np.asarray([value]), script.X_DUMMY)[0] for value in x]
        )

        np.testing.assert_allclose(script._landscape_value(metadata, x), expected_value)
        np.testing.assert_allclose(
            script._landscape_gradient(metadata, x), expected_grad
        )


def test_linear_matched_slope_reaches_coverage_boundary_above_threshold() -> None:
    manifest = load_experiment_manifest(MANIFEST_PATH)
    locations = {}
    for variant in manifest.variants:
        if (
            variant.axes["form"] == "linear"
            and variant.axes["theta0"] == 0.0
            and variant.axes["amplitude"] in (0.35, 0.42)
        ):
            config = get_config(manifest.base_preset, overrides=variant.overrides)
            metadata = script._variant_metadata(variant, config)
            locations[float(metadata["amplitude"])] = global_minimum(
                script.exact_landscape(config.objective, metadata)
            ).x

    assert 0.0 < locations[0.35] < 0.75
    assert locations[0.42] == pytest.approx(0.75)


def test_aggregate_tracks_exact_population_and_finite_run_distances(
    tmp_path: Path,
) -> None:
    rows = []
    for seed, x_k, population_success in (
        (101, 0.48, True),
        (102, 0.52, False),
    ):
        rows.append(
            {
                "variant": "test",
                "form": "smooth_nonconvex",
                "amplitude": 0.42,
                "slope": 1.0,
                "lower": 0.75,
                "upper": 1.25,
                "transition_width": 0.25,
                "theta0": 0.0,
                "sigma": 0.15,
                "run_seed": seed,
                "estimator": "finite_difference",
                "x_k": x_k,
                "exact_global_x": 0.59,
                "exact_global_value": 0.36,
                "population_global_x": 0.50,
                "population_global_value": 0.37,
                "reached_exact_global_basin": True,
                "reached_population_global_basin": population_success,
                "distance_to_exact_global": abs(x_k - 0.59),
                "distance_to_population_global": abs(x_k - 0.50),
                "distance_to_assigned_population": 0.01,
                "distance_to_truth": abs(x_k),
                "distance_to_support": 0.75 - x_k,
                "clean_regret": x_k**2,
                "upper_regret": 0.02,
            }
        )

    aggregate = script.aggregate_rows(rows)[0]

    assert aggregate["population_global_basin_rate"] == pytest.approx(0.5)
    assert aggregate["mean_distance_to_population_global"] == pytest.approx(0.02)
    assert aggregate["mean_distance_to_assigned_population"] == pytest.approx(0.01)
    script._plot_population_target_error([aggregate], tmp_path)
    assert (tmp_path / "population_target_error.png").exists()


def test_envelope_diagnostic_writes_true_and_upper_objective_plot(
    tmp_path: Path,
) -> None:
    manifest = load_experiment_manifest(MANIFEST_PATH)

    script._plot_envelope_diagnostics(manifest, tmp_path)
    image = tmp_path / "envelope_diagnostics.png"

    assert image.exists()
    assert image.stat().st_size > 10_000


def test_seed_convergence_plots_show_all_saved_conditions(tmp_path: Path) -> None:
    rows = _synthetic_seed_rows()

    script._plot_seed_convergence(rows, tmp_path, dpi=72)

    expected = {
        "seed_convergence_constant.png",
        "seed_convergence_linear.png",
        "seed_convergence_smooth_nonconvex.png",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert all((tmp_path / name).stat().st_size > 10_000 for name in expected)
    assert len(rows) == 864
    assert {int(row["run_seed"]) for row in rows} == set(range(101, 109))
    assert {float(row["theta0"]) for row in rows} == {0.0, 1.0}

    jitter = script._seed_jitter_map(rows)
    assert list(jitter) == list(range(101, 109))
    assert jitter[101] == pytest.approx(-1.0)
    assert jitter[108] == pytest.approx(1.0)

    panel = [
        row
        for row in rows
        if row["form"] == "linear"
        and row["estimator"] == "finite_difference"
    ]
    amplitudes, exact = script._reference_series(panel, "exact_global_x")
    _, population = script._reference_series(panel, "population_global_x")
    assert amplitudes == [0.0, 0.25, 0.35, 0.42, 0.6, 0.7]
    assert exact == pytest.approx([min(0.75, amplitude) for amplitude in amplitudes])
    assert population == pytest.approx(
        [min(0.70, amplitude + 0.01) for amplitude in amplitudes]
    )

    finite_difference_groups = {}
    stein_groups = {}
    for row in rows:
        key = (
            row["form"],
            row["amplitude"],
            row["theta0"],
            row["sigma"],
        )
        target = (
            finite_difference_groups
            if row["estimator"] == "finite_difference"
            else stein_groups
        )
        target.setdefault(key, set()).add(float(row["x_k"]))
    assert all(len(values) == 1 for values in finite_difference_groups.values())
    assert all(len(values) == 8 for values in stein_groups.values())


def _synthetic_seed_rows() -> list[dict[str, object]]:
    forms = {
        "constant": ([0.0, 0.42, 0.7], [0.15]),
        "linear": ([0.0, 0.25, 0.35, 0.42, 0.6, 0.7], [0.15]),
        "smooth_nonconvex": (
            [0.0, 0.25, 0.35, 0.42, 0.6, 0.7],
            [0.05, 0.15, 0.30],
        ),
    }
    rows: list[dict[str, object]] = []
    for form, (amplitudes, sigmas) in forms.items():
        for amplitude in amplitudes:
            exact_global = 0.0 if form == "constant" else min(0.75, amplitude)
            for sigma in sigmas:
                for estimator in ("finite_difference", "stein_difference"):
                    population_global = (
                        exact_global
                        if form == "constant"
                        else min(
                            0.70,
                            amplitude
                            + (0.01 if estimator == "finite_difference" else 0.02),
                        )
                    )
                    for theta0 in (0.0, 1.0):
                        for seed in range(101, 109):
                            seed_shift = (
                                0.0
                                if estimator == "finite_difference"
                                else (seed - 104.5) * 0.001
                            )
                            rows.append(
                                {
                                    "form": form,
                                    "amplitude": amplitude,
                                    "theta0": theta0,
                                    "sigma": sigma,
                                    "run_seed": seed,
                                    "estimator": estimator,
                                    "x_k": population_global
                                    + 0.01 * theta0
                                    + seed_shift,
                                    "x_star": 0.0,
                                    "exact_global_x": exact_global,
                                    "population_global_x": population_global,
                                }
                            )
    return rows
