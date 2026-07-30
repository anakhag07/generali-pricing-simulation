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


def test_envelope_diagnostic_writes_true_and_upper_objective_plot(
    tmp_path: Path,
) -> None:
    manifest = load_experiment_manifest(MANIFEST_PATH)

    script._plot_envelope_diagnostics(manifest, tmp_path)
    image = tmp_path / "envelope_diagnostics.png"

    assert image.exists()
    assert image.stat().st_size > 10_000
