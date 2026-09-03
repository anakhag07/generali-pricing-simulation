from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.policy_lcb import finite_grid
from experiments.policy_lcb.finite_grid import (
    ClippedDistanceRampSpec,
    calibration_quantile,
    clipped_distance_uncertainty,
    collect_variable_finite_grid_lcb_outputs,
    evaluate_variable_finite_grid_lcb_draw,
    evaluate_variable_finite_grid_lcb_seed,
    parse_variable_finite_grid_lcb_manifest,
    run_variable_finite_grid_lcb_manifest_seed,
    variable_finite_grid_lcb_seed_complete,
)


def _payload() -> dict[str, object]:
    return {
        "kind": "finite_grid_variable_lcb",
        "name": "variable-grid-test",
        "grid": {"type": "linspace", "lower": 0.0, "upper": 1.0, "count": 5},
        "true_value": {"type": "concave_quadratic", "linear": 5.0, "quadratic": 5.0},
        "uncertainty": {
            "type": "clipped_distance_ramp",
            "centers": [0.0, 0.5, 1.0],
            "minimum": 0.1,
            "maximum": 1.0,
            "ramp_radius": 0.5,
        },
        "surrogate": {"type": "independent_gaussian", "noise_scales": [0.0, 0.5]},
        "confidence": {
            "delta": 0.05,
            "calibrations": [
                {"name": "simultaneous", "type": "bonferroni_two_sided"},
                {"name": "pointwise", "type": "pointwise_two_sided"},
            ],
        },
        "seeds": {
            "master_noise_seed": 20260814,
            "run_seeds": {"type": "range", "start": 101, "count": 2},
        },
        "launch": {"mode": "local", "array": "seed"},
        "per_seed_plots": False,
    }


def _manifest():
    return parse_variable_finite_grid_lcb_manifest(_payload())


def test_uncertainty_profile_clips_and_moves_with_center() -> None:
    uncertainty = ClippedDistanceRampSpec(
        centers=(0.0, 0.5), minimum=0.1, maximum=1.0, ramp_radius=0.5
    )
    x = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

    centered_left = clipped_distance_uncertainty(x, 0.0, uncertainty)
    centered_middle = clipped_distance_uncertainty(x, 0.5, uncertainty)

    assert centered_left == pytest.approx([0.1, 0.55, 1.0, 1.0, 1.0])
    assert centered_middle == pytest.approx([1.0, 0.55, 0.1, 0.55, 1.0])


def test_one_noise_vector_is_paired_across_the_full_condition_cube() -> None:
    spec = _manifest().spec
    z = np.asarray([-0.5, -0.25, 0.0, 0.25, 0.5])
    result = evaluate_variable_finite_grid_lcb_draw(
        spec, run_seed=101, noise_seed=17, z=z
    )

    assert result.z == tuple(z)
    # The seed result stores one vector, not a draw per center, scale, or calibration.
    assert len(result.conditions) == 3 * 2 * 2
    assert len(result.selectors) == 3 * 2 * 2 * 3
    first = evaluate_variable_finite_grid_lcb_seed(spec, 101)
    replay = evaluate_variable_finite_grid_lcb_seed(spec, 101)
    second = evaluate_variable_finite_grid_lcb_seed(spec, 102)
    assert first == replay
    assert first.z != second.z


def test_coverage_is_invariant_across_positive_scales_and_centers() -> None:
    payload = _payload()
    payload["surrogate"] = {
        "type": "independent_gaussian",
        "noise_scales": [0.25, 0.5, 2.0],
    }
    spec = parse_variable_finite_grid_lcb_manifest(payload).spec
    bonferroni = next(
        item for item in spec.calibrations if item.kind == "bonferroni_two_sided"
    )
    q = calibration_quantile(bonferroni, spec.delta, spec.grid.count)
    result = evaluate_variable_finite_grid_lcb_draw(
        spec,
        run_seed=101,
        noise_seed=17,
        z=[0.0, q + 0.01, -q, 0.5, -0.5],
    )

    rows = [row for row in result.conditions if row.calibration == "simultaneous"]
    assert len({row.simultaneous_coverage for row in rows}) == 1
    assert len({row.fraction_covered for row in rows}) == 1
    assert not rows[0].simultaneous_coverage
    assert rows[0].fraction_covered == pytest.approx(0.8)


def test_uniform_lcb_and_nominal_select_exactly_the_same_grid_point() -> None:
    result = evaluate_variable_finite_grid_lcb_draw(
        _manifest().spec,
        run_seed=101,
        noise_seed=17,
        z=[3.0, -1.0, 0.0, 1.0, -3.0],
    )
    grouped: dict[tuple[float, float, str], dict[str, int]] = {}
    for row in result.selectors:
        key = (row.noise_scale, row.uncertainty_center, row.calibration)
        grouped.setdefault(key, {})[row.selector] = row.selected_index
    assert all(rows["nominal"] == rows["uniform_lcb"] for rows in grouped.values())


def test_deterministic_target_is_exact_argmax_of_true_value_minus_envelope() -> None:
    spec = _manifest().spec
    result = evaluate_variable_finite_grid_lcb_draw(
        spec, run_seed=101, noise_seed=17, z=np.zeros(spec.grid.count)
    )
    row = next(
        item
        for item in result.conditions
        if item.noise_scale == 0.5
        and item.uncertainty_center == 0.0
        and item.calibration == "simultaneous"
    )
    calibration = next(item for item in spec.calibrations if item.name == "simultaneous")
    grid = spec.grid.values()
    values = 5.0 * grid - 5.0 * grid**2
    sigma = clipped_distance_uncertainty(grid, 0.0, spec.uncertainty)
    expected = int(
        np.argmax(values - 0.5 * calibration_quantile(calibration, spec.delta, len(grid)) * sigma)
    )
    assert row.deterministic_target_index == expected


def test_certificate_holds_conditionally_on_simultaneous_coverage() -> None:
    spec = _manifest().spec
    result = evaluate_variable_finite_grid_lcb_draw(
        spec, run_seed=101, noise_seed=17, z=np.zeros(spec.grid.count)
    )

    assert all(row.simultaneous_coverage for row in result.conditions)
    assert all(row.certificate_slack >= -1e-12 for row in result.conditions)
    assert all(row.certificate_event_holds for row in result.conditions)


def test_deterministic_landscapes_are_mirror_symmetric() -> None:
    result = evaluate_variable_finite_grid_lcb_draw(
        _manifest().spec, run_seed=101, noise_seed=17, z=np.zeros(5)
    )
    rows = {
        (row.noise_scale, row.uncertainty_center, row.calibration): row
        for row in result.conditions
    }
    for noise_scale in (0.0, 0.5):
        for calibration in ("simultaneous", "pointwise"):
            left = rows[(noise_scale, 0.0, calibration)]
            right = rows[(noise_scale, 1.0, calibration)]
            assert left.deterministic_target_x == pytest.approx(
                1.0 - right.deterministic_target_x
            )


def test_zero_noise_has_perfect_coverage_zero_envelopes_and_optimal_selection() -> None:
    result = evaluate_variable_finite_grid_lcb_draw(
        _manifest().spec,
        run_seed=101,
        noise_seed=17,
        z=[20.0, -10.0, 4.0, 8.0, -30.0],
    )
    conditions = [row for row in result.conditions if row.noise_scale == 0.0]
    selectors = [row for row in result.selectors if row.noise_scale == 0.0]

    assert all(row.simultaneous_coverage for row in conditions)
    assert all(row.maximum_half_width == 0.0 for row in conditions)
    assert all(row.optimum_x == 0.5 for row in conditions)
    assert all(row.selected_x == 0.5 and row.regret == 0.0 for row in selectors)


def test_manifest_validation_resolves_seed_range_and_rejects_invalid_inputs() -> None:
    manifest = _manifest()
    assert manifest.spec.run_seeds == (101, 102)
    assert manifest.spec.grid.values() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    payload = copy.deepcopy(_payload())
    payload["uncertainty"]["minimum"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="positive"):
        parse_variable_finite_grid_lcb_manifest(payload)

    payload = copy.deepcopy(_payload())
    payload["confidence"]["calibrations"] = [  # type: ignore[index]
        {"name": "pointwise", "type": "pointwise_two_sided"}
    ]
    with pytest.raises(ValueError, match="Bonferroni and pointwise"):
        parse_variable_finite_grid_lcb_manifest(payload)


def test_committed_manifest_matches_the_characterization_cube() -> None:
    path = Path(__file__).parents[2] / "manifests" / "variable_lcb_envelope_characterization.json"
    with path.open(encoding="utf-8") as handle:
        manifest = parse_variable_finite_grid_lcb_manifest(json.load(handle), source_path=path)

    assert manifest.spec.grid.count == 101
    assert manifest.spec.uncertainty.centers == (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    assert manifest.spec.noise_scales == (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    assert manifest.spec.run_seeds[0] == 101
    assert manifest.spec.run_seeds[-1] == 2100
    assert len(manifest.spec.run_seeds) == 2000


def test_seed_persistence_resumability_aggregation_and_plot_generation(
    tmp_path: Path,
) -> None:
    manifest = _manifest()

    first = run_variable_finite_grid_lcb_manifest_seed(
        manifest, 0, runs_root=tmp_path
    )
    skipped = run_variable_finite_grid_lcb_manifest_seed(
        manifest, 0, runs_root=tmp_path
    )
    run_variable_finite_grid_lcb_manifest_seed(manifest, 1, runs_root=tmp_path)
    collected = collect_variable_finite_grid_lcb_outputs(manifest, runs_root=tmp_path)

    assert first["skipped"] is False
    assert skipped["skipped"] is True
    assert variable_finite_grid_lcb_seed_complete(manifest, 101, runs_root=tmp_path)
    assert collected["n_condition_rows"] == 2 * 3 * 2 * 2
    assert collected["n_selector_rows"] == 2 * 3 * 2 * 2 * 3
    project_dir = manifest.project_dir(tmp_path)
    for filename in (
        "EXPERIMENT.md",
        "seed_condition_metrics.csv",
        "seed_selector_metrics.csv",
        "experiment_1_noise_scale_summary.csv",
        "experiment_2_calibration_summary.csv",
        "experiment_3_envelope_shape_summary.csv",
        "experiment_4_center_summary.csv",
    ):
        assert (project_dir / filename).exists()
    for filename in (
        "experiment_1_noise_scale_regret.png",
        "experiment_1_validity.png",
        "experiment_1_realized_landscapes.png",
        "experiment_2_calibration_coverage.png",
        "experiment_2_calibration_regret.png",
        "experiment_2_calibration_regret_by_center.png",
        "experiment_3_envelope_shape_regret.png",
        "experiment_3_envelope_shape_regret_by_center.png",
        "experiment_4_center_regret.png",
        "experiment_4_center_selection.png",
    ):
        assert (project_dir / "plots" / filename).stat().st_size > 0
    with (project_dir / "experiment_4_center_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 2
    assert {float(row["uncertainty_center"]) for row in rows} == {0.0, 0.5, 1.0}


def test_calibration_regret_by_center_uses_calibration_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = []
    for calibration, calibration_type, offset in (
        ("familywise", "bonferroni_two_sided", 20.0),
        ("marginal", "pointwise_two_sided", 10.0),
    ):
        for noise_scale in (0.25, 0.5):
            rows.append(
                {
                    "uncertainty_center": 0.5,
                    "calibration": calibration,
                    "calibration_type": calibration_type,
                    "noise_scale": noise_scale,
                    "nominal_regret_mean": noise_scale,
                    "nominal_regret_q05": noise_scale - 0.1,
                    "nominal_regret_q95": noise_scale + 0.1,
                    "variable_lcb_regret_mean": offset + noise_scale,
                    "variable_lcb_regret_q05": offset + noise_scale - 0.1,
                    "variable_lcb_regret_q95": offset + noise_scale + 0.1,
                }
            )

    captured: dict[str, object] = {}

    def capture_facets(fig, facets, *args, **kwargs) -> None:
        captured["fig"] = fig
        captured["facets"] = facets

    monkeypatch.setattr(finite_grid, "_finish_facets", capture_facets)
    finite_grid._plot_calibration_regret_by_center(
        _manifest().spec,
        rows,
        tmp_path / "calibration-regret-by-center.pdf",
    )

    facets = captured["facets"]
    axis = facets[0][0]
    lines = {line.get_label(): line for line in axis.lines}
    assert set(lines) == {
        "Nominal (no envelope)",
        "Pointwise LCB",
        "Simultaneous LCB",
    }
    assert lines["Pointwise LCB"].get_ydata() == pytest.approx([10.25, 10.5])
    assert lines["Simultaneous LCB"].get_ydata() == pytest.approx([20.25, 20.5])
    finite_grid._load_pyplot().close(captured["fig"])
