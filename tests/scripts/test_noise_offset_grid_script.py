from __future__ import annotations

import numpy as np

from scripts import run_noise_offset_grid as script


def test_grid_offsets_and_seeds_reuse_saved_sweeps() -> None:
    # The reused fixed-noise curves come from the saved theta-offset sweeps, so
    # the grid axes must stay subsets of the original sweep grids and the
    # reused noise levels must match what those sweeps ran at.
    assert set(script.GRID_THETA_OFFSETS) <= {float(o) for o in script.REUSED_SWEEP_THETA_OFFSETS}
    assert set(script.RUN_SEEDS) <= set(script.REUSED_SWEEP_RUN_SEEDS)
    assert script.HOMO_FAMILY.reused_noise_level == 0.5
    assert script.HETERO_FAMILY.reused_noise_level == 1.0


def test_grid_override_lists_cover_noise_by_offset_product() -> None:
    for family in (script.HOMO_FAMILY, script.HETERO_FAMILY):
        override_list = script._build_grid_override_list(family)
        assert len(override_list) == len(family.new_noise_levels) * len(script.GRID_THETA_OFFSETS)
        expected_names = [
            script._grid_run_name(family, level, offset)
            for level in family.new_noise_levels
            for offset in script.GRID_THETA_OFFSETS
        ]
        assert [entry["_run_name"] for entry in override_list] == expected_names
        for entry, (level, offset) in zip(
            override_list,
            [(l, o) for l in family.new_noise_levels for o in script.GRID_THETA_OFFSETS],
        ):
            np.testing.assert_allclose(entry["theta0"], script.BASE_THETA + float(offset))
            noise = entry["objective"].noise
            if family is script.HOMO_FAMILY:
                assert noise.std == float(level)
            else:
                assert noise.growth == float(level)
                assert noise.base_std == 0.0


def test_grid_variant_name_round_trip() -> None:
    for family in (script.HOMO_FAMILY, script.HETERO_FAMILY):
        name = script._grid_run_name(family, 0.25, 0.05)
        assert script._parse_grid_variant(family, name) == (0.25, 0.05)
    assert script._parse_grid_variant(script.HOMO_FAMILY, "theta-offset-0.05") is None
    assert script._parse_grid_variant(script.HOMO_FAMILY, "noise-growth-1__theta-offset-1") is None


def test_task_specs_cover_families_variants_and_seeds() -> None:
    specs = script._task_specs(script.FAMILY_GROUPS["all"])
    expected = (
        len(script.HOMO_FAMILY.new_noise_levels) + len(script.HETERO_FAMILY.new_noise_levels)
    ) * len(script.GRID_THETA_OFFSETS) * len(script.RUN_SEEDS)
    assert len(specs) == expected
    projects = {spec[0] for spec in specs}
    assert projects == {script.HOMO_PROJECT_NAME, script.HETERO_PROJECT_NAME}
    seeds = {spec[3] for spec in specs}
    assert seeds == set(script.RUN_SEEDS)


def test_axis_labels_state_offset_and_gap_definitions() -> None:
    assert r"\theta_0 = \theta^{\mathrm{FO}}_{\mathrm{clean}} + \delta\,\mathbf{1}" in script.X_AXIS_LABEL
    assert "every coordinate" in script.X_AXIS_LABEL
    assert r"\|\hat{\theta}_{\mathrm{final}} - \theta^{\mathrm{FO}}_{\mathrm{clean}}\|_2" in script.THETA_DISTANCE_LABEL
    assert "train batch" in script.OBJECTIVE_GAP_LABEL
