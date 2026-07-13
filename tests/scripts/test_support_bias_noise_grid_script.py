from __future__ import annotations

import numpy as np

from objective.noise import (
    HeteroskedasticGaussianNoise,
    HomoskedasticGaussianNoise,
    NoisyObjective,
)
from objective.objectives import (
    BiasedObjective,
    PlantedLogisticObjective,
    UpperSupportHingeBias,
)
from scripts import run_support_bias_noise_grid as script


def test_objective_layers_noise_on_support_bias() -> None:
    # The per-variant objective must be NoisyObjective(BiasedObjective(planted))
    # with an UpperSupportHingeBias centered at u* and the requested noise family.
    for family in (script.HOMO_FAMILY, script.HETERO_FAMILY):
        objective = family.noisy_objective(0.1, family.new_noise_levels[-1])
        assert isinstance(objective, NoisyObjective)
        biased = objective.base_objective
        assert isinstance(biased, BiasedObjective)
        assert isinstance(biased.base_objective, PlantedLogisticObjective)
        bias = biased.bias
        assert isinstance(bias, UpperSupportHingeBias)
        assert bias.lambda_bias == 0.1
        assert bias.support_center == script.U_STAR
        assert bias.support_radius == script.SUPPORT_RADIUS
        if family is script.HOMO_FAMILY:
            assert isinstance(objective.noise, HomoskedasticGaussianNoise)
            assert objective.noise.std == float(family.new_noise_levels[-1])
        else:
            assert isinstance(objective.noise, HeteroskedasticGaussianNoise)
            assert objective.noise.growth == float(family.new_noise_levels[-1])
            assert objective.noise.base_std == 0.0
            assert objective.noise.u_center == script.U_STAR


def test_grid_override_lists_cover_noise_by_lambda_product() -> None:
    for family in (script.HOMO_FAMILY, script.HETERO_FAMILY):
        override_list = script._build_grid_override_list(family)
        assert len(override_list) == len(family.new_noise_levels) * len(script.LAMBDA_BIAS_VALUES)
        expected_names = [
            script._grid_run_name(family, level, lam)
            for level in family.new_noise_levels
            for lam in script.LAMBDA_BIAS_VALUES
        ]
        assert [entry["_run_name"] for entry in override_list] == expected_names
        # theta0 is a fixed cold start (not swept), so no override carries it.
        assert all("theta0" not in entry for entry in override_list)


def test_grid_variant_name_round_trip() -> None:
    for family in (script.HOMO_FAMILY, script.HETERO_FAMILY):
        name = script._grid_run_name(family, 0.5, 0.05)
        assert script._parse_grid_variant(family, name) == (0.5, 0.05)
    assert script._parse_grid_variant(script.HOMO_FAMILY, "lambda-0.05") is None
    # Cross-family names must not parse (guards against mixing homo/hetero dirs).
    assert script._parse_grid_variant(script.HOMO_FAMILY, "noise-growth-1__lambda-0.1") is None


def test_task_groups_are_one_per_family_noise_level() -> None:
    groups = script._task_groups(script.FAMILY_GROUPS["all"])
    n_levels = len(script.HOMO_FAMILY.new_noise_levels) + len(script.HETERO_FAMILY.new_noise_levels)
    assert len(groups) == n_levels
    projects = {family.project_name for family, _, _ in groups}
    assert projects == {script.HOMO_PROJECT_NAME, script.HETERO_PROJECT_NAME}
    for _, _, variants in groups:
        assert len(variants) == len(script.LAMBDA_BIAS_VALUES)
    total_runs = sum(len(variants) for _, _, variants in groups) * len(script.RUN_SEEDS)
    assert total_runs == n_levels * len(script.LAMBDA_BIAS_VALUES) * len(script.RUN_SEEDS)


def test_zeroth_order_estimators_only() -> None:
    # NoisyObjective.grad raises, so the grid must not enable first_order.
    assert "first_order" not in script.REQUIRED_ESTIMATORS
    assert set(script.REQUIRED_ESTIMATORS) == {"finite_difference", "stein_difference"}
    assert script.COMMON_OVERRIDES["enabled_estimators"] == script.REQUIRED_ESTIMATORS


def test_planted_and_bias_reconstruction_from_serialized_objective() -> None:
    # The reconstruction must descend the wrapper chain to the clean planted
    # objective and recover the bias support boundary for support-excess metrics.
    from experiments.config import _objective_to_dict

    objective = script.HOMO_FAMILY.noisy_objective(0.2, 0.5)
    summary = {"config": {"objective": _objective_to_dict(objective)}}
    planted, bias_dict = script._planted_and_bias(summary)
    assert isinstance(planted, PlantedLogisticObjective)
    assert bias_dict["type"] == "UpperSupportHingeBias"
    np.testing.assert_allclose(bias_dict["support_upper"], script.U_STAR + script.SUPPORT_RADIUS)


def test_axis_labels_state_bias_and_gap_definitions() -> None:
    assert r"\lambda_{\mathrm{bias}}" in script.X_AXIS_LABEL
    assert "noise layered on top" in script.X_AXIS_LABEL
    assert r"J_{\mathrm{clean}}(\theta^{\mathrm{FO}}_{\mathrm{clean}})" in script.TRUE_GAP_LABEL
    assert "support excess" in script.SUPPORT_EXCESS_LABEL
