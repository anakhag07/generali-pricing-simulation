from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from experiments.configs import get_config, list_configs
from experiments.manifest import load_experiment_manifest
from experiments.run import run_experiment
from objective import (
    ConstantThetaRegularizer,
    IntervalDistanceThetaRegularizer,
    RegularizedObjective,
    SmoothSaturatingIntervalThetaRegularizer,
    ThetaBiasedObjective,
    ZerothOrderProofObjective,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "manifests" / "zeroth_order_baseline.json"
BIAS_PATH = REPO_ROOT / "manifests" / "zeroth_order_functional_bias.json"
ENVELOPE_PATH = REPO_ROOT / "manifests" / "zeroth_order_envelopes.json"


def test_proof_preset_is_registered_with_fixed_initial_point() -> None:
    assert "zeroth_order_proof_base" in list_configs()
    config = get_config("zeroth_order_proof_base")

    assert isinstance(config.objective, ZerothOrderProofObjective)
    np.testing.assert_allclose(config.theta0, np.asarray([1.0]))
    assert config.step_rule == "constant"
    assert config.perturbation_space == "theta"


def test_baseline_manifest_has_two_independent_five_point_sweeps() -> None:
    manifest = load_experiment_manifest(BASELINE_PATH)
    sigma_variants = [v for v in manifest.variants if v.axes["sweep"] == "sigma"]
    m_variants = [v for v in manifest.variants if v.axes["sweep"] == "m"]

    assert [v.axes["sigma"] for v in sigma_variants] == [0.05, 0.075, 0.1125, 0.16875, 0.253125]
    assert [v.axes["m"] for v in m_variants] == [4, 8, 16, 32, 64]
    assert all(v.overrides["n_grad_samples"] == 128 for v in sigma_variants)
    assert all(v.overrides["enabled_estimators"] == ["stein_difference"] for v in m_variants)
    assert len(manifest.seeds.run_seeds) == 16
    assert manifest.seeds.vary == ("optimizer",)


def test_bias_manifest_has_requested_forms_and_alpha_grid() -> None:
    manifest = load_experiment_manifest(BIAS_PATH)
    observed: dict[str, list[float]] = {}
    for variant in manifest.variants:
        spec = variant.overrides["objective_modifications"][0]
        bias = spec["bias"]
        observed.setdefault(str(bias["type"]), []).append(float(bias["alpha"]))
        config = get_config(manifest.base_preset, overrides=variant.overrides)
        assert isinstance(config.objective, ThetaBiasedObjective)

    expected_alpha = [0.0, 0.025, 0.05, 0.1, 0.2]
    assert observed == {
        "LinearThetaBias": expected_alpha,
        "ArctanThetaBias": expected_alpha,
        "ArctanRemainderThetaBias": expected_alpha,
    }
    assert len(manifest.seeds.run_seeds) == 16
    assert manifest.seeds.vary == ("optimizer",)


def test_proof_preset_runs_small_fd_and_stein_smoke() -> None:
    config = get_config(
        "zeroth_order_proof_base",
        overrides={
            "t_steps": 3,
            "n_grad_samples": 4,
            "enabled_estimators": ("finite_difference", "stein_difference"),
        },
    )
    result = run_experiment(config)

    assert set(result.results) == {"finite_difference", "stein_difference"}
    assert all(item.theta.shape == (1,) for item in result.results.values())


def test_envelope_manifest_has_calibrated_conditional_sweep() -> None:
    manifest = load_experiment_manifest(ENVELOPE_PATH)

    assert len(manifest.variants) == 54
    assert Counter(variant.axes["form"] for variant in manifest.variants) == {
        "constant": 6,
        "linear": 12,
        "smooth_nonconvex": 36,
    }
    assert manifest.seeds.run_seeds == tuple(range(101, 109))
    assert manifest.seeds.vary == ("optimizer",)
    assert manifest.optimizer["t_steps"] == 1000
    assert manifest.optimizer["n_grad_samples"] == 64

    observed_types = set()
    for variant in manifest.variants:
        config = get_config(manifest.base_preset, overrides=variant.overrides)
        assert isinstance(config.objective, RegularizedObjective)
        assert len(config.objective.regularizers) == 1
        observed_types.add(type(config.objective.regularizers[0]))
    assert observed_types == {
        ConstantThetaRegularizer,
        IntervalDistanceThetaRegularizer,
        SmoothSaturatingIntervalThetaRegularizer,
    }


def test_envelope_manifest_runs_small_fd_and_stein_smoke() -> None:
    manifest = load_experiment_manifest(ENVELOPE_PATH)
    variant = next(
        variant
        for variant in manifest.variants
        if variant.name == "smooth-nonconvex-a-0p42-init-0-sigma-0p15"
    )
    config = get_config(
        manifest.base_preset,
        overrides={
            **variant.overrides,
            "t_steps": 3,
            "n_grad_samples": 4,
        },
    )
    result = run_experiment(config)

    assert set(result.results) == {"finite_difference", "stein_difference"}
    assert all(item.theta.shape == (1,) for item in result.results.values())
