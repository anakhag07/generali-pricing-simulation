from __future__ import annotations

import json

from experiments.configs import get_config
from objective.noise import HeteroskedasticGaussianNoise, HomoskedasticGaussianNoise, NoisyObjective
from scratch import run_planted_logistic_gradient_sample_noise_sweep as script


def test_task_specs_cover_asymmetric_estimator_grid() -> None:
    args = script._parse_args([])

    specs = script._task_specs(args)

    assert len(specs) == 72
    finite = [(variant, seed) for variant, seed in specs if variant.estimator == script.FINITE_DIFFERENCE]
    stein = [(variant, seed) for variant, seed in specs if variant.estimator == script.STEIN_DIFFERENCE]
    assert len(finite) == 18  # 2 families x 3 noise levels x 3 seeds
    assert len(stein) == 54  # finite grid plus 3 n_grad_samples values for Stein
    assert {variant.n_grad_samples for variant, _ in finite} == {None}
    assert {variant.n_grad_samples for variant, _ in stein} == {32, 64, 128}
    assert all("__ngrad-" not in variant.name for variant, _ in finite)
    assert all("__ngrad-" in variant.name for variant, _ in stein)
    assert {seed for _, seed in specs} == {7, 8, 9}


def test_noise_objective_and_seed_overrides_are_configured() -> None:
    args = script._parse_args(["--run-seeds", "11", "--anchor-seed", "7", "--t-steps", "5"])
    hetero_variant = next(
        variant
        for variant in script._variants(("heteroskedastic",))
        if variant.estimator == script.STEIN_DIFFERENCE and variant.n_grad_samples == 64
    )

    config = script._config_for_variant(hetero_variant, 11, args)

    assert config.enabled_estimators == (script.STEIN_DIFFERENCE,)
    assert config.n_grad_samples == 64
    assert config.t_steps == 5
    assert config.correctness.gradient_source == "denoised_exact"
    assert config.seed_setup is not None
    assert config.seed_setup.data_seed == 7
    assert config.seed_setup.split_seed == 7
    assert config.seed_setup.theta_seed == 7
    assert config.seed_setup.optimizer_seed == 11
    assert config.seed_setup.noise_seed == 11
    assert isinstance(config.objective, NoisyObjective)
    assert isinstance(config.objective.noise, HeteroskedasticGaussianNoise)
    assert config.objective.noise.growth == hetero_variant.noise_level
    assert config.objective.noise.base_std == 0.0
    assert config.objective.noise.u_center == 0.1


def test_finite_difference_varies_noise_without_n_grad_axis() -> None:
    args = script._parse_args(["--run-seeds", "11"])
    finite_variant = next(
        variant
        for variant in script._variants(("homoskedastic",))
        if variant.estimator == script.FINITE_DIFFERENCE and variant.noise_level == 0.5
    )

    config = script._config_for_variant(finite_variant, 11, args)

    assert config.enabled_estimators == (script.FINITE_DIFFERENCE,)
    assert config.n_grad_samples == get_config(script.BASE_PRESET).n_grad_samples
    assert finite_variant.n_grad_samples is None
    assert isinstance(config.objective, NoisyObjective)
    assert isinstance(config.objective.noise, HomoskedasticGaussianNoise)
    assert config.objective.noise.std == 0.5


def test_launch_plan_defaults_to_array_tasks() -> None:
    args = script._parse_args(["--families", "homoskedastic", "--run-seeds", "1", "2"])

    plan = script._build_launch_plan(args)

    assert plan.name == script.PROJECT_NAME
    assert plan.task_count == 24  # 3 noise levels x (1 FD + 3 Stein n_grad) x 2 seeds
    assert plan.default_array is True
    assert plan.requires_jax is False


def test_collector_writes_final_and_summary_csvs(tmp_path) -> None:
    summary_path = tmp_path / "summary-seed-7.json"
    summary_path.write_text(
        json.dumps(
            {
                "estimators": {
                    script.STEIN_DIFFERENCE: {
                        "final_value": 1.25,
                        "final_u": 0.2,
                        "runtime_sec": 3.5,
                        "optimizer_success": True,
                        "optimizer_status": 0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    payloads = [
        {
            "project": "proj",
            "variant": "stein_difference__homoskedastic-std-0.5__ngrad-64",
            "estimator": script.STEIN_DIFFERENCE,
            "run_seed": 7,
            "noise_family": "homoskedastic",
            "noise_level": 0.5,
            "noise_std": 0.5,
            "noise_growth": 0.0,
            "n_grad_samples": 64,
            "summary_json": str(summary_path),
            "run_dir": str(tmp_path / "seed-7"),
        }
    ]

    script._write_outputs_from_payloads(payloads, tmp_path, "proj")

    finals = tmp_path / "proj" / "gradient_sample_noise_sweep_finals.csv"
    summary = tmp_path / "proj" / "gradient_sample_noise_sweep_summary.csv"
    assert finals.exists()
    assert summary.exists()
    assert "stein_difference__homoskedastic-std-0.5__ngrad-64" in finals.read_text(encoding="utf-8")
    assert "final_value_mean" in summary.read_text(encoding="utf-8")
