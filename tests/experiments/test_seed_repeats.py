from __future__ import annotations

import json

from experiments.seed_repeats import SeedRepeatSpec, run_seed_repeats, seed_setup_for_repeat
from experiments.seed_repeats import _seed_repeat_output_dir


def test_seed_setup_for_repeat_varies_only_optimizer_by_default() -> None:
    spec = SeedRepeatSpec(base_preset="planted_logistic_base", run_seeds=(10, 11))

    setup = seed_setup_for_repeat(spec, run_seed=11)

    assert setup.run_seed == 11
    assert setup.data_seed == 10
    assert setup.split_seed == 10
    assert setup.theta_seed == 10
    assert setup.noise_seed == 10
    assert setup.optimizer_seed == 11


def test_seed_setup_for_repeat_allows_explicit_fixed_seeds() -> None:
    spec = SeedRepeatSpec(
        base_preset="planted_logistic_base",
        run_seeds=(10, 11),
        fixed_data_seed=100,
        fixed_split_seed=101,
        fixed_theta_seed=102,
        fixed_noise_seed=103,
        vary=("optimizer",),
    )

    setup = seed_setup_for_repeat(spec, run_seed=11)

    assert setup.data_seed == 100
    assert setup.split_seed == 101
    assert setup.theta_seed == 102
    assert setup.noise_seed == 103
    assert setup.optimizer_seed == 11


def test_seed_setup_for_repeat_all_mode_derives_streams_from_run_seed() -> None:
    spec = SeedRepeatSpec(
        base_preset="planted_logistic_base",
        run_seeds=(10, 11),
        vary=("all",),
    )

    setup = seed_setup_for_repeat(spec, run_seed=11)

    assert setup.run_seed == 11
    assert setup.data_seed is None
    assert setup.split_seed is None
    assert setup.theta_seed is None
    assert setup.noise_seed is None
    assert setup.optimizer_seed is None


def test_run_seed_repeats_writes_aggregate_outputs(tmp_path) -> None:
    spec = SeedRepeatSpec(
        base_preset="planted_logistic_base",
        run_seeds=(1, 2),
        overrides={
            "n_samples": 6,
            "t_steps": 1,
            "enabled_estimators": ("first_order",),
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        },
        output_root=str(tmp_path),
        project_name="seed-repeat-test",
    )

    output = run_seed_repeats(spec)

    assert (output.output_dir / "seed_repeats.csv").exists()
    assert (output.output_dir / "seed_repeats_summary.csv").exists()
    assert len(output.final_rows) == 2
    assert output.summary_rows[0]["estimator"] == "first_order"
    assert output.summary_rows[0]["n_runs"] == 2
    run_summary_path = next((output.output_dir / "runs").glob("seed-1__*/summary.json"))
    summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    assert summary["preset"]["preset_name"] == "planted_logistic_base"
    assert summary["preset"]["run_seed"] == 1


def test_seed_repeat_default_output_root_uses_results_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GENERALI_RESULTS_ROOT", str(tmp_path / "results"))
    spec = SeedRepeatSpec(base_preset="planted_logistic_base", run_seeds=(1,))

    output_dir = _seed_repeat_output_dir(spec)

    assert output_dir.parent == tmp_path / "results" / "seed-repeats"
