from __future__ import annotations

from experiments.configs import config_template, list_configs


def test_config_template_not_registered_as_runtime_preset() -> None:
    assert "config_template" not in list_configs()


def test_config_template_core_placeholders_exist_and_default_to_none() -> None:
    expected = {
        "STATE_DIM",
        "N_SAMPLES",
        "STEP_RULE",
        "OBJECTIVE",
        "THETA0",
        "BATCH_SIZE",
        "SEED",
        "T_STEPS",
        "STEP_SIZE",
        "GRAD_NORM_TOL",
        "FTOL",
        "SIGMA",
        "N_GRAD_SAMPLES",
        "VERBOSE",
        "PLOT",
        "PLOT_DIR",
        "ENABLED_ESTIMATORS",
        "WANDB_ENABLED",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_GROUP",
        "WANDB_JOB_TYPE",
        "WANDB_TAGS",
        "WANDB_MODE",
        "WANDB_LOG_PLOTS",
        "WANDB_ESTIMATOR_ALLOWLIST",
        "CORRECTNESS",
    }
    for name in expected:
        assert hasattr(config_template, name)
        assert getattr(config_template, name) is None


def test_config_template_extra_placeholders_and_config_default_none() -> None:
    expected = {
        "POLICY",
        "FIXED_BETA_1",
        "FIXED_BETA_2",
        "FIXED_BETA_3",
        "FIXED_BETA_4",
        "PLANTED_ALPHA",
        "PLANTED_BETA",
        "PLANTED_BIAS",
        "PLANTED_U_STAR",
        "CORRECTNESS_GRADIENT_SOURCE",
        "CORRECTNESS_NUMDIFF_METHOD",
        "CORRECTNESS_NUMDIFF_STEP",
        "CORRECTNESS_NUMDIFF_AGGREGATE",
        "CORRECTNESS_NUMDIFF_BOUNDS",
        "CONFIG",
    }
    for name in expected:
        assert hasattr(config_template, name)
        assert getattr(config_template, name) is None
