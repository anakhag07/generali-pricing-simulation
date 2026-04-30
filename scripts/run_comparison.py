"""Run a named preset comparison and write aggregate overlay plots."""

from __future__ import annotations

from data.loader import load_mean_observed_acceptance, load_x_array
from experiments.comparison_utils import ComparisonSpec, run_preset_comparison

PROJECT_NAME = "glm-policy-comparison"
_acceptance_floor = load_mean_observed_acceptance("glm")


COMPARISON_SPECS = (
    ComparisonSpec(
        name="constant-constrained",
        preset="real_data_glm_constant_policy_base",
    ),
    ComparisonSpec(
        name="linear-constrained",
        preset="real_data_glm_linear_policy_base",
    ),
    ComparisonSpec(
        name="softmax-linear-constrained",
        preset="real_data_glm_softmax_policy_base",
    ),
    ComparisonSpec(
        name="softmax-quadratic-constrained",
        preset="real_data_glm_softmax_policy_quadratic_base",
    ),
)

COMMON_OVERRIDES = {
    "n_samples": 5000,
    "x_fixed": load_x_array("glm", n_rows=5000),
    "enabled_estimators": (
        "first_order",
        # "finite_difference",
        # "spsa",
        # "stein_difference",
    ),
    # The comparison run writes aggregate overlay plots, so per-run plots are off by default.
    "plot": False,
    "verbose": True,
    "wandb_enabled": False,
    "step_rule": "trust-constr",
    "acceptance_floor": _acceptance_floor,
}


def main() -> None:
    results = run_preset_comparison(
        specs=COMPARISON_SPECS,
        common_overrides=COMMON_OVERRIDES,
        project_name=PROJECT_NAME,
        validate_shared_estimators=True,
        validate_shared_x=True,
    )
    print(f"Completed {len(results)} comparison runs for project '{PROJECT_NAME}'.")


if __name__ == "__main__":
    main()
