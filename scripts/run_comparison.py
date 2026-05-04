"""Run a named preset comparison and write aggregate overlay plots."""

from __future__ import annotations

from data.loader import load_mean_observed_acceptance, load_x_array, sample_csv_row_indices
from experiments.comparison_utils import ComparisonSpec, run_preset_comparison

PROJECT_NAME = "glm-policy-comparison"
SEED = 42
_acceptance_floor = load_mean_observed_acceptance("glm")
_row_indices = sample_csv_row_indices("glm", n_rows=5000, seed=SEED)


COMPARISON_SPECS = (
    ComparisonSpec(
        name="constant",
        preset="real_data_glm_constant_policy_base",
    ),
    ComparisonSpec(
        name="linear",
        preset="real_data_glm_linear_policy_base",
    ),
    ComparisonSpec(
        name="quadratic",
        preset="real_data_glm_linear_policy_quadratic_base",
    ),
    ComparisonSpec(
        name="third-order",
        preset="real_data_glm_linear_policy_cubic_base",
    ),
    ComparisonSpec(
        name="fourth-order",
        preset="real_data_glm_linear_policy_quartic_base",
    ),
    ComparisonSpec(
        name="softmax-linear",
        preset="real_data_glm_softmax_policy_base",
    ),
    ComparisonSpec(
        name="softmax-quadratic",
        preset="real_data_glm_softmax_policy_quadratic_base",
    ),
    ComparisonSpec(
        name="softmax-third-order",
        preset="real_data_glm_softmax_policy_cubic_base",
    ),
    ComparisonSpec(
        name="softmax-fourth-order",
        preset="real_data_glm_softmax_policy_quartic_base",
    ),
    ComparisonSpec(
        name="mlp",
        preset="real_data_glm_mlp_policy_base",
    ),
)

COMMON_OVERRIDES = {
    "n_samples": 5000,
    "x_fixed": load_x_array("glm", row_indices=_row_indices),
    "x_fixed_row_indices": _row_indices,
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
