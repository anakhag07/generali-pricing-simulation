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
        preset="real_data_glm_base",
        overrides={"policy_kind": "constant", "seed": 8},
    ),
    ComparisonSpec(
        name="linear",
        preset="real_data_glm_base",
        overrides={"policy_kind": "linear", "seed": 8},
    ),
    ComparisonSpec(
        name="quadratic",
        preset="real_data_glm_base",
        overrides={"policy_kind": "linear", "feature_order": "quadratic", "seed": 8},
    ),
    ComparisonSpec(
        name="third-order",
        preset="real_data_glm_base",
        overrides={"policy_kind": "linear", "feature_order": "cubic", "seed": 8},
    ),
    ComparisonSpec(
        name="fourth-order",
        preset="real_data_glm_base",
        overrides={"policy_kind": "linear", "feature_order": "quartic", "seed": 8},
    ),
    ComparisonSpec(
        name="softmax-linear",
        preset="real_data_glm_base",
        overrides={"policy_kind": "softmax"},
    ),
    ComparisonSpec(
        name="softmax-quadratic",
        preset="real_data_glm_base",
        overrides={"policy_kind": "softmax", "feature_order": "quadratic"},
    ),
    ComparisonSpec(
        name="softmax-third-order",
        preset="real_data_glm_base",
        overrides={"policy_kind": "softmax", "feature_order": "cubic"},
    ),
    ComparisonSpec(
        name="softmax-fourth-order",
        preset="real_data_glm_base",
        overrides={"policy_kind": "softmax", "feature_order": "quartic"},
    ),
    ComparisonSpec(
        name="mlp",
        preset="real_data_glm_base",
        overrides={"policy_kind": "mlp"},
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
