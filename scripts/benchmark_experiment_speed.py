"""Benchmark real-data objective caching and contour subsampling speed."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from data.loader import (
    ACCEPTANCE_STATE_COLS,
    LOSS_FEATURE_COLS,
    extract_glm_u_coef,
    load_model_artifacts,
    load_x_array,
)
from experiments.reporters import _contour_grid_size, _contour_x_samples
from objective.objectives.model_based import ModelBasedObjective
from objective.policy import SoftmaxPolicy
from reporting.visualization import theta_objective_contour_grid


def _make_glm_objective() -> ModelBasedObjective:
    acceptance_model, loss_model = load_model_artifacts("glm")
    return ModelBasedObjective(
        policy=SoftmaxPolicy(),
        acceptance_model=acceptance_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=9,
        u_coef=extract_glm_u_coef(acceptance_model),
    )


def _timed(callable_obj):
    start = time.perf_counter()
    result = callable_obj()
    return result, time.perf_counter() - start


def _speedup(before: float, after: float) -> float | None:
    if after <= 0.0:
        return None
    return before / after


def benchmark_repeated_value(x: np.ndarray) -> dict[str, object]:
    objective = _make_glm_objective()
    theta = np.zeros(objective.policy_theta_dim(), dtype=float)

    first_value, first_sec = _timed(lambda: objective.value(theta, x))
    counts_after_first = objective.eval_counts()
    second_value, second_sec = _timed(lambda: objective.value(theta, x))
    counts_after_second = objective.eval_counts()

    return {
        "first_value": float(first_value),
        "second_value": float(second_value),
        "first_sec": first_sec,
        "second_sec": second_sec,
        "speedup_second_over_first": _speedup(first_sec, second_sec),
        "loss_predict_calls_after_first": counts_after_first.get("loss_predict_calls", 0),
        "loss_predict_calls_after_second": counts_after_second.get("loss_predict_calls", 0),
        "loss_prediction_cache_hits_after_second": counts_after_second.get("loss_prediction_cache_hits", 0),
        "policy_features_cache_hits_after_second": counts_after_second.get("policy_features_cache_hits", 0),
    }


def benchmark_contours(x: np.ndarray, grid_size: int, sampled_grid_size: int | None = None) -> dict[str, object]:
    full_objective = _make_glm_objective()
    sampled_objective = _make_glm_objective()
    theta = np.zeros(full_objective.policy_theta_dim(), dtype=float)
    sampled_x = _contour_x_samples(x, sampled_objective, max_rows=200)
    sampled_grid = int(
        sampled_grid_size
        if sampled_grid_size is not None
        else min(grid_size, _contour_grid_size(sampled_objective))
    )

    _, full_sec = _timed(
        lambda: theta_objective_contour_grid(
            x,
            full_objective,
            theta,
            grid_size=grid_size,
        )
    )
    _, sampled_sec = _timed(
        lambda: theta_objective_contour_grid(
            sampled_x,
            sampled_objective,
            theta,
            grid_size=sampled_grid,
        )
    )

    return {
        "full_grid_size": int(grid_size),
        "sampled_grid_size": sampled_grid,
        "full_rows": int(x.shape[0]),
        "sampled_rows": int(sampled_x.shape[0]),
        "full_sec": full_sec,
        "sampled_sec": sampled_sec,
        "speedup_full_over_sampled": _speedup(full_sec, sampled_sec),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--sampled-grid-size", type=int, default=None)
    args = parser.parse_args()

    x = load_x_array("glm", n_rows=args.n_rows, seed=args.seed)
    payload = {
        "n_rows": int(x.shape[0]),
        "repeated_value": benchmark_repeated_value(x),
        "contours": benchmark_contours(
            x,
            grid_size=int(args.grid_size),
            sampled_grid_size=args.sampled_grid_size,
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
