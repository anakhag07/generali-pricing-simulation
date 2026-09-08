"""Small shared helpers for constructing and reporting optimized policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from objective.policy import SoftmaxPolicy


def artifact_policy_features(artifact: Any, frame: pd.DataFrame) -> np.ndarray:
    """Return finite policy features produced by a saved artifact preprocessor."""
    if artifact.preprocessor is None:
        raise ValueError("Artifact policy features require a saved preprocessor.")
    transformed = artifact.preprocessor.transform(
        frame.loc[:, list(artifact.x_feature_cols)]
    )
    features = np.asarray(transformed, dtype=float)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Artifact policy features must be a finite matrix.")
    return features


def constant_softmax_theta(
    policy: SoftmaxPolicy,
    feature_dim: int,
    action: float,
) -> np.ndarray:
    """Initialize a softmax policy to emit one interior action for every row."""
    fraction = (float(action) - policy.action_low) / policy.action_span
    if not 0.0 < fraction < 1.0:
        raise ValueError("action must lie strictly inside the policy bounds.")
    theta = np.zeros(policy.theta_dim(int(feature_dim)), dtype=float)
    theta[0] = np.log(fraction / (1.0 - fraction))
    return theta


def optimization_trace_summary(trace: Any) -> dict[str, Any]:
    """Return the common optimizer convergence fields used in provenance files."""
    return {
        "success": bool(trace.optimizer_success),
        "status": int(trace.optimizer_status),
        "message": str(trace.optimizer_message),
        "steps": max(0, len(trace.steps) - 1),
        "final_gradient_norm": float(trace.theta_grad_norms[-1]),
        "constraint_violation": (
            None
            if trace.constraint_violation is None
            else float(trace.constraint_violation)
        ),
        "optimality": (
            None
            if trace.optimizer_optimality is None
            else float(trace.optimizer_optimality)
        ),
    }


def load_acceptance_floor(path: str | Path) -> float:
    """Read and validate an acceptance floor from a saved NumPy policy artifact."""
    with np.load(path, allow_pickle=False) as artifact:
        floor = float(artifact["acceptance_floor"])
    if not 0.0 < floor < 1.0:
        raise ValueError("Acceptance floor must lie strictly between zero and one.")
    return floor


__all__ = [
    "artifact_policy_features",
    "constant_softmax_theta",
    "load_acceptance_floor",
    "optimization_trace_summary",
]
