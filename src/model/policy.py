"""Compatibility shim for policy APIs moved to objective.policy."""

from objective.policy import (
    POLICY_CONSTANT,
    POLICY_KINDS,
    POLICY_LINEAR,
    POLICY_SOFTMAX,
    PolicySpec,
    apply_policy,
    phi,
    phi_batch,
    policy_grad_theta,
    policy_u,
    policy_u_batch,
)

__all__ = [
    "POLICY_CONSTANT",
    "POLICY_KINDS",
    "POLICY_LINEAR",
    "POLICY_SOFTMAX",
    "PolicySpec",
    "apply_policy",
    "phi",
    "phi_batch",
    "policy_grad_theta",
    "policy_u",
    "policy_u_batch",
]
