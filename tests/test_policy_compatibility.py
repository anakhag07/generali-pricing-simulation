from __future__ import annotations

from model.policy import PolicySpec as ModelPolicySpec
from model.policy import policy_u as model_policy_u
from objective.policy import PolicySpec as ObjectivePolicySpec
from objective.policy import policy_u as objective_policy_u


def test_model_policy_shim_matches_objective_policy_api() -> None:
    assert ModelPolicySpec is ObjectivePolicySpec
    assert model_policy_u is objective_policy_u
