from __future__ import annotations

import numpy as np

from model import POLICY_LINEAR, PolicySpec, policy_u
from objective.base import StateVector


def test_model_package_exports_are_importable() -> None:
    policy = PolicySpec(theta=np.asarray([0.1, 0.2], dtype=float), kind=POLICY_LINEAR)
    u = policy_u(policy.theta, StateVector(values=np.asarray([0.5], dtype=float)), kind=policy.kind)
    assert isinstance(u, float)
