import numpy as np
import pytest

from experiments.envelopes.core import (
    FittedEnvelope, TargetDomain, check_finite_containment,
)


@pytest.mark.parametrize("contexts,expected,status", [
    ([2., 0., .5, 2.], [False, True, True, False], "violated"),
    ([0., .5, -.5], [True, True, True], "covered"),
])
def test_finite_context_coverage_requires_every_target_action(contexts, expected, status):
    target = TargetDomain(np.asarray(contexts)[:, None], [-1., 2.])
    envelope = FittedEnvelope(
        lambda X, a: np.zeros(len(a)), lambda X, a: np.ones(len(a)),
    )
    # At x=+/-0.5, action 2 touches the boundary. At x=2, both actions fail.
    result = check_finite_containment(envelope, lambda X, a: X[:, 0]*a, target)
    assert result.status == status
    assert result.domain == "finite_set"
    assert result.details["point_count"] == 2*len(contexts)
    assert result.details["context_count"] == len(contexts)
    assert result.details["contexts"] == [
        {"context_index": index, "covered": covered}
        for index, covered in enumerate(expected)
    ]
    if status == "violated":
        assert result.details["witness"] == {
            "context": [2.], "action": -1., "excess": -1.,
        }
    else:
        assert "witness" not in result.details
