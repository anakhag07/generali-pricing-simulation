"""Synthetic objectives with analytically known optima and no data dependency.

Nothing here may import `data`: these objectives are self-contained, which is what
keeps them usable as fast fixtures and reference benchmarks (enforced by
`tests/objective/test_package_boundary.py`). Two families live here:

- the `ladder` of policy-free theta-space benchmarks with a known $$w^*$$
- policy-based objectives with a known $$u^*$$ (`planted_logistic`,
  `fixed_regression`)
"""

from objective.objectives.synthetic.fixed_regression import FixedRegressionObjective
from objective.objectives.synthetic.ladder import (
    IMPLEMENTED_SYNTHETIC_LADDER,
    PiecewiseConvex,
    PiecewiseNonconvexDoubleWell,
    SmoothedNonconvex,
    StronglyConvexQuadratic,
    SYNTHETIC_LADDER,
    SyntheticFunction,
)
from objective.objectives.synthetic.planted_logistic import PlantedLogisticObjective
from objective.objectives.synthetic.proof_validation import ZerothOrderProofObjective

__all__ = [
    "FixedRegressionObjective",
    "IMPLEMENTED_SYNTHETIC_LADDER",
    "PiecewiseConvex",
    "PiecewiseNonconvexDoubleWell",
    "PlantedLogisticObjective",
    "SmoothedNonconvex",
    "StronglyConvexQuadratic",
    "SYNTHETIC_LADDER",
    "SyntheticFunction",
    "ZerothOrderProofObjective",
]
