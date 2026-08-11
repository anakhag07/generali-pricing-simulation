"""Compatibility alias for :mod:`experiments.policy_lcb.finite`."""

import sys

from experiments.policy_lcb import finite as _finite

sys.modules[__name__] = _finite
