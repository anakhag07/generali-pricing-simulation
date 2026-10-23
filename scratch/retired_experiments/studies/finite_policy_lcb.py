"""Compatibility alias for :mod:`experiments.policy_lcb.finite`."""

import sys

from scratch.retired_experiments.studies.policy_lcb import finite as _finite

sys.modules[__name__] = _finite
