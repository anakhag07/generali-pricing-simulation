from __future__ import annotations

import pytest

from experiments.runner import ExperimentConfig


def test_beta_2_must_be_negative() -> None:
    with pytest.raises(
        ValueError,
        match="beta_2 must be negative; acceptance probability should decrease as policy value increases.",
    ):
        ExperimentConfig(beta_2=0.0)

    with pytest.raises(
        ValueError,
        match="beta_2 must be negative; acceptance probability should decrease as policy value increases.",
    ):
        ExperimentConfig(beta_2=0.5)
