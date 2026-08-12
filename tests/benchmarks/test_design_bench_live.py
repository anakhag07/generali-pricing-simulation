from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.design_bench import (
    ANT_TASK,
    DKITTY_TASK,
    DesignBenchBridge,
    DesignBenchTaskSpec,
)


LEGACY_PYTHON = os.environ.get("DESIGN_BENCH_PYTHON")
pytestmark = [
    pytest.mark.design_bench_live,
    pytest.mark.skipif(
        not LEGACY_PYTHON,
        reason="Set DESIGN_BENCH_PYTHON to run legacy Design-Bench integration tests.",
    ),
]


@pytest.mark.parametrize("task_name", [ANT_TASK, DKITTY_TASK])
def test_exact_oracle_and_gradient_ascent_smoke(task_name: str, tmp_path: Path) -> None:
    assert LEGACY_PYTHON is not None
    bridge = DesignBenchBridge(LEGACY_PYTHON)
    task_root = tmp_path / task_name
    dataset = bridge.export_dataset(DesignBenchTaskSpec(task_name), task_root / "dataset")

    first = bridge.evaluate(dataset, dataset.x[:1], task_root / "first-evaluation")
    repeated = bridge.evaluate(dataset, dataset.x[:1], task_root / "repeated-evaluation")
    assert first.scores.shape == (1, 1)
    assert np.all(np.isfinite(first.scores))
    np.testing.assert_allclose(first.scores, repeated.scores, rtol=1e-6, atol=1e-6)

    baseline = bridge.run_gradient_ascent(
        dataset,
        task_root / "gradient-ascent-smoke",
        mode="smoke",
        seed=7,
    )
    assert baseline.raw_candidates.shape == (2, dataset.task.dimension)
    scored = bridge.evaluate(
        dataset,
        baseline.raw_candidates,
        task_root / "baseline-evaluation",
    )
    assert scored.scores.shape == (2, 1)
    assert np.all(np.isfinite(scored.scores))
