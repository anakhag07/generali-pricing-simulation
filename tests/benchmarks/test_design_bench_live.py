import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


LEGACY_PYTHON = os.environ.get("DESIGN_BENCH_PYTHON")
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "design_bench.py"
pytestmark = [
    pytest.mark.design_bench_live,
    pytest.mark.skipif(not LEGACY_PYTHON, reason="Set DESIGN_BENCH_PYTHON for live tests."),
]


@pytest.mark.parametrize(
    "task_name,dimension",
    [("AntMorphology-Exact-v0", 60), ("DKittyMorphology-Exact-v0", 56)],
)
def test_oracle_and_baseline_smoke(task_name, dimension, tmp_path):
    root = tmp_path / task_name
    dataset = root / "dataset"
    subprocess.run(
        [
            LEGACY_PYTHON,
            str(SCRIPT),
            "export-dataset",
            "--task",
            task_name,
            "--output",
            str(dataset),
        ],
        check=True,
    )
    candidate = root / "candidate.npy"
    np.save(candidate, np.load(dataset / "x.npy")[:1], allow_pickle=False)
    evaluation = root / "evaluation"
    subprocess.run(
        [
            LEGACY_PYTHON,
            str(SCRIPT),
            "evaluate",
            "--dataset",
            str(dataset),
            "--candidates",
            str(candidate),
            "--output",
            str(evaluation),
        ],
        check=True,
    )
    assert np.load(evaluation / "scores.npy").shape == (1, 1)

    baseline = root / "baseline"
    subprocess.run(
        [
            LEGACY_PYTHON,
            str(SCRIPT),
            "run-gradient-ascent",
            "--dataset",
            str(dataset),
            "--mode",
            "smoke",
            "--seed",
            "7",
            "--output",
            str(baseline),
        ],
        check=True,
    )
    assert np.load(baseline / "candidates.npy").shape == (2, dimension)
