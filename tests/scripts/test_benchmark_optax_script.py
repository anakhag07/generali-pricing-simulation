"""Benchmark script: argument defaults and planted-logistic row assembly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("optax")

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_optax_vs_trust_constr.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_optax_vs_trust_constr", _SCRIPT)
benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = benchmark
_SPEC.loader.exec_module(benchmark)


def test_parse_args_defaults() -> None:
    args = benchmark._parse_args([])
    assert args.state_dim == 199
    assert args.glm_rows == 20000
    assert not args.skip_glm


def test_planted_logistic_group_rows() -> None:
    args = benchmark._parse_args(
        ["--state-dim", "3", "--logistic-rows", "32", "--logistic-steps", "3"]
    )
    rows = benchmark.run_planted_logistic_group(args)

    assert [row["algorithm"] for row in rows] == ["l-bfgs-b", "optax-adam", "optax-sgd"]
    for row in rows:
        assert row["group"] == "planted_logistic"
        assert row["theta_dim"] == 4
        assert row["n_rows"] == 32
        assert row["wall_time_s"] > 0.0
        assert row["n_steps"] >= 1
        assert row["mean_u_gap"] != ""
    assert set(benchmark.FIELDNAMES) >= set(rows[0].keys())
