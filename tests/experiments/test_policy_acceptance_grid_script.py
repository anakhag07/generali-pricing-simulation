from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import plot_policy_acceptance_grid as script


class _FakeObjective:
    def _acceptance_proba(self, x_batch: pd.DataFrame, u_arr: np.ndarray) -> np.ndarray:
        base = x_batch["base"].to_numpy(dtype=float)
        u_values = np.asarray(u_arr, dtype=float).reshape(-1)
        return 1.0 / (1.0 + np.exp(-(base - 3.0 * u_values)))

    def _d_acceptance_du_batch(self, x_batch: pd.DataFrame, u_arr: np.ndarray) -> np.ndarray:
        del u_arr
        return x_batch["sensitivity"].to_numpy(dtype=float)

    def _loss_prediction(self, x_batch: pd.DataFrame) -> np.ndarray:
        return x_batch["loss"].to_numpy(dtype=float)


class _FakeArtifact:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.estimator = "first_order"
        self._frame = frame.reset_index(drop=True)
        self._objective = _FakeObjective()
        self.predict_calls = 0

    def build_objective(self) -> _FakeObjective:
        return self._objective

    def load_x(self, *, split: str = "all") -> pd.DataFrame:
        assert split == "all"
        return self._frame.copy()

    def row_indices(self, split: str) -> np.ndarray:
        assert split == "all"
        return np.arange(100, 100 + self._frame.shape[0], dtype=int)

    def predict_u(self, x_batch: pd.DataFrame, *, clip: bool = True) -> np.ndarray:
        del clip
        self.predict_calls += 1
        return x_batch["policy_u"].to_numpy(dtype=float)


def _frame(n_rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "base": np.linspace(-1.0, 1.0, n_rows),
            "sensitivity": np.linspace(0.01, 0.12, n_rows),
            "loss": np.linspace(100.0, 1200.0, n_rows),
            "policy_u": np.linspace(0.01, 0.12, n_rows),
        }
    )


def test_tertile_positions_are_stable_sorted_groups() -> None:
    scores = np.array([30.0, 10.0, 20.0, 60.0, 50.0, 40.0], dtype=float)

    buckets = script._tertile_positions(scores)

    np.testing.assert_array_equal(buckets["low"], np.array([1, 2], dtype=int))
    np.testing.assert_array_equal(buckets["medium"], np.array([0, 5], dtype=int))
    np.testing.assert_array_equal(buckets["high"], np.array([4, 3], dtype=int))


def test_seeded_bucket_sampling_is_reproducible() -> None:
    buckets = {
        "low": np.arange(0, 4, dtype=int),
        "medium": np.arange(4, 8, dtype=int),
        "high": np.arange(8, 12, dtype=int),
    }

    first = script._sample_bucket_positions(
        buckets,
        n_clients=2,
        rng=np.random.default_rng(123),
    )
    second = script._sample_bucket_positions(
        buckets,
        n_clients=2,
        rng=np.random.default_rng(123),
    )

    for name in script.BUCKET_NAMES:
        np.testing.assert_array_equal(first[name], second[name])
        assert first[name].shape == (2,)


def test_parser_defaults_to_unseeded_sampling() -> None:
    args = script._build_parser().parse_args(["--policy-artifact", "policy.json"])

    assert args.seed is None
    assert args.n_clients == 10
    assert args.u_min == 0.0
    assert args.u_max == 0.15


def test_run_acceptance_grid_writes_plots_and_sample_csv(monkeypatch, tmp_path) -> None:
    artifact = _FakeArtifact(_frame())
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    calls: dict[str, Path] = {}

    def fake_load_policy_artifact(path: Path):
        calls["artifact_path"] = path
        return artifact

    monkeypatch.setattr(script, "load_policy_artifact", fake_load_policy_artifact)
    output_dir = tmp_path / "out"
    args = script._build_parser().parse_args(
        [
            "--policy-artifact",
            str(artifact_dir),
            "--split",
            "all",
            "--u-min",
            "0.0",
            "--u-max",
            "0.15",
            "--u-count",
            "5",
            "--n-clients",
            "2",
            "--seed",
            "7",
            "--output-dir",
            str(output_dir),
            "--dpi",
            "80",
        ]
    )

    outputs = script.run_acceptance_grid(args)

    assert calls["artifact_path"] == artifact_dir / "policy.json"
    assert outputs["sensitivity_plot"].exists()
    assert outputs["predicted_loss_plot"].exists()
    assert outputs["sample_csv"].exists()
    assert outputs["summary_json"].exists()
    assert artifact.predict_calls == 6

    sampled = pd.read_csv(outputs["sample_csv"])
    assert sampled.shape[0] == 12
    assert set(sampled["plot"]) == {"sensitivity", "predicted_loss"}
    assert set(sampled["bucket"]) == {"low", "medium", "high"}
    assert sampled.groupby(["plot", "bucket"]).size().eq(2).all()
    assert sampled["policy_u_in_simulated_range"].all()
    assert sampled["csv_row_index"].between(100, 111).all()

    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["artifact_path"] == str(artifact_dir / "policy.json")
    assert summary["estimator"] == "first_order"
    assert summary["n_rows_scored"] == 12
    assert summary["n_clients_per_bucket"] == 2
    assert summary["seed"] == 7
