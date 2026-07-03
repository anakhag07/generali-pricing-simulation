from __future__ import annotations

from datetime import datetime
import json
from types import SimpleNamespace

import numpy as np

from experiments.config import CorrectnessSpec
from experiments.reporting.context import RunContext
import experiments.reporting.json_summary as json_summary
from experiments.reporting.json_summary import JsonReporter, build_summary_payload
from experiments.reporting.json_summary import _serialize_overrides
from experiments.seeds import SeedSetup


def _run_context(run_dir):
    return SimpleNamespace(run_dir=run_dir)


def test_json_reporter_defaults_to_summary_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(json_summary, "build_summary_payload", lambda ctx, result, **kwargs: {"ok": True})
    reporter = JsonReporter()

    reporter.on_end(_run_context(tmp_path), result=SimpleNamespace())

    assert (tmp_path / "summary.json").exists()
    assert json.loads((tmp_path / "summary.json").read_text()) == {"ok": True}


def test_json_reporter_writes_named_summary_in_variant_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(json_summary, "build_summary_payload", lambda ctx, result, **kwargs: {"seed": 7})
    variant_dir = tmp_path / "variant"
    seed_dir = variant_dir / "seeds" / "seed-7"
    seed_dir.mkdir(parents=True)
    reporter = JsonReporter(summary_name="summary-seed-7.json", summary_dir=variant_dir)

    reporter.on_end(_run_context(seed_dir), result=SimpleNamespace())

    assert (variant_dir / "summary-seed-7.json").exists()
    assert not (seed_dir / "summary-seed-7.json").exists()


def test_serialize_overrides_handles_non_json_values() -> None:
    serialized = _serialize_overrides(
        {
            "_run_name": "internal-name",
            "seed_setup": SeedSetup(run_seed=7, optimizer_seed=11),
            "correctness": CorrectnessSpec(gradient_source="numdiff", numdiff_bounds=(-1.0, 1.0)),
            "bounds": (-0.1, 0.2),
            "theta0": np.asarray([1.0, 2.0]),
            "object_value": object(),
        }
    )

    assert "_run_name" not in serialized
    assert serialized["seed_setup"]["run_seed"] == 7
    assert serialized["seed_setup"]["optimizer_seed"] == 11
    assert serialized["correctness"]["gradient_source"] == "numdiff"
    assert serialized["correctness"]["numdiff_bounds"] == [-1.0, 1.0]
    assert serialized["bounds"] == [-0.1, 0.2]
    assert serialized["theta0"] == [1.0, 2.0]
    assert isinstance(serialized["object_value"], str)


def test_summary_payload_includes_preset_metadata(tmp_path) -> None:
    run_context = RunContext(
        experiment_name="demo",
        run_id="rid",
        run_dir=tmp_path,
        plots_dir=tmp_path / "plots",
        started_at=datetime(2026, 1, 1),
        run_metadata={
            "preset_name": "planted_logistic_base",
            "variant_name": "sigma-0.1",
            "run_seed": 7,
            "overrides": {"sigma": 0.1, "_run_name": "internal"},
        },
    )
    config = SimpleNamespace(
        theta0=np.asarray([0.0]),
        train_fraction=1.0,
        test_fraction=0.0,
        objective=object(),
        to_dict=lambda: {},
    )
    result = SimpleNamespace(
        x_samples=np.zeros((2, 1)),
        x_test=None,
        train_indices=None,
        test_indices=None,
        config=config,
        results={},
        traces={},
        train_metrics={},
        test_metrics={},
        initial_value=0.0,
        initial_mean_acceptance=None,
        u_star=None,
        value_at_u_star=None,
        constant_u_baselines=[],
    )

    payload = build_summary_payload(run_context, result)

    assert payload["preset"] == {
        "preset_name": "planted_logistic_base",
        "variant_name": "sigma-0.1",
        "run_seed": 7,
        "overrides": {"sigma": 0.1},
    }
