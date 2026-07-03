from __future__ import annotations

import json
from types import SimpleNamespace

import experiments.reporting.json_summary as json_summary
from experiments.reporting.json_summary import JsonReporter


def _run_context(run_dir):
    return SimpleNamespace(run_dir=run_dir)


def test_json_reporter_defaults_to_summary_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        json_summary,
        "build_summary_payload",
        lambda ctx, result, summary_dir=None: {"ok": True},
    )
    reporter = JsonReporter()

    reporter.on_end(_run_context(tmp_path), result=SimpleNamespace())

    assert (tmp_path / "summary.json").exists()
    assert json.loads((tmp_path / "summary.json").read_text()) == {"ok": True}


def test_json_reporter_writes_named_summary_in_variant_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        json_summary,
        "build_summary_payload",
        lambda ctx, result, summary_dir=None: {"seed": 7},
    )
    variant_dir = tmp_path / "variant"
    seed_dir = variant_dir / "seeds" / "seed-7"
    seed_dir.mkdir(parents=True)
    reporter = JsonReporter(summary_name="summary-seed-7.json", summary_dir=variant_dir)

    reporter.on_end(_run_context(seed_dir), result=SimpleNamespace())

    assert (variant_dir / "summary-seed-7.json").exists()
    assert not (seed_dir / "summary-seed-7.json").exists()
