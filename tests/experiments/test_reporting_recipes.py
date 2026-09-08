from __future__ import annotations

from pathlib import Path

import experiments.reporting.recipes as recipes
from experiments.reporting.recipes import parse_manifest_reports, run_manifest_reports


def test_report_recipe_is_completion_aware(monkeypatch, tmp_path) -> None:
    calls: list[Path] = []

    def fake_runner(output_dir, options):
        del options
        calls.append(output_dir)
        output_dir.mkdir(parents=True)
        provenance = output_dir / "provenance.json"
        provenance.write_text("{}\n", encoding="utf-8")
        return [provenance]

    monkeypatch.setitem(
        recipes._REPORT_RUNNERS,
        "real_data_profit_dispersion",
        fake_runner,
    )
    reports = parse_manifest_reports(
        [{"name": "comparison", "recipe": "real_data_profit_dispersion"}]
    )

    first = run_manifest_reports(reports, project_dir=tmp_path)
    second = run_manifest_reports(reports, project_dir=tmp_path)

    assert calls == [tmp_path / "reports" / "comparison"]
    assert first[0]["skipped"] is False
    assert second[0]["skipped"] is True
