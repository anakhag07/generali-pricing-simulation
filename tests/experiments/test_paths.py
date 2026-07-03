from __future__ import annotations

from pathlib import Path

from experiments.paths import results_root


def test_results_root_honors_env_override(monkeypatch, tmp_path) -> None:
    override = tmp_path / "custom-results"
    monkeypatch.setenv("GENERALI_RESULTS_ROOT", str(override))

    assert results_root() == override.resolve()


def test_results_root_default_is_external_and_not_cwd(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GENERALI_RESULTS_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    assert results_root() == (Path.home() / "projects" / "generali-pricing" / "results").resolve()
