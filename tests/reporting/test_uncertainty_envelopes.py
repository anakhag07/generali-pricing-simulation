from __future__ import annotations

import json

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pytest

from experiments.envelopes.core import aggregate_coverage
from experiments.envelopes.reporting import write_reports
from experiments.provenance import file_record


def _summary(tmp_path, *, empirical=False):
    points = tmp_path / "points.npz"
    np.savez(points, radius=[1., 2.], excess=[-.25, 1.], weights=[1., 3.])
    rows = [{"index": 0, "status": "violated", "metrics": {"radius_mean": 1.75},
             "artifacts": {"points": file_record(points)}}]
    summary = {
        "name": "uncertainty-envelopes",
        "target": {"kind": "real_line", "context_count": 3},
        "coverage": {"analytical": {"kind": "lower_bound", "value": .95},
                     "empirical": aggregate_coverage(rows, 1) if empirical else None},
        "repetitions": rows,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def _capture_figures(monkeypatch):
    saved = []
    original = Figure.savefig

    def capture(figure, *args, **kwargs):
        saved.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return saved


def test_replay_preserves_inputs_and_single_containment_is_not_probability(tmp_path, monkeypatch):
    _summary(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    figures = _capture_figures(monkeypatch)
    paths = write_reports(tmp_path)
    assert [path.name for path in paths] == ["01_coverage.pdf", "02_radius_excess.pdf"]
    assert all(path.read_bytes().startswith(b"%PDF-") for path in paths)
    assert all(path.read_bytes() == data for path, data in before.items())
    probability, counts = figures[0].axes
    assert [tick.get_text() for tick in probability.get_xticklabels()] == ["Analytical\nlower bound"]
    assert probability.lines[0].get_ydata().tolist() == [.95]
    assert [patch.get_height() for patch in counts.patches] == [0, 1, 0, 0, 0]
    assert all(ax.get_xlabel() and ax.get_ylabel() for fig in figures for ax in fig.axes)
    assert not list(tmp_path.rglob("*.png"))


def test_histograms_keep_signed_excess_and_give_each_fit_equal_mass(tmp_path, monkeypatch):
    summary = _summary(tmp_path)
    second = tmp_path / "second.npz"
    np.savez(second, radius=[3.], excess=[2.], weights=[9.])
    summary["repetitions"].append({"index": 1, "status": "unknown", "metrics": {"radius_mean": 3.},
                                   "artifacts": {"points": file_record(second)}})
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    calls = []
    original = Axes.hist

    def capture(ax, values, *args, **kwargs):
        calls.append((np.array(values), np.array(kwargs["weights"])))
        return original(ax, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", capture)
    write_reports(tmp_path)
    np.testing.assert_allclose(calls[0][1], [.125, .375, .5])
    np.testing.assert_allclose(calls[1][0], [-.25, 1., 2.])


def test_empirical_probability_is_separate_from_analytical_bound(tmp_path, monkeypatch):
    _summary(tmp_path, empirical=True)
    figures = _capture_figures(monkeypatch)
    write_reports(tmp_path)
    probability = figures[0].axes[0]
    assert [tick.get_text() for tick in probability.get_xticklabels()] == [
        "Analytical\nlower bound", "Empirical\nestimate"]
    assert probability.lines[0].get_ydata().tolist() == [.95]
    assert probability.lines[1].get_ydata()[0] == 0.
    assert probability.get_legend().get_texts()[0].get_text() == "95% Wilson interval"


def test_unresolved_empirical_results_and_missing_cases_remain_visible(tmp_path, monkeypatch):
    summary = _summary(tmp_path)
    summary["repetitions"].extend([
        {"index": 1, "status": "unknown", "metrics": None, "artifacts": {}},
        {"index": 2, "status": "fit_failed", "metrics": None, "artifacts": {}},
    ])
    summary["coverage"]["empirical"] = aggregate_coverage(summary["repetitions"], 4)
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    figures = _capture_figures(monkeypatch)
    write_reports(tmp_path)
    probability, counts = figures[0].axes
    assert [patch.get_height() for patch in counts.patches] == [0, 1, 1, 1, 1]
    assert "Empirical\nunavailable" in [tick.get_text() for tick in probability.get_xticklabels()]
    assert len(probability.lines) == 1
    assert "Conditional on 1/4" in figures[1]._suptitle.get_text()


def test_empty_metric_panels_are_labeled(tmp_path, monkeypatch):
    summary = _summary(tmp_path)
    summary["repetitions"] = [{"index": 0, "status": "fit_failed", "metrics": None, "artifacts": {}}]
    summary["coverage"]["analytical"] = None
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    figures = _capture_figures(monkeypatch)
    write_reports(tmp_path)
    assert all("no available evaluations" in ax.get_title() for ax in figures[1].axes)


def test_tampered_point_artifact_is_rejected_before_reports(tmp_path):
    _summary(tmp_path)
    np.savez(tmp_path / "points.npz", radius=[99.], excess=[99.], weights=[1.])
    with pytest.raises(ValueError, match="hash/size mismatch"):
        write_reports(tmp_path)
    assert not (tmp_path / "reports").exists()


def test_empirical_aggregate_must_match_saved_outcomes(tmp_path):
    summary = _summary(tmp_path, empirical=True)
    summary["coverage"]["empirical"]["counts"]["covered"] = 1
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="counts disagree"):
        write_reports(tmp_path)
