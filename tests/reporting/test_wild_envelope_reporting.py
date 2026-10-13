"""Guard the statistical meanings of the wild-envelope comparison figures."""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.envelopes import wild_reporting


@pytest.fixture
def saved_report():
    manifest = {
        "baseline": {"n": 6, "B": 19, "delta": .05},
        "n_values": [3, 6], "B_values": [9, 19], "deltas": [.1, .05],
        "repetitions": 100, "noise": {"sigma_max": 1.},
    }
    rows = []
    for n in manifest["n_values"]:
        for B in manifest["B_values"]:
            for delta in manifest["deltas"]:
                for method in ("analytical", "wild_bootstrap"):
                    coverage = 1. if method == "analytical" else .7+n/60
                    rows.append({
                        "n": n, "B": B, "delta": delta, "method": method,
                        "auxiliary_coverage": coverage,
                        "auxiliary_lower": coverage-.05,
                        "auxiliary_upper": min(1., coverage+.05),
                        "transferred_coverage": 1., "transferred_lower": .96,
                        "transferred_upper": 1.,
                        "mean_width": 1/np.sqrt(n), "mean_radius": 100.,
                    })
    X = np.arange(6.)[:, None]
    a = np.linspace(-2., 2., 9)
    truth = X+a[None, :]
    example = {
        "X": X.copy(), "actions_grid": a, "truth": truth,
        "predictor": truth+2, "auxiliary": truth+.1,
        "analytical_width": np.ones_like(truth),
        "bootstrap_width": np.full_like(truth, .5),
        "sigma_grid": np.full_like(truth, .8),
    }
    return {"manifest": manifest, "metrics": rows}, {"X": X}, example


def test_auxiliary_coverage_is_not_replaced_by_transferred_coverage(saved_report):
    summary, _, _ = saved_report
    fig = wild_reporting._coverage_figure(summary, "n", "auxiliary")
    try:
        line = fig.axes[0].containers[1].lines[0]
        np.testing.assert_allclose(line.get_ydata(), [.75, .8])
        assert "all calibration contexts" in fig._supxlabel.get_text()
    finally:
        plt.close(fig)


def test_width_figure_uses_auxiliary_width_not_transferred_radius(saved_report):
    summary, _, example = saved_report
    fig = wild_reporting._width_figure(summary, example)
    try:
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), 1/np.sqrt([3, 6]))
        assert len(fig.axes[0].lines) == 2
    finally:
        plt.close(fig)


def test_shared_width_axis_contains_curves_for_every_delta(saved_report):
    summary, _, example = saved_report
    # A smaller delta produces much wider bands in later shared-axis panels.
    for row in summary["metrics"]:
        if row["delta"] == .05:
            row["mean_width"] *= 10
    fig = wild_reporting._width_figure(summary, example)
    try:
        for ax in fig.axes:
            lower, upper = ax.get_ylim()
            for line in ax.lines:
                values = line.get_ydata()
                assert np.all(values >= lower)
                assert np.all(values <= upper)
    finally:
        plt.close(fig)


def test_missing_and_duplicate_sweep_points_are_rejected(saved_report):
    summary, _, _ = saved_report
    broken = deepcopy(summary)
    broken["metrics"].pop()
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        wild_reporting._rows(broken, "n", .05, "wild_bootstrap")
    summary["metrics"].append(summary["metrics"][-1])
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        wild_reporting._rows(summary, "n", .05, "wild_bootstrap")


def test_example_must_use_original_saved_contexts(saved_report):
    _, snapshot, example = saved_report
    example["X"][0, 0] = 99
    with pytest.raises(ValueError, match="saved calibration contexts"):
        wild_reporting._validate_example(snapshot, example)


def test_all_reports_are_pdfs_and_leave_no_open_figures(tmp_path, saved_report):
    summary, snapshot, example = saved_report
    before = plt.get_fignums()
    paths = wild_reporting.write_reports(summary, snapshot, example, tmp_path)
    assert len(paths) == 7
    assert all(path.suffix == ".pdf" and path.read_bytes().startswith(b"%PDF") for path in paths)
    assert len(list(tmp_path.iterdir())) == 7
    assert plt.get_fignums() == before
