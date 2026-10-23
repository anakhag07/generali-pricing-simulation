"""Guard the statistical meanings of the wild-envelope comparison figures."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.envelopes import wild_reporting
from experiments.envelopes.core import wilson_interval


def test_finite_sample_terms_use_gaussian_fourth_moment_and_keep_vacuous_raw_bound():
    terms = wild_reporting._finite_sample_terms(5, .4419334613569795, 99, .05, .15, 1.)
    assert terms["K"]**4 == pytest.approx(3.)
    assert terms["C_p_kappa"] == pytest.approx(1866593.1182788361)
    assert terms["rho_n"] == pytest.approx(14548.852269906698)
    assert terms["tau_n"] == pytest.approx(7274.426134953349)
    assert terms["raw_lower_bound"] == pytest.approx(-21822.337904860047)
    assert terms["finite_B_correction"] == pytest.approx(.0095)
    assert terms["order_rank"] == 95
    assert terms["order_statistic_raw_lower_bound"] == pytest.approx(-21822.328404860047)
    assert terms["lower_bound"] == 0.


def test_finite_sample_terms_preserve_positive_bound_and_quantile_rounding():
    terms = wild_reporting._finite_sample_terms(1, 1e-30, 1, .05, 1., 1.)
    error = terms["rho_n"] + terms["tau_n"]
    assert terms["order_rank"] == 1
    assert terms["raw_lower_bound"] == pytest.approx(.475-error)
    assert terms["order_statistic_raw_lower_bound"] == pytest.approx(.5-error)
    assert 0 < terms["lower_bound"] == terms["raw_lower_bound"]
    with pytest.raises(ValueError, match="noise bounds"):
        wild_reporting._finite_sample_terms(1, .1, 99, .05, 0., 1.)


def test_exponential_bound_uses_its_fourth_moment():
    gaussian = wild_reporting._finite_sample_terms(5, .2, 99, .05, .15, 1.)
    exponential = wild_reporting._finite_sample_terms(5, .2, 99, .05, .15, 1., fourth_moment=9.)
    assert exponential["K"] == pytest.approx(np.sqrt(3))
    assert exponential["K"]**4 == pytest.approx(9.)
    assert exponential["rho_n"] > gaussian["rho_n"]
    assert exponential["raw_lower_bound"] < gaussian["raw_lower_bound"]


@pytest.mark.parametrize("event", ["auxiliary", "transferred"])
def test_distribution_comparison_uses_wild_counts_and_both_bounds(saved_report, event):
    baseline, _, _ = saved_report
    p = baseline["manifest"]
    pairs = set(zip(p["n_values"], p["B_values"]))
    baseline["metrics"] = [r for r in baseline["metrics"] if (r["n"], r["B"]) in pairs]
    variant = deepcopy(baseline)
    for row in variant["metrics"]:
        row[f"{event}_coverage"] -= .1
        row[f"{event}_lower"] -= .1
        row[f"{event}_upper"] -= .1
    bounds = [[{**r, "lower_bound": value, "raw_lower_bound": value}
               for r in baseline["metrics"] if r["method"] == "wild_bootstrap"] for value in (.4, .2)]
    fig = wild_reporting._distribution_coverage_figure(baseline, variant, "Exponential", event, bounds)
    try:
        for ax, delta in zip(fig.axes, p["deltas"]):
            assert len(ax.containers) == 2
            for container, summary in zip(ax.containers, (baseline, variant)):
                expected = wild_reporting._paired_rows(summary, delta, "wild_bootstrap")
                np.testing.assert_allclose(container.lines[0].get_ydata(), [r[f"{event}_coverage"] for r in expected])
            targets = [line for line in ax.lines if "bound (" in line.get_label()]
            assert len(targets) == 2
            np.testing.assert_allclose(targets[0].get_ydata(), .4)
            np.testing.assert_allclose(targets[1].get_ydata(), .2)
            assert all("OLS" not in label for label in ax.get_legend_handles_labels()[1])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("event", ["auxiliary", "transferred"])
def test_finite_sample_figure_keeps_wild_counts_and_omits_ols(saved_report, event):
    summary, _, _ = saved_report
    p = summary["manifest"]
    pairs = set(zip(p["n_values"], p["B_values"]))
    summary["metrics"] = [row for row in summary["metrics"] if (row["n"], row["B"]) in pairs]
    references = [{**row, "lower_bound": .3, "raw_lower_bound": .3}
                  for row in summary["metrics"] if row["method"] == "wild_bootstrap"]
    fig = wild_reporting._paired_metrics_figure(summary, event, finite_sample=references)
    try:
        for ax, delta in zip(fig.axes, p["deltas"]):
            assert len(ax.containers) == 1
            observed = ax.containers[0].lines[0]
            expected = wild_reporting._paired_rows(summary, delta, "wild_bootstrap")
            np.testing.assert_allclose(observed.get_ydata(), [r[f"{event}_coverage"] for r in expected])
            target = [line for line in ax.lines if line.get_label().startswith("Finite-sample")]
            assert len(target) == 1
            np.testing.assert_allclose(target[0].get_ydata(), [.3, .3])
            assert all("OLS" not in label and "Target" not in label for label in ax.get_legend_handles_labels()[1])
    finally:
        plt.close(fig)


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


@pytest.fixture
def saved_leverage_summaries(tmp_path):
    manifests = Path(__file__).parents[2] / "scratch/retired_experiments/wild_bootstrap_diagnostics/manifests"
    paths = []
    for index, label in enumerate(("0p5", "1", "3", "10")):
        manifest = json.loads((manifests / f"wild-bootstrap-leverage-bound-{label}.json").read_text())
        manifest.update(repetitions=8, repetitions_per_task=2)
        row = {
            **manifest["baseline"], "method": "wild_bootstrap", "repetitions": 8,
            "max_leverage": .015 + index * .005,
        }
        for event, count in (("auxiliary", 8-index), ("transferred", 8-max(0, index-1))):
            lower, upper = wilson_interval(count, 8)
            row.update({
                f"{event}_covered_count": count, f"{event}_coverage": count/8,
                f"{event}_lower": lower, f"{event}_upper": upper,
            })
        path = tmp_path / f"summary-{label}.json"
        path.write_text(json.dumps({"manifest": manifest, "metrics": [row]}))
        paths.append(path)
    return paths


def test_context_mean_radii_transfer_before_averaging():
    fit = np.array([[-2., 2.], [2., -2.]])
    example = {key: np.zeros_like(fit) for key in ("truth", "predictor", "residual_truth")}
    example["residual_fit"] = fit
    for method in ("analytical", "bootstrap"):
        example[f"{method}_width"] = np.ones_like(fit)
        example[f"{method}_radius_lower"] = np.maximum(1-fit, 0)
        example[f"{method}_radius_upper"] = np.maximum(1+fit, 0)
        example[f"{method}_lower"] = -example[f"{method}_radius_lower"]
        example[f"{method}_upper"] = example[f"{method}_radius_upper"]
    means = wild_reporting._mean_bands(example)
    np.testing.assert_array_equal(means["bootstrap_radius_lower"], [1.5, 1.5])
    np.testing.assert_array_equal(means["bootstrap_radius_upper"], [1.5, 1.5])
    np.testing.assert_array_equal(means["bootstrap_transferred_lower"], [-1.5, -1.5])
    np.testing.assert_array_equal(means["bootstrap_residual_lower"], [-1., -1.])


def test_normal_contextual_moved_point_report(tmp_path, saved_leverage_summaries):
    for index, path in enumerate(saved_leverage_summaries):
        summary = json.loads(path.read_text())
        summary["band_target"] = "residual"
        summary["manifest"]["design"] = {"distribution": "independent_standard_normal",
            "context_dim": 3, "action_override": {"index": 0, "value": [3, 10, 20, 40][index]}}
        path.write_text(json.dumps(summary))
    output = wild_reporting.write_leverage_report(saved_leverage_summaries, tmp_path / "normal")
    assert output.read_bytes().startswith(b"%PDF")
    provenance = json.loads((output.parent / "provenance.json").read_text())
    assert provenance["sweep_axis"] == "outlier_action"


@pytest.mark.parametrize("setting", ["n", "noise", "seeds"])
def test_leverage_report_rejects_unpaired_study_settings(tmp_path, saved_leverage_summaries, setting):
    path = saved_leverage_summaries[-1]
    summary = json.loads(path.read_text())
    manifest = summary["manifest"]
    if setting == "n":
        manifest["n_values"] = [301]
        manifest["baseline"]["n"] = summary["metrics"][0]["n"] = 301
    elif setting == "noise":
        manifest["noise"]["sigma_max"] = .75
    else:
        manifest["seeds"]["observations"] += 100
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="otherwise identical study manifests"):
        wild_reporting.write_leverage_report(saved_leverage_summaries, tmp_path / "report")


@pytest.mark.parametrize("event", ["auxiliary", "transferred"])
def test_leverage_report_rejects_coverage_inconsistent_with_saved_count(tmp_path, saved_leverage_summaries, event):
    path = saved_leverage_summaries[-1]
    summary = json.loads(path.read_text())
    summary["metrics"][0][f"{event}_coverage"] += .01
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Coverage estimate disagrees with saved count"):
        wild_reporting.write_leverage_report(saved_leverage_summaries, tmp_path / "report")


def test_leverage_report_preserves_counts_and_records_pdf_csv_provenance(tmp_path, saved_leverage_summaries):
    destination = tmp_path / "report"
    before = plt.get_fignums()
    output = wild_reporting.write_leverage_report(list(reversed(saved_leverage_summaries)), destination)
    assert output.suffix == ".pdf" and output.read_bytes().startswith(b"%PDF")
    assert plt.get_fignums() == before
    assert {path.name for path in destination.iterdir()} == {
        "coverage_vs_leverage.pdf", "coverage_vs_leverage.csv", "provenance.json",
    }
    with (destination / "coverage_vs_leverage.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["truncation_bound"]) for row in rows] == [.5, 1., 3., 10.]
    for path, row in zip(saved_leverage_summaries, rows):
        saved = json.loads(path.read_text())["metrics"][0]
        assert float(row["max_leverage"]) == saved["max_leverage"]
        for event in ("auxiliary", "transferred"):
            count = int(row[f"{event}_covered_count"])
            assert count == saved[f"{event}_covered_count"]
            assert float(row[f"{event}_coverage"]) == count/int(row["repetitions"])
            assert float(row[f"{event}_lower"]) == saved[f"{event}_lower"]
            assert float(row[f"{event}_upper"]) == saved[f"{event}_upper"]
    provenance = json.loads((destination / "provenance.json").read_text())
    assert {record["path"] for record in provenance["inputs"]} == {
        str(path.resolve()) for path in saved_leverage_summaries
    }
    assert {Path(record["path"]).name for record in provenance["outputs"]} == {
        "coverage_vs_leverage.pdf", "coverage_vs_leverage.csv",
    }
    for record in provenance["inputs"] + provenance["outputs"] + [provenance["source"]]:
        data = Path(record["path"]).read_bytes()
        assert record["bytes"] == len(data)
        assert record["sha256"] == hashlib.sha256(data).hexdigest()


def test_wide_bound_report_retains_all_conditions_when_points_coincide(tmp_path, saved_leverage_summaries):
    paths = [saved_leverage_summaries[index] for index in (0, 1, 3)]
    template = json.loads(paths[-1].read_text())
    for bound, label in ((100., "100"), (1000., "1000"), (None, "unbounded")):
        summary = deepcopy(template)
        summary["manifest"]["name"] = f"wild-bootstrap-leverage-{label}"
        summary["manifest"]["design"]["truncation_bound"] = bound
        path = tmp_path / f"summary-{label}.json"
        path.write_text(json.dumps(summary))
        paths.append(path)
    destination = tmp_path / "wide-report"
    wild_reporting.write_leverage_report(list(reversed(paths)), destination)
    with (destination / "coverage_vs_leverage.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["truncation_bound"]) if row["truncation_bound"] else None
            for row in rows] == [.5, 1., 10., 100., 1000., None]
    for row in rows[2:]:
        assert float(row["max_leverage"]) == template["metrics"][0]["max_leverage"]
        for event in ("auxiliary", "transferred"):
            assert int(row[f"{event}_covered_count"]) == template["metrics"][0][f"{event}_covered_count"]
    provenance = json.loads((destination / "provenance.json").read_text())
    assert {record["path"] for record in provenance["inputs"]} == {str(path.resolve()) for path in paths}
    assert provenance["coincident_points"] == [[.5], [1.], [10., 100., 1000., None]]


@pytest.fixture
def saved_outlier_summaries(tmp_path, saved_leverage_summaries):
    manifests = Path(__file__).parents[2] / "scratch/retired_experiments/wild_bootstrap_diagnostics/manifests"
    paths = []
    for index, value in enumerate((3, 10, 20, 40, 100, 1000)):
        template = saved_leverage_summaries[index % len(saved_leverage_summaries)]
        summary = json.loads(template.read_text())
        manifest = json.loads((manifests / f"wild-bootstrap-outlier-{value}.json").read_text())
        manifest.update(repetitions=8, repetitions_per_task=2)
        summary["manifest"] = manifest
        summary["metrics"][0]["max_leverage"] = (.03, .26, .58, .85, .97, .9997)[index]
        path = tmp_path / f"summary-outlier-{value}.json"
        path.write_text(json.dumps(summary))
        paths.append(path)
    return paths


def test_outlier_report_retains_fixed_bulk_and_sorted_conditions(tmp_path, saved_outlier_summaries):
    destination = tmp_path / "outlier-report"
    before = plt.get_fignums()
    output = wild_reporting.write_leverage_report(list(reversed(saved_outlier_summaries)), destination)
    assert output.read_bytes().startswith(b"%PDF")
    assert plt.get_fignums() == before
    with (destination / "coverage_vs_leverage.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["outlier_action"]) for row in rows] == [3., 10., 20., 40., 100., 1000.]
    assert [float(row["truncation_bound"]) for row in rows] == [3.] * 6
    for path, row in zip(saved_outlier_summaries, rows):
        saved = json.loads(path.read_text())["metrics"][0]
        assert float(row["max_leverage"]) == saved["max_leverage"]
        for event in ("auxiliary", "transferred"):
            assert int(row[f"{event}_covered_count"]) == saved[f"{event}_covered_count"]
            for statistic in ("coverage", "lower", "upper"):
                assert float(row[f"{event}_{statistic}"]) == saved[f"{event}_{statistic}"]
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["sweep_axis"] == "outlier_action"
    assert provenance["coincident_points"] == [[3.], [10.], [20.], [40.], [100.], [1000.]]
    assert {record["path"] for record in provenance["inputs"]} == {
        str(path.resolve()) for path in saved_outlier_summaries
    }


@pytest.mark.parametrize("setting", ["bulk_bound", "override_index"])
def test_outlier_report_rejects_changed_bulk_or_moved_row(tmp_path, saved_outlier_summaries, setting):
    path = saved_outlier_summaries[-1]
    summary = json.loads(path.read_text())
    design = summary["manifest"]["design"]
    if setting == "bulk_bound":
        design["truncation_bound"] = 10.
    else:
        design["action_override"]["index"] = 1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="otherwise identical study manifests"):
        wild_reporting.write_leverage_report(saved_outlier_summaries, tmp_path / "report")


def test_leverage_report_rejects_mixed_outlier_and_bound_sweeps(
        tmp_path, saved_outlier_summaries, saved_leverage_summaries):
    paths = [saved_leverage_summaries[0], *saved_outlier_summaries[:2]]
    with pytest.raises(ValueError, match="otherwise identical study manifests"):
        wild_reporting.write_leverage_report(paths, tmp_path / "report")


def test_residual_slices_use_saved_asymmetric_bounds(saved_report):
    summary, snapshot, example = saved_report
    summary['band_target'] = 'residual'
    example['residual_fit'] = example['auxiliary']-example['predictor']
    example['residual_truth'] = example['truth']-example['predictor']
    for method in ('analytical', 'bootstrap'):
        example[f'{method}_lower'] = np.minimum(example['predictor'], example['auxiliary']-example[f'{method}_width'])
        example[f'{method}_upper'] = np.maximum(example['predictor'], example['auxiliary']+example[f'{method}_width'])
    wild_reporting._validate_example(snapshot, example)
    fig = wild_reporting._slices_figure(summary, example, (0,), 'wild_bootstrap', 'Wild bootstrap')
    try:
        vertices = fig.axes[0].collections[0].get_paths()[0].vertices
        for action, lower, upper in zip(example['actions_grid'], example['bootstrap_lower'][0], example['bootstrap_upper'][0]):
            heights = vertices[vertices[:, 0] == action, 1]
            assert heights.min() == pytest.approx(lower)
            assert heights.max() == pytest.approx(upper)
    finally:
        plt.close(fig)


def test_paired_reports_reject_off_pair_rows(saved_report):
    summary, _, _ = saved_report
    summary['manifest']['sweep_mode'] = 'paired'
    with pytest.raises(ValueError, match='exactly the declared paired'):
        wild_reporting._paired_rows(summary, .05, 'wild_bootstrap')
    pairs = set(zip(summary['manifest']['n_values'], summary['manifest']['B_values']))
    summary['metrics'] = [r for r in summary['metrics'] if (r['n'], r['B']) in pairs]
    assert [(r['n'], r['B']) for r in wild_reporting._paired_rows(summary, .05, 'wild_bootstrap')] == [(3, 9), (6, 19)]
    fig = wild_reporting._paired_metrics_figure(summary, 'auxiliary')
    try:
        np.testing.assert_allclose(fig.axes[0].containers[1].lines[0].get_ydata(), [.75, .8])
        assert 'OLS variance bound' in fig.axes[0].get_legend_handles_labels()[1]
    finally:
        plt.close(fig)
