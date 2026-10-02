"""Gaussian-support envelope formula, certification, pairing, and artifacts."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import coverage_kernel_sweep as kernel
from experiments.bootstrap_band_sweep import dataset_streams as bootstrap_streams


MANIFEST = Path(__file__).parents[2]/"manifests"/"coverage_kernel_dense_n_sweep.json"


def test_kernel_formula_and_gradient():
    actions = np.array([-0.3, 0.1, 0.7])
    envelope = kernel.GaussianSupportEnvelope([0, 5, -5], actions, 0.1, 1.96)
    a = 0.25
    expected = np.sum(np.exp(-0.5*((actions-a)/0.1)**2))
    assert envelope.support(a) == pytest.approx(expected)
    assert envelope.radius(a) == pytest.approx(1.96/np.sqrt(expected))
    assert envelope.radius(2.0) > envelope.radius(a)
    for a in [-0.2, 0.2, 0.5, 1.1]:
        h = 1e-6
        derivative = (envelope.value(np.array([a+h]), None)-envelope.value(np.array([a-h]), None))/(2*h)
        assert envelope.grad(np.array([a]), None)[0] == pytest.approx(derivative, rel=1e-6, abs=1e-5)


def test_two_sided_real_line_coverage_and_witness():
    actions = np.array([-1., 0., 1.])
    covered = kernel.check_two_sided_coverage(actions, 0.1, np.zeros(3), 1.)
    assert covered["state"] == "covered"
    missed = kernel.check_two_sided_coverage(actions, 0.1, np.array([2., 0., 0.]), 1.)
    assert missed["state"] == "uncovered"
    assert missed["witness_statistic"] > 1.
    unresolved = kernel.check_two_sided_coverage(actions, 0.1, np.array([0.1, 0., 0.]), 1., max_intervals=1)
    assert unresolved["state"] in {"covered", "indeterminate"}


def test_dataset_streams_match_dense_bootstrap_prefix():
    payload = kernel.load_manifest(MANIFEST).payload
    bootstrap_payload = json.loads((MANIFEST.parent/"bootstrap_ols_dense_n_sweep.json").read_text())
    for index in (0, 3):
        seeds, x, error = kernel.dataset_streams(payload, index)
        old_seeds, old_x, old_error, _ = bootstrap_streams(bootstrap_payload, index)
        assert seeds == {key: old_seeds[key] for key in seeds}
        np.testing.assert_array_equal(x, old_x)
        np.testing.assert_array_equal(error, old_error)


def test_optimizer_action_is_solver_output_and_can_be_rejected(monkeypatch):
    payload = kernel.load_manifest(MANIFEST).payload
    config = {**payload["optimizer"], "starts": [0.0], "global_gap_tolerance": 1e-4}
    calls = []

    def fake_solver(theta, samples, objective, **kwargs):
        calls.append(float(theta[0]))
        return np.array([0.]), SimpleNamespace(optimizer_success=True,
                                               optimizer_status=0, optimizer_message="test")

    monkeypatch.setattr(kernel, "run_first_order_minimize", fake_solver)
    result = kernel.optimize_kernel_lcb(np.array([0., 5., -5.]), np.array([-1., 0., 1.]), 0.1, 1., config)
    assert calls == [0.]
    assert result["action"] == 0.
    assert result["optimization_state"] == "uncertified"


def test_small_end_to_end_sweep_and_pdf_outputs(tmp_path):
    payload = json.loads(MANIFEST.read_text())
    payload.update(name="kernel-test", datasets=2, N_values=[25, 50])
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = kernel.load_manifest(path)
    subprocess.run([sys.executable, "scripts/run_experiment_manifest.py", str(path),
                    "--runs-root", str(tmp_path), "--launch", "local"],
                   cwd=Path(__file__).parents[2], check=True)
    project = tmp_path/manifest.name
    for index in range(2):
        saved = json.loads((project/"datasets"/f"dataset-{index:03d}"/"summary.json").read_text())
        assert len(saved["rows"]) == 2
        assert all(row["coverage_state"] in {"covered", "uncovered", "indeterminate"} for row in saved["rows"])
        assert all(row["optimization_state"] == "certified" for row in saved["rows"])
    summary = json.loads((project/"summary.json").read_text())
    assert len(summary["metrics"]) == 2
    assert len(summary["plots"]) == 3
    for pdf in (project/"plots").glob("*.pdf"):
        assert pdf.read_bytes().startswith(b"%PDF-")


def test_manifest_rejects_bandwidth_change(tmp_path):
    payload = json.loads(MANIFEST.read_text())
    payload["bandwidth"] = 0.2
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bandwidth"):
        kernel.load_manifest(path)
