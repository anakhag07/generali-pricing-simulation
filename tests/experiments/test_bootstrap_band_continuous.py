from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from experiments.bootstrap_band import bootstrap_band, evaluate_band, features
from experiments.bootstrap_band_continuous import (
    VARIABLE, QuadraticErrorCertificate, nonnegative_on_real_line, replay_dataset,
)


@pytest.mark.parametrize("expression,expected", [
    (sp.Integer(0), True), (sp.Integer(1), True), (sp.Integer(-1), False),
    (VARIABLE, False), (VARIABLE**2, True), ((VARIABLE-1)**4, True),
    ((VARIABLE**2-1)**2, True), (VARIABLE**4-1, False),
    (-(VARIABLE**2+1), False), (VARIABLE**4+VARIABLE**2+1, True),
    ((VARIABLE-2)**2-sp.Rational(1, 10**16), False),
])
def test_exact_nonnegativity_including_repeated_roots_and_narrow_failure(expression, expected):
    assert nonnegative_on_real_line(sp.Poly(expression, VARIABLE, domain=sp.QQ)) is expected


@pytest.mark.parametrize("difference,expected", [
    ([0, 0, 0], 0), ([1, 0, 0], 1), ([0, 0, 1], 1), ([0, 1, 0], 1/np.sqrt(3)),
])
def test_known_all_real_statistics_include_tail_only_supremum(difference, expected):
    certificate = QuadraticErrorCertificate(np.eye(3), 1)
    d = certificate.difference(np.array(difference), np.zeros(3))
    low, high = certificate.supremum_interval(d, 1e-8)
    assert low <= expected + 1e-14 <= high + 1e-14
    assert high-low <= 2.1e-8*max(1, high)
    assert certificate.contains(d, high)
    if low > 0:
        assert not certificate.contains(d, low)


def test_tail_failure_cannot_be_hidden_by_finite_display_window():
    certificate = QuadraticErrorCertificate(np.eye(3), 1)
    d = certificate.difference(np.array([0, 0, 1]), np.zeros(3))
    assert not certificate.contains(d, .8)
    assert certificate.contains(d, 1)
    assert certificate.contains(d, 1.1)


def test_supremum_bracket_survives_bad_numerical_root_proposal(monkeypatch):
    certificate = QuadraticErrorCertificate(np.eye(3), 1)
    d = certificate.difference(np.array([0, 1, 0]), np.zeros(3))
    monkeypatch.setattr(np.polynomial.polynomial, "polyroots", lambda _: np.array([]))
    low, high = certificate.supremum_interval(d, 1e-8)
    assert low <= 1/np.sqrt(3) <= high


def test_replay_preserves_draws_verifies_identity_and_does_not_calibrate_with_truth():
    rng = np.random.default_rng(36)
    x = rng.normal(size=40)
    beta0 = np.array([0, 5, -5])
    y = features(x) @ beta0 + rng.normal(size=len(x))
    saved = bootstrap_band(x, y, np.linspace(0, 1, 51), draws=29, delta=.05, bootstrap_seed=41)
    saved.update(evaluate_band(saved, features(saved["grid"]) @ beta0))
    first = replay_dataset(saved, beta0, .05, 1e-8)
    second = replay_dataset(saved, beta0+[.1, 0, 0], .05, 1e-8)
    np.testing.assert_array_equal(first["bootstrap_beta"], saved["bootstrap_beta"])
    np.testing.assert_array_equal(first["bootstrap_supremum_upper"], second["bootstrap_supremum_upper"])
    assert first["critical_value"] == second["critical_value"]
    assert first["identity_coefficient_error"] < 1e-12
    assert first["critical_value"] >= saved["critical_value"] - 1e-12
    assert np.all(first["bootstrap_supremum_upper"] >= saved["bootstrap_maxima"]-1e-12)
    certificate = QuadraticErrorCertificate(first["covariance"], first["sigma_hat"])
    d = certificate.difference(first["beta_hat"], beta0)
    assert first["simultaneous_covered"] == certificate.contains(d, first["critical_value"])


def test_replay_manifest_outputs_exact_certificates_and_mathematical_pdfs(tmp_path):
    import json
    from pathlib import Path
    from experiments.bootstrap_band import load_bootstrap_band_manifest, run_case, collect_outputs
    from experiments.bootstrap_band_continuous import load_manifest, run_replay_case, source_cases, collect

    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "manifests/bootstrap_ols_grid_band.json").read_text())
    payload["stage2"] = {"sample_sizes": [100], "noise_stds": [.5, 1, 2], "datasets": 2}
    payload["bootstrap"]["draws"] = 9
    payload["grid"]["count"] = 11
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(json.dumps(payload))
    source_spec = load_bootstrap_band_manifest(source_manifest)
    for index in range(len(source_spec.cases())):
        run_case(source_spec, index, runs_root=tmp_path, force=False)
    collect_outputs(source_spec, runs_root=tmp_path)
    replay = load_manifest(root / "manifests/bootstrap_ols_continuous_replay.json")
    for source in source_cases(replay, tmp_path):
        run_replay_case(replay, source, tmp_path, False)
    collect(replay, tmp_path)
    project = tmp_path / replay.name
    summary = json.loads((project / "summary.json").read_text())
    assert len(summary["plots"]) == 4
    assert (project / "bootstrap_coefficients.csv").exists()
    for record in summary["plots"]:
        assert Path(record["path"]).read_bytes().startswith(b"%PDF-")
    for source in source_cases(replay, tmp_path):
        with np.load(project / source.name / "draws.npz") as archive:
            for index, coefficients in enumerate(archive["containment_exact_coefficients"]):
                poly = sp.Poly.from_list([sp.Rational(c) for c in coefficients[::-1]], VARIABLE, domain=sp.QQ)
                assert nonnegative_on_real_line(poly) == archive["simultaneous_covered"][index]
            assert np.max(archive["bootstrap_supremum_upper"]-archive["bootstrap_supremum_lower"]) < 1e-6
    assert not list(project.rglob("*.png"))
