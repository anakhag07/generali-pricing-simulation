from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import experiments.finite_policy_lcb as lcb


def _spec() -> lcb.FinitePolicyLCBSpec:
    return lcb.FinitePolicyLCBSpec(
        policies=(0.0, 0.5, 1.0),
        deltas=(0.2, 0.05),
        epsilon=0.0,
        master_noise_seed=20260807,
        run_seeds=(101, 102),
    )


def _manifest_payload() -> dict[str, object]:
    return {
        "kind": "finite_policy_lcb",
        "name": "lcb-test",
        "policies": [0.0, 0.5, 1.0],
        "true_value": {"type": "identity"},
        "surrogate": {"type": "policy_scaled_gaussian"},
        "deltas": [0.2, 0.05],
        "epsilon": 0.0,
        "seeds": {"master_noise_seed": 20260807, "run_seeds": [101, 102]},
        "launch": {"mode": "local", "array": "seed"},
    }


def test_formula_matches_value_surrogate_envelope_and_lcb() -> None:
    spec = _spec()
    z = np.asarray([8.0, 0.25, -0.5])

    result = lcb.evaluate_finite_policy_lcb_draw(
        spec,
        run_seed=101,
        noise_seed=17,
        z=z,
    )
    rows = [row for row in result.policy_results if row.delta == 0.2]
    quantile = lcb.lcb_quantile(0.2, 3)

    assert rows[0].true_value == 0.0
    assert rows[0].surrogate_value == 0.0
    assert rows[0].uncertainty_width == 0.0
    assert rows[0].lcb == 0.0
    assert rows[1].true_value == 0.5
    assert rows[1].surrogate_value == pytest.approx(0.5 + 0.5 * 0.25)
    assert rows[1].uncertainty_width == pytest.approx(2.0 * 0.5 * quantile)
    assert rows[1].lcb == pytest.approx(0.5 + 0.5 * 0.25 - 0.5 * quantile)


def test_noise_vector_is_paired_across_deltas_and_distinct_across_seeds() -> None:
    spec = _spec()

    first = lcb.evaluate_finite_policy_lcb_seed(spec, 101)
    replay = lcb.evaluate_finite_policy_lcb_seed(spec, 101)
    second = lcb.evaluate_finite_policy_lcb_seed(spec, 102)

    assert first == replay
    assert first.noise_seed != second.noise_seed
    assert first.z != second.z
    for policy in spec.policies:
        rows = [row for row in first.policy_results if row.policy == policy]
        assert len({row.z for row in rows}) == 1
        assert len({row.surrogate_value for row in rows}) == 1


def test_paired_selection_never_increases_as_confidence_strengthens() -> None:
    spec = _spec()

    for run_seed in spec.run_seeds:
        selected = [
            row.selected_policy
            for row in lcb.evaluate_finite_policy_lcb_seed(spec, run_seed).selections
        ]
        assert np.all(np.diff(selected) <= 0.0)


def test_exact_selector_uses_smallest_argmax_and_has_zero_gap() -> None:
    spec = _spec()
    quantile = lcb.lcb_quantile(spec.deltas[0], len(spec.policies))
    # Every policy has LCB zero; ascending order plus np.argmax chooses pi=0.
    z = [0.0, quantile - 1.0, quantile - 1.0]

    result = lcb.evaluate_finite_policy_lcb_draw(
        spec,
        run_seed=101,
        noise_seed=17,
        z=z,
    )

    assert result.selections[0].selected_policy == 0.0
    assert result.selections[0].lcb_gap == pytest.approx(0.0, abs=1e-15)
    assert result.selections[0].epsilon == 0.0


def test_oracle_inequality_holds_for_every_comparator_on_confidence_event() -> None:
    spec = _spec()
    result = lcb.evaluate_finite_policy_lcb_draw(
        spec,
        run_seed=101,
        noise_seed=17,
        z=np.zeros(3),
    )

    assert all(selection.simultaneous_coverage for selection in result.selections)
    assert all(not selection.oracle_violation for selection in result.selections)
    assert all(selection.worst_oracle_slack >= -1e-12 for selection in result.selections)


def test_analytic_joint_coverage_dominates_nominal_coverage() -> None:
    for delta in (0.5, 0.2, 0.1, 0.05, 0.01):
        exact = lcb.analytic_joint_coverage(delta, 11)
        assert exact == pytest.approx((1.0 - delta / 11.0) ** 11)
        assert exact >= 1.0 - delta


def test_wilson_interval_contains_boundary_empirical_rates() -> None:
    for successes in (0, 25):
        low, high = lcb._wilson_interval(successes, 25)
        empirical = successes / 25
        assert 0.0 <= low <= empirical <= high <= 1.0


def test_gaussian_surrogate_moments_match_construction() -> None:
    rng = np.random.default_rng(7)
    policies = np.asarray([0.0, 0.5, 1.0])
    surrogate = policies + policies * rng.normal(size=(50_000, policies.size))

    assert np.mean(surrogate, axis=0) == pytest.approx(policies, abs=0.012)
    assert np.var(surrogate, axis=0) == pytest.approx(policies**2, abs=0.015)


def test_manifest_parser_validates_exact_construction() -> None:
    manifest = lcb.parse_finite_policy_lcb_manifest(_manifest_payload())

    assert manifest.name == "lcb-test"
    assert manifest.spec == _spec()
    assert manifest.launch.mode == "local"
    assert manifest.launch.array == "seed"

    payload = _manifest_payload()
    payload["epsilon"] = 0.1
    with pytest.raises(ValueError, match="epsilon=0"):
        lcb.parse_finite_policy_lcb_manifest(payload)

    payload = _manifest_payload()
    payload["surrogate"] = {"type": "other"}
    with pytest.raises(ValueError, match="surrogate.type"):
        lcb.parse_finite_policy_lcb_manifest(payload)


def test_seed_execution_completion_and_collection(monkeypatch, tmp_path: Path) -> None:
    manifest = lcb.parse_finite_policy_lcb_manifest(_manifest_payload())
    monkeypatch.setattr(lcb, "_write_plots", lambda *args, **kwargs: None)

    first = lcb.run_finite_policy_lcb_manifest_seed(manifest, 0, runs_root=tmp_path)
    skipped = lcb.run_finite_policy_lcb_manifest_seed(manifest, 0, runs_root=tmp_path)
    lcb.run_finite_policy_lcb_manifest_seed(manifest, 1, runs_root=tmp_path)
    collected = lcb.collect_finite_policy_lcb_outputs(manifest, runs_root=tmp_path)

    assert first["skipped"] is False
    assert skipped["skipped"] is True
    assert lcb.finite_policy_lcb_seed_complete(manifest, 101, runs_root=tmp_path)
    assert collected["n_policy_rows"] == 2 * 2 * 3
    assert collected["n_selection_rows"] == 2 * 2
    project_dir = manifest.project_dir(tmp_path)
    assert (project_dir / "EXPERIMENT.md").exists()
    for filename in (
        "seed_policy_values.csv",
        "seed_selections.csv",
        "policy_summary.csv",
        "coverage_summary.csv",
        "oracle_summary.csv",
    ):
        assert (project_dir / filename).exists()
    with (project_dir / "seed_policy_values.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 12
