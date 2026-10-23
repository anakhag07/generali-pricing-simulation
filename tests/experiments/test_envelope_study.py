import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from scratch.retired_experiments.studies.envelopes import study


@pytest.fixture
def payload():
    path = Path(__file__).parents[2]/"scratch/retired_experiments/uncertainty_envelopes/manifests/uncertainty-envelopes.json"
    result = json.loads(path.read_text())
    result["calibration"].update(n=8, context_dim=1)
    result["problem"]["truth"]["coefficients"] = [0., .5, 1.]
    result["problem"]["predictor"]["coefficients"] = [.2, .4, .8]
    result["tightness"]["actions"]["count"] = 5
    return result


def manifest(tmp_path, payload):
    path = tmp_path/f"{payload['name']}.json"
    path.write_text(json.dumps(payload))
    return study.load_manifest(path)


def make_empirical(payload, repetitions=3):
    payload["coverage"].update(mode="empirical", repetitions=repetitions, redraw="responses")


def custom_adapter(tmp_path, monkeypatch):
    name = "envelope_fixture_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    (tmp_path/f"{name}.py").write_text('''
import numpy as np
from experiments.envelopes.core import FittedEnvelope

TRUTH_SHIFT = 0.

def nonlinear_truth(options):
    scale = options.get("scale", 1.0)
    shift = TRUTH_SHIFT
    return lambda X, actions: scale * np.sin(actions) + X[:, 0] ** 2 + shift

def observed_radius(data, predictor, target, settings, rng):
    assert set(vars(data)) == {"X", "actions", "y"}
    assert not hasattr(data, "truth")
    radius = float(np.sqrt(np.mean((data.y - predictor(data.X, data.actions)) ** 2)))
    radius += settings["options"].get("padding", 0.)
    return FittedEnvelope(
        predictor, lambda X, actions: np.full(len(actions), radius),
        state={"radius": radius, "response_sum": float(data.y.sum())},
        metadata={"construction": "observed_radius"},
    )

def failing_radius(data, predictor, target, settings, rng):
    raise RuntimeError("deliberately failed calibration")
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


def test_analytical_launch_fits_once_and_replay_does_not_refit(tmp_path, payload, monkeypatch):
    count = []
    original = study.fit_linear_transfer

    def observed_fit(*args):
        count.append(args[0])
        return original(*args)

    monkeypatch.setattr(study, "fit_linear_transfer", observed_fit)
    spec = manifest(tmp_path, payload)
    result = study.run_study(spec, tmp_path/"runs", plots=False)
    assert len(count) == 1
    assert result["requested_repetitions"] == 1
    assert result["complete"] and result["successful"]
    assert result["coverage"]["empirical"] is None
    assert result["coverage"]["analytical"]["kind"] == "lower_bound"
    assert result["coverage"]["analytical"]["value"] == pytest.approx(.95)
    row = result["repetitions"][0]
    assert row["status"] == "covered"
    assert row["containment"]["domain"] == "real_line"
    assert row["metrics"]["excess_mean"] >= 0
    assert set(row["artifacts"]) == {"calibration", "fit", "points"}
    repeated = study.run_study(spec, tmp_path/"runs", plots=False)
    assert len(count) == 1
    assert repeated["repetitions"] == result["repetitions"]
    assert list((tmp_path/"runs"/spec.name).glob("*.pdf")) == []


def test_empirical_launch_freezes_design_and_refits_each_new_response_vector(tmp_path, payload, monkeypatch):
    make_empirical(payload)
    calls = []
    original = study.fit_linear_transfer

    def observed_fit(data, *args):
        calls.append(data)
        return original(data, *args)

    monkeypatch.setattr(study, "fit_linear_transfer", observed_fit)
    spec = manifest(tmp_path, payload)
    result = study.run_study(spec, tmp_path/"runs", plots=False)
    assert len(calls) == 3
    for data in calls[1:]:
        np.testing.assert_array_equal(data.X, calls[0].X)
        np.testing.assert_array_equal(data.actions, calls[0].actions)
        assert not np.array_equal(data.y, calls[0].y)
    rows = result["repetitions"]
    assert len({row["seeds"]["observations"] for row in rows}) == 3
    assert len({row["seeds"]["method"] for row in rows}) == 3
    assert all(row["seeds"]["method"] != row["seeds"]["observations"] for row in rows)
    states = [json.loads(Path(row["artifacts"]["fit"]["path"]).read_text())["state"] for row in rows]
    assert states[0]["beta"] != states[1]["beta"]
    np.testing.assert_array_equal(states[0]["covariance"], states[1]["covariance"])
    empirical = result["coverage"]["empirical"]
    assert empirical["requested"] == 3
    assert empirical["counts"]["fit_failed"] == 0
    assert empirical["estimate"] is not None
    assert empirical["interval_method"] == "Wilson"


def test_diagnostic_resolution_does_not_change_fit_or_continuum_coverage(tmp_path, payload):
    original = study.run_study(manifest(tmp_path, payload), tmp_path/"runs", plots=False)
    payload["name"] += "-dense-diagnostics"
    payload["tightness"]["actions"]["count"] = 47
    denser = study.run_study(manifest(tmp_path, payload), tmp_path/"runs", plots=False)
    first, second = original["repetitions"][0], denser["repetitions"][0]
    assert first["containment"] == second["containment"]
    assert first["analytical"] == second["analytical"]
    assert first["seeds"] == second["seeds"]
    first_state = json.loads(Path(first["artifacts"]["fit"]["path"]).read_text())
    second_state = json.loads(Path(second["artifacts"]["fit"]["path"]).read_text())
    assert first_state == second_state
    with np.load(first["artifacts"]["points"]["path"]) as first_points:
        with np.load(second["artifacts"]["points"]["path"]) as second_points:
            assert len(first_points["actions"]) == 8*5
            assert len(second_points["actions"]) == 8*47


def test_replay_rejects_changed_saved_artifact(tmp_path, payload):
    spec = manifest(tmp_path, payload)
    result = study.run_study(spec, tmp_path/"runs", plots=False)
    fit_path = Path(result["repetitions"][0]["artifacts"]["fit"]["path"])
    fit_path.write_text(fit_path.read_text()+"\n")
    with pytest.raises(ValueError, match="Saved artifact changed"):
        study.collect(spec, tmp_path/"runs", plots=False)


@pytest.mark.parametrize("operation", ["resume", "collect"])
def test_changed_saved_coverage_decision_is_rejected(tmp_path, payload, operation):
    spec = manifest(tmp_path, payload)
    study.run_study(spec, tmp_path/"runs", plots=False)
    path = tmp_path/"runs"/spec.name/"repetitions"/"000000"/"summary.json"
    row = json.loads(path.read_text())
    row["status"] = row["containment"]["status"] = "violated"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Saved artifact changed"):
        if operation == "resume":
            study.run_repetition(spec, 0, tmp_path/"runs")
        else:
            study.collect(spec, tmp_path/"runs", plots=False)


def test_changed_prepared_model_is_rejected_without_source_or_config_change(tmp_path, payload, monkeypatch):
    module_name = custom_adapter(tmp_path, monkeypatch)
    payload["problem"]["truth"] = {"type": "python", "callable": f"{module_name}:nonlinear_truth"}
    payload["coverage"].update(procedure="none", containment_checker="finite_evaluation")
    payload["target"]["actions"] = {"type": "finite_set", "values": [-1., 0., 1.]}
    spec = manifest(tmp_path, payload)
    study.run_study(spec, tmp_path/"runs", plots=False)
    # Each returned model is deterministic, but an external state change alters
    # the next prepared model without changing its source file or options.
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "TRUTH_SHIFT", 1.)
    with pytest.raises(ValueError, match="inputs/configuration/source changed"):
        study.run_repetition(spec, 0, tmp_path/"runs")


def test_changed_numeric_runtime_is_rejected_on_reuse(tmp_path, payload, monkeypatch):
    spec = manifest(tmp_path, payload)
    study.run_study(spec, tmp_path/"runs", plots=False)
    real_version = study.version
    monkeypatch.setattr(study, "version", lambda name: "different-runtime" if name == "numpy" else real_version(name))
    with pytest.raises(ValueError, match="inputs/configuration/source changed"):
        study.collect(spec, tmp_path/"runs", plots=False)


def test_saved_calibration_uses_saved_responses_only_for_analytical_mode(tmp_path, payload):
    X = np.array([[-1.], [-1.], [1.], [1.]])
    actions = np.array([-1., 1., -1., 1.])
    saved_y = np.array([101., 102., 103., 104.])
    path = tmp_path/"observations.npz"
    np.savez_compressed(path, X=X, actions=actions, y=saved_y)
    payload["calibration"] = {
        "type": "saved", "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "noise": {"type": "gaussian", "std": 1.},
    }
    original = study.run_study(manifest(tmp_path, payload), tmp_path/"runs", plots=False)
    assert original["successful"]
    with np.load(original["repetitions"][0]["artifacts"]["calibration"]["path"]) as data:
        np.testing.assert_array_equal(data["X"], X)
        np.testing.assert_array_equal(data["actions"], actions)
        np.testing.assert_array_equal(data["y"], saved_y)
    payload["name"] += "-redraw-saved"
    make_empirical(payload, repetitions=2)
    redrawn = study.run_study(manifest(tmp_path, payload), tmp_path/"runs", plots=False)
    assert redrawn["successful"]
    responses = []
    for row in redrawn["repetitions"]:
        with np.load(row["artifacts"]["calibration"]["path"]) as data:
            np.testing.assert_array_equal(data["X"], X)
            np.testing.assert_array_equal(data["actions"], actions)
            assert not np.array_equal(data["y"], saved_y)
            responses.append(data["y"].copy())
    assert not np.array_equal(*responses)


@pytest.mark.parametrize("change", ["configuration", "source"])
def test_resume_rejects_changed_configuration_or_source(tmp_path, payload, change):
    extra = tmp_path/"additional_source.py"
    extra.write_text("VALUE = 1\n")
    payload["source_files"] = [extra.name]
    spec = manifest(tmp_path, payload)
    study.run_study(spec, tmp_path/"runs", plots=False)
    if change == "configuration":
        payload["envelope"]["delta"] = .1
        spec = manifest(tmp_path, payload)
    else:
        extra.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="inputs/configuration/source changed"):
        study.collect(spec, tmp_path/"runs", plots=False)


def test_failed_python_radius_builder_stays_in_denominator(tmp_path, payload, monkeypatch):
    module = custom_adapter(tmp_path, monkeypatch)
    make_empirical(payload)
    payload["envelope"] = {"type": "python", "callable": f"{module}:failing_radius"}
    payload["coverage"].update(procedure="none", containment_checker="finite_evaluation")
    payload["target"]["actions"] = {"type": "finite_set", "values": [-1., 0., 1.]}
    result = study.run_study(manifest(tmp_path, payload), tmp_path/"runs", plots=False)
    empirical = result["coverage"]["empirical"]
    assert empirical["requested"] == 3
    assert empirical["counts"]["fit_failed"] == 3
    assert empirical["estimate"] is None and empirical["interval"] is None
    assert result["complete"] and not result["successful"]
    assert all(row["error"]["stage"] == "fit" for row in result["repetitions"])


def test_nonlinear_truth_and_python_radius_execute_on_declared_finite_target(tmp_path, payload, monkeypatch):
    module = custom_adapter(tmp_path, monkeypatch)
    make_empirical(payload, repetitions=2)
    payload["problem"]["truth"] = {
        "type": "python", "callable": f"{module}:nonlinear_truth", "options": {"scale": 2.},
    }
    payload["envelope"] = {
        "type": "python", "callable": f"{module}:observed_radius", "options": {"padding": 3.},
    }
    payload["coverage"].update(procedure="none", containment_checker="finite_evaluation")
    payload["target"]["actions"] = {"type": "finite_set", "values": [-1., 0., 1.]}
    result = study.run_study(manifest(tmp_path, payload), tmp_path/"runs", plots=False)
    assert result["complete"] and result["successful"]
    assert result["target"]["kind"] == "finite_set"
    assert result["coverage"]["analytical"] is None
    assert result["coverage"]["empirical"]["estimate"] is not None
    assert all(row["containment"]["details"]["point_count"] == 24 for row in result["repetitions"])
    row = result["repetitions"][0]
    with np.load(row["artifacts"]["points"]["path"]) as points:
        expected = 2*np.sin(points["actions"])+points["X"][:, 0]**2
        np.testing.assert_allclose(points["truth"], expected)
        np.testing.assert_allclose(points["excess"], points["radius"]-points["absolute_error"])
    contract = json.loads((tmp_path/"runs"/payload["name"]/"contract.json").read_text())
    assert any(record["path"].endswith(f"{module}.py") for record in contract["sources"])


def test_finite_checker_cannot_silently_certify_real_line(tmp_path, payload):
    payload["coverage"]["containment_checker"] = "finite_evaluation"
    with pytest.raises(ValueError, match="cannot certify a real-line"):
        manifest(tmp_path, payload)


@pytest.mark.parametrize("invalid", ["nonlinear_truth", "small_width", "small_sigma_bound"])
def test_analytical_theorem_rejects_incompatible_launches(tmp_path, payload, monkeypatch, invalid):
    payload["coverage"]["containment_checker"] = "finite_evaluation"
    payload["target"]["actions"] = {"type": "finite_set", "values": [0., 1.]}
    if invalid == "nonlinear_truth":
        module = custom_adapter(tmp_path, monkeypatch)
        payload["problem"]["truth"] = {"type": "python", "callable": f"{module}:nonlinear_truth"}
    elif invalid == "small_width":
        payload["envelope"]["width_multiplier"] = .5
    else:
        payload["envelope"]["sigma_bound"] = .5
    with pytest.raises(ValueError, match="moment bound assumptions"):
        manifest(tmp_path, payload)
