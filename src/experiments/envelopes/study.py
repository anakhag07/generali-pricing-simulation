"""Manifest execution and replay for the uncertainty-envelopes experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace

import numpy as np

from experiments.launch import LaunchPlan
from experiments.paths import results_root
from experiments.provenance import array_sha256, file_record
from experiments.seeds import derive_seed
from experiments.slurm import CPU_PROFILE
from experiments.sweep_reporting import write_rows_csv
from .core import (
    CalibrationData, ContainmentResult, FittedEnvelope, ProbabilityStatement,
    TargetDomain, aggregate_coverage, cartesian_locations,
    check_finite_containment, evaluate_tightness, values,
)
from .linear import LinearModel, check_linear_containment, fit_linear_transfer, linear_guarantee

MANIFEST_KIND = "uncertainty_envelopes"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, default=_jsonable, allow_nan=False)+"\n"
    # Project-level metadata can be prepared concurrently by Slurm tasks.
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _number(value, name, minimum=0, strict=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value < minimum or (strict and value == minimum)):
        raise ValueError(f"Invalid {name}.")
    return float(value)


def _callable(spec):
    if not isinstance(spec, dict) or spec.get("type") != "python":
        raise ValueError("Custom adapters require type=python and callable=module:function.")
    reference = spec.get("callable", "")
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("Python callable must be module:function.")
    module, name = reference.split(":")
    result = getattr(importlib.import_module(module), name)
    if not callable(result):
        raise ValueError(f"Not callable: {reference}")
    return result


def _model(spec):
    if spec.get("type") == "linear":
        return LinearModel(np.asarray(spec["coefficients"], dtype=float))
    factory = _callable(spec)
    result = factory(dict(spec.get("options", {})))
    if not callable(result):
        raise ValueError("Model factory must return a deterministic callable (X, actions).")
    return result


def fit_constant_radius(data, predictor, target, settings, rng):
    """Build a fixed-radius baseline without a statistical coverage claim."""
    radius = _number(settings["radius"], "radius")
    return FittedEnvelope(predictor, lambda X, a: np.full(len(a), radius),
                          state={"radius": radius}, metadata={"construction": "constant_radius"})


@dataclass(frozen=True)
class EnvelopeManifest:
    """Validated choices for construction, coverage, and diagnostic locations."""

    payload: dict
    source_path: Path
    launch: SimpleNamespace

    @property
    def name(self):
        """Return the results-directory name."""
        return self.payload["name"]

    @property
    def repetitions(self):
        """Return one analytical realization or the requested empirical count."""
        c = self.payload["coverage"]
        return 1 if c["mode"] == "analytical" else c["repetitions"]


def load_manifest(path):
    """Validate a launch, rejecting unsupported mathematical combinations early."""
    path = Path(path).resolve()
    p = _read_json(path)
    if p.get("kind") != MANIFEST_KIND:
        raise ValueError(f"Require kind={MANIFEST_KIND}.")
    name = p.get("name")
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name)):
        raise ValueError("Require a path-safe experiment name.")
    calibration = p["calibration"]
    if calibration["type"] == "synthetic":
        _integer(calibration["n"], "calibration.n")
        _integer(calibration["context_dim"], "context_dim")
    elif calibration["type"] == "saved":
        if not calibration.get("path") or len(calibration.get("sha256", "")) != 64:
            raise ValueError("Saved calibration data requires path and SHA256.")
    else:
        raise ValueError("Calibration type must be synthetic or saved.")
    noise = calibration["noise"]
    if noise["type"] == "gaussian":
        _number(noise["std"], "noise.std")
    else:
        _callable(noise)
    if set(p["seeds"]) != {"contexts", "actions", "observations", "method"}:
        raise ValueError("Require separate contexts/actions/observations/method seed roots.")
    for key, seed in p["seeds"].items():
        _integer(seed, f"seeds.{key}", minimum=0)
    if len(set(p["seeds"].values())) != 4:
        raise ValueError("Seed roots must be distinct.")
    truth, predictor = _model(p["problem"]["truth"]), _model(p["problem"]["predictor"])
    method = p["envelope"]
    if method["type"] == "linear_known_variance_transfer":
        _number(method["sigma_bound"], "sigma_bound", strict=True)
        if not 0 < _number(method["delta"], "delta") < 1:
            raise ValueError("Require 0 < delta < 1.")
        _number(method.get("width_multiplier", 1.), "width_multiplier")
        if calibration["type"] == "synthetic" and calibration["n"] < calibration["context_dim"]+2:
            raise ValueError("Linear OLS requires n >= context_dim+2.")
    elif method["type"] == "constant_radius":
        _number(method["radius"], "radius")
    else:
        _callable(method)
    c = p["coverage"]
    if c["mode"] == "empirical":
        _integer(c["repetitions"], "coverage.repetitions", minimum=2)
        if c.get("redraw") != "responses":
            raise ValueError("Empirical coverage must explicitly redraw responses at fixed design.")
    elif c["mode"] == "analytical":
        if "repetitions" in c or "redraw" in c:
            raise ValueError("Analytical mode has one realization, not outer repetitions.")
    else:
        raise ValueError("Coverage mode must be analytical or empirical.")
    if not 0 < c.get("confidence", .95) < 1:
        raise ValueError("Invalid empirical interval confidence.")
    action_spec = p["target"]["actions"]
    if p["target"].get("contexts") != "calibration":
        raise ValueError("Initial target contexts must be the frozen calibration contexts.")
    if action_spec["type"] == "finite_set":
        TargetDomain(np.zeros((1, 1)), action_spec["values"])
    elif action_spec["type"] != "real_line":
        raise ValueError("Target actions must be real_line or finite_set.")
    checker = c["containment_checker"]
    if checker == "affine_transfer_all_real":
        if (method["type"] != "linear_known_variance_transfer"
                or not isinstance(truth, LinearModel) or not isinstance(predictor, LinearModel)
                or action_spec["type"] != "real_line"):
            raise ValueError("Affine real-line checker requires the linear transfer construction and linear models.")
    elif checker == "finite_evaluation":
        if action_spec["type"] != "finite_set":
            raise ValueError("Finite evaluation cannot certify a real-line target.")
    else:
        _callable(checker)
    procedure = c["procedure"]
    if procedure == "linear_moment_bound":
        if (method["type"] != "linear_known_variance_transfer" or not isinstance(truth, LinearModel)
                or noise["type"] != "gaussian" or noise["std"] > method["sigma_bound"]
                or method.get("width_multiplier", 1.) < 1):
            raise ValueError("Linear moment bound assumptions are incompatible with this launch.")
    elif procedure != "none":
        _callable(procedure)
    _diagnostic_actions(p["tightness"]["actions"])
    weights = p["tightness"].get("weights", "uniform")
    if not isinstance(weights, list) and weights != "uniform":
        raise ValueError("Tightness weights must be uniform or an explicit vector.")
    launch = p["launch"]
    if launch["mode"] not in {"local", "slurm", "auto"} or launch["array"] not in {"none", "repetition"}:
        raise ValueError("Invalid launch policy.")
    if launch.get("array_max_parallel") is not None:
        _integer(launch["array_max_parallel"], "array_max_parallel")
    return EnvelopeManifest(p, path, SimpleNamespace(**{**launch, "array_max_parallel": launch.get("array_max_parallel")}))


def _diagnostic_actions(spec):
    if spec["type"] == "linspace":
        _integer(spec["count"], "diagnostic count", minimum=2)
        lo, hi = float(spec["min"]), float(spec["max"])
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError("Diagnostic window requires finite min < max.")
        return np.linspace(lo, hi, spec["count"])
    if spec["type"] == "values":
        return TargetDomain(np.zeros((1, 1)), spec["values"]).actions
    raise ValueError("Diagnostic actions must use linspace or values.")


def _source_records(manifest):
    paths = set(Path(__file__).parent.glob("*.py"))
    root = Path(__file__).parents[1]
    paths.update(root/name for name in ("bootstrap_band_continuous.py", "offline_bootstrap_math.py",
                                      "provenance.py", "seeds/streams.py", "policy_lcb/common.py"))
    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "python":
                path = inspect.getsourcefile(_callable(value))
                if path is None:
                    raise ValueError("Python adapters require inspectable source for provenance.")
                paths.add(Path(path))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(manifest.payload)
    for source in manifest.payload.get("source_files", []):
        paths.add((manifest.source_path.parent/source).resolve())
    return [file_record(path) for path in sorted(paths)]


def _design(manifest):
    p, spec = manifest.payload, manifest.payload["calibration"]
    if spec["type"] == "saved":
        source = (manifest.source_path.parent/spec["path"]).resolve()
        record = file_record(source)
        if record["sha256"] != spec["sha256"]:
            raise ValueError("Saved calibration checksum mismatch.")
        with np.load(source, allow_pickle=False) as a:
            data = CalibrationData(a["X"], a["actions"], a["y"])
        return data.X, data.actions, data.y, record
    X = np.random.default_rng(p["seeds"]["contexts"]).normal(size=(spec["n"], spec["context_dim"]))
    actions = np.random.default_rng(p["seeds"]["actions"]).normal(size=spec["n"])
    return X, actions, None, None


def _prepare(manifest, runs_root, force=False):
    p = manifest.payload
    project = Path(results_root() if runs_root is None else runs_root)/manifest.name
    X, actions, saved_y, input_record = _design(manifest)
    truth, predictor = _model(p["problem"]["truth"]), _model(p["problem"]["predictor"])
    prepared = {"contexts": array_sha256(X), "actions": array_sha256(actions)}
    dx, da = cartesian_locations(X, _diagnostic_actions(p["tightness"]["actions"]))
    for label, model in (("truth", truth), ("predictor", predictor)):
        prepared[label+"_observations"] = array_sha256(values(model(X, actions), len(X), label))
        prepared[label+"_diagnostics"] = array_sha256(values(model(dx, da), len(dx), label))
    target_actions = p["target"]["actions"]
    target = TargetDomain(X, None if target_actions["type"] == "real_line" else target_actions["values"])
    contract = {"schema_version": 1, "manifest": p, "sources": _source_records(manifest),
                "input": input_record, "prepared": prepared,
                "runtime": {"python": platform.python_version(),
                            **{name: version(name) for name in ("numpy", "scipy", "sympy", "matplotlib")}}}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, default=_jsonable).encode()).hexdigest()
    path = project/"contract.json"
    if path.exists() and _read_json(path)["sha256"] != digest and not force:
        raise ValueError("Experiment inputs/configuration/source changed; choose a new name or use --force.")
    _write_json(path, {"sha256": digest, **contract})
    return project, digest, X, actions, saved_y, truth, predictor, target


def _builder(spec):
    if spec["type"] == "linear_known_variance_transfer":
        return fit_linear_transfer
    if spec["type"] == "constant_radius":
        return fit_constant_radius
    return _callable(spec)


def _checker(spec):
    if spec == "affine_transfer_all_real":
        return check_linear_containment
    if spec == "finite_evaluation":
        return check_finite_containment
    return _callable(spec)


def _guarantee(manifest, data, truth, predictor, target):
    p = manifest.payload
    spec = p["coverage"]["procedure"]
    if spec == "none":
        return None
    function = linear_guarantee if spec == "linear_moment_bound" else _callable(spec)
    statement = function(data, truth, predictor, target, p["envelope"], p["calibration"]["noise"].get("std"))
    if not isinstance(statement, ProbabilityStatement):
        raise TypeError("Probability procedures must return ProbabilityStatement.")
    return statement.to_dict()


def _verify(record):
    if file_record(record["path"]) != record:
        raise ValueError(f"Saved artifact changed: {record['path']}")


def _read_completed_case(path):
    completion = _read_json(path.with_name("completion.json"))["summary"]
    if completion["path"] != str(path.resolve()):
        raise ValueError("Repetition completion record points to a different summary.")
    _verify(completion)
    row = _read_json(path)
    for record in row["artifacts"].values():
        _verify(record)
    return row


def run_repetition(manifest, index, runs_root=None, force=False):
    """Fit and evaluate one calibration realization, saving replayable artifacts."""
    if not 0 <= index < manifest.repetitions:
        raise IndexError(index)
    project, digest, X, actions, saved_y, truth, predictor, target = _prepare(manifest, runs_root, force)
    destination = project/"repetitions"/f"{index:06d}"
    summary_path = destination/"summary.json"
    if summary_path.exists() and summary_path.with_name("completion.json").exists() and not force:
        previous = _read_completed_case(summary_path)
        if previous["contract_sha256"] != digest:
            raise ValueError("Repetition contract differs; use --force.")
        return previous
    destination.mkdir(parents=True, exist_ok=True)
    p = manifest.payload
    seeds = {key: derive_seed(p["seeds"][key], f"repetition:{index}") for key in ("observations", "method")}
    row = {"index": index, "contract_sha256": digest, "seeds": seeds,
           "status": "fit_failed", "containment": None, "analytical": None,
           "metrics": None, "artifacts": {}}
    stage = "observations"
    try:
        mean = values(truth(X, actions), len(X), "truth")
        if saved_y is not None and p["coverage"]["mode"] == "analytical":
            y = saved_y
        else:
            rng = np.random.default_rng(seeds["observations"])
            noise = p["calibration"]["noise"]
            y = (mean+noise["std"]*rng.normal(size=len(X)) if noise["type"] == "gaussian"
                 else _callable(noise)(X, actions, mean, dict(noise.get("options", {})), rng))
        data = CalibrationData(X, actions, y)
        calibration_path = destination/"calibration.npz"
        np.savez_compressed(calibration_path, X=data.X, actions=data.actions, y=data.y)
        row["artifacts"]["calibration"] = file_record(calibration_path)
        stage = "fit"
        fitted = _builder(p["envelope"])(data, predictor, target, p["envelope"], np.random.default_rng(seeds["method"]))
        if not isinstance(fitted, FittedEnvelope):
            raise TypeError("Envelope builder must return FittedEnvelope.")
        state_path = destination/"fit.json"
        _write_json(state_path, {"state": fitted.state, "metadata": fitted.metadata})
        row["artifacts"]["fit"] = file_record(state_path)
        stage = "coverage"
        decision = _checker(p["coverage"]["containment_checker"])(fitted, truth, target)
        if not isinstance(decision, ContainmentResult) or decision.domain != target.kind:
            raise ValueError("Checker must return a ContainmentResult on the declared target.")
        row["containment"] = decision.to_dict()
        row["analytical"] = _guarantee(manifest, data, truth, predictor, target)
        stage = "tightness"
        dx, da = cartesian_locations(X, _diagnostic_actions(p["tightness"]["actions"]))
        weights = p["tightness"].get("weights", "uniform")
        points, metrics = evaluate_tightness(fitted, truth, dx, da, None if isinstance(weights, str) else weights)
        point_path = destination/"points.npz"
        np.savez_compressed(point_path, X=dx, **points)
        row["artifacts"]["points"] = file_record(point_path)
        row.update(status=decision.status, metrics=metrics)
    except Exception as error:
        row["error"] = {"stage": stage, "type": type(error).__name__, "message": str(error)}
    _write_json(summary_path, row)
    _write_json(summary_path.with_name("completion.json"), {"summary": file_record(summary_path)})
    return row


def _experiment_text(manifest):
    return f'''# {manifest.name}

Configuration is the source of truth. Truth and predictor are deterministic
factories and deterministic evaluators after preparation. Synthetic contexts and logged actions use independent named
streams and remain fixed across all repetitions. Observation vectors are
independent across repetitions; method randomness uses a separate stream.
Analytical mode uses one calibration realization; empirical mode redraws only
responses, never the design or frozen predictor. A saved response vector is used
in analytical mode; empirical mode deliberately redraws it under the declared
noise law. The law of externally saved data is an assumption, not verified by
its checksum. There is no inner bootstrap unless an explicit adapter provides it.

Target: the saved calibration contexts crossed with the configured real line or
finite action set. Built-in linear functions are evaluated directly off-grid.
Custom adapters must define deterministic evaluation everywhere on their target.
No interpolation of diagnostic samples defines a stochastic function. Within-fit
prediction errors share fitted parameters; they are not independent point noise.
Continuum polynomial checks interpret stored float coefficients/covariance as
exact rational numbers; they do not certify floating-point fitting/evaluation
roundoff. Finite checks make a finite-set floating-point statement only.

Analytical lower bounds, exact probabilities, and realized containment are
separate result fields. Negative excess radius is retained. Diagnostic means and
quantiles use the saved finite locations/weights, not a uniform measure on R.
Empirical estimates require every requested repetition to be decisively checked;
failures, unknowns, and missing records are counted. No action optimization or
global optimum reference is performed. No certificate tolerance is inferred
from plotting resolution. Figure bins summarize saved diagnostic values only.

Named seed roots and any paired parameter comparisons must be retained in copied
manifests. Reusing the same roots pairs the frozen design and response streams;
changing only a diagnostic grid does not alter fitting or coverage. Separate
names preserve each condition's artifacts. Python callbacks execute trusted local
code; list additional imported source dependencies in source_files for hashing.
Restart the Python process after editing imported callbacks. The contract hashes
source, inputs, prepared design/model outputs at observed and diagnostic points,
and numeric-library versions. These finite fingerprints detect drift but do not
prove that custom functions agree everywhere. Per-repetition completion records
checksum each summary and its artifacts before resume or collection.

```json
{json.dumps(manifest.payload, indent=2)}
```
'''


def collect(manifest, runs_root=None, force=False, *, plots=True):
    """Validate saved cases and regenerate summaries without refitting envelopes."""
    project, digest, X, actions, saved_y, truth, predictor, target = _prepare(manifest, runs_root, force)
    rows, records = [], []
    for index in range(manifest.repetitions):
        path = project/"repetitions"/f"{index:06d}"/"summary.json"
        if not path.exists() or not path.with_name("completion.json").exists():
            continue
        row = _read_completed_case(path)
        if row["contract_sha256"] != digest or row["index"] != index:
            raise ValueError("Saved repetition is incompatible with this experiment.")
        records.append(file_record(path))
        rows.append(row)
    statements = [row["analytical"] for row in rows if row["analytical"] is not None]
    if statements and any(statement != statements[0] for statement in statements):
        raise ValueError("Analytical probability statements differ across fixed-design repetitions.")
    empirical = (aggregate_coverage(rows, manifest.repetitions, manifest.payload["coverage"].get("confidence", .95))
                 if manifest.payload["coverage"]["mode"] == "empirical" else None)
    summary = {"name": manifest.name, "manifest": manifest.payload, "contract_sha256": digest,
               "target": {"kind": target.kind, "context_count": len(X), "context_dim": X.shape[1]},
               "coverage": {"analytical": statements[0] if statements else None, "empirical": empirical},
               "requested_repetitions": manifest.repetitions, "repetitions": rows,
               "repetition_records": records, "complete": len(rows) == manifest.repetitions,
               "successful": len(rows) == manifest.repetitions and all(row["status"] != "fit_failed" for row in rows),
               "plots": []}
    table = [{"repetition": row["index"], "status": row["status"], **(row["metrics"] or {})} for row in rows]
    keys = ["repetition", "status"] + sorted({key for row in table for key in row} - {"repetition", "status"})
    write_rows_csv(project/"metrics.csv", table, keys)
    (project/"EXPERIMENT.md").write_text(_experiment_text(manifest))
    _write_json(project/"summary.json", summary)
    if plots:
        from .reporting import write_reports
        from .diagnostics import write_diagnostic_reports
        paths = write_reports(project) + write_diagnostic_reports(project)
        summary["plots"] = [file_record(path) for path in paths]
        _write_json(project/"summary.json", summary)
    print(f"Uncertainty envelopes: {project} ({len(rows)}/{manifest.repetitions} repetitions; "
          f"{sum(row['status'] == 'fit_failed' for row in rows)} failed)", flush=True)
    return summary


def run_study(manifest, runs_root=None, force=False, *, plots=True):
    """Run or resume an uncertainty-envelope study through its selected adapters."""
    for index in range(manifest.repetitions):
        run_repetition(manifest, index, runs_root, force)
    return collect(manifest, runs_root, force, plots=plots)


def build_launch_plan(manifest, *, runs_root=None, force=False):
    """Use the shared local/Slurm runner with one task per calibration repetition."""
    return LaunchPlan(
        name=manifest.name, task_count=manifest.repetitions, requires_jax=False,
        run_task=lambda index, context: run_repetition(manifest, index, context.runs_root, force),
        run_all=lambda context: run_study(manifest, context.runs_root, force),
        collect=lambda context: collect(manifest, context.runs_root, force),
        runs_root=runs_root, default_launch=manifest.launch.mode,
        default_array=manifest.launch.array == "repetition",
        slurm_profile=replace(CPU_PROFILE, cpus_per_task=1, memory="4G"),
    )
