"""Shared preparation of the established synthetic design and explicit saved/Generali inputs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, log_ndtr, ndtr
from scipy.stats import t, truncnorm

from experiments.synthetic import FiniteFourierGPSpec, FourierGPDraw
from experiments.provenance import file_record, array_sha256
from experiments.seeds import derive_seed


@dataclass
class PreparedSetting:
    """Prepared observed inputs, predictor and provenance; truth is evaluation-only."""

    X: np.ndarray
    arrays: dict
    provenance: dict
    predictor: object | None = None
    observations: object | None = None
    multipliers: np.ndarray | None = None
    truth: object | None = None


def repetition_draws(payload, index):
    """Replay the existing paired wild-study streams at their original full shapes."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("repetition must be a nonnegative integer.")
    seeds = {key: derive_seed(payload["seeds"][key], f"repetition:{index}")
             for key in ("observations", "method")}
    nmax, bmax = max(payload["n_values"]), max(payload["B_values"])
    innovations = np.random.default_rng(seeds["observations"]).normal(size=nmax)
    if payload["noise"].get("innovation_distribution", "standard_normal") == "centered_exponential":
        innovations = -log_ndtr(-innovations)-1.
    multipliers = 2*np.random.default_rng(seeds["method"]).integers(0, 2, size=(bmax, nmax), dtype=np.int8)-1
    return innovations, multipliers, seeds


def prepare_setting(spec, *, base_path=None):
    """Prepare an existing synthetic setup, supplied observations, or explicit Generali rows."""
    from experiments.envelopes.core import CalibrationData
    from experiments.envelopes.fitting import predictor_from_spec
    from experiments.envelopes.linear import LinearModel, QuadraticModel

    def path(value):
        candidate = Path(value)
        return candidate if candidate.is_absolute() else Path(base_path or '.') / candidate

    source = spec["source"]
    if source == "wild_study":
        from experiments.envelopes.wild_study import load_manifest
        manifest_path = path(spec["manifest"])
        p = load_manifest(manifest_path).payload
        n, index = spec["n"], spec["repetition"]
        if n not in p["n_values"]:
            raise ValueError("n must be a declared source-study prefix; edit the family setup for a new design.")
        snapshot, designs, _ = _environment(json.dumps(p, sort_keys=True))
        innovations, multipliers, seeds = repetition_draws(p, index)
        design = designs[p["n_values"].index(n)]
        model = QuadraticModel if p["design"].get("action_degree", 1) == 2 else LinearModel
        predictor, truth = model(p["predictor"]["coefficients"]), model(p["truth"]["coefficients"])
        X, actions = snapshot["X"][:n], snapshot["actions"][:n]
        y = design.design @ np.asarray(p["truth"]["coefficients"]) + snapshot["sigma"][:n]*innovations[:n]
        data = CalibrationData(X, actions, y)
        return PreparedSetting(X, {"X": X, "actions": actions, "y": y},
            {"source": source, "manifest": file_record(manifest_path), "resolved_source": p,
             "n": n, "repetition": index, "seeds": seeds, "seed_roots": p["seeds"],
             "multiplier_shape": list(multipliers.shape), "multiplier_sha256": array_sha256(multipliers)},
            predictor, data, multipliers[:, :n], truth)
    if source == "observations":
        artifact = path(spec["path"])
        with np.load(artifact, allow_pickle=False) as saved:
            arrays = dict(saved)
        data = CalibrationData(arrays["X"], arrays["actions"], arrays["y"])
        if not spec.get("quantity") or not spec.get("provenance"):
            raise ValueError("Observed inputs require an explicit quantity and provenance description.")
        predictor = predictor_from_spec(spec["predictor"])
        return PreparedSetting(data.X, arrays,
            {"source": source, "artifact": file_record(artifact), "specification": dict(spec)},
            predictor, data, arrays.get("multipliers"))
    if source == "generali_support":
        return _prepare_generali_support(spec)
    raise ValueError("setting.source must be wild_study, observations or generali_support.")


def _prepare_generali_support(spec):
    from data.dataset_metadata import DATASET_PATH
    from data.loader import load_x_frame, load_observed_u_array
    from data.coverage import mixed_customer_embedding, local_support_matrix
    from experiments.configs.real_data_factory import _artifact_policy_features
    from reporting.real_data import resolve_real_data_model
    from reporting.profit_dispersion import predict_acceptance_matrix, exact_spline_acceptance_matrix
    rows = np.asarray(spec["row_indices"])
    if rows.ndim != 1 or len(rows) < 2 or not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0) or len(np.unique(rows)) != len(rows):
        raise ValueError("Generali preparation requires at least two distinct explicit source row indices.")
    quantity = spec["quantity"]
    if quantity not in {"profit", "cost", "acceptance"}:
        raise ValueError("Declare quantity=profit, cost or acceptance; no response is inferred.")
    grid = np.asarray(spec["action_grid"], dtype=float)
    model = resolve_real_data_model(spec["model"], eligible_rows=rows)
    frame = load_x_frame(model.spec.acceptance_model, row_indices=rows)
    if model.spec.acceptance_strategy == "exact_spline":
        acceptance = exact_spline_acceptance_matrix(model.acceptance_artifact, frame, grid, model.exact_spline_weights, n_jobs=1)
    else:
        acceptance = predict_acceptance_matrix(model.acceptance_artifact, frame, grid)
    predicted = acceptance if quantity == "acceptance" else model.profit_matrix(frame, grid, n_jobs=1)
    if quantity == "cost":
        predicted = -predicted
    X = _artifact_policy_features(model.acceptance_artifact, frame)
    observed = load_observed_u_array(model.spec.acceptance_model, row_indices=rows)
    embedding = mixed_customer_embedding(model.acceptance_artifact, frame)
    support = local_support_matrix(embedding, observed, grid,
        n_neighbors=int(spec["n_neighbors"]), action_bandwidth=float(spec["action_bandwidth"]), n_jobs=1)
    arrays = {"X": X, "action_grid": grid, "predictor_grid": predicted,
              "acceptance_grid": acceptance, "support": support, "row_indices": rows}
    return PreparedSetting(X, arrays, {"source": "generali_support", "specification": dict(spec),
        "dataset": file_record(DATASET_PATH), "model": model.provenance(),
        "row_indices": rows.tolist(), "policy_inputs": "saved acceptance-artifact preprocessing",
        "quantity": quantity, "truth": None})

def noise_scale(X, actions, settings, gp_cos, gp_sin):
    """Evaluate the background scale; observation-row overrides are applied separately."""
    if settings["type"] == "constant":
        return np.full(np.shape(actions), settings["sigma_max"], dtype=float)
    draw = FourierGPDraw(FiniteFourierGPSpec(settings["rank"], settings["lengthscale"]),
                         tuple(gp_cos), tuple(gp_sin))
    t = np.asarray(actions) + np.asarray(X) @ np.asarray(settings["context_projection"])
    return settings["sigma_min"] + (settings["sigma_max"]-settings["sigma_min"]) * expit(
        settings["amplitude"] * draw.evaluate(t))


def _standardized_t3(normal_draws):
    # Symmetry uses the small tail: no upper-CDF rounding, clipping or new draws.
    return np.sign(normal_draws) * t.isf(ndtr(-np.abs(normal_draws)), df=3) / np.sqrt(3.)


@lru_cache(maxsize=4)
def _environment(payload_json):
    from experiments.envelopes.wild import WildLinearDesign
    p = json.loads(payload_json)
    nmax, d, seeds = max(p["n_values"]), p["design"]["context_dim"], p["seeds"]
    context_rng, action_rng = (np.random.default_rng(seeds[key]) for key in ("contexts", "actions"))
    if p["design"]["distribution"] == "independent_truncated_standard_normal":
        bound = p["design"]["truncation_bound"]
        bound = np.inf if bound is None else bound
        # Shared uniforms pair bounds; samples follow a truncated law, not clipping.
        X = truncnorm.ppf(context_rng.random((nmax, d)), -bound, bound)
        actions = truncnorm.ppf(action_rng.random(nmax), -bound, bound)
    else:
        X = context_rng.normal(size=(nmax, d))
        actions = action_rng.normal(size=nmax)
        if p["design"]["distribution"] == "independent_standardized_t3":
            X, actions = _standardized_t3(X), _standardized_t3(actions)
    if "action_override" in p["design"]:
        override = p["design"]["action_override"]
        # This prescribed design change consumes no draws; noise uses its new location.
        actions[override["index"]] = override["value"]
    coefficients = (np.empty((2, 0)) if p["noise"]["type"] == "constant" else
                    np.random.default_rng(seeds["profile"]).normal(size=(2, p["noise"]["rank"])))
    sigma = noise_scale(X, actions, p["noise"], *coefficients)
    if "moved_observation_sigma" in p["noise"]:
        # Override only the moved observation, retaining the background GP and cap.
        sigma[p["design"]["action_override"]["index"]] = p["noise"]["moved_observation_sigma"]
    spec = p["diagnostics"]["actions"]
    grid = np.linspace(spec["min"], spec["max"], spec["count"])
    snapshot = {"X": X, "actions": actions, "sigma": sigma,
                "gp_cos": coefficients[0], "gp_sin": coefficients[1], "actions_grid": grid}
    designs, mean_ell = [], []
    for n in p["n_values"]:
        design = WildLinearDesign(X[:n], actions[:n],
                                  action_degree=p["design"].get("action_degree", 1),
                                  supremum_tolerance=p.get("supremum_tolerance", 1e-8))
        contexts = X[:1] if d == 0 else X[:n]
        phi = design.features(np.repeat(contexts, len(grid), axis=0), np.tile(grid, len(contexts)))
        mean_ell.append(float(np.sqrt(np.einsum("ij,jk,ik->i", phi, design.covariance, phi)).mean()))
        designs.append(design)
    return snapshot, designs, np.array(mean_ell)
