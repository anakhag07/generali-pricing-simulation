"""Construction-neutral contracts and diagnostics for uncertainty envelopes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from scipy.stats import norm

def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a Wilson score interval with exact endpoints at all-fail/all-pass."""
    if trials <= 0:
        raise ValueError("trials must be positive.")
    z_value = float(norm.ppf(0.5 + confidence / 2.0))
    proportion = successes / trials
    denominator = 1.0 + z_value**2 / trials
    center = (proportion + z_value**2 / (2.0 * trials)) / denominator
    radius = (
        z_value
        * np.sqrt(proportion * (1.0 - proportion) / trials + z_value**2 / (4.0 * trials**2))
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, center - radius)
    high = 1.0 if successes == trials else min(1.0, center + radius)
    return float(low), float(high)


Model = Callable[[np.ndarray, np.ndarray], np.ndarray]


def locations(X, actions):
    """Validate paired finite contexts/actions, allowing zero context columns."""
    X, actions = np.asarray(X, dtype=float), np.asarray(actions, dtype=float)
    if (X.ndim != 2 or len(X) == 0
            or actions.shape != (len(X),)
            or not np.isfinite(X).all() or not np.isfinite(actions).all()):
        raise ValueError("Require finite X of shape (n,d) and actions of shape (n,).")
    return X, actions


def values(result, size, name):
    """Require one finite model value per queried context/action pair."""
    result = np.asarray(result, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must return {size} finite values, shape ({size},).")
    return result


@dataclass(frozen=True)
class CalibrationData:
    """Observed rows only; synthetic truth is deliberately absent."""

    X: np.ndarray
    actions: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        X, actions = locations(self.X, self.actions)
        for name, array in (("X", X), ("actions", actions),
                            ("y", values(self.y, len(X), "responses"))):
            copy = array.copy()
            copy.flags.writeable = False
            object.__setattr__(self, name, copy)


@dataclass(frozen=True)
class TargetDomain:
    """Observed context fibers crossed with R or an explicitly finite action set."""

    contexts: np.ndarray
    actions: np.ndarray | None = None

    def __post_init__(self):
        X = np.asarray(self.contexts, dtype=float)
        locations(X, np.zeros(len(X)))
        X = X.copy()
        X.flags.writeable = False
        object.__setattr__(self, "contexts", X)
        if self.actions is not None:
            a = np.asarray(self.actions, dtype=float)
            if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all():
                raise ValueError("Finite target actions must be a nonempty finite vector.")
            a = a.copy()
            a.flags.writeable = False
            object.__setattr__(self, "actions", a)

    @property
    def kind(self):
        """Return the scope of the simultaneous containment event."""
        return "real_line" if self.actions is None else "finite_set"

    def points(self):
        """Expand a finite target into paired evaluation locations."""
        if self.actions is None:
            raise ValueError("The real line cannot be expanded into a finite target.")
        return cartesian_locations(self.contexts, self.actions)


def cartesian_locations(contexts, actions):
    """Expand contexts by actions in context-major order."""
    X, a = np.asarray(contexts, dtype=float), np.asarray(actions, dtype=float)
    return locations(np.repeat(X, len(a), axis=0), np.tile(a, len(X)))


@dataclass(frozen=True)
class EnvelopeValues:
    """Pointwise endpoints; ``radius`` is available only for symmetric bands."""

    center: np.ndarray
    radius: np.ndarray | None
    components: dict[str, np.ndarray] = field(default_factory=dict)
    radius_lower: np.ndarray | None = None
    radius_upper: np.ndarray | None = None

    @property
    def lower(self):
        """Return the lower endpoint without symmetrizing residual transfer."""
        return self.center - (self.radius if self.radius_lower is None else self.radius_lower)

    @property
    def upper(self):
        """Return the upper endpoint without symmetrizing residual transfer."""
        return self.center + (self.radius if self.radius_upper is None else self.radius_upper)


@dataclass
class FittedEnvelope:
    """Deterministic fitted callables with replayable numeric state and metadata."""

    center: Model
    radius: Model | None = None
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    components: Callable | None = None
    radii: Callable | None = None
    action_derivatives: Callable | None = None

    def evaluate(self, X, actions):
        """Evaluate paired queries and reject invalid or negative radii."""
        X, actions = locations(X, actions)
        center = values(self.center(X, actions), len(X), "center")
        if (self.radius is None) == (self.radii is None):
            raise ValueError("Specify either a symmetric radius or separate lower/upper radii.")
        if self.radii is None:
            radius = values(self.radius(X, actions), len(X), "radius")
            lower = upper = radius
        else:
            radius = None
            lower, upper = self.radii(X, actions)
            lower = values(lower, len(X), "lower radius")
            upper = values(upper, len(X), "upper radius")
        if np.any(lower < 0) or np.any(upper < 0):
            raise ValueError("Envelope radii must be nonnegative.")
        parts = {} if self.components is None else self.components(X, actions)
        if not isinstance(parts, Mapping):
            raise ValueError("Envelope components must be a mapping.")
        parts = {key: values(value, len(X), f"component {key}") for key, value in parts.items()}
        return EnvelopeValues(center, radius, parts, lower, upper)

    def derivative(self, X, actions, endpoint):
        """Return a declared endpoint action derivative, rejecting query-only methods."""
        if endpoint not in {"center", "lower", "upper"}:
            raise ValueError("Endpoint must be center, lower or upper.")
        if self.action_derivatives is None:
            raise ValueError("This fitted band does not provide first-order derivatives.")
        X, actions = locations(X, actions)
        return values(self.action_derivatives(X, actions)[endpoint], len(X), "action derivative")


@dataclass(frozen=True)
class ContainmentResult:
    """A realized-domain decision, distinct from a coverage probability."""

    status: str
    method: str
    domain: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in {"covered", "violated", "unknown"}:
            raise ValueError("Invalid containment status.")
        if self.domain not in {"real_line", "finite_set"}:
            raise ValueError("Invalid containment domain.")

    def to_dict(self):
        """Serialize the decision and its certificate semantics."""
        return asdict(self)


@dataclass(frozen=True)
class ProbabilityStatement:
    """An analytical probability statement with explicit event and assumptions."""

    kind: str
    value: float
    event: str
    conditioning: str
    assumptions: tuple[str, ...]
    derivation: str

    def __post_init__(self):
        if self.kind not in {"lower_bound", "exact_probability"}:
            raise ValueError("Analytical probability kind must be explicit.")
        if not np.isfinite(self.value) or not 0 <= self.value <= 1:
            raise ValueError("Probability must be in [0,1].")
        if not all((self.event, self.conditioning, self.assumptions, self.derivation)):
            raise ValueError("Probability statements require event, assumptions and derivation.")

    def to_dict(self):
        """Serialize the analytical statement without calling it an estimate."""
        return asdict(self)


def evaluate_tightness(envelope, truth, X, actions, weights=None):
    """Return pointwise radius/excess and summaries over a declared finite measure."""
    X, actions = locations(X, actions)
    evaluated = envelope.evaluate(X, actions)
    actual = values(truth(X, actions), len(X), "truth")
    weight = np.ones(len(X)) if weights is None else values(weights, len(X), "weights")
    if np.any(weight < 0) or weight.sum() <= 0:
        raise ValueError("Diagnostic weights must be nonnegative with positive sum.")
    weight = weight / weight.sum()
    error = np.abs(evaluated.center - actual)
    half_width = evaluated.radius if evaluated.radius is not None else (evaluated.upper-evaluated.lower)/2
    excess = (evaluated.radius-error if evaluated.radius is not None else
              np.minimum(actual-evaluated.lower, evaluated.upper-actual))
    points = {"actions": actions, "center": evaluated.center, "truth": actual,
              "radius": half_width, "absolute_error": error,
              "excess": excess, "weights": weight}
    if evaluated.radius is None:
        points.update(lower=evaluated.lower, upper=evaluated.upper,
                      radius_lower=evaluated.radius_lower, radius_upper=evaluated.radius_upper)
    points.update({f"component_{key}": value for key, value in evaluated.components.items()})
    summary = {}
    for key in ("radius", "excess", "absolute_error"):
        value = points[key]
        # Inverse weighted empirical CDF; this is a diagnostic quantile, not an action selection.
        order = np.argsort(value)
        positive = weight[order] > 0
        ordered, cumulative = value[order][positive], np.cumsum(weight[order][positive])
        summary[f"{key}_mean"] = float(weight @ value)
        for q in (.5, .9):
            index = min(int(np.searchsorted(cumulative, q, side="left")), len(ordered)-1)
            summary[f"{key}_q{int(100*q)}"] = float(ordered[index])
    summary["negative_excess_weight"] = float(weight @ (points["excess"] < 0))
    return points, summary


def check_finite_containment(envelope, truth, target):
    """Check precisely the declared finite set, without claiming continuum coverage."""
    X, a = target.points()
    evaluated = envelope.evaluate(X, a)
    actual = values(truth(X, a), len(a), "truth")
    margin = (evaluated.radius-np.abs(evaluated.center-actual) if evaluated.radius is not None else
              np.minimum(actual-evaluated.lower, evaluated.upper-actual))
    failed = np.flatnonzero(margin < 0)
    covered = (margin >= 0).reshape(len(target.contexts), -1).all(axis=1)
    details = {
        "point_count": len(a), "context_count": len(target.contexts),
        "contexts": [{"context_index": index, "covered": bool(value)}
                     for index, value in enumerate(covered)],
        "numerics": "floating-point evaluation; no rounding enclosure",
    }
    if len(failed):
        index = int(failed[0])
        details["witness"] = {"context": X[index].tolist(), "action": float(a[index]),
                              "excess": float(margin[index])}
    return ContainmentResult("violated" if len(failed) else "covered",
                             "finite_evaluation", target.kind, details)


def aggregate_coverage(records: Sequence[dict], expected: int, confidence=.95):
    """Retain every repetition; suppress estimates with missing or unresolved cases."""
    if expected < 1 or len(records) > expected or not 0 < confidence < 1:
        raise ValueError("Invalid coverage aggregation dimensions or confidence.")
    counts = {key: 0 for key in ("covered", "violated", "unknown", "fit_failed")}
    for row in records:
        status = row["status"]
        if status not in counts:
            raise ValueError(f"Unrecognized coverage status: {status}")
        counts[status] += 1
    missing = expected-len(records)
    decisive = counts["covered"]+counts["violated"] == expected
    interval = wilson_interval(counts["covered"], expected, confidence=confidence) if decisive else None
    return {"kind": "empirical_estimate", "requested": expected, "counts": counts,
            "missing": missing, "estimate": counts["covered"]/expected if decisive else None,
            "interval": interval, "interval_method": "Wilson" if decisive else None,
            "confidence": confidence, "conditioning": "fixed logged contexts/actions and frozen predictor"}
