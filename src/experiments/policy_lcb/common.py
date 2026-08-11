"""Shared contracts and numerical helpers for policy-LCB experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import norm


PolicyLCBLaunchMode = Literal["auto", "local", "slurm"]
PolicyLCBLaunchArray = Literal["none", "seed"]
ORACLE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PolicyLCBLaunchSpec:
    """Launch settings shared by finite and continuous policy-LCB manifests."""

    mode: PolicyLCBLaunchMode
    array: PolicyLCBLaunchArray
    array_max_parallel: int | None = None


def gaussian_lcb_quantile(delta: float, multiplicity: int = 1) -> float:
    """Return a two-sided Gaussian quantile with Bonferroni multiplicity."""
    delta_value = float(delta)
    count = int(multiplicity)
    if not 0.0 < delta_value < 1.0:
        raise ValueError("delta must lie in (0, 1).")
    if count <= 0:
        raise ValueError("multiplicity must be positive.")
    return float(norm.ppf(1.0 - delta_value / (2.0 * count)))


def independent_joint_coverage(delta: float, multiplicity: int) -> float:
    """Return exact coverage for independent Bonferroni-calibrated Gaussians."""
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie in (0, 1).")
    if int(multiplicity) <= 0:
        raise ValueError("multiplicity must be positive.")
    return float((1.0 - float(delta) / int(multiplicity)) ** int(multiplicity))


def shared_gaussian_coverage(delta: float) -> float:
    """Return exact simultaneous coverage when one Gaussian is shared by all policies."""
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie in (0, 1).")
    return 1.0 - float(delta)


def sample_std(values: np.ndarray) -> float:
    """Return sample standard deviation, using zero for singleton groups."""
    values_arr = np.asarray(values, dtype=float)
    return float(np.std(values_arr, ddof=1)) if values_arr.size > 1 else 0.0


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


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON object, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON object."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required object-valued manifest field."""
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a JSON object.")
    return value


def require_type(payload: Mapping[str, Any], key: str, expected: str) -> None:
    """Validate a nested manifest discriminator."""
    value = required_mapping(payload, key)
    if value.get("type") != expected:
        raise ValueError(f"{key}.type must be {expected!r}.")


def number_sequence(value: object, field: str) -> tuple[float, ...]:
    """Normalize a JSON number sequence to a float tuple."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence.")
    return tuple(float(item) for item in value)


def path_part(value: object) -> str:
    """Normalize a manifest name to a result-directory slug."""
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    return "".join(character for character in text if character.isalnum() or character == "-")


__all__ = [
    "ORACLE_TOLERANCE",
    "PolicyLCBLaunchArray",
    "PolicyLCBLaunchMode",
    "PolicyLCBLaunchSpec",
    "gaussian_lcb_quantile",
    "independent_joint_coverage",
    "number_sequence",
    "path_part",
    "read_json",
    "required_mapping",
    "require_type",
    "sample_std",
    "shared_gaussian_coverage",
    "wilson_interval",
    "write_json_atomic",
]
