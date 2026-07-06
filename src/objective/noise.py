"""Deterministic additive noise models for objective value oracles."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtri

from objective.base import Objective, Policy
from objective.utils import _policy_value


_MAX_SEED = 2**63 - 1
_FLOAT64_DTYPE = np.dtype("<f8")


class ObjectiveNoise:
    r"""Additive action-level noise field $$\delta(x, u)$$ for noisy objectives."""

    def values(self, x_batch: Any, u: np.ndarray) -> np.ndarray:
        """Return additive noise with the same shape as ``u``."""
        raise NotImplementedError

    def with_seed(self, seed: int) -> "ObjectiveNoise":
        """Return a copy using the supplied experiment noise seed."""
        raise NotImplementedError


@dataclass(frozen=True)
class NoNoise(ObjectiveNoise):
    """Zero additive noise field."""

    def values(self, x_batch: Any, u: np.ndarray) -> np.ndarray:
        del x_batch
        return np.zeros_like(np.asarray(u, dtype=float), dtype=float)

    def with_seed(self, seed: int) -> "NoNoise":
        del seed
        return self


@dataclass(frozen=True)
class HomoskedasticGaussianNoise(ObjectiveNoise):
    """Deterministic Gaussian field with constant standard deviation."""

    std: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        std = float(self.std)
        if not np.isfinite(std) or std < 0.0:
            raise ValueError("std must be finite and nonnegative.")
        object.__setattr__(self, "std", std)
        if self.seed is not None:
            object.__setattr__(self, "seed", _validate_seed(self.seed))

    def with_seed(self, seed: int) -> "HomoskedasticGaussianNoise":
        return replace(self, seed=_validate_seed(seed))

    def values(self, x_batch: Any, u: np.ndarray) -> np.ndarray:
        """Return deterministic $$N(0, \text{std}^2)$$ noise keyed by exact ``(x, u)``."""
        u_arr = _validate_u_field(u)
        if self.std == 0.0:
            return np.zeros_like(u_arr, dtype=float)
        if self.seed is None:
            raise ValueError("HomoskedasticGaussianNoise requires a seed before evaluation.")
        return self.std * _unit_normal_field(self.seed, x_batch, u_arr)


@dataclass(frozen=True)
class HeteroskedasticGaussianNoise(ObjectiveNoise):
    r"""Deterministic Gaussian field whose std grows with action distance from ``u_center``.

    $$\delta(x, u) = \big(\sigma_0 + \gamma\,|u - u_c|\big)\,\varepsilon(x, u; s)$$ with the
    same hash-keyed unit-normal field $$\varepsilon(x, u; s)$$ as the homoskedastic adapter,
    so evaluations near ``u_center`` (typically the planted optimum) stay nearly noiseless
    while far-from-optimum queries become increasingly noisy. With ``growth = 0`` this is
    exactly ``HomoskedasticGaussianNoise(std=base_std)``. The seed is controlled by the
    experiment ``noise_seed`` stream through ``with_seed``.
    """

    base_std: float = 0.0
    growth: float = 1.0
    u_center: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        base_std = float(self.base_std)
        growth = float(self.growth)
        u_center = float(self.u_center)
        if not np.isfinite(base_std) or base_std < 0.0:
            raise ValueError("base_std must be finite and nonnegative.")
        if not np.isfinite(growth) or growth < 0.0:
            raise ValueError("growth must be finite and nonnegative.")
        if not np.isfinite(u_center):
            raise ValueError("u_center must be finite.")
        object.__setattr__(self, "base_std", base_std)
        object.__setattr__(self, "growth", growth)
        object.__setattr__(self, "u_center", u_center)
        if self.seed is not None:
            object.__setattr__(self, "seed", _validate_seed(self.seed))

    def with_seed(self, seed: int) -> "HeteroskedasticGaussianNoise":
        return replace(self, seed=_validate_seed(seed))

    def std_values(self, u: np.ndarray) -> np.ndarray:
        r"""Return the local noise std $$\sigma(u) = \sigma_0 + \gamma\,|u - u_c|$$."""
        u_arr = _validate_u_field(u)
        return self.base_std + self.growth * np.abs(u_arr - self.u_center)

    def values(self, x_batch: Any, u: np.ndarray) -> np.ndarray:
        r"""Return deterministic $$N(0, \sigma(u)^2)$$ noise keyed by exact ``(x, u)``."""
        u_arr = _validate_u_field(u)
        if self.base_std == 0.0 and self.growth == 0.0:
            return np.zeros_like(u_arr, dtype=float)
        if self.seed is None:
            raise ValueError("HeteroskedasticGaussianNoise requires a seed before evaluation.")
        return self.std_values(u_arr) * _unit_normal_field(self.seed, x_batch, u_arr)


@dataclass(frozen=True)
class NoisyObjective(Objective):
    r"""Objective wrapper exposing $$\hat{M}(x,u)=M(x,u)+\delta(x,u)$$ values."""

    base_objective: Objective
    noise: ObjectiveNoise
    policy: Policy | None = None

    def __post_init__(self) -> None:
        base_policy = getattr(self.base_objective, "policy", None)
        if self.policy is None:
            if base_policy is not None:
                object.__setattr__(self, "policy", base_policy)
            return
        if base_policy is self.policy:
            return
        try:
            updated_base = replace(self.base_objective, policy=self.policy)
        except TypeError as exc:
            raise ValueError("base_objective policy could not be replaced.") from exc
        object.__setattr__(self, "base_objective", updated_base)

    def with_noise_seed(self, seed: int) -> "NoisyObjective":
        """Return a wrapper copy whose noise is controlled by ``seed``."""
        return replace(self, noise=self.noise.with_seed(seed))

    def value(self, theta: np.ndarray, x_batch: Any) -> float:
        """Return mean noisy objective value for ``theta`` on ``x_batch``."""
        theta_arr = np.asarray(theta, dtype=float)
        base_value = float(self.base_objective.value(theta_arr, x_batch))
        u_arr = self._clip_u(_policy_value(self.base_objective, theta_arr, x_batch))
        return base_value + float(np.mean(self.noise.values(x_batch, u_arr)))

    def base_value(self, theta: np.ndarray, x_batch: Any) -> float:
        """Return the wrapped clean objective value used for reporting."""
        base_value_fn = getattr(self.base_objective, "base_value", None)
        if callable(base_value_fn):
            return float(base_value_fn(theta, x_batch))
        return float(self.base_objective.value(theta, x_batch))

    def grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        """No analytical gradient exists for hash-keyed noisy objective values."""
        del theta, x_batch
        raise NotImplementedError(
            "NoisyObjective has no analytical gradient for M_hat = M + delta(x, u). "
            "Use value-based zeroth-order estimators, or call base_objective.grad(...) "
            "to inspect the true non-noisy objective gradient."
        )

    def value_at_u(self, x_batch: Any, u: float) -> float:
        """Return mean noisy objective value at a fixed action ``u``."""
        value_at_u_fn = getattr(self.base_objective, "value_at_u", None)
        if not callable(value_at_u_fn):
            raise ValueError("base_objective does not support value_at_u(x_batch, u).")
        row_count = _row_count(x_batch)
        u_arr = self._clip_u(np.full(row_count, float(u), dtype=float))
        return float(value_at_u_fn(x_batch, float(u))) + float(np.mean(self.noise.values(x_batch, u_arr)))

    def base_value_at_u(self, x_batch: Any, u: float) -> float:
        """Return the wrapped clean objective value at a fixed action ``u``."""
        base_value_at_u_fn = getattr(self.base_objective, "base_value_at_u", None)
        if callable(base_value_at_u_fn):
            return float(base_value_at_u_fn(x_batch, u))
        value_at_u_fn = getattr(self.base_objective, "value_at_u", None)
        if callable(value_at_u_fn):
            return float(value_at_u_fn(x_batch, u))
        raise ValueError("base_objective does not support value_at_u(x_batch, u).")

    def _value_batch(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
        """Return per-row noisy action-level objective values."""
        u_values = _validate_u_vector(u_arr, _row_count(x_batch))
        base_values = _base_action_values(self.base_objective, x_batch, u_values)
        return base_values + self.noise.values(x_batch, u_values)

    def _value_batch_many(self, x_batch: Any, u_matrix: np.ndarray) -> np.ndarray:
        """Return noisy action-level values for many action vectors."""
        u_values = np.asarray(u_matrix, dtype=float)
        if u_values.ndim != 2:
            raise ValueError("u_matrix must be 2D.")
        n_rows = _row_count(x_batch)
        if u_values.shape[1] != n_rows:
            raise ValueError("u_matrix must have shape (n_evaluations, n_rows).")
        value_many_fn = getattr(self.base_objective, "_value_batch_many", None)
        if callable(value_many_fn):
            base_values = np.asarray(value_many_fn(x_batch, u_values), dtype=float)
        else:
            base_values = np.vstack(
                [_base_action_values(self.base_objective, x_batch, u_row) for u_row in u_values]
            )
        if base_values.shape != u_values.shape:
            raise ValueError("base objective returned unexpected value matrix shape.")
        return base_values + self.noise.values(x_batch, u_values)

    def policy_value(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        """Delegate policy action evaluation to the wrapped objective."""
        return _policy_value(self.base_objective, theta, x_batch)

    def policy_grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        """Delegate policy Jacobian evaluation to the wrapped objective."""
        grad_fn = getattr(self.base_objective, "policy_grad", None)
        if callable(grad_fn):
            return np.asarray(grad_fn(theta, x_batch), dtype=float)
        if self.policy is None:
            raise ValueError("base_objective does not expose a policy gradient.")
        return np.asarray(self.policy.grad(theta, x_batch), dtype=float)

    def policy_weighted_grad(self, theta: np.ndarray, x_batch: Any, weights: np.ndarray) -> np.ndarray:
        """Delegate weighted policy-gradient evaluation to the wrapped objective."""
        weighted_grad_fn = getattr(self.base_objective, "policy_weighted_grad", None)
        if callable(weighted_grad_fn):
            return np.asarray(weighted_grad_fn(theta, x_batch, weights), dtype=float)
        if self.policy is None:
            raise ValueError("base_objective does not expose a weighted policy gradient.")
        return np.asarray(self.policy.weighted_grad(theta, x_batch, weights), dtype=float)

    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        clip_fn = getattr(self.base_objective, "_clip_u", None)
        if callable(clip_fn):
            return np.asarray(clip_fn(u), dtype=float)
        return np.asarray(u, dtype=float)

    def __getattr__(self, name: str) -> Any:
        if name == "grad":
            raise AttributeError(name)
        return getattr(self.base_objective, name)


def _base_action_values(objective: Objective, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
    value_batch_fn = getattr(objective, "_value_batch", None)
    if callable(value_batch_fn):
        values = np.asarray(value_batch_fn(x_batch, u_arr), dtype=float)
        if values.shape != u_arr.shape:
            raise ValueError("base objective _value_batch returned unexpected shape.")
        return values
    value_at_u_fn = getattr(objective, "value_at_u", None)
    if not callable(value_at_u_fn):
        raise ValueError("base_objective must expose _value_batch or value_at_u.")
    values = np.empty_like(u_arr, dtype=float)
    for idx, u_val in enumerate(u_arr):
        values[idx] = float(value_at_u_fn(_slice_rows(x_batch, idx, idx + 1), float(u_val)))
    return values


def _row_count(x_batch: Any) -> int:
    return int(x_batch.shape[0])


def _slice_rows(x_batch: Any, start: int, stop: int) -> Any:
    if hasattr(x_batch, "iloc"):
        return x_batch.iloc[start:stop].reset_index(drop=True)
    return np.asarray(x_batch)[start:stop]


def _validate_u_field(u: np.ndarray) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    if u_arr.ndim not in {1, 2}:
        raise ValueError("u must be a 1D action vector or 2D action matrix.")
    if not np.isfinite(u_arr).all():
        raise ValueError("u must contain only finite values.")
    return u_arr


def _unit_normal_field(seed: int, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
    """Unit-normal field shared by the Gaussian noise adapters, keyed by exact (x, u, seed)."""
    row_hashes = _row_fingerprints(x_batch)
    n_rows = len(row_hashes)
    if u_arr.ndim == 1:
        if u_arr.shape != (n_rows,):
            raise ValueError("1D u must have one value per x_batch row.")
        return _standard_normals(seed, row_hashes, u_arr)
    if u_arr.shape[1] != n_rows:
        raise ValueError("2D u must have shape (n_evaluations, n_rows).")
    return np.vstack([_standard_normals(seed, row_hashes, u_row) for u_row in u_arr])


def _validate_u_vector(u: np.ndarray, n_rows: int) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float).reshape(-1)
    if u_arr.shape != (n_rows,):
        raise ValueError("u_arr must have one value per x_batch row.")
    if not np.isfinite(u_arr).all():
        raise ValueError("u_arr must contain only finite values.")
    return u_arr


def _row_fingerprints(x_batch: Any) -> tuple[bytes, ...]:
    if hasattr(x_batch, "iloc") and hasattr(x_batch, "columns"):
        frame = x_batch.reset_index(drop=True)
        columns = tuple(str(col) for col in frame.columns)
        return tuple(_fingerprint_row(tuple(row), columns=columns) for row in frame.itertuples(index=False, name=None))
    x_arr = np.asarray(x_batch)
    if x_arr.ndim != 2:
        raise ValueError("x_batch must be 2D.")
    if x_arr.dtype != object:
        x_contig = np.ascontiguousarray(x_arr)
        prefix = f"array|{x_contig.dtype}|{x_contig.shape[1]}|".encode("utf-8")
        return tuple(_digest(prefix + row.tobytes()) for row in x_contig)
    return tuple(_fingerprint_row(tuple(row), columns=None) for row in x_arr)


def _fingerprint_row(row: tuple[Any, ...], *, columns: tuple[str, ...] | None) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    if columns is None:
        h.update(b"array-object-row")
    else:
        h.update(b"dataframe-row")
        for col in columns:
            _update_token(h, col.encode("utf-8"))
    for value in row:
        _update_token(h, _stable_value_bytes(value))
    return h.digest()


def _stable_value_bytes(value: Any) -> bytes:
    if value is None:
        return b"none"
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if isinstance(value, (bool, np.bool_)):
        return b"bool:" + bytes([int(value)])
    if isinstance(value, (int, np.integer)):
        return b"int:" + int(value).to_bytes(8, byteorder="little", signed=True)
    if isinstance(value, (float, np.floating)):
        val = float(value)
        if np.isnan(val):
            return b"float:nan"
        return b"float:" + np.asarray(val, dtype=_FLOAT64_DTYPE).tobytes()
    if pd.isna(value):
        return b"na"
    return b"repr:" + repr(value).encode("utf-8")


def _standard_normals(seed: int, row_hashes: tuple[bytes, ...], u_arr: np.ndarray) -> np.ndarray:
    uints = np.fromiter(
        (_noise_uint64(seed, row_hash, float(u_val)) for row_hash, u_val in zip(row_hashes, u_arr)),
        dtype=np.uint64,
        count=len(row_hashes),
    )
    mantissa = uints >> np.uint64(11)
    uniforms = np.ldexp(mantissa.astype(np.float64) + 0.5, -53)
    return np.asarray(ndtri(uniforms), dtype=float)


def _noise_uint64(seed: int, row_hash: bytes, u_val: float) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(b"homoskedastic-gaussian-noise-v1")
    h.update(int(seed).to_bytes(8, byteorder="little", signed=False))
    h.update(row_hash)
    h.update(np.asarray(float(u_val), dtype=_FLOAT64_DTYPE).tobytes())
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def _digest(payload: bytes) -> bytes:
    return hashlib.blake2b(payload, digest_size=16).digest()


def _update_token(h: "hashlib._Hash", payload: bytes) -> None:
    h.update(len(payload).to_bytes(8, byteorder="little", signed=False))
    h.update(payload)


def _validate_seed(seed: int) -> int:
    seed_int = int(seed)
    if seed_int < 0 or seed_int > _MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {_MAX_SEED}].")
    return seed_int


__all__ = [
    "HeteroskedasticGaussianNoise",
    "HomoskedasticGaussianNoise",
    "NoisyObjective",
    "NoNoise",
    "ObjectiveNoise",
]
