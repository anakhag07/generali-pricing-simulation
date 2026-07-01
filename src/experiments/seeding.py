"""Seed stream helpers for reproducible experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np


_MAX_SEED = 2**63 - 1


@dataclass(frozen=True)
class SeedSetup:
    """Optional per-process seeds for one experiment run."""

    run_seed: int
    data_seed: int | None = None
    split_seed: int | None = None
    theta_seed: int | None = None
    noise_seed: int | None = None
    optimizer_seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_seed", _validate_seed(self.run_seed, "run_seed"))
        for field_name in ("data_seed", "split_seed", "theta_seed", "noise_seed", "optimizer_seed"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _validate_seed(value, field_name))

    def to_dict(self) -> dict[str, int | None]:
        """Serialize the configured seed overrides."""
        return {
            "run_seed": int(self.run_seed),
            "data_seed": _optional_int(self.data_seed),
            "split_seed": _optional_int(self.split_seed),
            "theta_seed": _optional_int(self.theta_seed),
            "noise_seed": _optional_int(self.noise_seed),
            "optimizer_seed": _optional_int(self.optimizer_seed),
        }


@dataclass(frozen=True)
class ResolvedSeedSetup:
    """Concrete seed values used by every stochastic process in a run."""

    run_seed: int
    data_seed: int
    split_seed: int
    theta_seed: int
    noise_seed: int
    optimizer_seed: int

    def __post_init__(self) -> None:
        for field_name in (
            "run_seed",
            "data_seed",
            "split_seed",
            "theta_seed",
            "noise_seed",
            "optimizer_seed",
        ):
            object.__setattr__(self, field_name, _validate_seed(getattr(self, field_name), field_name))

    def to_dict(self) -> dict[str, int]:
        """Serialize the resolved seed values used by the run."""
        return {
            "run_seed": int(self.run_seed),
            "data_seed": int(self.data_seed),
            "split_seed": int(self.split_seed),
            "theta_seed": int(self.theta_seed),
            "noise_seed": int(self.noise_seed),
            "optimizer_seed": int(self.optimizer_seed),
        }


def seed_setup_from_mapping(setup: SeedSetup | Mapping[str, int | None]) -> SeedSetup:
    """Normalize a ``SeedSetup`` or dictionary payload into ``SeedSetup``."""
    if isinstance(setup, SeedSetup):
        return setup
    return SeedSetup(
        run_seed=int(setup["run_seed"]),
        data_seed=_optional_seed_from_mapping(setup, "data_seed"),
        split_seed=_optional_seed_from_mapping(setup, "split_seed"),
        theta_seed=_optional_seed_from_mapping(setup, "theta_seed"),
        noise_seed=_optional_seed_from_mapping(setup, "noise_seed"),
        optimizer_seed=_optional_seed_from_mapping(setup, "optimizer_seed"),
    )


def resolve_seed_setup(
    seed_setup: SeedSetup | Mapping[str, int | None] | None,
    legacy_seed: int,
) -> ResolvedSeedSetup:
    """Resolve explicit seed streams, deriving missing streams from ``run_seed``."""
    if seed_setup is None:
        seed = _validate_seed(legacy_seed, "legacy_seed")
        return ResolvedSeedSetup(
            run_seed=seed,
            data_seed=seed,
            split_seed=seed,
            theta_seed=seed,
            noise_seed=seed,
            optimizer_seed=seed,
        )
    setup = seed_setup_from_mapping(seed_setup)
    run_seed = int(setup.run_seed)
    return ResolvedSeedSetup(
        run_seed=run_seed,
        data_seed=_resolve_or_derive(setup.data_seed, run_seed, "data"),
        split_seed=_resolve_or_derive(setup.split_seed, run_seed, "split"),
        theta_seed=_resolve_or_derive(setup.theta_seed, run_seed, "theta"),
        noise_seed=_resolve_or_derive(setup.noise_seed, run_seed, "noise"),
        optimizer_seed=_resolve_or_derive(setup.optimizer_seed, run_seed, "optimizer"),
    )


def rng_from_seed(seed: int) -> np.random.Generator:
    """Return a NumPy generator for a validated integer seed."""
    return np.random.default_rng(_validate_seed(seed, "seed"))


def derive_seed(parent_seed: int, label: str) -> int:
    """Derive a stable child seed from a parent seed and semantic label."""
    parent = _validate_seed(parent_seed, "parent_seed")
    if not label:
        raise ValueError("label must be non-empty.")
    label_entropy = _label_entropy(label)
    sequence = np.random.SeedSequence([parent, label_entropy])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def optimizer_rngs(
    seeds: ResolvedSeedSetup,
    estimator_name: str,
) -> tuple[np.random.Generator, np.random.Generator]:
    """Return independent batch and gradient RNGs for one estimator."""
    batch_seed = derive_seed(seeds.optimizer_seed, f"{estimator_name}:batch")
    gradient_seed = derive_seed(seeds.optimizer_seed, f"{estimator_name}:gradient")
    return rng_from_seed(batch_seed), rng_from_seed(gradient_seed)


def _resolve_or_derive(value: int | None, run_seed: int, label: str) -> int:
    if value is not None:
        return int(value)
    return derive_seed(run_seed, label)


def _optional_seed_from_mapping(
    setup: Mapping[str, int | None],
    key: str,
) -> int | None:
    value = setup.get(key)
    return None if value is None else int(value)


def _validate_seed(value: int, name: str) -> int:
    seed = int(value)
    if seed < 0 or seed > _MAX_SEED:
        raise ValueError(f"{name} must be an integer in [0, {_MAX_SEED}].")
    return seed


def _label_entropy(label: str) -> int:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _optional_int(value: int | None) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "ResolvedSeedSetup",
    "SeedSetup",
    "derive_seed",
    "optimizer_rngs",
    "resolve_seed_setup",
    "rng_from_seed",
    "seed_setup_from_mapping",
]
