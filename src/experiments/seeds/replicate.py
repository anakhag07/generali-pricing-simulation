"""Seed-stream replication policy for multi-seed sweeps.

Decides which seed streams move across replicates of the same experiment variant.
Streams named in ``vary`` follow each replicate's ``run_seed``; every other stream
is pinned to a shared ``anchor_seed`` so that (by default) data, split, and noise
stay identical across seeds and only policy initialization changes.
"""

from __future__ import annotations

from typing import Literal, Mapping

from experiments.seeds.streams import SeedSetup

SeedStream = Literal["data", "split", "theta", "noise", "optimizer", "all"]

_STREAMS: tuple[SeedStream, ...] = ("data", "split", "theta", "noise", "optimizer")
_ALLOWED: frozenset[str] = frozenset(_STREAMS) | {"all"}


def validate_vary(vary: tuple[SeedStream, ...]) -> None:
    """Validate a ``vary`` tuple of seed streams, raising on unknown/illegal values."""
    unknown = sorted(set(vary) - _ALLOWED)
    if unknown:
        raise ValueError(f"Unknown seed streams: {', '.join(unknown)}.")
    if "all" in vary and len(vary) > 1:
        raise ValueError("vary=('all',) cannot be combined with other seed streams.")


def replicate_seed_setup(
    run_seed: int,
    anchor_seed: int,
    *,
    vary: tuple[SeedStream, ...] = ("theta",),
    fixed: Mapping[str, int | None] | None = None,
) -> SeedSetup:
    """Build a per-replicate ``SeedSetup`` for one ``run_seed`` in a seed sweep.

    Streams in ``vary`` follow ``run_seed``; all other streams are pinned to
    ``anchor_seed`` so data/split/noise stay identical across replicates.
    ``vary=('all',)`` leaves streams unset so seeding derives each per ``run_seed``.
    A non-``None`` entry in ``fixed`` pins that stream regardless of ``vary``.
    """
    validate_vary(vary)
    fixed_map = dict(fixed or {})
    vary_all = "all" in vary
    return SeedSetup(
        run_seed=int(run_seed),
        data_seed=_stream_seed("data", run_seed, anchor_seed, vary, fixed_map, vary_all),
        split_seed=_stream_seed("split", run_seed, anchor_seed, vary, fixed_map, vary_all),
        theta_seed=_stream_seed("theta", run_seed, anchor_seed, vary, fixed_map, vary_all),
        noise_seed=_stream_seed("noise", run_seed, anchor_seed, vary, fixed_map, vary_all),
        optimizer_seed=_stream_seed("optimizer", run_seed, anchor_seed, vary, fixed_map, vary_all),
    )


def _stream_seed(
    stream: SeedStream,
    run_seed: int,
    anchor_seed: int,
    vary: tuple[SeedStream, ...],
    fixed_map: Mapping[str, int | None],
    vary_all: bool,
) -> int | None:
    fixed_seed = fixed_map.get(stream)
    if fixed_seed is not None:
        return int(fixed_seed)
    if vary_all:
        return None
    if stream in vary:
        return int(run_seed)
    return int(anchor_seed)


__all__ = [
    "SeedStream",
    "replicate_seed_setup",
    "validate_vary",
]
