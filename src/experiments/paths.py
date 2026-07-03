"""Filesystem path helpers for experiment outputs."""

from __future__ import annotations

import os
from pathlib import Path


RESULTS_ROOT_ENV = "GENERALI_RESULTS_ROOT"


def results_root() -> Path:
    """Return the shared external results root for experiment artifacts."""
    override = os.environ.get(RESULTS_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "projects" / "generali-pricing" / "results").resolve()


__all__ = ["RESULTS_ROOT_ENV", "results_root"]
