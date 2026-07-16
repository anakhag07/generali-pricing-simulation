"""Enforces the generali/synthetic seam in `objective.objectives`.

The split is by provenance: generali objectives are bound to the real dataset and
trained artifacts under `src/data`; synthetic objectives are self-contained. That
distinction is only worth having if it is checkable, and "does it import data?" is
the checkable form -- synthetic objectives must stay usable as fast fixtures and
reference benchmarks in a checkout with no data artifacts.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_OBJECTIVES = pathlib.Path(__file__).resolve().parents[2] / "src" / "objective" / "objectives"
_SYNTHETIC = _OBJECTIVES / "synthetic"


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def _synthetic_sources() -> list[pathlib.Path]:
    return sorted(_SYNTHETIC.glob("*.py"))


def test_synthetic_package_is_populated() -> None:
    """Guards against the glob silently matching nothing and vacuously passing."""
    names = {path.name for path in _synthetic_sources()}
    assert {"ladder.py", "planted_logistic.py"} <= names


@pytest.mark.parametrize("source", _synthetic_sources(), ids=lambda p: p.name)
def test_synthetic_objectives_do_not_depend_on_data(source: pathlib.Path) -> None:
    offenders = sorted(
        module
        for module in _imported_modules(source)
        if module == "data" or module.startswith("data.")
    )
    assert not offenders, (
        f"{source.name} imports {offenders}; synthetic objectives must stay free of "
        "data artifacts. If it genuinely needs the dataset, it belongs under generali/."
    )
