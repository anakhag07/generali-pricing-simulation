from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETIRED_REAL_DATA = ROOT / "scratch" / "retired_real_data_analysis"


def _python_files(root: Path):
    return sorted(root.glob("*.py"))


def test_real_data_drivers_do_not_import_other_scripts() -> None:
    violations: list[str] = []
    for path in _python_files(RETIRED_REAL_DATA):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("scripts"):
                violations.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("scripts"):
                        violations.append(f"{path.name}:{node.lineno}")
    assert violations == []


def test_scripts_never_call_scipy_minimize_directly() -> None:
    violations: list[str] = []
    roots = (ROOT / "scripts", RETIRED_REAL_DATA)
    for root in roots:
        for path in _python_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "scipy.optimize":
                    imported = {alias.name for alias in node.names}
                    if imported.intersection({"minimize", "minimize_scalar"}):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert violations == []


def test_retired_real_data_plots_only_name_pdf_outputs() -> None:
    violations = [
        path.name
        for path in _python_files(RETIRED_REAL_DATA)
        if ".png" in path.read_text(encoding="utf-8")
    ]
    assert violations == []
