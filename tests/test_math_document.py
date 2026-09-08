import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATH = ROOT / "MATH.md"


def _text() -> str:
    return MATH.read_text(encoding="utf-8")


def test_math_document_has_numbered_outline_and_sections() -> None:
    text = _text()
    headings = re.findall(r"^## (\d+)\. (.+)$", text, flags=re.MULTILINE)

    assert [int(number) for number, _ in headings] == list(range(1, 11))
    assert "## Outline" in text
    for number, title in headings:
        slug = re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-")
        anchor = f"#{number}-{slug}"
        assert f"]({anchor})" in text


def test_math_document_defines_precedence_and_real_data_processing() -> None:
    text = _text()
    precedence = [
        "current implementation",
        "tests;",
        "this file;",
        "`README.md`;",
        "`AGENTS.md`.",
    ]

    positions = [text.index(item) for item in precedence]
    assert positions == sorted(positions)
    assert "`src/data/dataset.csv` is the canonical" in text
    assert "Loaders\nnever modify it" in text
    assert "Acceptance state" in text
    assert "Loss state" in text
    assert "Model and policy frames" in text


def test_math_document_uses_renderable_display_math_delimiters() -> None:
    lines = _text().splitlines()
    delimiter_lines = [line for line in lines if "$$" in line]

    assert len(delimiter_lines) % 2 == 0
    assert all(line.strip() == "$$" for line in delimiter_lines)
    assert _text().count("```") % 2 == 0


def test_paths_in_implementation_index_exist() -> None:
    section = _text().split("## 10. Implementation and Verification Index", 1)[1]
    paths = re.findall(r"`((?:src|tests)/[^`]+)`", section)

    assert paths
    for path in paths:
        assert (ROOT / path).exists(), path
