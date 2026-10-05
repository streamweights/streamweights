"""The docs ship finished: no placeholders, no em-dashes, a README under 150 lines in the
promised order, and every image and doc it links exists."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "streamweights/data/examples").rglob("README.md"))]


def test_no_placeholders_or_em_dashes():
    for p in DOCS:
        text = p.read_text()
        assert "@@" not in text and "<!--TABLE-->" not in text, p
        assert "—" not in text, f"em-dash in {p}"


def test_readme_shape():
    assert len(README.splitlines()) < 150
    assert README.splitlines()[2] == "Build your own model on the Mac you already own."
    heads = re.findall(r"^## (.+)$", README, re.M)
    assert heads == ["Quick start", "How it works in one picture", "Which path, and how long",
                     "Copy or surpass", "Ship it", "What build does", "Requirements",
                     "Under the hood", "Status", "Feedback", "License"]
    assert "badge" not in README.lower() and "shields.io" not in README


def test_links_resolve():
    for m in re.finditer(r"\]\(((?:docs|streamweights)/[^)#]+)", README):
        assert (ROOT / m.group(1)).exists(), m.group(1)
