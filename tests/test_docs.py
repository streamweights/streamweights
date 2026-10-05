"""The docs ship finished: no placeholders, no em-dashes, no number that no report measured,
a README under 150 lines in the promised order, every link resolves, and cli.md and
models.md match the code."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
DOCS = [ROOT / "README.md", ROOT / "CLAUDE.md", *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "streamweights/data/examples").rglob("README.md"))]
REPORTS = sorted((ROOT / "docs/reports").glob("*.md"))
PLACEHOLDERS = ("@@", "<!--TABLE-->", "TBD", "TODO", "FIXME", "XXX", "lorem", "placeholder")


def test_no_placeholders_or_em_dashes():
    for p in DOCS:
        text = p.read_text()
        for bad in PLACEHOLDERS:
            assert bad not in text, f"{bad!r} in {p}"
        assert "—" not in text, f"em-dash in {p}"


def test_readme_shape():
    assert len(README.splitlines()) < 150
    assert README.splitlines()[2] == "Build your own model on the Mac you already own."
    heads = re.findall(r"^## (.+)$", README, re.M)
    assert heads == ["Quick start", "How it works in one picture", "Which path, and how long",
                     "Copy or surpass", "Ship it", "What build does", "Requirements",
                     "Linux and other platforms", "Under the hood", "Status", "Feedback",
                     "License"]
    assert "badge" not in README.lower() and "shields.io" not in README
    assert "pip install git+https://github.com/streamweights/streamweights" in README
    assert "uv tool install git+https://github.com/streamweights/streamweights" in README
    assert "docs/linux.md" in README and "issues/1" in README
    assert "arrives with the full proof run" in README


def test_links_resolve():
    for p in (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))):
        for m in re.finditer(r"\]\(([^)#]+)", p.read_text()):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (p.parent / target).exists(), f"{p.name} links to {target}"


# a number with a unit is a measurement; it must appear in a report
UNIT = r"(?:h|min|s|ms|tok/s|tokens/s|TFLOP/s|GB/s|tokens|rows|steps)"
MEASURED = re.compile(rf"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?({UNIT})\b")
# facts that are specifications, not measurements
SPEC = {"150 lines", "3.10 or", "20 GB", "48 GB"}


def _normalized(text: str) -> str:
    text = re.sub(r"(\d)\s*h\s*(\d+)\s*m\b", r"\1 h \2 m", text)
    return text.replace("**", "")


def test_every_number_in_the_readme_was_measured():
    corpus = _normalized("\n".join(p.read_text() for p in REPORTS)
                         + (ROOT / "docs/reports/data/calibration-m4pro.json").read_text())
    for m in MEASURED.finditer(README):
        num, unit = m.group(1), m.group(2)
        token = f"{num} {unit}"
        if token in SPEC or unit in ("rows", "steps"):
            continue
        assert (token in corpus or f"{num}{unit}" in corpus
                or re.search(rf"\b{re.escape(num)}\s?{re.escape(unit)}\b", corpus)), \
            f"README says {token!r}, which no report in docs/reports contains"


def test_score_tables_match_the_quick_build_report():
    report = (ROOT / "docs/reports/009-phase3.5.md").read_text()
    for row in re.findall(r"^\| (qwen2\.5:0\.5b\S*) \| (your model|base \(untrained\)) \| "
                          r"([\d.]+) \| (\d+) \|$", README, re.M):
        assert " | ".join(row) in report.replace(" ", " "), row


def _cli(*args):
    import os
    import tempfile
    env = {**os.environ, "COLUMNS": "80", "SPILL_HOME": tempfile.mkdtemp(prefix="spill-docs-")}
    r = subprocess.run([sys.executable, "-m", "streamweights", *args], capture_output=True,
                       text=True, env=env, cwd=ROOT, timeout=300)
    return r.stdout


def test_cli_md_is_the_real_help_output():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_cli_docs.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode == 0, "docs/cli.md is out of date: python scripts/make_cli_docs.py"


def test_help_lists_commands_in_loop_order_one_line_each():
    out = _cli("--help")
    cmds = re.findall(r"^  (\w+)\s{2,}(.+)$", out, re.M)
    assert [c for c, _ in cmds] == ["build", "example", "run", "distill", "tune", "eval",
                                    "export", "models", "adapters", "runs", "status", "tail",
                                    "resume", "doctor", "check"]
    assert all(len(d) < 62 for _, d in cmds)


def test_every_command_help_ends_with_one_example():
    for c in ("build", "example", "run", "distill", "tune", "eval", "export", "models",
              "adapters", "runs", "status", "tail", "resume", "doctor", "check"):
        lines = [l for l in _cli(c, "--help").rstrip().splitlines()]
        assert lines[-1].lstrip().startswith("Example: spill ") or \
            lines[-2].lstrip().startswith("Example: spill "), c


def test_models_md_matches_spill_models():
    live = {}
    for line in _cli("models").splitlines():
        m = re.match(r"^(\S+:\S+)\s+(\S+)\s+(.+?)\s{2,}([\d.]+)G\s+([\d.]+)G\s+\S+\s+(\d+)G\+", line)
        if m:
            live[m.group(1)] = m.groups()[1:]
    assert live, "spill models printed no table"
    doc = (ROOT / "docs/models.md").read_text()
    for tag, (params, fam, bf16, q8, disk) in live.items():
        row = re.search(rf"^\| {re.escape(tag)} \| (\S+) \| (.+?) \| ([\d.]+) GB \| ([\d.]+) GB "
                        rf"\| \w+ \| (\d+) GB\+ \|$", doc, re.M)
        assert row, f"{tag} missing from docs/models.md"
        assert row.group(1) == params and row.group(2) == fam.strip()
        assert abs(float(row.group(3)) - float(bf16)) < 0.51, tag
        assert abs(float(row.group(4)) - float(q8)) < 0.51, tag
        assert row.group(5) == disk, tag
