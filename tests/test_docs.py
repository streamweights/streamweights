"""The docs ship finished: no placeholders, no em-dashes, no number that no report measured,
a README under 170 lines in the promised order, every link resolves, and cli.md and
models.md match the code."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
GUIDES = sorted((ROOT / "docs/guides").glob("*.md"))
DOCS = [ROOT / "README.md", ROOT / "CLAUDE.md", ROOT / "CONTRIBUTING.md",
        *sorted((ROOT / "docs").glob("*.md")), *GUIDES,
        *sorted((ROOT / "docs/outreach").glob("*.md")), ROOT / "docs/reports/index.md",
        *sorted((ROOT / "streamweights/data/examples").rglob("README.md"))]
REPORTS = sorted((ROOT / "docs/reports").glob("*.md"))
TRANSCRIPTS = sorted((ROOT / "docs/reports").glob("*.txt"))
PLACEHOLDERS = ("@@", "<!--TABLE-->", "TBD", "TODO", "FIXME", "XXX", "lorem", "placeholder")


def test_no_placeholders_or_em_dashes():
    for p in DOCS:
        if not p.exists():
            continue
        text = p.read_text()
        for bad in PLACEHOLDERS:
            assert bad not in text, f"{bad!r} in {p}"
        assert "—" not in text, f"em-dash in {p}"


def test_readme_shape():
    lines = README.splitlines()
    assert len(lines) < 170
    assert lines[2] == "**A 70B model doesn't fit on your laptop. Build your own model from it anyway.**"
    assert lines[4].startswith("Distill, fine-tune and evaluate on whatever hardware you have.")
    assert lines[6].startswith("[Measured: Llama 3.3 70B, 141 GB unquantized, run on a 48 GB "
                               "MacBook Pro.](docs/reports/")
    heads = re.findall(r"^## (.+)$", README, re.M)
    assert heads == ["Your examples to a model", "Understanding results",
                     "Continuing on another machine", "Sample projects", "When a teacher is useful",
                     "Platforms and explicitly untested paths", "Under the hood", "Prior art",
                     "Status"]
    assert "shields.io" not in README
    badges = re.findall(r"!\[[^\]]*\]\(([^)]*badge[^)]*)\)", README)       # one badge, for the relay workflow only
    assert badges == ["https://github.com/streamweights/streamweights/actions/workflows/relay.yml/badge.svg"]
    assert "pip install git+https://github.com/streamweights/streamweights" in README
    assert "uv tool install git+https://github.com/streamweights/streamweights" in README
    assert "docs/linux.md" in README and "issues/1" in README
    assert "arrives with the full proof run" in README
    first = README.split("\n## ")[0].lower()
    for word in ("fine-tune", "llm", "mac", "linux", "lora", "distill", "70b",
                 "bigger than ram", "local"):
        assert word in first, word
    for img in re.findall(r"!\[([^\]]*)\]", README):
        assert len(img) > 20, "image alt text must be descriptive"


def _slug(heading: str) -> str:
    """The anchor GitHub and MkDocs derive from a heading."""
    text = re.sub(r"`|\*\*|\*", "", heading.strip().lower())
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s", "-", text)


def _anchors(path: Path) -> set:
    out, fenced = set(), False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            fenced = not fenced
        m = None if fenced else re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if m:
            out.add(_slug(m.group(1)))
    return out


def _links(path: Path):
    text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    for m in re.finditer(r"\]\(([^)\s]+)", text):
        yield m.group(1)


def check_links(paths):
    """Every relative link and #anchor in the given markdown files; returns the broken ones."""
    broken = []
    for p in paths:
        for target in _links(p):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            file_part, _, frag = target.partition("#")
            dest = (p.parent / file_part) if file_part else p
            if not dest.exists():
                broken.append(f"{p.relative_to(ROOT)} -> {target} (no such file)")
            elif frag and dest.suffix == ".md" and frag not in _anchors(dest):
                broken.append(f"{p.relative_to(ROOT)} -> {target} (no such anchor)")
    return broken


DOC_PAGES = [ROOT / "README.md", ROOT / "CONTRIBUTING.md",
             *sorted((ROOT / "docs").rglob("*.md"))]


def test_links_and_anchors_resolve():
    pages = [p for p in DOC_PAGES if p.exists() and "paste-sets" not in p.parts]
    assert check_links(pages) == []


def test_link_checker_catches_a_broken_anchor(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\n[ok](b.md#hello-there) [bad](b.md#nope) "
                                   "[gone](c.md)\n")
    (tmp_path / "b.md").write_text("## Hello there\n")
    global ROOT
    saved, ROOT = ROOT, tmp_path
    try:
        broken = check_links([tmp_path / "a.md"])
    finally:
        ROOT = saved
    assert len(broken) == 2 and "nope" in broken[0] and "c.md" in broken[1]


# a number with a unit is a measurement; it must appear in a report
UNIT = r"(?:h|min|s|ms|tok/s|tokens/s|TFLOP/s|GB/s|tokens|rows|steps)"
MEASURED = re.compile(rf"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?({UNIT})\b")
# facts that are specifications, not measurements
SPEC = {"150 lines", "3.10 or", "20 GB", "48 GB"}


def _normalized(text: str) -> str:
    text = re.sub(r"(\d)\s*h\s*(\d+)\s*m\b", r"\1 h \2 m", text)
    return text.replace("**", "")


def test_every_number_in_the_readme_and_guides_was_measured():
    corpus = _normalized("\n".join(p.read_text() for p in [*REPORTS, *TRANSCRIPTS])
                         + (ROOT / "docs/reports/data/calibration-m4pro.json").read_text())
    for page in (ROOT / "README.md", *GUIDES):
        text = re.sub(r"\A---\n.*?\n---\n", "", page.read_text(), flags=re.S)
        for m in MEASURED.finditer(text):
            num, unit = m.group(1), m.group(2)
            token = f"{num} {unit}"
            if token in SPEC or unit in ("rows", "steps"):
                continue
            assert (token in corpus or f"{num}{unit}" in corpus
                    or re.search(rf"\b{re.escape(num)}\s?{re.escape(unit)}\b", corpus)), \
                f"{page.name} says {token!r}, which no report in docs/reports contains"


def test_score_tables_match_the_quick_build_report():
    report = ((ROOT / "docs/reports/009-phase3.5.md").read_text()
              + (ROOT / "docs/reports/014-build-anywhere.md").read_text())
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
    assert [c for c, _ in cmds] == ["init", "plan", "build", "report", "compare", "example", "run",
                                    "distill", "tune", "eval", "export", "test", "bundle", "move",
                                    "models", "adapters", "runs", "status", "tail", "resume",
                                    "doctor", "check"]
    assert all(len(d) < 62 for _, d in cmds)


def test_every_command_help_ends_with_one_example():
    for c in ("init", "plan", "build", "report", "compare", "example", "run", "distill", "tune",
              "eval", "export", "test", "bundle", "move", "models", "adapters", "runs", "status",
              "tail", "resume", "doctor", "check"):
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


def test_site_home_is_rendered_from_the_readme_with_working_links():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_site_home.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout


def test_guides_are_answer_shaped():
    titles = ["Turn a CSV of examples into an evaluated model", "Fine-tune an LLM on a Mac", "Run a 70B model on a 48 GB Mac",
              "Distill a large model into a small one locally",
              "LoRA fine-tuning without a big GPU",
              "Resume a fine-tuning job on a different machine",
              "Run fine-tuning on spot instances with SkyPilot",
              "Start a fine-tuning job on one machine and finish it on another"]
    found = {}
    for g in GUIDES:
        text = g.read_text()
        assert text.startswith("---\ndescription: "), g.name
        title = re.search(r"^# (.+)$", text, re.M).group(1)
        found[title] = text
    assert sorted(found) == sorted(titles)
    for title, text in found.items():
        assert "## Try it" in text and "github.com/streamweights/streamweights" in text, title
        opening = text.split("\n# ", 1)[1].split("\n\n", 2)[1]
        assert len(re.findall(r"[.!?](?:\s|$)", opening)) == 2, f"{title}: two-sentence answer"
