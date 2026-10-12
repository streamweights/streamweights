"""The docs ship finished: no placeholders, no em-dashes, no number that no report measured,
a README in the promised six sections, its output taken from a real MLX run, every link resolves, and cli.md and
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


HEADLINE = "**Build a small model for your task, on hardware you control.**"
SUPPORT = ("Bring labeled examples. Fine-tune locally, compare against simple baselines, and "
           "export to GGUF or safetensors. Pause and resume supported training runs across Mac and Linux.")
SECTIONS = ["What it does", "Try it", "Use your own data", "Understand the result", "Go further",
            "Support and limitations"]


def _section(title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", README, re.M | re.S)
    assert m, title
    return m.group(1)


def test_readme_shape():
    lines = README.splitlines()
    assert lines[2] == HEADLINE and lines[4] == SUPPORT
    assert re.findall(r"^## (.+)$", README, re.M) == SECTIONS
    assert "shields.io" not in README
    badges = re.findall(r"!\[[^\]]*\]\(([^)]*badge[^)]*)\)", README)       # one badge, for the relay workflow only
    assert badges == ["https://github.com/streamweights/streamweights/actions/workflows/relay.yml/badge.svg"]
    for img in re.findall(r"!\[([^\]]*)\]", README):
        assert len(img) > 20, "image alt text must be descriptive"
    from scripts.readme_counts import count
    before = __import__("json").loads((ROOT / "docs/reports/data/018-readme-counts-before.json").read_text())
    assert count(README)["prose_words"] <= before["prose_words"] / 2 + 10, "the README grew back"


def test_what_it_does_names_both_jobs_without_implementation_terms():
    s = _section("What it does").lower()
    assert "text classification" in s and "json extraction" in s
    for term in ("lora", "mlx", "pytorch", "embedding", "bf16", "adapter", "fenc", "lineage", "distill"):
        assert term not in s, term


def test_try_it_has_prerequisites_then_commands_then_the_models_answer():
    s = _section("Try it")
    before_commands = s.split("```", 1)[0]
    for need in ("Python 3.10", "0.93 GB", "92 MB", "Apple silicon", "not training time"):
        assert need in before_commands, need
    assert "# setup" in s and "# training" in s and "# export, then inference" in s
    blocks = re.findall(r"```\n(.*?)```", s, re.S)
    assert blocks[0].startswith("# setup")
    answer = next(b for b in blocks if "change_pin" in b)
    assert answer.rstrip().endswith("change_pin"), "Try it ends the commands on the model's answer"
    assert "SPILL_MIN_FREE_GB=2" in s.split("```", 1)[1].split("full capture")[-1] or "SPILL_MIN_FREE_GB=2" in s
    assert s.index("spill plan tickets") > s.index("change_pin")
    assert "spill report tickets" in s


def test_disk_prerequisite_matches_the_code_and_the_capture_difference_is_stated():
    import json
    src = (ROOT / "streamweights/registry.py").read_text()
    assert 'environ.get("SPILL_MIN_FREE_GB", 20)' in src, "the default reserve is 20 GB"
    s = _section("Try it")
    assert "20 GB free-space reserve (`SPILL_MIN_FREE_GB`)" in s and "about 23 GB free" in s
    assert "nearly full disk" in s and "with `SPILL_MIN_FREE_GB=2`, not the default" in s
    cap = json.loads((ROOT / "docs/reports/data/018-readme-capture.json").read_text())
    assert "SPILL_MIN_FREE_GB" in (ROOT / "scripts/capture_readme.py").read_text()
    assert "SPILL_MIN_FREE_GB=2" in (ROOT / "docs/reports/018-readme-pass.md").read_text()
    assert cap["ok"]


def test_own_data_section_shows_formats_schema_and_the_same_commands():
    s = _section("Use your own data")
    assert "The example is over" in s
    assert "text,label" in s and "text,json" in s
    assert "shape only" in s, "the snippets are labeled as format examples, not datasets"
    assert '""order_id"": ""A-1655""' in s, "JSON cells use doubled quotes"
    assert "spill init my-tickets.csv --input text --output label --project my-tickets" in s
    assert "spill init orders.csv --input text --output json --schema schema.json --project orders" in s
    for cmd in ("spill build my-tickets", "spill export my-tickets", "spill build orders",
                "spill export orders"):
        assert cmd in s, cmd
    shown = re.search(r"`schema.json` says.*?```\n(.*?)```", s, re.S).group(1)
    fixture = ROOT / "tests/fixtures/own-data/schema.json"
    import json
    assert json.loads(shown) == json.loads(fixture.read_text())
    assert shown.strip() == fixture.read_text().strip(), "the README shows the fixture verbatim"


def test_understand_the_result_shows_the_baseline_win_and_links_the_export_check():
    s = _section("Understand the result")
    assert "scored 0.111 higher than the trained student" in s
    assert "whether fine-tuning beats a simpler approach" in s
    assert "export-diagnosis" in s
    assert "four-row" not in s


def test_go_further_and_support_sections():
    s = _section("Go further")
    for need in ("Continue an interrupted training run", "JSON extraction", "Offline bundles", "Large teachers",
                 "docs/portability.md#continuing-on-another-machine-step-by-step", "[Measured: Llama 3.3 70B"):
        assert need in s, need
    assert s.index("Continue an interrupted") < s.index("JSON extraction")
    assert "spill move" not in s and "s3://" not in s, "the S3 walkthrough lives in docs/portability.md"
    walk = (ROOT / "docs/portability.md").read_text()
    for cmd in ("spill build tickets2 --stop-after train:12", "spill move tickets2 s3://my-bucket/tickets2",
                "spill resume s3://my-bucket/tickets2"):
        assert cmd in walk, cmd
    t = _section("Support and limitations")
    assert "| platform | status |" in t and "issues/1" in t
    for need in ("small models", "CUDA is untested", "differ in quality"):
        assert need in t, need
    for link in ("docs/prior-art.md", "docs/architecture.md", "docs/plan.md", "/discussions"):
        assert link in t, link


def test_moved_pages_are_in_the_nav_and_one_click_from_the_readme():
    nav = (ROOT / "mkdocs.yml").read_text()
    for page in ("prior-art.md", "architecture.md", "portability.md", "plan.md", "reports/018-readme-pass.md",
                 "guides/turn-a-csv-of-examples-into-an-evaluated-model.md"):
        assert page in nav, page
        assert page.split("/")[-1] in README or page in README or page == "reports/018-readme-pass.md", page
    for anchor in ("your-own-data-in-detail", "sample-projects", "when-a-teacher-is-useful"):
        assert f"#{anchor}" in README


def test_readme_concept_names_are_only_linked_names():
    prose = re.sub(r"```.*?```", "", README, flags=re.S)
    for term in ("lora", "ownership fenc", "lineage", "distillation", "split rule", "evaluation protocol"):
        assert term not in prose.lower(), term


def test_every_output_line_in_the_readme_is_from_the_real_mlx_capture():
    import json
    cap = json.loads((ROOT / "docs/reports/data/018-readme-capture.json").read_text())
    assert cap["ok"] and cap["device"].startswith("Device(gpu"), "the capture is a real Metal run"
    captured = "\n".join(s["stdout"] for s in cap["steps"])
    commands = {s["command"] for s in cap["steps"]} | {"pip install git+https://github.com/streamweights/streamweights"}
    shown = 0
    for title in ("Try it", "Understand the result"):
        for block in re.findall(r"```\n(.*?)```", _section(title), re.S):
            for line in block.splitlines():
                if not line.strip() or line.startswith("#") or line.strip() in commands or line.strip() == "...":
                    continue
                assert line.strip() in captured, f"README shows output that was not captured: {line!r}"
                shown += 1
    assert shown >= 4
    table = re.findall(r"^\| (.+?) \| ([\d.]+) \| (\d+) \|$", _section("Understand the result"), re.M)
    assert [t[0] for t in table] == ["embedding baseline (MiniLM + logistic regression)",
                                      "student, prompted, untrained", "student, trained"]
    report = next(s["stdout"] for s in cap["steps"] if s["command"].startswith("spill report"))
    for name, acc, rows in table:
        cells = next(l for l in report.splitlines() if l.startswith(f"| {name} |")).split("|")[1:-1]
        assert (cells[0].strip(), cells[1].strip(), cells[3].strip()) == (name, acc, rows), name
    assert "```" not in re.search(r"^\| comparator.*?\n\n", _section("Understand the result"), re.S | re.M).group(0)


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
                                    "distill", "tune", "eval", "evaluate", "export", "test", "bundle", "move",
                                    "models", "adapters", "runs", "status", "tail", "resume",
                                    "doctor", "check"]
    assert all(len(d) < 62 for _, d in cmds)


def test_every_command_help_ends_with_one_example():
    for c in ("init", "plan", "build", "report", "compare", "example", "run", "distill", "tune",
              "eval", "evaluate", "export", "test", "bundle", "move", "models", "adapters", "runs", "status",
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
