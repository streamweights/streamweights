# no-mlx-needed
"""The commands in the README and the user guides, one by one.

Inventory: every actionable product command in README.md, docs/guides/ and the user-facing
reference pages is found by reading the markdown. Each must be mapped in
tests/docs_command_map.toml (and each mapping must be used). The real CLI parses every mapped
command (command, options, required arguments) without loading or downloading a model. Commands
that need a larger model are run with an explicit tiny equivalent. With SPILL_DOCS_RUN=1 the
small-model commands are executed, in map order, in one scratch directory, with an allow-list
that refuses any model but qwen2.5:0.5b and the pinned embedding model. Historical verbatim
directives in docs/paste-sets/ are archived instructions and are never read here."""

import os
import re
import shlex
import shutil
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:     # Python 3.10
    import tomli as tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAP = tomllib.loads((Path(__file__).parent / "docs_command_map.toml").read_text())
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs/guides").glob("*.md")),
         *[ROOT / "docs" / n for n in ("portability.md", "linux.md", "schedulers.md", "llamacpp.md",
                                       "formats.md", "models.md", "plan.md")]]
START = re.compile(r"^(spill|pip install|uv tool|uv pip|docker|python -m|python scripts|\./)")


def inventory() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for page in PAGES:
        text = page.read_text()
        for m in re.finditer(r"```[a-z]*\n(.*?)```", text, re.S):
            lines = m.group(1).splitlines()
            i = 0
            while i < len(lines):
                s = lines[i].strip()
                s = s[2:] if s.startswith("$ ") else s
                if START.match(s):
                    while s.endswith("\\") and i + 1 < len(lines):
                        i += 1
                        s = s[:-1].rstrip() + " " + lines[i].strip()
                    found.setdefault(_norm(s), set()).add(page.name)
                i += 1
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        for m in re.finditer(r"`(spill [^`]+)`", prose):
            found.setdefault(_norm(m.group(1)), set()).add(page.name)
    return found


def _norm(s: str) -> str:
    s = re.sub(r"\s+#.*$", "", " ".join(s.split()))
    return s.strip()


def rules():
    rs = MAP["rule"]
    return [r for r in rs if r["action"] != "reference"] + [r for r in rs if r["action"] == "reference"]


def match(cmd: str):
    for r in rules():
        if re.fullmatch(r["pattern"], cmd) or r["pattern"] == cmd:
            return r
    return None


def fill(cmd: str, extra: dict | None = None) -> str:
    table = {**MAP["placeholders"], **(extra or {})}
    return re.sub(r"<([^>]+)>", lambda m: table.get(m.group(1), m.group(0)), cmd)


_GROUP = None


def parse_only(line: str) -> str:
    """Parse one `spill ...` command against the real CLI; raises on an unknown command or option
    or a missing required argument. Nothing runs."""
    global _GROUP
    from typer.main import get_command
    from streamweights.cli import app
    _GROUP = _GROUP or get_command(app)
    argv = shlex.split(line)[1:]
    if "--config" in argv:          # main() rewrites --config <job.json> into the stored arguments
        i = argv.index("--config")
        del argv[i:i + 2]
        argv += {"build": ["x"], "tune": ["m", "t.jsonl", "--name", "n"]}[argv[0]]
    ctx = _GROUP.make_context("spill", list(argv), resilient_parsing=False)
    rest = list(getattr(ctx, "_protected_args", [])) + list(ctx.args)
    name, cmd, args = _GROUP.resolve_command(ctx, rest)
    cmd.make_context(name, args, parent=ctx, resilient_parsing=False)
    return name


def parts(cmd: str) -> list[str]:
    return [p.strip() for p in cmd.split("&&")]


def test_every_actionable_command_is_mapped_and_every_mapping_is_used():
    inv = inventory()
    assert len(inv) > 40
    unmapped = sorted(c for c in inv if match(c) is None)
    assert unmapped == [], f"add these to tests/docs_command_map.toml: {unmapped}"
    used = {id(match(c)) for c in inv}
    stale = [r["pattern"] for r in MAP["rule"] if id(r) not in used]
    assert stale == [], f"mapped but no longer in the docs: {stale}"
    assert "paste-sets" in MAP["exclusions"]["historical"]
    assert not any("paste-sets" in str(p) for p in PAGES)


def test_every_mapped_spill_command_parses_against_the_real_cli():
    bad = []
    for cmd in inventory():
        r = match(cmd)
        if r["action"] in ("reference", "setup", "exclude"):
            continue
        for part in parts(fill(cmd, r.get("fill"))):
            if not part.startswith("spill "):
                continue
            try:
                parse_only(part)
            except BaseException as e:                    # click raises several types
                bad.append(f"{part!r}: {type(e).__name__}: {str(e)[:80]}")
    assert bad == []


def test_substituted_and_tiny_equivalents_also_parse_and_name_only_approved_models():
    approved = {"qwen2.5:0.5b"}
    for cmd in inventory():
        r = match(cmd)
        if r["action"] == "substitute":
            for part in parts(fill(cmd, r.get("fill"))):
                if not part.startswith("spill "):
                    continue
                new = part
                for a, b in r.get("sub", {}).items():
                    new = new.replace(a, b)
                new = (new + " " + r.get("append", "")).strip()
                parse_only(new)
                for tok in re.findall(r"(?:qwen2\.5|llama3\.3):[\w.]+", new):
                    assert tok in approved, (cmd, new)
        if r["action"] == "tiny":
            for t in r["tiny"]:
                c = t["cmd"] if isinstance(t, dict) else t
                parse_only(c.replace("{tmp}", "/tmp/x"))


def test_the_harness_refuses_unapproved_models_before_any_download(monkeypatch):
    from streamweights import guard
    from streamweights.errors import SpillError
    monkeypatch.setenv("SPILL_ALLOWED_MODELS", "qwen2.5:0.5b,sentence-transformers/all-MiniLM-L6-v2")
    for tag in ("qwen2.5:7b", "llama3.3:70b"):
        with pytest.raises(SpillError):
            guard.check(tag)


# ---------------------------------------------------------------- execution (SPILL_DOCS_RUN=1)

RUN = os.environ.get("SPILL_DOCS_RUN") == "1"


def _env(work: Path) -> dict:
    env = {**os.environ, "SPILL_HEADLESS": "0", "SPILL_ALLOWED_MODELS":
           "qwen2.5:0.5b,sentence-transformers/all-MiniLM-L6-v2,embedding:minilm,Qwen/Qwen2.5-0.5B-Instruct",
           "COLUMNS": "120"}
    return env


def _spill(argline: str, cwd: Path, env: dict, ok=(0,)):
    argv = shlex.split(argline)[1:]
    p = subprocess.run([sys.executable, "-m", "streamweights.cli", *argv], capture_output=True,
                       text=True, cwd=cwd, env=env, timeout=3600)
    return p.returncode in ok, p


@pytest.mark.skipif(not RUN, reason="set SPILL_DOCS_RUN=1 to execute the small-model commands "
                                    "(about ten minutes)")
def test_execute_the_small_model_commands_on_tiny_data(tmp_path):
    from streamweights import example as E
    work = tmp_path / "docs"
    work.mkdir()
    env = _env(work)
    tiny = E.DATA / "banking77" / "tiny"
    import json
    for name in ("train.jsonl", "evals.jsonl"):
        shutil.copyfile(tiny / name, work / name)
    (work / "prompts.jsonl").write_text("".join(
        json.dumps({"prompt": json.loads(l)["prompt"]}) + "\n"
        for l in (tiny / "train.jsonl").read_text().splitlines()))
    inv = inventory()
    ordered = []
    for r in MAP["rule"]:
        if r["action"] in ("run", "substitute", "tiny"):
            ordered += [(r, c) for c in sorted(inv) if match(c) is r]
    done, failures = set(), []
    for r, cmd in ordered:
        if r.get("needs_env") and os.environ.get(r["needs_env"]) != "1":
            cmd_run = r["fallback"]
            seq = [cmd_run]
        elif r["action"] == "tiny":
            seq = [(t["cmd"], tuple(t.get("exit", [0]))) if isinstance(t, dict) else (t, (0,))
                   for t in r["tiny"]]
        else:
            real = fill(cmd, r.get("fill"))
            seq = []
            for part in parts(real):
                new = part
                for a, b in r.get("sub", {}).items():
                    new = new.replace(a, b)
                if part is parts(real)[-1]:
                    new = (new + " " + r.get("append", "")).strip()
                seq.append(new)
        for item in seq:
            line, ok = item if isinstance(item, tuple) else (item, tuple(r.get("ok_exit", [0])))
            line = line.replace("{tmp}", str(work))
            if line in done:
                continue
            if not line.startswith("spill "):
                continue
            good, p = _spill(line, work, env, ok)
            done.add(line)
            if not good:
                failures.append(f"{line} -> exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}")
    assert failures == [], "\n".join(failures)
