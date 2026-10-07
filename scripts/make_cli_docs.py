"""Generate docs/cli.md from the real `spill --help` output.

  python scripts/make_cli_docs.py          write docs/cli.md
  python scripts/make_cli_docs.py --check  exit 1 if docs/cli.md is out of date
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "cli.md"
ORDER = ["build", "example", "run", "distill", "tune", "eval", "export",
         "models", "adapters", "runs", "status", "tail", "resume", "doctor", "check"]


HOME = tempfile.mkdtemp(prefix="spill-docs-")


def help_text(*args: str) -> str:
    r = subprocess.run([sys.executable, "-m", "streamweights", *args, "--help"],
                       capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "COLUMNS": "80", "SPILL_HOME": HOME})
    if r.returncode:
        raise SystemExit(r.stderr)
    return re.sub(r"[ \t]+$", "", r.stdout, flags=re.M).strip("\n")


def render() -> str:
    parts = ["---",
             "description: Every spill command with its real --help output, in the order you use "
             "them: build, example, run, distill, tune, eval, export, models, adapters, runs, "
             "status, tail, resume, doctor, check.",
             "---", "", "# Commands", "",
             "Generated from the real `--help` output by `scripts/make_cli_docs.py`; "
             "`tests/test_docs.py` fails if it is out of date. Commands are in the order "
             "you use them.", "", "```", help_text(), "```", ""]
    for c in ORDER:
        parts += [f"## spill {c}", "", "```", help_text(c), "```", ""]
    return "\n".join(parts)


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        sys.exit(0 if OUT.read_text() == text else 1)
    OUT.write_text(text)
    print("wrote", OUT)
