"""Fresh-install verification: what a stranger does, with timestamps.

  python scripts/fresh_install.py [--ref BRANCH_OR_SHA]

In a new temporary directory with a clean venv and an empty HOME it runs
  pip install git+https://github.com/streamweights/streamweights
  spill doctor
  spill example banking77 --quick && spill build banking77-quick
  spill export qwen2.5:0.5b+banking77-quick --gguf
and writes the full transcript to docs/reports/010-fresh-install.txt.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "reports" / "010-fresh-install.txt"
URL = "git+https://github.com/streamweights/streamweights"


def main() -> int:
    ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else None
    work = Path(tempfile.mkdtemp(prefix="spill-fresh-"))
    home = work / "home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home), "XDG_CACHE_HOME": str(home / ".cache")}
    for k in ("SPILL_HOME", "SPILL_NO_MLX", "SPILL_DEVICE", "VIRTUAL_ENV", "PYTHONPATH"):
        env.pop(k, None)
    t0 = time.monotonic()
    lines: list[str] = []
    marks: dict[str, float] = {}

    def log(text: str) -> None:
        el = time.monotonic() - t0
        line = f"[{time.strftime('%H:%M:%S')} +{el:7.1f}s] {text}"
        lines.append(line)
        print(line, flush=True)

    def run(label: str, cmd: str, cwd: Path) -> int:
        log(f"$ {cmd}")
        marks[label + ":start"] = time.monotonic() - t0
        p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=0)
        buf = ""
        while True:
            ch = p.stdout.read(1)
            if not ch:
                break
            if ch in "\r\n":
                if buf.strip():
                    log(buf.rstrip())
                buf = ""
            else:
                buf += ch
        if buf.strip():
            log(buf.rstrip())
        p.wait()
        marks[label + ":end"] = time.monotonic() - t0
        log(f"(exit {p.returncode}, {marks[label + ':end'] - marks[label + ':start']:.1f} s)")
        return p.returncode

    py = "python3.12" if subprocess.run("command -v python3.12", shell=True,
                                        capture_output=True).returncode == 0 else "python3"
    log(f"fresh directory {work}, empty HOME, {py}")
    spec = URL + (f"@{ref}" if ref else "")
    steps = [
        ("venv", f"{py} -m venv venv", work),
        ("install", f"venv/bin/pip install --quiet {spec}", work),
        ("doctor", "venv/bin/spill doctor", work),
        ("example", "venv/bin/spill example banking77 --quick", work),
        ("build", "venv/bin/spill build banking77-quick", work),
        ("export", "venv/bin/spill export qwen2.5:0.5b+banking77-quick --gguf", work),
    ]
    rc = 0
    for label, cmd, cwd in steps:
        r = run(label, cmd, cwd)
        if r and not rc:
            rc = r
            break
    log(f"install to end of build: {marks.get('build:end', 0) - marks['install:start']:.1f} s"
        if "build:end" in marks else "build did not finish")
    OUT.write_text("\n".join(lines) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
