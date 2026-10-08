"""Fresh-install gate: build the wheel, install it into a clean environment with the documented
extra, and run the public CLI from a directory that is not the checkout.

  python scripts/fresh_wheel.py [--keep] [--home DIR]

Steps (each timed): build the wheel; create a venv with uv; install the wheel[cloud]; check that
`import streamweights` resolves to site-packages and not the source tree; run `spill --help`,
`spill example banking77 --tiny`, `spill init`, `spill plan`; and `spill example snips --tiny`
then `spill init` for the extraction task. No model is downloaded (plan loads nothing). Writes a
JSON summary to stdout."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd, **kw):
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return {"cmd": " ".join(map(str, cmd))[:200], "exit": p.returncode,
            "seconds": round(time.monotonic() - t0, 1), "out": (p.stdout + p.stderr)[-1500:]}


def main():
    keep = "--keep" in sys.argv
    work = Path(tempfile.mkdtemp(prefix="spill-wheel-"))
    home = Path(sys.argv[sys.argv.index("--home") + 1]) if "--home" in sys.argv else work / "home"
    home.mkdir(parents=True, exist_ok=True)
    steps = []
    dist = work / "dist"
    steps.append(sh([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist), str(ROOT)]))
    wheel = next(dist.glob("*.whl"))
    venv = work / "venv"
    steps.append(sh(["uv", "venv", "--python", "3.12", str(venv)]))
    py = venv / "bin" / "python"
    steps.append(sh(["uv", "pip", "install", "--python", str(py), f"{wheel}[cloud]"]))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "SPILL_HOME")}
    env.update(SPILL_HOME=str(home), PATH=f"{venv / 'bin'}:{env['PATH']}", SPILL_HEADLESS="0")
    run = work / "run"
    run.mkdir()
    where = sh([str(py), "-c", "import streamweights, sys; print(streamweights.__file__)"], cwd=run, env=env)
    steps.append(where)
    loc = where["out"].strip().splitlines()[-1]
    ok_loc = "site-packages" in loc and str(ROOT) not in loc
    for cmd in (["spill", "--help"], ["spill", "example", "banking77", "--tiny"],
                ["spill", "init", "banking77-tiny/banking77.csv", "--input", "text", "--output", "label",
                 "--project", "tickets"],
                ["spill", "plan", "tickets"],
                ["spill", "example", "snips", "--tiny"],
                ["spill", "init", "snips-tiny/snips.csv", "--input", "text", "--output", "json",
                 "--schema", "snips-tiny/schema.json", "--project", "slots"],
                ["spill", "plan", "slots"]):
        steps.append(sh(cmd, cwd=run, env=env))
    spill_bin = shutil.which("spill", path=env["PATH"])
    summary = {"wheel": wheel.name, "wheel_bytes": wheel.stat().st_size,
               "imported_from": loc, "outside_checkout": ok_loc, "spill_bin": spill_bin,
               "steps": steps, "all_ok": ok_loc and all(s["exit"] == 0 for s in steps),
               "work": str(work)}
    print(json.dumps(summary, indent=1))
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
