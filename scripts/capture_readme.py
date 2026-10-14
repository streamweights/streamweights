"""Capture the real output shown in the README (directive 018).

  python scripts/capture_readme.py --out docs/reports/data/018-readme-capture.json

Runs the README's Try it commands in a fresh project directory on the Apple GPU through MLX (the
script fails if mx.default_device() is not the GPU), and stores each command with its exact stdout.
Nothing in the README output blocks is typed by hand: tests/test_docs.py checks that every output
line shown there is a line of this capture."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    ["spill", "example", "banking77", "--tiny"],
    ["spill", "init", "banking77-tiny/banking77.csv", "--input", "text", "--output", "label", "--project", "tickets"],
    ["spill", "plan", "tickets"],
    ["spill", "build", "tickets"],
    ["spill", "report", "tickets"],
    ["spill", "export", "tickets"],
    ["python", "tickets/exports/*/run_safetensors.py", "Can I change my PIN at any ATM?"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.pop("SPILL_DEVICE", None)
    import mlx.core as mx
    if mx.default_device() != mx.gpu:
        raise SystemExit(f"mx.default_device() is {mx.default_device()}, not the GPU")
    work = Path(tempfile.mkdtemp(prefix="readme-capture-"))
    home = work / "home"
    (home / "state").mkdir(parents=True)
    (home / "models").symlink_to(ROOT / "models")
    for f in ("hardware.json", "calibration.json"):
        if (ROOT / "state" / f).exists():
            (home / "state" / f).write_bytes((ROOT / "state" / f).read_bytes())
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env.update(SPILL_HOME=str(home), SPILL_HEADLESS="0", SPILL_MIN_FREE_GB="2", COLUMNS="100")
    run = work / "run"
    run.mkdir()
    res = {"device": str(mx.default_device()), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": []}
    for cmd in STEPS:
        t0 = time.monotonic()
        if cmd[0] == "spill":
            argv = [sys.executable, "-m", "streamweights.cli", *cmd[1:]]
            p = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=run, timeout=1800)
        else:
            p = subprocess.run(["bash", "-c", f'{sys.executable} {cmd[1]} "{cmd[2]}"'], capture_output=True,
                               text=True, env=env, cwd=run, timeout=1800)
        res["steps"].append({"command": " ".join(cmd if cmd[0] != "python" else [cmd[0], cmd[1], f'"{cmd[2]}"']),
                             "exit": p.returncode, "seconds": round(time.monotonic() - t0, 1),
                             "stdout": p.stdout, "stderr_tail": p.stderr[-600:]})
        print(cmd[1] if len(cmd) > 1 else cmd, p.returncode, flush=True)
        if p.returncode != 0:
            break
    res["ok"] = all(s["exit"] == 0 for s in res["steps"])
    Path(a.out).write_text(json.dumps(res, indent=1))
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
