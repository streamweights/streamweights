"""Capture the directive 020 gates on real models (qwen2.5:0.5b student, qwen2.5:1.5b teacher and
judge), commands and their exact output, in a fresh project directory.

  python scripts/capture_020.py --flow judged --engine mlx --out docs/reports/data/020-judged-mlx.json
  python scripts/capture_020.py --flow judged --engine torch-cpu --out docs/reports/data/020-judged-torch-cpu.json
  python scripts/capture_020.py --flow nojudge --engine mlx --out docs/reports/data/020-nojudge-mlx.json
  python scripts/capture_020.py --flow labeled --engine mlx --out docs/reports/data/020-labeled-mlx.json
  python scripts/capture_020.py --flow kd --engine mlx --out docs/reports/data/020-kd-mlx.json

Flows
  judged    an open-ended project: init (teacher and judge named), plan, build, report, export
  nojudge   the same project with no teacher and no judge: reference overlap, the plan's judge
            recommendation; SPILL_ALLOWED_MODELS lets only the student through, so any other model
            would have failed before it was fetched or loaded
  labeled   a labeled classification project with no teacher: no teacher stage; SPILL_ALLOWED_MODELS
            lets only the student and the embedding baseline through
  kd        the same teacher and student trained with --distill-loss ce and with mix, compared
  deadline  spill plan --deadline with room to spare and with none, and a build that refuses to start
            (a step starting with "!" is expected to fail)

With --engine mlx the script fails unless MLX's default device is the GPU (Metal). With
--engine torch-cpu it forces the PyTorch CPU engine. Model downloads are detected by listing the
models directory before and after."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FLOWS = {
    "judged": [
        ["spill", "example", "dolly-qa", "--tiny"],
        ["spill", "init", "dolly-qa-tiny/dolly-qa.csv", "--input", "prompt", "--output", "answer",
         "--project", "dolly", "--teacher", "qwen2.5:1.5b", "--judge", "qwen2.5:1.5b"],
        ["spill", "plan", "dolly"],
        ["spill", "build", "dolly"],
        ["spill", "report", "dolly"],
        ["spill", "export", "dolly"],
        ["python", "dolly/exports/*/run_safetensors.py",
         "Passage: Final Fantasy is a Japanese science fantasy anthology media franchise created by "
         "Hironobu Sakaguchi.\n\nQuestion: Who created Final Fantasy?"],
    ],
    "nojudge": [
        ["spill", "example", "dolly-qa", "--tiny"],
        ["spill", "init", "dolly-qa-tiny/dolly-qa.csv", "--input", "prompt", "--output", "answer",
         "--project", "dolly"],
        ["spill", "plan", "dolly"],
        ["spill", "build", "dolly"],
        ["spill", "report", "dolly"],
    ],
    "labeled": [
        ["spill", "example", "banking77", "--tiny"],
        ["spill", "init", "banking77-tiny/banking77.csv", "--input", "text", "--output", "label",
         "--project", "tickets"],
        ["spill", "plan", "tickets"],
        ["spill", "build", "tickets"],
    ],
    "deadline": [
        ["spill", "example", "dolly-qa", "--tiny"],
        ["spill", "init", "dolly-qa-tiny/dolly-qa.csv", "--input", "prompt", "--output", "answer",
         "--project", "dolly", "--teacher", "qwen2.5:1.5b"],
        ["spill", "plan", "dolly", "--deadline", "30m"],
        ["spill", "plan", "dolly", "--deadline", "5s"],
        ["!spill", "build", "dolly", "--deadline", "5s"],
    ],
    "kd": [
        ["spill", "example", "dolly-qa", "--tiny"],
        ["spill", "init", "dolly-qa-tiny/dolly-qa.csv", "--input", "prompt", "--output", "answer",
         "--project", "dolly", "--teacher", "qwen2.5:1.5b", "--judge", "none"],
        ["spill", "build", "dolly", "--distill-loss", "ce"],
        ["spill", "build", "dolly", "--distill-loss", "mix"],
        ["spill", "compare", "dolly"],
    ],
}


def models_listing(models: Path) -> dict:
    out = {}
    for p in sorted(models.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            out[str(p.relative_to(models))] = p.stat().st_size
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True, choices=sorted(FLOWS))
    ap.add_argument("--engine", required=True, choices=["mlx", "torch-cpu"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true", help="keep the scratch project directory")
    a = ap.parse_args()
    os.environ.pop("SPILL_DEVICE", None)
    device = "cpu"
    if a.engine == "mlx":
        import mlx.core as mx
        if mx.default_device() != mx.gpu:
            raise SystemExit(f"mx.default_device() is {mx.default_device()}, not the GPU")
        device = str(mx.default_device())
    work = Path(tempfile.mkdtemp(prefix="cap020-"))
    home = work / "home"
    (home / "state").mkdir(parents=True)
    (home / "models").symlink_to(ROOT / "models")
    for f in ("hardware.json", "calibration.json"):
        if (ROOT / "state" / f).exists():
            (home / "state" / f).write_bytes((ROOT / "state" / f).read_bytes())
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env.update(SPILL_HOME=str(home), SPILL_HEADLESS="0", SPILL_MIN_FREE_GB=os.environ.get("SPILL_MIN_FREE_GB", "2"),
               COLUMNS="110")
    if a.flow in ("nojudge", "labeled"):
        # the allow-list refuses any other model before a byte is fetched or a weight is loaded
        env["SPILL_ALLOWED_MODELS"] = "qwen2.5:0.5b,embedding:minilm,Qwen/Qwen2.5-0.5B-Instruct"
    run = work / "run"
    run.mkdir()
    before = models_listing(ROOT / "models")
    res = {"flow": a.flow, "engine": a.engine, "device": device,
           "allowed_models": env.get("SPILL_ALLOWED_MODELS"),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": []}
    for cmd in FLOWS[a.flow]:
        t0 = time.monotonic()
        expect_fail = cmd[0].startswith("!")
        cmd = [cmd[0].lstrip("!"), *cmd[1:]]
        if cmd[0] == "spill":
            argv = [sys.executable, "-m", "streamweights.cli", *cmd[1:]]
            if cmd[1] in ("plan", "build", "export", "evaluate", "test") and "--engine" not in cmd:
                argv += ["--engine", a.engine] if cmd[1] != "export" else []
            p = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=run, timeout=7200)
            shown = " ".join(cmd + (["--engine", a.engine] if cmd[1] in ("plan", "build") and "--engine" not in cmd else []))
        else:
            p = subprocess.run(["bash", "-c", f'{sys.executable} {cmd[1]} "$0"', cmd[2]], capture_output=True,
                               text=True, env=env, cwd=run, timeout=1800)
            shown = f'python {cmd[1]} "..."'
        res["steps"].append({"command": shown, "exit": p.returncode,
                             "expect_fail": expect_fail,
                             "seconds": round(time.monotonic() - t0, 1), "stdout": p.stdout,
                             "stderr_tail": p.stderr[-800:]})
        print(shown, p.returncode, flush=True)
        if (p.returncode != 0) != expect_fail:
            break
    after = models_listing(ROOT / "models")
    res["models_downloaded"] = {k: v for k, v in after.items() if k not in before}
    res["models_changed"] = {k: v for k, v in after.items() if k in before and before[k] != v}
    res["ok"] = all((s["exit"] != 0) == s.get("expect_fail", False) for s in res["steps"]) \
        and len(res["steps"]) == len(FLOWS[a.flow])
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(a.out).write_text(json.dumps(res, indent=1))
    if not a.keep:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
