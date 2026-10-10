"""Gate runner for directive 015: the whole developer journey, per task and per engine, with
the public CLI, on the permitted small models.

  python scripts/gates_015.py --home <fresh SPILL_HOME> --work <dir> --out <json> \\
        [--engines mlx torch-cpu] [--tasks classification json] [--no-gguf]

For every (task, engine): spill example <name> --tiny, spill init, spill plan, spill build,
spill report, spill test, spill export (safetensors, and GGUF q8_0 unless --no-gguf), then the
exported artifacts' own scripts. Every duration is the wall time of that command here. The
JSON it writes is what docs/reports/015-workflow.md quotes: nothing in the report is typed in."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = {
    "classification": {
        "example": "banking77", "csv": "banking77.csv", "init": ["--input", "text", "--output", "label"],
        "extra": [],
    },
    "json": {
        "example": "snips", "csv": "snips.csv", "init": ["--input", "text", "--output", "json"],
        "extra": ["--schema", "{dir}/schema.json"],
    },
}


def run(cmd, env, cwd, timeout=3600):
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, "-m", "streamweights.cli", *cmd], capture_output=True,
                       text=True, env=env, cwd=cwd, timeout=timeout)
    return {"cmd": "spill " + " ".join(cmd), "seconds": round(time.monotonic() - t0, 1),
            "exit": p.returncode, "stdout": p.stdout[-6000:], "stderr": p.stderr[-1500:]}


def newest(d: Path):
    xs = sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))
    return xs[-1] if xs else None


def journey(task: str, engine: str, work: Path, env, gguf: bool) -> dict:
    t = TASKS[task]
    d = work / f"{task}-{engine}"
    d.mkdir(parents=True, exist_ok=True)
    out = {"task": task, "engine": engine, "steps": []}
    step = lambda r: (out["steps"].append({k: r[k] for k in ("cmd", "seconds", "exit")}), r)[1]
    r = step(run(["example", t["example"], "--tiny"], env, d))
    ex = d / f"{t['example']}-tiny"
    proj = d / "project"
    r = step(run(["init", str(ex / t["csv"]), *t["init"], *[x.format(dir=ex) for x in t["extra"]],
                  "--project", str(proj)], env, d))
    out["init"] = r["stdout"].strip().splitlines()[-4:]
    r = step(run(["plan", str(proj), "--engine", engine], env, d))
    out["plan"] = r["stdout"]
    r = step(run(["build", str(proj), "--engine", engine], env, d))
    out["build_exit"] = r["exit"]
    out["build_tail"] = r["stdout"].strip().splitlines()[-8:]
    if r["exit"] != 0:
        out["error"] = r["stderr"]
        return out
    run_dir = newest(proj / "runs")
    m = json.loads((run_dir / "manifest.json").read_text())
    out["run"] = {"id": m["run_id"], "metric": m["metric"], "table": m["table"],
                  "stages": {k: {"seconds": v["seconds"], "producers": v["producers"]}
                             for k, v in m["stages"].items()},
                  "models": {k: {"tag": v["tag"], "repo": v["repo"], "revision": v["revision"]}
                             for k, v in m["models"].items() if v},
                  "dependencies": m["dependencies"], "findings": m["findings"],
                  "protocol_sha256": m["protocol"]["sha256"]}
    step(run(["report", str(proj)], env, d))
    r = step(run(["test", str(proj), "--engine", engine], env, d))
    t_rec = json.loads((newest(proj / "tests") / "record.json").read_text()) if r["exit"] == 0 else None
    out["test"] = t_rec and {"table": t_rec["table"], "rows": t_rec["rows"], "seconds": t_rec["seconds"],
                             "holdout_note": t_rec["holdout_note"]}
    # the same run evaluated on the other engine (a separate record), and compared two ways
    other = "torch-cpu" if engine == "mlx" else "mlx"
    r = step(run(["evaluate", str(proj), "--engine", other], env, d))
    if r["exit"] == 0:
        erec = json.loads((newest(proj / "evaluations") / "record.json").read_text())
        out["evaluate"] = {"id": erec["id"], "engine_requested": other, "table": erec["table"],
                           "vs_original": erec["vs_original"], "rows": erec["rows"],
                           "conditions": {k: v["conditions"] for k, v in erec["comparators"].items()},
                           "seconds": erec["seconds"]}
        c1 = step(run(["compare", str(proj)], env, d))
        c2 = step(run(["compare", str(proj), "--use", f"{m['run_id']}={erec['id']}"], env, d))
        out["compare_default"] = c1["stdout"]
        out["compare_explicit"] = c2["stdout"]
    cmd = ["export", str(proj)] + (["--gguf", "q8_0"] if gguf else [])
    r = step(run(cmd, env, d, timeout=7200))
    out["export_exit"] = r["exit"]
    if r["exit"] != 0:
        out["export_error"] = (r["stderr"] or r["stdout"])[-800:]
        return out
    e_dir = newest(proj / "exports")
    rec = json.loads((e_dir / "record.json").read_text())
    out["export"] = {"id": rec["id"], "format": rec["format"], "status": rec["status"],
                     "verification": {k: {"runtime": v["runtime"], "diffs": v["prediction_differences"]["count"],
                                          "of": v["prediction_differences"]["of"], "primary": v["primary"],
                                          "load_failure": v["load_failure"]}
                                      for k, v in rec["verification"]["runs"].items()},
                     "deployment": rec["deployment"], "scripts": rec["verification"]["scripts"],
                     "seconds": rec["seconds"]}
    # inference through the exported artifact, from outside the project
    sample = json.loads((proj / "data" / "val.jsonl").read_text().splitlines()[0])["input"]
    out["inference"] = {}
    for fmt, script in rec["scripts"].items():
        t0 = time.monotonic()
        p = subprocess.run([sys.executable, str(e_dir / script), sample], capture_output=True, text=True,
                           env=env, cwd="/")
        out["inference"][fmt] = {"input": sample, "output": p.stdout.strip()[:400], "exit": p.returncode,
                                 "seconds": round(time.monotonic() - t0, 1)}
    return out


def hardware() -> dict:
    from streamweights import machine
    info = machine.info()
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                     text=True).stdout.strip() or platform.processor()
    except OSError:
        info["cpu"] = platform.processor()
    info["python"] = platform.python_version()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engines", nargs="+", default=["mlx", "torch-cpu"])
    ap.add_argument("--tasks", nargs="+", default=["classification", "json"])
    ap.add_argument("--no-gguf", action="store_true")
    ap.add_argument("--models-from", default=str(ROOT / "models"))
    a = ap.parse_args()
    home, work = Path(a.home), Path(a.work)
    (home / "state").mkdir(parents=True, exist_ok=True)
    for f in ("hardware.json", "calibration.json"):
        if (ROOT / "state" / f).exists() and not (home / "state" / f).exists():
            (home / "state" / f).write_bytes((ROOT / "state" / f).read_bytes())
    if not (home / "models").exists():
        (home / "models").symlink_to(a.models_from)
    for n in ("bin", "tools"):
        if (ROOT / n).exists() and not (home / n).exists():
            (home / n).symlink_to(ROOT / n)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(SPILL_HOME=str(home), SPILL_HEADLESS="0", SPILL_MIN_FREE_GB="2")
    import streamweights
    res = {"imported_from": streamweights.__file__, "hardware": hardware(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "journeys": []}
    for task in a.tasks:
        for engine in a.engines:
            print(f"== {task} on {engine}", flush=True)
            j = journey(task, engine, work, env, not a.no_gguf)
            res["journeys"].append(j)
            print(json.dumps({k: j.get(k) for k in ("task", "engine", "build_exit", "export_exit")}), flush=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
