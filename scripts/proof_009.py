"""The item 11 loop proof, run the way a stranger would: no flags, no supervision.

  python scripts/proof_009.py            # runs every step that is not done yet
  python scripts/proof_009.py --step quick

Steps (each is resumable; state in docs/reports/009-banking77/proof-state.json):

  prep      plugged in? disk? models (the 70B is linked from the main checkout, read-only)
  quick     spill example banking77 --quick && spill build banking77-quick
  surpass   spill build banking77 --compare llama3.3:70b   (prompts.jsonl moved aside)
  copy      spill build banking77-copy                      (train.jsonl moved aside)
  collect   the combined table, per-row results, stage estimates versus actual

Wall time per step is wall clock around the shell command (sleeping or not). The per-stage
estimate printed before each stage and the actual are read from each folder's
.build/state.json.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "reports" / "009-banking77"
WORK = ROOT / "proof"
STATE = REPORT / "proof-state.json"
LOGS = REPORT / "logs"
SPILL = [str(ROOT / ".venv" / "bin" / "spill")]


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"steps": {}}


def save_state(s: dict) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def sh(cmd: list[str], log: str, cwd: Path = WORK, env: dict | None = None) -> tuple[int, float]:
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with open(LOGS / f"{log}.log", "ab") as f:
        f.write(f"\n$ {' '.join(cmd)}  [{time.strftime('%FT%T%z')}]\n".encode())
        f.flush()
        r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, **(env or {})})
    return r.returncode, time.monotonic() - t0


def power() -> dict:
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return {"on_ac": "AC Power" in out, "raw": out.strip().splitlines()[:2]}


def build_state(folder: Path) -> dict:
    p = folder / ".build" / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def run_build(step: str, folder: str, extra: list[str], state: dict) -> None:
    """Build, and on an interrupted or failed attempt resume it (up to 3 attempts)."""
    rec = state["steps"].setdefault(step, {"attempts": []})
    for attempt in range(3):
        cmd = SPILL + (["build", folder, *extra] if attempt == 0 else ["resume", str(WORK / folder)])
        rc, wall = sh(cmd, step)
        rec["attempts"].append({"cmd": " ".join(cmd[1:]), "rc": rc, "wall_s": round(wall, 1),
                                "ended": time.strftime("%FT%T%z")})
        save_state(state)
        if rc == 0 and build_state(WORK / folder).get("finished"):
            rec["done"] = True
            rec["wall_s"] = round(sum(a["wall_s"] for a in rec["attempts"]), 1)
            rec["interventions"] = len(rec["attempts"]) - 1
            save_state(state)
            return
    rec["failed"] = True
    save_state(state)


def step_prep(state: dict) -> None:
    rec = state["steps"].setdefault("prep", {})
    rec["power"] = power()
    rec["disk_free_gb"] = round(shutil.disk_usage(ROOT).free / 2**30, 1)
    link = ROOT / "models" / "llama3.3-70b"
    main = Path("/Users/amrishkapoor/spillway/models/llama3.3-70b")
    if not link.exists() and main.exists():
        (ROOT / "models").mkdir(exist_ok=True)
        link.symlink_to(main)
        rec["linked_70b"] = str(main)
    WORK.mkdir(exist_ok=True)
    rec["done"] = True
    save_state(state)
    print("prep:", rec)


def step_quick(state: dict) -> None:
    f = WORK / "banking77-quick"
    if not f.exists():
        sh(SPILL + ["example", "banking77", "--quick"], "quick")
    run_build("quick", "banking77-quick", [], state)


def step_surpass(state: dict) -> None:
    f = WORK / "banking77"
    if not f.exists():
        sh(SPILL + ["example", "banking77"], "surpass")
        shutil.move(f / "prompts.jsonl", f / "prompts.jsonl.aside")
    run_build("surpass", "banking77", ["--compare", "llama3.3:70b"], state)


def step_copy(state: dict) -> None:
    f = WORK / "banking77-copy"
    if not f.exists():
        f.mkdir()
        src = WORK / "banking77"
        for name in ("evals.jsonl", "instructions.txt"):
            shutil.copyfile(src / name, f / name)
        shutil.copyfile(src / "prompts.jsonl.aside", f / "prompts.jsonl")   # train.jsonl left out
    run_build("copy", "banking77-copy", [], state)


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def step_collect(state: dict) -> None:
    out = {}
    for step, folder in (("quick", "banking77-quick"), ("surpass", "banking77"),
                         ("copy", "banking77-copy")):
        st = build_state(WORK / folder)
        if st:
            out[step] = st
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "build-states.json").write_text(json.dumps(out, indent=2))
    # combined table
    def score(step, stage_id):
        for s in out.get(step, {}).get("stages", []):
            if s["id"] == stage_id and s["status"] == "done":
                return s["result"].get("score"), s["result"].get("rows")
        return None, None
    table = [
        ("llama3.3:70b untrained", *score("surpass", "eval:compare:llama3.3:70b")),
        ("qwen2.5:7b untrained", *score("surpass", "eval:base")),
        ("qwen2.5:7b trained on your labels", *score("surpass", "eval:tuned")),
        ("qwen2.5:7b distilled from the 70B", *score("copy", "eval:tuned")),
    ]
    md = ["| model | score | rows |", "|---|---|---|"]
    md += [f"| {n} | {('%.3f' % s) if s is not None else 'n/a'} | {r or 'n/a'} |"
           for n, s, r in table]
    (REPORT / "table.md").write_text("\n".join(md) + "\n")
    # per-stage estimate versus actual
    lines = ["| build | stage | estimate (s) | actual (s) | ratio |", "|---|---|---|---|---|"]
    for step, st in out.items():
        for s in st["stages"]:
            e, a = s.get("est_s") or 0, s.get("actual_s")
            lines.append(f"| {step} | {s['label']} | {e:.0f} | {a if a is not None else '-'} | "
                         f"{(a / e):.2f} |" if a and e else
                         f"| {step} | {s['label']} | {e:.0f} | {a if a is not None else '-'} | - |")
    (REPORT / "stages.md").write_text("\n".join(lines) + "\n")
    # per-row results, gzipped
    for p in WORK.glob("*/*.out.jsonl"):
        dest = REPORT / "rows" / f"{p.parent.name}__{p.name}.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest, "wt") as g:
            g.write(p.read_text())
    state["steps"].setdefault("collect", {})["done"] = True
    save_state(state)
    print((REPORT / "table.md").read_text())


STEPS = {"prep": step_prep, "quick": step_quick, "surpass": step_surpass, "copy": step_copy,
         "collect": step_collect}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=list(STEPS))
    a = ap.parse_args()
    state = load_state()
    for name, fn in STEPS.items():
        if a.step and a.step != name:
            continue
        if state["steps"].get(name, {}).get("done") and name != "collect":
            continue
        print(f"== {name} [{time.strftime('%FT%T')}]", flush=True)
        fn(state)
        if state["steps"].get(name, {}).get("failed"):
            print(f"step {name} failed after retries; continuing with the rest", flush=True)


if __name__ == "__main__":
    main()
