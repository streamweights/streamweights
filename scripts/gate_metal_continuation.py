"""MLX-on-Metal continuation gate: a guided build killed while it publishes a checkpoint and
finished on the other engine, in both directions, then moved between locations.

  python scripts/gate_metal_continuation.py --out docs/reports/data/016-metal.json \\
        [--s3 s3://bucket/prefix]   (S3 endpoint and credentials from the usual AWS_* variables)

Runs on an Apple silicon Mac with the MLX GPU device and fails if mx.default_device() is not the
GPU (SPILL_DEVICE=cpu would make it the CPU). The CLI is the public one, each build is a separate
process, and every stop is a process killed (os._exit) by a deterministic hook at the exact point
a checkpoint is published; machine power loss is not tested.

  reference       an uninterrupted MLX-GPU build of the same project
  chain A          mlx -> killed after publishing step 10 -> resume on torch-cpu -> killed after
                   publishing step 15 -> resume on mlx -> complete
  chain B          the same, starting on torch-cpu
  move chains      one engine killed after step 10, `spill move` to a local folder (and to S3 if
                   given), `spill resume` there on the other engine, in both directions

Per run it checks that every published checkpoint's step, data cursor and optimizer step agree
and carry over (the history of each checkpoint extends the previous one), that each engine change
is recorded in the manifest, and reports the final validation score next to the reference."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT_EVERY = 5


def gpu_check() -> dict:
    if os.environ.get("SPILL_DEVICE", "").lower() == "cpu":
        raise SystemExit("SPILL_DEVICE=cpu is set: this gate needs the MLX GPU device")
    import mlx.core as mx
    dev = mx.default_device()
    if dev != mx.gpu:
        raise SystemExit(f"mx.default_device() is {dev}, not the GPU")
    return {"mlx_default_device": str(dev), "mlx": __import__("mlx").__version__ if hasattr(
        __import__("mlx"), "__version__") else "unknown"}


class Env:
    def __init__(self, work: Path):
        self.work = work
        self.home = work / "home"
        (self.home / "state").mkdir(parents=True, exist_ok=True)
        if not (self.home / "models").exists():
            (self.home / "models").symlink_to(ROOT / "models")
        for f in ("hardware.json", "calibration.json"):
            if (ROOT / "state" / f).exists() and not (self.home / "state" / f).exists():
                shutil.copyfile(ROOT / "state" / f, self.home / "state" / f)
        self.base = {**os.environ, "SPILL_HOME": str(self.home), "SPILL_HEADLESS": "0",
                     "SPILL_MIN_FREE_GB": "2", "SPILL_LEASE_S": "2"}
        self.base.pop("SPILL_DEVICE", None)

    def spill(self, args, hooks=None, expect=(0,), cwd=None):
        env = {**self.base, **{f"SPILL_HOOK_{k}": v for k, v in (hooks or {}).items()}}
        t0 = time.monotonic()
        p = subprocess.run([sys.executable, "-m", "streamweights.cli", *map(str, args)], env=env,
                           capture_output=True, text=True, cwd=cwd or self.work, timeout=3600)
        if p.returncode not in expect:
            raise RuntimeError(f"spill {' '.join(map(str, args))} -> {p.returncode}\n{p.stdout[-800:]}\n{p.stderr[-800:]}")
        return {"exit": p.returncode, "seconds": round(time.monotonic() - t0, 1), "stdout": p.stdout}


def make_project(env: Env, name: str) -> Path:
    from streamweights import example as E
    from streamweights.project import config as C
    from streamweights.project.init import create_project
    proj = env.work / name
    create_project(E.DATA / "banking77" / "tiny" / "banking77.csv", proj, {"input": "text", "output": "label"},
                   "classification")
    cfg = C.load(proj)
    cfg["training"].update(epochs=2.0, ckpt_every=CKPT_EVERY, micro_batch=4)
    C.save(proj, cfg)
    return proj


def run_info(state_root: str, rid: str) -> dict:
    from streamweights.portable.store import Store
    from streamweights.project import coordinator as CO
    doc = {d["run_id"]: d for d in CO.list_runs_at(state_root)}[rid]
    uri = f"{state_root}/runs/{rid}"
    store = Store(uri)
    states = []
    for d in store.ls("payloads"):
        if d.startswith("ckpt-") and store.exists(f"payloads/{d}/PAYLOAD.json"):
            man = json.loads(store.read(f"payloads/{d}/PAYLOAD.json"))
            st = json.loads(store.read(f"payloads/{d}/state.json"))
            states.append({"generation": man["generation"], "seq": man["seq"], "state": st})
    states.sort(key=lambda x: (x["generation"], x["seq"]))
    return {"doc": doc, "states": states}


def check_carry_over(info: dict) -> list[str]:
    """Problems with the published checkpoints of one run (empty when they carry over)."""
    prob, prev = [], None
    published = (info["doc"].get("checkpoint") or {}).get("seq")
    for x in info["states"]:
        s = x["state"]
        if s["opt_step"] != s["step"]:
            prob.append(f"step {s['step']}: optimizer step {s['opt_step']}")
        if s["data_cursor"]["step"] != s["step"]:
            prob.append(f"step {s['step']}: data cursor {s['data_cursor']}")
        if prev is not None and s["step"] > prev["step"] and prev["generation"] != x["generation"]:
            ph, h = prev["history"], s["history"]
            # the new history keeps the earlier ranges (up to the step it resumed from)
            if h[0]["range"][0] != 0 or h[0]["engine"] != ph[0]["engine"]:
                prob.append(f"step {s['step']}: history does not start where the run did")
        prev = {"step": s["step"], "generation": x["generation"], "history": s["history"]}
    return prob


def final(env: Env, project: Path, rid: str, state_root: str | None = None) -> dict:
    m = json.loads((project / "runs" / rid / "manifest.json").read_text())
    tbl = {t["comparator"]: t for t in m["table"]}
    prod = m["stages"]["train"]["producers"]
    return {"trained_accuracy": tbl["trained"]["primary"], "rows": tbl["trained"]["rows"],
            "untrained_accuracy": tbl["untrained"]["primary"], "baseline_accuracy": tbl["baseline"]["primary"],
            "train_producers": [{"engine": p["engine"], "numerics": p.get("numerics"), "quanta": p.get("quanta")}
                                for p in prod], "engine_transitions": m["engine_transitions"]}


def chain(env: Env, name: str, engines: list[str], ref: dict) -> dict:
    """engines: [first, second, third]: first is killed after step 10, second after step 15, the
    third completes."""
    proj = make_project(env, name)
    sr = str(proj / ".spill")
    steps = []
    env.spill(["build", proj, "--engine", engines[0]], hooks={"checkpoint_after_control": "crash:2"}, expect=(137,))
    rid = __import__("streamweights.project.coordinator", fromlist=["x"]).list_runs(proj)[0]["run_id"]
    c = run_info(sr, rid)["doc"]["checkpoint"]
    steps.append({"engine": engines[0], "killed_after_published_step": c["step"], "cursor": c["cursor"]})
    time.sleep(2.5)
    out = env.spill(["resume", proj, "--engine", engines[1]], hooks={"checkpoint_after_control": "crash:1"}, expect=(137,))
    c2 = run_info(sr, rid)["doc"]["checkpoint"]
    steps.append({"engine": engines[1], "restored_step": c["step"], "killed_after_published_step": c2["step"],
                  "cursor": c2["cursor"]})
    time.sleep(2.5)
    out = env.spill(["resume", proj, "--engine", engines[2]])
    steps.append({"engine": engines[2], "restored_step": c2["step"],
                  "restore_line_seen": f"restored the committed checkpoint: step {c2['step']}" in out["stdout"]})
    info = run_info(sr, rid)
    res = final(env, proj, rid)
    problems = check_carry_over(info)
    expected_engines = [e.replace("torch-cpu", "torch-cpu") for e in engines]
    got = [p["engine"] for p in res["train_producers"]]
    transitions = res["engine_transitions"]
    ok = (not problems and c["step"] == 10 and c2["step"] == 15 and steps[-1]["restore_line_seen"]
          and got[0] == engines[0] and got[-1] == engines[2] and len(transitions) == 2
          and info["doc"]["status"] == "completed")
    return {"name": name, "engines": engines, "steps": steps, "carry_over_problems": problems,
            "published_checkpoints": [(x["state"]["step"], x["state"]["opt_step"]) for x in info["states"]],
            "recorded_producers": got, "engine_transitions": transitions, "final": res,
            "reference_trained_accuracy": ref["trained_accuracy"], "ok": bool(ok)}


def move_chain(env: Env, name: str, first: str, second: str, dest: str, kind: str) -> dict:
    from streamweights.project import coordinator as CO
    proj = make_project(env, name)
    sr = str(proj / ".spill")
    env.spill(["build", proj, "--engine", first], hooks={"checkpoint_after_control": "crash:2"}, expect=(137,))
    rid = CO.list_runs(proj)[0]["run_id"]
    pub = run_info(sr, rid)["doc"]["checkpoint"]["step"]
    mv = env.spill(["move", proj, dest, "--wait", "30"])
    src = run_info(sr, rid)["doc"]
    out = env.spill(["resume", dest, "--engine", second])
    dest_root = f"{dest}/.spill"
    info = run_info(dest_root, rid)
    problems = check_carry_over(info)
    local = dest if kind == "local" else None
    if kind == "local":
        res = final(env, Path(dest), rid)
    else:
        from streamweights.project import remote as RM
        res = final(env, RM.working_copy(dest), rid)
    ok = (src["status"] == "transferred" and info["doc"]["status"] == "completed" and not problems
          and f"restored the committed checkpoint: step {pub}" in out["stdout"]
          and [p["engine"] for p in res["train_producers"]][0] == first
          and [p["engine"] for p in res["train_producers"]][-1] == second)
    return {"name": name, "move": kind, "dest": dest if kind == "local" else dest, "from": first, "to": second,
            "published_step_before_move": pub, "source_status_after_move": src["status"],
            "move_seconds": mv["seconds"], "carry_over_problems": problems,
            "published_checkpoints": [(x["state"]["step"], x["state"]["opt_step"]) for x in info["states"]],
            "final": res, "ok": bool(ok)}


def run_all(work: Path, s3: str | None = None) -> dict:
    os.environ["SPILL_HOME"] = str(Path(work) / "home")       # before any streamweights import
    out = {"device": gpu_check(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    sys.path.insert(0, str(ROOT))
    from streamweights import machine
    out["hardware"] = machine.info()
    try:
        out["hardware"]["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                                capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    env = Env(work)
    proj = make_project(env, "reference")
    env.spill(["build", proj, "--engine", "mlx"])
    from streamweights.project import coordinator as CO
    rid = CO.list_runs(proj)[0]["run_id"]
    ref = final(env, proj, rid)
    ref["engine_check"] = [p["engine"] for p in ref["train_producers"]]
    out["reference"] = ref
    out["chains"] = [chain(env, "chain-a", ["mlx", "torch-cpu", "mlx"], ref),
                     chain(env, "chain-b", ["torch-cpu", "mlx", "torch-cpu"], ref)]
    moves = []
    for first, second in (("mlx", "torch-cpu"), ("torch-cpu", "mlx")):
        d = work / f"moved-{first}-to-{second}" / "p"
        moves.append(move_chain(env, f"move-local-{first}-{second}", first, second, str(d), "local"))
        if s3:
            moves.append(move_chain(env, f"move-s3-{first}-{second}", first, second,
                                    f"{s3.rstrip('/')}/{first}-{second}-{int(time.time())}", "s3"))
    out["moves"] = moves
    if not s3:
        out["s3_note"] = ("no S3 endpoint was given: Docker is not available here, so the MinIO move is "
                          "covered by CI (tests/test_move.py, job object-store)")
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["all_ok"] = all(c["ok"] for c in out["chains"] + out["moves"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/tmp/metal-gate")
    ap.add_argument("--s3")
    a = ap.parse_args()
    res = run_all(Path(a.work), a.s3)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"all_ok": res["all_ok"], "chains": [(c["name"], c["ok"]) for c in res["chains"]],
                      "moves": [(m["name"], m["ok"]) for m in res["moves"]]}))
    sys.exit(0 if res["all_ok"] else 1)
