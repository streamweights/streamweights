"""The item 7 gates of directive 014 on the 0.5B, on this Mac.

  python scripts/gates_014.py quick     torch-cpu build vs MLX, and the noise between two MLX runs
  python scripts/gates_014.py resume    a build stopped on one engine, finished on the other, both ways
  python scripts/gates_014.py remote    --state on the in-memory filesystem, and on a separate directory
  python scripts/gates_014.py headless  SIGTERM mid-stage: exit 75, a checkpoint, a resume that completes
  python scripts/gates_014.py relay     spill example relay && ./relay/relay.sh, in every mode here
  python scripts/gates_014.py noise     the tiny build's score noise, for the CI relay check

Every run gets its own SPILL_HOME (models linked in), so no run is answered by another's cache.
Results merge into docs/reports/014-gates.json.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "reports" / "014-gates.json"
WORK = REPO / "state" / "files" / "gates-014"
EXAMPLES = REPO / "streamweights" / "data" / "examples"
TOL_ROWS = 0.02          # two rows of 100: the floor under the measured noise
sys.path.insert(0, str(REPO))


def save(key, value):
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    cur[key] = value
    OUT.write_text(json.dumps(cur, indent=1, default=str) + "\n")


_N = [0]


def home() -> Path:
    """A fresh SPILL_HOME with the downloaded models and the probed hardware linked in."""
    _N[0] += 1
    h = WORK / f"home-{int(time.time())}-{_N[0]}"
    (h / "state").mkdir(parents=True)
    for f in ("hardware.json", "calibration.json"):
        if (REPO / "state" / f).exists():
            shutil.copy(REPO / "state" / f, h / "state" / f)
    (h / "models").symlink_to(REPO / "models")
    return h


def env_for(h: Path, extra=None):
    e = {k: v for k, v in os.environ.items() if k not in ("SPILL_HEADLESS", "SPILL_ENGINE")}
    e.update(SPILL_HOME=str(h), SPILL_HEADLESS="0", SPILL_NO_NOTIFY="1")
    e.update(extra or {})
    return e


def spill(args, h, extra_env=None, timeout=3600, cwd=None):
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, "-m", "streamweights", *map(str, args)], env=env_for(h, extra_env),
                       capture_output=True, text=True, timeout=timeout, cwd=cwd)
    return p, time.monotonic() - t0


def make_folder(kind: str, dest: Path) -> Path:
    if dest.parent.exists():
        shutil.rmtree(dest.parent)
    dest.parent.mkdir(parents=True)
    p, _ = spill(["example", "banking77", f"--{kind}"], home(), cwd=dest.parent)
    assert p.returncode == 0, p.stderr
    made = dest.parent / f"banking77-{kind}"
    made.rename(dest)
    return dest


def state_doc(path) -> dict:
    return json.loads((Path(path) / "state.json").read_text())


def scores(doc) -> dict:
    return {s["id"]: s["result"]["score"] for s in doc["stages"] if s["kind"] == "eval"}


def audit(state_uri: str) -> dict:
    """Rows and steps in a build's state: none missing, none repeated."""
    from streamweights.portable import checkpoint as pc
    from streamweights.portable import rows as pr
    from streamweights.portable.store import Store
    st = Store(state_uri)
    out = {"tune": None, "row_jobs": {}}
    for stage in st.ls("stages"):
        sub = st.sub(f"stages/{stage}")
        if sub.exists("ckpt"):
            ck = pc.load_tune(sub)
            steps = [s for s, _ in ck.state["losses"]]
            out["tune"] = {"final_step": ck.step, "steps_recorded": len(steps),
                           "contiguous_1_to_n": steps == list(range(1, ck.step + 1)),
                           "history": [{"range": h["range"], "engine": h["engine"],
                                        "hardware": h["hardware"], "os": h.get("os"),
                                        "numerics": h["numerics"].get("base")}
                                       for h in ck.state["history"]]}
        for where in [sub] + [sub.sub(l) for l in sub.ls("") if sub.exists(f"{l}/rows")]:
            if where.exists("rows"):
                segs = pr.committed_segments(where)
                ids = [i for s in segs for i in s["ids"]]
                out["row_jobs"][f"{stage}"] = {
                    "rows": len(ids), "unique": len(set(ids)), "segments": len(segs),
                    "engines": sorted({s.get("engine") for s in segs})}
    out["clean"] = (out["tune"] is None or (out["tune"]["contiguous_1_to_n"])) and all(
        j["rows"] == j["unique"] for j in out["row_jobs"].values())
    return out


def build_leg(folder, state, engine, h, *, stop=None, resume=False, extra=None, env=None):
    args = (["resume", folder, "--state", state, "--engine", engine] if resume else
            ["build", folder, "--state", state, "--engine", engine])
    if stop:
        args += ["--stop-after", stop]
    args += extra or []
    p, wall = spill(args, h, env)
    return p, wall


# ---------------------------------------------------------------- gate 1

def cached(label, fn):
    """Run `fn` once; a gate that is called again (each call fits a tool timeout) reuses it."""
    f = WORK / "results" / f"{label}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        return json.loads(f.read_text())
    r = fn()
    f.write_text(json.dumps(r, default=str))
    return r


def quick():
    out = {}

    def one(label, engine, env=None, extra_home=None):
        return cached(f"quick-{label}", lambda: _one(label, engine, env))

    def _one(label, engine, env=None):
        h = home()
        folder = make_folder("quick", WORK / f"q-{label}" / "banking77-quick")
        state = WORK / f"q-{label}" / "state"
        p, wall = spill(["build", folder, "--engine", engine, "--state", state], h, env)
        assert p.returncode == 0, (p.stdout[-500:], p.stderr[-800:])
        doc = state_doc(state)
        sc = scores(doc)
        r = {"engine": engine, "wall_s": round(wall, 1), "base": sc["eval:base"],
             "tuned": sc["eval:tuned"], "env": env or {},
             "stage_s": {s["id"]: s["actual_s"] for s in doc["stages"]},
             "hardware": doc["stages"][1]["producers"][0]["hardware"],
             "numerics": doc["stages"][1]["producers"][0]["numerics"]}
        print(label, json.dumps(r), flush=True)
        return r

    out["mlx_a"] = one("mlx-a", "mlx")
    out["mlx_b_other_batch_shape"] = one("mlx-b", "mlx", {"SPILL_BUILD_BATCH": "eval=7,micro=2,accum=2"})
    out["torch_cpu"] = one("torch", "torch-cpu")
    a, b, c = out["mlx_a"], out["mlx_b_other_batch_shape"], out["torch_cpu"]
    noise = {"base": abs(a["base"] - b["base"]), "tuned": abs(a["tuned"] - b["tuned"])}
    tol = max(noise["base"], noise["tuned"], TOL_ROWS)
    out["noise_two_mlx_runs_different_batch_shapes"] = noise
    out["tolerance"] = {"value": tol, "rule": f"max(two-MLX-run noise on base and tuned, {TOL_ROWS})"}
    out["torch_cpu_beats_base"] = c["tuned"] > c["base"]
    out["torch_cpu_vs_mlx_a"] = {"base": abs(c["base"] - a["base"]), "tuned": abs(c["tuned"] - a["tuned"])}
    out["pass"] = (out["torch_cpu_beats_base"] and a["tuned"] > a["base"]
                   and out["torch_cpu_vs_mlx_a"]["tuned"] <= tol
                   and out["torch_cpu_vs_mlx_a"]["base"] <= tol)
    save("1_quick_torch_cpu_vs_mlx", out)
    print("PASS" if out["pass"] else "FAIL", json.dumps({k: out[k] for k in (
        "noise_two_mlx_runs_different_batch_shapes", "tolerance", "torch_cpu_vs_mlx_a")}), flush=True)


# ---------------------------------------------------------------- gate 2

def resume_gate():
    res = json.loads(OUT.read_text()).get("1_quick_torch_cpu_vs_mlx") if OUT.exists() else None
    ref = res["mlx_a"] if res else None
    tol = res["tolerance"]["value"] if res else TOL_ROWS
    out = {"reference_uninterrupted_mlx": ref, "tolerance": tol}
    for first, then in (("mlx", "torch-cpu"), ("torch-cpu", "mlx")):
        h = home()
        label = f"{first}-then-{then}"
        folder = make_folder("quick", WORK / f"r-{label}" / "banking77-quick")
        state = str(WORK / f"r-{label}" / "state")
        p1, w1 = build_leg(folder, state, first, h, stop="tune:100")
        assert p1.returncode == 0, p1.stderr[-800:]
        mid = state_doc(state)
        p2, w2 = build_leg(folder, state, then, h, resume=True)
        assert p2.returncode == 0, p2.stderr[-800:]
        doc = state_doc(state)
        sc, au = scores(doc), audit(state)
        r = {"first_leg": {"engine": first, "wall_s": round(w1, 1),
                           "stages": [s["status"] for s in mid["stages"]],
                           "tune_steps_done": mid["stages"][1]["result"].get("step")},
             "second_leg": {"engine": then, "wall_s": round(w2, 1)},
             "base": sc["eval:base"], "tuned": sc["eval:tuned"],
             "stage_engines": {s["id"]: [p["engine"] for p in s["producers"]] for s in doc["stages"]},
             "stage_os": {s["id"]: [p["os"] for p in s["producers"]] for s in doc["stages"]},
             "audit": au, "finished": doc["finished"]}
        if ref:
            r["diff_vs_uninterrupted"] = {"base": abs(r["base"] - ref["base"]),
                                          "tuned": abs(r["tuned"] - ref["tuned"])}
        r["pass"] = (doc["finished"] and au["clean"] and au["tune"]["final_step"] == 250
                     and all(j["rows"] == 100 for j in au["row_jobs"].values())
                     and r["stage_engines"]["tune"] == [first, then]
                     and r["stage_engines"]["eval:base"] == [first]
                     and r["stage_engines"]["eval:tuned"] == [then]
                     and (not ref or (r["diff_vs_uninterrupted"]["tuned"] <= tol
                                      and r["diff_vs_uninterrupted"]["base"] <= tol)))
        out[label] = r
        print(label, json.dumps(r, default=str)[:900], "PASS" if r["pass"] else "FAIL", flush=True)
    out["pass"] = all(out[k]["pass"] for k in out if "-then-" in k)
    save("2_resume_across_engines", out)


# ---------------------------------------------------------------- gate 3

def remote():
    out = {}
    # (a) fsspec's in-memory filesystem: in one process, since the filesystem lives in it
    h = home()
    folder = make_folder("tiny", WORK / "m" / "banking77-tiny")
    code = f"""
import json, os, sys
os.environ.update(SPILL_HOME={str(h)!r}, SPILL_HEADLESS="0", SPILL_NO_NOTIFY="1")
sys.path.insert(0, {str(REPO)!r})
from typer.testing import CliRunner
from streamweights.cli import app
from streamweights.portable.build_state import BuildState
R = CliRunner()
uri = "memory://gate3-build"
a = R.invoke(app, ["build", {str(folder)!r}, "--engine", "mlx", "--state", uri, "--stop-after", "tune:20"])
mid = BuildState(uri).read()
b = R.invoke(app, ["resume", {str(folder)!r}, "--state", uri, "--engine", "torch-cpu"])
doc = BuildState(uri).read()
print(json.dumps({{"leg1_exit": a.exit_code, "leg2_exit": b.exit_code,
  "after_leg1": [s["status"] for s in mid["stages"]], "finished": doc["finished"],
  "scores": {{s["id"]: s["result"]["score"] for s in doc["stages"] if s["kind"] == "eval"}},
  "engines": {{s["id"]: [p["engine"] for p in s["producers"]] for s in doc["stages"]}},
  "state_files": sorted(f for f in BuildState(uri).store.fs.find(BuildState(uri).store.root))[:40]}}))
"""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stderr[-800:]
    out["memory_filesystem"] = json.loads(p.stdout.strip().splitlines()[-1])
    m = out["memory_filesystem"]
    m["pass"] = (m["leg1_exit"] == 0 and m["leg2_exit"] == 0 and m["finished"]
                 and m["after_leg1"] == ["done", "interrupted", "pending"]
                 and m["engines"]["tune"] == ["mlx", "torch-cpu"])
    # (b) a separate local directory, as shared storage, resumed from a copy of the folder
    h = home()
    folder = make_folder("tiny", WORK / "d" / "machine-a" / "banking77-tiny")
    shared = WORK / "d" / "shared-storage"
    if shared.exists():
        shutil.rmtree(shared)
    p1, _ = build_leg(folder, str(shared), "torch-cpu", h, stop="tune:20")
    assert p1.returncode == 0, p1.stderr[-600:]
    mid = state_doc(shared)
    other = WORK / "d" / "machine-b" / "banking77-tiny"                  # a second machine's copy
    if other.parent.exists():
        shutil.rmtree(other.parent)
    other.parent.mkdir(parents=True)
    shutil.copytree(folder, other, ignore=shutil.ignore_patterns(".build", "*.out.jsonl"))
    h2 = home()
    p2, _ = build_leg(other, str(shared), "mlx", h2, resume=True)
    assert p2.returncode == 0, p2.stderr[-600:]
    doc = state_doc(shared)
    au = audit(str(shared))
    out["separate_directory"] = {
        "after_leg1": [s["status"] for s in mid["stages"]], "finished": doc["finished"],
        "scores": scores(doc), "audit": au,
        "engines": {s["id"]: [p["engine"] for p in s["producers"]] for s in doc["stages"]},
        "pass": doc["finished"] and au["clean"] and [s["status"] for s in mid["stages"]] == [
            "done", "interrupted", "pending"]}
    out["pass"] = out["memory_filesystem"]["pass"] and out["separate_directory"]["pass"]
    save("3_remote_state", out)
    print(json.dumps(out, default=str)[:1500], "PASS" if out["pass"] else "FAIL", flush=True)


# ---------------------------------------------------------------- gate 4

def sigterm_run(folder, state, h, trigger):
    cmd = [sys.executable, "-m", "streamweights", "build", str(folder), "--state", str(state),
           "--engine", "torch-cpu", "--headless"]
    p = subprocess.Popen(cmd, env=env_for(h) | {"SPILL_HEADLESS": "1"}, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    evs, sent, t_sig = [], False, None
    for line in p.stdout:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        evs.append(e)
        if not sent and trigger(e):
            p.send_signal(signal.SIGTERM)
            sent, t_sig = True, time.time()
    rc = p.wait(timeout=300)
    return rc, evs, (time.time() - t_sig) if t_sig else None, p.stderr.read()


def headless():
    out = {}
    for label, trigger, stage in (
            ("mid_tune", lambda e: e["event"] == "step" and e["step"] >= 12, "tune"),
            ("mid_eval", lambda e: e["event"] == "row" and e["step"] >= 5, "eval:base")):
        h = home()
        folder = make_folder("tiny", WORK / f"h-{label}" / "banking77-tiny")
        state = WORK / f"h-{label}" / "state"
        rc, evs, dt, err = sigterm_run(folder, state, h, trigger)
        doc = state_doc(state)
        kinds = [e["event"] for e in evs]
        ckpt = [e for e in evs if e["event"] == "checkpoint"]
        mid = {s["id"]: s["status"] for s in doc["stages"]}
        p2 = subprocess.run([sys.executable, "-m", "streamweights", "resume", str(folder), "--state",
                             str(state), "--headless", "--engine", "torch-cpu"],
                            env=env_for(h) | {"SPILL_HEADLESS": "1"}, capture_output=True, text=True,
                            timeout=1800)
        ev2 = [json.loads(l) for l in p2.stdout.splitlines() if l.startswith("{")]
        doc2 = state_doc(state)
        au = audit(str(state))
        r = {"killed_in_stage": stage, "exit_code": rc, "seconds_signal_to_exit": round(dt, 2),
             "events_tail": kinds[-4:], "checkpoint_events": [{"step": c["step"]} for c in ckpt],
             "state_after_signal": mid, "resume_exit": p2.returncode,
             "resume_done_event": {k: ev2[-1].get(k) for k in ("event", "complete", "seconds")},
             "finished": doc2["finished"], "audit": au, "scores": scores(doc2)}
        r["pass"] = (rc == 75 and bool(ckpt) and kinds[-1] == "preempted" and mid[stage] == "interrupted"
                     and p2.returncode == 0 and ev2[-1]["event"] == "done" and doc2["finished"] and au["clean"])
        out[label] = r
        print(label, json.dumps(r, default=str)[:900], "PASS" if r["pass"] else "FAIL", flush=True)
    out["pass"] = all(out[k]["pass"] for k in out if k.startswith("mid_"))
    save("4_headless_sigterm", out)


# ---------------------------------------------------------------- gate 5

def relay():
    out = {}
    docker = shutil.which("docker")
    out["docker"] = ("installed" if docker else "not installed on this machine: Docker mode could not "
                     "run here; the CI relay (relay.yml) stands in for cross-OS proof")
    d = WORK / "relay-modes"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    h = home()
    env = env_for(h) | {"PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"}

    def sh(args, cwd, timeout=3600):
        t0 = time.monotonic()
        p = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return p, time.monotonic() - t0

    # engine-switch mode
    p, _ = sh([sys.executable, "-m", "streamweights", "example", "relay"], d)
    assert p.returncode == 0, p.stderr
    p, wall = sh(["./relay/relay.sh", "--no-docker"], d)
    out["engine_switch"] = {"exit": p.returncode, "wall_s": round(wall, 1), "output": p.stdout}
    print(p.stdout, flush=True)
    doc = state_doc(d / "relay-state")
    ref = state_doc(d / "relay-ref-state")
    out["engine_switch"]["engines"] = {s["id"]: [x["engine"] for x in s["producers"]] for s in doc["stages"]}
    out["engine_switch"]["scores"] = scores(doc)
    out["engine_switch"]["reference_scores"] = scores(ref)
    out["engine_switch"]["audit"] = audit(str(d / "relay-state"))
    out["engine_switch"]["pass"] = (p.returncode == 0 and doc["finished"] and ref["finished"]
                                    and out["engine_switch"]["audit"]["clean"]
                                    and len(set(sum(out["engine_switch"]["engines"].values(), []))) == 2)
    # two-machine mode: stop, copy the two folders, run the printed command there
    d2 = WORK / "relay-modes2"
    if d2.exists():
        shutil.rmtree(d2)
    d2.mkdir(parents=True)
    p, _ = sh([sys.executable, "-m", "streamweights", "example", "relay"], d2)
    p, wall = sh(["./relay/relay.sh", "--two-machines"], d2)
    printed = p.stdout
    other = WORK / "relay-machine-b"
    if other.exists():
        shutil.rmtree(other)
    other.mkdir(parents=True)
    shutil.copytree(d2 / "relay", other / "relay")
    shutil.copytree(d2 / "relay-state", other / "relay-state")
    h2 = home()
    env2 = env_for(h2) | {"PATH": env["PATH"]}
    cmd = "spill resume relay --state relay-state"
    assert cmd in printed, printed
    p2 = subprocess.run(cmd.split(), cwd=other, env=env2, capture_output=True, text=True, timeout=1800)
    t = subprocess.run("spill build relay --state relay-state --table".split(), cwd=other, env=env2,
                       capture_output=True, text=True, timeout=300)
    doc = state_doc(other / "relay-state")
    out["two_machines"] = {"start_exit": p.returncode, "printed": printed, "resume_exit": p2.returncode,
                           "table": t.stdout, "finished": doc["finished"],
                           "audit": audit(str(other / "relay-state")),
                           "pass": p.returncode == 0 and p2.returncode == 0 and doc["finished"]}
    out["pass"] = out["engine_switch"]["pass"] and out["two_machines"]["pass"]
    save("5_relay_example", out)
    print(out["two_machines"]["table"], "PASS" if out["pass"] else "FAIL", flush=True)


# ---------------------------------------------------------------- the CI noise floor

def noise():
    """Tiny-build scores across the ways a CI relay can differ from its reference: engine, batch
    shape, numerics (MLX bf16, CPU float32), and a build moved between engines."""
    runs = {}

    def one(label, engine, env=None, stop_then=None):
        runs[label] = cached(f"noise-{label}", lambda: _one(label, engine, env, stop_then))
        print(label, runs[label], flush=True)

    def _one(label, engine, env=None, stop_then=None):
        h = home()
        folder = make_folder("tiny", WORK / f"n-{label}" / "banking77-tiny")
        state = WORK / f"n-{label}" / "state"
        if stop_then:
            p, _ = build_leg(folder, str(state), engine, h, stop="tune:20", env=env)
            assert p.returncode == 0, p.stderr[-500:]
            p, _ = build_leg(folder, str(state), stop_then, h, resume=True, env=env)
        else:
            p, _ = build_leg(folder, str(state), engine, h, env=env)
        assert p.returncode == 0, p.stderr[-500:]
        sc = scores(state_doc(state))
        return {"base": sc["eval:base"], "tuned": sc["eval:tuned"]}

    one("mlx", "mlx")
    one("mlx_other_batch_shape", "mlx", {"SPILL_BUILD_BATCH": "eval=7,micro=2,accum=2"})
    one("torch_cpu_float32", "torch-cpu", {"SPILL_TORCH_DTYPE": "float32"})
    one("torch_cpu_other_batch_shape", "torch-cpu", {"SPILL_TORCH_DTYPE": "float32",
                                                     "SPILL_BUILD_BATCH": "eval=7,micro=2,accum=2"})
    one("mlx_then_torch_cpu", "mlx", stop_then="torch-cpu")
    one("torch_cpu_then_mlx", "torch-cpu", stop_then="mlx")
    tuned = [r["tuned"] for r in runs.values()]
    base = [r["base"] for r in runs.values()]
    row = 1 / 20
    floor = {
        "rows_in_the_exam": 20, "one_row": row,
        "tuned_spread": round(max(tuned) - min(tuned), 4), "base_spread": round(max(base) - min(base), 4),
        "tolerance": round(max(max(tuned) - min(tuned), max(base) - min(base), row) + row, 4),
        "runs": runs,
        "justification": (
            "The tiny build grades 20 rows, so a score moves in steps of 0.05. The tolerance is the "
            "largest spread between any two of these runs of the same build (two engines, two batch "
            "shapes on each, bf16 on MLX and float32 on the CPU, and builds moved between engines in both "
            "directions; bf16 on this Mac's CPU, which has no bf16 hardware, is hundreds of times slower and was not run), plus one more row. A CI relay score that lands outside it differs from "
            "its uninterrupted reference by more than anything a change of engine, batch shape or "
            "numerics did here."),
        "measured_on": "Apple M4 Pro, macOS, qwen2.5:0.5b, " + time.strftime("%F"),
    }
    (REPO / "docs" / "reports" / "014-noise-floor.json").write_text(json.dumps(floor, indent=1) + "\n")
    save("6_tiny_noise_floor", floor)
    print(json.dumps(floor, indent=1))


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    {"quick": quick, "resume": resume_gate, "remote": remote, "headless": headless,
     "relay": relay, "noise": noise}[sys.argv[1]]()
