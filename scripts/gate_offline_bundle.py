"""Offline bundle gate.

  python scripts/gate_offline_bundle.py prepare --work DIR          (online) build the tiny
        classification project with a real run, its bundle (DIR/bundle) and the wheel (DIR/dist)
  python scripts/gate_offline_bundle.py offline --bundle B --out J  (no network) in a fresh, empty
        SPILL_HOME and Hugging Face cache: verify and install the bundle, report, run inference
        from it, fork a new run and complete a training run, then reject a corrupted and a missing
        asset. Every child process runs with scripts/netguard on PYTHONPATH (any connection attempt
        raises and is logged; the log must be empty) and HF_HUB_OFFLINE=1 as a second layer. The
        proof that no network exists is the environment this runs in: a container started with
        --network none (Linux CI) or sandbox-exec with a deny-network profile (macOS). The gate
        first checks that a real connection attempt from this process fails."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUARD = ROOT / "scripts" / "netguard"


def sh(cmd, env=None, cwd=None, timeout=3600):
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)
    return {"cmd": " ".join(str(c) for c in cmd)[:160], "exit": p.returncode,
            "seconds": round(time.monotonic() - t0, 1), "out": (p.stdout + p.stderr)[-1200:]}


def prepare(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    home = work / "prep-home"
    (home / "state").mkdir(parents=True, exist_ok=True)
    if not (home / "models").exists():
        (home / "models").symlink_to(a.models_from)
    env = {**os.environ, "SPILL_HOME": str(home), "SPILL_HEADLESS": "0", "SPILL_MIN_FREE_GB": "2"}
    spill = [sys.executable, "-m", "streamweights.cli"]
    steps = []
    for cmd in (["example", "banking77", "--tiny"],
                ["init", "banking77-tiny/banking77.csv", "--input", "text", "--output", "label",
                 "--project", "tickets"],
                ["build", "tickets", "--engine", "torch-cpu"],
                ["bundle", "tickets", str(work / "bundle")]):
        r = sh(spill + cmd, env=env, cwd=work)
        steps.append({k: r[k] for k in ("cmd", "exit", "seconds")})
        if r["exit"] != 0:
            raise SystemExit(f"prepare failed: {r}")
    r = sh([sys.executable, "-m", "build", "--wheel", "--outdir", str(work / "dist"), str(ROOT)])
    steps.append({k: r[k] for k in ("cmd", "exit", "seconds")})
    if r["exit"] != 0:
        raise SystemExit(f"wheel build failed: {r}")
    (work / "prepare.json").write_text(json.dumps(steps, indent=1))
    print(json.dumps(steps))


def offline(a):
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "checks": []}

    def check(name, ok, detail=""):
        res["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:400]})
        print(("ok   " if ok else "FAIL ") + name, flush=True)

    # 0. the environment really has no network (this process is not under the guard)
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=5).close()
        check("the environment has no network", False, "a connection to 1.1.1.1:443 succeeded")
    except OSError as e:
        check("the environment has no network", True, f"connect to 1.1.1.1:443 failed: {e}")
    try:
        socket.getaddrinfo("huggingface.co", 443)
        check("DNS is unavailable", False, "huggingface.co resolved")
    except OSError as e:
        check("DNS is unavailable", True, f"{e}")

    work = Path(a.work or "/tmp/offline-gate")
    shutil.rmtree(work, ignore_errors=True)
    home, hf, proj, netlog = work / "home", work / "hf", work / "project", work / "netlog.txt"
    for d in (home / "state", hf):
        d.mkdir(parents=True)
    netlog.write_text("")
    check("SPILL_HOME and the Hugging Face cache start empty",
          not any(home.glob("models/*")) and not any(hf.iterdir()))
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN",)}
    env.update(SPILL_HOME=str(home), HF_HOME=str(hf), HF_HUB_CACHE=str(hf / "hub"),
               HF_HUB_OFFLINE="1", SPILL_HEADLESS="0", SPILL_MIN_FREE_GB="1", SPILL_NETLOG=str(netlog),
               PYTHONPATH=str(GUARD), SPILL_ALLOWED_MODELS="qwen2.5:0.5b,embedding:minilm")
    spill = [sys.executable, "-m", "streamweights.cli"]
    import streamweights
    res["imported_from"] = streamweights.__file__
    bundle = Path(a.bundle)

    def run(name, cmd, ok=(0,), expect=None):
        r = sh(spill + cmd, env=env, cwd=work)
        good = r["exit"] in ok and (expect is None or expect in r["out"])
        check(name, good, r["out"][-300:] if not good else f"exit {r['exit']} in {r['seconds']} s")
        return r

    run("bundle verifies (checksums)", ["bundle", "--verify", str(bundle)], expect="every checksum matches")
    run("bundle installs into the empty cache and the project is copied out",
        ["bundle", "--install", str(bundle), str(proj)])
    for m in ("qwen2.5-0.5b/bf16-st/model.safetensors", "embedding-minilm/model.safetensors"):
        check(f"installed {m}", (home / "models" / m).exists())
    run("report from the bundled project", ["report", str(proj)], expect="index:")
    run("plan needs nothing from the network", ["plan", str(proj), "--engine", "torch-cpu"],
        expect="next: spill build")
    runs = sorted((proj / "runs").iterdir())
    adapter = runs[-1] / "artifacts" / "adapter"
    sample = work / "in.jsonl"
    rows = [json.loads(l) for l in (proj / "data" / "val.jsonl").read_text().splitlines()[:3]]
    sample.write_text("".join(json.dumps({"prompt": r["input"]}) + "\n" for r in rows))
    out = work / "answers.jsonl"
    run("inference from the bundle (base plus the bundled adapter)",
        ["run", f"qwen2.5:0.5b+{adapter}", str(sample), "--out", str(out), "--engine", "torch-cpu"])
    answers = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    check("inference produced one answer per row", len(answers) == len(rows),
          [x["response"]["body"]["choices"][0]["message"]["content"] for x in answers])
    r = run("fork a new run and complete a training run offline",
            ["build", str(proj), "--new-run", "--engine", "torch-cpu"])
    new = sorted((proj / "runs").iterdir())
    check("a new completed run exists next to the bundled one", len(new) == len(runs) + 1)
    if len(new) > len(runs):
        m = json.loads((new[-1] / "manifest.json").read_text())
        check("the fork names the bundled run as its parent", m["parent"] == runs[-1].name, m["parent"])
        res["fork_table"] = m["table"]
    # corruption and a missing asset, inside the offline environment
    bad = work / "bad-bundle"
    shutil.copytree(bundle, bad, symlinks=True)
    tok = bad / "assets/models/qwen2.5-0.5b/bf16-st/tokenizer.json"
    tok.write_bytes(tok.read_bytes()[:-8] + b"corrupt!")
    r = run("a corrupted asset is rejected", ["bundle", "--verify", str(bad)], ok=(1,),
            expect="corrupted assets/models/qwen2.5-0.5b/bf16-st/tokenizer.json")
    run("a corrupted bundle is refused by install too", ["bundle", "--install", str(bad), str(work / "x1")],
        ok=(1,), expect="not intact")
    shutil.copyfile(bundle / "assets/models/qwen2.5-0.5b/bf16-st/tokenizer.json", tok)
    (bad / "assets/models/embedding-minilm/model.safetensors").unlink()
    run("a missing asset is rejected", ["bundle", "--verify", str(bad)], ok=(1,),
        expect="missing assets/models/embedding-minilm/model.safetensors")
    attempts = netlog.read_text().strip().splitlines()
    check("no network call was attempted by any spill process", attempts == [], attempts[:5])
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    res["all_ok"] = all(c["ok"] for c in res["checks"])
    Path(a.out).write_text(json.dumps(res, indent=1))
    raise SystemExit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "offline"])
    ap.add_argument("--work")
    ap.add_argument("--bundle")
    ap.add_argument("--out")
    ap.add_argument("--models-from", default=str(ROOT / "models"))
    a = ap.parse_args()
    prepare(a) if a.mode == "prepare" else offline(a)
