"""Captured evaluation request and applied decoding, per engine (directive 017).

  python scripts/gate_decoding_capture.py --out docs/reports/data/017-decoding.json

For mlx (on the Metal GPU: the gate fails if mx.default_device() is not the GPU) and torch-cpu, runs
`spill run qwen2.5:0.5b` on a batch-shaped request that carries every recorded decoding setting, then
reads back (a) the request as the job stored it and (b) the settings each result row says were
applied. Then sends a request with a setting the engine cannot apply and records the refusal."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BODY = {"temperature": 0, "top_p": 1.0, "stop": [], "seed": 3, "greedy": True, "max_tokens": 5}


def row(cid, **extra):
    b = {"messages": [{"role": "user", "content": "Say hi."}], **BODY, **extra}
    return {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": b}


def run(home, engine, rows, work):
    f = work / f"in-{engine}.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = work / f"out-{engine}-{len(rows)}-{rows[0]['body'].get('temperature')}.jsonl"
    env = {k: v for k, v in os.environ.items() if k != "SPILL_DEVICE"}
    env.update(SPILL_HOME=str(home), SPILL_HEADLESS="0")
    p = subprocess.run([sys.executable, "-m", "streamweights.cli", "run", "qwen2.5:0.5b", str(f), "--out", str(out),
                        "--engine", engine], capture_output=True, text=True, env=env, cwd=work, timeout=900)
    return p, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="decoding-gate-"))
    home = work / "home"
    (home / "state").mkdir(parents=True)
    (home / "models").symlink_to(ROOT / "models")
    os.environ["SPILL_HOME"] = str(home)
    import mlx.core as mx
    if mx.default_device() != mx.gpu:
        raise SystemExit(f"mx.default_device() is {mx.default_device()}, not the GPU")
    res = {"mlx_default_device": str(mx.default_device()), "engines": {}}
    for engine in ("mlx", "torch-cpu"):
        p, out = run(home, engine, [row("a"), row("b")], work)
        rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
        jobs = sorted((home / "jobs").iterdir())
        sent = [json.loads(l) for l in (jobs[-1] / "input.jsonl").read_text().splitlines()]
        bad, bad_out = run(home, engine, [row("c", temperature=0.7)], work)
        res["engines"][engine] = {
            "exit": p.returncode, "request_as_sent": sent[0]["body"],
            "applied_per_row": [r["streamweights"]["decoding"] for r in rows],
            "engine_impl": rows[0]["streamweights"]["engine"] if rows else None,
            "hardware": rows[0]["streamweights"].get("hardware") if rows else None,
            "max_completion_tokens": max(r["response"]["body"]["usage"]["completion_tokens"] for r in rows),
            "unsupported_request_exit": bad.returncode, "unsupported_request_message": bad.stderr.strip()[-300:],
            "unsupported_wrote_output": bad_out.exists()}
    res["ok"] = all(e["exit"] == 0 and e["unsupported_request_exit"] == 1 and not e["unsupported_wrote_output"]
                    and all(d["requested"] == {k: BODY[k] for k in ("temperature", "top_p", "stop", "seed", "greedy")}
                            for d in e["applied_per_row"]) and e["max_completion_tokens"] <= 5
                    and "cannot apply" in e["unsupported_request_message"] for e in res["engines"].values())
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"ok": res["ok"], "device": res["mlx_default_device"]}))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
