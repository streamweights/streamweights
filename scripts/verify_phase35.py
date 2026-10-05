"""Phase 3.5 verification runs. Each subcommand writes a JSON report next to the others in
docs/reports/009-banking77/ and prints one line.

  prefix-ab MODEL [--rows 20] [--file prompts.jsonl --instructions instructions.txt]
      Greedy output with and without shared-prefix reuse on the first N rows (with the
      label prompt as system prompt, as the teacher sees them): identical or not, wall
      time of each, tokens the prefix saved, and the cost-model prefill time removed.

  export-verify MODEL+ADAPTER [--rows 20] [--evals evals.jsonl]
      The merged model's greedy output equals base+adapter on N prompts (engine to engine),
      and the GGUF converted from it runs in llama.cpp (llama-server, the same prompts).

Run with the same environment as the proof (SPILL_DEVICE unset: Metal).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "docs" / "reports" / "009-banking77"
EXAMPLE = ROOT / "streamweights" / "data" / "examples" / "banking77"


def spill(*args, env=None, check=True):
    cmd = [sys.executable, "-m", "streamweights", *map(str, args)]
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})},
                       cwd=ROOT)
    if check and r.returncode:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r, time.monotonic() - t0


def with_system(src: Path, system: str, dest: Path, n: int, max_tokens: int = 16) -> Path:
    rows = []
    for i, line in enumerate(src.read_text().splitlines()[:n]):
        o = json.loads(line)
        rows.append({"custom_id": f"r{i:03d}", "max_tokens": max_tokens, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": o["prompt"]}]})
    dest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return dest


def outputs(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[r["custom_id"]] = r["response"]["body"]["choices"][0]["message"]["content"]
    return out


def manifest_for(out_file: Path) -> dict:
    """The run manifest whose results file this is (the newest manifest of that input)."""
    ms = [json.loads(p.read_text()) for p in (ROOT / "runs").glob("*/manifest.json")]
    ms = [m for m in ms if m.get("kind") == "run"]
    return max(ms, key=lambda m: m["started"])


def prefix_ab(a) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    tmp = REPORTS / "tmp"
    tmp.mkdir(exist_ok=True)
    src = Path(a.file) if a.file else EXAMPLE / "prompts.jsonl"
    ins = Path(a.instructions) if a.instructions else EXAMPLE / "instructions.txt"
    f = with_system(src, ins.read_text().strip(), tmp / "ab.jsonl", a.rows)
    res = {}
    # a third run changes only the batch composition of the unshared path: bf16 output is not
    # bit-invariant to batch shape, so the disagreement between two unshared runs is the noise
    # floor the reuse comparison has to be read against
    for label, extra in (("reuse", []), ("no_reuse", ["--no-prefix-reuse"]),
                         ("no_reuse_batch7", ["--no-prefix-reuse", "--parallel", "7"])):
        out = tmp / f"ab-{label}.jsonl"
        r, wall = spill("run", a.model, f, "--out", out, "--quiet", *extra)
        m = manifest_for(out)
        res[label] = {"wall_s": round(wall, 1), "run_id": m["id"],
                      "started": m["started"], "ended": m["ended"],
                      "prefix_reuse": m.get("prefix_reuse"), "tokens": m.get("tokens")}
        res[label]["outputs"] = outputs(out)
    same = res["reuse"]["outputs"] == res["no_reuse"]["outputs"]
    diff = [k for k in res["reuse"]["outputs"]
            if res["reuse"]["outputs"][k] != res["no_reuse"]["outputs"].get(k)]
    saved = ((res["reuse"]["prefix_reuse"] or {}).get("tokens_saved") or 0)
    floor = [k for k in res["no_reuse"]["outputs"]
             if res["no_reuse"]["outputs"][k] != res["no_reuse_batch7"]["outputs"].get(k)]
    rep = {"model": a.model, "rows": a.rows, "identical": same, "differing_rows": diff,
           "noise_floor_differing_rows": floor,
           "wall_s": {k: res[k]["wall_s"] for k in ("reuse", "no_reuse")},
           "wall_s_removed": round(res["no_reuse"]["wall_s"] - res["reuse"]["wall_s"], 1),
           "prefix_tokens": (res["reuse"]["prefix_reuse"] or {}).get("prefix_tokens"),
           "prefill_tokens_saved": saved, "runs": {k: res[k]["run_id"] for k in res},
           "outputs": res["reuse"]["outputs"]}
    (REPORTS / f"prefix-ab-{a.model.replace(':', '-')}.json").write_text(json.dumps(rep, indent=2))
    print(f"prefix-ab {a.model}: identical={same} on {a.rows} rows; reuse "
          f"{rep['wall_s']['reuse']} s vs {rep['wall_s']['no_reuse']} s without "
          f"({rep['wall_s_removed']} s removed, {saved:,} prefill tokens saved); differing "
          f"rows with reuse: {len(diff)}, between two unshared runs of different batch shape "
          f"(noise floor): {len(floor)}")
    return rep


def export_verify(a) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    tmp = REPORTS / "tmp"
    tmp.mkdir(exist_ok=True)
    base, _, adapter = a.model.partition("+")
    evals = Path(a.evals) if a.evals else EXAMPLE / "evals.jsonl"
    rows = []
    for i, line in enumerate(evals.read_text().splitlines()[:a.rows]):
        rows.append({"custom_id": f"r{i:03d}", "max_tokens": 16,
                     "messages": [{"role": "user", "content": json.loads(line)["prompt"]}]})
    f = tmp / "export.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out_a = tmp / "export-adapter.jsonl"
    spill("run", a.model, f, "--out", out_a, "--quiet")
    dest = tmp / "merged"
    r, _ = spill("export", a.model, "--out", dest, "--gguf", "q8_0")
    gguf = next(dest.glob("*.gguf"))
    out_m = tmp / "export-merged.jsonl"
    spill("run", dest, f, "--out", out_m, "--quiet")        # merged safetensors dir as the model
    A, M = outputs(out_a), outputs(out_m)
    same = A == M
    diff = [k for k in A if A[k] != M.get(k)]
    # GGUF in llama.cpp: llama-server, one request per row, greedy
    g = gguf_outputs(gguf, rows)
    agree = sum(1 for k in A if g.get(k, "").strip() == A[k].strip())
    rep = {"model": a.model, "rows": a.rows, "merged_equals_adapter": same,
           "differing_rows": diff, "gguf": gguf.name, "gguf_bytes": gguf.stat().st_size,
           "gguf_runs": bool(g), "gguf_agreement": f"{agree}/{len(A)}",
           "adapter_outputs": A, "merged_outputs": M, "gguf_outputs": g}
    (REPORTS / "export-verify.json").write_text(json.dumps(rep, indent=2))
    print(f"export-verify {a.model}: merged == base+adapter: {same} ({len(diff)} differ of "
          f"{a.rows}); GGUF {gguf.name} ran in llama.cpp, agrees with the engine on {agree}/{len(A)}")
    return rep


def gguf_outputs(gguf: Path, rows: list[dict]) -> dict[str, str]:
    import socket
    import urllib.request
    sys.path.insert(0, str(ROOT))
    from streamweights.llamacpp import ensure_llama_server
    server = ensure_llama_server()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = open(REPORTS / "tmp" / "llama-server.log", "wb")
    p = subprocess.Popen([str(server), "-m", str(gguf), "--port", str(port), "-ngl", "99",
                          "--no-webui", "-c", "4096"], stdout=log, stderr=log)
    out = {}
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            raise SystemExit("llama-server did not come up; see llama-server.log")
        for r in rows:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({"messages": r["messages"], "max_tokens": r["max_tokens"],
                                 "temperature": 0}).encode(),
                headers={"Content-Type": "application/json"})
            body = json.loads(urllib.request.urlopen(req, timeout=120).read())
            out[r["custom_id"]] = body["choices"][0]["message"]["content"]
    finally:
        p.terminate()
        p.wait(timeout=20)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prefix-ab")
    a.add_argument("model")
    a.add_argument("--rows", type=int, default=20)
    a.add_argument("--file")
    a.add_argument("--instructions")
    e = sub.add_parser("export-verify")
    e.add_argument("model")
    e.add_argument("--rows", type=int, default=20)
    e.add_argument("--evals")
    args = ap.parse_args()
    {"prefix-ab": prefix_ab, "export-verify": export_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
