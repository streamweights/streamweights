"""Phase-0 wedge measurement: run the eval set at several --parallel settings,
recording aggregate tokens/s, wall time, effective disk read rate (vm_stat
pageins + iostat), and page-cache behavior.

Usage: python scripts/measure_wedge.py <model> <quant> <parallel,...> <rows> <timeout_s> <ctx>
Writes state/measurements-<model>-<quant>.json
"""

from __future__ import annotations

import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = 16384  # Apple Silicon page size


def vm_pageins() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r'Pageins:\s+(\d+)', out)
    return int(m.group(1)) if m else 0


def iostat_mb() -> float:
    # cumulative MB transferred on disk0 since boot
    out = subprocess.run(["iostat", "-Id", "disk0"], capture_output=True, text=True).stdout
    line = out.strip().splitlines()[-1].split()
    return float(line[2])


def run_setting(model: str, quant: str, parallel: int, input_file: Path,
                timeout_s: int, ctx: int) -> dict:
    t0 = time.time()
    pg0, io0 = vm_pageins(), iostat_mb()
    samples = []
    proc = subprocess.Popen(
        [str(REPO / ".venv/bin/spill"), "run", model, str(input_file),
         "--quant", quant, "--parallel", str(parallel), "--context", str(ctx)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=REPO,
    )
    interrupted = False
    last_io = io0
    while proc.poll() is None:
        time.sleep(10)
        now_io = iostat_mb()
        samples.append(round((now_io - last_io) / 10, 1))  # MB/s over the window
        last_io = now_io
        if time.time() - t0 > timeout_s and not interrupted:
            proc.send_signal(signal.SIGINT)  # engine checkpoints cleanly
            interrupted = True
        if time.time() - t0 > timeout_s + 180:
            proc.kill()
            break
    wall = time.time() - t0
    pg1, io1 = vm_pageins(), iostat_mb()

    # newest job dir carries the truth
    jobs = sorted((REPO / "jobs").iterdir(), key=lambda p: p.name)
    meta = json.loads((jobs[-1] / "meta.json").read_text())
    comp_tokens = prompt_tokens = rows_done = 0
    lat = []
    rp = jobs[-1] / "results.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            r = json.loads(line)
            t = r["streamweights"]["tokens"]
            comp_tokens += t.get("completion", 0) or 0
            prompt_tokens += t.get("prompt", 0) or 0
            if r["streamweights"].get("latency_s"):
                lat.append(r["streamweights"]["latency_s"])
            rows_done += 1
    return {
        "requested_parallel": parallel,
        "effective_parallel": meta.get("parallel"),
        "oom_halved": meta.get("parallel") != parallel,
        "rows_done": rows_done,
        "interrupted_at_timeout": interrupted,
        "wall_s": round(wall, 1),
        "completion_tokens": comp_tokens,
        "prompt_tokens": prompt_tokens,
        "agg_tokens_per_sec": round(comp_tokens / wall, 2) if wall else 0,
        "disk_read_gb_vmstat": round((pg1 - pg0) * PAGE / 1024**3, 2),
        "disk_xfer_gb_iostat": round((io1 - io0) / 1024, 2),
        "disk_mbps_avg": round(sum(samples) / len(samples), 1) if samples else 0,
        "disk_mbps_p95": sorted(samples)[int(len(samples) * 0.95)] if samples else 0,
        "job": meta.get("id"),
        "median_latency_s": sorted(lat)[len(lat) // 2] if lat else None,
    }


def main():
    model, quant = sys.argv[1], sys.argv[2]
    parallels = [int(x) for x in sys.argv[3].split(",")]
    rows, timeout_s, ctx = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

    src = REPO / "examples/evals-2000.jsonl"
    subset = REPO / f"state/evals-first-{rows}.jsonl"
    subset.write_text("\n".join(src.read_text().splitlines()[:rows]) + "\n")

    out_path = REPO / f"state/measurements-{model.replace(':', '-')}-{quant}.json"
    results = []
    if out_path.exists():
        results = json.loads(out_path.read_text())
    done = {r["requested_parallel"] for r in results}
    for p in parallels:
        if p in done:
            continue
        print(f"=== parallel {p}", flush=True)
        r = run_setting(model, quant, p, subset, timeout_s, ctx)
        results.append(r)
        out_path.write_text(json.dumps(results, indent=2))
        print(json.dumps(r, indent=2), flush=True)
        time.sleep(15)  # let the page cache settle between settings


if __name__ == "__main__":
    main()
