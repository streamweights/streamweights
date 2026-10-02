"""Phase 1 measurement: 70B bf16 streaming at several batch settings.

Usage: python scripts/measure_phase1.py <settings-comma: auto,16,64,128> <rows> <timeout_s>
Writes state/measurements-llama3.3-70b-bf16.json
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


def iostat_mb() -> float:
    out = subprocess.run(["iostat", "-Id", "disk0"], capture_output=True, text=True).stdout
    return float(out.strip().splitlines()[-1].split()[2])


def run_setting(setting: str, input_file: Path, timeout_s: int) -> dict:
    cmd = [str(REPO / ".venv/bin/spill"), "run", "llama3.3:70b", str(input_file)]
    if setting != "auto":
        cmd += ["--parallel", setting]
    t0 = time.time()
    io0 = iostat_mb()
    samples = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=REPO)
    interrupted = False
    last_io = io0
    out_lines = []
    import threading

    def pump():
        for line in proc.stdout:
            out_lines.append(line)
    threading.Thread(target=pump, daemon=True).start()
    while proc.poll() is None:
        time.sleep(10)
        now = iostat_mb()
        samples.append(round((now - last_io) / 10, 1))
        last_io = now
        if time.time() - t0 > timeout_s and not interrupted:
            proc.send_signal(signal.SIGINT)
            interrupted = True
        if time.time() - t0 > timeout_s + 600:
            proc.kill()
            break
    wall = time.time() - t0
    io1 = iostat_mb()
    text = "".join(out_lines)

    jobs = sorted((REPO / "jobs").iterdir(), key=lambda p: p.name)
    jd = jobs[-1]
    meta = json.loads((jd / "meta.json").read_text())
    comp = prompt = rows_done = 0
    rp = jd / "results.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            r = json.loads(line)
            t = r["streamweights"]["tokens"]
            comp += t.get("completion", 0)
            prompt += t.get("prompt", 0)
            rows_done += 1
    batch = re.search(r"batch (\d+)", text)
    gen = [int(x) for x in re.findall(r"gen_tokens (\d+)", text)]
    passes = [float(x) for x in re.findall(r"median pass ([\d.]+)s", text)]
    peaks = [float(x) for x in re.findall(r"peak mem ([\d.]+)G", text)]
    prefills = [float(x) for x in re.findall(r"prefill ([\d.]+)s", text)]
    cal = json.loads((REPO / "state/calibration.json").read_text())
    return {
        "setting": setting,
        "batch": int(batch.group(1)) if batch else None,
        "rows_done": rows_done,
        "interrupted_at_timeout": interrupted,
        "wall_s": round(wall, 1),
        "completion_tokens": comp,
        "prompt_tokens": prompt,
        "agg_tokens_per_sec": round(comp / wall, 3),
        "gen_tokens_incl_unfinished": gen[-1] if gen else None,
        "gen_tokens_per_sec": round(gen[-1] / wall, 3) if gen else None,
        "median_pass_s": passes[-1] if passes else cal.get("measured_pass_s"),
        "prefill_s": prefills[-1] if prefills else None,
        "peak_mem_gb": max(peaks) if peaks else None,
        "disk_mbps_avg_active": round(sum(s for s in samples if s > 500) /
                                      max(1, len([s for s in samples if s > 500])), 1),
        "disk_gb_total": round((io1 - io0) / 1024, 1),
        "isolated_read_mbps": cal.get("isolated_read_mbps"),
        "job": meta.get("id"),
        "pre_run_line": next((l.strip() for l in out_lines if l.startswith("spill:")), None),
        "why_line": next((l.strip() for l in out_lines if "why:" in l), None),
    }


def main():
    settings = sys.argv[1].split(",")
    rows, timeout_s = int(sys.argv[2]), int(sys.argv[3])
    src = REPO / "examples/evals-2000.jsonl"
    subset = REPO / f"state/evals-first-{rows}.jsonl"
    subset.write_text("\n".join(src.read_text().splitlines()[:rows]) + "\n")
    out_path = REPO / "state/measurements-llama3.3-70b-bf16.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["setting"] for r in results}
    for s in settings:
        if s in done:
            continue
        print(f"=== setting {s}", flush=True)
        r = run_setting(s, subset, timeout_s)
        results.append(r)
        out_path.write_text(json.dumps(results, indent=2))
        print(json.dumps(r, indent=2), flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
