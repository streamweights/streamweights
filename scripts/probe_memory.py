"""Phase 1.5 memory calibration: measure peak at two batch sizes on the longest
rows, fit peak(B) = base + slope*B, store per-quant mem_model in calibration.json.

Usage: python scripts/probe_memory.py <quant bf16|8bit> <B1> <B2> [max_tokens]
"""

import json
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from streamweights.engines.base import MemoryBudget, ModelSpec  # noqa: E402
from streamweights.engines.mlx_stream import (MlxStreamEngine,  # noqa: E402
                                              load_calibration, save_calibration)

QUANT_DIRS = {"bf16": "bf16-st", "8bit": "mlx-8bit"}


def probe(quant: str, batch: int, rows: list[dict], max_tokens: int) -> float:
    import mlx.core as mx
    mx.reset_peak_memory()
    peaks = []
    eng = MlxStreamEngine(pass_cb=lambda i: peaks.append(i["peak_gb"]))
    spec = ModelSpec("llama3.3:70b", quant,
                     REPO / "models/llama3.3-70b" / QUANT_DIRS[quant], {}, 4096)
    budget = MemoryBudget(38654705664, batch_override=batch)
    done = 0
    for cr in eng.run_batch(rows, spec, budget):
        done += 1
    print(f"  B={batch}: {done} rows, passes={len(eng.last_pass_times)}, "
          f"peak={max(peaks):.2f} GB", flush=True)
    return max(peaks) * 1024**3


def main():
    quant = sys.argv[1]
    b1, b2 = int(sys.argv[2]), int(sys.argv[3])
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    lines = (REPO / "examples/evals-2000.jsonl").read_text().splitlines()
    rows = [json.loads(l) for l in lines]
    rows.sort(key=lambda r: -len(r["body"]["messages"][0]["content"]))  # longest first
    probe_rows = rows[:b2]
    for r in probe_rows:
        r["body"]["max_tokens"] = max_tokens

    mean_len_chars = sum(len(r["body"]["messages"][0]["content"]) for r in probe_rows) / b2
    mean_cost_tokens = mean_len_chars / 4 + 16 + max_tokens  # chars/4 approx + template

    p1 = probe(quant, b1, probe_rows[:b1], max_tokens)
    p2 = probe(quant, b2, probe_rows, max_tokens)
    slope = (p2 - p1) / (b2 - b1)
    base = p1 - slope * b1
    per_seq_token = slope / mean_cost_tokens
    cal = load_calibration()
    cal.setdefault("mem_model", {})[quant] = {
        "base_bytes": int(base),
        "per_seq_token_bytes": int(per_seq_token),
        "probe_batches": [b1, b2],
        "probe_mean_cost_tokens": round(mean_cost_tokens),
        "probe_peaks_gb": [round(p1 / 1024**3, 2), round(p2 / 1024**3, 2)],
    }
    save_calibration(cal)
    print(json.dumps(cal["mem_model"][quant], indent=1))


if __name__ == "__main__":
    main()
