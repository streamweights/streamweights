"""The cost model behind every pre-run estimate, README time and diagram.

  decode on a model that streams from disk is disk-bound: one weight pass per token
      step, about 34 s per pass for the 70B here (measured, state/calibration.json)
  decode on a model that is resident is memory-bound: one pass reads the weights
  prefill is compute-bound:   2 x params x tokens  / achieved FLOP/s
  training is compute-bound:  6 x params x tokens  / achieved FLOP/s
      (forward, recompute, backward)

Achieved FLOP/s comes from the tune runs recorded in the calibration file (Phase 3
writes `tune_rates`); until one exists it is 5 TFLOP/s, and the pre-run line says so.
Shared-prefix reuse removes (rows - 1) x prefix tokens from the prefill term.

Token counts before a model is downloaded are chars / 4, which is close enough for a
pre-run estimate; the actual run prints measured numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DEFAULT_TFLOPS = 5.0
# resident decode: one pass reads the weights once; effective bytes/s is far below the
# datasheet figure once attention, sampling and launch overhead are included
RESIDENT_BYTES_PER_S = 120e9
STREAM_PASS_S_FALLBACK = 34.0

# parameters in billions (all registry tags); HF repos fall back to bytes / 2
PARAMS_B = {"qwen2.5:0.5b": 0.494, "qwen2.5:7b": 7.62, "qwen2.5:32b": 32.8,
            "llama3.3:70b": 70.55}
ARCH = {"qwen2.5:0.5b": (24, 2, 64), "qwen2.5:7b": (28, 4, 128),
        "qwen2.5:32b": (64, 8, 128), "llama3.3:70b": (80, 8, 128)}
BF16_BYTES = {"qwen2.5:0.5b": 0.99e9, "qwen2.5:7b": 15.23e9, "qwen2.5:32b": 65.5e9,
              "llama3.3:70b": 141.1e9}


def register_model(tag: str, bytes_: int, arch: tuple[int, int, int] | None = None) -> None:
    """Sizes for models outside the registry (HF repo ids), so estimates use the real size."""
    if tag not in PARAMS_B:
        PARAMS_B[tag] = bytes_ / 2 / 1e9
        BF16_BYTES[tag] = float(bytes_)
        if arch:
            ARCH[tag] = arch


def achieved_tflops(cal: dict) -> tuple[float, str]:
    """(TFLOP/s, where it came from)."""
    if cal.get("achieved_tflops"):
        return float(cal["achieved_tflops"]), "measured"
    rates = cal.get("tune_rates") or {}
    if rates:
        best = max(rates.values())
        return best / 1e12, "measured by spill tune"
    return DEFAULT_TFLOPS, "assumed, nothing measured yet"


def params_b(tag: str, size_bytes: int | None = None) -> float:
    if tag in PARAMS_B:
        return PARAMS_B[tag]
    return (size_bytes or 2e9) / 2 / 1e9


def model_bytes(tag: str) -> float:
    return BF16_BYTES.get(tag, 2e9)


def fits_resident(tag: str, working_set: int) -> bool:
    return model_bytes(tag) <= working_set * 0.70


def stream_pass_s(cal: dict, tag: str) -> float:
    # per-model rate first: measured_pass_s is overwritten by whichever model streamed last
    rate = (cal.get("engine_rates") or {}).get(f"{tag}|bf16")
    if rate:
        return model_bytes(tag) / (rate * 1024 * 1024)
    if tag == "llama3.3:70b" and cal.get("measured_pass_s"):
        return float(cal["measured_pass_s"])
    iso = cal.get("isolated_read_mbps")
    if iso:
        return model_bytes(tag) / (iso * 1024 * 1024 * 0.8)
    return STREAM_PASS_S_FALLBACK * model_bytes(tag) / BF16_BYTES["llama3.3:70b"]


def prefill_s(tag: str, tokens: float, tflops: float) -> float:
    return 2 * params_b(tag) * 1e9 * tokens / (tflops * 1e12)


def train_s(tag: str, tokens: float, tflops: float, epochs: float = 1.0) -> float:
    return 6 * params_b(tag) * 1e9 * tokens * epochs / (tflops * 1e12)


def kv_bytes_per_token(tag: str) -> int:
    layers, kv_heads, hd = ARCH.get(tag, (32, 8, 128))
    return 2 * layers * kv_heads * hd * 2


def common_prefix_chars(texts: list[str]) -> int:
    if not texts:
        return 0
    a, b = min(texts), max(texts)
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class Workload:
    """What a stage will push through a model."""
    rows: int
    prefix_tokens: float          # shared by every row (computed once with reuse)
    suffix_tokens: float          # mean private prompt tokens per row
    out_tokens: float             # mean generated tokens per row
    max_tokens: int = 16


@dataclass
class StageEstimate:
    seconds: float
    detail: str
    parts: dict = field(default_factory=dict)


def decode_batch(tag: str, wl: Workload, working_set: int, reuse: bool) -> int:
    """Rows per batch: what the KV budget allows (private tokens only when the prefix is
    shared), at 75% of the working set minus the weights' working buffers."""
    per_row = (wl.suffix_tokens + wl.max_tokens + (0 if reuse else wl.prefix_tokens)) \
        * kv_bytes_per_token(tag)
    resident = fits_resident(tag, working_set)
    base = model_bytes(tag) if resident else 8e9
    avail = max(working_set * 0.75 - base, 1e9)
    return int(max(1, min(512, avail // max(per_row, 1))))


def eval_seconds(tag: str, wl: Workload, cal: dict, working_set: int, tflops: float,
                 reuse: bool = True, adapter: bool = False) -> StageEstimate:
    resident = fits_resident(tag, working_set)
    batch = decode_batch(tag, wl, working_set, reuse)
    pass_s = (model_bytes(tag) / RESIDENT_BYTES_PER_S) if resident else stream_pass_s(cal, tag)
    prefill_tokens = wl.rows * wl.suffix_tokens + (wl.prefix_tokens if reuse
                                                   else wl.rows * wl.prefix_tokens)
    pre = prefill_s(tag, prefill_tokens, tflops)
    chunks = math.ceil(wl.rows / batch)
    dec = chunks * (wl.out_tokens + 1) * pass_s
    # streamed: each chunk's prefill rides on the first pass; the weight stream is the floor
    total = pre + dec
    detail = (f"{wl.rows} rows, batch {batch}, {'resident' if resident else 'streamed'}: "
              f"prefill {fmt_dur(pre)} ({prefill_tokens:,.0f} tokens"
              f"{', shared prefix once' if reuse and wl.prefix_tokens else ''}) + "
              f"decode {fmt_dur(dec)} ({chunks} x {wl.out_tokens + 1:.0f} passes x "
              f"{pass_s:.2f} s)")
    return StageEstimate(total, detail, {"prefill_s": pre, "decode_s": dec, "batch": batch,
                                         "pass_s": pass_s, "prefill_tokens": prefill_tokens})


def tune_seconds(tag: str, tokens: float, epochs: float, tflops: float, cal: dict,
                 working_set: int, steps: int | None = None) -> StageEstimate:
    resident = fits_resident(tag, working_set)
    own = (cal.get("tune_rates") or {}).get(f"{tag}|{'resident' if resident else 'streamed'}")
    if own:                      # this model's own measured rate beats the machine-wide one
        tflops = own / 1e12
    compute = train_s(tag, tokens, tflops, epochs)
    stream = 0.0
    if not resident:
        # two weight streams per optimizer step on the streamed path
        steps = steps or max(1, int(tokens * epochs / 8192))
        stream = steps * 2 * stream_pass_s(cal, tag)
    total = max(compute, stream) if stream else compute
    detail = (f"{tokens * epochs:,.0f} token-passes x 6 x {params_b(tag):.2f}B params / "
              f"{tflops:.3g} TFLOP/s = {fmt_dur(compute)}"
              + (f"; weight streams floor {fmt_dur(stream)}" if stream else ""))
    return StageEstimate(total, detail, {"compute_s": compute, "stream_s": stream})


def fmt_dur(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    if s < 48 * 3600:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"
