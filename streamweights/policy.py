"""Quant policy. bf16 default; drops are loud, never silent.

Phase 1 recalibration: the ETA comes from the measured engine read rate in
state/calibration.json (refreshed after every run from actual pass times), not
from the raw NVMe probe. The 8-bit drop rule: drop only if the bf16 estimate
exceeds 24 hours for the submitted job, or disk cannot hold bf16 plus 20 GB.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass

from .registry import GIB, MIN_FREE_AFTER_DOWNLOAD, REPO_ROOT, Model

MAX_JOB_HOURS = 24
MB = 1024 * 1024


@dataclass
class QuantChoice:
    quant: str
    reason: str  # why this quant (printed; never silent)
    est_seconds: float | None = None
    pass_seconds: float | None = None


def engine_read_rate(calibration: dict, hardware: dict,
                     key: str | None = None) -> tuple[float, str]:
    """Bytes/sec the streaming engine actually achieves, best available source.
    Rates are keyed per model|quant; generic fallbacks follow."""
    rates = calibration.get("engine_rates", {})
    if key and rates.get(key):
        return rates[key] * MB, "measured engine rate"
    if calibration.get("engine_read_mbps"):
        return calibration["engine_read_mbps"] * MB, "measured engine rate (machine)"
    if calibration.get("isolated_read_mbps"):
        return calibration["isolated_read_mbps"] * MB, "isolated read ceiling"
    return hardware["nvme_seq_read"]["bytes_per_sec"], "probed NVMe rate (uncalibrated)"


def estimate_job_seconds(model_bytes: int, n_rows: int, max_tokens: int,
                         batch: int, rate_bytes_s: float) -> tuple[float, float]:
    """(job_seconds, pass_seconds). One weight stream per pass; one prefill pass
    plus up to max_tokens decode passes per chunk of `batch` rows."""
    pass_s = model_bytes / rate_bytes_s
    chunks = math.ceil(n_rows / max(1, batch))
    return chunks * (max_tokens + 1) * pass_s, pass_s


def choose_quant_v2(model_bytes: int, bf16_downloaded: bool, n_rows: int,
                    max_tokens: int, est_batch: int, calibration: dict,
                    hardware: dict, explicit: str | None = None,
                    rate_key: str | None = None) -> QuantChoice:
    if explicit:
        return QuantChoice(explicit, f"--quant {explicit} (explicit opt-in)")

    rate, rate_src = engine_read_rate(calibration, hardware, key=rate_key)
    est_s, pass_s = estimate_job_seconds(model_bytes, n_rows, max_tokens, est_batch, rate)

    free = shutil.disk_usage(REPO_ROOT).free
    if not bf16_downloaded and free - model_bytes < MIN_FREE_AFTER_DOWNLOAD:
        return QuantChoice(
            "8bit",
            f"dropped to 8-bit: disk free ({free / GIB:.0f} GB) cannot hold bf16 "
            f"({model_bytes / GIB:.0f} GB) plus 20 GB",
            est_seconds=est_s / 2, pass_seconds=pass_s / 2)

    if est_s > MAX_JOB_HOURS * 3600:
        return QuantChoice(
            "8bit",
            f"dropped to 8-bit: bf16 estimate {est_s / 3600:.1f} h for this job exceeds "
            f"{MAX_JOB_HOURS} h (pass {pass_s:.0f} s at {rate_src} {rate / MB:.0f} MB/s)",
            est_seconds=est_s / 2, pass_seconds=pass_s / 2)

    return QuantChoice("bf16", f"bf16 (batch-tier default); est from {rate_src}",
                       est_seconds=est_s, pass_seconds=pass_s)


# ---------------------------------------------------------------------------
# Phase 0 rule set, retained for the non-Apple GGUF path.

MAX_SECONDS_PER_PASS = 120


def choose_quant(model: Model, hardware: dict, explicit: str | None = None) -> QuantChoice:
    if explicit:
        return QuantChoice(explicit, f"--quant {explicit} (explicit opt-in)")
    bf16_bytes = model.quants["bf16"].bytes
    gpu = hardware.get("gpu", {})
    nvme_rate = hardware.get("nvme_seq_read", {}).get("bytes_per_sec", 0)
    if not gpu.get("bf16_compute"):
        return QuantChoice("Q8_0", "dropped to Q8_0: GPU lacks bf16 compute support")
    free = shutil.disk_usage(REPO_ROOT).free
    already = model.quants["bf16"].downloaded(model.name)
    if not already and free - bf16_bytes < MIN_FREE_AFTER_DOWNLOAD:
        return QuantChoice(
            "Q8_0",
            f"dropped to Q8_0: disk free ({free / GIB:.0f} GB) cannot hold bf16 "
            f"({bf16_bytes / GIB:.0f} GB) plus 20 GB")
    if nvme_rate and bf16_bytes / nvme_rate > MAX_SECONDS_PER_PASS:
        return QuantChoice(
            "Q8_0",
            f"dropped to Q8_0: probed NVMe rate ({nvme_rate / GIB:.1f} GB/s) implies "
            f"{bf16_bytes / nvme_rate:.0f} s per forward pass at bf16 (> {MAX_SECONDS_PER_PASS} s)")
    return QuantChoice("bf16", "bf16 (batch-tier default)")


# ---------------------------------------------------------------------------
# Phase 2: prefill-bound estimates for teacher-forced scoring.

ASSUMED_TFLOPS = 6.0   # effective bf16 matmul rate used only when nothing is measured


def estimate_score_seconds(total_tokens: int, model_bytes: int, quant: str,
                           calibration: dict, key: str, streamed: bool,
                           read_rate: float | None, group_tokens: int = 16384
                           ) -> tuple[float, str]:
    """Scoring is prefill-bound: time = tokens / measured prefill rate, with the
    weight-read floor (one full read per group of up to `group_tokens`) when the
    model streams. Source of the rate, best first:
      measured   state/calibration.json prefill_rates[key] from an earlier --score run
      scaled     another model's measured prefill rate, scaled by weight bytes
      assumed    ASSUMED_TFLOPS over 2 x params (params ~ bytes / bytes-per-param)
    Returns (seconds, source sentence)."""
    pr = calibration.get("prefill_rates", {})
    if key in pr:
        rate, src = pr[key]["tok_s"], f"measured prefill rate {pr[key]['tok_s']:.0f} tok/s"
    elif pr:
        k2, v = max(pr.items(), key=lambda kv: kv[1]["tokens"])
        ref_bytes = v.get("model_bytes")
        # rate ~ 1/params; without the reference's bytes, fall back to the assumption
        if ref_bytes:
            rate = v["tok_s"] * ref_bytes / model_bytes
            src = (f"scaled from {k2} measured {v['tok_s']:.0f} tok/s by weight bytes")
        else:
            rate = None
    else:
        rate = None
    if rate is None:
        bpp = {"bf16": 2.0, "8bit": 1.0, "4bit": 0.5}.get(quant, 2.0)
        params = model_bytes / bpp
        rate = ASSUMED_TFLOPS * 1e12 / (2 * params)
        src = f"uncalibrated: {ASSUMED_TFLOPS:.0f} TFLOP/s assumed, {rate:.0f} tok/s"
    compute_s = total_tokens / rate
    if streamed and read_rate:
        floor = math.ceil(total_tokens / group_tokens) * (model_bytes / read_rate)
        if floor > compute_s:
            return floor, src + "; weight-read bound"
    return compute_s, src
