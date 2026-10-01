"""Quant policy. On the batch tier bf16 is the default; drops are loud, never silent."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from .registry import GIB, MIN_FREE_AFTER_DOWNLOAD, REPO_ROOT, Model

MAX_SECONDS_PER_PASS = 120


@dataclass
class QuantChoice:
    quant: str
    reason: str  # why this quant (printed; never silent)


def choose_quant(model: Model, hardware: dict, explicit: str | None = None) -> QuantChoice:
    """bf16 default. Drop to Q8_0 only on one of three stated rules.
    Never below Q8_0 without an explicit flag."""
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
            f"({bf16_bytes / GIB:.0f} GB) plus 20 GB",
        )

    if nvme_rate and bf16_bytes / nvme_rate > MAX_SECONDS_PER_PASS:
        return QuantChoice(
            "Q8_0",
            f"dropped to Q8_0: probed NVMe rate ({nvme_rate / GIB:.1f} GB/s) implies "
            f"{bf16_bytes / nvme_rate:.0f} s per forward pass at bf16 (> {MAX_SECONDS_PER_PASS} s)",
        )

    return QuantChoice("bf16", "bf16 (batch-tier default)")
