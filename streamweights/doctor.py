"""spill doctor: one screen about this machine and what it can do overnight.

  chip, RAM, Metal working set, free disk, Python, package version
  downloaded models, interrupted jobs and builds
  the largest model this machine can run overnight for eval
  the achieved TFLOP/s the training and prefill estimates use
"""

from __future__ import annotations

import math
import platform
import shutil
import sys

from . import estimate as est
from .platforms import platform_line

GIB = 1024**3
OVERNIGHT_H = 12
REF_ROWS, REF_PROMPT_TOKENS, REF_NEW_TOKENS = 2000, 300, 64
DISK_FLOOR = 20 * GIB


def overnight_eval_seconds(params_b: float, rate_bytes_s: float, ws: int, tflops: float
                           ) -> float:
    """Reference eval (2,000 rows, 300-token prompts, 64 new tokens) on a bf16 model of
    `params_b` billion parameters: the model's weights are read once per pass when it does
    not fit the working set, and prefill is compute-bound."""
    size = params_b * 2e9
    kv = 2 * max(8, int(params_b ** 0.5 * 3)) * 8 * 128 * 2      # rough per-token KV for the size
    per_row = (REF_PROMPT_TOKENS + REF_NEW_TOKENS) * kv
    resident = size <= ws * 0.70
    avail = max(ws * 0.75 - (size if resident else 8e9), 1e9)
    batch = max(1, min(512, int(avail // per_row)))
    pass_s = size / (est.RESIDENT_BYTES_PER_S if resident else rate_bytes_s)
    decode = math.ceil(REF_ROWS / batch) * (REF_NEW_TOKENS + 1) * pass_s
    prefill = 2 * params_b * 1e9 * REF_ROWS * REF_PROMPT_TOKENS / (tflops * 1e12)
    return decode + prefill


def largest_overnight(rate_bytes_s: float, ws: int, free_disk: int, tflops: float,
                      hours: float = OVERNIGHT_H) -> tuple[float, str]:
    """(billions of parameters, what limits it): the largest bf16 model whose reference
    eval finishes within `hours` and whose weights fit on disk with the floor kept."""
    best, limit = 0.0, "time"
    for p in range(1, 400):
        if p * 2e9 + DISK_FLOOR > free_disk:
            return float(best), "disk"
        if overnight_eval_seconds(float(p), rate_bytes_s, ws, tflops) > hours * 3600:
            return float(best), "time"
        best = p
    return float(best), limit


def _existing(p):
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def downloaded_models(registry_tags, models_dir) -> list[str]:
    from .registry import MODELS_DIR, safetensors_downloaded
    out = []
    for tag in registry_tags:
        d = (models_dir / tag.replace(":", "-"))
        bits = []
        if safetensors_downloaded(tag):
            bits.append("bf16")
        for q in ("8bit", "4bit"):
            if (d / f"mlx-{q}" / "config.json").exists():
                bits.append(q)
        if bits:
            out.append(f"{tag} ({', '.join(bits)})")
    hf = models_dir / "hf"
    if hf.exists():
        out += [p.name.replace("__", "/") + " (bf16)" for p in sorted(hf.iterdir())
                if p.is_dir() and (p / "config.json").exists()]
    return out


def report(hw: dict, cal: dict, *, registry_tags, models_dir, interrupted_lines: list[str],
           version: str, free_disk: int | None = None, adapters: int = 0,
           mlx: bool = True) -> str:
    ws = hw.get("gpu", {}).get("vram_bytes", 0)
    ram = hw.get("ram_total_bytes", 0)
    free = free_disk if free_disk is not None else shutil.disk_usage(_existing(models_dir)).free
    tf, tf_src = est.achieved_tflops(cal)
    rates = cal.get("engine_rates") or {}
    rate = (max(rates.values()) * 1024 * 1024 if rates else
            (cal.get("isolated_read_mbps", 0) * 1024 * 1024 * 0.5 or
             hw.get("nvme_seq_read", {}).get("bytes_per_sec", 1e9) * 0.5))
    rate_src = "measured streaming rate" if rates else "estimated from the disk probe"
    big, limit = largest_overnight(rate, ws, free, tf)
    have = downloaded_models(registry_tags, models_dir)
    chip = hw.get("cpu") or platform.processor() or platform.machine() or "unknown"
    mem = (f"{ram / GIB:.0f} GB RAM, Metal working set {ws / GIB:.0f} GB" if mlx
           else f"{ram / GIB:.0f} GB RAM, no Metal")
    lines = [
        f"chip         {chip}, {hw.get('cpu_cores', '?')} cores",
        f"memory       {mem}",
        f"disk         {free / GIB:.0f} GB free",
        f"software     Python {sys.version.split()[0]}, streamweights {version}",
        f"models       " + (", ".join(have) if have else "none downloaded yet"),
        f"adapters     {adapters}",
        f"interrupted  " + ("; ".join(interrupted_lines) if interrupted_lines else "none"),
        f"overnight    largest bf16 model for a {REF_ROWS:,}-row eval inside {OVERNIGHT_H} h: "
        f"about {big:.0f}B parameters ({big * 2:.0f} GB; limited by {limit}; disk stream "
        f"{rate / 1e9:.1f} GB/s, {rate_src})",
        f"training     {tf:.3g} TFLOP/s used for prefill and training estimates ({tf_src}); "
        f"training and prefill scale with GPU cores, evals with the disk",
    ]
    if not mlx:
        lines = [l for l in lines if not l.startswith(("overnight", "training"))]
        lines.append(f"platform     {platform.system()} {platform.machine()}: {platform_line()}")
    return "\n".join(lines)
