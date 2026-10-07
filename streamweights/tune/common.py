"""Everything about a tune job that does not depend on the engine: the prepared job, the
per-step log, writing the finished adapter, and the lifecycle helpers MLX and PyTorch share."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import adapters as adapters_mod
from ..jobs.engine import Job
from . import budget as bud
from . import lora_core as lo
from .checkpointer import Checkpointer
from .data import BatchPlan, DataStats, Example
from .spec import TuneSpec, data_hash  # noqa: F401

GIB = 1024**3
ASSUMED_FLOPS = 3.5e12          # effective bf16 flops/s until a run measures it
RESIDENT_FRACTION = 0.35        # weights must be under this share of the working set


@dataclass
class Prepared:
    spec: TuneSpec
    config: dict
    tokenizer: object
    examples: list[Example]
    stats: DataStats
    plan: BatchPlan
    shapes: dict
    size_bytes: int
    n_layers: int
    lora_params: int
    budget: bud.TuneBudget | None = None
    est_step_s: float = 0.0
    est_total_s: float = 0.0
    est_note: str = ""
    why: str = ""
    working_set: int = 0
    stream_stats: dict = field(default_factory=dict)
    cp: Checkpointer | None = None
    dtype_note: str = ""
    abandoned: bool = False        # the step in flight was dropped to meet a preemption deadline
    snapshot: object = None        # (step, params, opt) as of the last completed step


def adapters_dir() -> Path:
    return adapters_mod.ADAPTERS_DIR


def _dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in Path(d).glob("*.safetensors"))


def decide_path(size_bytes: int, working_set: int, forced: str = "auto") -> str:
    if forced in ("resident", "streamed"):
        return forced
    return "resident" if size_bytes <= RESIDENT_FRACTION * working_set else "streamed"



class StepLog:
    """Per-step bookkeeping: losses.jsonl, live.json, the progress callback."""

    def __init__(self, job: Job, steps: int, start: int, cb, ema: float = 0.3, peak_fn=None):
        self.job, self.steps, self.cb, self.ema = job, steps, cb, ema
        self.peak_fn = peak_fn or (lambda: 0.0)
        self.t_last = time.monotonic()
        self.avg = None
        self.start = start
        self.tokens = 0
        self.t0 = time.monotonic()
        self.f = open(job.dir / "losses.jsonl", "a")

    def close(self):
        self.f.close()

    def reset_clock(self):
        self.t_last = time.monotonic()

    def step(self, step: int, loss: float, tokens: int):
        now = time.monotonic()
        dt = now - self.t_last
        self.t_last = now
        self.avg = dt if self.avg is None else self.ema * dt + (1 - self.ema) * self.avg
        self.tokens += tokens
        peak = self.peak_fn()
        info = {"step": step, "steps": self.steps, "loss": loss, "step_s": dt,
                "avg_step_s": self.avg, "eta_s": (self.steps - step) * self.avg,
                "peak_gb": peak, "tokens": tokens,
                "tok_s": self.tokens / max(1e-9, now - self.t0)}
        self.f.write(json.dumps({k: info[k] for k in ("step", "loss", "step_s", "peak_gb")}) + "\n")
        self.f.flush()
        try:
            (self.job.dir / "live.json").write_text(json.dumps(info))
        except OSError:
            pass
        self.job.write_meta(done=step, total=self.steps, eta_seconds=round(info["eta_s"]),
                            tokens_per_sec=round(info["tok_s"], 2))
        if self.cb:
            self.cb(info)
        return info


def truncate_losses(job: Job, step: int) -> None:
    p = job.dir / "losses.jsonl"
    if not p.exists():
        return
    keep = [l for l in p.read_text().splitlines() if l.strip() and json.loads(l)["step"] <= step]
    p.write_text("".join(l + "\n" for l in keep))


# ------------------------------------------------------------ streamed path

def write_adapter(prep: Prepared, flat: dict, steps_done: int, job: Job, base: str,
                  final_loss: float | None) -> Path:
    spec = prep.spec
    dest = adapters_dir() / spec.name
    extra = {"base": spec.model, "quant": spec.quant, "tune_job": job.id, "path": spec.path,
             "steps": steps_done, "examples": prep.stats.examples,
             "data_sha256": data_hash(Path(spec.data)), "max_seq": spec.max_seq,
             "micro_batch": spec.micro_batch, "grad_accum": spec.grad_accum,
             "lr": spec.lr, "schedule": spec.schedule, "seed": spec.seed,
             "final_loss": final_loss}
    lo.save_adapter_dir(dest, flat, spec.lora(), base, prep.n_layers, prep.shapes, extra)
    log = job.dir / "losses.jsonl"
    if log.exists():
        shutil.copy(log, dest / "train_log.jsonl")
    return dest




def with_stop_after(spec: TuneSpec, stop, progress_cb):
    """`stop_after`: set the stop event once that many steps are done (tests, demos, and the
    cross-hardware gates stop a job part-way and continue it elsewhere)."""
    if not spec.stop_after:
        return progress_cb
    inner = progress_cb

    def cb(info):
        if inner:
            inner(info)
        if info["step"] >= spec.stop_after:
            stop.set()
    return cb
