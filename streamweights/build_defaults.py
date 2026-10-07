"""Default student and teacher for `spill build`, chosen from the probed machine and the
engine's calibrated rates, never declared.

  Apple silicon  qwen2.5:7b student, llama3.3:70b teacher (the models the engine is built for)
  CUDA           the same when device memory and disk allow; otherwise the largest that fit
  CPU            qwen2.5:0.5b student, and the largest teacher whose estimated distill time
                 for this folder's prompts is under 12 hours

A model the user names is never second-guessed: if its estimate is long the build says so
and goes on.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import build as B
from . import estimate as est

STUDENT_APPLE = "qwen2.5:7b"
TEACHER_APPLE = "llama3.3:70b"
SMALL = "qwen2.5:0.5b"
TEACHERS = ["llama3.3:70b", "qwen2.5:32b", "qwen2.5:7b", "qwen2.5:0.5b"]   # largest first
CPU_TEACHER_LIMIT_S = 12 * 3600
LONG_RUN_S = 24 * 3600
DISK_FLOOR = 20 * 1024**3


@dataclass
class Choice:
    student: str
    teacher: str
    notes: list[str] = field(default_factory=list)      # the "why", one clause per choice
    forced: set = field(default_factory=set)            # roles the user named

    def why(self) -> str:
        return "; ".join(self.notes)


def _gb(n: float) -> str:
    return f"{n / 1e9:.0f} GB"


def distill_estimate(folder: B.Folder, tag: str, cal: dict, working_set: int, engine: str,
                     read_rate: float | None) -> est.StageEstimate | None:
    """Estimated seconds for `tag` to answer the folder's prompts.jsonl on `engine`."""
    if not folder.prompts:
        return None
    cls = B.is_classification(folder.evals)
    mt = B.classification_max_tokens(folder.evals) if cls else B.DEFAULT_MAX_TOKENS
    wl = B.workload(folder.prompts, folder.instructions, mt, cls)
    tf, _ = est.tflops_for(cal, engine, tag)
    return est.eval_seconds(tag, wl, cal, working_set, tf, reuse=True, engine=engine,
                            read_rate=read_rate)


def choose(folder: B.Folder, engine: str, cal: dict, working_set: int, free_bytes: int,
           read_rate: float | None = None, student: str | None = None,
           teacher: str | None = None, downloaded=lambda tag: False) -> Choice:
    """Pure: everything it needs is passed in, so it can be tested without a machine."""
    c = Choice(student or STUDENT_APPLE, teacher or TEACHER_APPLE)
    if student:
        c.forced.add("student")
    if teacher:
        c.forced.add("teacher")
    needs_teacher = folder.prompts is not None
    disk_ok = lambda tag: downloaded(tag) or est.model_bytes(tag) + DISK_FLOOR <= free_bytes

    if engine == "mlx":
        if not student:
            c.notes.append(f"student {c.student}: the Apple silicon default")
        if needs_teacher and not teacher:
            c.notes.append(f"teacher {c.teacher}: the Apple silicon default")
        return c

    if engine == "torch-cuda":
        if not student:
            fits = est.model_bytes(STUDENT_APPLE) <= working_set * 0.70
            c.student = STUDENT_APPLE if fits and disk_ok(STUDENT_APPLE) else SMALL
            c.notes.append(
                f"student {c.student}: " + (
                    f"{STUDENT_APPLE} ({_gb(est.model_bytes(STUDENT_APPLE))}) fits 70% of the "
                    f"{_gb(working_set)} device memory and the disk" if c.student == STUDENT_APPLE
                    else f"{STUDENT_APPLE} does not fit {_gb(working_set)} of device memory "
                         f"and the disk, so the largest that fits"))
        if needs_teacher and not teacher:
            pick = next((t for t in TEACHERS if disk_ok(t)), SMALL)
            c.teacher = pick
            c.notes.append(
                f"teacher {pick}: " + (
                    f"streams from disk, {_gb(est.model_bytes(pick))} of the "
                    f"{_gb(free_bytes)} free" if pick == TEACHER_APPLE else
                    f"the largest teacher the {_gb(free_bytes)} of free disk holds "
                    f"({TEACHER_APPLE} needs {_gb(est.model_bytes(TEACHER_APPLE) + DISK_FLOOR)})"))
        return c

    # torch-cpu
    if not student:
        c.student = SMALL
        c.notes.append(f"student {SMALL}: the CPU default")
    if needs_teacher and not teacher:
        times = {}
        pick = None
        for t in TEACHERS:
            if not disk_ok(t):
                continue
            e = distill_estimate(folder, t, cal, working_set, engine, read_rate)
            times[t] = e.seconds
            if pick is None and e.seconds <= CPU_TEACHER_LIMIT_S:
                pick = t
        pick = pick or SMALL
        c.teacher = pick
        n = len(B._rows(folder.prompts))
        if times.get(pick) is not None:
            tail = ""
            bigger = [t for t in TEACHERS if t in times and times[t] > CPU_TEACHER_LIMIT_S]
            if bigger:
                tail = f"; {bigger[0]} would take {est.fmt_dur(times[bigger[0]])}"
            c.notes.append(f"teacher {pick}: the largest whose estimated distill of {n:,} "
                           f"prompts on {engine} is under 12 h ({est.fmt_dur(times[pick])}{tail})")
        else:
            c.notes.append(f"teacher {pick}: the largest that fits the disk")
    return c


def machine_inputs(engine: str, hw: dict, cal: dict) -> tuple[int, int, float | None]:
    """(working set bytes, free disk bytes, probed read rate) for the engine on this machine."""
    from .registry import MODELS_DIR
    if engine == "mlx":
        ws = hw["gpu"]["vram_bytes"]
    else:
        from .engines.torch_common import memory_total_bytes
        ws = memory_total_bytes(engine)
    probe = MODELS_DIR if MODELS_DIR.exists() else Path.home()
    free = shutil.disk_usage(probe).free
    from .policy import engine_read_rate
    try:
        rate, _ = engine_read_rate(cal, hw)
    except (KeyError, TypeError):
        rate = None
    return int(ws), int(free), rate
