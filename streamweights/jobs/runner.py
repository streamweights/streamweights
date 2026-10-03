"""Engine-agnostic job runner: checkpoint per row, results.jsonl in OpenAI batch
shape + streamweights metadata, SIGINT-clean, resumable."""

from __future__ import annotations

import json
import signal
import threading
import time
import uuid

from .. import runs
from ..engines.base import CompletedRow, MemoryBudget, ModelSpec
from .engine import Job, Progress


def result_row(cr: CompletedRow, job: Job, engine_name: str,
               prov: dict | None = None) -> dict:
    ok = cr.error is None
    body = {
        "choices": [{"finish_reason": cr.finish_reason, "index": 0,
                     "message": {"role": "assistant", "content": cr.content}}],
        "model": job.model,
        "object": "chat.completion",
        "usage": {"prompt_tokens": cr.prompt_tokens,
                  "completion_tokens": cr.completion_tokens,
                  "total_tokens": cr.prompt_tokens + cr.completion_tokens},
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
    }
    row = {
        "id": f"batch_req_{uuid.uuid4().hex[:12]}",
        "custom_id": cr.custom_id,
        "response": {"status_code": 200, "request_id": body["id"], "body": body} if ok else None,
        "error": None if ok else {"code": "engine_error", "message": cr.error},
        "streamweights": {
            "tier": "batch", "engine": engine_name, "quant": job.quant,
            "batch": cr.batch_size,
            "tokens": {"prompt": cr.prompt_tokens, "completion": cr.completion_tokens},
            "latency_s": cr.latency_s,
            **({"provenance": prov} if prov else {}),
        },
    }
    if cr.logprobs is not None:
        row["logprobs"] = cr.logprobs
    if cr.extra:
        row.update(cr.extra)
    return row


def run_job(job: Job, engine, spec: ModelSpec, budget: MemoryBudget,
            progress_cb=None) -> Progress:
    progress_cb = progress_cb or (lambda p: None)  # may be a no-op; the engine's
    # per-pass callback owns the single updating progress line
    done = job.done_ids()
    rows = []
    for line in job.input_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["custom_id"] not in done:
                rows.append(r)

    prog = Progress(total=job.total, done=len(done))
    stop_event = threading.Event()
    spec.extra["stop_event"] = stop_event
    prev = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev[sig] = signal.signal(sig, lambda *a: stop_event.set())
        except ValueError:
            pass

    meta0 = job.read_meta()
    prov = meta0.get("provenance")
    if meta0.get("run_id") and done:
        runs.mark_resumed(meta0["run_id"])
    job.write_meta(status="running", engine=engine.name)
    ckpt = open(job.checkpoint_path, "a")
    results = open(job.results_path, "a")
    try:
        for cr in engine.run_batch(rows, spec, budget):
            results.write(json.dumps(result_row(cr, job, engine.name, prov)) + "\n")
            results.flush()
            ckpt.write(cr.custom_id + "\n")
            ckpt.flush()
            prog.done += 1
            prog.completion_tokens += cr.completion_tokens
            prog.prompt_tokens += cr.prompt_tokens
            job.write_meta(done=prog.done,
                           tokens_per_sec=round(prog.tokens_per_sec, 2),
                           eta_seconds=round(prog.eta_seconds or 0))
            progress_cb(prog)
        status = "interrupted" if stop_event.is_set() else "completed"
        if prog.done >= prog.total:
            status = "completed"
        job.write_meta(status=status, done=prog.done,
                       tokens_per_sec=round(prog.tokens_per_sec, 2))
        if meta0.get("run_id"):
            runs.finish_run(meta0["run_id"], status=status, rows_done=prog.done,
                            tokens={"prompt": prog.prompt_tokens,
                                    "completion": prog.completion_tokens},
                            results_path=job.results_path)
    except BaseException:
        if meta0.get("run_id"):
            runs.finish_run(meta0["run_id"], status="failed", rows_done=prog.done,
                            results_path=job.results_path)
        job.write_meta(status="failed")
        raise
    finally:
        ckpt.close()
        results.close()
        for sig, h in prev.items():
            try:
                signal.signal(sig, h)
            except ValueError:
                pass
    return prog
