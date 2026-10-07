"""Engine-agnostic job runner: checkpoint per row, results.jsonl in OpenAI batch
shape + streamweights metadata, SIGINT-clean, resumable."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
import uuid

from .. import runs
from ..engines.base import CompletedRow, MemoryBudget, ModelSpec
from .engine import Job, Progress


def result_row(cr: CompletedRow, job: Job, engine_name: str,
               prov: dict | None = None, producer: dict | None = None) -> dict:
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
            **({"hardware": producer["hardware"], "numerics": producer["numerics"]}
               if producer else {}),
        },
    }
    if cr.logprobs is not None:
        row["logprobs"] = cr.logprobs
    if cr.extra:
        row.update(cr.extra)
    return row


def producer_of(engine, spec: ModelSpec) -> dict:
    """What produces a row: engine, hardware, numerics (stamped on every result row and on
    every pushed segment of portable state)."""
    d = engine.describe(spec) if hasattr(engine, "describe") else {}
    return {"engine": engine.name, "hardware": d.get("device", "unknown"),
            "numerics": d.get("numerics", {"base": spec.quant})}


def _attach_state(job: Job, spec: ModelSpec, producer: dict):
    """Open the portable state for this job when --state is set: restore what other machines
    already finished into this job's directory, and return the sync object."""
    from .. import runtime
    uri = runtime.state_uri()
    if not uri:
        return None
    import hashlib

    from ..errors import SpillError
    from ..portable.rows import RowSync
    from ..portable.store import Store
    store = Store(uri)
    ident = {"model": spec.name, "quant": spec.quant,
             "input_sha256": hashlib.sha256(job.input_path.read_bytes()).hexdigest(),
             "rows": job.total, "options": job.read_meta().get("options", {})}
    if store.exists("rows/identity.json"):
        have = json.loads(store.read("rows/identity.json"))
        if have.get("input_sha256") != ident["input_sha256"] or have.get("model") != ident["model"]:
            raise SpillError(f"the state at {uri} belongs to a different job (model "
                             f"{have.get('model')}, another input)",
                             "spill ... --state <a new location>")
    else:
        store.write("rows/identity.json", json.dumps(ident).encode())
    sync = RowSync(store, job.results_path, job.checkpoint_path, producer)
    restored = sync.attach()
    if restored:
        runtime.emit("restore", step=restored, uri=uri)
    return sync


def run_job(job: Job, engine, spec: ModelSpec, budget: MemoryBudget,
            progress_cb=None) -> Progress:
    from .. import runtime
    from ..headless import QuantumAbandoned
    env = runtime.ENV
    progress_cb = progress_cb or (lambda p: None)  # may be a no-op; the engine's
    # per-pass callback owns the single updating progress line
    producer = producer_of(engine, spec)
    sync = _attach_state(job, spec, producer)
    done = job.done_ids()
    rows = []
    for line in job.input_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["custom_id"] not in done:
                rows.append(r)

    prog = Progress(total=job.total, done=len(done))
    stop_event = env.stop if env.guard is not None else threading.Event()
    spec.extra["stop_event"] = stop_event
    prev = {}
    if env.guard is None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev[sig] = signal.signal(sig, lambda *a: stop_event.set())
            except ValueError:
                pass

    meta0 = job.read_meta()
    prov = meta0.get("provenance")
    if meta0.get("run_id") and done:
        runs.mark_resumed(meta0["run_id"])
    job.write_meta(status="running", engine=engine.name, pid=os.getpid())
    ckpt = open(job.checkpoint_path, "a")
    results = open(job.results_path, "a")
    gen = engine.run_batch(rows, spec, budget)
    abandoned = False
    try:
        try:
            for cr in gen:
                results.write(json.dumps(result_row(cr, job, engine.name, prov, producer)) + "\n")
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
                runtime.emit("row", step=prog.done, total=prog.total, loss=None,
                             tokens_per_s=round(prog.tokens_per_sec, 2),
                             eta_s=round(prog.eta_seconds, 1) if prog.eta_seconds else None,
                             peak_mem_gb=_peak(), custom_id=cr.custom_id)
                if sync is not None:
                    ckpt.flush()
                    if sync.maybe_push(prog.done):
                        runtime.emit("checkpoint", step=prog.done, uri=runtime.state_uri(),
                                     rows=prog.done)
                if env.stop_after and prog.done >= env.stop_after and prog.done < prog.total:
                    env.stopped_early = True
                    stop_event.set()
                    break
        except QuantumAbandoned:
            abandoned = True
        finally:
            gen.close()
        status = "interrupted" if stop_event.is_set() or abandoned else "completed"
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
        if sync is not None:
            if env.guard is not None:
                env.guard.disarm()
            n = sync.push()
            if n:
                runtime.emit("checkpoint", step=prog.done, uri=runtime.state_uri(),
                             rows=prog.done)
        for sig, h in prev.items():
            try:
                signal.signal(sig, h)
            except ValueError:
                pass
    runtime.ENV.summary.update(rows_done=prog.done, rows_total=prog.total, job=job.id,
                               results=str(job.results_path))
    return prog


def _peak():
    from ..headless import peak_mem_gb
    return peak_mem_gb()
