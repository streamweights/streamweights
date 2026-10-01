"""Spillway gateway: OpenAI files+batches shapes and Ollama /api/tags, port 11435.
Runs beside Ollama (11434). The CLI uses this if running, in-process otherwise."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from . import probe as probe_mod
from .jobs.engine import BatchEngine, Job, JOBS_DIR, compute_offload
from .policy import choose_quant
from .registry import GIB, REPO_ROOT, download, load_registry

app = FastAPI(title="spillway")

FILES_DIR = REPO_ROOT / "state" / "files"
BATCHES: dict[str, dict] = {}  # batch_id -> OpenAI batch object (+ _job_id)
_batch_lock = threading.Lock()


def _file_meta_path(file_id: str) -> Path:
    return FILES_DIR / f"{file_id}.json"


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- OpenAI /v1/files ----------

@app.post("/v1/files")
async def create_file(file: UploadFile, purpose: str = "batch"):
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    file_id = "file-" + uuid.uuid4().hex[:24]
    data = await file.read()
    (FILES_DIR / file_id).write_bytes(data)
    obj = {
        "id": file_id, "object": "file", "bytes": len(data),
        "created_at": int(time.time()), "filename": file.filename or "input.jsonl",
        "purpose": purpose,
    }
    _file_meta_path(file_id).write_text(json.dumps(obj))
    return obj


@app.get("/v1/files/{file_id}")
def get_file(file_id: str):
    p = _file_meta_path(file_id)
    if not p.exists():
        raise HTTPException(404, "file not found")
    return json.loads(p.read_text())


@app.get("/v1/files/{file_id}/content", response_class=PlainTextResponse)
def file_content(file_id: str):
    p = FILES_DIR / file_id
    if not p.exists():
        raise HTTPException(404, "file not found")
    return p.read_text()


# ---------- OpenAI /v1/batches ----------

def _batch_obj(batch_id: str) -> dict:
    b = BATCHES.get(batch_id)
    if not b:
        raise HTTPException(404, "batch not found")
    job_id = b.get("_job_id")
    if job_id:
        try:
            meta = Job.load(job_id).read_meta()
            done, total = meta.get("done", 0), meta.get("total", 0)
            b["request_counts"] = {"total": total, "completed": done, "failed": 0}
            status = meta.get("status", "created")
            b["status"] = {"created": "validating", "loading": "in_progress",
                           "running": "in_progress", "completed": "completed",
                           "interrupted": "cancelled", "failed": "failed"}.get(status, "in_progress")
            if b["status"] == "completed" and not b.get("output_file_id"):
                out_id = "file-" + uuid.uuid4().hex[:24]
                FILES_DIR.mkdir(parents=True, exist_ok=True)
                src = Job.load(job_id).results_path
                (FILES_DIR / out_id).write_bytes(src.read_bytes())
                _file_meta_path(out_id).write_text(json.dumps(
                    {"id": out_id, "object": "file", "bytes": src.stat().st_size,
                     "created_at": int(time.time()), "filename": "results.jsonl",
                     "purpose": "batch_output"}))
                b["output_file_id"] = out_id
        except FileNotFoundError:
            pass
    return {k: v for k, v in b.items() if not k.startswith("_")}


def _run_job_thread(job: Job, model_name: str, quant: str):
    hw = probe_mod.load()
    reg = load_registry()
    m = reg[model_name]
    paths = download(m, quant)
    ngl, _ = compute_offload(m, quant, job.ctx, hw["gpu"]["vram_bytes"], job.parallel)
    engine = BatchEngine(job, m, paths, ngl)
    asyncio.run(engine.run())


@app.post("/v1/batches")
def create_batch(body: dict):
    input_file_id = body.get("input_file_id")
    src = FILES_DIR / (input_file_id or "")
    if not input_file_id or not src.exists():
        raise HTTPException(400, "unknown input_file_id")
    # model comes from the first row's body (OpenAI batch rows carry it)
    first = json.loads(src.read_text().splitlines()[0])
    model_name = first["body"]["model"]
    reg = load_registry()
    if model_name not in reg:
        raise HTTPException(400, f"unknown model {model_name}")
    hw = probe_mod.load()
    m = reg[model_name]
    choice = choose_quant(m, hw)
    ctx = 4096
    _, n = compute_offload(m, choice.quant, ctx, hw["gpu"]["vram_bytes"])
    job = Job.create(src, model_name, choice.quant, ctx, n)
    batch_id = "batch_" + uuid.uuid4().hex[:24]
    obj = {
        "id": batch_id, "object": "batch", "endpoint": body.get("endpoint", "/v1/chat/completions"),
        "errors": None, "input_file_id": input_file_id,
        "completion_window": body.get("completion_window", "24h"),
        "status": "validating", "output_file_id": None, "error_file_id": None,
        "created_at": int(time.time()), "in_progress_at": None, "expires_at": None,
        "finalizing_at": None, "completed_at": None, "failed_at": None,
        "expired_at": None, "cancelling_at": None, "cancelled_at": None,
        "request_counts": {"total": job.total, "completed": 0, "failed": 0},
        "metadata": body.get("metadata"),
        "_job_id": job.id,
    }
    with _batch_lock:
        BATCHES[batch_id] = obj
    threading.Thread(target=_run_job_thread, args=(job, model_name, choice.quant),
                     daemon=True).start()
    return _batch_obj(batch_id)


@app.get("/v1/batches")
def list_batches(limit: int = 20):
    return {"object": "list",
            "data": [_batch_obj(bid) for bid in list(BATCHES)][:limit],
            "has_more": False}


@app.get("/v1/batches/{batch_id}")
def get_batch(batch_id: str):
    return _batch_obj(batch_id)


@app.post("/v1/batches/{batch_id}/cancel")
def cancel_batch(batch_id: str):
    b = BATCHES.get(batch_id)
    if not b:
        raise HTTPException(404, "batch not found")
    job_id = b.get("_job_id")
    if job_id:
        Job.load(job_id).write_meta(status="interrupted")
    b["status"] = "cancelled"
    b["cancelled_at"] = int(time.time())
    return _batch_obj(batch_id)


# ---------- Ollama shape ----------

@app.get("/api/tags")
def tags():
    reg = load_registry()
    models = []
    for name, m in reg.items():
        for qn, q in m.quants.items():
            if q.downloaded(name):
                models.append({
                    "name": name, "model": name,
                    "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "size": q.bytes,
                    "digest": "",
                    "details": {"format": "gguf", "family": name.split(":")[0],
                                "parameter_size": name.split(":")[1].upper(),
                                "quantization_level": qn},
                })
                break
        else:
            models.append({"name": name, "model": name,
                           "modified_at": "", "size": m.quants["bf16"].bytes,
                           "digest": "", "details": {"format": "gguf",
                           "family": name.split(":")[0],
                           "parameter_size": name.split(":")[1].upper(),
                           "quantization_level": "not downloaded"}})
    return {"models": models}


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=11435)


if __name__ == "__main__":
    main()
