"""llamacpp: the Phase 0 GGUF path behind the engine interface.
Retained as the non-Apple path (and for explicit GGUF quants)."""

from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

import httpx

from ..jobs.engine import LLAMA_SERVER, _free_port
from .base import CompletedRow, MemoryBudget, ModelSpec


class LlamaCppEngine:
    name = "llamacpp"

    def run_batch(self, rows: list[dict], spec: ModelSpec,
                  budget: MemoryBudget) -> Iterator[CompletedRow]:
        parallel = budget.batch_override or 8
        ngl = spec.extra.get("n_gpu_layers", 999)
        port = _free_port()
        cmd = [str(LLAMA_SERVER), "-m", str(spec.path), "--load-mode", "mmap",
               "-ngl", str(ngl), "--parallel", str(parallel),
               "-c", str(spec.ctx * parallel), "--cont-batching",
               "--port", str(port), "--host", "127.0.0.1", "--no-webui"]
        log = open(spec.extra.get("log_path", "/dev/null"), "ab")
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
        stop_event = spec.extra.get("stop_event")
        try:
            with httpx.Client() as c:
                t0 = time.monotonic()
                while time.monotonic() - t0 < 1800:
                    if proc.poll() is not None:
                        raise RuntimeError("llama-server failed to start")
                    try:
                        if c.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(1)

            def one(r):
                t = time.monotonic()
                body = dict(r["body"])
                body["model"] = spec.name
                resp = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                  json=body, timeout=3600).json()
                usage = resp.get("usage", {})
                choice = (resp.get("choices") or [{}])[0]
                return CompletedRow(
                    custom_id=r["custom_id"],
                    content=choice.get("message", {}).get("content", ""),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    latency_s=round(time.monotonic() - t, 3),
                    finish_reason=choice.get("finish_reason", "stop"),
                    error=json.dumps(resp.get("error")) if resp.get("error") else None,
                    batch_size=parallel)

            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futs = [pool.submit(one, r) for r in rows]
                for f in as_completed(futs):
                    if stop_event and stop_event.is_set():
                        break
                    yield f.result()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(20)
                except subprocess.TimeoutExpired:
                    proc.kill()
