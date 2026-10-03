"""Batch engine: OpenAI batch JSONL in, llama-server with continuous batching
underneath, checkpointed results.jsonl out. Interruptible and resumable."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..registry import GIB, REPO_ROOT, Model

JOBS_DIR = REPO_ROOT / "jobs"
LLAMA_SERVER = REPO_ROOT / "bin" / "llama-server"

VRAM_HEADROOM = 0.90        # fraction of VRAM usable for weights+KV
WEIGHT_VRAM_FRACTION = 0.75  # cap GPU-offloaded weights so KV has room


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def compute_offload(model: Model, quant: str, ctx: int, vram_bytes: int,
                    parallel: int | None = None) -> tuple[int, int]:
    """Return (n_gpu_layers, n_parallel). KV-bound: N comes from free VRAM
    after weights at the requested per-slot context."""
    weights = model.quants[quant].bytes
    n_layers = model.arch["n_layers"]
    per_layer = weights / n_layers
    kv_per_slot = model.kv_bytes_per_token() * ctx
    budget = int(vram_bytes * VRAM_HEADROOM)
    if weights <= WEIGHT_VRAM_FRACTION * budget:
        ngl = n_layers + 1  # fully resident (+1 covers the output layer)
        gpu_weights = weights
    else:
        ngl = max(1, int(WEIGHT_VRAM_FRACTION * budget / per_layer))
        gpu_weights = int(ngl * per_layer)
    n = max(1, (budget - gpu_weights) // kv_per_slot)
    n = min(n, 256)
    if parallel is not None:
        n = parallel  # explicit override (measurement mode); OOM halving still applies
    return ngl, int(n)


@dataclass
class Job:
    id: str
    dir: Path
    model: str
    quant: str
    ctx: int
    parallel: int
    total: int

    @property
    def input_path(self) -> Path:
        return self.dir / "input.jsonl"

    @property
    def checkpoint_path(self) -> Path:
        return self.dir / "checkpoint"

    @property
    def results_path(self) -> Path:
        return self.dir / "results.jsonl"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    def done_ids(self) -> set[str]:
        if not self.checkpoint_path.exists():
            return set()
        return set(self.checkpoint_path.read_text().split())

    def write_meta(self, **kw) -> None:
        meta = self.read_meta()
        meta.update(kw)
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(self.meta_path)

    def read_meta(self) -> dict:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text())
            except json.JSONDecodeError:
                pass
        return {}

    @classmethod
    def create(cls, input_jsonl: Path, model: str, quant: str, ctx: int,
               parallel: int, out: Path | None = None,
               rows: list[dict] | None = None, options: dict | None = None) -> "Job":
        """`rows`: already-validated rows in the batch shape (the normalized form of
        a chat or eval file); otherwise the input file's lines are copied verbatim."""
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        d = JOBS_DIR / job_id
        d.mkdir(parents=True)
        if rows is not None:
            lines = [json.dumps(r) for r in rows]
        else:
            lines = [l for l in input_jsonl.read_text().splitlines() if l.strip()]
        (d / "input.jsonl").write_text("\n".join(lines) + "\n")
        job = cls(job_id, d, model, quant, ctx, parallel, len(lines))
        job.write_meta(id=job_id, model=model, quant=quant, ctx=ctx, parallel=parallel,
                       total=len(lines), status="created", out=str(out) if out else None,
                       input_path=str(input_jsonl), options=options or {},
                       created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        return job

    @classmethod
    def load(cls, job_id: str) -> "Job":
        d = JOBS_DIR / job_id
        meta = json.loads((d / "meta.json").read_text())
        return cls(job_id, d, meta["model"], meta["quant"], meta["ctx"],
                   meta["parallel"], meta["total"])

    @classmethod
    def latest(cls) -> "Job | None":
        if not JOBS_DIR.exists():
            return None
        dirs = sorted((p for p in JOBS_DIR.iterdir() if (p / "meta.json").exists()),
                      key=lambda p: p.name)
        return cls.load(dirs[-1].name) if dirs else None


@dataclass
class Progress:
    total: int = 0
    done: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    started: float = field(default_factory=time.monotonic)

    @property
    def tokens_per_sec(self) -> float:
        el = time.monotonic() - self.started
        return self.completion_tokens / el if el > 0 else 0.0

    @property
    def eta_seconds(self) -> float | None:
        if self.done == 0:
            return None
        el = time.monotonic() - self.started
        return el / self.done * (self.total - self.done)


class BatchEngine:
    def __init__(self, job: Job, model: Model, gguf_paths: list[Path],
                 n_gpu_layers: int, tier: str = "batch",
                 progress_cb=None, extra_server_args: list[str] | None = None):
        self.job = job
        self.model = model
        self.gguf = gguf_paths[0]
        self.ngl = n_gpu_layers
        self.tier = tier
        self.progress_cb = progress_cb or (lambda p: None)
        self.extra_server_args = extra_server_args or []
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self._stop = asyncio.Event()

    # ---------- server lifecycle ----------

    def _start_server(self, parallel: int) -> None:
        self.port = _free_port()
        cmd = [
            str(LLAMA_SERVER), "-m", str(self.gguf),
            "--load-mode", "mmap",
            "-ngl", str(self.ngl),
            "--parallel", str(parallel),
            "-c", str(self.job.ctx * parallel),
            "--cont-batching",
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--no-webui",
            *self.extra_server_args,
        ]
        log = open(self.job.dir / "llama-server.log", "ab")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=log,
                                     start_new_session=True)

    async def _wait_healthy(self, timeout: float = 1800) -> bool:
        t0 = time.monotonic()
        async with httpx.AsyncClient() as c:
            while time.monotonic() - t0 < timeout:
                if self.proc.poll() is not None:
                    return False  # died (OOM or load failure)
                try:
                    r = await c.get(f"http://127.0.0.1:{self.port}/health", timeout=2)
                    if r.status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.0)
        return False

    def _stop_server(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    # ---------- run ----------

    async def run(self) -> Progress:
        job = self.job
        parallel = job.parallel
        while True:
            self._start_server(parallel)
            job.write_meta(status="loading", parallel=parallel, port=self.port)
            if await self._wait_healthy():
                break
            self._stop_server()
            if parallel <= 1:
                job.write_meta(status="failed", error="server failed to start at parallel=1")
                raise RuntimeError("llama-server failed to start even at parallel=1 "
                                   f"(see {job.dir}/llama-server.log)")
            parallel //= 2  # halve on OOM
            job.write_meta(note=f"server OOM/failed; halved parallel to {parallel}")
        job.parallel = parallel

        done = job.done_ids()
        rows = []
        for line in job.input_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["custom_id"] not in done:
                rows.append(row)

        prog = Progress(total=job.total, done=len(done))
        job.write_meta(status="running", parallel=parallel)

        ckpt = open(job.checkpoint_path, "a")
        results = open(job.results_path, "a")
        lock = asyncio.Lock()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except (NotImplementedError, RuntimeError):
                pass

        sem = asyncio.Semaphore(parallel)

        async def one(row: dict, client: httpx.AsyncClient):
            async with sem:
                if self._stop.is_set():
                    return
                body = dict(row["body"])
                body["model"] = job.model  # the job's model is authoritative
                t0 = time.monotonic()
                try:
                    r = await client.post(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                          json=body, timeout=3600)
                    latency = time.monotonic() - t0
                    rbody = r.json()
                    usage = rbody.get("usage", {})
                    out = {
                        "id": f"batch_req_{uuid.uuid4().hex[:12]}",
                        "custom_id": row["custom_id"],
                        "response": {"status_code": r.status_code,
                                     "request_id": rbody.get("id", ""),
                                     "body": rbody},
                        "error": None,
                        "streamweights": {
                            "tier": self.tier,
                            "quant": job.quant,
                            "batch": parallel,
                            "tokens": {"prompt": usage.get("prompt_tokens", 0),
                                       "completion": usage.get("completion_tokens", 0)},
                            "latency_s": round(latency, 3),
                        },
                    }
                except (httpx.HTTPError, json.JSONDecodeError) as e:
                    if self._stop.is_set():
                        return
                    out = {
                        "id": f"batch_req_{uuid.uuid4().hex[:12]}",
                        "custom_id": row["custom_id"],
                        "response": None,
                        "error": {"code": "request_failed", "message": str(e)},
                        "streamweights": {"tier": self.tier, "quant": job.quant,
                                     "batch": parallel, "tokens": {}, "latency_s": None},
                    }
                async with lock:
                    results.write(json.dumps(out) + "\n")
                    results.flush()
                    ckpt.write(row["custom_id"] + "\n")
                    ckpt.flush()
                    prog.done += 1
                    tok = out["streamweights"]["tokens"]
                    prog.completion_tokens += tok.get("completion", 0) or 0
                    prog.prompt_tokens += tok.get("prompt", 0) or 0
                    if prog.done % 5 == 0 or prog.done == prog.total:
                        job.write_meta(done=prog.done,
                                       tokens_per_sec=round(prog.tokens_per_sec, 2),
                                       eta_seconds=round(prog.eta_seconds or 0))
                self.progress_cb(prog)

        try:
            async with httpx.AsyncClient() as client:
                tasks = [asyncio.create_task(one(r, client)) for r in rows]
                gathered = asyncio.ensure_future(asyncio.gather(*tasks, return_exceptions=True))
                stopper = asyncio.create_task(self._stop.wait())
                await asyncio.wait([gathered, stopper], return_when=asyncio.FIRST_COMPLETED)
                stopper.cancel()
                if self._stop.is_set():
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    job.write_meta(status="interrupted", done=prog.done,
                                   tokens_per_sec=round(prog.tokens_per_sec, 2))
                else:
                    job.write_meta(status="completed", done=prog.done,
                                   tokens_per_sec=round(prog.tokens_per_sec, 2),
                                   eta_seconds=0)
        finally:
            ckpt.close()
            results.close()
            self._stop_server()
        return prog
