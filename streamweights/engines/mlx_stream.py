"""mlx_stream: the Phase 1 streaming runner.

Weights live on NVMe, never resident. A reader thread fills a ring of
preallocated host buffers with one transformer layer each (large preads,
F_NOCACHE), running ahead of compute; compute binds each layer's bytes to a
single reusable TransformerBlock and pushes the whole batch through it.
One full weight read per forward pass, amortized over the batch.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np

from ..registry import GIB, REPO_ROOT
from .base import CompletedRow, MemoryBudget, ModelSpec

CALIBRATION_JSON = REPO_ROOT / "state" / "calibration.json"

MB = 1024 * 1024

ST_DTYPES = {
    "BF16": (np.uint16, mx.bfloat16, 2),
    "F16": (np.float16, mx.float16, 2),
    "F32": (np.float32, mx.float32, 4),
    "U32": (np.uint32, mx.uint32, 4),   # packed quantized weights
    "I32": (np.int32, mx.int32, 4),
}


# ---------------------------------------------------------------- index

@dataclass
class TensorLoc:
    name: str          # safetensors key
    shard: Path
    offset: int        # absolute byte offset in shard
    nbytes: int
    shape: tuple
    st_dtype: str


@dataclass
class LayerPlan:
    """All tensors of one streamed unit (a transformer block), with a coalesced read plan."""
    layer_id: int
    tensors: list[TensorLoc]
    segments: list[tuple[Path, int, int, int]] = field(default_factory=list)
    # (shard, file_offset, length, buf_offset); tensor buf offsets assigned in order
    tensor_buf_offsets: dict[str, int] = field(default_factory=dict)
    nbytes: int = 0

    def build(self):
        self.tensors.sort(key=lambda t: (str(t.shard), t.offset))
        buf_off = 0
        segs: list[list] = []
        for t in self.tensors:
            self.tensor_buf_offsets[t.name] = buf_off
            if segs and segs[-1][0] == t.shard and segs[-1][1] + segs[-1][2] == t.offset:
                segs[-1][2] += t.nbytes
            else:
                segs.append([t.shard, t.offset, t.nbytes, buf_off])
            buf_off += t.nbytes
        self.segments = [tuple(s) for s in segs]
        self.nbytes = buf_off


class SafetensorsIndex:
    """Parse all shard headers (no tensor data) and group tensors by transformer
    block in execution order, plus embeddings, final norm, lm_head."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text())
        self.tensors: dict[str, TensorLoc] = {}
        for shard in sorted(self.model_dir.glob("*.safetensors")):
            with open(shard, "rb") as f:
                (hlen,) = np.frombuffer(f.read(8), dtype=np.uint64)
                header = json.loads(f.read(int(hlen)))
                base = 8 + int(hlen)
                for name, meta in header.items():
                    if name == "__metadata__":
                        continue
                    s, e = meta["data_offsets"]
                    self.tensors[name] = TensorLoc(
                        name, shard, base + s, e - s, tuple(meta["shape"]), meta["dtype"])

        self.n_layers = self.config["num_hidden_layers"]
        self.layers: list[LayerPlan] = []
        for k in range(self.n_layers):
            prefix = f"model.layers.{k}."
            plan = LayerPlan(k, [t for n, t in self.tensors.items() if n.startswith(prefix)])
            if not plan.tensors:
                raise ValueError(f"no tensors found for layer {k}")
            plan.build()
            self.layers.append(plan)
        self.max_layer_bytes = max(p.nbytes for p in self.layers)

        self.embed = self.tensors["model.embed_tokens.weight"]
        self.final_norm = self.tensors["model.norm.weight"]
        self.lm_head = self.tensors.get("lm_head.weight")  # None if tied
        self.tied = self.lm_head is None


# ---------------------------------------------------------------- ring reader

class _Fds:
    """Per-shard fds with F_NOCACHE so streamed weights never pollute the page cache."""

    def __init__(self):
        self._fds: dict[Path, int] = {}
        self._lock = threading.Lock()

    def get(self, shard: Path) -> int:
        with self._lock:
            fd = self._fds.get(shard)
            if fd is None:
                fd = os.open(shard, os.O_RDONLY)
                if hasattr(fcntl, "F_NOCACHE"):
                    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                self._fds[shard] = fd
            return fd

    def close(self):
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


class RingReader:
    """N preallocated host buffers; a producer thread fills them with layers in
    schedule order using a thread pool of large preads; compute consumes in the
    same order. The free-slot semaphore guarantees a slot is never overwritten
    while still in use."""

    def __init__(self, index: SafetensorsIndex, n_slots: int = 3,
                 chunk_bytes: int = 16 * MB, n_threads: int = 4):
        self.index = index
        self.n_slots = n_slots
        self.chunk = chunk_bytes
        self.bufs = [bytearray(index.max_layer_bytes) for _ in range(n_slots)]
        self.free = threading.Semaphore(n_slots)
        self.ready: dict[int, int] = {}       # schedule seq -> slot id
        self.cond = threading.Condition()
        self.pool = ThreadPoolExecutor(max_workers=n_threads)
        self.fds = _Fds()
        self.bytes_read = 0
        self.read_seconds = 0.0
        self._stop = threading.Event()
        self._producer: threading.Thread | None = None

    def _read_segment(self, fd: int, file_off: int, length: int, buf: bytearray, buf_off: int):
        mv = memoryview(buf)
        done = 0
        while done < length:
            n = min(self.chunk, length - done)
            got = os.preadv(fd, [mv[buf_off + done: buf_off + done + n]], file_off + done)
            if got <= 0:
                raise IOError("short pread")
            done += got

    def _fill_slot(self, slot: int, plan: LayerPlan):
        t0 = time.monotonic()
        futs = []
        for shard, foff, length, boff in plan.segments:
            fd = self.fds.get(shard)
            # split large segments across the pool in chunk-sized pieces
            pos = 0
            while pos < length:
                n = min(self.chunk, length - pos)
                futs.append(self.pool.submit(
                    self._read_segment, fd, foff + pos, n, self.bufs[slot], boff + pos))
                pos += n
        for f in futs:
            f.result()
        self.bytes_read += plan.nbytes
        self.read_seconds += time.monotonic() - t0

    def start(self, schedule: list[int]):
        """schedule: list of layer ids in the exact order compute will consume them."""
        def produce():
            slot_order = list(range(self.n_slots))
            next_slot = 0
            for seq, layer_id in enumerate(schedule):
                if self._stop.is_set():
                    return
                self.free.acquire()
                if self._stop.is_set():
                    return
                slot = slot_order[next_slot % self.n_slots]
                next_slot += 1
                self._fill_slot(slot, self.index.layers[layer_id])
                with self.cond:
                    self.ready[seq] = slot
                    self.cond.notify_all()
        self._producer = threading.Thread(target=produce, daemon=True)
        self._producer.start()

    def get(self, seq: int) -> tuple[int, memoryview]:
        with self.cond:
            while seq not in self.ready:
                self.cond.wait(timeout=60)
        slot = self.ready.pop(seq)
        return slot, memoryview(self.bufs[slot])

    def release(self, slot: int):
        self.free.release()

    def stop(self):
        self._stop.set()
        self.free.release()  # unblock producer
        if self._producer:
            self._producer.join(timeout=10)
        self.pool.shutdown(wait=False)
        self.fds.close()


# ---------------------------------------------------------------- calibration

def calibrate(index: SafetensorsIndex, n_layers: int = 4,
              chunks_mb=(4, 16, 64), threads=(2, 4, 8)) -> dict:
    """Measure isolated read rate (no compute) over a grid; persist the best."""
    results = []
    for cmb in chunks_mb:
        for nt in threads:
            ring = RingReader(index, n_slots=3, chunk_bytes=cmb * MB, n_threads=nt)
            sched = list(range(min(n_layers, index.n_layers)))
            t0 = time.monotonic()
            ring.start(sched)
            total = 0
            for seq in range(len(sched)):
                slot, _ = ring.get(seq)
                total += index.layers[sched[seq]].nbytes
                ring.release(slot)
            dt = time.monotonic() - t0
            ring.stop()
            mbps = total / dt / MB
            results.append({"chunk_mb": cmb, "threads": nt, "mbps": round(mbps, 1)})
    best = max(results, key=lambda r: r["mbps"])
    cal = load_calibration()
    cal.update({
        "isolated_read_mbps": best["mbps"],
        "chunk_mb": best["chunk_mb"],
        "threads": best["threads"],
        "grid": results,
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    save_calibration(cal)
    return cal


def load_calibration() -> dict:
    if CALIBRATION_JSON.exists():
        try:
            return json.loads(CALIBRATION_JSON.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def save_calibration(cal: dict):
    CALIBRATION_JSON.parent.mkdir(exist_ok=True)
    CALIBRATION_JSON.write_text(json.dumps(cal, indent=2) + "\n")


# ---------------------------------------------------------------- compute

def _model_modules(config: dict):
    """One reusable TransformerBlock + args for the model's architecture."""
    mt = config["model_type"]
    if mt == "llama":
        from mlx_lm.models.llama import ModelArgs, TransformerBlock
    elif mt == "qwen2":
        from mlx_lm.models.qwen2 import ModelArgs, TransformerBlock
    else:
        raise ValueError(f"mlx_stream: unsupported architecture {mt}")
    args = ModelArgs.from_dict(config)
    block = TransformerBlock(args)
    q = config.get("quantization")
    if q:
        import mlx.nn as nn
        nn.quantize(block, group_size=q["group_size"], bits=q["bits"])
    return block, args


class StreamKVCache:
    """Per-layer KV cache compatible with mlx_lm attention modules."""

    STEP = 256

    def __init__(self, offset: int = 0):
        self.keys = None
        self.values = None
        self.offset = offset  # rope position of the next token

    def update_and_fetch(self, keys, values):
        B, H, L, D = keys.shape
        prev = self._len
        if self.keys is None:
            cap = ((prev + L + self.STEP - 1) // self.STEP) * self.STEP
            self.keys = mx.zeros((B, H, cap, D), keys.dtype)
            self.values = mx.zeros((B, H, cap, D), values.dtype)
            self._used = prev
        if self._used + L > self.keys.shape[2]:
            grow = ((L + self.STEP - 1) // self.STEP) * self.STEP
            self.keys = mx.concatenate(
                [self.keys, mx.zeros((B, H, grow, D), keys.dtype)], axis=2)
            self.values = mx.concatenate(
                [self.values, mx.zeros((B, H, grow, D), values.dtype)], axis=2)
        self.keys[..., self._used:self._used + L, :] = keys
        self.values[..., self._used:self._used + L, :] = values
        self._used += L
        self.offset += L
        return self.keys[..., :self._used, :], self.values[..., :self._used, :]

    @property
    def _len(self):
        return getattr(self, "_used", 0)

    def keep_rows(self, idx):
        if self.keys is not None:
            idx_arr = mx.array(idx)
            self.keys = self.keys[idx_arr]
            self.values = self.values[idx_arr]


def _bind(block, plan: LayerPlan, buf: memoryview, prefix: str) -> float:
    """Wrap slot bytes as mx arrays with right dtype/shape, bind to the block.
    Returns seconds spent (the copy cost; mx.array() copies from the numpy view)."""
    from mlx.utils import tree_unflatten
    t0 = time.monotonic()
    pairs = []
    for t in plan.tensors:
        npdt, mxdt, isz = ST_DTYPES[t.st_dtype]
        boff = plan.tensor_buf_offsets[t.name]
        arr = np.frombuffer(buf[boff:boff + t.nbytes], dtype=npdt).reshape(t.shape)
        a = mx.array(arr)
        if t.st_dtype == "BF16":
            a = a.view(mx.bfloat16)
        pairs.append((t.name.removeprefix(prefix), a))
    block.update(tree_unflatten(pairs))
    mx.eval(block.parameters())
    return time.monotonic() - t0


def _load_resident(loc: TensorLoc) -> mx.array:
    with open(loc.shard, "rb") as f:
        f.seek(loc.offset)
        raw = f.read(loc.nbytes)
    npdt, mxdt, _ = ST_DTYPES[loc.st_dtype]
    a = mx.array(np.frombuffer(raw, dtype=npdt).reshape(loc.shape))
    return a.view(mx.bfloat16) if loc.st_dtype == "BF16" else a


def _load_resident_maybe_quantized(index: "SafetensorsIndex", base: str) -> mx.array:
    """Load `base`.weight; if the checkpoint is quantized, dequantize to bf16."""
    loc = index.tensors[f"{base}.weight"]
    w = _load_resident(loc)
    q = index.config.get("quantization")
    if q and loc.st_dtype in ("U32", "I32"):
        scales = _load_resident(index.tensors[f"{base}.scales"])
        biases = _load_resident(index.tensors[f"{base}.biases"])
        w = mx.dequantize(w, scales, biases, group_size=q["group_size"],
                          bits=q["bits"]).astype(mx.bfloat16)
        mx.eval(w)
    return w


# ---------------------------------------------------------------- budget

@dataclass
class BudgetMath:
    working_set: int
    margin_bytes: int
    ring_bytes: int
    resident_bytes: int
    activation_bytes: int
    kv_budget: int
    per_seq_kv: int
    batch: int
    reason: str


def compute_batch(index: SafetensorsIndex, budget: MemoryBudget,
                  seq_costs: list[int], max_tokens: int, kv_per_token: int,
                  n_ring: int = 3) -> BudgetMath:
    """Batch = (working set - margin - ring - resident - activations) / per-seq KV,
    with per-seq KV from the ACTUAL token lengths of the input file."""
    ws = budget.working_set_bytes
    margin = int(ws * budget.margin)
    ring = n_ring * index.max_layer_bytes
    resident = index.embed.nbytes + index.final_norm.nbytes + (
        0 if index.tied else index.lm_head.nbytes)
    avail0 = ws - margin - ring - resident
    hidden = index.config["hidden_size"]
    mean_cost = sum(seq_costs) / max(1, len(seq_costs))

    # activations scale with batch; fixed-point in two passes
    batch = max(1, int(avail0 / max(1, mean_cost)))
    for _ in range(3):
        act = int(batch * (max(seq_costs) / max(1, kv_per_token)) * hidden * 2 * 3)
        kv_budget = avail0 - act
        nb = max(1, int(kv_budget / max(1, mean_cost)))
        if nb == batch:
            break
        batch = nb
    batch = min(batch, max(1, len(seq_costs)), 512)
    if budget.batch_override:
        batch = budget.batch_override
    act = int(batch * (max(seq_costs) / max(1, kv_per_token)) * hidden * 2 * 3)
    kv_budget = avail0 - act
    reason = (f"batch {batch}: ({ws / GIB:.1f}G working set - {margin / GIB:.1f}G margin(15%) "
              f"- {ring / GIB:.1f}G ring({n_ring}x{index.max_layer_bytes / GIB:.2f}G layer) "
              f"- {resident / GIB:.1f}G resident(embed+norm+lm_head) "
              f"- {act / GIB:.1f}G activations) / {mean_cost / GIB:.3f}G per-seq KV "
              f"(actual prompt lens + {max_tokens} max_tokens)")
    return BudgetMath(ws, margin, ring, resident, act, kv_budget,
                      int(mean_cost), batch, reason)


# ---------------------------------------------------------------- engine

class ResidentProvider:
    """Serves pre-built weight trees from memory; same loop, no ring, no disk."""

    def __init__(self, index: SafetensorsIndex):
        from mlx.utils import tree_unflatten
        self.trees = {}
        for plan in index.layers:
            pairs = []
            for t in plan.tensors:
                pairs.append((t.name.removeprefix(f"model.layers.{plan.layer_id}."),
                              _load_resident(t)))
            self.trees[plan.layer_id] = tree_unflatten(pairs)
        mx.eval(self.trees)


class MlxStreamEngine:
    name = "mlx_stream"

    def __init__(self, progress_note=None, resident=False):
        self.note = progress_note or (lambda s: None)
        self.resident = resident
        self.last_pass_times: list[float] = []

    def run_batch(self, rows: list[dict], spec: ModelSpec,
                  budget: MemoryBudget) -> Iterator[CompletedRow]:
        from mlx_lm.utils import load_tokenizer

        index = SafetensorsIndex(spec.path)
        cal = load_calibration()
        if "isolated_read_mbps" not in cal:
            self.note("calibrating read rate (no compute)...")
            cal = calibrate(index)
        chunk_mb, n_threads = cal.get("chunk_mb", 16), cal.get("threads", 4)

        tokenizer = load_tokenizer(spec.path)
        eos_ids = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])
        block, args = _model_modules(index.config)
        kv_per_token = (2 * index.config["num_hidden_layers"]
                        * index.config["num_key_value_heads"]
                        * (index.config["hidden_size"] // index.config["num_attention_heads"])
                        * 2)

        prompts = []
        for r in rows:
            toks = tokenizer.apply_chat_template(
                r["body"]["messages"], add_generation_prompt=True)
            prompts.append((r, toks, r["body"].get("max_tokens", 128)))
        # sort by length so chunks pad little; per-seq KV from actual lengths
        prompts.sort(key=lambda p: len(p[1]))
        max_tokens_job = max(p[2] for p in prompts)
        seq_costs = [(len(t) + mt) * kv_per_token for _, t, mt in prompts]
        bm = compute_batch(index, budget, seq_costs, max_tokens_job, kv_per_token)
        self.note(bm.reason)

        provider = ResidentProvider(index) if self.resident else None

        def bind_layer(k, ring, seqno):
            """Bind layer k's weights to the block; from memory or from the ring."""
            if provider is not None:
                block.update(provider.trees[k])
                return None
            slot, buf = ring.get(seqno)
            bind_costs.append(_bind(block, index.layers[k], buf, f"model.layers.{k}."))
            return slot

        embed_w = _load_resident_maybe_quantized(index, "model.embed_tokens")
        norm_w = _load_resident(index.final_norm).astype(mx.bfloat16)
        lm_w = embed_w if index.tied else _load_resident_maybe_quantized(index, "lm_head")
        eps = index.config.get("rms_norm_eps", 1e-5)

        stop_event = spec.extra.get("stop_event")
        bind_costs = []
        n_layers = index.n_layers

        for c0 in range(0, len(prompts), bm.batch):
            chunk = prompts[c0:c0 + bm.batch]
            if stop_event and stop_event.is_set():
                return
            B = len(chunk)
            lens = [len(t) for _, t, _ in chunk]
            Lpad = max(lens)
            deltas = [Lpad - l for l in lens]
            t_chunk0 = time.monotonic()

            # ---- prefill: per-seq causal (exact; uniform rope shift per row) ----
            caches = [StreamKVCache() for _ in range(n_layers)]
            # per-seq hidden states during prefill
            hs = [embed_w[mx.array(t)][None] for _, t, _ in chunk]
            sched = list(range(n_layers))
            ring = None
            if provider is None:
                ring = RingReader(index, 3, chunk_mb * MB, n_threads)
                ring.start(sched)
            prefill_t0 = time.monotonic()
            seq_caches = [[StreamKVCache(offset=deltas[i]) for i in range(B)]
                          for _ in range(n_layers)]
            for seq, k in enumerate(sched):
                if stop_event and stop_event.is_set():
                    if ring:
                        ring.stop()
                    return
                slot = bind_layer(k, ring, seq)
                for i in range(B):
                    hs[i] = block(hs[i], mask="causal", cache=seq_caches[k][i])
                mx.eval(*[h for h in hs])
                if slot is not None:
                    ring.release(slot)
            if ring:
                ring.stop()
            prefill_s = time.monotonic() - prefill_t0

            # merge per-seq caches into batched caches with physical left-pad
            H = index.config["num_key_value_heads"]
            D = index.config["hidden_size"] // index.config["num_attention_heads"]
            for k in range(n_layers):
                ks, vs = [], []
                for i in range(B):
                    sk = seq_caches[k][i]
                    kk = sk.keys[..., :sk._len, :]
                    vv = sk.values[..., :sk._len, :]
                    if deltas[i]:
                        pad = mx.zeros((1, H, deltas[i], D), kk.dtype)
                        kk = mx.concatenate([pad, kk], axis=2)
                        vv = mx.concatenate([pad, vv], axis=2)
                    ks.append(kk)
                    vs.append(vv)
                c = caches[k]
                c.keys = mx.concatenate(ks, axis=0)
                c.values = mx.concatenate(vs, axis=0)
                c._used = Lpad
                c.offset = Lpad
                mx.eval(c.keys, c.values)
            del seq_caches
            mx.clear_cache()

            # last real token logits -> first generated token
            last_h = mx.concatenate([h[:, -1:, :] for h in hs], axis=0)
            del hs
            logits = mx.fast.rms_norm(last_h, norm_w, eps) @ lm_w.T
            tokens = mx.argmax(logits, axis=-1)  # [B,1]
            mx.eval(tokens)

            generated = [[int(tokens[i, 0])] for i in range(B)]
            active = list(range(B))
            delta_vec = list(deltas)
            maxtoks = [mt for _, _, mt in chunk]
            finished = [False] * B

            def finish(i_global, reason):
                r, toks, _ = chunk[i_global]
                out = generated[i_global]
                if out and out[-1] in eos_ids:
                    out = out[:-1]
                yield_row = CompletedRow(
                    custom_id=r["custom_id"],
                    content=tokenizer.decode(out),
                    prompt_tokens=len(toks),
                    completion_tokens=len(generated[i_global]),
                    latency_s=round(time.monotonic() - t_chunk0, 3),
                    finish_reason=reason, batch_size=B)
                return yield_row

            # immediate EOS check
            for i in list(active):
                if generated[i][-1] in eos_ids or len(generated[i]) >= maxtoks[i]:
                    finished[i] = True

            # ---- decode: batched, one weight stream per token ----
            decode_t0 = time.monotonic()
            n_decode_passes = 0
            ring = None
            if provider is None:
                ring = RingReader(index, 3, chunk_mb * MB, n_threads)
                # schedule enough passes for the worst case; stop early via ring.stop()
                ring.start([k for _ in range(max(maxtoks)) for k in range(n_layers)])
            seqno = 0
            try:
                while active:
                    # emit finished rows, drop them from the batch at this pass boundary
                    if any(finished[i] for i in active):
                        keep_pos = [p for p, i in enumerate(active) if not finished[i]]
                        for i in list(active):
                            if finished[i]:
                                yield finish(i, "stop" if generated[i][-1] in eos_ids
                                             else "length")
                        active = [i for i in active if not finished[i]]
                        if not active:
                            break
                        for c in caches:
                            c.keep_rows(keep_pos)
                        tokens = tokens[mx.array(keep_pos)]
                        delta_vec = [delta_vec[p] for p in keep_pos]
                        mx.eval(tokens)
                    if stop_event and stop_event.is_set():
                        return
                    pass_t0 = time.monotonic()
                    x = embed_w[tokens]          # [b,1,D]
                    K = caches[0].offset + 1
                    pad_mask = (mx.arange(K)[None, None, None, :]
                                < mx.array(delta_vec)[:, None, None, None])
                    mask = mx.where(pad_mask, mx.array(-mx.inf, x.dtype),
                                    mx.array(0, x.dtype))
                    for k in range(n_layers):
                        slot = bind_layer(k, ring, seqno)
                        seqno += 1
                        x = block(x, mask=mask, cache=caches[k])
                        mx.eval(x)
                        if slot is not None:
                            ring.release(slot)
                    logits = mx.fast.rms_norm(x, norm_w, eps) @ lm_w.T
                    tokens = mx.argmax(logits, axis=-1)
                    mx.eval(tokens)
                    self.last_pass_times.append(time.monotonic() - pass_t0)
                    n_decode_passes += 1
                    self._gen_tokens = getattr(self, "_gen_tokens", 0) + len(active)
                    if n_decode_passes % 10 == 0:
                        self.note(f"gen progress: passes {n_decode_passes}, "
                                  f"active {len(active)}, gen_tokens {self._gen_tokens}, "
                                  f"median pass {float(np.median(self.last_pass_times)):.1f}s, "
                                  f"peak mem {mx.get_peak_memory() / GIB:.1f}G")
                    for pos, i in enumerate(active):
                        generated[i].append(int(tokens[pos, 0]))
                        if generated[i][-1] in eos_ids or len(generated[i]) >= maxtoks[i]:
                            finished[i] = True
            finally:
                if ring:
                    ring.stop()
                mx.clear_cache()

            self.note(f"chunk done: prefill {prefill_s:.1f}s, "
                      f"{n_decode_passes} decode passes, "
                      f"median pass {np.median(self.last_pass_times or [0]):.1f}s, "
                      f"bind cost median {np.median(bind_costs or [0]) * 1000:.0f}ms/layer, "
                      f"peak mem {mx.get_peak_memory() / GIB:.1f}G")

        # refresh calibration with measured pass times (policy rule source)
        model_bytes_total = sum(p.nbytes for p in index.layers)
        # only streaming-scale models produce a trustworthy engine rate; small
        # models sit in page cache and would pollute the calibration
        if self.last_pass_times and provider is None and model_bytes_total > 10 * GIB:
            cal = load_calibration()
            med = float(np.median(self.last_pass_times))
            cal["engine_read_mbps"] = round(model_bytes_total / med / MB, 1)
            cal["measured_pass_s"] = round(med, 2)
            cal["bind_ms_per_layer"] = round(float(np.median(bind_costs)) * 1000, 1)
            save_calibration(cal)
