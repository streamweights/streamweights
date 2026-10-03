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

from .. import logits as lg
from ..errors import SpillError
from ..registry import GIB, REPO_ROOT
from .base import CompletedRow, MemoryBudget, ModelSpec

CALIBRATION_JSON = REPO_ROOT / "state" / "calibration.json"

# prefill is processed alongside decode, at most this many prompt tokens per
# pass, so a wave of newcomers never turns one pass into a 28-minute wall
PREFILL_TOKENS_PER_PASS = 2048

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

def collect_eos_ids(model_dir: Path, tokenizer) -> set[int]:
    """Every EOS id the model declares: generation_config.json and config.json
    (int or list), the tokenizer's own ids, and the chat template's end-of-turn
    token (for Llama 3.x that includes 128009)."""
    ids: set[int] = set()
    for fname in ("generation_config.json", "config.json"):
        p = Path(model_dir) / fname
        if p.exists():
            v = json.loads(p.read_text()).get("eos_token_id")
            if isinstance(v, int):
                ids.add(v)
            elif isinstance(v, list):
                ids.update(int(x) for x in v)
    if getattr(tokenizer, "eos_token_ids", None):
        ids.update(tokenizer.eos_token_ids)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(tokenizer.eos_token_id)
    # chat template end-of-turn token, resolved through the vocabulary
    for tok in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                ids.add(tid)
        except Exception:
            pass
    return ids


def _model_modules(config: dict):
    """One reusable TransformerBlock + args + family for the architecture."""
    import importlib

    from .supported import NOT_YET, classify
    fam = classify(config)
    if fam.state == NOT_YET or not fam.module:
        raise ValueError(
            f"unsupported architecture: {fam.label} is '{fam.state}' "
            f"({fam.note}) — see: spill models --architectures")
    mod = importlib.import_module(f"mlx_lm.models.{fam.module}")
    args = mod.ModelArgs.from_dict(config)
    block = mod.TransformerBlock(args)
    q = config.get("quantization")
    if q:
        import mlx.nn as nn
        nn.quantize(block, group_size=q["group_size"], bits=q["bits"])
    return block, args, fam


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

    def compact(self, n: int):
        """Drop n dead left-pad columns. Rotary positions live in the stored
        K values, so physical shifting is exact; offset stays absolute."""
        if self.keys is not None and n > 0:
            self.keys = self.keys[..., n:, :]
            self.values = self.values[..., n:, :]
            self._used -= n


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
                  n_ring: int = 3, calibration: dict | None = None,
                  quant: str | None = None) -> BudgetMath:
    """Batch sizing. Preferred: measured-memory mode — peak(B) calibrated from
    probe runs at two batch sizes (state/calibration.json mem_model), solved for
    85% of the Metal working set. Fallback: the Phase 1 analytic formula."""
    ws = budget.working_set_bytes
    mem = (calibration or {}).get("mem_model", {}).get(quant or "")  # quant is "model|quant" composite when measured
    if mem:
        target = int(ws * 0.75)
        mean_cost_tokens = sum(seq_costs) / max(1, len(seq_costs)) / max(1, kv_per_token)
        per_seq_mean = mem["per_seq_token_bytes"] * mean_cost_tokens
        batch = max(1, int((target - mem["base_bytes"]) / max(1, per_seq_mean)))
        batch = min(batch, max(1, len(seq_costs)), 512)
        if budget.batch_override:
            batch = budget.batch_override
        reason = (f"batch ~{batch} (admission by memory, not count): "
                  f"(0.75×{ws / GIB:.1f}G target − {mem['base_bytes'] / GIB:.1f}G measured base) "
                  f"÷ ({mem['per_seq_token_bytes'] / 1024:.0f} KB/seq-token × "
                  f"{mean_cost_tokens:.0f} mean tokens (actual prompt lens + {max_tokens} "
                  f"max_tokens)); rows admitted while calibrated cost fits; "
                  f"probes at B={mem['probe_batches']}")
        return BudgetMath(ws, int(ws * 0.15), 0, 0, 0, target - mem["base_bytes"],
                          int(per_seq_mean), batch, reason)
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

    def __init__(self, progress_note=None, resident=False, pass_cb=None):
        self.note = progress_note or (lambda s: None)
        self.resident = resident
        self.pass_cb = pass_cb
        self.last_pass_times: list[float] = []
        self._bind_costs: list[float] = []

    def run_batch(self, rows: list[dict], spec: ModelSpec,
                  budget: MemoryBudget):
        from itertools import cycle

        from mlx_lm.utils import load_tokenizer

        index = SafetensorsIndex(spec.path)
        cal = load_calibration()
        if "isolated_read_mbps" not in cal:
            self.note("calibrating read rate (no compute)...")
            cal = calibrate(index)
        chunk_mb, n_threads = cal.get("chunk_mb", 16), cal.get("threads", 4)

        tokenizer = load_tokenizer(spec.path)
        eos_ids = collect_eos_ids(spec.path, tokenizer)
        block, args, fam = _model_modules(index.config)
        cfg = index.config
        D = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        kv_per_token = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * D * 2
        n_layers = index.n_layers
        H = cfg["num_key_value_heads"]
        eps = cfg.get("rms_norm_eps", 1e-5)

        K_lp = lg.validate_k(spec.extra.get("logprobs"))
        full_dir = spec.extra.get("full_logits_dir")
        scoring = spec.extra.get("mode") == "score"
        adapter = spec.extra.get("adapter")   # adapters.LoraAdapter or None
        score_info: dict[str, tuple[int, list[int]]] = {}
        prompts = []
        for r in rows:
            if scoring:
                from ..formats import tokenize_scored
                n_prompt, full_ids = tokenize_scored(tokenizer, r["body"]["messages"], eos_ids)
                score_info[r["custom_id"]] = (n_prompt, full_ids)
                prompts.append((r, full_ids, 0))
                continue
            toks = tokenizer.apply_chat_template(
                r["body"]["messages"], add_generation_prompt=True)
            prompts.append((r, toks, r["body"].get("max_tokens", 128)))
        prompts.sort(key=lambda p: -len(p[1]))  # longest first: the cache
        # length is set by the head of the queue, every later row admits
        # freely, and expensive rows are spread through the run, not walled
        total_rows = len(prompts)
        max_tokens_job = max([p[2] for p in prompts] or [0])
        seq_costs = [(len(t) + mt) * kv_per_token for _, t, mt in prompts]
        bm = compute_batch(index, budget, seq_costs, max_tokens_job, kv_per_token,
                           calibration=cal, quant=f"{spec.name}|{spec.quant}")
        auto_batch = bm.batch

        provider = ResidentProvider(index) if self.resident else None

        if adapter is not None:
            adapter.prepare(block)   # one-time: LoRA-aware linears, passthrough until attached

        def bind_layer(k, ring, seqno):
            if provider is not None:
                block.update(provider.trees[k])
                slot = None
            else:
                slot, buf = ring.get(seqno)
                self._bind_costs.append(
                    _bind(block, index.layers[k], buf, f"model.layers.{k}."))
            if adapter is not None:
                adapter.attach(block, k)   # resident LoRA deltas, after the base is bound
            return slot

        self._bind_costs = []
        embed_w = _load_resident_maybe_quantized(index, "model.embed_tokens")
        norm_w = _load_resident(index.final_norm).astype(mx.bfloat16)
        if fam.norm_plus_one:
            norm_w = norm_w + 1.0
        lm_w = embed_w if index.tied else _load_resident_maybe_quantized(index, "lm_head")
        embed_scale = (index.config["hidden_size"] ** 0.5) if fam.embed_scale else None
        softcap = (index.config.get(fam.final_softcap_key)
                   if fam.final_softcap_key else None)

        def embed(ids):
            h = embed_w[ids]
            return (h * embed_scale).astype(h.dtype) if embed_scale else h

        def lm_logits(h):
            logits = mx.fast.rms_norm(h, norm_w, eps) @ lm_w.T
            if softcap:
                logits = mx.tanh(logits / softcap) * softcap
            return logits

        vocab = int(lm_w.shape[0])
        if full_dir:
            lg.check_full_logits(total_rows, max_tokens_job, vocab)
            Path(full_dir).mkdir(parents=True, exist_ok=True)
        lp_cap = None
        if K_lp or full_dir:
            # batch x vocab x 2 bytes per step (plus log-prob temporaries) must fit
            # in 5% of the working set, on top of the weights/KV budget
            lp_cap = lg.logits_batch_cap(vocab, int(0.05 * budget.working_set_bytes), K_lp)
            auto_batch = min(auto_batch, lp_cap)
        stop_event = spec.extra.get("stop_event")
        max_passes = spec.extra.get("max_passes")  # probe/timebox hook

        pending = list(prompts)          # sorted ascending by prompt length
        mem = cal.get("mem_model", {}).get(f"{spec.name}|{spec.quant}")
        admit_budget = committed = None
        per_tok_cost = None
        if mem and not budget.batch_override:
            admit_budget = int(budget.working_set_bytes * 0.75) - mem["base_bytes"]
            if K_lp or full_dir:
                admit_budget -= lg.step_bytes(min(auto_batch, 512), int(
                    index.config.get("vocab_size", 0)))
            per_tok_cost = mem["per_seq_token_bytes"]
            committed = 0

        def row_cost(p):
            return per_tok_cost * (len(p[1]) + p[2]) if per_tok_cost else 0

        if committed is not None and pending:
            # blunt upfront clamp: the worst pending row's full-horizon physical
            # tier must fit the whole batch (incremental top-ups cannot exceed it)
            worst = max(len(p[1]) + p[2] for p in pending)
            tier = ((worst + 255) // 256) * 256
            phys_cap = max(1, int(admit_budget // (tier * kv_per_token)) - 1)
            auto_batch = min(auto_batch, phys_cap)
        self.note(f"admission: mode={'measured' if committed is not None else 'count'}, "
                  f"auto_batch={auto_batch}")

        def may_admit(p, n_active, K_now, rem_max, pending_cost=0):
            if committed is None:
                return n_active < auto_batch
            if n_active == 0:
                return True
            if n_active >= 512 or (lp_cap and n_active >= lp_cap):
                return False
            # (1) calibrated per-row cost must fit the budget
            if committed + pending_cost + row_cost(p) > admit_budget:
                return False
            # (2) physical allocation: every row's cache occupies the batch's
            # padded length, allocated in 256-token steps — the term the
            # per-row model misses (this is what let 512 short rows in)
            horizon = K_now + max(rem_max, p[2])
            phys_len = ((horizon + 255) // 256) * 256
            phys = (n_active + 1) * phys_len * kv_per_token
            return phys <= admit_budget
        # active-row state (parallel lists)
        act_rows: list = []              # (row, toks, max_tokens)
        act_start: list[int] = []        # first valid cache column per row
        act_gen: list[list[int]] = []    # generated ids per row
        act_lp: list[list] = []          # per-token (id, logprob, top ids, top logprobs)
        act_full: list[list] = []        # per-token float16 logits (--full-logits)
        act_t0: list[float] = []         # when each row entered a pass
        caches = [StreamKVCache() for _ in range(n_layers)]
        tokens = None                    # [b, 1] next input ids
        completed = 0
        gen_tokens_total = 0
        t_job0 = time.monotonic()
        pass_no = 0

        ring = None
        if provider is None:
            ring = RingReader(index, 3, chunk_mb * MB, n_threads)
            ring.start(cycle(range(n_layers)))
        seqno = 0

        def emit(i):
            r, toks, mt = act_rows[i]
            out = act_gen[i]
            stopped = bool(out) and out[-1] in eos_ids
            text_ids = out[:-1] if stopped else out
            recs = None
            if K_lp and act_lp[i]:
                ids, tls, tis, tvs = zip(*act_lp[i])
                recs = lg.records_from_arrays(ids, tls, tis, tvs)
            if full_dir and act_full[i]:
                np.save(Path(full_dir) / f"{lg.safe_name(r['custom_id'])}.npy",
                        np.stack(act_full[i]))
            return CompletedRow(
                custom_id=r["custom_id"], content=tokenizer.decode(text_ids),
                prompt_tokens=len(toks), completion_tokens=len(out),
                latency_s=round(time.monotonic() - act_t0[i], 3),
                finish_reason="stop" if stopped else "length",
                batch_size=len(act_rows), logprobs=recs)

        def score_groups():
            """Teacher-forced scoring: prefill only, no sampling. Sequences are
            grouped by token budget; one weight pass (one trip through every
            layer) serves the whole group. For each row, the hidden states at
            the target positions go through the final norm and lm_head, and the
            teacher's top-k plus the target token's log-prob are recorded."""
            nonlocal seqno
            k_top = K_lp or 32
            max_group_tokens = spec.extra.get("score_group_tokens", 16384)
            max_group_rows = spec.extra.get("score_group_rows", 256)
            todo = list(prompts)           # ascending by length
            done_rows = scored = 0
            t_start = time.monotonic()
            pass_i = 0
            while todo:
                if stop_event and stop_event.is_set():
                    return
                group, tok_sum = [], 0
                while todo and len(group) < max_group_rows and (
                        not group or tok_sum + len(todo[0][1]) <= max_group_tokens):
                    group.append(todo.pop(0))
                    tok_sum += len(group[-1][1])
                t0 = time.monotonic()
                hs = [embed(mx.array(t))[None] for _, t, _ in group]
                for k in range(n_layers):
                    slot = bind_layer(k, ring, seqno)
                    seqno += 1
                    for a in range(len(group)):
                        la = hs[a].shape[1]
                        amask = "causal" if not fam.needs_array_mask else \
                            mx.where(mx.arange(la)[:, None] >= mx.arange(la)[None, :],
                                     mx.array(0, hs[a].dtype), mx.array(-mx.inf, hs[a].dtype))
                        hs[a] = block(hs[a], mask=amask, cache=StreamKVCache())
                        mx.eval(hs[a])   # one sequence's intermediates live at a time
                    if slot is not None:
                        ring.release(slot)
                out_rows = []
                for a, (r, toks, _) in enumerate(group):
                    n_prompt, full_ids = score_info[r["custom_id"]]
                    tgt = mx.array(full_ids[n_prompt:])
                    h = hs[a][0, n_prompt - 1:len(full_ids) - 1, :]
                    recs, lps, full_chunks = [], [], []
                    for c0 in range(0, h.shape[0], 256):   # bound logits to 256 x vocab
                        lg_c = lm_logits(h[c0:c0 + 256])
                        tl, ti, tv = lg.topk_mlx(lg_c, tgt[c0:c0 + 256], k_top)
                        recs += lg.records_from_arrays(
                            full_ids[n_prompt + c0:n_prompt + c0 + len(tl)], tl, ti, tv)
                        lps += [float(x) for x in tl]
                        if full_dir:
                            full_chunks.append(np.array(lg_c.astype(mx.float16)))
                    if full_dir:
                        np.save(Path(full_dir) / f"{lg.safe_name(r['custom_id'])}.npy",
                                np.concatenate(full_chunks))
                    tgt_ids = full_ids[n_prompt:]
                    stopped = bool(tgt_ids) and tgt_ids[-1] in eos_ids
                    out_rows.append(CompletedRow(
                        custom_id=r["custom_id"],
                        content=tokenizer.decode(tgt_ids[:-1] if stopped else tgt_ids),
                        prompt_tokens=n_prompt, completion_tokens=len(tgt_ids),
                        latency_s=0.0, finish_reason="scored", batch_size=len(group),
                        logprobs=recs,
                        extra={"score": {
                            "mode": "teacher_forced", "n_target": len(tgt_ids),
                            "logprob_sum": round(sum(lps), 6),
                            "logprob_mean": round(sum(lps) / len(lps), 6),
                            "perplexity": round(float(np.exp(-sum(lps) / len(lps))), 6)}}))
                mx.clear_cache()
                pass_s = time.monotonic() - t0
                pass_i += 1
                self.last_pass_times.append(pass_s)
                scored += sum(len(score_info[r["custom_id"]][1]) for r, _, _ in group)
                for cr in out_rows:
                    cr.latency_s = round(pass_s, 3)
                    yield cr
                done_rows += len(group)
                if self.pass_cb:
                    el = time.monotonic() - t_start
                    rate = scored / el if el else 0
                    left = sum(len(t) for _, t, _ in todo)
                    self.pass_cb({
                        "rows_done": done_rows, "total": total_rows, "pass_no": pass_i,
                        "pass_s": pass_s, "tok_s": rate,
                        "eta_s": left / rate if rate else None,
                        "quant": spec.quant, "batch": len(group),
                        "peak_gb": (mx.get_peak_memory() / GIB)
                        if mx.default_device() == mx.gpu else 0.0,
                        "slots": [{"custom_id": r["custom_id"],
                                   "tokens": len(score_info[r["custom_id"]][1])
                                   - score_info[r["custom_id"]][0],
                                   "tail": tokenizer.decode(
                                       score_info[r["custom_id"]][1][-30:])[-80:]
                                   .replace("\n", " ")} for r, _, _ in group[:8]]})
            el = time.monotonic() - t_start
            if scored and el > 0:     # measured prefill rate, the basis for --score estimates
                cal2 = load_calibration()
                cal2.setdefault("prefill_rates", {})[f"{spec.name}|{spec.quant}"] = {
                    "tok_s": round(scored / el, 1), "tokens": scored,
                    "rows": done_rows, "engine": "resident" if provider else "stream",
                    "model_bytes": sum(p.nbytes for p in index.layers),
                    "adapter": bool(adapter)}
                save_calibration(cal2)

        try:
            if scoring:
                yield from score_groups()
                return
            while act_rows or pending:
                if stop_event and stop_event.is_set():
                    return
                if max_passes and pass_no >= max_passes:
                    return

                K = caches[0]._len    # physical cache length == next write slot
                abs_off = caches[0].offset  # absolute rope position of next slot
                # direct backpressure guard: no admission while live Metal
                # allocation exceeds 85% of the working set (the 75% steady-state
                # budget is the primary control; this catches modeling errors)
                mem_pressure = (mx.default_device() == mx.gpu and
                                mx.get_active_memory() > 0.85 * budget.working_set_bytes)
                # ---- admission ----
                admits = []
                cap = auto_batch
                if not act_rows and pending:
                    if committed is not None:
                        if mem_pressure:
                            mx.clear_cache()  # return pooled memory before refilling
                            group = pending[:1]
                        else:
                            group = []
                            gcost = 0
                            ptoks = 0
                            Lp = 0
                            for p in pending:
                                Lp = max(Lp, len(p[1]))
                                if (len(group) >= cap
                                        or ptoks + len(p[1]) > PREFILL_TOKENS_PER_PASS
                                        or not may_admit(p, len(group), Lp, p[2],
                                                         pending_cost=gcost)):
                                    break
                                group.append(p)
                                gcost += row_cost(p)
                                ptoks += len(p[1])
                            group = group or pending[:1]
                    else:
                        group = pending[:cap]
                    pending = pending[len(group):]
                    Lpad = max(len(t) for _, t, _ in group)
                    # fresh group: prompt ends align at slot Lpad-1; next write = Lpad
                    admits = [(p, Lpad - len(p[1])) for p in group]
                    caches = [StreamKVCache() for _ in range(n_layers)]
                    K = Lpad - 1   # so that s = K+1-L matches Lpad-L below
                    abs_off = Lpad - 1
                else:
                    rem_max = max((act_rows[i][2] - len(act_gen[i])
                                   for i in range(len(act_rows))), default=0)
                    acost = 0
                    ptoks = 0
                    while pending and not mem_pressure and may_admit(pending[0],
                                                len(act_rows) + len(admits), K,
                                                rem_max, pending_cost=acost):
                        cand = pending[0]
                        if ptoks + len(cand[1]) > PREFILL_TOKENS_PER_PASS:
                            break
                        if len(cand[1]) <= K + 1:
                            admits.append((cand, K + 1 - len(cand[1])))
                            acost += row_cost(cand)
                            ptoks += len(cand[1])
                            pending = pending[1:]
                        else:
                            break

                pass_t0 = time.monotonic()
                admit_h = [embed(mx.array(t))[None] for (_, t, _), _ in admits]
                # physical pad start s vs absolute rope start: differ after compaction
                admit_caches = [[StreamKVCache(offset=abs_off + 1 - len(p[1]))
                                 for p, _ in admits] for _ in range(n_layers)]
                x = embed(tokens) if act_rows else None   # [b,1,D]
                if act_rows:
                    Kcur = K + 1  # physical columns
                    pad_mask = (mx.arange(Kcur)[None, None, None, :]
                                < mx.array(act_start)[:, None, None, None])
                    mask = mx.where(pad_mask, mx.array(-mx.inf, x.dtype),
                                    mx.array(0, x.dtype))
                    if fam.needs_array_mask:
                        # eager GQA attention scores are 5-D [B, Hkv, rep, L, K]
                        mask = mask[:, :, None, :, :]

                for k in range(n_layers):
                    slot = bind_layer(k, ring, seqno)
                    seqno += 1
                    if act_rows:
                        x = block(x, mask=mask, cache=caches[k])
                    for a in range(len(admits)):
                        la = admit_h[a].shape[1]
                        amask = "causal" if not fam.needs_array_mask else \
                            mx.where(mx.arange(la)[:, None] >= mx.arange(la)[None, :],
                                     mx.array(0, admit_h[a].dtype),
                                     mx.array(-mx.inf, admit_h[a].dtype))
                        admit_h[a] = block(admit_h[a], mask=amask,
                                           cache=admit_caches[k][a])
                    ev = ([x] if act_rows else []) + admit_h
                    if ev:
                        mx.eval(*ev)
                    if slot is not None:
                        ring.release(slot)

                # merge admits into the batch (prompt end aligned at slot K)
                if admits:
                    new_len = caches[0]._len if act_rows else (K + 1)
                    for k in range(n_layers):
                        c = caches[k]
                        ks, vs = ([c.keys[..., :c._len, :]] if act_rows else []), \
                                 ([c.values[..., :c._len, :]] if act_rows else [])
                        for a in range(len(admits)):
                            sk = admit_caches[k][a]
                            kk = sk.keys[..., :sk._len, :]
                            vv = sk.values[..., :sk._len, :]
                            padlen = new_len - kk.shape[2]
                            if padlen > 0:
                                z = mx.zeros((1, H, padlen, D), kk.dtype)
                                kk = mx.concatenate([z, kk], axis=2)
                                vv = mx.concatenate([z, vv], axis=2)
                            ks.append(kk)
                            vs.append(vv)
                        c.keys = mx.concatenate(ks, axis=0)
                        c.values = mx.concatenate(vs, axis=0)
                        c._used = new_len
                        c.offset = (abs_off + 1) if act_rows else new_len
                        mx.eval(c.keys, c.values)
                        for a in range(len(admits)):
                            admit_caches[k][a] = None
                    for (p, s) in admits:
                        act_rows.append(p)
                        act_start.append(s)
                        act_gen.append([])
                        act_lp.append([])
                        act_full.append([])
                        act_t0.append(pass_t0)
                        if committed is not None:
                            committed += row_cost(p)
                    mx.clear_cache()

                # logits for this pass: actives' decode step + admits' first token
                hs_last = []
                if x is not None:
                    hs_last.append(x)
                hs_last += [h[:, -1:, :] for h in admit_h]
                if not hs_last:
                    break
                h_all = mx.concatenate(hs_last, axis=0)
                logits = lm_logits(h_all)
                tokens = mx.argmax(logits, axis=-1)
                mx.eval(tokens)
                step_lp = step_full = None
                if K_lp:
                    step_lp = lg.topk_mlx(logits[:, 0, :], tokens[:, 0], K_lp)
                if full_dir:
                    step_full = np.array(logits[:, 0, :].astype(mx.float16))
                pass_s = time.monotonic() - pass_t0
                self.last_pass_times.append(pass_s)
                pass_no += 1

                for i in range(len(act_rows)):
                    act_gen[i].append(int(tokens[i, 0]))
                    if step_lp is not None:
                        tl, ti, tv = step_lp
                        act_lp[i].append((int(tokens[i, 0]), tl[i], ti[i], tv[i]))
                    if step_full is not None:
                        act_full[i].append(step_full[i])
                gen_tokens_total += len(act_rows)

                # finish + compress
                done_idx = [i for i in range(len(act_rows))
                            if act_gen[i][-1] in eos_ids
                            or len(act_gen[i]) >= act_rows[i][2]]
                if done_idx:
                    for i in done_idx:
                        yield emit(i)
                        if committed is not None:
                            committed -= row_cost(act_rows[i])
                    completed += len(done_idx)
                    keep = [i for i in range(len(act_rows)) if i not in set(done_idx)]
                    if keep:
                        kidx = mx.array(keep)
                        tokens = tokens[kidx]
                        for c in caches:
                            c.keep_rows(keep)
                        mx.eval(tokens)
                    else:
                        tokens = None
                    act_rows = [act_rows[i] for i in keep]
                    act_start = [act_start[i] for i in keep]
                    act_gen = [act_gen[i] for i in keep]
                    act_lp = [act_lp[i] for i in keep]
                    act_full = [act_full[i] for i in keep]
                    act_t0 = [act_t0[i] for i in keep]
                    if not act_rows:
                        caches = [StreamKVCache() for _ in range(n_layers)]
                        mx.clear_cache()
                    else:
                        trim = min(act_start)
                        if trim >= 256:  # dead left-pad columns: compact physically
                            for c in caches:
                                c.compact(trim)
                            act_start = [st - trim for st in act_start]
                            mx.clear_cache()

                if self.pass_cb:
                    elapsed = time.monotonic() - t_job0
                    rate = gen_tokens_total / elapsed if elapsed else 0
                    done_avg = (gen_tokens_total / max(1, completed)
                                if completed else max_tokens_job)
                    remaining = (total_rows - completed) * min(done_avg, max_tokens_job)
                    slots = []
                    for i in range(min(8, len(act_rows))):
                        txt = tokenizer.decode(act_gen[i][-40:]) if act_gen[i] else ""
                        slots.append({"custom_id": act_rows[i][0]["custom_id"],
                                      "tokens": len(act_gen[i]),
                                      "tail": txt[-80:].replace("\n", " ")})
                    self.pass_cb({
                        "rows_done": completed, "total": total_rows,
                        "pass_no": pass_no, "pass_s": pass_s,
                        "tok_s": rate,
                        "eta_s": remaining / rate if rate else None,
                        "quant": spec.quant, "batch": len(act_rows),
                        "peak_gb": (mx.get_peak_memory() / GIB) if mx.default_device() == mx.gpu else 0.0,
                        "slots": slots,
                    })
        finally:
            if ring:
                ring.stop()
            mx.clear_cache()

        model_bytes_total = sum(p.nbytes for p in index.layers)
        if self.last_pass_times and provider is None and model_bytes_total > 10 * GIB:
            cal = load_calibration()
            med = float(np.median(self.last_pass_times))
            # keyed per model|quant: a smaller model's (often page-cache-warm)
            # rate must never masquerade as another model's streaming rate
            cal.setdefault("engine_rates", {})[f"{spec.name}|{spec.quant}"] = \
                round(model_bytes_total / med / MB, 1)
            cal["measured_pass_s"] = round(med, 2)
            cal["bind_ms_per_layer"] = round(float(np.median(self._bind_costs or [0])) * 1000, 1)
            save_calibration(cal)
