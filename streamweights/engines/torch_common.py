"""The PyTorch engines' shared parts: devices and dtypes, the tokenizer, KV caches, the model
core (Hugging Face transformers, built on the meta device, one reusable decoder layer whose
weights are bound from memory or from the streaming ring), and LoRA slots.

Everything an engine needs from transformers is here, so torch_resident.py and torch_stream.py
differ only in where a layer's weights come from. The model classes are transformers' own
(Llama, Qwen2, Qwen3, Mistral, Phi 3, Gemma 2); this module never reimplements attention, the
MLP, or the norms.
"""

from __future__ import annotations

import math
import os
import platform
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import SpillError
from ..ring import RingReader, SafetensorsIndex, TensorLoc

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:           # torch is a base dependency; guarded so import errors stay one line
    torch = None
    F = None
    HAVE_TORCH = False

GIB = 1024**3


def require_torch() -> None:
    if not HAVE_TORCH:
        raise SpillError("the torch engines need PyTorch, which is not importable here",
                         "pip install torch")


def _st_torch(name: str):
    return {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[name]


# ---------------------------------------------------------------- devices and dtypes

def cuda_available() -> bool:
    return HAVE_TORCH and torch.cuda.is_available()


_BF16_CPU: bool | None = None


def cpu_bf16_ok() -> bool:
    """True when this CPU multiplies bf16 matrices about as fast as float32 (hardware bf16:
    AVX512-BF16, AMX, ARM BF16). Measured, not guessed: on CPUs without it torch falls back to
    a path hundreds of times slower."""
    global _BF16_CPU
    if _BF16_CPU is None:
        require_torch()
        def rate(dt):
            # the op the layers run (a linear over a model-shaped weight): the 2-D matmul
            # path of a CPU without bf16 hardware is far slower than the linear path
            x = torch.randn(128, 512, dtype=dt)
            w = torch.randn(1024, 512, dtype=dt)
            for _ in range(2):
                F.linear(x, w)
            t0 = time.perf_counter()
            for _ in range(8):
                F.linear(x, w)
            return 8 / (time.perf_counter() - t0)
        _BF16_CPU = rate(torch.bfloat16) >= 0.6 * rate(torch.float32)
    return _BF16_CPU


def resolve_dtype(engine: str, requested: str | None = None) -> tuple["torch.dtype", str]:
    """(dtype for the base weights, the sentence the pre-run line shows). Adapters and the
    optimizer are float32 regardless. CUDA: bf16. CPU: bf16 where this CPU multiplies bf16
    quickly, otherwise float32."""
    require_torch()
    requested = requested or os.environ.get("SPILL_TORCH_DTYPE")
    if requested in ("float32", "f32", "fp32"):
        return torch.float32, "float32 base weights (requested)"
    if requested in ("bf16", "bfloat16"):
        return torch.bfloat16, "bf16 base weights (requested)"
    if engine == "torch-cuda":
        if not torch.cuda.is_bf16_supported():
            return torch.bfloat16, ("bf16 base weights on the GPU (this GPU has no native bf16; "
                                    "expect it to be slow)")
        return torch.bfloat16, "bf16 base weights on the GPU"
    if cpu_bf16_ok():
        return torch.bfloat16, "bf16 base weights (this CPU has fast bf16 matmul)"
    return torch.float32, "float32 base weights (this CPU has no fast bf16 matmul)"


def numerics_for(dtype, adapter: bool = True) -> dict:
    name = {torch.bfloat16: "bf16", torch.float32: "float32", torch.float16: "float16"}[dtype]
    return {"base": name, "adapter": "float32", "optimizer": "float32"}


def device_for(engine: str):
    require_torch()
    if engine == "torch-cuda":
        if not torch.cuda.is_available():
            raise SpillError("--engine torch-cuda but no CUDA device is visible",
                             "spill doctor")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def device_label(engine: str) -> str:
    if engine == "torch-cuda":
        try:
            return f"cuda:{torch.cuda.get_device_name(torch.cuda.current_device())}"
        except Exception:
            return "cuda"
    from ..probe import _cpu_name
    return f"cpu:{_cpu_name()}"


def memory_total_bytes(engine: str) -> int:
    """The memory a job may plan against: device memory on CUDA, RAM on CPU."""
    if engine == "torch-cuda":
        return int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pages = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
        return int(pages * os.sysconf("SC_PAGE_SIZE")) if pages else 8 * GIB


def peak_bytes(device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if platform.system() == "Darwin" else peak * 1024)
    except Exception:
        return 0


# ---------------------------------------------------------------- tokenizer

class TorchTokenizer:
    """transformers' tokenizer behind the small surface the engines use (the same surface the
    mlx-lm TokenizerWrapper offers): apply_chat_template -> list[int], decode, EOS lookups."""

    def __init__(self, path: Path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.eos_token_id = self.tok.eos_token_id
        self.pad_token_id = self.tok.pad_token_id

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kw):
        out = self.tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt,
                                           tokenize=tokenize, return_dict=False, **kw)
        if hasattr(out, "keys"):
            out = out["input_ids"]
        return list(out)

    def decode(self, ids):
        return self.tok.decode(list(ids))

    def convert_tokens_to_ids(self, tok):
        return self.tok.convert_tokens_to_ids(tok)


def load_tokenizer(path: Path) -> TorchTokenizer:
    return TorchTokenizer(Path(path))


# ---------------------------------------------------------------- KV caches

class TorchKV:
    """Per-layer key/value store for one batch, laid out [B, H, columns, D]. It is also the
    object the transformers attention modules call `update` on, so it stands in for a Cache."""

    STEP = 256

    def __init__(self, shared=None):
        self.keys = None
        self.values = None
        self.used = 0
        self.shared = shared          # (K, V) [1, H, P, D] of a prefix every row begins with

    def update(self, key, value, layer_idx=None, cache_kwargs=None):
        B, H, L, D = key.shape
        if self.keys is None:
            cap = ((L + self.STEP - 1) // self.STEP) * self.STEP
            self.keys = key.new_zeros((B, H, cap, D))
            self.values = value.new_zeros((B, H, cap, D))
        if self.used + L > self.keys.shape[2]:
            grow = ((L + self.STEP - 1) // self.STEP) * self.STEP
            pad = key.new_zeros((B, H, grow, D))
            self.keys = torch.cat([self.keys, pad], dim=2)
            self.values = torch.cat([self.values, pad.clone()], dim=2)
        self.keys[:, :, self.used:self.used + L] = key
        self.values[:, :, self.used:self.used + L] = value
        self.used += L
        k, v = self.keys[:, :, :self.used], self.values[:, :, :self.used]
        if self.shared is not None:
            sk, sv = self.shared
            if k.shape[0] != 1:
                sk = sk.expand(k.shape[0], -1, -1, -1)
                sv = sv.expand(v.shape[0], -1, -1, -1)
            k, v = torch.cat([sk, k], dim=2), torch.cat([sv, v], dim=2)
        return k, v

    def keep_rows(self, idx: list[int]) -> None:
        if self.keys is not None:
            t = torch.as_tensor(idx, device=self.keys.device)
            self.keys = self.keys.index_select(0, t)
            self.values = self.values.index_select(0, t)

    def compact(self, n: int) -> None:
        """Drop n dead left-pad columns. Rotary positions live in the stored keys, so shifting
        is exact."""
        if self.keys is not None and n > 0:
            self.keys = self.keys[:, :, n:].contiguous()
            self.values = self.values[:, :, n:].contiguous()
            self.used -= n

    @property
    def nbytes(self) -> int:
        return 0 if self.keys is None else self.keys.numel() * self.keys.element_size() * 2


def _finfo_min(dtype):
    return torch.finfo(dtype).min


def attention_mask(q_pos, k_pos, k_valid, dtype, window=None):
    """Additive mask [B, 1, Tq, K]: key j is visible to query i when its position is not in
    the future, the key is real (not left padding), and, for sliding-window layers, within
    `window` positions."""
    allowed = (k_pos[:, None, :] <= q_pos[:, :, None]) & k_valid[:, None, :]
    if window:
        allowed &= (q_pos[:, :, None] - k_pos[:, None, :]) < window
    m = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    return m.masked_fill(~allowed, _finfo_min(dtype)).unsqueeze(1)


def layer_window(cfg, k: int) -> int | None:
    lt = getattr(cfg, "layer_types", None)
    sw = getattr(cfg, "sliding_window", None)
    if lt:
        return sw if (lt[k] == "sliding_attention" and sw) else None
    if getattr(cfg, "use_sliding_window", False) is False and cfg.model_type != "mistral":
        return None
    return sw or None


# ---------------------------------------------------------------- LoRA slots

class _Slot:
    __slots__ = ("a", "b", "scale", "p", "seed")

    def __init__(self, a, b, scale, p=0.0, seed=None):
        self.a, self.b, self.scale, self.p, self.seed = a, b, scale, p, seed


def _lora_forward(self, x):
    y = F.linear(x, self.weight, self.bias)
    slot = self.__dict__.get("_lora")
    if slot is None:
        return y
    xi = x.to(slot.a.dtype)
    if slot.p and slot.seed is not None:
        g = torch.Generator(device=x.device)
        g.manual_seed(int(slot.seed))
        keep = torch.rand(xi.shape, generator=g, device=x.device) >= slot.p
        xi = xi * keep / (1.0 - slot.p)
    z = (xi @ slot.a) @ slot.b
    return y + (slot.scale * z).to(y.dtype)


_LORA_CLASS = None


def lora_class():
    global _LORA_CLASS
    if _LORA_CLASS is None:
        class LoraLinear(torch.nn.Linear):
            forward = _lora_forward
        _LORA_CLASS = LoraLinear
    return _LORA_CLASS


def make_lora_aware(layer, paths) -> None:
    """One-time: every targeted linear in the reusable layer becomes slot-aware (a plain
    passthrough until a slot is attached)."""
    cls = lora_class()
    for path in paths:
        try:
            mod = layer.get_submodule(path)
        except AttributeError:
            raise SpillError(f"the adapter targets module {path}, which this architecture's "
                             f"layer does not have")
        if type(mod) is torch.nn.Linear:
            mod.__class__ = cls
        mod.__dict__["_lora"] = None


def set_slot(layer, path: str, slot: _Slot | None) -> None:
    layer.get_submodule(path).__dict__["_lora"] = slot


class TorchAdapter:
    """An inference adapter (adapters.LoraAdapter with numpy arrays) as float32 tensors on the
    device; the base weights are never merged."""

    def __init__(self, ad, device):
        self.ad = ad
        self.targets = sorted(ad.targets)
        self.layers = {k: {p: (torch.as_tensor(np.asarray(a), dtype=torch.float32,
                                               device=device),
                               torch.as_tensor(np.asarray(b), dtype=torch.float32,
                                               device=device), s)
                           for p, (a, b, s) in mods.items()}
                       for k, mods in ad.layers.items()}

    def prepare(self, layer):
        make_lora_aware(layer, self.targets)

    def attach(self, layer, k: int):
        mods = self.layers.get(k, {})
        for p in self.targets:
            m = mods.get(p)
            set_slot(layer, p, _Slot(*m) if m else None)


# ---------------------------------------------------------------- the model core

class Core:
    """transformers' model on the meta device with one real decoder layer. Construction costs
    no weight memory; `bind` points the reusable layer's parameters at a layer's bytes."""

    def __init__(self, index: SafetensorsIndex, dtype, device, attn_impl: str | None = None):
        require_torch()
        from transformers import AutoConfig, AutoModelForCausalLM

        from .supported import NOT_YET, classify
        self.index = index
        cfg_dict = index.config
        fam = classify(cfg_dict)
        if fam.state == NOT_YET or not fam.module:
            raise SpillError(f"unsupported architecture: {fam.label} is '{fam.state}' "
                             f"({fam.note})", "spill models --architectures")
        if cfg_dict.get("quantization"):
            raise SpillError("the torch engines run bf16, float16 or float32 weights; this "
                             "directory is quantized", "spill run <model> <file> --engine mlx")
        self.fam = fam
        self.dtype, self.device = dtype, device
        cfg = AutoConfig.from_pretrained(str(index.model_dir))
        cfg._attn_implementation = attn_impl or (
            "eager" if cfg.model_type in ("gemma2",) else "sdpa")
        self.cfg = cfg
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
        model.requires_grad_(False)
        inner = model.model
        self.layer = inner.layers[0]
        self.layer.to_empty(device=device)
        self.layer.requires_grad_(False)
        self.layer.eval()
        self.params = dict(self.layer.named_parameters())
        self.norm = inner.norm
        self.norm.to_empty(device=device)
        rot_cls = type(inner.rotary_emb)
        self.rotary = rot_cls(config=cfg, device=device)
        self.n_layers = index.n_layers
        self.hidden = cfg.hidden_size
        self.embed_scale = (cfg.hidden_size ** 0.5) if cfg.model_type in ("gemma2", "gemma") \
            else None
        self.softcap = getattr(cfg, "final_logit_softcapping", None)
        self.windows = [layer_window(cfg, k) for k in range(self.n_layers)]
        # embeddings, final norm and lm_head are small enough to stay resident
        self.embed_w = self.read_tensor(index.embed)
        self.norm.weight.data = self.read_tensor(index.final_norm)
        self.lm_w = self.embed_w if index.tied else self.read_tensor(index.lm_head)
        self.vocab = int(self.lm_w.shape[0])
        self.adapter: TorchAdapter | None = None
        del model

    # -- weights

    def read_tensor(self, loc: TensorLoc):
        with open(loc.shard, "rb") as f:
            f.seek(loc.offset)
            raw = bytearray(f.read(loc.nbytes))
        t = torch.frombuffer(raw, dtype=torch.uint8).view(_st_torch(loc.st_dtype)).reshape(loc.shape)
        return t.to(device=self.device, dtype=self.dtype)

    def bind_from_buffer(self, plan, u8) -> float:
        """Point the reusable layer's parameters at tensors in `u8` (a uint8 tensor over the
        ring slot, or a device buffer holding the same layout). Zero-copy when the storage
        dtype and device already match."""
        t0 = time.perf_counter()
        for t in plan.tensors:
            boff = plan.tensor_buf_offsets[t.name]
            name = t.name[len(f"model.layers.{plan.layer_id}."):]
            arr = u8[boff:boff + t.nbytes].view(_st_torch(t.st_dtype)).reshape(t.shape)
            if arr.dtype != self.dtype or arr.device != self.device:
                arr = arr.to(device=self.device, dtype=self.dtype)
            self.params[name].data = arr
        return time.perf_counter() - t0

    def bind_tensors(self, tensors: dict) -> None:
        for name, arr in tensors.items():
            self.params[name].data = arr

    # -- forward pieces

    def embed(self, ids):
        h = F.embedding(ids, self.embed_w)
        if self.embed_scale:
            h = h * torch.tensor(self.embed_scale, dtype=h.dtype)
        return h

    def rope(self, h, position_ids):
        return self.rotary(h, position_ids)

    def layer_forward(self, h, k: int, *, mask, position_ids, pos_emb, cache=None):
        out = self.layer(hidden_states=h, attention_mask=mask, position_ids=position_ids,
                         past_key_values=cache, use_cache=cache is not None,
                         position_embeddings=pos_emb)
        return out[0] if isinstance(out, tuple) else out

    def logits(self, h):
        x = self.norm(h)
        lg = F.linear(x, self.lm_w)
        if self.softcap:
            lg = torch.tanh(lg / self.softcap) * self.softcap
        return lg


# ---------------------------------------------------------------- weight providers

class ResidentProvider:
    """Every layer's weights in memory, converted once; binding is a pointer swap."""

    def __init__(self, core: Core):
        self.core = core
        self.layers: list[dict] = []
        for plan in core.index.layers:
            d = {}
            for t in plan.tensors:
                name = t.name[len(f"model.layers.{plan.layer_id}."):]
                d[name] = core.read_tensor(t)
            self.layers.append(d)
        self.bind_seconds = 0.0
        self.wait_seconds = 0.0

    def bind(self, k: int):
        self.core.bind_tensors(self.layers[k])
        return None

    def release(self, slot):
        pass

    def close(self):
        pass

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for d in self.layers for t in d.values())


class RingProvider:
    """Layers stream from the NVMe ring in schedule order. On CPU the compute reads the host
    buffers directly. `schedule` is an iterator of layer ids in consumption order."""

    def __init__(self, core: Core, schedule, note=None):
        from ..ring import ring_settings
        self.core = core
        chunk_mb, threads, mode = ring_settings(core.index, note)
        self.ring = RingReader(core.index, 3, chunk_mb * 1024 * 1024, threads, mode=mode)
        self.mode = mode
        self.u8 = [torch.frombuffer(b, dtype=torch.uint8) for b in self.ring.bufs]
        self.seq = 0
        self.bind_seconds = 0.0
        self.wait_seconds = 0.0
        self.ring.start(schedule)

    def bind(self, k: int):
        t0 = time.perf_counter()
        slot, _ = self.ring.get(self.seq)
        self.wait_seconds += time.perf_counter() - t0
        self.seq += 1
        self.bind_seconds += self.core.bind_from_buffer(self.core.index.layers[k], self.u8[slot])
        return slot

    def release(self, slot):
        if slot is not None:
            self.ring.release(slot)

    def close(self):
        self.ring.stop()


class CudaRingProvider:
    """The ring with pinned host buffers; a side stream copies upcoming layers to the device
    while the compute stream works on the current one (layer k computes while k+1 .. k+N
    load). Device buffers are a small pool; a buffer returns to the pool once the compute
    stream has passed the layer that used it."""

    N_DEVICE = 3

    def __init__(self, core: Core, schedule, note=None):
        from ..ring import ring_settings
        self.core = core
        chunk_mb, threads, mode = ring_settings(core.index, note)
        dev = core.device

        self.host = []                  # the pinned tensors themselves (numpy views share them)

        def pinned(n):
            t = torch.empty(n, dtype=torch.uint8).pin_memory()
            self.host.append(t)
            return t.numpy()
        self.ring = RingReader(core.index, 4, chunk_mb * 1024 * 1024, threads, mode=mode,
                               alloc=pinned)
        self.mode = mode
        n = core.index.max_layer_bytes
        self.free = queue.Queue()
        for _ in range(self.N_DEVICE):
            self.free.put((torch.empty(n, dtype=torch.uint8, device=dev), None))
        self.side = torch.cuda.Stream(device=dev)
        self.ready: dict[int, tuple] = {}
        self.cond = threading.Condition()
        self.seq = 0
        self.bind_seconds = 0.0
        self.wait_seconds = 0.0
        self._stop = threading.Event()
        self.error = None
        self.ring.start(schedule)
        self._copier = threading.Thread(target=self._copy_loop, args=(core.index,), daemon=True)
        self._copier.start()

    def _copy_loop(self, index):
        try:
            torch.cuda.set_device(self.core.device)
            seq = 0
            while not self._stop.is_set():
                slot, _ = self.ring.get(seq)
                nbytes = index.layers[self.ring.last_layer].nbytes
                dbuf, done_ev = self.free.get()
                with torch.cuda.stream(self.side):
                    if done_ev is not None:
                        self.side.wait_event(done_ev)
                    dbuf[:nbytes].copy_(self.host[slot][:nbytes], non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(self.side)
                ev.synchronize()                 # the pinned slot is free once the copy ran
                self.ring.release(slot)
                with self.cond:
                    self.ready[seq] = (dbuf, ev)
                    self.cond.notify_all()
                seq += 1
        except BaseException as e:               # includes the ring's own reader errors
            self.error = e
            with self.cond:
                self.cond.notify_all()

    def bind(self, k: int):
        t0 = time.perf_counter()
        with self.cond:
            while self.seq not in self.ready:
                if self.error is not None:
                    raise self.error
                self.cond.wait(timeout=60)
            dbuf, ev = self.ready.pop(self.seq)
        torch.cuda.current_stream().wait_event(ev)      # GPU-side wait, no CPU stall
        self.wait_seconds += time.perf_counter() - t0
        self.seq += 1
        self.bind_seconds += self.core.bind_from_buffer(self.core.index.layers[k], dbuf)
        return dbuf

    def release(self, dbuf):
        if dbuf is not None:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            self.free.put((dbuf, ev))

    def close(self):
        self._stop.set()
        self.ring.stop()


def linear_shapes_meta(index: SafetensorsIndex, only: list[str] | None = None
                       ) -> dict[str, tuple[int, int]]:
    """{module path: (in, out)} of every linear in one decoder layer, from a meta-device model
    (no weight memory) of the same transformers class the engines run."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from ..tune.lora_core import select_shapes
    cfg = AutoConfig.from_pretrained(str(index.model_dir))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    found = {p: (m.in_features, m.out_features)
             for p, m in model.model.layers[0].named_modules() if isinstance(m, torch.nn.Linear)}
    return select_shapes(found, only)
