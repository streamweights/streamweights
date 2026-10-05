"""Streamed LoRA training: the base model never sits in memory.

One micro-batch is two trips through the layers, each a full weight stream:

  forward   layers 0..L-1 in order. Each layer's weights come from the NVMe ring;
            the micro-batch's activations go through it, and the layer's INPUT is
            kept (not its output; the output is the next layer's input).
  backward  layers L-1..0, weights streamed again. For each layer the forward is
            recomputed from its saved input and differentiated with a vector-
            Jacobian product taken with respect to the LoRA parameters and the
            layer input only. Base weights are constants: they get no gradient.
            The input gradient becomes the upstream gradient of the layer below.

Two weight streams per micro-batch (forward, backward-with-recompute), not three.
LoRA parameters for every layer, their gradients and AdamW state stay resident
(they are small). The loss head (final norm, lm_head, cross entropy) runs on
chunks of positions so [B, S, vocab] logits never exist at once.

The ring is the Phase 1 ring: the same reader, slots and binding as inference,
fed a schedule that runs the layers forward then backward, repeating.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ..adapters import _resolve, _Slot
from ..engines.mlx_stream import (MB, ResidentProvider, RingReader, SafetensorsIndex,
                                  _bind, _load_resident, _load_resident_maybe_quantized,
                                  _model_modules, calibrate, load_calibration)
from . import lora as lo
from .budget import head_chunk_positions


def make_optimizer(cfg: lo.LoraConfig, steps: int):
    lr = (optim.cosine_decay(cfg.lr, max(1, steps)) if cfg.schedule == "cosine" else cfg.lr)
    return optim.AdamW(learning_rate=lr, weight_decay=cfg.weight_decay)


def masked_ce(logits, targets, mask):
    """Token-mean cross entropy over masked positions: exactly mlx-lm's default_loss
    arithmetic (bf16 logits, float32 sum), so both trainers compute one loss."""
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


class StreamedTrainer:
    def __init__(self, model_dir: Path, cfg: lo.LoraConfig, params: dict | None = None,
                 resident_weights: bool = False, note=None):
        self.note = note or (lambda s: None)
        self.cfg = cfg
        self.index = SafetensorsIndex(model_dir)
        self.config = self.index.config
        self.block, self.args, self.fam = _model_modules(self.config)
        self.L = self.index.n_layers
        self.shapes = lo.linear_shapes(self.block, cfg.targets)
        self.paths = list(self.shapes)
        self.params = params if params is not None else lo.init_params(
            self.shapes, self.L, cfg.rank, cfg.seed)
        self._prepare_block()

        self.provider = ResidentProvider(self.index) if resident_weights else None
        self.ring: RingReader | None = None
        if self.provider is None:
            cal = load_calibration()
            if "isolated_read_mbps" not in cal:
                self.note("calibrating read rate (no compute)...")
                cal = calibrate(self.index)
            self.ring = RingReader(self.index, 3, cal.get("chunk_mb", 16) * MB,
                                   cal.get("threads", 4))
        self._seq = 0
        self._ring_started = False
        self.bind_seconds = 0.0
        self.wait_seconds = 0.0        # blocked on the ring (reads not ahead of compute)
        self.micro_seconds = 0.0       # wall time inside micro_batch()
        self.micro_tokens = 0          # padded tokens pushed through (B x T)

        ix = self.index
        self.embed_w = _load_resident_maybe_quantized(ix, "model.embed_tokens")
        self.norm_w = _load_resident(ix.final_norm).astype(mx.bfloat16)
        if self.fam.norm_plus_one:
            self.norm_w = self.norm_w + 1.0
        self.lm_w = (self.embed_w if ix.tied
                     else _load_resident_maybe_quantized(ix, "lm_head"))
        self.embed_scale = (self.config["hidden_size"] ** 0.5) if self.fam.embed_scale else None
        self.softcap = (self.config.get(self.fam.final_softcap_key)
                        if self.fam.final_softcap_key else None)
        self.eps = self.config.get("rms_norm_eps", 1e-5)
        self.vocab = int(self.lm_w.shape[0])
        self.chunk = head_chunk_positions(self.vocab)

    # ---- plumbing

    def _prepare_block(self):
        from ..adapters import LoraAdapter
        ad = LoraAdapter("train", Path("."), "mlx-lm", self.cfg.rank, "", None, {},
                         set(self.paths))
        ad.prepare(self.block)

    def close(self):
        if self.ring is not None:
            self.ring.stop()

    def _start_ring(self):
        def schedule():
            L = self.L
            while True:
                yield from range(L)
                yield from range(L - 1, -1, -1)
        self.ring.start(schedule())
        self._ring_started = True

    def _bind_layer(self, k: int):
        """Bind layer k's base weights into the block; returns the ring slot to release."""
        if self.provider is not None:
            self.block.update(self.provider.trees[k])
            return None
        t0 = time.monotonic()
        slot, buf = self.ring.get(self._seq)
        self.wait_seconds += time.monotonic() - t0
        self._seq += 1
        self.bind_seconds += _bind(self.block, self.index.layers[k], buf, f"model.layers.{k}.")
        return slot

    def _release(self, slot):
        if slot is not None:
            self.ring.release(slot)

    def _set_slots(self, k: int, ab: dict, micro: int):
        c = self.cfg
        for j, path in enumerate(self.paths):
            a, b = ab[path]
            key = None
            if c.dropout:
                seed = ((c.seed * 1000003 + micro) * 4093 + k) * 64 + j
                key = mx.random.key(seed)
            _resolve(self.block, path).__dict__["_lora"] = _Slot(
                a, b, c.scale, c.dropout, key)

    def _layer_params(self, k: int) -> dict:
        return {p: (self.params[lo.param_name(k, p, "a")], self.params[lo.param_name(k, p, "b")])
                for p in self.paths}

    def _mask(self, T: int, dtype):
        if not self.fam.needs_array_mask:
            return "causal"
        return mx.where(mx.arange(T)[:, None] >= mx.arange(T)[None, :],
                        mx.array(0, dtype), mx.array(-mx.inf, dtype))

    def _embed(self, ids):
        h = self.embed_w[ids]
        return (h * self.embed_scale).astype(h.dtype) if self.embed_scale else h

    def _logits(self, h):
        logits = mx.fast.rms_norm(h, self.norm_w, self.eps) @ self.lm_w.T
        if self.softcap:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        return logits

    # ---- one micro-batch

    def micro_batch(self, inputs, targets, mask, micro: int):
        """Forward + backward for one micro-batch. Returns (loss, grads, ntoks) where
        grads is {param name: float32 array} of the token-mean loss."""
        if self.ring is not None and not self._ring_started:
            self._start_ring()
        t_micro = time.monotonic()
        ids = mx.array(inputs)
        B, T = ids.shape
        tgt = mx.array(targets).reshape(-1)
        msk = mx.array(mask).reshape(-1)
        ntoks = msk.sum()

        h = self._embed(ids)
        mx.eval(h)
        m = self._mask(T, h.dtype)
        saved: list = []
        for k in range(self.L):
            slot = self._bind_layer(k)
            self._set_slots(k, self._layer_params(k), micro)
            saved.append(h)                      # this layer's input
            h = self.block(h, mask=m, cache=None)
            mx.eval(h)
            self._release(slot)

        hf = h.reshape(B * T, -1)
        total = mx.array(0.0)
        dh_chunks = []

        def head_loss(hc, tc, mc):
            ce = nn.losses.cross_entropy(self._logits(hc), tc) * mc
            return ce.astype(mx.float32).sum() / ntoks

        vg = mx.value_and_grad(head_loss)
        for c0 in range(0, B * T, self.chunk):
            sl = slice(c0, c0 + self.chunk)
            lc, gc = vg(hf[sl], tgt[sl], msk[sl])
            mx.eval(lc, gc)
            total = total + lc
            dh_chunks.append(gc)
        loss = float(total.item())
        g = mx.concatenate(dh_chunks, axis=0).reshape(B, T, -1)
        del hf, h, dh_chunks

        grads: dict = {}
        nparam = 2 * len(self.paths)
        for k in range(self.L - 1, -1, -1):
            slot = self._bind_layer(k)
            x = saved[k]
            lp = self._layer_params(k)
            flat = [t for p in self.paths for t in lp[p]]

            def f(*a, _x=x, _k=k):
                ab = {p: (a[2 * j], a[2 * j + 1]) for j, p in enumerate(self.paths)}
                self._set_slots(_k, ab, micro)
                return self.block(a[nparam] if len(a) > nparam else _x, mask=m, cache=None)

            primals = flat + ([x] if k > 0 else [])      # no input gradient below layer 0
            _, vjps = mx.vjp(f, primals, [g])
            mx.eval(vjps)
            for j, p in enumerate(self.paths):
                grads[lo.param_name(k, p, "a")] = vjps[2 * j]
                grads[lo.param_name(k, p, "b")] = vjps[2 * j + 1]
            if k > 0:
                g = vjps[nparam]
            saved[k] = None
            self._release(slot)
        mx.clear_cache()
        self.micro_seconds += time.monotonic() - t_micro
        self.micro_tokens += B * T
        return loss, grads, int(ntoks.item())

    # ---- optimizer step

    def apply(self, optimizer, grads: dict):
        self.params = optimizer.apply_gradients(grads, self.params)
        mx.eval(self.params, optimizer.state)
