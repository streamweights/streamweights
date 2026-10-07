"""LoRA training on PyTorch: the optimizer, the loss, and the two trainers.

  StreamedTrainer   the same design as the MLX streamed trainer: one micro-batch is two trips
                    through the layers. Forward saves each layer's INPUT; backward, with the
                    weights streamed again in reverse order, recomputes each layer from its
                    saved input and takes a vector-Jacobian product with respect to the LoRA
                    parameters and the layer input only. Base weights are constants.
  PeftTrainer       the whole model in memory, trained through PEFT's LoRA layers and
                    autograd. The reference the streamed trainer is gated against.

Both consume the same BatchPlan, start from the same numpy init (lora_core.init_params_np),
use the same optimizer, and define a step's loss the same way (the mean of its micro-batches'
token-mean losses). Parameters are float32 [in, r] and [r, out] under the portable names; the
optimizer state is float32 and is exactly what the portable checkpoint stores.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..engines.torch_common import (Core, ResidentProvider, RingProvider, _Slot, attention_mask,
                                    make_lora_aware, set_slot, torch)
from ..ring import SafetensorsIndex
from . import lora_core as lo
from .budget import head_chunk_positions

F = torch.nn.functional if torch is not None else None


# ---------------------------------------------------------------- optimizer and loss

class AdamW:
    """AdamW with exactly the arithmetic the MLX trainers use (so a run can move between
    engines): no bias correction, decoupled weight decay applied to the parameter first,

        m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g^2
        p = p (1 - lr wd);      p = p - lr m / (sqrt(v) + eps)

    with the learning rate read from a cosine (or constant) schedule at the step count BEFORE
    the step is counted. State is float32 and round-trips through the portable checkpoint."""

    def __init__(self, params: dict, lr: float, steps: int, schedule: str = "cosine",
                 weight_decay: float = 0.01, betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr, self.steps, self.schedule = lr, max(1, steps), schedule
        self.wd, self.b1, self.b2, self.eps = weight_decay, betas[0], betas[1], eps
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}
        self.step = 0

    def lr_at(self, n: int) -> float:
        if self.schedule != "cosine":
            return self.lr
        s = min(n, self.steps)
        return float(np.float32(0.5 * (1.0 + math.cos(math.pi / self.steps * s)) * self.lr))

    def apply(self, params: dict, grads: dict) -> None:
        lr = torch.tensor(self.lr_at(self.step), dtype=torch.float32)
        self.step += 1
        with torch.no_grad():
            for k, p in params.items():
                g = grads[k]
                m = self.m[k].mul_(self.b1).add_(g, alpha=1 - self.b1)
                v = self.v[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                p.mul_(1 - lr * self.wd)
                p.sub_(lr * m / (v.sqrt() + self.eps))

    def state(self) -> dict:
        return {"step": self.step, "m": self.m, "v": self.v}

    def load(self, state: dict) -> None:
        self.step = int(state["step"])
        for k in self.m:
            self.m[k] = torch.as_tensor(np.array(state["m"][k]), dtype=torch.float32)
            self.v[k] = torch.as_tensor(np.array(state["v"][k]), dtype=torch.float32)


def masked_ce(logits, targets, mask):
    """Token-mean cross entropy over masked positions, float32."""
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                         reduction="none") * mask.reshape(-1)
    ntoks = mask.sum()
    return ce.sum() / ntoks, ntoks


def to_params(np_params: dict) -> dict:
    return {k: torch.as_tensor(np.array(v), dtype=torch.float32).contiguous()
            for k, v in np_params.items()}


# ---------------------------------------------------------------- streamed trainer

class StreamedTrainer:
    def __init__(self, model_dir: Path, cfg: lo.LoraConfig, params: dict | None, *, dtype,
                 device, shapes: dict, resident_weights: bool = False, note=None):
        self.note = note or (lambda s: None)
        self.cfg = cfg
        self.index = SafetensorsIndex(model_dir)
        self.core = Core(self.index, dtype, device)
        self.L = self.index.n_layers
        self.paths = list(shapes)
        self.device, self.dtype = device, dtype
        np_params = params if params is not None else lo.init_params_np(
            shapes, self.L, cfg.rank, cfg.seed)
        self.params = to_params(np_params)
        make_lora_aware(self.core.layer, self.paths)

        def schedule():
            while True:
                yield from range(self.L)
                yield from range(self.L - 1, -1, -1)
        if resident_weights:
            self.provider = ResidentProvider(self.core)
        elif device.type == "cuda":
            from ..engines.torch_common import CudaRingProvider
            self.provider = CudaRingProvider(self.core, schedule(), self.note)
        else:
            self.provider = RingProvider(self.core, schedule(), self.note)
        self.bind_seconds = self.wait_seconds = self.micro_seconds = 0.0
        self.micro_tokens = 0
        self.vocab = self.core.vocab
        self.chunk = head_chunk_positions(self.vocab)

    def close(self):
        self.provider.close()

    @property
    def ring(self):
        return getattr(self.provider, "ring", None)

    def _slots(self, k: int, leaf: dict, micro: int):
        c = self.cfg
        for j, path in enumerate(self.paths):
            seed = None
            if c.dropout:
                seed = ((c.seed * 1000003 + micro) * 4093 + k) * 64 + j
            a, b = leaf[lo.param_name(k, path, "a")], leaf[lo.param_name(k, path, "b")]
            set_slot(self.core.layer, path, _Slot(a, b, c.scale, c.dropout, seed))

    def _rope_mask(self, B: int, T: int, h):
        dev = self.device
        pos = torch.arange(T, device=dev)[None].expand(B, -1)
        cs = self.core.rope(h, pos)
        valid = torch.ones(B, T, dtype=torch.bool, device=dev)
        masks = {}
        for w in set(self.core.windows):
            masks[w] = attention_mask(pos, pos, valid, h.dtype, w)
        return pos, cs, masks

    def micro_batch(self, inputs, targets, mask, micro: int):
        """Forward + backward for one micro-batch. Returns (loss, grads, ntoks); grads are
        float32 {portable name: tensor} of the token-mean loss."""
        t_micro = time.monotonic()
        core, dev = self.core, self.device
        ids = torch.as_tensor(inputs, dtype=torch.long, device=dev)
        B, T = ids.shape
        tgt = torch.as_tensor(targets, dtype=torch.long, device=dev).reshape(-1)
        msk = torch.as_tensor(mask, dtype=torch.float32, device=dev).reshape(-1)
        ntoks = msk.sum()
        with torch.no_grad():
            h = core.embed(ids)
        pos, cs, masks = self._rope_mask(B, T, h)
        saved: list = []
        with torch.no_grad():
            for k in range(self.L):
                slot = self.provider.bind(k)
                self._slots(k, self.params, micro)
                saved.append(h)
                h = core.layer_forward(h, k, mask=masks[core.windows[k]], position_ids=pos,
                                       pos_emb=cs)
                self.provider.release(slot)
        hf = h.reshape(B * T, -1)
        total = 0.0
        dh_chunks = []
        for c0 in range(0, B * T, self.chunk):
            sl = slice(c0, c0 + self.chunk)
            hc = hf[sl].detach().requires_grad_(True)
            with torch.enable_grad():
                ce = F.cross_entropy(core.logits(hc).float(), tgt[sl], reduction="none")
                lc = (ce * msk[sl]).sum() / ntoks
                (gc,) = torch.autograd.grad(lc, hc)
            total += float(lc.detach())
            dh_chunks.append(gc)
        loss = total
        g = torch.cat(dh_chunks, dim=0).reshape(B, T, -1)
        del hf, h, dh_chunks

        grads: dict = {}
        for k in range(self.L - 1, -1, -1):
            slot = self.provider.bind(k)
            leaf_names = [lo.param_name(k, p, ab) for p in self.paths for ab in ("a", "b")]
            leaf = {n: self.params[n].detach().clone().requires_grad_(True) for n in leaf_names}
            self._slots(k, {**self.params, **leaf}, micro)
            x = saved[k].detach()
            if k > 0:
                x = x.requires_grad_(True)
            with torch.enable_grad():
                y = core.layer_forward(x, k, mask=masks[core.windows[k]], position_ids=pos,
                                       pos_emb=cs)
                wrt = [leaf[n] for n in leaf_names] + ([x] if k > 0 else [])
                out = torch.autograd.grad(y, wrt, grad_outputs=g)
            for n, gr in zip(leaf_names, out):
                grads[n] = gr.detach()
            if k > 0:
                g = out[-1].detach()
            saved[k] = None
            self.provider.release(slot)
        self.micro_seconds += time.monotonic() - t_micro
        self.micro_tokens += B * T
        return loss, grads, int(ntoks.item())


# ---------------------------------------------------------------- PEFT reference trainer

class PeftTrainer:
    """The model fully in memory, LoRA through PEFT, ordinary autograd."""

    def __init__(self, model_dir: Path, cfg: lo.LoraConfig, params: dict | None, *, dtype,
                 device, shapes: dict, note=None):
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM
        self.cfg, self.device = cfg, device
        self.index = SafetensorsIndex(model_dir)
        self.L = self.index.n_layers
        self.paths = list(shapes)
        base = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype).to(device)
        base.requires_grad_(False)
        base.eval()
        pcfg = PeftLoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                              target_modules=list(self.paths), bias="none",
                              task_type="CAUSAL_LM")
        self.model = get_peft_model(base, pcfg)
        np_params = params if params is not None else lo.init_params_np(
            shapes, self.L, cfg.rank, cfg.seed)
        self.params = to_params(np_params)
        self.mods = {}
        for k in range(self.L):
            for p in self.paths:
                mod = self.model.base_model.model.model.layers[k].get_submodule(p)
                self.mods[(k, p)] = mod
        self.model.train(False)         # dropout, when asked for, is applied by PEFT in train mode
        if cfg.dropout:
            self.model.train(True)
        self.micro_seconds = 0.0
        self.micro_tokens = 0
        self.wait_seconds = 0.0
        self.bind_seconds = 0.0

    def close(self):
        pass

    def _load(self):
        with torch.no_grad():
            for (k, p), mod in self.mods.items():
                a = self.params[lo.param_name(k, p, "a")]
                b = self.params[lo.param_name(k, p, "b")]
                mod.lora_A["default"].weight.copy_(a.T)
                mod.lora_B["default"].weight.copy_(b.T)

    def micro_batch(self, inputs, targets, mask, micro: int):
        t0 = time.monotonic()
        self._load()
        for q in self.model.parameters():
            q.grad = None
        ids = torch.as_tensor(inputs, dtype=torch.long, device=self.device)
        tgt = torch.as_tensor(targets, dtype=torch.long, device=self.device)
        msk = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        with torch.enable_grad():
            logits = self.model(input_ids=ids).logits
            loss, ntoks = masked_ce(logits, tgt, msk)
            loss.backward()
        grads = {}
        for (k, p), mod in self.mods.items():
            grads[lo.param_name(k, p, "a")] = mod.lora_A["default"].weight.grad.T.detach().clone()
            grads[lo.param_name(k, p, "b")] = mod.lora_B["default"].weight.grad.T.detach().clone()
        self.micro_seconds += time.monotonic() - t0
        self.micro_tokens += int(ids.numel())
        return float(loss.detach()), grads, int(ntoks.item())
