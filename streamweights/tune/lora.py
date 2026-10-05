"""LoRA parameters for training: configuration, shared initialization, and the
adapter directory (PEFT and mlx-lm layouts side by side).

Parameters are a flat dict {"layers.<k>.<module path>.lora_a": [in, r] f32,
"...lora_b": [r, out] f32}. Both trainers (mlx-lm resident, streamed) start from
the same `init_params(seed)` values, so "identical init" is true by construction
rather than by hoping two RNG streams line up.

One adapter directory serves both ecosystems. adapter_model.safetensors is PEFT
(A [r, in], B [out, r], HF key names); adapters.safetensors is mlx-lm. The single
adapter_config.json carries both vocabularies (peft_type/r/lora_alpha/... and
fine_tune_type/num_layers/lora_parameters/...), which do not collide.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..errors import SpillError

_KEY = re.compile(r"^layers\.(\d+)\.(.+)\.lora_([ab])$")


@dataclass
class LoraConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: list[str] | None = None      # module paths in a block; None = every linear
    lr: float = 1e-4
    weight_decay: float = 0.01
    schedule: str = "cosine"              # cosine | constant
    seed: int = 0

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def validate(self) -> None:
        if self.rank < 1:
            raise SpillError("--rank must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise SpillError("--dropout must be in [0, 1)")
        if self.lr <= 0:
            raise SpillError("--lr must be positive")


def linear_shapes(block, only: list[str] | None = None) -> dict[str, tuple[int, int]]:
    """{module path: (in_features, out_features)} for every linear in one
    transformer block (quantized linears report their logical width)."""
    out: dict[str, tuple[int, int]] = {}
    for path, mod in block.named_modules():
        if isinstance(mod, nn.QuantizedLinear):
            out_f, packed = mod.weight.shape
            out[path] = (packed * 32 // mod.bits, out_f)
        elif isinstance(mod, nn.Linear):
            out_f, in_f = mod.weight.shape
            out[path] = (in_f, out_f)
    if only:
        miss = [t for t in only if t not in out]
        if miss:
            raise SpillError(f"--targets {', '.join(miss)} not found in this architecture's "
                             f"block; linear modules are: {', '.join(sorted(out))}")
        out = {p: out[p] for p in only}
    if not out:
        raise SpillError("no linear modules to adapt in this architecture's block")
    return dict(sorted(out.items()))


def param_name(k: int, path: str, ab: str) -> str:
    return f"layers.{k}.{path}.lora_{ab}"


def init_params(shapes: dict[str, tuple[int, int]], n_layers: int, rank: int,
                seed: int) -> dict[str, mx.array]:
    """mlx-lm's LoRALinear init (A uniform in +-1/sqrt(in), B zero), float32, from
    one deterministic key stream over (layer, module) in sorted order."""
    key = mx.random.key(seed)
    params: dict[str, mx.array] = {}
    keys = mx.random.split(key, n_layers * len(shapes))
    i = 0
    for k in range(n_layers):
        for path, (din, dout) in shapes.items():
            s = 1.0 / math.sqrt(din)
            params[param_name(k, path, "a")] = mx.random.uniform(
                low=-s, high=s, shape=(din, rank), key=keys[i]).astype(mx.float32)
            params[param_name(k, path, "b")] = mx.zeros((rank, dout), mx.float32)
            i += 1
    mx.eval(params)
    return params


def n_param_bytes(shapes: dict[str, tuple[int, int]], n_layers: int, rank: int) -> int:
    return n_layers * sum(rank * (i + o) for i, o in shapes.values()) * 4


def by_layer(params: dict[str, mx.array], scale: float) -> dict[int, dict[str, tuple]]:
    """Flat dict -> {layer: {path: (A, B, scale)}}, the shape adapters.LoraAdapter uses."""
    layers: dict[int, dict] = {}
    for name, arr in params.items():
        m = _KEY.match(name)
        if not m:
            raise SpillError(f"unexpected adapter parameter name {name}")
        k, path, ab = int(m.group(1)), m.group(2), m.group(3)
        slot = layers.setdefault(k, {}).setdefault(path, [None, None])
        slot[0 if ab == "a" else 1] = arr
    return {k: {p: (ab[0], ab[1], scale) for p, ab in mods.items()} for k, mods in layers.items()}


# ------------------------------------------------------------ directory layouts

def _union_config(cfg: LoraConfig, base: str, n_layers: int, shapes, extra: dict) -> dict:
    names = sorted({p.rsplit(".", 1)[-1] for p in shapes})
    return {
        # PEFT vocabulary
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": cfg.rank,
        "lora_alpha": cfg.alpha, "lora_dropout": cfg.dropout, "bias": "none",
        "target_modules": names, "base_model_name_or_path": base,
        "use_dora": False, "use_rslora": False, "fan_in_fan_out": False,
        # mlx-lm vocabulary
        "model": base, "fine_tune_type": "lora", "num_layers": n_layers,
        "lora_parameters": {"rank": cfg.rank, "scale": cfg.scale, "dropout": cfg.dropout,
                            "keys": sorted(shapes)},
        # ours
        "streamweights": extra,
    }


def save_adapter_dir(d: Path, params: dict[str, mx.array], cfg: LoraConfig, base: str,
                     n_layers: int, shapes, extra: dict | None = None) -> None:
    """Write both layouts atomically enough for a crash not to leave a half adapter:
    tensors first, config last (readers key off the config)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    mx_tensors = {f"model.{k}": v for k, v in params.items()}
    mx.save_safetensors(str(d / "adapters.tmp.safetensors"), mx_tensors)
    peft = {}
    for name, arr in params.items():
        m = _KEY.match(name)
        k, path, ab = m.group(1), m.group(2), m.group(3)
        peft[f"base_model.model.model.layers.{k}.{path}.lora_{ab.upper()}.weight"] = arr.T
    mx.save_safetensors(str(d / "adapter_model.tmp.safetensors"), peft)
    (d / "adapters.tmp.safetensors").replace(d / "adapters.safetensors")
    (d / "adapter_model.tmp.safetensors").replace(d / "adapter_model.safetensors")
    (d / "adapter_config.json").write_text(json.dumps(
        _union_config(cfg, base, n_layers, shapes, extra or {}), indent=2))


def read_adapter_params(d: Path) -> dict[str, mx.array]:
    """Flat params from an adapter directory's mlx-lm layout (our canonical form)."""
    d = Path(d)
    f = d / "adapters.safetensors"
    if not f.exists():
        raise SpillError(f"{d} has no adapters.safetensors")
    out = {}
    for name, arr in mx.load(str(f)).items():
        out[name[len("model."):] if name.startswith("model.") else name] = arr
    return out


def config_from_dir(d: Path) -> tuple[LoraConfig, dict]:
    raw = json.loads((Path(d) / "adapter_config.json").read_text())
    lp = raw.get("lora_parameters", {})
    cfg = LoraConfig(rank=int(raw.get("r", lp.get("rank", 16))),
                     alpha=float(raw.get("lora_alpha", 32.0)),
                     dropout=float(raw.get("lora_dropout", lp.get("dropout", 0.0))),
                     targets=lp.get("keys"))
    return cfg, raw


def cosine_sim(a: mx.array, b: mx.array) -> float:
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    den = mx.sqrt((a * a).sum()) * mx.sqrt((b * b).sum())
    return float(((a * b).sum() / mx.maximum(den, 1e-30)).item())
