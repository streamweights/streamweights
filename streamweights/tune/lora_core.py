"""LoRA configuration, naming, initialization and adapter directories: everything about a LoRA
parameter set that does not depend on MLX or PyTorch, in numpy float32.

Parameters are a flat dict {"layers.<k>.<module path>.lora_a": [in, r] f32,
"...lora_b": [r, out] f32}. Every trainer on every engine starts from `init_params_np(seed)`,
so "identical init" is true by construction rather than by hoping two RNG streams line up.

One adapter directory serves both ecosystems. adapter_model.safetensors is PEFT
(A [r, in], B [out, r], HF key names); adapters.safetensors is mlx-lm. The single
adapter_config.json carries both vocabularies (peft_type/r/lora_alpha/... and
fine_tune_type/num_layers/lora_parameters/...), which do not collide.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import SpillError
from ..portable.checkpoint import to_numpy_f32

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


def select_shapes(found: dict[str, tuple[int, int]],
                  only: list[str] | None) -> dict[str, tuple[int, int]]:
    """Restrict {module path: (in, out)} to the requested targets, with the one-line errors."""
    out = dict(found)
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


def init_params_np(shapes: dict[str, tuple[int, int]], n_layers: int, rank: int,
                   seed: int) -> dict[str, np.ndarray]:
    """LoRA init (A uniform in +-1/sqrt(in), B zero), float32, from one independent stream per
    (layer, module in sorted order). Deterministic and engine-neutral."""
    params: dict[str, np.ndarray] = {}
    for k in range(n_layers):
        for j, (path, (din, dout)) in enumerate(shapes.items()):
            rng = np.random.default_rng(np.random.SeedSequence([seed, k, j]))
            s = 1.0 / math.sqrt(din)
            params[param_name(k, path, "a")] = rng.uniform(-s, s, (din, rank)).astype(np.float32)
            params[param_name(k, path, "b")] = np.zeros((rank, dout), np.float32)
    return params


def n_param_bytes(shapes: dict[str, tuple[int, int]], n_layers: int, rank: int) -> int:
    return n_layers * sum(rank * (i + o) for i, o in shapes.values()) * 4


def by_layer(params: dict, scale: float) -> dict[int, dict[str, tuple]]:
    """Flat dict -> {layer: {path: (A, B, scale)}}, the shape adapters use."""
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


def save_adapter_dir(d: Path, params: dict, cfg: LoraConfig, base: str,
                     n_layers: int, shapes, extra: dict | None = None) -> None:
    """Write both layouts atomically enough for a crash not to leave a half adapter:
    tensors first, config last (readers key off the config). `params` may hold numpy, torch
    or mlx arrays."""
    from safetensors.numpy import save_file
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    flat = {k: to_numpy_f32(v) for k, v in params.items()}
    save_file({f"model.{k}": v for k, v in flat.items()}, str(d / "adapters.tmp.safetensors"))
    peft = {}
    for name, arr in flat.items():
        m = _KEY.match(name)
        k, path, ab = m.group(1), m.group(2), m.group(3)
        peft[f"base_model.model.model.layers.{k}.{path}.lora_{ab.upper()}.weight"] = \
            np.ascontiguousarray(arr.T)
    save_file(peft, str(d / "adapter_model.tmp.safetensors"))
    (d / "adapters.tmp.safetensors").replace(d / "adapters.safetensors")
    (d / "adapter_model.tmp.safetensors").replace(d / "adapter_model.safetensors")
    (d / "adapter_config.json").write_text(json.dumps(
        _union_config(cfg, base, n_layers, shapes, extra or {}), indent=2))


def read_adapter_params(d: Path) -> dict[str, np.ndarray]:
    """Flat float32 numpy params from an adapter directory's mlx-lm layout (our canonical
    form)."""
    from safetensors.numpy import load_file
    d = Path(d)
    f = d / "adapters.safetensors"
    if not f.exists():
        raise SpillError(f"{d} has no adapters.safetensors")
    return {(n[len("model."):] if n.startswith("model.") else n): a.astype(np.float32)
            for n, a in load_file(str(f)).items()}


def config_from_dir(d: Path) -> tuple[LoraConfig, dict]:
    raw = json.loads((Path(d) / "adapter_config.json").read_text())
    lp = raw.get("lora_parameters", {})
    cfg = LoraConfig(rank=int(raw.get("r", lp.get("rank", 16))),
                     alpha=float(raw.get("lora_alpha", 32.0)),
                     dropout=float(raw.get("lora_dropout", lp.get("dropout", 0.0))),
                     targets=lp.get("keys"))
    return cfg, raw


def cosine_np(a, b) -> float:
    a = to_numpy_f32(a).reshape(-1).astype(np.float64)
    b = to_numpy_f32(b).reshape(-1).astype(np.float64)
    den = math.sqrt(float(a @ a)) * math.sqrt(float(b @ b))
    return float(a @ b) / max(den, 1e-30)
