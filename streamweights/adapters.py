"""LoRA adapters at inference: PEFT and mlx-lm layouts, applied at the layer boundary.

The engines bind one transformer layer's base weights into a single reusable
TransformerBlock (from NVMe in the streamed engine, from memory in the resident
one). Adapter tensors are small and stay resident; after a layer's base weights
are bound, `attach(block, k)` points each targeted linear at that layer's
(A, B, scale). The linear's forward is then

    y = base(x) + (scale * ((x @ A) @ B)).astype(x.dtype)

which is exactly mlx-lm's LoRALinear. The base weights are never merged or
modified, so the same streamed bytes serve any adapter.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path

from .errors import SpillError
from .registry import REPO_ROOT

ADAPTERS_DIR = REPO_ROOT / "adapters"

_MLX_KEY = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)\.lora_([ab])$")
_PEFT_KEY = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+?)\.lora_([AB])(?:\.[A-Za-z0-9_]+)?\.weight$")


class _Slot:
    """Plain holder (not an array/dict/list, so nn.Module keeps it out of the
    parameter tree and the base weight binding never sees it)."""
    __slots__ = ("a", "b", "scale", "p", "key")

    def __init__(self, a, b, scale, p=0.0, key=None):
        # p/key: training-only input dropout on the LoRA branch (explicit key so a
        # recomputed forward draws the same mask as the original one)
        self.a, self.b, self.scale, self.p, self.key = a, b, scale, p, key


_CLASSES: dict = {}


def _lora_class(cls):
    import mlx.core as mx
    if cls not in _CLASSES:
        class _Lora(cls):  # type: ignore[misc, valid-type]
            def __call__(self, x):
                y = cls.__call__(self, x)
                slot = self.__dict__.get("_lora")
                if slot is None:
                    return y
                xi = x
                if slot.p and slot.key is not None:
                    keep = mx.random.bernoulli(1.0 - slot.p, x.shape, key=slot.key)
                    xi = (1.0 / (1.0 - slot.p)) * keep * x
                z = (xi @ slot.a) @ slot.b
                return y + (slot.scale * z).astype(x.dtype)
        _Lora.__name__ = f"Lora{cls.__name__}"
        _CLASSES[cls] = _Lora
    return _CLASSES[cls]


def _resolve(block, path: str):
    return reduce(getattr, path.split("."), block)


@dataclass
class LoraAdapter:
    id: str
    path: Path
    layout: str                     # "mlx-lm" | "peft"
    rank: int
    hash: str                       # sha256 over weights file + adapter_config.json
    base: str | None = None         # base model the adapter declares, if any
    layers: dict = field(default_factory=dict)   # layer -> {module path: (A, B, scale)}
    targets: set = field(default_factory=set)

    def info(self) -> dict:
        return {"id": self.id, "hash": self.hash, "layout": self.layout,
                "rank": self.rank, "base": self.base,
                "layers": sorted(self.layers), "modules": sorted(self.targets)}

    def validate(self, config: dict) -> None:
        n = config["num_hidden_layers"]
        bad = [k for k in self.layers if k >= n]
        if bad:
            raise SpillError(f"adapter {self.id} has layer {max(bad)} but the base model has "
                             f"only {n} layers; it was trained on a different base")
        hidden = config["hidden_size"]
        for k, mods in self.layers.items():
            for path, (a, b, _) in mods.items():
                if path.endswith(("q_proj", "gate_proj", "up_proj")) and a.shape[0] != hidden:
                    raise SpillError(
                        f"adapter {self.id} layer {k} {path} expects input width "
                        f"{a.shape[0]} but the base model's hidden size is {hidden}; "
                        f"it was trained on a different base")

    def prepare(self, block) -> None:
        """One-time: make every targeted linear LoRA-aware (passthrough until attached)."""
        for path in sorted(self.targets):
            try:
                mod = _resolve(block, path)
            except AttributeError:
                raise SpillError(f"adapter {self.id} targets module {path}, which this "
                                 f"architecture's block does not have")
            if "_lora" in mod.__dict__:
                continue
            mod.__class__ = _lora_class(type(mod))
            mod.__dict__["_lora"] = None

    def attach(self, block, k: int) -> None:
        mods = self.layers.get(k, {})
        for path in self.targets:
            mod = _resolve(block, path)
            m = mods.get(path)
            mod.__dict__["_lora"] = _Slot(*m) if m else None


# ------------------------------------------------------------ loading

def _sha(files: list[Path]) -> str:
    import hashlib
    h = hashlib.sha256()
    for f in files:
        with open(f, "rb") as fh:
            while True:
                b = fh.read(8 << 20)
                if not b:
                    break
                h.update(b)
    return h.hexdigest()


def detect_layout(d: Path) -> str | None:
    if (d / "adapters.safetensors").exists():
        return "mlx-lm"
    if (d / "adapter_model.safetensors").exists():
        return "peft"
    return None


def load_adapter_dir(d: Path, adapter_id: str | None = None) -> LoraAdapter:
    from .platforms import mlx_available
    mx = None
    if mlx_available():
        import mlx.core as mx
    d = Path(d)
    layout = detect_layout(d)
    if layout is None:
        raise SpillError(f"{d} is not a LoRA adapter directory: expected adapters.safetensors "
                         f"(mlx-lm) or adapter_model.safetensors (PEFT) plus adapter_config.json",
                         "spill adapters")
    cfg_path = d / "adapter_config.json"
    if not cfg_path.exists():
        raise SpillError(f"{d} has no adapter_config.json (needed for rank and scale)")
    cfg = json.loads(cfg_path.read_text())
    wfile = d / ("adapters.safetensors" if layout == "mlx-lm" else "adapter_model.safetensors")

    if layout == "mlx-lm":
        ft = cfg.get("fine_tune_type", "lora")
        if ft != "lora":
            raise SpillError(f"adapter {d.name}: fine_tune_type '{ft}' is not supported; "
                             f"LoRA only (DoRA and full fine-tunes are not applied at inference)")
        lp = cfg.get("lora_parameters", {})
        scale = float(lp.get("scale", 20.0))
        rank = int(lp.get("rank", 8))
        base = cfg.get("model")
        pat = _MLX_KEY
    else:
        if str(cfg.get("peft_type", "LORA")).upper() != "LORA" or cfg.get("use_dora"):
            raise SpillError(f"adapter {d.name}: only plain LoRA is supported "
                             f"(peft_type={cfg.get('peft_type')}, use_dora={cfg.get('use_dora')})")
        rank = int(cfg["r"])
        alpha = float(cfg.get("lora_alpha", rank))
        scale = alpha / (rank ** 0.5 if cfg.get("use_rslora") else rank)
        base = cfg.get("base_model_name_or_path")
        pat = _PEFT_KEY

    if mx is not None:
        raw = mx.load(str(wfile))
    else:                              # export on a platform without MLX: numpy float32
        from .safetensors_np import load_f32
        raw = load_f32(wfile)
    pairs: dict = {}
    for name, arr in raw.items():
        m = pat.search(name)
        if not m:
            continue
        layer, path, ab = int(m.group(1)), m.group(2), m.group(3).lower()
        pairs.setdefault((layer, path), {})[ab] = arr
    if not pairs:
        raise SpillError(f"{wfile.name} has no LoRA tensors this loader recognizes "
                         f"(layout {layout})")
    layers: dict = {}
    targets: set = set()
    for (layer, path), ab in pairs.items():
        if set(ab) != {"a", "b"}:
            raise SpillError(f"adapter {d.name}: layer {layer} {path} has only one of A/B")
        a, b = ab["a"], ab["b"]
        if layout == "peft":       # PEFT stores A [r, in], B [out, r]; we use [in, r], [r, out]
            a, b = a.T, b.T
        if a.shape[1] != b.shape[0]:
            raise SpillError(f"adapter {d.name}: layer {layer} {path} rank mismatch "
                             f"{tuple(a.shape)} vs {tuple(b.shape)}")
        layers.setdefault(layer, {})[path] = (a, b, scale)
        targets.add(path)
    if mx is not None:
        mx.eval([t for mods in layers.values() for m in mods.values() for t in m[:2]])
    return LoraAdapter(adapter_id or d.name, d, layout, rank, _sha([wfile, cfg_path]),
                       base, layers, targets)


def resolve_adapter(spec: str) -> LoraAdapter:
    """Local directory, a name under <data>/adapters/, or a Hugging Face repo id."""
    p = Path(spec).expanduser()
    # a directory only wins when it is an adapter: `spill build my-folder` names its adapter
    # after the folder, and the folder itself must not shadow it
    if p.is_dir() and (detect_layout(p) or not (ADAPTERS_DIR / spec).is_dir()):
        return load_adapter_dir(p, spec)
    local = ADAPTERS_DIR / spec
    if local.is_dir():
        return load_adapter_dir(local, spec)
    if spec.startswith((".", "/", "~")):
        raise SpillError(f"adapter directory {spec} does not exist", "spill adapters")
    if re.fullmatch(r"[\w.-]+/[\w.-]+", spec):
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import RepositoryNotFoundError
        try:
            d = snapshot_download(spec, allow_patterns=[
                "adapter_model.safetensors", "adapters.safetensors", "adapter_config.json"])
        except RepositoryNotFoundError:
            raise SpillError(f"no adapter directory or Hugging Face repo named '{spec}'",
                             "spill adapters")
        return load_adapter_dir(Path(d), spec)
    raise SpillError(f"adapter '{spec}' is not a directory, not under {ADAPTERS_DIR}, and not "
                     f"a Hugging Face repo id (org/name)", "spill adapters")


def split_model_adapter(arg: str) -> tuple[str, str | None]:
    """'llama3.3:70b+./my-adapter' -> ('llama3.3:70b', './my-adapter')."""
    base, plus, adapter = arg.partition("+")
    return (base, adapter) if plus and adapter else (arg, None)


def list_local_adapters() -> list[dict]:
    out = []
    if ADAPTERS_DIR.exists():
        for d in sorted(ADAPTERS_DIR.iterdir()):
            layout = detect_layout(d) if d.is_dir() else None
            if not layout:
                continue
            cfg = json.loads((d / "adapter_config.json").read_text()) \
                if (d / "adapter_config.json").exists() else {}
            wfile = d / ("adapters.safetensors" if layout == "mlx-lm"
                         else "adapter_model.safetensors")
            rank = (cfg.get("lora_parameters", {}).get("rank") if layout == "mlx-lm"
                    else cfg.get("r"))
            out.append({"id": d.name, "layout": layout, "rank": rank,
                        "base": cfg.get("model") or cfg.get("base_model_name_or_path"),
                        "bytes": wfile.stat().st_size, "path": str(d)})
    return out
