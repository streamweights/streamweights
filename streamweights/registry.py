"""Model registry: models.yaml + download policy + resident-bytes estimates."""

from __future__ import annotations

import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from .errors import SpillError

def _home() -> Path:
    """Data root: the git checkout when running from one, else ~/.streamweights
    (installed packages must not write into site-packages). SPILL_HOME overrides."""
    import os
    env = os.environ.get("SPILL_HOME")
    if env:
        return Path(env).expanduser()
    cand = Path(__file__).resolve().parent.parent
    if (cand / "pyproject.toml").exists():
        return cand
    return Path.home() / ".streamweights"


REPO_ROOT = _home()
REPO_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_YAML = Path(__file__).resolve().parent / "models.yaml"
MODELS_DIR = REPO_ROOT / "models"

GIB = 1024**3
import os as _os  # noqa: E402

MIN_FREE_AFTER_DOWNLOAD = int(float(_os.environ.get("SPILL_MIN_FREE_GB", 20)) * GIB)   # CI lowers it

KV_BYTES_PER_ELT = 2  # f16 KV cache


@dataclass
class Quant:
    name: str
    repo: str | None
    convert_from: str | None
    files: list[str]
    bytes: int

    def local_paths(self, model: str) -> list[Path]:
        d = MODELS_DIR / model.replace(":", "-") / self.name
        if self.files:
            return [d / Path(f).name for f in self.files]
        # converted-locally quants: any gguf in the dir
        return sorted(d.glob("*.gguf"))

    def downloaded(self, model: str) -> bool:
        paths = self.local_paths(model)
        return bool(paths) and all(p.exists() for p in paths)


@dataclass
class Model:
    name: str
    arch: dict
    quants: dict[str, Quant]

    def kv_bytes_per_token(self) -> int:
        a = self.arch
        return 2 * a["n_layers"] * a["n_kv_heads"] * a["head_dim"] * KV_BYTES_PER_ELT

    def resident_bytes(self, quant: str, ctx: int, n_parallel: int = 1) -> int:
        """Weights + KV estimate. Uses the local GGUF header when present."""
        q = self.quants[quant]
        arch = self.arch
        if q.downloaded(self.name):
            hdr = read_gguf_arch(q.local_paths(self.name)[0])
            if hdr:
                arch = hdr
        kv = 2 * arch["n_layers"] * arch["n_kv_heads"] * arch["head_dim"] * KV_BYTES_PER_ELT
        return q.bytes + kv * ctx * n_parallel


def read_gguf_arch(path: Path) -> dict | None:
    """Minimal GGUF v3 metadata reader: block_count, head_count_kv, head dim."""
    TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

            def rstr():
                (n,) = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "replace")

            def rval(t):
                if t == 8:
                    return rstr()
                if t == 9:
                    (et,) = struct.unpack("<I", f.read(4))
                    (n,) = struct.unpack("<Q", f.read(8))
                    return [rval(et) for _ in range(n)]
                fmt = TYPES[t]
                return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

            meta = {}
            for _ in range(n_kv):
                key = rstr()
                (t,) = struct.unpack("<I", f.read(4))
                v = rval(t)
                if any(key.endswith(s) for s in (".block_count", ".head_count", ".head_count_kv",
                                                 ".embedding_length", ".key_length")):
                    meta[key] = v
            def find(suffix):
                for k, v in meta.items():
                    if k.endswith(suffix):
                        return v
                return None
            layers, kv_heads = find(".block_count"), find(".head_count_kv")
            heads, emb = find(".head_count"), find(".embedding_length")
            head_dim = find(".key_length") or (emb // heads if heads and emb else None)
            if layers and kv_heads and head_dim:
                if isinstance(kv_heads, list):
                    kv_heads = max(kv_heads)
                return {"n_layers": layers, "n_kv_heads": kv_heads, "head_dim": head_dim}
    except (OSError, struct.error, KeyError):
        pass
    return None


def load_registry() -> dict[str, Model]:
    raw = yaml.safe_load(MODELS_YAML.read_text())
    out = {}
    for name, m in raw["models"].items():
        quants = {
            qn: Quant(qn, q.get("repo"), q.get("convert_from"), q.get("files", []), q["bytes"])
            for qn, q in m["quants"].items()
        }
        out[name] = Model(name, m["arch"], quants)
    return out


def download(model: Model, quant_name: str, progress: bool = True) -> list[Path]:
    """Download (or convert) a quant. Enforces the disk policy: print size and
    destination, never start a download that would leave under 20 GB free."""
    q = model.quants[quant_name]
    dest = MODELS_DIR / model.name.replace(":", "-") / quant_name
    if q.downloaded(model.name):
        return q.local_paths(model.name)
    free = shutil.disk_usage(REPO_ROOT).free
    if free - q.bytes < MIN_FREE_AFTER_DOWNLOAD:
        raise SpillError(
            f"refusing download: {q.bytes / GIB:.1f} GB to {dest} would leave "
            f"{(free - q.bytes) / GIB:.1f} GB free (20 GB floor)", "spill doctor")
    print(f"downloading {model.name} {quant_name}: {q.bytes / GIB:.1f} GB -> {dest}", file=sys.stderr)
    dest.mkdir(parents=True, exist_ok=True)
    if q.repo:
        from huggingface_hub import hf_hub_download
        paths = []
        for f in q.files:
            p = hf_hub_download(q.repo, f, local_dir=dest / "_hf")
            target = dest / Path(f).name
            if not target.exists():
                target.hardlink_to(p)
            paths.append(target)
        return paths
    raise SpillError(
        f"{model.name} has no published bf16 GGUF (convert from {q.convert_from} with "
        f"llama.cpp's convert_hf_to_gguf.py --outtype bf16 into {dest})",
        f"spill run {model.name} sample --quant Q8_0")


if __name__ == "__main__":
    reg = load_registry()
    for name, m in reg.items():
        print(name)
        for qn, q in m.quants.items():
            print(f"  {qn:7s} {q.bytes / GIB:6.1f} GB  downloaded={q.downloaded(name)}  "
                  f"resident@4k={m.resident_bytes(qn, 4096) / GIB:.1f} GB  "
                  f"src={q.repo or ('convert:' + q.convert_from)}")


# ---------- Phase 1: safetensors + mlx quants ----------

def load_extras() -> tuple[dict, dict]:
    raw = yaml.safe_load(MODELS_YAML.read_text())
    return raw.get("safetensors", {}), raw.get("mlx_quants", {})


def safetensors_spec(model_name: str) -> dict | None:
    st, _ = load_extras()
    return st.get(model_name)


def mlx_quant_repo(model_name: str, bits: str) -> str | None:
    _, mq = load_extras()
    return (mq.get(model_name) or {}).get(bits)


def safetensors_dir(model_name: str) -> Path:
    spec = safetensors_spec(model_name)
    return MODELS_DIR / model_name.replace(":", "-") / spec["dir"]


def safetensors_downloaded(model_name: str) -> bool:
    d = safetensors_dir(model_name)
    return d.exists() and any(d.glob("*.safetensors")) and (d / "config.json").exists()


def download_safetensors(model_name: str) -> Path:
    """bf16 safetensors with the 20 GB floor checked before the first byte, one progress
    line with speed and ETA, and resume after an interruption."""
    spec = safetensors_spec(model_name)
    dest = safetensors_dir(model_name)
    if safetensors_downloaded(model_name):
        return dest
    from .download import fetch
    return fetch(spec["repo"], dest, ["*.safetensors", "*.json"],
                 f"{model_name} bf16 safetensors",
                 recovery=f"spill run {model_name} <input> --quant 8bit")
