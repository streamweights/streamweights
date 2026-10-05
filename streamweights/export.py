"""spill export <base>+<adapter>: merge the adapter into the base and write the result.

  merged safetensors   W' = W + scale * (A @ B)^T per targeted linear, computed in float32
                       and written back in the base's dtype, shard by shard (so the memory
                       needed is one shard, not the model); config, tokenizer and index
                       files are copied beside them
  --gguf [bf16|q8_0|q4_k_m]
                       llama.cpp's own converter, downloaded at the release tag of the
                       llama.cpp binaries we ship (never vendored), run in a private
                       venv; q4_k_m is a bf16 conversion followed by llama-quantize.
                       An Ollama Modelfile is written beside the GGUF.
  --ollama             `ollama create <name>` when ollama is installed, and the run line
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .errors import SpillError
from .registry import REPO_ROOT

GGUF_TYPES = ("bf16", "q8_0", "q4_k_m")
EXPORTS_DIR = REPO_ROOT / "exports"
TOOLS_DIR = REPO_ROOT / "tools"
LLAMACPP_RELEASE = "b11311"
SOURCE_URL = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/{rel}.tar.gz"
CONVERT_FILES = ("convert_hf_to_gguf.py", "conversion", "gguf-py")   # what the converter imports
CONVERT_DEPS = ["torch", "numpy", "sentencepiece", "transformers", "protobuf", "safetensors"]

_WEIGHT = re.compile(r"^model\.layers\.(\d+)\.(.+)\.weight$")
COPY_SUFFIXES = (".json", ".model", ".txt", ".tiktoken", ".jinja")


def export_name(base: str, adapter_id: str) -> str:
    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")
    return f"{safe(base)}+{safe(Path(adapter_id).name)}"


# ------------------------------------------------------------ merge

def merge_adapter(model_dir: Path, adapter, out_dir: Path, say=lambda s: None) -> dict:
    """Write the merged model to out_dir. Returns {"modules": n, "shards": n, "dtype": str}."""
    import mlx.core as mx
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    if cfg.get("quantization"):
        raise SpillError("cannot merge an adapter into a quantized base; export from the bf16 "
                         "safetensors (the default)", "spill export <base>+<adapter>")
    adapter.validate(cfg)
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise SpillError(f"{model_dir} has no .safetensors files")
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = sum(len(m) for m in adapter.layers.values())
    merged = 0
    dtype = None
    for i, shard in enumerate(shards):
        tensors = mx.load(str(shard))
        out = {}
        for name, w in tensors.items():
            m = _WEIGHT.match(name)
            if m:
                layer, path = int(m.group(1)), m.group(2)
                ab = adapter.layers.get(layer, {}).get(path)
                if ab is not None:
                    a, b, scale = ab
                    delta = (a.astype(mx.float32) @ b.astype(mx.float32)) * scale   # [in, out]
                    if delta.T.shape != w.shape:
                        raise SpillError(
                            f"adapter {adapter.id} layer {layer} {path} makes a {tuple(delta.T.shape)} "
                            f"update for a {tuple(w.shape)} weight; it was trained on a "
                            f"different base")
                    dtype = w.dtype
                    w = (w.astype(mx.float32) + delta.T).astype(w.dtype)
                    merged += 1
            out[name] = w
        mx.eval(list(out.values()))
        mx.save_safetensors(str(out_dir / shard.name), out, metadata={"format": "pt"})
        say(f"merged shard {i + 1}/{len(shards)}: {shard.name}")
        del tensors, out
        mx.clear_cache()
    if merged != expected:
        raise SpillError(f"adapter {adapter.id} has {expected} modules but only {merged} matched "
                         f"a weight in the base; it was trained on a different base or "
                         f"architecture")
    for f in model_dir.iterdir():
        if f.is_file() and f.suffix in COPY_SUFFIXES and f.name != "adapter_config.json":
            shutil.copyfile(f, out_dir / f.name)
    (out_dir / "merge.json").write_text(json.dumps({
        "adapter": adapter.id, "adapter_hash": adapter.hash, "rank": adapter.rank,
        "modules": merged, "base_dir": str(model_dir), "dtype": str(dtype)}, indent=2))
    return {"modules": merged, "shards": len(shards), "dtype": str(dtype)}


# ------------------------------------------------------------ gguf

def ensure_converter(say=lambda s: None, runner=subprocess.run) -> tuple[Path, Path]:
    """(python, convert_hf_to_gguf.py): a private venv with the converter's requirements.
    Downloaded once, reused after."""
    tools = TOOLS_DIR
    venv = tools / "convert-venv"
    py = venv / "bin" / "python"
    src = tools / f"llama.cpp-convert-{LLAMACPP_RELEASE}"
    script = src / "convert_hf_to_gguf.py"
    if script.exists() and py.exists():
        return py, script
    tools.mkdir(parents=True, exist_ok=True)
    if not script.exists():
        say(f"downloading llama.cpp {LLAMACPP_RELEASE} converter")
        _fetch_converter_source(src)
    if not py.exists():
        say("creating the converter environment (torch, transformers, gguf; once, ~1 GB)")
        uv = shutil.which("uv")
        if uv:
            runner([uv, "venv", "--python", "3.12", str(venv)], check=True, capture_output=True)
            runner([uv, "pip", "install", "--python", str(py), *CONVERT_DEPS,
                    str(script.parent / "gguf-py")], check=True)
        else:
            runner([sys.executable, "-m", "venv", str(venv)], check=True)
            runner([str(py), "-m", "pip", "install", *CONVERT_DEPS,
                    str(script.parent / "gguf-py")], check=True)
    return py, script


def _fetch_converter_source(dest: Path) -> None:
    """The converter script plus the packages it imports, from the release's source archive."""
    import io
    import tarfile
    import urllib.request
    with urllib.request.urlopen(SOURCE_URL.format(rel=LLAMACPP_RELEASE), timeout=120) as r:
        data = r.read()
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            rel = parts[1] if len(parts) > 1 else ""
            if rel and rel.split("/")[0] in CONVERT_FILES and (m.isfile() or m.isdir()):
                m.name = rel
                tf.extract(m, tmp, filter="data")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)


def llama_quantize_bin() -> Path:
    from .llamacpp import BIN_DIR, ensure_llama_server
    ensure_llama_server()
    for cand in sorted(BIN_DIR.glob("llama-*/llama-quantize")) + [BIN_DIR / "llama-quantize"]:
        if cand.exists():
            return cand
    raise SpillError("llama-quantize not found in the llama.cpp release; q4_k_m needs it",
                     "spill export <base>+<adapter> --gguf q8_0")


def convert_gguf(merged_dir: Path, name: str, kind: str, say=lambda s: None,
                 runner=subprocess.run) -> Path:
    kind = kind.lower()
    if kind not in GGUF_TYPES:
        raise SpillError(f"--gguf must be one of {', '.join(GGUF_TYPES)}, not {kind}",
                         f"spill export ... --gguf q8_0")
    py, script = ensure_converter(say, runner)
    out = Path(merged_dir) / f"{name}-{kind}.gguf"
    direct = "bf16" if kind in ("bf16", "q4_k_m") else "q8_0"
    first = out if kind != "q4_k_m" else Path(merged_dir) / f"{name}-bf16.gguf"
    say(f"converting to GGUF {direct} with llama.cpp's converter")
    r = runner([str(py), str(script), str(merged_dir), "--outfile", str(first),
                "--outtype", direct], capture_output=True, text=True)
    if r.returncode != 0:
        raise SpillError("llama.cpp's converter failed: "
                         + (r.stderr or r.stdout or "").strip().splitlines()[-1][:200],
                         "spill export ... --debug")
    if kind == "q4_k_m":
        say("quantizing to Q4_K_M with llama-quantize")
        r = runner([str(llama_quantize_bin()), str(first), str(out), "Q4_K_M"],
                   capture_output=True, text=True)
        if r.returncode != 0:
            raise SpillError("llama-quantize failed: "
                             + (r.stderr or r.stdout or "").strip().splitlines()[-1][:200])
        if first.exists() and first != out:
            first.unlink()
    return out


def write_modelfile(gguf: Path, merged_dir: Path) -> Path:
    """An Ollama Modelfile beside the GGUF. The chat template is taken from the tokenizer
    config (Go-template translation is Ollama's job for known families via `FROM`'s
    metadata; the GGUF embeds the Jinja template, which current Ollama reads)."""
    mf = Path(gguf).with_name("Modelfile")
    mf.write_text(f"FROM ./{Path(gguf).name}\n")
    return mf


def ollama_create(name: str, modelfile: Path, runner=subprocess.run) -> str | None:
    """`ollama create <name> -f Modelfile`; returns the `ollama run` line, or None if
    ollama is not installed."""
    if not shutil.which("ollama"):
        return None
    r = runner(["ollama", "create", name, "-f", str(modelfile)], capture_output=True, text=True,
               cwd=str(Path(modelfile).parent))
    if r.returncode != 0:
        raise SpillError("ollama create failed: " + (r.stderr or r.stdout).strip()[-200:])
    return f"ollama run {name}"
