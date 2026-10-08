"""Model identity: the exact bytes a run was built on.

A curated tag (qwen2.5:0.5b) is not an identity: it resolves to a repository, a pinned
revision and the sha256 of every file the workflow reads (models.yaml `pins`). A copy on disk
is verified against those hashes, so a changed file is caught offline; a missing copy is
re-acquired at the pinned revision. A Hugging Face repo id or a local directory has no pin:
its identity is the hashes of its files."""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

from ..errors import SpillError
from ..registry import MODELS_DIR, load_pins, safetensors_dir, safetensors_spec
from .common import sha_file

CORE_SUFFIXES = (".safetensors", ".json", ".txt", ".model")
EMBEDDING_TAG = "embedding:minilm"


def _cached_sha(path: Path) -> str:
    cache = path.parent / ".spill-sha-cache.json"
    try:
        data = json.loads(cache.read_text()) if cache.exists() else {}
    except (json.JSONDecodeError, OSError):
        data = {}
    st = path.stat()
    key = f"{st.st_size}:{st.st_mtime_ns}"
    ent = data.get(path.name)
    if ent and ent["key"] == key:
        return ent["sha256"]
    sha = sha_file(path)
    data[path.name] = {"key": key, "sha256": sha}
    try:
        cache.write_text(json.dumps(data))
    except OSError:
        pass
    return sha


def hash_dir(d: Path, names=None) -> dict:
    out = {}
    for p in sorted(Path(d).iterdir()):
        if not p.is_file() or p.name.startswith(".") or p.suffix not in CORE_SUFFIXES:
            continue
        if names is not None and p.name not in names:
            continue
        out[p.name] = {"sha256": _cached_sha(p), "bytes": p.stat().st_size}
    return out


def _pinned_files(pin: dict) -> dict:
    return pin["files"]


def verify_dir(tag: str, d: Path) -> dict:
    """Check a directory against a tag's pin; returns the file hashes that matched. Missing
    optional files are fine; a missing or different required file is not."""
    pin = load_pins()[tag]
    got = {}
    for name, meta in _pinned_files(pin).items():
        p = Path(d) / name
        if not p.exists():
            if name in ("model.safetensors", "config.json", "tokenizer.json"):
                raise SpillError(f"{tag}: {name} is missing from {d}; re-fetch the pinned "
                                 f"revision {pin['revision'][:12]}", f"spill models --fetch {tag}")
            continue
        if p.stat().st_size != meta["bytes"] or _cached_sha(p) != meta["sha256"]:
            raise SpillError(f"{tag}: {name} in {d} does not match the pinned revision "
                             f"{pin['revision'][:12]} (hash differs)",
                             f"remove {d} and run the command again to re-fetch it")
        got[name] = {"sha256": meta["sha256"], "bytes": meta["bytes"]}
    return got


def student_identity(model: str, fetch: bool = True) -> dict:
    """The identity record of a student or teacher: curated tag, HF repo id or directory."""
    pins = load_pins()
    if model in pins:
        spec = safetensors_spec(model)
        d = safetensors_dir(model)
        if not (d / "model.safetensors").exists():
            if not fetch:
                return {"tag": model, "repo": pins[model]["repo"], "revision": pins[model]["revision"],
                        "format": pins[model]["format"], "files": pins[model]["files"],
                        "present": False, "dir": str(d)}
            from ..registry import download_safetensors
            download_safetensors(model)
        files = verify_dir(model, d)
        return {"tag": model, "repo": pins[model]["repo"], "revision": pins[model]["revision"],
                "format": pins[model]["format"], "license": pins[model].get("license"),
                "files": files, "present": True, "dir": str(d)}
    p = Path(model).expanduser()
    if p.is_dir() and (p / "config.json").exists():
        return {"tag": str(p), "repo": None, "revision": None, "format": "safetensors (local directory)",
                "files": hash_dir(p), "present": True, "dir": str(p.resolve())}
    from ..resolve import download_hf, resolve_model
    res = resolve_model(model)
    d = download_hf(res) if fetch else res.local_dir
    return {"tag": model, "repo": res.name, "revision": res.revision or "unpinned",
            "format": "safetensors", "files": hash_dir(d) if Path(d).exists() else {},
            "present": Path(d).exists(), "dir": str(d)}


def embedding_dir() -> Path:
    return MODELS_DIR / "embedding-minilm"


def embedding_identity(fetch: bool = True) -> dict:
    pin = load_pins()[EMBEDDING_TAG]
    d = embedding_dir()
    if not (d / "model.safetensors").exists():
        if not fetch:
            return {"tag": EMBEDDING_TAG, "repo": pin["repo"], "revision": pin["revision"],
                    "format": pin["format"], "files": pin["files"], "present": False, "dir": str(d)}
        ensure_embedding()
    files = verify_dir(EMBEDDING_TAG, d)
    return {"tag": EMBEDDING_TAG, "repo": pin["repo"], "revision": pin["revision"],
            "format": pin["format"], "license": pin.get("license"), "files": files,
            "present": True, "dir": str(d)}


def ensure_embedding(say=lambda s: None) -> Path:
    """The pinned embedding model (92 MB), fetched at its revision if absent."""
    from .. import guard
    pin = load_pins()[EMBEDDING_TAG]
    d = embedding_dir()
    if (d / "model.safetensors").exists():
        verify_dir(EMBEDDING_TAG, d)
        return d
    guard.check(pin["repo"])
    from huggingface_hub import snapshot_download
    say(f"fetching {pin['repo']}@{pin['revision'][:12]} "
        f"({sum(m['bytes'] for m in pin['files'].values()) / 1e6:.0f} MB) -> {d}")
    snapshot_download(pin["repo"], revision=pin["revision"], local_dir=d,
                      allow_patterns=list(pin["files"]))
    verify_dir(EMBEDDING_TAG, d)
    return d


def identity_fingerprint(ident: dict | None) -> dict | None:
    """What goes into a run's identity hash: revision and file hashes, never a path."""
    if ident is None:
        return None
    return {"tag": ident["tag"], "repo": ident.get("repo"), "revision": ident.get("revision"),
            "files": {k: v["sha256"] for k, v in sorted(ident["files"].items())}}


def dependency_versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("streamweights", "torch", "transformers", "peft", "safetensors", "numpy",
                "mlx", "mlx-lm", "jsonschema", "huggingface-hub", "tokenizers", "boto3", "s3fs"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out
