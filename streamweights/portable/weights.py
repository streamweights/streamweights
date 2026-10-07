"""Weights cache and staging. Weights live per machine under SPILL_HOME/models; with
`--weights <uri>` a machine stages a model directory from shared storage (s3://, gs://, az://,
a path) instead of Hugging Face. Staging is a plain copy of the safetensors shards and the
small config and tokenizer files; a file whose size already matches is not copied again, so a
second job on the same machine pays nothing."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import time
from pathlib import Path

from ..errors import SpillError
from ..registry import MODELS_DIR
from .store import Store

PATTERNS = ("*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken", "*.jinja")


def staged_dir(uri: str) -> Path:
    key = hashlib.sha256(uri.rstrip("/").encode()).hexdigest()[:12]
    leaf = uri.rstrip("/").rsplit("/", 1)[-1] or "weights"
    return MODELS_DIR / "staged" / f"{leaf}-{key}"


def stage_weights(uri: str, note=None) -> tuple[Path, dict]:
    """Copy a model directory from `uri` into the per-machine cache. Returns (local directory,
    {"bytes": copied, "seconds": s, "files": n, "reused": n, "total_bytes": all})."""
    store = Store(uri)
    names = [n for n in store.ls() if any(fnmatch.fnmatch(n, pat) for pat in PATTERNS)]
    if "config.json" not in names or not any(n.endswith(".safetensors") for n in names):
        raise SpillError(f"{uri} is not a safetensors model directory (needs config.json and "
                         f"*.safetensors)", "spill tune --help")
    dest = staged_dir(uri)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    copied = reused = total = 0
    todo = []
    for n in names:
        size = store.size(n)
        total += size
        f = dest / n
        if f.exists() and f.stat().st_size == size:
            reused += 1
        else:
            todo.append((n, size))
    need = sum(s for _, s in todo)
    if todo and note:
        note(f"staging {need / 1e9:.2f} GB of weights from {uri} into {dest}")
    for n, size in todo:
        copied += store.copy_to_local(n, dest / n)
    secs = time.monotonic() - t0
    info = {"bytes": copied, "seconds": round(secs, 3), "files": len(todo), "reused": reused,
            "total_bytes": total, "uri": uri, "dir": str(dest)}
    (dest / ".staged.json").write_text(json.dumps(info))
    return dest, info
