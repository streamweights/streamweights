"""Small shared helpers: canonical JSON, hashing, atomic file writes, text normalization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import uuid
from pathlib import Path


def canon(obj) -> str:
    """Canonical JSON: sorted keys, no whitespace, non-ASCII kept. The input of every fingerprint."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_obj(obj) -> str:
    return sha_bytes(canon(obj).encode())


def sha_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fsync_dir(path: str | Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(path: str | Path, data: bytes) -> None:
    """Temp file in the same directory, fsync, rename, fsync the directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def write_json(path: str | Path, obj, indent: int | None = 2) -> None:
    atomic_write(path, (json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False)
                        + "\n").encode())


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_jsonl(path: str | Path, rows) -> None:
    atomic_write(path, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                               for r in rows).encode())


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


_WS = re.compile(r"\s+")


def norm_dup(text: str) -> str:
    """Duplicate normalization (recorded in the project config): Unicode NFKC, casefold,
    whitespace runs collapsed to one space, ends stripped."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def norm_label(text: str) -> str:
    """Label normalization: the same rules as duplicates (NFKC, casefold, collapse, strip)."""
    return norm_dup(text)


def norm_group(text: str) -> str:
    return norm_dup(text)


def now() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def new_id(prefix: str = "") -> str:
    import time
    return f"{prefix}{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
