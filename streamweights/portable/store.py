"""Storage for portable job state: a local path, or s3://, gs://, az://, memory:// through
fsspec. One small interface; the commit protocol lives in checkpoint.py and rows.py.

Local writes are atomic (temp file, then rename). Object stores publish a whole object on a
single put, and multi-file checkpoints are made atomic by writing a commit marker last, so a
reader only ever trusts directories that carry one.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from ..errors import SpillError

CLOUD_SCHEMES = {"s3": "s3fs", "s3a": "s3fs", "gs": "gcsfs", "gcs": "gcsfs",
                 "az": "adlfs", "abfs": "adlfs", "abfss": "adlfs"}


def is_uri(s: str | os.PathLike) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", str(s)))


def scheme_of(s: str) -> str:
    m = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*)://", str(s))
    return m.group(1).lower() if m else "file"


def _filesystem(uri: str):
    import fsspec
    try:
        return fsspec.core.url_to_fs(uri)
    except ImportError as e:
        pkg = CLOUD_SCHEMES.get(scheme_of(uri), "the fsspec backend for this scheme")
        raise SpillError(f"{scheme_of(uri)}:// needs {pkg}",
                         'pip install "streamweights[cloud]"') from e
    except ValueError as e:
        raise SpillError(f"cannot open {uri}: {e}") from e


class Store:
    """A directory-like location. Paths passed in are relative to the root."""

    def __init__(self, uri: str | os.PathLike):
        self.uri = str(uri)
        if is_uri(self.uri):
            self.fs, root = _filesystem(self.uri)
            self.local = scheme_of(self.uri) == "file"
        else:
            import fsspec
            self.fs = fsspec.filesystem("file")
            root = str(Path(self.uri).expanduser().resolve())
            self.local = True
        self.root = root.rstrip("/") or "/"
        self.scheme = scheme_of(self.uri)

    def _p(self, rel: str) -> str:
        rel = rel.strip("/")
        return f"{self.root}/{rel}" if rel else self.root

    def sub(self, rel: str) -> "Store":
        base = self.uri.rstrip("/")
        return Store(f"{base}/{rel.strip('/')}")

    def exists(self, rel: str) -> bool:
        return bool(self.fs.exists(self._p(rel)))

    def read(self, rel: str) -> bytes:
        with self.fs.open(self._p(rel), "rb") as f:
            return f.read()

    def write(self, rel: str, data: bytes) -> None:
        """Atomic: temp then rename locally; one put on an object store or in memory."""
        path = self._p(rel)
        parent = path.rsplit("/", 1)[0]
        if self.scheme not in ("s3", "s3a", "gs", "gcs", "az", "abfs", "abfss"):
            self.fs.makedirs(parent, exist_ok=True)
        if self.local:
            tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        else:
            with self.fs.open(path, "wb") as f:
                f.write(data)

    def ls(self, rel: str = "") -> list[str]:
        """Immediate child names (not paths) under rel; [] when it does not exist."""
        path = self._p(rel)
        if not self.fs.exists(path):
            return []
        names = []
        for entry in self.fs.ls(path, detail=False):
            name = str(entry).rstrip("/").rsplit("/", 1)[-1]
            if name:
                names.append(name)
        return sorted(set(names))

    def size(self, rel: str) -> int:
        return int(self.fs.size(self._p(rel)))

    def rm(self, rel: str) -> None:
        path = self._p(rel)
        if self.fs.exists(path):
            self.fs.rm(path, recursive=True)

    def put_file(self, local: str | Path, rel: str) -> None:
        self.write(rel, Path(local).read_bytes())

    def get_file(self, rel: str, local: str | Path) -> None:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(local) + f".{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_bytes(self.read(rel))
        os.replace(tmp, local)

    def copy_to_local(self, rel: str, local: str | Path, chunk: int = 16 << 20) -> int:
        """Stream one object to a local file (never holding it in memory); returns bytes."""
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(local) + f".{uuid.uuid4().hex[:8]}.tmp")
        n = 0
        with self.fs.open(self._p(rel), "rb") as src, open(tmp, "wb") as dst:
            while True:
                b = src.read(chunk)
                if not b:
                    break
                dst.write(b)
                n += len(b)
        os.replace(tmp, local)
        return n

    def __repr__(self) -> str:
        return f"Store({self.uri})"
