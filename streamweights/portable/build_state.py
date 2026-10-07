"""Portable state for `spill build`: where a build's stage reached, per-stage checkpoints,
intermediate files and the input fingerprint live, in the same portable layout as the tune
and row jobs.

  <state>/state.json              the plan, the stage reached, per-stage results, the engine,
                                  hardware, OS and numerics that produced each stage, the
                                  input fingerprint. One atomic object, written after
                                  everything it names, so it is the commit marker.
  <state>/files/<name>            intermediate files (the teacher's answers)
  <state>/adapters/<name>/        the tuned adapter, both layouts, once the tune stage is done
  <state>/stages/<slug>/          each stage's own state: ckpt/ for a tune (a portable
                                  checkpoint), rows/ for distill and eval (committed segments)

`<state>` is the build folder's `.build/` by default, or `--state <uri>`: a path, s3://, gs://,
az:// or memory://. Any machine that can read it can continue the build.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .store import Store

SCHEMA = 1


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tree_sha(d: Path) -> str:
    """One hash over a directory's files (names and contents, sorted)."""
    h = hashlib.sha256()
    for f in sorted(p for p in Path(d).rglob("*") if p.is_file()):
        h.update(str(f.relative_to(d)).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
    return h.hexdigest()


class BuildState:
    def __init__(self, uri: str | Path):
        self.uri = str(uri)
        self.store = Store(self.uri)

    # ---- the state document
    def read(self) -> dict | None:
        if not self.store.exists("state.json"):
            return None
        try:
            return json.loads(self.store.read("state.json"))
        except (ValueError, OSError):
            return None

    def write(self, doc: dict) -> None:
        self.store.write("state.json", json.dumps({"schema": SCHEMA, **doc}, indent=2).encode())

    # ---- stage locations
    def stage_uri(self, stage_id: str) -> str:
        import re
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stage_id).strip("-")
        return f"{self.uri.rstrip('/')}/stages/{slug}"

    # ---- files and directories
    def put_file(self, name: str, local: Path) -> dict:
        data = Path(local).read_bytes()
        self.store.write(name, data)
        return {"name": name, "sha256": _sha_bytes(data), "bytes": len(data)}

    def get_file(self, name: str, local: Path) -> None:
        self.store.get_file(name, local)

    def has(self, name: str) -> bool:
        return self.store.exists(name)

    def put_dir(self, prefix: str, local_dir: Path) -> dict:
        d = Path(local_dir)
        names = []
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            rel = str(f.relative_to(d))
            self.store.write(f"{prefix}/{rel}", f.read_bytes())
            names.append(rel)
        self.store.write(f"{prefix}/MANIFEST.json", json.dumps(names).encode())  # last
        return {"name": prefix, "sha256": tree_sha(d), "files": len(names)}

    def get_dir(self, prefix: str, local_dir: Path) -> None:
        d = Path(local_dir)
        names = json.loads(self.store.read(f"{prefix}/MANIFEST.json"))
        d.mkdir(parents=True, exist_ok=True)
        for rel in names:
            self.store.get_file(f"{prefix}/{rel}", d / rel)

    def has_dir(self, prefix: str) -> bool:
        return self.store.exists(f"{prefix}/MANIFEST.json")
