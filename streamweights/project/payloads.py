"""Immutable payloads: checkpoints, stage outputs and snapshots are written once, to a unique
location, with a manifest of sizes and content hashes written last. A payload is current only
when the control object points at its manifest (control.py); nothing here overwrites a
published payload, and orphans left by a dead or superseded writer are simply unreferenced.

  <state>/payloads/<kind>-g<generation>-s<seq>-<attempt>/<files...>
  <state>/payloads/<kind>-g<generation>-s<seq>-<attempt>/MANIFEST.json     written last
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from ..errors import SpillError
from ..portable.store import Store
from . import hooks
from .common import sha_bytes, sha_file


def payload_dir(kind: str, generation: int, seq: int, attempt: str) -> str:
    return f"payloads/{kind}-g{generation:06d}-s{seq:09d}-{attempt}"


def new_attempt() -> str:
    return uuid.uuid4().hex[:12]


def write_payload(store: Store, kind: str, generation: int, seq: int, attempt: str,
                  files: dict[str, Path | bytes], meta: dict | None = None) -> dict:
    """Write `files` ({name: local path or bytes}) then the manifest; verify every file by
    reading it back; return the reference the control object will hold. Refuses to write
    into a location that already exists."""
    d = payload_dir(kind, generation, seq, attempt)
    if store.exists(f"{d}/MANIFEST.json") or store.ls(d):
        raise SpillError(f"payload {d} already exists; payloads are never overwritten",
                         "run the command again")
    entries = {}
    for i, (name, src) in enumerate(sorted(files.items())):
        data = src if isinstance(src, bytes) else Path(src).read_bytes()
        store.write(f"{d}/{name}", data)
        entries[name] = {"bytes": len(data), "sha256": sha_bytes(data)}
        if i == 0:
            hooks.fire("payload_mid_write")
    hooks.fire("payload_before_manifest")
    manifest = {"schema": 1, "kind": kind, "generation": generation, "seq": seq,
                "attempt": attempt, "files": entries, **(meta or {})}
    mbytes = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    store.write(f"{d}/MANIFEST.json", mbytes)
    verify_payload(store, d, sha_bytes(mbytes))
    hooks.fire("payload_written")
    return {"dir": d, "manifest_sha256": sha_bytes(mbytes), "attempt": attempt,
            "generation": generation, "seq": seq}


def verify_payload(store: Store, d: str, manifest_sha256: str | None = None) -> dict:
    """The manifest of payload `d`, after checking that every file it lists is present with
    the recorded size and hash. Raises SpillError on any difference."""
    try:
        mbytes = store.read(f"{d}/MANIFEST.json")
    except (FileNotFoundError, OSError):
        raise SpillError(f"payload {d} has no manifest; it was never completed",
                         "spill resume <project>")
    if manifest_sha256 and sha_bytes(mbytes) != manifest_sha256:
        raise SpillError(f"payload {d}: the manifest differs from the one that was published",
                         "spill resume <project>")
    manifest = json.loads(mbytes)
    for name, meta in manifest["files"].items():
        try:
            data = store.read(f"{d}/{name}")
        except (FileNotFoundError, OSError):
            raise SpillError(f"payload {d}: {name} is missing", "spill resume <project>")
        if len(data) != meta["bytes"] or sha_bytes(data) != meta["sha256"]:
            raise SpillError(f"payload {d}: {name} does not match its recorded hash",
                             "spill resume <project>")
    return manifest


def read_payload(store: Store, ref: dict, dest: Path) -> dict:
    """Verify the payload a control pointer names and copy its files into `dest`."""
    manifest = verify_payload(store, ref["dir"], ref["manifest_sha256"])
    dest = Path(dest)
    for name in manifest["files"]:
        store.get_file(f"{ref['dir']}/{name}", dest / name)
    return manifest


def dir_files(d: Path, exclude=()) -> dict[str, Path]:
    return {str(p.relative_to(d)): p for p in sorted(Path(d).rglob("*"))
            if p.is_file() and p.name not in exclude}
