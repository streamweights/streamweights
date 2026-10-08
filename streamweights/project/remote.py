"""A project at another location: a local directory elsewhere, or s3://bucket/prefix.

The authority for its live runs is the control object there (`<uri>/.spill/runs/<id>/`);
payloads stay there too and are read on demand. What `spill resume <uri>` keeps on this
machine is a working copy of the small project files (config, data, completed runs) and the
attempt's staging; it is a cache, never a second authority: nothing is published except
through the remote control object, which this machine must acquire first."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..errors import SpillError
from ..portable.store import Store, is_uri, scheme_of
from ..registry import REPO_ROOT
from .common import sha_bytes

SKIP_PREFIX = (".spill/",)


def is_remote(target: str) -> bool:
    return is_uri(target) and scheme_of(target) != "file"


def working_copy(uri: str) -> Path:
    return REPO_ROOT / "remote" / hashlib.sha256(uri.encode()).hexdigest()[:12]


def _relfiles(store: Store) -> list[str]:
    root = store.root
    out = []
    for p in store.fs.find(root):
        rel = str(p)[len(root):].lstrip("/") if str(p).startswith(root) else str(p)
        if not any(rel.startswith(x) for x in SKIP_PREFIX):
            out.append(rel)
    return sorted(out)


def pull_project(uri: str, dest: Path) -> int:
    """Copy the project files (everything but .spill/) to `dest`. Files named by the latest
    transfer manifest are verified against its checksums. Returns the file count."""
    store = Store(uri)
    dest.mkdir(parents=True, exist_ok=True)
    manifests = []
    for tid in store.ls(".spill/transfers"):
        try:
            manifests.append(json.loads(store.read(f".spill/transfers/{tid}/MOVE_MANIFEST.json")))
        except (OSError, ValueError):
            continue
    manifests.sort(key=lambda m: m.get("created", 0))
    expect = manifests[-1]["files"] if manifests else {}
    n = 0
    for rel in _relfiles(store):
        data = store.read(rel)
        want = expect.get(rel)
        if want and rel != "REPORT.md" and sha_bytes(data) != want["sha256"]:
            raise SpillError(f"{rel} at {uri} does not match the transfer manifest checksum",
                             "re-run `spill move` at the source")
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.read_bytes() == data:
            n += 1
            continue
        if p.exists() and rel.startswith("runs/"):
            continue                                   # an installed snapshot is never rewritten
        p.write_bytes(data)
        n += 1
    return n


def push_completed(uri: str, project: Path, run_id: str) -> None:
    """After completion, publish the installed snapshot and the index to the remote project."""
    store = Store(uri)
    base = Path(project) / "runs" / run_id
    for p in sorted(base.rglob("*")):
        if p.is_file():
            rel = p.relative_to(project).as_posix()
            if not store.exists(rel):
                store.write(rel, p.read_bytes())
    idx = Path(project) / "REPORT.md"
    if idx.exists():
        store.write("REPORT.md", idx.read_bytes())
