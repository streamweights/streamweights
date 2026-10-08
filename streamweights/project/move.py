"""spill move: a controlled handoff of a project's committed state to another location.

  1  quiesce     ask the running writer to stop at its next committed boundary (a bounded wait)
  2  hold        take ownership of every unfinished run under a handoff owner id derived from the
                 transfer id (so a retry is the same owner) and record handoff state `preparing`
  3  copy        copy a consistent snapshot of the project: every file and, per run, the payloads
                 its control object points at; write a checksummed manifest
  4  verify      read everything back from the destination and compare sizes and sha256
  5  prepare     create the destination's control objects in state `incoming`: they cannot be
                 acquired, so the destination cannot run
  6  fence       mark the SOURCE control objects `transferred` under a new generation: from here
                 nothing can publish to the source, including this process
  7  activate    only now flip the destination's control objects from `incoming` to `idle`

If anything stops between 1 and 7 the same command, run again, recovers the recorded transfer:
completed steps are not redone (files that already verify are not copied again), a fenced
source is never re-opened, and at no point are both locations acquirable. The source bytes are
kept. The command prints the receiving command only after step 7."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..errors import SpillError
from ..portable.store import Store, is_uri
from . import caps, control as ctl, hooks
from .common import sha_bytes, sha_file
from .runstate import run_uri

SKIP_NAMES = {".DS_Store"}


@dataclass
class MoveResult:
    transfer_id: str
    dest: str
    runs: list
    files: int
    bytes: int
    resumed: bool


def _controls(project: Path) -> list[tuple[str, dict]]:
    from .coordinator import list_runs
    return [(d["run_id"], d) for d in list_runs(project)]


def _project_files(project: Path, with_exports: bool) -> list[str]:
    """Everything that belongs to the project and is not mutable run state (.spill/)."""
    out = []
    for p in sorted(Path(project).rglob("*")):
        if not p.is_file() or p.name in SKIP_NAMES:
            continue
        rel = p.relative_to(project).as_posix()
        if rel.split("/")[0] == ".spill":
            continue
        if rel.split("/")[0] == "exports" and not with_exports and "/artifacts/" in "/" + rel:
            continue                    # bulk export artifacts stay where they were made
        out.append(rel)
    return out


def _payload_files(project: Path, run_id: str, doc: dict) -> list[str]:
    """The payload directories a run's control object points at: current checkpoint, accepted
    stages, completed snapshot. Orphans are not moved."""
    base = Path(project) / ".spill" / "runs" / run_id
    dirs = []
    if doc.get("checkpoint"):
        dirs.append(doc["checkpoint"]["dir"])
    for ref in doc.get("stages", {}).values():
        dirs.append(ref["dir"])
    if doc.get("completed"):
        dirs.append(doc["completed"]["dir"])
    out = []
    for d in dirs:
        for p in sorted((base / d).rglob("*")):
            if p.is_file():
                out.append(p.relative_to(project).as_posix())
    return out


def _tid(dest: str, docs: dict) -> tuple[str, bool]:
    """Reuse the transfer recorded on an unfinished run for this destination (a retry),
    otherwise a new one."""
    for rid, d in docs.items():
        h = d.get("handoff") or {}
        if h.get("dest") == dest and d["status"] in (ctl.HANDOFF, ctl.TRANSFERRED):
            return h["transfer_id"], True
    return uuid.uuid4().hex[:16], False


def move(project: Path, dest: str, say=print, wait_s: float = 120.0, with_exports: bool = False,
         local_backend_factory=None) -> MoveResult:
    project = Path(project).resolve()
    dest = str(dest).rstrip("/")
    if not is_uri(dest):
        dest = str(Path(dest).expanduser().resolve())
    docs = dict(_controls(project))
    if not docs:
        raise SpillError(f"{project.name} has no runs to move", f"spill build {project}")
    for rid, d in docs.items():
        if d["status"] == ctl.TRANSFERRED and (d["handoff"] or {}).get("dest") != dest:
            raise SpillError(f"run {rid} was already handed to {d['handoff']['dest']}",
                             f"spill resume {d['handoff']['dest']}")
    tid, retry = _tid(dest, docs)
    if retry:
        say(f"recovering the recorded transfer {tid} to {dest}")
    active = [rid for rid, d in docs.items() if d["status"] in (ctl.IDLE, ctl.RUNNING, ctl.HANDOFF)]
    transferred = [rid for rid, d in docs.items() if d["status"] == ctl.TRANSFERRED]
    pending = [rid for rid in transferred if (docs[rid].get("handoff") or {}).get("transfer_id") == tid]
    done_runs = [rid for rid, d in docs.items() if d["status"] == ctl.COMPLETED]
    dest_state = f"{dest}/.spill"
    for rid in active:                                    # the destination must be able to hold it
        caps.open_backend(run_uri(dest_state, rid))
    src_backend = {rid: caps.open_backend(run_uri(str(project / ".spill"), rid)) for rid in docs}
    owner = f"mover:{tid}"
    auths = {rid: ctl.Authority(src_backend[rid], owner=owner, lease_s=300.0) for rid in docs}

    # 1 quiesce, 2 hold
    for rid in active:
        d = docs[rid]
        if d["status"] == ctl.HANDOFF and (d.get("handoff") or {}).get("transfer_id") == tid:
            continue                                      # already ours from an earlier attempt
        auths[rid].request_quiesce(tid)
    deadline = time.monotonic() + wait_s
    for rid in active:
        while True:
            d = auths[rid].read()
            if d["owner"] in (None, owner) or d["lease_expires"] < time.time():
                break
            if time.monotonic() > deadline:
                raise SpillError(f"run {rid} is still being written by {d['owner']}; it did not "
                                 f"stop within {wait_s:.0f} s", "stop that build, or move --wait <seconds>")
            time.sleep(0.2)
        d = auths[rid].read()
        if d["status"] != ctl.HANDOFF:
            auths[rid].acquire()
            auths[rid].begin_handoff(tid, dest)
        else:
            auths[rid].acquire()                          # same owner id: a retry takes it again
    hooks.fire("move_held")

    # 3 copy: a consistent snapshot, verified unchanged afterwards
    files = _project_files(project, with_exports)
    docs_now = {rid: auths[rid].read() for rid in docs}
    for rid in docs:
        files += _payload_files(project, rid, docs_now[rid])
    files = sorted(set(files))
    store = Store(dest)
    before = {rel: sha_file(project / rel) for rel in files}
    manifest = {"transfer_id": tid, "source": str(project), "created": time.time(),
                "files": {rel: {"bytes": (project / rel).stat().st_size, "sha256": before[rel]}
                          for rel in files},
                "source_revisions": {rid: docs_now[rid]["revision"] for rid in docs}}
    copied = 0
    for rel in files:
        if _verifies(store, rel, manifest["files"][rel]):
            continue
        store.write(rel, (project / rel).read_bytes())
        copied += 1
        hooks.fire("move_mid_copy")
    mbytes = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    store.write(f".spill/transfers/{tid}/MOVE_MANIFEST.json", mbytes)
    for rid in active:
        auths[rid].set_handoff_state(tid, "copied", files=len(files), copied_now=copied)
    hooks.fire("move_after_copy")
    # the source must not have changed while it was copied
    for rel in files:
        if sha_file(project / rel) != before[rel]:
            raise SpillError(f"{rel} changed while the project was being copied; nothing was "
                             f"handed over", f"spill move {project.name} {dest}   (run it again)")

    # 4 verify from the destination
    for rel, meta in manifest["files"].items():
        if not _verifies(store, rel, meta):
            raise SpillError(f"{rel} at {dest} does not match its checksum after the copy",
                             f"spill move {project.name} {dest}   (run it again)")
    for rid in active:
        auths[rid].set_handoff_state(tid, "verified")
    hooks.fire("move_after_verify")

    # 5 destination control objects: incoming for unfinished runs, a copy for completed ones
    dest_docs = {}
    for rid in active:
        src = docs_now[rid]
        gen_after = src["generation"] + 1                 # what the source reaches when fenced
        inc = ctl.new_doc(rid, src["identity"], ctl.INCOMING, generation=gen_after)
        inc.update(checkpoint=src.get("checkpoint"), stages=src.get("stages", {}),
                   handoff={"transfer_id": tid, "from": str(project), "state": "incoming",
                            "source_generation": src["generation"]},
                   history=src.get("history", [])[-20:] + [
                       {"rev": 0, "event": "moved-in", "at": time.time()}],
                   locations=src.get("locations", []))
        ctl.Authority.create(caps.open_backend(run_uri(dest_state, rid)), rid, src["identity"],
                             ctl.INCOMING, gen_after)      # no-op when it already exists
        b = caps.open_backend(run_uri(dest_state, rid))
        cur, tok = b.read()
        if cur["handoff"] is None:                        # fresh: write the full incoming document
            inc["revision"] = cur["revision"] + 1
            b.cas(tok, inc)
        dest_docs[rid] = gen_after
    for rid in done_runs:
        src = docs_now[rid]
        b = caps.open_backend(run_uri(dest_state, rid))
        if b.read()[0] is None:
            ctl.Authority.create(b, rid, src["identity"])
            cur, tok = b.read()
            copy = dict(src)
            copy.update(revision=cur["revision"] + 1, owner=None, lease_expires=0.0,
                        locations=src.get("locations", []) + [
                            {"moved_from": str(project), "transfer_id": tid}])
            b.cas(tok, copy)
    hooks.fire("move_before_fence")

    # 6 fence the source
    for rid in active:
        auths[rid].mark_transferred(tid)
    hooks.fire("move_after_fence")

    # 7 activate the destination
    for rid in active + pending:
        a = ctl.Authority(caps.open_backend(run_uri(dest_state, rid)), owner=f"activator:{tid}")
        gen = auths[rid].read()["generation"]
        a.activate_incoming(tid, gen)
    for rid in done_runs + transferred:
        try:
            ctl.Authority(src_backend[rid], owner=owner).add_location(
                {"moved_to": dest, "transfer_id": tid})
        except SpillError:
            pass
    hooks.fire("move_after_activate")
    nbytes = sum(m["bytes"] for m in manifest["files"].values())
    return MoveResult(tid, dest, active + pending + done_runs, len(files), nbytes, retry)


def _verifies(store: Store, rel: str, meta: dict) -> bool:
    try:
        if store.size(rel) != meta["bytes"]:
            return False
        return sha_bytes(store.read(rel)) == meta["sha256"]
    except (FileNotFoundError, OSError):
        return False
