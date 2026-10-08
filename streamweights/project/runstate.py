"""A run's mutable state and how it becomes accepted, immutable history.

  <state>/runs/<run id>/control.json     the authority (control.py)
  <state>/runs/<run id>/control.lock     the stable lock file (local disk only)
  <state>/runs/<run id>/payloads/...     immutable payloads (payloads.py): checkpoints, stage
                                         outputs, the completion snapshot
  <state>/runs/<run id>/attempts/<id>/   this attempt's staging (always local): the engines
                                         write here, nothing in it is accepted until a
                                         payload is published and the control object points
                                         at it

`<state>` is `<project>/.spill` for a local run, or an s3:// prefix whose control object is
the one authority. Completion installs the accepted snapshot at <project>/runs/<id>/."""

from __future__ import annotations

import json
import os
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from ..errors import SpillError
from ..portable.store import Store, is_uri
from . import caps, control as ctl, hooks, payloads as pl
from .common import atomic_write, sha_bytes, sha_file, write_json


def run_uri(state_root: str, run_id: str) -> str:
    return f"{str(state_root).rstrip('/')}/runs/{run_id}"


@dataclass
class Recovery:
    """What a resume found: what is committed, and what is and is not known about the rest."""
    step: int | None
    cursor: dict | None
    seq: int | None
    generation_of_payload: int | None
    known_replay: int | None          # exact only when a local record of the lost progress exists
    bound: str

    def line(self) -> str:
        if self.step is None:
            return "no checkpoint is committed yet; this run starts from step 0"
        s = (f"last committed checkpoint: step {self.step}, data cursor "
             f"{json.dumps(self.cursor, sort_keys=True)}")
        if self.known_replay is not None:
            return s + f"; {self.known_replay} step(s) done after it were not committed and are redone"
        return s + f"; progress after it was not committed ({self.bound})"


class RunState:
    def __init__(self, state_root: str, run_id: str, work_root: Path, owner: str | None = None,
                 lock_timeout: float | None = None):
        self.state_root, self.run_id = str(state_root), run_id
        self.uri = run_uri(state_root, run_id)
        self.remote = is_uri(self.uri) and not self.uri.startswith("file://")
        self.backend = caps.open_backend(self.uri)
        self.auth = ctl.Authority(self.backend, owner=owner, lock_timeout=lock_timeout)
        self.store = Store(self.uri)
        self.attempt = pl.new_attempt()
        self.work = Path(work_root) / "runs" / run_id / "attempts" / self.attempt
        self.next_seq = 1
        self._renew_stop = None

    # ---- lifecycle
    @classmethod
    def create(cls, state_root, run_id, identity, work_root, **kw) -> "RunState":
        rs = cls(state_root, run_id, work_root, **kw)
        ctl.Authority.create(rs.backend, run_id, identity)
        return rs

    @classmethod
    def open(cls, state_root, run_id, work_root, **kw) -> "RunState":
        rs = cls(state_root, run_id, work_root, **kw)
        if rs.auth.read() is None:
            raise SpillError(f"no run {run_id} at {rs.uri}", "spill runs")
        return rs

    def doc(self) -> dict:
        d = self.auth.read()
        if d is None:
            raise SpillError(f"the control object of run {self.run_id} is missing at {self.uri}",
                             "spill resume <project>")
        return d

    def acquire(self, heartbeat: bool = True) -> dict:
        d = self.auth.acquire()
        self.work.mkdir(parents=True, exist_ok=True)
        ck = d.get("checkpoint")
        self.next_seq = (ck["seq"] + 1) if ck else 1
        if heartbeat:
            self._renew_stop = ctl.renewer(self.auth)
        return d

    def release(self) -> None:
        if self._renew_stop is not None:
            self._renew_stop.set()
        try:
            self.auth.release()
        except SpillError:
            pass

    def quiesce_requested(self) -> bool:
        return self.auth.quiesce_requested()

    def _held(self) -> int:
        if self.auth.gen is None:
            raise ctl.Superseded("this writer never acquired the run", "spill resume <project>")
        return self.auth.gen

    # ---- checkpoints
    def publish_checkpoint(self, ckpt_dir: Path, step: int, cursor: dict | None) -> dict:
        """Copy a committed engine checkpoint into an immutable payload, verify it, then point
        the control object at it. Until the pointer moves, the previous checkpoint is current."""
        seq = self.next_seq
        gen = self._held()
        ref = pl.write_payload(self.store, "ckpt", gen, seq, self.attempt,
                               pl.dir_files(Path(ckpt_dir)), meta={"step": step, "cursor": cursor})
        ref = {**ref, "step": step, "cursor": cursor}
        hooks.fire("checkpoint_before_control")
        self.auth.publish_checkpoint(ref)
        hooks.fire("checkpoint_after_control")
        self.next_seq = seq + 1
        return ref

    def recovery(self, ckpt_every: int | None = None, local_progress: int | None = None
                 ) -> Recovery:
        d = self.doc()
        ck = d.get("checkpoint")
        if not ck:
            return Recovery(None, None, None, None, None, "nothing was committed")
        known = None
        if local_progress is not None and local_progress >= ck["step"]:
            known = local_progress - ck["step"]
        bound = (f"up to {ckpt_every} optimizer step(s), the checkpoint interval, may be redone; "
                 f"the exact number is unknown" if ckpt_every else "the amount to redo is unknown")
        return Recovery(ck["step"], ck.get("cursor"), ck["seq"], ck["generation"], known, bound)

    def restore_checkpoint(self, dest: Path) -> dict | None:
        """The checkpoint the control object points at, verified and copied into `dest`.
        Never the newest directory name or object listing."""
        ck = self.doc().get("checkpoint")
        if not ck:
            return None
        pl.read_payload(self.store, ck, dest)
        return ck

    # ---- stages
    def publish_stage(self, stage: str, out_dir: Path, result: dict) -> dict:
        files = pl.dir_files(Path(out_dir))
        files["result.json"] = (json.dumps(result, indent=1, sort_keys=True) + "\n").encode()
        ref = pl.write_payload(self.store, f"stage-{stage.replace(':', '_')}", self._held(), 0,
                               self.attempt, files)
        hooks.fire("stage_before_control")
        self.auth.publish_stage(stage, {**ref, "stage": stage, "status": result.get("status", "done")})
        return ref

    def accepted_stage(self, stage: str) -> dict | None:
        return self.doc().get("stages", {}).get(stage)

    def fetch_stage(self, stage: str, dest: Path) -> dict | None:
        ref = self.accepted_stage(stage)
        if ref is None:
            return None
        pl.read_payload(self.store, ref, dest)
        return json.loads((Path(dest) / "result.json").read_text())

    # ---- completion
    def complete(self, snapshot_dir: Path, meta: dict | None = None) -> dict:
        """Write the snapshot as an immutable payload, then make the fenced transition. A stale
        writer fails at the transition and leaves only an unreferenced payload."""
        ref = pl.write_payload(self.store, "snapshot", self._held(), 0, self.attempt,
                               pl.dir_files(Path(snapshot_dir)), meta=meta)
        hooks.fire("complete_before_control")
        self.auth.complete({**ref, "completed_at": time.time()})
        return ref

    def install_snapshot(self, project: Path) -> Path:
        """Materialize the accepted snapshot at <project>/runs/<id>/ (idempotent, read-only
        afterwards). Safe to call by anyone who reads a completed control object."""
        d = self.doc()
        if d["status"] != ctl.COMPLETED or not d.get("completed"):
            raise SpillError(f"run {self.run_id} is not completed", "spill resume <project>")
        dest = Path(project) / "runs" / self.run_id
        ref = d["completed"]
        if dest.exists() and (dest / ".snapshot").exists() and \
                (dest / ".snapshot").read_text().strip() == ref["manifest_sha256"]:
            return dest
        tmp = dest.with_name(f".{self.run_id}.install-{self.attempt}")
        shutil.rmtree(tmp, ignore_errors=True)
        pl.read_payload(self.store, ref, tmp)
        (tmp / ".snapshot").write_text(ref["manifest_sha256"] + "\n")
        if dest.exists():                      # an interrupted earlier install
            make_writable(dest)
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dest)
        make_readonly(dest)
        return dest


def make_readonly(d: Path) -> None:
    for p in sorted(Path(d).rglob("*"), reverse=True):
        try:
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        except OSError:
            pass
    try:
        Path(d).chmod(Path(d).stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    except OSError:
        pass


def make_writable(d: Path) -> None:
    for p in [Path(d), *Path(d).rglob("*")]:
        try:
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass


def tree_digest(d: Path, skip=()) -> dict:
    """{relative path: sha256} for every file under d: what 'byte-identical' is checked with."""
    return {str(p.relative_to(d)): sha_file(p) for p in sorted(Path(d).rglob("*"))
            if p.is_file() and p.name not in skip}
