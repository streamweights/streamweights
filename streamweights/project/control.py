"""One control object per run: who owns it, under which fencing generation, and what has
been accepted.

The control object is the single authority for a live run. It holds the owner id, a
monotonically increasing fencing generation, a lease expiry, a control revision, the run or
handoff status and the pointers to the accepted checkpoint, the accepted stage results and
the completed snapshot. Every transition (acquire, renew, publish a checkpoint or a stage,
complete, hand off) is a compare-and-swap of that one object:

  local disk  an OS-level exclusive lock (flock) on a stable lock file held for the short
              read-validate-replace section; the object is replaced atomically
  S3          If-None-Match: * to create, If-Match <etag> to replace; the object is never
              deleted and recreated, so fencing never resets

The client validates owner, generation and the allowed transition against what it just read
before it writes; the store only validates the ETag (S3) or the lock (local). On a conflict
or an uncertain outcome the client rereads and reconciles: if it has been superseded it
stops, and it retries only when the same owner and generation still permit the transition,
rebuilt from the newest state, never a replay of an old document.

Lease assumptions: leases are wall-clock times written by the owner. Neither S3 nor the
local lock checks them. A lease that expired only allows a successor to compete for
ownership by a successful conditional acquisition, which creates a newer generation; the
old writer is then rejected at its next transition. Clocks of competing workers are assumed
to differ by much less than the lease (default 120 s) and owners renew before it runs out.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..errors import SpillError
from . import hooks
from .common import atomic_write, fsync_dir

SCHEMA = 1
LEASE_S = 120.0

# statuses
IDLE, RUNNING, HANDOFF, TRANSFERRED, INCOMING, COMPLETED, FAILED, BUNDLED = (
    "idle", "running", "handoff", "transferred", "incoming", "completed", "failed", "bundled")


class Conflict(Exception):
    """The conditional write lost: someone replaced the object since it was read."""


class Uncertain(Exception):
    """The request's outcome is unknown (timeout, dropped response)."""


class AlreadyExists(Exception):
    pass


class Superseded(SpillError):
    """This writer no longer owns the run: a newer generation, a handoff or a completion."""


class NotAllowed(SpillError):
    """The transition is not allowed in the control object's current state."""


@dataclass
class Token:
    """What a conditional write is conditioned on: the ETag (S3) or the revision (local)."""
    value: str


def new_owner_id() -> str:
    import socket
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


# ------------------------------------------------------------ backends

class LocalBackend:
    """Control object as a file next to a stable lock file. `locked()` is the critical
    section: read, validate, replace happen inside it, so there is no read-then-write gap."""

    kind = "local"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / "control.json"
        self.lock_path = self.root / "control.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.lock_path.exists():
            with open(self.lock_path, "a"):
                pass
        self._mutex = threading.RLock()

    @contextlib.contextmanager
    def locked(self, timeout: float | None = None):
        import fcntl
        with self._mutex:
            fd = os.open(self.lock_path, os.O_RDWR)
            try:
                deadline = None if timeout is None else time.monotonic() + timeout
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | (0 if timeout is None else fcntl.LOCK_NB))
                        break
                    except BlockingIOError:
                        if time.monotonic() > deadline:
                            raise LockContended(f"control lock {self.lock_path} is held by "
                                                f"another process")
                        time.sleep(0.01)
                try:
                    yield
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def read(self):
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return None, None
        return json.loads(data), Token(str(json.loads(data)["revision"]))

    def create(self, doc: dict) -> Token:
        with self.locked():
            if self.path.exists():
                raise AlreadyExists()
            atomic_write(self.path, _enc(doc))
        return Token(str(doc["revision"]))

    def cas(self, token: Token, doc: dict) -> Token:
        """Only called inside `locked()`: the revision must still be the one we read."""
        cur, tok = self.read()
        if tok is None or tok.value != token.value:
            raise Conflict()
        hooks.fire("control_before_replace")
        atomic_write(self.path, _enc(doc))
        return Token(str(doc["revision"]))


class LockContended(SpillError):
    def __init__(self, msg):
        super().__init__(msg, "wait for the other process, or stop it")


class S3Backend:
    """Control object in S3 (or an S3-compatible store such as MinIO) through boto3's
    conditional writes. `uri` is s3://bucket/prefix/to/run-dir."""

    kind = "s3"

    def __init__(self, uri: str, client=None):
        from urllib.parse import urlparse
        u = urlparse(uri)
        self.bucket, self.prefix = u.netloc, u.path.strip("/")
        self.key = f"{self.prefix}/control.json" if self.prefix else "control.json"
        self.uri = uri
        self._c = client

    @property
    def client(self):
        if self._c is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as e:
                raise SpillError("s3:// needs boto3", 'pip install "streamweights[cloud]"') from e
            self._c = boto3.client("s3", config=Config(
                retries={"max_attempts": 1, "mode": "standard"}, connect_timeout=10,
                read_timeout=30))
        return self._c

    @contextlib.contextmanager
    def locked(self, timeout=None):
        yield                          # no lock: the ETag condition is the only guard

    def _classify(self, e):
        from botocore.exceptions import ClientError
        if isinstance(e, ClientError):
            code = e.response.get("Error", {}).get("Code", "")
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            if code in ("PreconditionFailed", "ConditionalRequestConflict") or status in (409, 412):
                return Conflict()
            if status >= 500 or code in ("RequestTimeout", "SlowDown", "InternalError"):
                return Uncertain(str(e))
            return None
        if isinstance(e, Uncertain):
            return e
        from botocore.exceptions import BotoCoreError
        if isinstance(e, BotoCoreError):
            return Uncertain(str(e))
        return None

    def read(self):
        from botocore.exceptions import ClientError
        try:
            r = self.client.get_object(Bucket=self.bucket, Key=self.key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None, None
            raise
        return json.loads(r["Body"].read()), Token(r["ETag"])

    def create(self, doc: dict) -> Token:
        try:
            r = self.client.put_object(Bucket=self.bucket, Key=self.key, Body=_enc(doc),
                                       IfNoneMatch="*", ContentType="application/json")
        except Exception as e:
            k = self._classify(e)
            if isinstance(k, Conflict):
                raise AlreadyExists()
            if k is not None:
                raise k
            raise
        if hooks.fault("cas_response"):
            raise Uncertain("injected: the response was lost")
        return Token(r["ETag"])

    def cas(self, token: Token, doc: dict) -> Token:
        hooks.fire("control_before_replace")
        try:
            r = self.client.put_object(Bucket=self.bucket, Key=self.key, Body=_enc(doc),
                                       IfMatch=token.value, ContentType="application/json")
        except Exception as e:
            k = self._classify(e)
            if k is not None:
                raise k
            raise
        if hooks.fault("cas_response"):
            raise Uncertain("injected: the response was lost")
        return Token(r["ETag"])


def _enc(doc: dict) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()


# ------------------------------------------------------------ the authority

def new_doc(run_id: str, identity: str, status: str = IDLE, generation: int = 0,
            parent: str | None = None) -> dict:
    return {"schema": SCHEMA, "run_id": run_id, "identity": identity, "parent": parent, "revision": 0,
            "generation": generation, "owner": None, "lease_expires": 0.0, "status": status,
            "checkpoint": None, "stages": {}, "completed": None, "handoff": None,
            "quiesce": None, "locations": [], "history": []}


class Authority:
    """One writer's handle on a run's control object. `owner` is this worker's id; `gen` is
    the generation it holds after `acquire`."""

    MAX_TRIES = 8

    def __init__(self, backend, owner: str | None = None, lease_s: float | None = None,
                 clock=time.time, lock_timeout: float | None = None):
        self.b = backend
        self.lock_timeout = lock_timeout
        lease_s = lease_s if lease_s is not None else float(os.environ.get("SPILL_LEASE_S", LEASE_S))
        self.owner = owner or new_owner_id()
        self.lease_s = lease_s
        self.clock = clock
        self.gen: int | None = None
        self._mu = threading.Lock()          # renewal and publication in one worker never overlap

    # ---- the transition engine
    def _transition(self, name: str, fn, confirm=None):
        """Read the control object, let `fn(doc)` validate it and return the new document
        (or raise Superseded / NotAllowed), then write conditionally. On a conflict or an
        uncertain outcome reread and ask `fn` again against the newest document: a
        transition that is no longer permitted stops, one that still is is rebuilt from the
        newest state. `confirm(doc)` says whether a document shows our own transition landed."""
        with self._mu:
            last: Exception | None = None
            for _ in range(self.MAX_TRIES):
                with self.b.locked(self.lock_timeout):
                    doc, tok = self.b.read()
                    if doc is None:
                        raise NotAllowed(f"no control object for this run ({name})")
                    new = fn(doc)
                    new = dict(new)
                    new["revision"] = doc["revision"] + 1
                    new["history"] = (doc.get("history", []) + [
                        {"rev": new["revision"], "event": name, "owner": new.get("owner"),
                         "generation": new.get("generation"), "at": self.clock()}])[-40:]
                    hooks.fire("transition_validated")
                    try:
                        self.b.cas(tok, new)
                        hooks.fire("transition_written")
                        return new
                    except Conflict as e:
                        last = e
                    except Uncertain as e:
                        last = e
                        again, _ = self.b.read()
                        if again is not None and again["revision"] == new["revision"] \
                                and (confirm(again) if confirm else again == new):
                            return again
                time.sleep(0.01)
            raise SpillError(f"could not complete the {name} transition after "
                             f"{self.MAX_TRIES} tries ({last})", "run the command again")

    # ---- lifecycle
    @staticmethod
    def create(backend, run_id: str, identity: str, status: str = IDLE,
               generation: int = 0, parent: str | None = None) -> dict:
        doc = new_doc(run_id, identity, status, generation, parent)
        doc["history"] = [{"rev": 0, "event": "create", "at": time.time()}]
        try:
            backend.create(doc)
        except AlreadyExists:
            cur, _ = backend.read()
            if cur is not None:
                return cur
        except Uncertain:
            cur, _ = backend.read()
            if cur is not None:
                return cur
            raise
        return doc

    def read(self) -> dict | None:
        doc, _ = self.b.read()
        return doc

    def _check_owner(self, doc: dict, allow=(RUNNING,)):
        if doc["status"] == COMPLETED:
            raise Superseded("this run is completed; a completed run cannot be changed",
                             "spill build <project> to start a new run")
        if doc["status"] == TRANSFERRED:
            raise Superseded(f"this run was handed to {doc['handoff']['dest']}",
                             f"spill resume {doc['handoff']['dest']}")
        if doc["owner"] != self.owner or doc["generation"] != self.gen:
            raise Superseded(f"this writer was superseded (it held generation {self.gen}; "
                             f"generation {doc['generation']} is owned by {doc['owner']})",
                             "spill resume <project>")
        if doc["status"] not in allow:
            raise NotAllowed(f"the run is {doc['status']}, which does not allow this")

    def acquire(self) -> dict:
        """Become the owner under a new generation. Allowed when nobody owns the run or the
        owner's lease has expired (a competing acquisition then decides)."""
        target_holder: dict = {}

        def fn(doc):
            if doc["status"] == COMPLETED:
                raise NotAllowed("this run is completed; further training starts a new run",
                                 "spill build <project>")
            if doc["status"] == TRANSFERRED:
                raise NotAllowed(f"this run was handed to {doc['handoff']['dest']}",
                                 f"spill resume {doc['handoff']['dest']}")
            if doc["status"] == BUNDLED:
                raise NotAllowed("this run is a read-only copy inside a bundle; a training copy "
                                 "forks a new run that names it as its parent",
                                 "spill build <project> --new-run")
            if doc["status"] == INCOMING:
                raise NotAllowed("this location has not been activated yet; the handoff to it "
                                 "is not committed", "spill move <project> <uri>   (retry)")
            now = self.clock()
            held = doc["owner"] is not None and doc["lease_expires"] > now \
                and doc["owner"] != self.owner
            if held:
                raise NotAllowed(f"the run is owned by {doc['owner']} (generation "
                                 f"{doc['generation']}, lease until "
                                 f"{time.strftime('%H:%M:%S', time.localtime(doc['lease_expires']))})",
                                 "wait for it to finish, stop it, or wait out the lease")
            new = dict(doc)
            new.update(owner=self.owner, generation=doc["generation"] + 1,
                       lease_expires=now + self.lease_s, status=RUNNING if doc["status"] != HANDOFF
                       else HANDOFF)
            target_holder["gen"] = new["generation"]
            return new

        new = self._transition("acquire", fn, confirm=lambda d: d["owner"] == self.owner
                               and d["generation"] == target_holder.get("gen"))
        self.gen = new["generation"]
        return new

    def renew(self) -> dict:
        def fn(doc):
            self._check_owner(doc, allow=(RUNNING, HANDOFF))
            new = dict(doc)
            new["lease_expires"] = self.clock() + self.lease_s
            return new
        return self._transition("renew", fn, confirm=lambda d: d["owner"] == self.owner
                                and d["generation"] == self.gen)

    def release(self) -> dict:
        def fn(doc):
            self._check_owner(doc, allow=(RUNNING, HANDOFF))
            new = dict(doc)
            new.update(owner=None, lease_expires=0.0, quiesce=None,
                       status=IDLE if doc["status"] == RUNNING else doc["status"])
            return new
        return self._transition("release", fn, confirm=lambda d: d["owner"] is None)

    # ---- publication
    def publish_checkpoint(self, ref: dict) -> dict:
        """`ref`: {seq, step, cursor, dir, manifest_sha256, attempt, generation}. Becomes
        current only through this transition; its sequence must be above the current one."""
        def fn(doc):
            self._check_owner(doc)
            cur = doc["checkpoint"]
            if cur and ref["seq"] <= cur["seq"]:
                raise NotAllowed(f"checkpoint sequence {ref['seq']} would not advance the "
                                 f"published sequence {cur['seq']}")
            if ref["generation"] != doc["generation"]:
                raise Superseded(f"the payload was written under generation {ref['generation']}, "
                                 f"not the current {doc['generation']}")
            new = dict(doc)
            new["checkpoint"] = ref
            new["lease_expires"] = max(doc["lease_expires"], self.clock() + self.lease_s)
            return new
        return self._transition("checkpoint", fn,
                                confirm=lambda d: (d["checkpoint"] or {}).get("seq") == ref["seq"])

    def publish_stage(self, stage: str, ref: dict) -> dict:
        """Accept a stage's result: its attempt, outputs and their hashes. A stage already
        accepted keeps its result; republishing the same one is a no-op."""
        def fn(doc):
            self._check_owner(doc)
            have = doc["stages"].get(stage)
            if have and have["manifest_sha256"] != ref["manifest_sha256"]:
                raise NotAllowed(f"stage {stage} was already accepted with other outputs")
            new = dict(doc)
            new["stages"] = {**doc["stages"], stage: ref}
            return new
        return self._transition(f"stage:{stage}", fn,
                                confirm=lambda d: stage in d["stages"])

    def complete(self, ref: dict) -> dict:
        """The fenced transition that makes the run complete: after it nothing can change."""
        def fn(doc):
            self._check_owner(doc)
            new = dict(doc)
            new.update(status=COMPLETED, completed=ref, owner=None, lease_expires=0.0,
                       quiesce=None)
            return new
        return self._transition("complete", fn, confirm=lambda d: d["status"] == COMPLETED)

    def add_location(self, rec: dict) -> dict:
        """Record where the run's files also live (outside the completed snapshot). Allowed
        in any state; it touches nothing else."""
        def fn(doc):
            new = dict(doc)
            new["locations"] = doc.get("locations", []) + [rec]
            return new
        return self._transition("location", fn,
                                confirm=lambda d: rec in d.get("locations", []))

    # ---- handoff
    def request_quiesce(self, transfer_id: str) -> dict:
        """Ask the running owner to stop at its next committed boundary and release."""
        def fn(doc):
            if doc["status"] != RUNNING:
                return doc
            new = dict(doc)
            new["quiesce"] = {"transfer_id": transfer_id, "at": self.clock()}
            return new
        return self._transition("quiesce", fn)

    def quiesce_requested(self) -> bool:
        doc = self.read()
        return bool(doc and doc.get("quiesce") and doc["owner"] == self.owner)

    def begin_handoff(self, transfer_id: str, dest: str) -> dict:
        def fn(doc):
            self._check_owner(doc, allow=(RUNNING,))
            new = dict(doc)
            new.update(status=HANDOFF, quiesce=None,
                       handoff={"transfer_id": transfer_id, "dest": dest, "state": "preparing",
                                "source_generation": doc["generation"]})
            return new
        return self._transition("handoff-begin", fn,
                                confirm=lambda d: (d["handoff"] or {}).get("transfer_id")
                                == transfer_id)

    def set_handoff_state(self, transfer_id: str, state: str, **extra) -> dict:
        def fn(doc):
            self._check_owner(doc, allow=(HANDOFF,))
            if doc["handoff"]["transfer_id"] != transfer_id:
                raise NotAllowed("another transfer is recorded on this run")
            new = dict(doc)
            new["handoff"] = {**doc["handoff"], "state": state, **extra}
            return new
        return self._transition(f"handoff-{state}", fn,
                                confirm=lambda d: (d["handoff"] or {}).get("state") == state)

    def mark_transferred(self, transfer_id: str) -> dict:
        """Fence the source: after this nobody, including this process, can publish to it."""
        def fn(doc):
            if doc["status"] == TRANSFERRED and doc["handoff"]["transfer_id"] == transfer_id:
                return doc
            self._check_owner(doc, allow=(HANDOFF,))
            if doc["handoff"]["transfer_id"] != transfer_id:
                raise NotAllowed("another transfer is recorded on this run")
            new = dict(doc)
            new.update(status=TRANSFERRED, owner=None, lease_expires=0.0,
                       generation=doc["generation"] + 1,
                       handoff={**doc["handoff"], "state": "transferred"})
            return new
        new = self._transition("handoff-transferred", fn,
                               confirm=lambda d: d["status"] == TRANSFERRED)
        return new

    def activate_incoming(self, transfer_id: str, generation: int) -> dict:
        """Destination side: incoming -> idle, only for the matching committed transfer."""
        def fn(doc):
            if doc["status"] != INCOMING:
                if doc["status"] in (IDLE, RUNNING) and (doc["handoff"] or {}).get(
                        "transfer_id") == transfer_id:
                    return doc
                raise NotAllowed(f"the destination is {doc['status']}, not incoming")
            if (doc["handoff"] or {}).get("transfer_id") != transfer_id:
                raise NotAllowed("the destination was created by another transfer")
            new = dict(doc)
            new.update(status=IDLE, generation=max(doc["generation"], generation))
            new["handoff"] = {**doc["handoff"], "state": "activated"}
            return new
        return self._transition("handoff-activate", fn,
                                confirm=lambda d: d["status"] != INCOMING)


def renewer(auth: Authority, every: float | None = None):
    """A daemon thread that renews the lease until stopped; returns the stop Event."""
    stop = threading.Event()
    period = every or auth.lease_s / 3

    def run():
        while not stop.wait(period):
            try:
                auth.renew()
            except SpillError:
                return
            except Exception:
                continue
    threading.Thread(target=run, daemon=True).start()
    return stop
