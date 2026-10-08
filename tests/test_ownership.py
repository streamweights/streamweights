# no-mlx-needed
"""Ownership and publication: deterministic hooks, separate processes, local disk and MinIO."""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from streamweights.errors import SpillError
from streamweights.project import caps, control as ctl, payloads as pl
from streamweights.project.runstate import RunState

HERE = Path(__file__).parent
BUCKET = os.environ.get("SPILL_TEST_S3_BUCKET")
BACKENDS = ["local"] + (["s3"] if BUCKET else [pytest.param("s3", marks=pytest.mark.skip(
    reason="no S3 endpoint: set SPILL_TEST_S3_BUCKET (and AWS_ENDPOINT_URL, credentials) "
           "to run the MinIO cases"))])


@pytest.fixture(params=BACKENDS)
def backend(request, tmp_path):
    if request.param == "local":
        root = str(tmp_path / "state")
    else:
        root = f"s3://{BUCKET}/t-{uuid.uuid4().hex[:10]}"
    return request.param, root


def make_run(backend, tmp_path, run="r1"):
    kind, root = backend
    return RunState.create(root, run, "identity-1", tmp_path / "work")


def spawn(backend, owner, ops, env=None, lease=None, lock_timeout=None, run="r1"):
    kind, root = backend
    spec = {"backend": f"{kind}::{root}", "owner": owner, "ops": ops, "lease": lease,
            "lock_timeout": lock_timeout, "run": run}
    e = {**os.environ, **(env or {})}
    return subprocess.Popen([sys.executable, str(HERE / "ownership_worker.py"), json.dumps(spec)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=e,
                            cwd=str(HERE.parent))


def finish(p, timeout=60):
    out, err = p.communicate(timeout=timeout)
    lines = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    return lines, err


def wait_for(path: Path, timeout=30):
    t = time.monotonic()
    while not path.exists():
        assert time.monotonic() - t < timeout, f"{path} never appeared"
        time.sleep(0.02)


def doc_of(backend, tmp_path):
    return RunState.open(backend[1], "r1", tmp_path / "w2").doc()


def test_a_pauses_outside_critical_section_lease_expires_b_publishes_a_rejected(backend, tmp_path):
    rs = make_run(backend, tmp_path)
    gate, reached = tmp_path / "gate", tmp_path / "reached"
    a = spawn(backend, "A", [["acquire"], ["ckpt", 1], ["ckpt", 2]], lease=1.0,
              env={"SPILL_HOOK_checkpoint_before_control": f"pause:{gate}:{reached}"})
    # A's first checkpoint already pauses before its control update: nothing is committed yet
    wait_for(reached)
    assert rs.doc()["checkpoint"] is None
    time.sleep(1.3)                                        # A's lease expires
    lines_b, _ = finish(spawn(backend, "B", [["acquire"], ["ckpt", 7], ["stage", "train"]]))
    assert [l["ok"] for l in lines_b] == [True, True, True]
    before = rs.doc()
    gate.write_text("go")
    lines_a, _ = finish(a)
    ck_results = [l for l in lines_a if l["op"] == "ckpt"]
    assert ck_results[0]["ok"] is False and ck_results[0]["error"] == "Superseded"
    after = rs.doc()
    assert after["checkpoint"] == before["checkpoint"] and after["checkpoint"]["step"] == 7
    assert after["generation"] == before["generation"] == 2 and after["owner"] == "B"
    assert after["stages"] == before["stages"]
    # B's payload intact; A's orphan exists but is not referenced
    pl.verify_payload(rs.store, after["checkpoint"]["dir"], after["checkpoint"]["manifest_sha256"])
    dirs = [d for d in rs.store.ls("payloads") if d.startswith("ckpt-")]
    assert len(dirs) == 2 and after["checkpoint"]["dir"].split("/")[-1] in dirs
    assert [h["event"] for h in after["history"]].count("acquire") == 2


def test_two_writers_race_to_acquire_exactly_one_wins(backend, tmp_path):
    rs = make_run(backend, tmp_path)
    gate = tmp_path / "gate"
    hook = {"SPILL_HOOK_transition_validated": f"pause:{gate}:{tmp_path}/reached-%s"}
    procs = []
    for i in range(3):
        env = {"SPILL_HOOK_transition_validated": f"pause:{gate}:{tmp_path}/reached{i}"}
        if backend[0] == "local":
            env = {}           # the lock serializes them; no pause needed (and none possible)
        procs.append(spawn(backend, f"W{i}", [["acquire"], ["ckpt", 1]], env=env, lease=30,
                           lock_timeout=None))
    if backend[0] == "s3":
        for i in range(3):
            wait_for(tmp_path / f"reached{i}")
        gate.write_text("go")
    results = [finish(p)[0] for p in procs]
    winners = [r for r in results if r[0]["ok"]]
    assert len(winners) == 1
    d = rs.doc()
    assert d["generation"] == 1 and d["owner"] == winners[0][0]["owner"]
    assert winners[0][1]["ok"] is True and d["checkpoint"]["step"] == 1
    for r in results:
        if not r[0]["ok"]:
            assert r[0]["error"] == "NotAllowed" and r[1]["ok"] is False


def test_local_a_holds_lock_b_gets_contention_then_a_stale_after_b_acquires(tmp_path):
    backend = ("local", str(tmp_path / "state"))
    rs = make_run(backend, tmp_path)
    gate, reached = tmp_path / "gate", tmp_path / "reached"
    a = spawn(backend, "A", [["acquire"], ["ckpt", 1], ["ckpt", 2]], lease=1.0,
              env={"SPILL_HOOK_transition_validated": f"pause:{gate}:{reached}:2"})
    wait_for(reached)                    # A holds the lock, validated, before its checkpoint write
    b_lines, _ = finish(spawn(backend, "B", [["acquire"]], lock_timeout=0.3))
    assert b_lines[0]["ok"] is False and b_lines[0]["error"] == "LockContended"
    gate.write_text("go")
    a_lines, _ = finish(a)
    assert [l["ok"] for l in a_lines] == [True, True, True]
    time.sleep(1.3)                                       # A's lease expires
    b2, _ = finish(spawn(backend, "B", [["acquire"], ["ckpt", 9]]))
    assert b2[0]["ok"] and b2[0]["generation"] > a_lines[0]["generation"]
    a2, _ = finish(spawn(backend, "A", [["ckpt", 99]], run="r1"))
    assert a2[0]["ok"] is False and a2[0]["error"] == "Superseded"
    assert rs.doc()["checkpoint"]["step"] == 9


def test_s3_a_validates_pauses_before_conditional_write_b_publishes(tmp_path):
    if not BUCKET:
        pytest.skip("no S3 endpoint: set SPILL_TEST_S3_BUCKET")
    backend = ("s3", f"s3://{BUCKET}/t-{uuid.uuid4().hex[:10]}")
    rs = make_run(backend, tmp_path)
    gate, reached = tmp_path / "gate", tmp_path / "reached"
    # A: acquire (no hook yet for the first transition), then pause validated-before-write on
    # its checkpoint transition. The hook fires on every transition, so pause on the second.
    a = spawn(backend, "A", [["acquire"], ["ckpt", 1]], lease=1.0,
              env={"SPILL_HOOK_transition_validated": f"pause:{gate}:{reached}:2"})
    wait_for(reached)        # A owns generation 1 and has validated its checkpoint transition
    assert rs.doc()["checkpoint"] is None
    time.sleep(1.3)                                        # A's lease runs out
    b1, _ = finish(spawn(backend, "B", [["acquire"], ["ckpt", 5]], lease=30))
    assert [l["ok"] for l in b1] == [True, True]
    gate.write_text("go")
    lines, _ = finish(a)
    # A's conditional write carries the old ETag: it fails, and A cannot retry into B's
    # generation (the reread shows generation 2 owned by B)
    assert lines[0]["ok"] is True and lines[1]["ok"] is False and lines[1]["error"] == "Superseded"
    d = rs.doc()
    assert d["owner"] == "B" and d["checkpoint"]["step"] == 5 and d["generation"] == 2


def test_lost_response_reconciles_without_replay(backend, tmp_path):
    rs = make_run(backend, tmp_path)
    if backend[0] == "local":
        pytest.skip("a lost response is an object-store failure: a local replace is not uncertain")
    lines, _ = finish(spawn(backend, "A", [["acquire"], ["ckpt", 3]], lease=30,
                            env={"SPILL_HOOK_cas_response": "lose"}))
    assert [l["ok"] for l in lines] == [True, True]
    d = rs.doc()
    events = [h["event"] for h in d["history"]]
    assert events.count("acquire") == 1 and events.count("checkpoint") == 1
    assert d["checkpoint"]["seq"] == 1 and d["generation"] == 1


def test_renewal_publication_race_preserves_newer_state(tmp_path):
    if not BUCKET:
        pytest.skip("no S3 endpoint: set SPILL_TEST_S3_BUCKET")
    backend = ("s3", f"s3://{BUCKET}/t-{uuid.uuid4().hex[:10]}")
    rs = make_run(backend, tmp_path)
    rs.acquire(heartbeat=False)
    other = ctl.Authority(rs.backend, owner=rs.auth.owner)
    other.gen = rs.auth.gen
    d = tmp_path / "ck"
    d.mkdir()
    (d / "f").write_text("x")

    real_cas = rs.backend.cas
    state = {"n": 0}

    def cas(token, doc):
        if state["n"] == 0 and doc["history"][-1]["event"] == "renew":
            state["n"] = 1
            ref = pl.write_payload(rs.store, "ckpt", rs.auth.gen, 1, "inner", {"f": d / "f"})
            other.publish_checkpoint({**ref, "step": 4, "cursor": {}})       # lands first
        return real_cas(token, doc)
    rs.backend.cas = cas
    rs.auth.renew()                                        # conflicts, reconciles, retries
    cur = rs.doc()
    assert cur["checkpoint"]["step"] == 4 and cur["owner"] == rs.auth.owner
    assert cur["generation"] == 1
    # a second publication with a lower sequence is refused (no pointer regression)
    ref2 = pl.write_payload(rs.store, "ckpt", rs.auth.gen, 1, "late", {"f": d / "f"})
    with pytest.raises(SpillError):
        rs.auth.publish_checkpoint({**ref2, "step": 1, "cursor": {}})
    assert rs.doc()["checkpoint"]["step"] == 4


def test_stale_writer_cannot_publish_stage_or_complete(backend, tmp_path):
    rs = make_run(backend, tmp_path)
    gate, reached = tmp_path / "gate", tmp_path / "reached"
    a = spawn(backend, "A", [["acquire"], ["stage", "train"], ["complete"]], lease=1.0,
              env={"SPILL_HOOK_stage_before_control": f"pause:{gate}:{reached}"})
    wait_for(reached)
    time.sleep(1.3)
    lines_b, _ = finish(spawn(backend, "B", [["acquire"], ["stage", "train"], ["complete"]]))
    assert [l["ok"] for l in lines_b] == [True, True, True]
    done = rs.doc()
    assert done["status"] == "completed"
    proj = tmp_path / "proj"
    snap = rs.install_snapshot(proj)
    before = {p.name: p.read_text() for p in snap.iterdir() if p.is_file()}
    gate.write_text("go")
    lines_a, _ = finish(a)
    assert [l["ok"] for l in lines_a] == [True, False, False]       # acquire ok, stage and complete rejected
    assert lines_a[1]["error"] == "Superseded" or "NotAllowed"
    after = rs.doc()
    assert after["stages"] == done["stages"] and after["completed"] == done["completed"]
    assert {p.name: p.read_text() for p in snap.iterdir() if p.is_file()} == before
    assert "report by B" in (snap / "report.md").read_text()
    # an installed snapshot cannot be re-installed over by anyone
    assert rs.install_snapshot(proj) == snap


def test_capability_refusals_before_any_publication(tmp_path):
    pytest.importorskip("botocore", reason="S3 needs the cloud extra: pip install -e .[cloud]")
    class NoCas:
        """An S3 look-alike that ignores conditional headers."""
        def put_object(self, **kw):
            return {"ETag": '"abc"'}

        def delete_object(self, **kw):
            pass
    b = ctl.S3Backend("s3://bkt/run", client=NoCas())
    with pytest.raises(SpillError) as e:
        caps.check_s3(b)
    assert "no conditional create" in e.value.message
    with pytest.raises(SpillError) as e:
        caps.check_local(tmp_path / "x", fstype=lambda p: "nfs4")
    assert "network" in e.value.message
    with pytest.raises(SpillError) as e:
        caps.check_local(tmp_path / "y", fstype=lambda p: "weirdfs")
    assert "not verified" in e.value.message
    assert not (tmp_path / "x" / "control.json").exists()
    assert caps.check_local(tmp_path / "z")["backend"] == "local"


def test_never_deleted_or_recreated_and_completed_is_final(backend, tmp_path):
    rs = make_run(backend, tmp_path)
    again = ctl.Authority.create(rs.backend, "r1", "other-identity")      # create again: no reset
    assert again["identity"] == "identity-1"
    rs.acquire(heartbeat=False)
    snap = tmp_path / "s"
    snap.mkdir()
    (snap / "report.md").write_text("r")
    rs.complete(snap)
    with pytest.raises(SpillError):
        rs.auth.publish_checkpoint({"seq": 5, "generation": 1, "step": 1, "dir": "x",
                                    "manifest_sha256": "y"})
    other = ctl.Authority(rs.backend)
    with pytest.raises(SpillError) as e:
        other.acquire()
    assert "completed" in e.value.message


@pytest.mark.parametrize("point,committed", [
    ("payload_mid_write", 1), ("payload_before_manifest", 1), ("checkpoint_before_control", 1),
    ("checkpoint_after_control", 2)])
def test_killed_during_checkpoint_publication_recovers_last_accepted(backend, tmp_path, point,
                                                                     committed):
    """Process termination (os._exit) at each point of publishing checkpoint 2. The recovery
    is the one the control object points at, intact, with its step and cursor. (Termination of
    the process only: power loss of the machine is not tested.)"""
    rs = make_run(backend, tmp_path)
    # the hook fires once per ckpt: the 2nd ckpt dies, the 1st completes
    a = spawn(backend, "A", [["acquire"], ["ckpt", 10], ["ckpt", 20]], lease=1.0,
              env={f"SPILL_HOOK_{point}": "crash:2"})
    lines, _ = finish(a)
    assert a.returncode == 137 and [l["op"] for l in lines] == ["acquire", "ckpt"]
    time.sleep(1.2)
    b, _ = finish(spawn(backend, "B", [["acquire"], ["restore", 5]], lease=30))
    r = b[1]
    assert r["ok"] and r["step"] == committed * 10 and r["cursor"] == {"micro_batch_index": committed * 10}
    assert r["files"] == ["COMMIT", "PAYLOAD.json", "params.bin"] or \
        r["files"] == ["COMMIT", "params.bin"]
    assert r["params"].startswith(f"params-A-{committed * 10}")
    assert "last committed checkpoint" in r["recovery"] and "unknown" in r["recovery"]
    assert "at most one" not in r["recovery"]
    d = rs.doc()
    assert d["checkpoint"]["seq"] == committed
