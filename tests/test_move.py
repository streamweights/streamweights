# no-mlx-needed
"""spill move and resume <uri>: a controlled handoff, recoverable at every step, never two
writable authorities. Local destination directories and a MinIO bucket."""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from streamweights.errors import SpillError
from streamweights.project import control as ctl
from streamweights.project import coordinator as CO
from streamweights.project import move as M
from streamweights.project import plan as P
from streamweights.project import remote as RM
from streamweights.project.common import read_json, sha_file
from streamweights.project.runstate import RunState, tree_digest

from .test_project_build import FakeExecutor, make_project, pytestmark  # noqa: F401

BUCKET = os.environ.get("SPILL_TEST_S3_BUCKET")
DESTS = ["local"] + (["s3"] if BUCKET else [pytest.param("s3", marks=pytest.mark.skip(
    reason="no S3 endpoint: set SPILL_TEST_S3_BUCKET (and AWS_ENDPOINT_URL, credentials)"))])
HERE = Path(__file__).parent


@pytest.fixture(params=DESTS)
def dest(request, tmp_path):
    if request.param == "local":
        return str(tmp_path / "elsewhere" / "proj")
    return f"s3://{BUCKET}/mv-{uuid.uuid4().hex[:10]}"


def stopped_project(tmp_path):
    proj = make_project(tmp_path)
    plan = P.make_plan(proj, engine="torch-cpu")
    co = CO.Coordinator(proj, FakeExecutor(interrupt_train_at=5), "torch-cpu", say=lambda s: None)
    assert co.build(plan).status == "stopped"
    return proj, CO.list_runs(proj)[0]["run_id"]


def dest_state(dest):
    return f"{dest}/.spill"


def dest_doc(dest, rid):
    return {d["run_id"]: d for d in CO.list_runs_at(dest_state(dest))}.get(rid)


def acquirable(state_root, rid, tmp_path, who):
    """Can a fresh writer take this run right now? (It takes it, so use once per state.)"""
    rs = RunState.open(state_root, rid, tmp_path / f"w-{who}")
    try:
        rs.acquire(heartbeat=False)
        return True
    except SpillError:
        return False


def resume_at(dest, tmp_path, ex=None):
    remote = RM.is_remote(dest)
    if remote:
        project = RM.working_copy(dest)
        RM.pull_project(dest, project)
    else:
        project = Path(dest)
    co = CO.Coordinator(project, ex or FakeExecutor(), "torch-cpu", say=lambda s: None,
                        state_root_uri=dest_state(dest), remote_project=dest if remote else None)
    rid = sorted(d["run_id"] for d in CO.list_runs_at(dest_state(dest))
                 if d["status"] in ("idle", "running"))[-1]
    return co.resume(rid), project


def test_move_then_resume_continues_the_same_run_from_the_committed_checkpoint(dest, tmp_path):
    proj, rid = stopped_project(tmp_path)
    src_before = read_json(proj / ".spill" / "runs" / rid / "control.json")
    r = M.move(proj, dest, say=lambda s: None)
    assert r.files > 5 and r.bytes > 0
    src = read_json(proj / ".spill" / "runs" / rid / "control.json")
    assert src["status"] == "transferred" and src["handoff"]["dest"] == dest.rstrip("/")
    assert src["generation"] > src_before["generation"] and src["checkpoint"] == src_before["checkpoint"]
    d = dest_doc(dest, rid)
    assert d["status"] == "idle" and d["checkpoint"] == src_before["checkpoint"]
    assert d["handoff"]["state"] == "activated" and d["generation"] >= src["generation"]
    ex = FakeExecutor()
    out, project = resume_at(dest, tmp_path, ex)
    assert out.status == "completed" and out.run_id == rid and ex.restored_from == 4
    assert ex.calls == ["train", "eval:untrained", "eval:trained"]
    done = dest_doc(dest, rid)
    assert done["status"] == "completed"
    if RM.is_remote(dest):                                    # the snapshot was published back
        from streamweights.portable.store import Store
        assert Store(dest).exists(f"runs/{rid}/manifest.json")
    # the source stays fenced: nothing can take it, and the bytes are still there
    assert not acquirable(str(proj / ".spill"), rid, tmp_path, "src")
    assert (proj / "data" / "train.jsonl").exists() and (proj / ".spill" / "runs" / rid / "payloads").exists()


@pytest.mark.parametrize("point", ["move_mid_copy", "move_after_copy", "move_after_verify",
                                   "move_before_fence", "move_after_fence", "move_after_activate"])
def test_an_interrupted_handoff_recovers_and_two_places_are_never_writable(dest, tmp_path, point):
    proj, rid = stopped_project(tmp_path)
    p = subprocess.run([sys.executable, str(HERE / "move_worker.py"), str(proj), dest],
                       capture_output=True, text=True, cwd=str(HERE.parent),
                       env={**os.environ, f"SPILL_HOOK_{point}": "crash"})
    assert p.returncode == 137, p.stderr[-400:]
    src = read_json(proj / ".spill" / "runs" / rid / "control.json")
    d = dest_doc(dest, rid)
    # the invariant at the moment of the kill: neither place can run unless the handoff finished
    if point in ("move_mid_copy", "move_after_copy", "move_after_verify", "move_before_fence"):
        assert src["status"] == "handoff" and d is None or d["status"] == "incoming"
        assert not acquirable(str(proj / ".spill"), rid, tmp_path, "src-mid")      # held by the mover
        if d:
            assert not acquirable(dest_state(dest), rid, tmp_path, "dst-mid")
    elif point == "move_after_fence":
        assert src["status"] == "transferred" and d["status"] == "incoming"
        assert not acquirable(str(proj / ".spill"), rid, tmp_path, "src-f")
        assert not acquirable(dest_state(dest), rid, tmp_path, "dst-f")            # fenced and not yet active
    else:
        assert src["status"] == "transferred" and d["status"] == "idle"
        assert not acquirable(str(proj / ".spill"), rid, tmp_path, "src-a")
    # the same command, run again, finishes the recorded transfer
    r = M.move(proj, dest, say=lambda s: None, wait_s=5)
    src2 = read_json(proj / ".spill" / "runs" / rid / "control.json")
    d2 = dest_doc(dest, rid)
    assert src2["status"] == "transferred" and d2["status"] == "idle"
    assert src2["handoff"]["transfer_id"] == d2["handoff"]["transfer_id"]
    if point != "move_mid_copy":
        assert r.resumed
    assert not acquirable(str(proj / ".spill"), rid, tmp_path, "src-end")
    ex = FakeExecutor()
    out, _ = resume_at(dest, tmp_path, ex)
    assert out.status == "completed" and ex.restored_from == 4


def test_a_running_writer_is_quiesced_at_its_next_committed_boundary(tmp_path):
    proj = make_project(tmp_path)
    plan = P.make_plan(proj, engine="torch-cpu")
    state = str(proj / ".spill")
    rs = RunState.create(state, "run-live", plan.identity, proj / ".spill")
    rs.acquire(heartbeat=False)
    other = ctl.Authority(rs.backend, owner="mover-x")
    other.request_quiesce("t1")
    assert rs.quiesce_requested()
    rs.release()
    assert rs.doc()["status"] == "idle" and rs.doc()["owner"] is None


def test_a_move_refuses_when_the_writer_never_stops(tmp_path):
    proj = make_project(tmp_path)
    plan = P.make_plan(proj, engine="torch-cpu")
    rs = RunState.create(str(proj / ".spill"), "run-live", plan.identity, proj / ".spill")
    rs.acquire(heartbeat=False)
    with pytest.raises(SpillError) as e:
        M.move(proj, str(tmp_path / "dst"), say=lambda s: None, wait_s=0.5)
    assert "still being written" in e.value.message
    assert rs.doc()["status"] == "running"                     # nothing was fenced


def test_a_changed_file_during_the_copy_aborts_without_fencing(tmp_path, monkeypatch):
    proj, rid = stopped_project(tmp_path)
    real = M.hooks.fire

    def fire(name):
        if name == "move_after_copy":
            (proj / "data" / "train.jsonl").write_bytes(b"changed\n")
        real(name)
    monkeypatch.setattr(M.hooks, "fire", fire)
    with pytest.raises(SpillError) as e:
        M.move(proj, str(tmp_path / "dst"), say=lambda s: None)
    assert "changed while the project was being copied" in e.value.message
    assert read_json(proj / ".spill" / "runs" / rid / "control.json")["status"] == "handoff"


def test_resume_names_the_differences_when_the_weights_are_not_the_run_s(tmp_path, monkeypatch):
    proj, rid = stopped_project(tmp_path)
    from streamweights.project import modelid
    real = modelid.student_identity

    def other(model, fetch=True):
        i = real(model, fetch)
        i = {**i, "files": {**i["files"], "model.safetensors": {"sha256": "0" * 64, "bytes": 1}}}
        return i
    monkeypatch.setattr(P.modelid, "student_identity", other)
    co = CO.Coordinator(proj, FakeExecutor(), "torch-cpu", say=lambda s: None)
    with pytest.raises(SpillError) as e:
        co.resume(rid)
    assert "differs in" in e.value.message and "model" in e.value.message


def test_pulling_a_project_rejects_a_corrupted_file(tmp_path):
    proj, rid = stopped_project(tmp_path)
    dst = str(tmp_path / "elsewhere" / "proj")
    M.move(proj, dst, say=lambda s: None)
    (Path(dst) / "data" / "val.jsonl").write_bytes(b"corrupt\n")
    with pytest.raises(SpillError) as e:
        RM.pull_project(dst, tmp_path / "wc")
    assert "does not match the transfer manifest" in e.value.message


def test_completed_runs_move_without_becoming_trainable(tmp_path):
    proj = make_project(tmp_path)
    plan = P.make_plan(proj, engine="torch-cpu")
    out = CO.Coordinator(proj, FakeExecutor(), "torch-cpu", say=lambda s: None).build(plan)
    before = tree_digest(out.snapshot)
    dst = str(tmp_path / "moved")
    M.move(proj, dst, say=lambda s: None)
    assert tree_digest(Path(dst) / "runs" / out.run_id) == before and tree_digest(out.snapshot) == before
    d = dest_doc(dst, out.run_id)
    assert d["status"] == "completed"
    with pytest.raises(SpillError):
        RunState.open(dest_state(dst), out.run_id, tmp_path / "w").acquire()
    src = read_json(proj / ".spill" / "runs" / out.run_id / "control.json")
    assert src["status"] == "completed" and src["locations"][-1]["moved_to"] == dst


def test_a_stale_writer_cannot_publish_to_the_source_after_the_handoff(dest, tmp_path):
    proj, rid = stopped_project(tmp_path)
    stale = RunState.open(str(proj / ".spill"), rid, tmp_path / "stale-work")
    stale.auth.lease_s = 0.05
    stale.acquire(heartbeat=False)                       # a writer that then stalls
    import time
    time.sleep(0.2)
    M.move(proj, dest, say=lambda s: None, wait_s=5)
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "f").write_text("x")
    with pytest.raises(SpillError) as e:
        stale.publish_checkpoint(ck, 99, {"step": 99})
    assert "handed to" in e.value.message or "superseded" in e.value.message
    with pytest.raises(SpillError):
        stale.complete(ck)
    d = dest_doc(dest, rid)
    assert d["status"] == "idle" and d["checkpoint"]["step"] == 4       # the destination is untouched
    assert read_json(proj / ".spill" / "runs" / rid / "control.json")["checkpoint"]["step"] == 4
