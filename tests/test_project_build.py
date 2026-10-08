# no-mlx-needed
"""The coordinator: run identity, resume, immutable history. Stages run through a fake executor
(no model is loaded); the model identity is the real pinned qwen2.5:0.5b (hashed from disk)."""

import csv
import json
import os
import shutil
from pathlib import Path

import pytest

from streamweights.errors import SpillError
from streamweights.project import config as C
from streamweights.project import contract as K
from streamweights.project import control as ctl
from streamweights.project import coordinator as CO
from streamweights.project import plan as P
from streamweights.project.common import read_json, read_jsonl, write_json, write_jsonl
from streamweights.project.index import records, render
from streamweights.project.init import create_project
from streamweights.project.runstate import RunState, make_writable, tree_digest
from streamweights.project.stagefns import StageOutcome

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "qwen2.5-0.5b" / "bf16-st"
pytestmark = pytest.mark.skipif(
    not (MODEL / "model.safetensors").exists() and os.environ.get("SPILL_REQUIRE_FIXTURE_MODEL") != "1",
    reason="needs the pinned qwen2.5:0.5b (its identity is hashed from disk)")

CLASSES = ("billing", "shipping", "login", "refund")


def make_project(tmp_path, name="proj", n=120, seed=0, system=None):
    p = tmp_path / f"{name}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for i in range(n):
            w.writerow([f"ticket {i} about {CLASSES[i % 4]} problem number {i}", CLASSES[i % 4]])
    sysf = None
    if system:
        sysf = tmp_path / "system.txt"
        sysf.write_text(system)
    create_project(p, tmp_path / name, {"input": "text", "output": "label"}, "classification",
                   system_file=sysf, seed=seed)
    return tmp_path / name


class FakeExecutor:
    """Runs a stage without a model. `calls` records every stage it was asked to run."""
    name = "fake"

    def __init__(self, interrupt_train_at=None, train_steps=6):
        self.calls = []
        self.interrupt_train_at = interrupt_train_at
        self.train_steps = train_steps
        self.restored_from = None

    def run(self, desc, ctx):
        self.calls.append(desc.id)
        out = ctx.out_dir
        out.mkdir(parents=True, exist_ok=True)
        cfg = C.parse((ctx.inputs_dir / "streamweights.toml").read_bytes())
        val = read_jsonl(ctx.inputs_dir / "val.jsonl")
        labels = cfg["contract"]["labels"]
        prod = [{"engine": "fake", "hardware": "cpu", "numerics": {"base": "float32"},
                 "os": "test", "host": "h", "quanta": "all"}]
        if desc.kind in ("baseline", "eval"):
            right = {"baseline": 0.9, "untrained": 0.25, "trained": 0.8, "teacher": 0.7}[
                "baseline" if desc.kind == "baseline" else desc.params["comparator"]]
            cmp = "baseline" if desc.kind == "baseline" else desc.params["comparator"]
            preds, items = [], []
            for i, r in enumerate(val):
                text = r["output"] if (i / len(val)) < right else \
                    next(l for l in labels if l != r["output"])
                sc = K.score_class(text, r["output"], labels)
                items.append(sc)
                preds.append({"id": r["id"], "gold": r["output"], "text": text, "finish_reason": "stop",
                              "pred": sc["pred"], "correct": sc["correct"], "valid": sc["valid"],
                              "failed": False, "comparator": cmp})
            m = K.agg_class(items, [r["output"] for r in val], labels)
            m["truncated"] = 0
            write_jsonl(out / "predictions.jsonl", preds)
            write_json(out / "metrics.json", m)
            return StageOutcome("done", m, prod, 0.1)
        if desc.kind == "train":
            start = 0
            lat = ctx.stage_dir / "state" / "ckpt" / "LATEST"
            if lat.exists():
                start = json.loads(lat.read_text())["step"]
                d = ctx.stage_dir / "state" / "ckpt" / f"step-{start:09d}"
                assert (d / "COMMIT").exists() and (d / "params.safetensors").exists()
                self.restored_from = start
            for step in range(start + 1, self.train_steps + 1):
                if self.interrupt_train_at and step == self.interrupt_train_at and start == 0:
                    return StageOutcome("interrupted", {"step": step - 1}, prod, 0.1)
                if step % 2 == 0:
                    d = ctx.stage_dir / "state" / "ckpt" / f"step-{step:09d}"
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "params.safetensors").write_bytes(os.urandom(64))
                    (d / "opt.safetensors").write_bytes(os.urandom(64))
                    state = {"data_cursor": {"micro_batch_index": step * 2, "step": step}}
                    (d / "state.json").write_text(json.dumps(state))
                    (d / "COMMIT").write_text(json.dumps({"step": step}))
                    ctx.on_checkpoint(step, d, state)
            (out / "adapter").mkdir()
            (out / "adapter" / "adapter_model.safetensors").write_bytes(b"fake adapter")
            (out / "adapter" / "adapter_config.json").write_text("{}")
            write_json(out / "training.manifest.json", {"rows": 1})
            write_jsonl(out / "training.rows.jsonl", [{"id": "x", "input": "y", "target": "z", "origin": "own"}])
            return StageOutcome("done", {"steps": self.train_steps, "final_loss": 0.5}, prod, 0.1)
        raise AssertionError(desc.kind)


def build(project, executor=None, **kw):
    plan = P.make_plan(project, engine="torch-cpu")
    co = CO.Coordinator(project, executor or FakeExecutor(), "torch-cpu", say=lambda s: None, **kw)
    return plan, co


def test_build_completes_and_the_snapshot_is_complete_and_read_only(tmp_path):
    proj = make_project(tmp_path)
    plan, co = build(proj)
    out = co.build(plan)
    assert out.status == "completed" and out.snapshot == proj / "runs" / out.run_id
    for f in ("report.md", "manifest.json", "results.json", "disagreements.jsonl", "inputs/train.jsonl",
              "predictions/trained.jsonl", "predictions/baseline.jsonl", "artifacts/adapter/adapter_config.json"):
        assert (out.snapshot / f).exists(), f
    m = read_json(out.snapshot / "manifest.json")
    assert m["identity"] == plan.identity and m["models"]["student"]["revision"]
    assert m["models"]["student"]["files"]["model.safetensors"]["sha256"].startswith("fdf756fa")
    assert not any("dir" in v for v in m["models"].values() if v)
    assert not (out.snapshot / "inputs" / "test.jsonl").exists()          # the test rows are not copied in
    assert m["final_test"]["used_by_this_run"] is False
    with pytest.raises(PermissionError):
        (out.snapshot / "report.md").write_text("changed")
    ctl_doc = RunState.open(str(proj / ".spill"), out.run_id, proj / ".spill").doc()
    assert ctl_doc["status"] == "completed" and ctl_doc["completed"]["manifest_sha256"]
    idx = (proj / "REPORT.md").read_text()
    assert out.run_id in idx and "Not used yet" in idx


def test_same_inputs_do_not_rebuild_and_new_run_references_its_parent(tmp_path):
    proj = make_project(tmp_path)
    plan, co = build(proj)
    first = co.build(plan)
    ex2 = FakeExecutor()
    _, co2 = build(proj, ex2)
    again = co2.build(plan)
    assert again.status == "already-complete" and again.run_id == first.run_id and ex2.calls == []
    new = co2.build(plan, new_run=True)
    assert new.status == "completed" and new.run_id != first.run_id
    assert read_json(new.snapshot / "manifest.json")["parent"] == first.run_id
    assert RunState.open(str(proj / ".spill"), new.run_id, proj / ".spill").doc()["parent"] == first.run_id


def test_changed_settings_or_data_make_a_new_run_and_leave_old_snapshots_byte_identical(tmp_path):
    proj = make_project(tmp_path)
    plan, co = build(proj)
    a = co.build(plan)
    before = tree_digest(a.snapshot)
    cfg = C.load(proj)
    cfg["training"]["epochs"] = 3.0
    C.save(proj, cfg)
    plan2, co2 = build(proj)
    assert plan2.identity != plan.identity
    b = co2.build(plan2)
    assert b.run_id != a.run_id
    rows = read_jsonl(proj / "data" / "train.jsonl")
    write_jsonl(proj / "data" / "train.jsonl", rows[:-3])           # changed data
    plan3, co3 = build(proj)
    c = co3.build(plan3)
    assert len({a.run_id, b.run_id, c.run_id}) == 3
    assert tree_digest(a.snapshot) == before
    ids = [r["run_id"] for r in records(proj, "runs")]
    assert ids == sorted([a.run_id, b.run_id, c.run_id])


def test_prompt_system_text_and_schema_changes_are_new_runs(tmp_path):
    proj = make_project(tmp_path, system="Be brief.")
    plan, _ = build(proj)
    cfg = C.load(proj)
    cfg["task"]["system"] = "Be brief and polite."
    C.save(proj, cfg)
    assert P.make_plan(proj, engine="torch-cpu").identity != plan.identity


def test_resume_continues_the_same_run_from_the_committed_checkpoint(tmp_path):
    proj = make_project(tmp_path)
    ex = FakeExecutor(interrupt_train_at=5)
    plan, co = build(proj, ex)
    out = co.build(plan)
    assert out.status == "stopped" and ex.calls == ["baseline", "train"]
    rs = RunState.open(str(proj / ".spill"), CO.list_runs(proj)[0]["run_id"], proj / ".spill")
    d = rs.doc()
    assert d["status"] == "idle" and d["checkpoint"]["step"] == 4 and d["checkpoint"]["seq"] == 2
    assert "baseline" in d["stages"] and "train" not in d["stages"]
    rec = rs.recovery(ckpt_every=10)
    assert "step 4" in rec.line() and "unknown" in rec.line()
    ex2 = FakeExecutor()
    plan2, co2 = build(proj, ex2)
    out2 = co2.build(plan2)
    assert out2.status == "completed" and out2.run_id == rs.run_id and out2.resumed
    assert ex2.calls == ["train", "eval:untrained", "eval:trained"]       # baseline was accepted, not redone
    assert ex2.restored_from == 4
    done = RunState.open(str(proj / ".spill"), rs.run_id, proj / ".spill").doc()
    assert done["checkpoint"]["seq"] == 3 and done["checkpoint"]["step"] == 6
    assert [h["event"] for h in done["history"]].count("acquire") == 2


def test_resume_uses_the_frozen_inputs_not_the_live_project_files(tmp_path):
    proj = make_project(tmp_path)
    plan, co = build(proj, FakeExecutor(interrupt_train_at=3))
    assert co.build(plan).status == "stopped"
    rid = CO.list_runs(proj)[0]["run_id"]
    write_jsonl(proj / "data" / "train.jsonl", read_jsonl(proj / "data" / "train.jsonl")[:20])
    cfg = C.load(proj)
    cfg["training"]["lr"] = 0.5
    C.save(proj, cfg)
    ex = FakeExecutor()
    co2 = CO.Coordinator(proj, ex, "torch-cpu", say=lambda s: None)
    out = co2.resume(rid)
    assert out.status == "completed" and out.run_id == rid
    m = read_json(out.snapshot / "manifest.json")
    assert m["config_resolved"]["training"]["lr"] == 1e-4
    assert len(read_jsonl(out.snapshot / "inputs" / "train.jsonl")) > 20
    # the new inputs are a new run, and say what differs
    said = []
    plan2 = P.make_plan(proj, engine="torch-cpu")
    CO.Coordinator(proj, FakeExecutor(), "torch-cpu", say=said.append).build(plan2)
    assert plan2.identity != m["identity"]


def test_a_completed_run_cannot_resume_or_be_acquired(tmp_path):
    proj = make_project(tmp_path)
    plan, co = build(proj)
    out = co.build(plan)
    with pytest.raises(SpillError) as e:
        co.resume(out.run_id)
    assert "cannot resume training" in e.value.message
    with pytest.raises(SpillError):
        RunState.open(str(proj / ".spill"), out.run_id, proj / ".spill").acquire()


def test_a_stale_worker_cannot_complete_or_alter_a_finished_snapshot(tmp_path):
    proj = make_project(tmp_path)
    state = str(proj / ".spill")
    plan = P.make_plan(proj, engine="torch-cpu")
    stale = RunState.create(state, "run-stale", plan.identity, proj / ".spill")
    stale.auth.lease_s = 0.01
    stale.acquire(heartbeat=False)
    import time
    time.sleep(0.05)
    winner = RunState.open(state, "run-stale", proj / ".spill")
    winner.acquire(heartbeat=False)
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "report.md").write_text("winner")
    winner.complete(snap)
    dest = winner.install_snapshot(proj)
    before = tree_digest(dest)
    snap2 = tmp_path / "snap2"
    snap2.mkdir()
    (snap2 / "report.md").write_text("stale writer")
    with pytest.raises(SpillError):
        stale.complete(snap2)
    with pytest.raises(SpillError):
        stale.publish_stage("train", snap2, {"status": "done"})
    assert tree_digest(dest) == before
    assert winner.install_snapshot(proj) == dest and tree_digest(dest) == before


def test_final_test_labels_are_never_read_by_build_plan_or_init_inference(tmp_path, monkeypatch):
    proj = make_project(tmp_path)
    test_rows = read_jsonl(proj / "data" / "test.jsonl")
    secret = {r["output"] for r in test_rows}
    # poison the test file: any consumer that reads its labels would see the sentinel
    poisoned = [{**r, "output": "ZZZ_TEST_ONLY_LABEL"} for r in test_rows]
    write_jsonl(proj / "data" / "test.jsonl", poisoned)
    plan = P.make_plan(proj, engine="torch-cpu")
    assert "ZZZ_TEST_ONLY_LABEL" not in json.dumps(plan.cfg)
    seen = []
    real_read = P.read_jsonl

    def spy(path, *a, **k):
        seen.append(Path(path).name)
        return real_read(path, *a, **k)
    monkeypatch.setattr(P, "read_jsonl", spy)
    P.make_plan(proj, engine="torch-cpu")
    assert "test.jsonl" not in seen
    ex = FakeExecutor()
    co = CO.Coordinator(proj, ex, "torch-cpu", say=lambda s: None)
    out = co.build(P.make_plan(proj, engine="torch-cpu"))
    blob = "".join(p.read_text() for p in out.snapshot.rglob("*") if p.is_file()
                   and p.suffix in (".jsonl", ".json", ".md", ".toml"))
    assert "ZZZ_TEST_ONLY_LABEL" not in blob
    assert not (out.snapshot / "inputs" / "test.jsonl").exists()
    # the vocabulary and the schema come from the training rows
    assert set(plan.cfg["contract"]["labels"]) <= {c for c in CLASSES} and secret <= set(CLASSES)


def test_unusable_split_is_refused_by_build(tmp_path):
    from streamweights.cli import app
    from typer.testing import CliRunner
    p = tmp_path / "t.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for i in range(10):
            w.writerow([f"t{i}", CLASSES[i % 2]])
    create_project(p, tmp_path / "small", {"input": "text", "output": "label"}, "classification")
    r = CliRunner().invoke(app, ["build", str(tmp_path / "small")])
    assert r.exit_code == 1 and "cannot be built" in r.output and "Try:" in r.output
