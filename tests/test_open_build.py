# no-mlx-needed
"""An open-ended project through the coordinator: judged stage, the judge in the run's identity
and protocol, the teacher as judge by default, and compare refusing runs judged differently.
Stages run through a fake executor except the judge, which is the real stage with a scripted
model (no weights are loaded)."""

import csv
import os
from pathlib import Path

import pytest

from streamweights.project import compare as CM
from streamweights.project import config as C
from streamweights.project import coordinator as CO
from streamweights.project import judge as J
from streamweights.project import plan as P
from streamweights.project import scoring as SC
from streamweights.project import stagefns as SF
from streamweights.project.common import read_json, read_jsonl, write_json, write_jsonl
from streamweights.project.init import create_project
from streamweights.project.stagefns import StageOutcome

from .test_judge import ScriptedBackend
from .test_project_build import MODEL, FakeExecutor

pytestmark = pytest.mark.skipif(
    not (MODEL / "model.safetensors").exists() and os.environ.get("SPILL_REQUIRE_FIXTURE_MODEL") != "1",
    reason="needs the pinned qwen2.5:0.5b (its identity is hashed from disk)")


class OpenExecutor(FakeExecutor):
    def run(self, desc, ctx):
        if desc.kind == "judge":
            self.calls.append(desc.id)
            return SF.run_stage(desc, ctx)
        if desc.kind in ("baseline", "eval"):
            self.calls.append(desc.id)
            cfg = C.parse((ctx.inputs_dir / "streamweights.toml").read_bytes())
            val = read_jsonl(ctx.inputs_dir / "val.jsonl")
            cmp = "baseline" if desc.kind == "baseline" else desc.params["comparator"]
            items, preds = [], []
            for i, r in enumerate(val):
                text = r["output"] if cmp == "trained" or (cmp == "teacher" and i % 2) else "unrelated words"
                sc, rec = SC.score_row(cfg, None, None, text, r)
                items.append(sc)
                preds.append({"id": r["id"], "gold": r["output"], "text": text, "finish_reason": "stop",
                              "failed": False, "comparator": cmp, **rec})
            m = SC.aggregate(cfg, items, [r["output"] for r in val], None)
            m["truncated"] = 0
            ctx.out_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(ctx.out_dir / "predictions.jsonl", preds)
            write_json(ctx.out_dir / "metrics.json", m)
            return StageOutcome("done", m, [{"engine": "fake", "hardware": "cpu", "numerics": {},
                                             "os": "t", "host": "h", "quanta": "all"}], 0.1,
                                notes={"conditions": self.conditions})
        if desc.kind == "distill":
            self.calls.append(desc.id)
            ctx.out_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(ctx.out_dir / "teacher.jsonl",
                        [{"custom_id": r["id"], "completion": "teacher says " + r["output"]}
                         for r in read_jsonl(ctx.inputs_dir / "train.jsonl")])
            return StageOutcome("done", {"rows": 1}, [{"engine": "fake", "hardware": "cpu", "numerics": {},
                                                       "os": "t", "host": "h", "quanta": "all"}], 0.1)
        return super().run(desc, ctx)


def make_open(tmp_path, name="op", **kw):
    p = tmp_path / f"{name}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "out"])
        for i in range(60):
            w.writerow([f"What does passage {i} say about subject {i}?",
                        f"Passage {i} says that subject {i} is described at length." if i % 3 else f"Subject {i}"])
    create_project(p, tmp_path / name, {"input": "text", "output": "out"}, **kw)
    return tmp_path / name


@pytest.fixture
def scripted(monkeypatch):
    be = ScriptedBackend()
    monkeypatch.setattr(SF, "_backend", lambda: be)
    monkeypatch.setattr("streamweights.cli_build._row_producers", lambda p: [])
    monkeypatch.setattr(SF, "choose_engine_name", lambda: "torch-cpu")
    return be


def run(proj, **kw):
    plan = P.make_plan(proj, engine="torch-cpu")
    ex = OpenExecutor()
    return plan, ex, CO.Coordinator(proj, ex, "torch-cpu", say=lambda s: None, **kw).build(plan)


def test_open_project_without_a_judge_builds_with_overlap_and_recommends_one(tmp_path):
    proj = make_open(tmp_path)
    plan = P.make_plan(proj, engine="torch-cpu")
    assert [s.id for s in plan.stages] == ["baseline", "train", "eval:untrained", "eval:trained"]
    rec = plan.judge_rec
    assert rec and rec["tag"] in {t for t, _ in J.CANDIDATES} and rec["bytes"] and rec["tag"] != "qwen2.5:0.5b"
    text = "\n".join(P.render(plan))
    assert f"judge     judge: none named. Judged scores need a judge; recommended: {rec['tag']}" in text
    assert "scoring: judge: reference overlap against your outputs; judged scores need a judge" in text
    # nothing but the student is a download
    assert [d["model"] for d in plan.downloads] == ["qwen2.5:0.5b"]
    _, ex, out = run(proj)
    assert out.status == "completed" and "judge" not in ex.calls
    m = read_json(out.snapshot / "manifest.json")
    assert m["judge"] is None and m["metric_key"] == "token_f1" and "judge" not in m["protocol"]["fields"]
    assert "Reference overlap" in (out.snapshot / "report.md").read_text()
    assert {t["comparator"] for t in out.table} == {"baseline", "untrained", "trained"}


def test_teacher_judges_by_default_and_its_row_is_reference_only(tmp_path, scripted):
    proj = make_open(tmp_path, teacher="qwen2.5:1.5b")
    plan = P.make_plan(proj, engine="torch-cpu")
    assert plan.cfg["model"]["judge"] == "qwen2.5:1.5b" and plan.cfg["evaluation"]["metric"] == "judge_mean"
    assert plan.stages[-1].id == "judge" and plan.stages[-1].params["excluded"].keys() == {"teacher"}
    assert [d["model"] for d in plan.downloads].count("qwen2.5:1.5b") == 1       # one download, two roles
    _, ex, out = run(proj)
    assert out.status == "completed" and "judge" in ex.calls
    m = read_json(out.snapshot / "manifest.json")
    jm = m["judge"]
    assert "teacher" in jm["excluded"] and "teacher" not in jm["pointwise"]
    tab = {t["comparator"]: t for t in m["table"]}
    assert tab["teacher"]["primary"] is None and tab["trained"]["primary"] is not None
    report = (out.snapshot / "report.md").read_text()
    assert "Judged evaluation (one model's assessment, not accuracy)" in report
    assert "reference only" in report and "may not judge its own outputs" in report
    assert (out.snapshot / "judgments.jsonl").exists()
    assert any("one model's assessment" in f for f in m["findings"])


def test_judge_identity_rubric_and_prompt_change_the_run_and_the_protocol(tmp_path, scripted):
    proj = make_open(tmp_path, judge="qwen2.5:1.5b")
    plan1, _, a = run(proj)
    assert "judge" in read_json(a.snapshot / "manifest.json")["protocol"]["fields"]
    (proj / "rubric.md").write_text("Score 5 only for answers that quote the passage.")
    plan2, _, b = run(proj)
    assert plan2.identity != plan1.identity and a.run_id != b.run_id
    views = CM.load_views([proj])
    res = CM.compare(views)
    assert res["incompatible"] and "rubric" in res["incompatible"][0]["reasons"][0]
    assert not res["outcomes"]                                         # not ranked
    # an unchanged rubric re-uses the run
    _, ex, c = run(proj)
    assert c.status == "already-complete"


def test_judge_none_opts_out_and_exact_projects_ignore_a_judge(tmp_path):
    proj = make_open(tmp_path, teacher="qwen2.5:1.5b", judge="none")
    plan = P.make_plan(proj, engine="torch-cpu")
    assert all(s.id != "judge" for s in plan.stages) and plan.cfg["evaluation"]["metric"] == "token_f1"
    p = tmp_path / "c.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "out"])
        for i in range(40):
            w.writerow([f"ticket {i}", ["a", "b", "c"][i % 3]])
    create_project(p, tmp_path / "cls", {"input": "text", "output": "out"}, judge="qwen2.5:1.5b")
    plan = P.make_plan(tmp_path / "cls", engine="torch-cpu")
    assert all(s.id != "judge" for s in plan.stages)
    assert any("recorded but unused" in n for n in plan.notes)
    assert [d["model"] for d in plan.downloads if d["role"] != "embedding"] == ["qwen2.5:0.5b"]
