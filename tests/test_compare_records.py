# no-mlx-needed
"""spill compare (rank only comparable runs) and spill test (a separate, immutable record)."""

import csv
import json
from pathlib import Path

import pytest

from streamweights.errors import SpillError
from streamweights.project import compare as CM
from streamweights.project import config as C
from streamweights.project import coordinator as CO
from streamweights.project import plan as P
from streamweights.project import testrec
from streamweights.project.common import read_json, read_jsonl
from streamweights.project.init import create_project
from streamweights.project.runstate import tree_digest
from streamweights.project.stagefns import StageOutcome

from .test_project_build import CLASSES, FakeExecutor, pytestmark  # noqa: F401


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    return path


def rows(lo, hi):
    return [(f"ticket {i} about {CLASSES[i % 4]} problem number {i}", CLASSES[i % 4]) for i in range(lo, hi)]


def project(tmp_path, name, train=(0, 80), val=(200, 224), test=(300, 324), system=None, labels=None):
    tr = write_csv(tmp_path / f"{name}-tr.csv", rows(*train))
    va = write_csv(tmp_path / "val.csv", rows(*val))
    te = write_csv(tmp_path / "test.csv", rows(*test))
    sysf = None
    if system:
        sysf = tmp_path / f"{name}-sys.txt"
        sysf.write_text(system)
    create_project(tr, tmp_path / name, {"input": "text", "output": "label"}, "classification",
                   system_file=sysf, val=va, test=te, labels=labels)
    return tmp_path / name


def built(proj):
    plan = P.make_plan(proj, engine="torch-cpu")
    return CO.Coordinator(proj, FakeExecutor(), "torch-cpu", say=lambda s: None).build(plan)


def test_runs_with_different_training_data_but_the_same_rows_and_protocol_are_ranked(tmp_path):
    a = built(project(tmp_path, "a", train=(0, 80)))
    b = built(project(tmp_path, "b", train=(0, 100)))
    runs = CM.load_runs([a.snapshot, b.snapshot])
    res = CM.compare(runs)
    assert res["ranked"] and set(res["ranked"][0]) == {a.run_id, b.run_id} and not res["refusals"]
    diffs = next(iter(res["differences"].values()))
    assert "data" in diffs and any("train_sha256" in x for x in diffs["data"])
    text = "\n".join(CM.render(res))
    assert "ranking (same evaluation rows, metric and protocol)" in text and "not a statistical claim" in text


def test_different_evaluation_rows_get_an_explanation_and_no_winner(tmp_path):
    a = built(project(tmp_path, "a"))
    b = built(project(tmp_path, "b", val=(210, 240)))
    res = CM.compare(CM.load_runs([a.snapshot, b.snapshot]))
    assert not res["ranked"] and res["refusals"]
    assert "the evaluation rows differ" in res["refusals"][0]["reasons"][0]
    assert "no ranking" in "\n".join(CM.render(res))


def test_different_prompts_or_vocabulary_are_a_different_protocol(tmp_path):
    a = built(project(tmp_path, "a"))
    b = built(project(tmp_path, "b", system="Answer with the department only."))
    res = CM.compare(CM.load_runs([a.snapshot, b.snapshot]))
    assert not res["ranked"]
    assert "the evaluation protocols differ in" in res["refusals"][0]["reasons"][0]
    assert "prompts" in res["refusals"][0]["reasons"][0]


def test_a_changed_metric_definition_is_refused(tmp_path):
    a = built(project(tmp_path, "a"))
    b = built(project(tmp_path, "b"))
    ma, mb = (read_json(x.snapshot / "manifest.json") for x in (a, b))
    mb["metric_definition"]["metric_version"] = "2"
    mb["metric_version"] = "2"
    res = CM.compare([{**ma, "_dir": "a"}, {**mb, "_dir": "b"}])
    assert not res["ranked"] and "metric definitions differ" in res["refusals"][0]["reasons"][0]


def fake_run_stage(desc, ctx):
    """What spill test would get from the engines: every comparator right on every other row."""
    from streamweights.project import contract as K
    cfg = C.parse((ctx.inputs_dir / "streamweights.toml").read_bytes())
    val = read_jsonl(ctx.inputs_dir / "val.jsonl")          # these are the TEST rows here
    labels = cfg["contract"]["labels"]
    items, preds = [], []
    for i, r in enumerate(val):
        text = r["output"] if i % 2 == 0 else next(l for l in labels if l != r["output"])
        sc = K.score_class(text, r["output"], labels)
        items.append(sc)
        preds.append({"id": r["id"], "gold": r["output"], "text": text, "correct": sc["correct"]})
    m = K.agg_class(items, [r["output"] for r in val], labels)
    m["truncated"] = 0
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    (ctx.out_dir / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    return StageOutcome("done", m, [{"engine": "fake"}], 0.1)


def test_test_creates_a_separate_immutable_record_and_counts_each_use(tmp_path, monkeypatch):
    proj = project(tmp_path, "a")
    out = built(proj)
    run_before = tree_digest(out.snapshot)
    monkeypatch.setattr(testrec, "run_stage", fake_run_stage)
    r1 = testrec.run_test(proj, say=lambda s: None)
    assert r1["status"] == "completed" and r1["source_run"] == out.run_id and r1["uses_before"] == 0
    assert r1["test_sha256"] == read_json(out.snapshot / "manifest.json")["data"]["test_sha256"]
    assert "not consulted before" in r1["holdout_note"]
    r2 = testrec.run_test(proj, out.run_id, say=lambda s: None)
    assert r2["uses_before"] == 1 and "not an untouched holdout" in r2["holdout_note"]
    d1 = proj / "tests" / r1["id"]
    snap1 = tree_digest(d1)
    with pytest.raises(PermissionError):
        (d1 / "record.json").write_text("{}")
    assert tree_digest(out.snapshot) == run_before                       # the run did not change
    idx = (proj / "REPORT.md").read_text()
    assert "scored 2 time(s)" in idx and "not an untouched holdout" in idx
    assert tree_digest(d1) == snap1
    # the run's own report never mentions a test score
    rep = (out.snapshot / "report.md").read_text()
    assert "final test" in rep and "scored only by `spill test`" in rep


def test_test_refuses_when_the_test_file_changed_and_keeps_failures_visible(tmp_path, monkeypatch):
    proj = project(tmp_path, "a")
    out = built(proj)

    def boom(desc, ctx):
        raise RuntimeError("engine fell over")
    monkeypatch.setattr(testrec, "run_stage", boom)
    with pytest.raises(SpillError) as e:
        testrec.run_test(proj, say=lambda s: None)
    assert "failed record was kept" in e.value.message
    failed = [r for r in json.loads(json.dumps([read_json(p / "record.json")
              for p in sorted((proj / "tests").iterdir())])) if r["status"] == "failed"]
    assert len(failed) == 1 and "engine fell over" in failed[0]["error"]
    # a failed record is not a use of the holdout
    from streamweights.project.index import test_uses
    assert sum(test_uses(proj).values()) == 0
    (proj / "data" / "test.jsonl").write_text((proj / "data" / "test.jsonl").read_text() + "\n")
    with pytest.raises(SpillError) as e:
        testrec.run_test(proj, say=lambda s: None)
    assert "changed since run" in e.value.message
