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


def views(*snaps, uses=None):
    return CM.load_views([s for s in snaps], uses)


def test_common_evaluator_ranks_runs_with_different_training_data(tmp_path):
    a = built(project(tmp_path, "a", train=(0, 80)))
    b = built(project(tmp_path, "b", train=(0, 100)))
    res = CM.compare(views(a.snapshot, b.snapshot))
    assert [o["label"] for o in res["outcomes"]] == ["common evaluator"] and not res["incompatible"]
    assert len(res["outcomes"][0]["ranked"]) == 2
    diffs = next(iter(res["differences"].values()))
    assert "data" in diffs and any("train_sha256" in x for x in diffs["data"])
    text = "\n".join(CM.render(res))
    assert "common evaluator: ranked by" in text and "not a statistical claim" in text
    assert "(the run's original evaluation; no --use given)" in text


def test_cross_runtime_is_ranked_with_the_differences_and_a_caution(tmp_path):
    a = built(project(tmp_path, "a"))
    other = {"engine": "mlx", "engine_impl": "mlx_resident", "device": "apple-gpu", "numerics": {"base": "bf16"},
             "weight_dtype": "bf16", "adapter_dtype": "float32", "compute_dtype": "bf16",
             "versions": {"mlx": "0.32", "torch": "2.0", "transformers": "5.0"}, "decoding_applied": {}}
    plan = P.make_plan(project(tmp_path, "b"), engine="torch-cpu")
    b = CO.Coordinator(tmp_path / "b", FakeExecutor(conditions=other), "torch-cpu", say=lambda s: None).build(plan)
    res = CM.compare(views(a.snapshot, b.snapshot))
    o = res["outcomes"][0]
    assert o["label"] == "cross-runtime" and not res["incompatible"]
    assert any(c.startswith("engine: torch-cpu vs mlx") for c in o["condition_differences"])
    assert any("weight_dtype" in c for c in o["condition_differences"])
    text = "\n".join(CM.render(res))
    assert "cross-runtime: ranked by" in text and "evaluation-runtime effects" in text


def reasons_for(tmp_path, tweak, name):
    a = built(project(tmp_path, f"a-{name}"))
    pb = project(tmp_path, f"b-{name}", system="x" if name == "prompts" else None,
                 val=(210, 240) if name == "rows" else (200, 224))
    cfg = C.load(pb)
    tweak(cfg)
    C.save(pb, cfg)
    b = CO.Coordinator(pb, FakeExecutor(), "torch-cpu", say=lambda s: None).build(P.make_plan(pb, engine="torch-cpu"))
    return CM.compare(views(a.snapshot, b.snapshot))


def test_incompatible_pairs_get_no_ranking_and_say_why(tmp_path):
    cases = {
        "rows": (lambda c: None, "the evaluation rows differ"),
        "prompts": (lambda c: None, "the prompts differ"),
        "decoding": (lambda c: c["evaluation"]["decoding"].update(seed=9), "the decoding policy differs"),
        "metric": (lambda c: c["evaluation"].update(metric="macro_f1"), "the metric definition or version differs"),
    }
    for name, (tweak, want) in cases.items():
        res = reasons_for(tmp_path, tweak, name)
        assert not res["outcomes"] and res["incompatible"], name
        assert any(want in r for r in res["incompatible"][0]["reasons"]), (name, res["incompatible"])
        assert "incompatible: no ranking" in "\n".join(CM.render(res))


def test_a_metric_version_change_is_incompatible(tmp_path):
    a = built(project(tmp_path, "a"))
    b = built(project(tmp_path, "b"))
    va, vb = views(a.snapshot, b.snapshot)
    vb["protocol"] = json.loads(json.dumps(vb["protocol"]))
    vb["protocol"]["fields"]["metric_version"] = "2"
    vb["protocol"]["sha256"] = "x"
    res = CM.compare([va, vb])
    assert not res["outcomes"] and "metric definition or version differs" in res["incompatible"][0]["reasons"][0]


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
    return StageOutcome("done", m, [{"engine": "fake"}], 0.1,
                        notes={"conditions": {"engine": "torch-cpu", "device": "cpu", "numerics": {"base": "float32"},
                                              "weight_dtype": "float32", "versions": {"torch": "2.0"},
                                              "decoding_applied": {"temperature": 0.0}}})


def test_evaluate_makes_a_separate_record_from_validation_rows_only_and_leaves_the_run_identical(tmp_path, monkeypatch):
    from streamweights.project import evalrec
    proj = project(tmp_path, "a")
    out = built(proj)
    before = tree_digest(out.snapshot)
    seen = []
    def spy(desc, ctx):
        seen.append(sha := __import__("streamweights.project.common", fromlist=["x"]).sha_file(ctx.inputs_dir / "val.jsonl"))
        return fake_run_stage(desc, ctx)
    monkeypatch.setattr(testrec, "run_stage", spy)
    rec = evalrec.run_evaluate(proj, say=lambda s: None)
    assert rec["status"] == "completed" and rec["id"].startswith("eval-") and rec["source_run"] == out.run_id
    assert set(seen) == {rec["val_sha256"]} and rec["val_sha256"] != read_json(out.snapshot / "manifest.json")["data"]["test_sha256"]
    assert rec["rows_source"] == "the run's frozen validation inputs"
    assert rec["comparators"]["trained"]["conditions"]["engine"] == "torch-cpu"
    assert set(rec["vs_original"]) == {"baseline", "untrained", "trained"} and rec["original_metric_version"] == "1"
    assert tree_digest(out.snapshot) == before
    d = proj / "evaluations" / rec["id"]
    with pytest.raises(PermissionError):
        (d / "record.json").write_text("{}")
    from streamweights.project.index import test_uses
    assert sum(test_uses(proj).values()) == 0                              # not a use of the final test
    assert rec["id"] in (proj / "REPORT.md").read_text()


def test_compare_uses_an_explicitly_selected_evaluation_and_says_what_it_used(tmp_path, monkeypatch):
    from streamweights.project import evalrec
    proj = project(tmp_path, "a")
    out = built(proj)
    other = built(project(tmp_path, "b", train=(0, 100)))
    monkeypatch.setattr(testrec, "run_stage", fake_run_stage)
    rec = evalrec.run_evaluate(proj, say=lambda s: None)
    default = CM.compare(views(out.snapshot, other.snapshot))
    text = "\n".join(CM.render(default))
    assert f"evaluation used for {out.run_id}: {out.run_id}/original (the run's original evaluation; no --use given; others exist: {rec['id']})" in text
    chosen = CM.compare(CM.load_views([proj, other.snapshot], {out.run_id: rec["id"]}))
    text2 = "\n".join(CM.render(chosen))
    assert f"evaluation used for {out.run_id}: {rec['id']} (chosen with --use)" in text2
    assert any(r["evaluation"] == rec["id"] and r["explicit"] for r in chosen["rows"])
    with pytest.raises(ValueError) as e:
        CM.load_views([proj], {out.run_id: "eval-nope"})
    assert "has no completed evaluation" in str(e.value)


def test_a_parent_is_described_as_lineage_only(tmp_path):
    proj = project(tmp_path, "a")
    first = built(proj)
    plan = P.make_plan(proj, engine="torch-cpu")
    second = CO.Coordinator(proj, FakeExecutor(), "torch-cpu", say=lambda s: None).build(plan, new_run=True)
    assert "experiment lineage only" in (second.snapshot / "report.md").read_text()
    assert "not from the parent's weights" in (second.snapshot / "report.md").read_text()
    text = "\n".join(CM.render(CM.compare(CM.load_views([proj]))))
    assert f"has parent {first.run_id}: experiment lineage only" in text


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
