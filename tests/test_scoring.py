# no-mlx-needed
"""Scoring selection: exact, json and judge are chosen from the outputs; script: overrides;
a project made before scoring selection keeps its method and its protocol fingerprint."""

import csv
import json

import pytest

from streamweights.errors import SpillError
from streamweights.project import config as C
from streamweights.project import contract as K
from streamweights.project import report as R
from streamweights.project import scoring as SC
from streamweights.project.init import create_project

MAP = {"input": "text", "output": "out"}


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "out"])
        w.writerows(rows)
    return path


def fixtures(tmp_path):
    labels = write(tmp_path / "labels.csv", [(f"ticket {i} about {['billing', 'login', 'ship'][i % 3]}",
                                              ["billing", "login", "ship"][i % 3]) for i in range(60)])
    objs = write(tmp_path / "objs.csv", [(f"order {i}: replace item {i}",
                                          json.dumps({"order": f"A-{i}", "action": "replace"})) for i in range(60)])
    free = write(tmp_path / "free.csv", [(f"passage {i} mentions topic number {i}",
                                          f"The passage says topic number {i} is discussed in detail.") for i in range(60)])
    return labels, objs, free


def test_three_fixtures_pick_exact_json_judge(tmp_path):
    labels, objs, free = fixtures(tmp_path)
    got = {}
    for name, f in (("exact", labels), ("json", objs), ("judge", free)):
        r = create_project(f, tmp_path / f"p-{name}", MAP)
        got[name] = (r.task, r.cfg["task"]["type"], r.cfg["evaluation"]["scoring"])
    assert got == {"exact": ("exact", "classification", "exact"), "json": ("json", "json", "json"),
                   "judge": ("judge", "open", "judge")}


def test_score_flag_overrides_the_automatic_choice(tmp_path):
    labels, _, free = fixtures(tmp_path)
    r = create_project(labels, tmp_path / "a", MAP, score="judge")
    assert r.cfg["task"]["type"] == "open" and r.cfg["evaluation"]["scoring"] == "judge"
    with pytest.raises(SpillError) as e:
        create_project(write(tmp_path / "many.csv", [(f"t{i}", f"label{i}") for i in range(300)]),
                       tmp_path / "c", MAP, score="exact")
    assert "at most 200" in e.value.line()


def test_script_override_scores_each_row(tmp_path):
    labels, _, _ = fixtures(tmp_path)
    s = tmp_path / "my_score.py"
    s.write_text("def score(row):\n    return 1.0 if row['text'] == row['gold'] else 0.25\n")
    r = create_project(labels, tmp_path / "p", MAP, score=f"script:{s}")
    ev = r.cfg["evaluation"]
    assert ev["scoring"] == "script" and ev["metric"] == "script_score"
    assert SC.scoring_of(r.cfg) == "script" and r.cfg["task"]["type"] == "classification"
    assert (tmp_path / "p" / "score.py").exists()
    fn = SC.load_script(tmp_path / "p" / "score.py")
    sc, rec = SC.score_row(r.cfg, None, None, "billing", {"id": "x", "input": "i", "output": "billing"}, fn)
    assert sc["script_score"] == 1.0 and rec["script_score"] == 1.0
    m = SC.aggregate(r.cfg, [sc, SC.score_row(r.cfg, None, None, "login", {"id": "y", "input": "i", "output": "billing"}, fn)[0]],
                     ["billing", "billing"], None)
    assert m["script_score"] == pytest.approx(0.625) and m["accuracy"] == 0.5
    assert C.check_metric("classification", "script_score") == "script_score"
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n")
    with pytest.raises(SpillError) as e:
        create_project(labels, tmp_path / "q", MAP, score=f"script:{bad}")
    assert "no score(row)" in e.value.line()


def test_legacy_config_keeps_its_method_and_protocol(tmp_path):
    labels, objs, _ = fixtures(tmp_path)
    for f, task, want in ((labels, "classification", "exact"), (objs, "json", "json")):
        r = create_project(f, tmp_path / want, MAP, task=task)
        legacy = json.loads(json.dumps(r.cfg))
        del legacy["evaluation"]["scoring"]
        del legacy["model"]["judge"]
        assert SC.scoring_of(legacy) == want
        fp = R.protocol_fingerprint(legacy, None if task == "classification" else
                                    json.loads((tmp_path / want / "schema.json").read_text()), "x", ["a"])
        assert not {"scoring", "judge", "script_sha256"} & set(fp["fields"])
        assert C.metric_version(legacy) == C.METRIC_VERSIONS[task]


def test_reference_overlap_scorer():
    s = K.score_open("The cat sat on the mat", "the cat sat on a mat")
    assert 0.8 < s["token_f1"] < 0.9 and s["rouge_l"] > 0.8 and not s["exact_match"]
    assert K.score_open("Same words", "same   WORDS")["exact_match"]
    z = K.score_open(None, "anything")
    assert z["failed"] and z["token_f1"] == 0.0
    m = K.agg_open([s, z], ["the cat sat on a mat", "anything"])
    assert m["rows"] == 2 and m["inference_failures"] == 1 and m["token_f1"] < s["token_f1"]


def test_a_legacy_project_builds_and_compares_with_a_new_one_on_the_same_data(tmp_path):
    """A project made before scoring selection (no scoring key, no judge key, no kd settings in its
    toml) still plans and builds, and its run is comparable with a new project's run on the same
    rows: the protocol fingerprint carries nothing the old one lacked."""
    import os
    from streamweights.project import compare as CM
    from streamweights.project import coordinator as CO
    from streamweights.project import plan as P
    from .test_project_build import MODEL, FakeExecutor
    if not (MODEL / "model.safetensors").exists():
        pytest.skip("needs the pinned qwen2.5:0.5b")
    labels, _, _ = fixtures(tmp_path)
    for name in ("old", "new"):
        create_project(labels, tmp_path / name, MAP, task="classification")
    cfg = C.load(tmp_path / "old")
    for k in ("scoring",):
        del cfg["evaluation"][k]
    del cfg["model"]["judge"], cfg["model"]["student_why"]
    for k in ("distill_loss", "kd_temp", "kd_alpha", "kd_topk"):
        del cfg["training"][k]
    C.save(tmp_path / "old", cfg)
    runs = {}
    for name in ("old", "new"):
        plan = P.make_plan(tmp_path / name, engine="torch-cpu")
        runs[name] = CO.Coordinator(tmp_path / name, FakeExecutor(), "torch-cpu", say=lambda s: None).build(plan)
        assert runs[name].status == "completed"
    assert SC.scoring_of(C.load(tmp_path / "old")) == "exact"
    views = CM.load_views([tmp_path / "old", tmp_path / "new"])
    res = CM.compare(views)
    assert not res["incompatible"] and res["outcomes"] and res["outcomes"][0]["label"] == "common evaluator"
