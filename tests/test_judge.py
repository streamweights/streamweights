# no-mlx-needed
"""Judged evaluation: parsing, both-order pairwise verdicts, self-judge exclusion, calibration,
and the judge's place in the evaluation-protocol fingerprint. The judge model is scripted; the
stage code, the prompts and the arithmetic are the real ones."""

import csv
import json
from pathlib import Path

import pytest

from streamweights.project import compare as CM
from streamweights.project import judge as J
from streamweights.project import report as R
from streamweights.project import stagefns as SF
from streamweights.project.common import read_json, read_jsonl, write_json, write_jsonl
from streamweights.project.init import create_project


def test_parse_pointwise_and_pairwise():
    assert J.parse_pointwise("Rationale: ok.\nScore: 4")[0] == 4
    assert J.parse_pointwise("Rationale: bad\nscore - 2")[0] == 2
    assert J.parse_pointwise("Score: 5\nactually Score: 3")[0] == 3
    assert J.parse_pointwise("no number here")[0] is None
    assert J.parse_pointwise("Score: 7")[0] is None
    assert J.parse_pointwise(None) == (None, "")
    assert J.parse_pairwise("Rationale: x\nBetter: A")[0] == "A"
    assert J.parse_pairwise("Better: **tie**")[0] == "tie"
    assert J.parse_pairwise("Better: Answer B")[0] == "B"
    assert J.parse_pairwise("unclear")[0] is None


def test_both_orders_cancel_position_bias():
    # a judge that always says A: x wins one order and loses the other -> a tie, a disagreement
    first = J.to_models("A", "x", "y")            # x shown as A -> x
    second = J.to_models("A", "y", "x")           # y shown as A -> y
    assert J.combine_orders(first, second) == ("tie", True)
    # a judge that follows content: x preferred in both orders
    assert J.combine_orders(J.to_models("A", "x", "y"), J.to_models("B", "y", "x")) == ("x", False)
    # a tie against a win is a disagreement, and a tie
    assert J.combine_orders("x", "tie") == ("tie", True)
    assert J.combine_orders("tie", "tie") == ("tie", False)
    assert J.combine_orders(None, "x") == (None, None)
    s = J.pairwise_summary([{"verdict": "x", "disagreed": False, "picks": ["A", "B"]},
                            {"verdict": "tie", "disagreed": True, "picks": ["A", "A"]},
                            {"verdict": None, "disagreed": None, "picks": [None, "A"]}])
    assert (s["wins"], s["ties"], s["losses"], s["scored"], s["unscored"]) == (1, 1, 0, 2, 1)
    assert s["order_disagreement_rate"] == 0.5 and s["first_position_pick_rate"] == 0.8


def test_self_judging_is_excluded_from_judging():
    p = J.plan_judging(["baseline", "untrained", "trained", "teacher"], "qwen2.5:0.5b",
                       "qwen2.5:1.5b", "qwen2.5:1.5b")
    assert list(p["excluded"]) == ["teacher"] and "may not judge its own outputs" in p["excluded"]["teacher"]
    assert "teacher" not in p["judged"] and ["trained", "teacher"] not in p["pairs"]
    assert p["pairs"] == [["trained", "untrained"], ["trained", "baseline"]]
    # a judge equal to the student excludes both student comparators
    q = J.plan_judging(["baseline", "untrained", "trained"], "qwen2.5:0.5b", None, "Qwen/Qwen2.5-0.5B-Instruct")
    assert set(q["excluded"]) == {"untrained", "trained"} and q["pairs"] == []
    assert J.same_model("qwen2.5:1.5b", "qwen/qwen2.5-1.5b-instruct")
    assert not J.same_model("qwen2.5:1.5b", "qwen2.5:7b")


def test_calibration_applies_only_to_checkable_rows():
    items = [{"reference": "Paris", "output": "The capital is Paris.", "score": 5},
             {"reference": "Paris", "output": "It is Lyon.", "score": 2},
             {"reference": "Paris", "output": "Berlin.", "score": 4},          # judge-only positive
             {"reference": "a long reference answer that cannot be matched word for word by check",
              "output": "x", "score": 3},
             {"reference": "Rome", "output": "Rome", "score": None}]
    c = J.calibration(items)
    assert c["rows"] == 3 and c["agreement"] == pytest.approx(2 / 3)
    assert (c["both_positive"], c["both_negative"], c["judge_only"], c["match_only"]) == (1, 1, 1, 0)
    assert J.calibration([{"reference": "a very long reference " * 5, "output": "x", "score": 3}]) is None


def _csv(path, n=24):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "out"])
        for i in range(n):
            w.writerow([f"What is item {i}?", f"Item {i}" if i % 2 else f"The item number {i} is a long form answer here."])
    return path


class ScriptedBackend:
    """A judge that scores by output length and always prefers position A when asked to compare
    answers of equal length, so both orders disagree on those rows."""
    def __init__(self):
        self.requests = []

    def eval(self, model, f):
        rows = read_jsonl(f)
        self.requests = rows
        out = []
        for r in rows:
            msg = r["body"]["messages"][0]["content"]
            if r["custom_id"].startswith("pt:"):
                score = 5 if "ANSWER TO GRADE:\nItem" in msg else 2
                reply = f"Rationale: scripted.\nScore: {score}"
            else:
                reply = "Rationale: scripted.\nBetter: A"
            out.append({"custom_id": r["custom_id"], "response": {"body": {"choices": [
                {"message": {"content": reply}, "finish_reason": "stop"}]}}, "streamweights": {}})
        res = f.with_name("results.jsonl")
        write_jsonl(res, out)
        return {"results_path": str(res), "run_id": "r", "cached": False}


def test_judge_stage_end_to_end_with_scripted_judge(tmp_path, monkeypatch):
    proj = tmp_path / "p"
    create_project(_csv(tmp_path / "d.csv"), proj, {"input": "text", "output": "out"},
                   judge="qwen2.5:1.5b", teacher="qwen2.5:1.5b")
    att = tmp_path / "att"
    (att / "inputs").mkdir(parents=True)
    for n in ("streamweights.toml",):
        (att / "inputs" / n).write_bytes((proj / n).read_bytes())
    val = read_jsonl(proj / "data" / "val.jsonl")
    write_jsonl(att / "inputs" / "val.jsonl", val)
    pf = {}
    for c in ("baseline", "untrained", "trained"):
        d = tmp_path / c
        d.mkdir()
        write_jsonl(d / "predictions.jsonl", [{"id": r["id"], "text": r["output"] if c == "trained" else "Zzz"}
                                              for r in val])
        pf[c] = str(d / "predictions.jsonl")
    be = ScriptedBackend()
    monkeypatch.setattr(SF, "_backend", lambda: be)
    monkeypatch.setattr("streamweights.cli_build._row_producers", lambda p: [])
    monkeypatch.setattr(SF, "choose_engine_name", lambda: "torch-cpu")
    plan_j = J.plan_judging(["baseline", "untrained", "trained", "teacher"], "qwen2.5:0.5b",
                            "qwen2.5:1.5b", "qwen2.5:1.5b")
    desc = SF.StageDesc("judge", "judge", "j", {"judge": "qwen2.5:1.5b", **plan_j,
                                                "pred_files": pf})
    sd = tmp_path / "stage"
    ctx = SF.StageContext(att, sd, "torch-cpu")
    out = SF.run_stage(desc, ctx)
    m = read_json(ctx.out_dir / "metrics.json")
    assert out.status == "done" and "teacher" in m["excluded"] and "teacher" not in m["pointwise"]
    assert set(m["pointwise"]) == {"baseline", "untrained", "trained"}
    assert m["pointwise"]["untrained"]["judge_mean"] == 2.0           # "Zzz" never starts with Item
    assert m["pointwise"]["trained"]["judge_mean"] > 2.0
    pairs = {tuple(p["pair"]): p for p in m["pairwise"]}
    assert set(pairs) == {("trained", "untrained"), ("trained", "baseline")}
    # the always-A judge disagrees with itself across orders on every pair row
    p = pairs[("trained", "untrained")]
    assert p["order_disagreement_rate"] == 1.0 and p["ties"] == p["scored"] and p["wins"] == 0
    # both orders were requested
    ids = [r["custom_id"] for r in be.requests]
    assert sum(i.startswith("pw:trained:untrained:xa") for i in ids) == len(val)
    assert sum(i.startswith("pw:trained:untrained:ya") for i in ids) == len(val)
    # per-example records carry verdicts and rationales
    recs = read_jsonl(ctx.out_dir / "judgments.jsonl")
    assert {r["kind"] for r in recs} == {"pointwise", "pairwise"}
    assert all(r["rationale"] == "scripted." for r in recs if r["kind"] == "pointwise")
    assert m["calibration"]["rows"] > 0                               # the "Item n" references are short
    # folded into the comparators' metrics, the teacher marked reference only
    comps = {c: {"metrics": {"rows": 1}} for c in ("baseline", "trained", "teacher")}
    J.apply_to_metrics(comps, m)
    assert comps["trained"]["metrics"]["judge_mean"] == m["pointwise"]["trained"]["judge_mean"]
    assert comps["teacher"]["metrics"]["judge_mean"] is None and "judge_excluded" in comps["teacher"]["metrics"]
    lines = R.judge_findings(m)
    assert any("one model's assessment" in l for l in lines) and any("may not judge its own" in l for l in lines)


def test_judge_protocol_changes_make_runs_incompatible():
    ident = {"tag": "qwen2.5:1.5b", "repo": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "r1", "files": {"m": "h"}}
    base = J.protocol(ident, J.DEFAULT_RUBRIC, "default rubric")

    def view(jp, run):
        fields = {"rows": {"n": 1}, "prompts": {}, "decoding": {}, "max_tokens": 1, "postprocessing": "",
                  "metric": "judge_mean", "metric_version": "1", "contract": {}, "judge": jp}
        return {"run_id": run, "eval_id": run, "protocol": {"fields": fields, "sha256": run}}
    same = CM.protocol_differences(view(base, "a"), view(base, "b"))
    assert same == []
    new_rubric = J.protocol(ident, J.DEFAULT_RUBRIC + "\nBe strict.", "rubric.md")
    d = CM.protocol_differences(view(base, "a"), view(new_rubric, "b"))
    assert d and "rubric" in d[0] and d[0].startswith("the judge protocol differs")
    other = J.protocol({**ident, "revision": "r2"}, J.DEFAULT_RUBRIC, "default rubric")
    assert "judge model identity" in CM.protocol_differences(view(base, "a"), view(other, "b"))[0]
    tmpl = json.loads(json.dumps(base))
    tmpl["templates"]["pointwise"] += " Be brief."
    assert "judge prompt templates" in CM.protocol_differences(view(base, "a"), view(tmpl, "b"))[0]
    dec = json.loads(json.dumps(base))
    dec["decoding"]["max_tokens"] = 99
    assert "judge decoding" in CM.protocol_differences(view(base, "a"), view(dec, "b"))[0]
    assert "judged and the other was not" in CM.protocol_differences(view(base, "a"), view(None, "b"))[0]
