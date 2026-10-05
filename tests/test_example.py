import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from streamweights import build as B
from streamweights import example as E
from streamweights import formats
from streamweights.cli import app
from streamweights.errors import SpillError

DATA = Path(E.DATA) / "banking77"


def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def test_shipped_files_have_the_promised_shapes():
    ev, tr, pr = rows(DATA / "evals.jsonl"), rows(DATA / "train.jsonl"), rows(DATA / "prompts.jsonl")
    assert (len(ev), len(tr), len(pr)) == (300, 2000, 2000)
    assert all(set(r) == {"prompt", "expected"} for r in ev)
    assert all(set(r) == {"prompt", "answer"} for r in tr)
    assert all(set(r) == {"prompt"} for r in pr)
    for f in ("evals.jsonl", "train.jsonl", "prompts.jsonl"):
        formats.check_file(DATA / f)


def test_no_leakage_between_exam_and_homework():
    ev = {r["prompt"] for r in rows(DATA / "evals.jsonl")}
    tr = {r["prompt"] for r in rows(DATA / "train.jsonl")}
    pr = {r["prompt"] for r in rows(DATA / "prompts.jsonl")}
    assert not ev & tr and not ev & pr and not tr & pr
    q_ev = {r["prompt"] for r in rows(DATA / "quick" / "evals.jsonl")}
    q_tr = {r["prompt"] for r in rows(DATA / "quick" / "train.jsonl")}
    assert q_ev <= ev and q_tr <= tr and not q_ev & q_tr


def test_77_labels_and_the_instructions_list_them_all():
    labels = {r["answer"] for r in rows(DATA / "train.jsonl")}
    assert len(labels) == 77
    ins = (DATA / "instructions.txt").read_text()
    assert all(f"- {l}\n" in ins for l in labels)
    assert {r["expected"] for r in rows(DATA / "evals.jsonl")} <= labels
    assert "Refund_not_showing_up" not in ins and "?" not in "".join(sorted(labels))


def test_quick_variant_sizes_and_coverage():
    assert len(rows(DATA / "quick" / "evals.jsonl")) == 100
    qt = rows(DATA / "quick" / "train.jsonl")
    assert len(qt) == 500 and len({r["answer"] for r in qt}) == 77
    assert not (DATA / "quick" / "prompts.jsonl").exists()


def test_create_full_and_quick(tmp_path):
    f = E.create("banking77", False, tmp_path)
    assert f.name == "banking77" and sorted(p.name for p in f.iterdir() if p.suffix == ".jsonl") == \
        ["evals.jsonl", "prompts.jsonl", "train.jsonl"]
    q = E.create("banking77", True, tmp_path)
    assert q.name == "banking77-quick" and not (q / "prompts.jsonl").exists()
    assert json.loads((q / "spill.json").read_text()) == {"student": "qwen2.5:0.5b"}
    with pytest.raises(SpillError, match="already exists"):
        E.create("banking77", True, tmp_path)
    with pytest.raises(SpillError, match="no example named"):
        E.create("nope", False, tmp_path)


def test_quick_folder_builds_with_the_small_student_and_no_flags(tmp_path):
    q = E.create("banking77", True, tmp_path)
    f = B.read_folder(q)
    assert f.settings["student"] == "qwen2.5:0.5b"
    plan = B.make_plan(f, f.settings["student"], "llama3.3:70b", None, [], 2.0, {}, 36 * 1024**3)
    assert plan.path_kind == "train" and [s.id for s in plan.stages] == ["eval:base", "tune", "eval:tuned"]
    assert plan.classification and plan.max_tokens == 16
    assert "max_tokens 16" in B.pre_run_line(plan)
    # the untrained model's file carries the 77-label prompt; the student's does not
    assert "- card_arrival" in f.instructions


def test_cli_example(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(app, ["example", "banking77", "--quick"])
    assert r.exit_code == 0 and "next: spill build banking77-quick" in r.output
    r = CliRunner().invoke(app, ["example", "banking77", "--quick"])
    assert r.exit_code == 1 and "already exists" in r.output
