# no-mlx-needed
"""The teacher is optional and never implied: labels alone build without a teacher stage and
without any download beyond the student and the embedding baseline; a teacher (or unlabeled
prompts, which need one) adds the distill stage, and its answers train the student with the loss
the tokenizers allow."""

import csv
import json
import os

import pytest

from streamweights.errors import SpillError
from streamweights.project import plan as P
from streamweights.project import training
from streamweights.project.common import read_jsonl
from streamweights.project.init import create_project

from .test_project_build import MODEL

pytestmark = pytest.mark.skipif(
    not (MODEL / "model.safetensors").exists() and os.environ.get("SPILL_REQUIRE_FIXTURE_MODEL") != "1",
    reason="needs the pinned qwen2.5:0.5b")


def labeled_csv(path, n=60):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for i in range(n):
            w.writerow([f"ticket {i} about {['billing', 'login', 'ship'][i % 3]}", ["billing", "login", "ship"][i % 3]])
    return path


def test_labels_alone_build_without_a_teacher_or_any_extra_download(tmp_path):
    create_project(labeled_csv(tmp_path / "d.csv"), tmp_path / "p", {"input": "text", "output": "label"})
    plan = P.make_plan(tmp_path / "p", engine="torch-cpu")
    assert [s.id for s in plan.stages] == ["baseline", "train", "eval:untrained", "eval:trained"]
    assert [d["model"] for d in plan.downloads] == ["qwen2.5:0.5b", "embedding:minilm"]
    assert plan.models["teacher"] is None and plan.kd["mode"] == "ce" and plan.kd["logprobs"] is None
    text = "\n".join(P.render(plan))
    assert "teacher   none" in text and "   judge     " not in text
    train = [s for s in plan.stages if s.id == "train"][0]
    assert train.params["teacher"] is False and train.depends == []


def test_a_named_teacher_adds_the_distill_stage_and_keeps_its_top_k(tmp_path):
    create_project(labeled_csv(tmp_path / "d.csv"), tmp_path / "p", {"input": "text", "output": "label"},
                   teacher="qwen2.5:1.5b")
    plan = P.make_plan(tmp_path / "p", engine="torch-cpu")
    assert plan.stages[0].id == "distill" and plan.stages[0].params["logprobs"] == 20
    train = [s for s in plan.stages if s.id == "train"][0]
    assert train.params["distill_loss"] == "mix" and train.params["kd_temp"] == 1.0
    assert plan.kd["why"] == "log-probs kept: same tokenizer"
    assert "the teacher's answers train the student with mix" in "\n".join(P.render(plan))
    from streamweights.project import config as C
    cfg = C.load(tmp_path / "p")
    cfg["training"]["distill_loss"] = "ce"
    C.save(tmp_path / "p", cfg)
    plan = P.make_plan(tmp_path / "p", engine="torch-cpu")
    assert plan.stages[0].params["logprobs"] is None and plan.kd["mode"] == "ce"


def test_the_loss_follows_the_tokenizers():
    sid = {"files": {"tokenizer.json": {"sha256": "a"}}}
    same, other = {"files": {"tokenizer.json": {"sha256": "a"}}}, {"files": {"tokenizer.json": {"sha256": "b"}}}
    cfg = {"training": {"distill_loss": "auto", "kd_topk": 8}, "model": {"teacher": "t", "student": "s"}}
    assert P._kd_plan(cfg, sid, same)["mode"] == "mix" and P._kd_plan(cfg, sid, same)["logprobs"] == 8
    d = P._kd_plan(cfg, sid, other)
    assert d["mode"] == "ce" and d["logprobs"] is None and "tokenizers differ" in d["why"]
    cfg["training"]["distill_loss"] = "kl"
    with pytest.raises(SpillError) as e:
        P._kd_plan(cfg, sid, other)
    assert "share a tokenizer" in e.value.line()
    assert P._kd_plan(cfg, sid, None)["mode"] == "ce"


def test_unlabeled_prompts_need_a_teacher_and_never_include_evaluation_inputs(tmp_path):
    d = labeled_csv(tmp_path / "d.csv")
    u = tmp_path / "u.csv"
    with open(u, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text"])
        for i in range(30):
            w.writerow([f"ticket {i} about {['billing', 'login', 'ship'][i % 3]}" if i < 3 else f"extra prompt {i}"])   # 3 collide with labeled rows
        w.writerow(["extra prompt 10"])                                                  # a duplicate
    res = create_project(d, tmp_path / "p", {"input": "text", "output": "label"}, unlabeled=u)
    assert res.cfg["data"]["unlabeled"]["rows"] == 27 and res.sizes["unlabeled"] == 27
    assert any("matched a training, validation or test input" in w for w in res.warnings)
    unl = read_jsonl(tmp_path / "p" / "data" / "unlabeled.jsonl")
    evalset = {r["input"] for n in ("train", "val", "test") for r in read_jsonl(tmp_path / "p" / "data" / f"{n}.jsonl")}
    assert not ({r["input"] for r in unl} & evalset)
    with pytest.raises(SpillError) as e:
        P.make_plan(tmp_path / "p", engine="torch-cpu")
    assert "unlabeled prompts and no teacher" in e.value.line() and "--teacher" in e.value.line()
    from streamweights.project import config as C
    cfg = C.load(tmp_path / "p")
    cfg["model"]["teacher"] = "qwen2.5:1.5b"
    C.save(tmp_path / "p", cfg)
    plan = P.make_plan(tmp_path / "p", engine="torch-cpu")
    dist = plan.stages[0]
    assert dist.id == "distill" and {i["name"] for i in dist.inputs} == {"train.jsonl", "unlabeled.jsonl"}
    assert "27 unlabeled prompts" in dist.label


def test_training_rows_record_origin_and_carry_the_teachers_top_k(tmp_path):
    res = create_project(labeled_csv(tmp_path / "d.csv"), tmp_path / "p", {"input": "text", "output": "label"})
    cfg = res.cfg
    train = read_jsonl(tmp_path / "p" / "data" / "train.jsonl")[:3]
    unl = [{"id": "u-1", "input": "a new prompt"}]
    t = lambda: {"token_ids": [1], "top_ids": [[1, 2]], "top_lps": [[-0.1, -2.0]], "tokenizer_sha256": "x"}
    answers = {train[0]["id"]: train[0]["output"], train[1]["id"]: "nonsense label", "u-1": "billing"}
    fields = {train[0]["id"]: t(), train[1]["id"]: t(), "u-1": t()}
    rows, man = training.build_training(cfg, train, answers, 2.0, fields, unl)
    by = lambda o: [r for r in rows if r["origin"] == o]
    assert len(by("own")) == 6 and len(by("teacher")) == 1 and len(by("teacher_unlabeled")) == 1
    assert man["teacher_rows_dropped"] == 2 and man["teacher_unlabeled_rows"] == 1
    assert man["teacher_rows_with_logprobs"] == 2 and all("teacher" in r for r in by("teacher") + by("teacher_unlabeled"))
    assert not any("teacher" in r for r in by("own"))
    chat = training.chat_rows(cfg, rows)
    assert sum("teacher" in c for c in chat) == 2
    rows0, man0 = training.build_training(cfg, train, None)
    assert man0["policy"] == "user labels only" and len(rows0) == 3 and man0["teacher_rows"] == 0
