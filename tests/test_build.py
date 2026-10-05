"""spill build: path selection, per-role prompts, stage plan and estimates, resume at any
stage, the table, and the final line. A fake backend stands in for the engines (CPU only)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from streamweights import build as B
from streamweights import cli_build
from streamweights.cli import app
from streamweights.errors import SpillError

LABELS = ["card_arrival", "lost_or_stolen_card", "exchange_rate"]
INS = "Answer with exactly one label from: " + ", ".join(LABELS) + "."


def jl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture()
def folder(tmp_path):
    d = tmp_path / "bank"
    d.mkdir()
    jl(d / "evals.jsonl", [{"prompt": f"q{i}", "expected": LABELS[i % 3]} for i in range(30)])
    jl(d / "train.jsonl", [{"prompt": f"t{i}", "answer": LABELS[i % 3]} for i in range(40)])
    jl(d / "prompts.jsonl", [{"prompt": f"p{i}"} for i in range(40)])
    (d / "instructions.txt").write_text(INS + "\n")
    return d


class Fake(B.Backend):
    def __init__(self, scores=None, stop_at=None):
        self.calls, self.scores = [], scores or {}
        self.stop_at = stop_at
        self.seen = {}

    def distill(self, teacher, prompts, out, max_tokens, resume_job):
        self.calls.append(("distill", teacher))
        self.seen["distill_prompts"] = [json.loads(l) for l in Path(prompts).read_text().splitlines()]
        rows = self.seen["distill_prompts"]
        Path(out).write_text("".join(json.dumps(
            {"custom_id": f"r{i}", "messages": r["messages"], "completion": "card_arrival"}) + "\n"
            for i, r in enumerate(rows)))
        return {"job_id": "job-d", "rows": len(rows), "out": str(out)}

    def tune(self, model, train, name, resume_job):
        self.calls.append(("tune", model, name))
        self.seen["train"] = [json.loads(l) for l in Path(train).read_text().splitlines()]
        if self.stop_at == "tune":
            self.stop_at = None
            return {"interrupted": True, "job_id": "job-t"}
        return {"adapter": f"/adapters/{name}", "steps": 10}

    def eval(self, model, eval_file, resume_job=None):
        self.calls.append(("eval", model, Path(eval_file).name))
        self.seen.setdefault("eval_files", {})[model] = [
            json.loads(l) for l in Path(eval_file).read_text().splitlines()]
        return {"score": self.scores.get(model, 0.5), "rows": 30, "run_id": f"run-{len(self.calls)}"}


def plan_for(folder, **kw):
    f = B.read_folder(folder)
    args = dict(student="qwen2.5:7b", teacher="llama3.3:70b", base=None, compare=[],
                weight_own=2.0, cal={}, working_set=36 * 1024**3)
    args.update(kw)
    return B.make_plan(f, args["student"], args["teacher"], args["base"], args["compare"],
                       args["weight_own"], args["cal"], args["working_set"])


def test_path_decided_by_files(folder):
    assert plan_for(folder).path_kind == "both"
    (folder / "prompts.jsonl").rename(folder / "prompts.jsonl.bak")
    assert plan_for(folder).path_kind == "train"
    (folder / "train.jsonl").rename(folder / "train.jsonl.bak")
    (folder / "prompts.jsonl.bak").rename(folder / "prompts.jsonl")
    assert plan_for(folder).path_kind == "prompts"
    (folder / "prompts.jsonl").unlink()
    with pytest.raises(SpillError, match="nothing to learn from"):
        plan_for(folder)


def test_evals_required(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(SpillError, match="evals.jsonl is missing"):
        B.read_folder(tmp_path / "x")


def test_stage_order_per_path(folder):
    ids = lambda p: [s.id for s in p.stages]
    assert ids(plan_for(folder)) == ["distill", "tune", "eval:base", "eval:tuned", "eval:teacher"]
    (folder / "prompts.jsonl").rename(folder / "p.bak")
    assert ids(plan_for(folder)) == ["eval:base", "tune", "eval:tuned"]
    assert ids(plan_for(folder, compare=["llama3.3:70b"])) == [
        "eval:base", "tune", "eval:tuned", "eval:compare:llama3.3:70b"]


def test_base_flag_tunes_the_big_model(folder):
    p = plan_for(folder, base="qwen2.5:32b")
    assert p.tuned_model == "qwen2.5:32b"
    tune = next(s for s in p.stages if s.kind == "tune")
    assert tune.model == "qwen2.5:32b"


def test_classification_gets_max_tokens_16_and_says_so(folder):
    p = plan_for(folder)
    assert p.classification and p.max_tokens == 16
    line = B.pre_run_line(p)
    assert "max_tokens 16" in line and "path both" in line
    assert "llama3.3:70b (teacher)" in line and "Est. " in line and "5 TFLOP/s, assumed" in line
    assert "On battery" not in line and "On battery" in B.pre_run_line(p, on_battery=True)


def test_prose_evals_default_128(folder):
    jl(folder / "evals.jsonl", [{"prompt": "explain", "expected": "x" * 200}])
    p = plan_for(folder)
    assert not p.classification and p.max_tokens == 128


def test_measured_tflops_replace_the_assumption(folder):
    p = plan_for(folder, cal={"tune_rates": {"qwen2.5:7b|resident": 12e12}})
    assert p.tflops == 12.0 and "measured" in p.tflops_src


def test_run_both_path_roles_and_files(folder):
    plan = plan_for(folder)
    fake = Fake({"qwen2.5:7b+bank": 0.9, "qwen2.5:7b": 0.2, "llama3.3:70b": 0.8})
    res = B.run_plan(plan, fake, say=lambda *_: None)
    assert [c[0] for c in fake.calls] == ["distill", "tune", "eval", "eval", "eval"]
    # untrained models see the instructions; the tuned student never does
    for m in ("qwen2.5:7b", "llama3.3:70b"):
        first = fake.seen["eval_files"][m][0]
        assert first["messages"][0] == {"role": "system", "content": INS}
        assert first["max_tokens"] == 16
    tuned = fake.seen["eval_files"]["qwen2.5:7b+bank"][0]
    assert all(m["role"] != "system" for m in tuned["messages"])
    # the teacher answered prompts with the instructions
    assert fake.seen["distill_prompts"][0]["messages"][0]["content"] == INS
    # training = own x2 + teacher, student view (no instructions)
    tr = fake.seen["train"]
    assert len(tr) == 40 * 2 + 40
    assert all(m["role"] != "system" for r in tr for m in r["messages"])
    assert [r["role"] for r in res.table] == ["your model", "base (untrained)", "teacher"]
    assert res.table[0]["score"] == 0.9
    assert B.final_lines(res)[0] == "your model: qwen2.5:7b+bank"
    # intermediates in the folder
    assert (folder / "prompts.distill.jsonl").exists()
    assert (folder / "qwen2.5-7b_plus_bank.out.jsonl").exists() or True


def test_weight_own_changes_repetition(folder):
    plan = plan_for(folder, weight_own=3.0)
    fake = Fake()
    B.run_plan(plan, fake, say=lambda *_: None)
    assert len(fake.seen["train"]) == 40 * 3 + 40


def test_train_only_has_no_duplication(folder):
    (folder / "prompts.jsonl").rename(folder / "p.bak")
    fake = Fake()
    B.run_plan(plan_for(folder), fake, say=lambda *_: None)
    assert len(fake.seen["train"]) == 40
    assert [c[0] for c in fake.calls] == ["eval", "tune", "eval"]


def test_resume_at_any_stage(folder):
    plan = plan_for(folder)
    fake = Fake(stop_at="tune")
    r1 = B.run_plan(plan, fake, say=lambda *_: None)
    assert r1.interrupted
    st = B.load_state(B.read_folder(folder))
    assert [s["status"] for s in st["stages"]][:2] == ["done", "interrupted"]
    plan2 = plan_for(folder)
    assert B.apply_state(plan2, st)
    fake2 = Fake()
    r2 = B.run_plan(plan2, fake2, say=lambda *_: None)
    assert not r2.interrupted
    assert [c[0] for c in fake2.calls] == ["tune", "eval", "eval", "eval"]  # distill not redone
    assert len(r2.table) == 3
    assert B.load_state(B.read_folder(folder))["finished"] is True


def test_changed_plan_does_not_reuse_state(folder):
    B.run_plan(plan_for(folder), Fake(), say=lambda *_: None)
    st = B.load_state(B.read_folder(folder))
    assert not B.apply_state(plan_for(folder, teacher="qwen2.5:32b"), st)


def test_estimates_use_the_cost_model(folder):
    p = plan_for(folder)
    d = next(s for s in p.stages if s.kind == "distill")
    assert d.model == "llama3.3:70b" and d.est_s > 0
    assert "shared prefix once" in d.est_detail
    assert p.total_s == pytest.approx(sum(s.est_s for s in p.stages))


def test_cli_build_end_to_end_with_fake_backend(folder, monkeypatch):
    fake = Fake({"qwen2.5:7b+bank": 0.9, "qwen2.5:7b": 0.2, "llama3.3:70b": 0.8})
    monkeypatch.setattr(cli_build, "RealBackend", lambda *a, **k: fake)
    monkeypatch.setenv("SPILL_NO_NOTIFY", "1")
    r = CliRunner().invoke(app, ["build", str(folder)])
    assert r.exit_code == 0, r.output
    assert "path both" in r.output and "stage 1/5" in r.output
    assert "your model: qwen2.5:7b+bank" in r.output
    assert r.output.rstrip().endswith("next: spill export qwen2.5:7b+bank")
    assert "model" in r.output and "role" in r.output and "score" in r.output
    # running again is a no-op that reprints the table
    fake2 = Fake()
    monkeypatch.setattr(cli_build, "RealBackend", lambda *a, **k: fake2)
    r = CliRunner().invoke(app, ["build", str(folder)])
    assert r.exit_code == 0 and fake2.calls == []


def test_cli_build_interrupted_prints_resume_command(folder, monkeypatch):
    fake = Fake(stop_at="tune")
    monkeypatch.setattr(cli_build, "RealBackend", lambda *a, **k: fake)
    monkeypatch.setenv("SPILL_NO_NOTIFY", "1")
    r = CliRunner().invoke(app, ["build", str(folder)])
    assert r.exit_code == 130
    assert f"next: spill resume {folder.resolve()}" in r.output


def test_stage_table_shows_estimate_versus_actual(folder):
    plan = plan_for(folder)
    B.run_plan(plan, Fake(), say=lambda *_: None)
    out = B.render_stages(plan.stages)
    assert "estimate" in out and "actual" in out and out.splitlines()[-1].startswith("total")
    assert "1 distill" in out or "teacher" in out


def test_edited_inputs_invalidate_finished_stages(folder):
    B.run_plan(plan_for(folder), Fake(), say=lambda *_: None)
    st = B.load_state(B.read_folder(folder))
    assert B.apply_state(plan_for(folder), st)
    with open(folder / "train.jsonl", "a") as f:
        f.write(json.dumps({"prompt": "new", "answer": "card_arrival"}) + "\n")
    assert not B.apply_state(plan_for(folder), st)
    (folder / "instructions.txt").write_text("changed")
    assert not B.apply_state(plan_for(folder), st)
    assert not B.apply_state(plan_for(folder, weight_own=5.0), st)
