"""CLI-level checks on the 0.5b model (CPU): distill generation, distill --score,
check, runs, and that the second --score estimate comes from the measured rate."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from streamweights.cli import app

REPO = Path(__file__).resolve().parent.parent
need_model = pytest.mark.skipif(not (REPO / "models/qwen2.5-0.5b/bf16-st/config.json").exists(),
                                reason="0.5b safetensors not available")
runner = CliRunner()


def _write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_check_reports_first_error_with_line_number(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text(json.dumps({"messages": [{"role": "user", "content": "ok"}]}) + "\n"
                 + json.dumps({"messages": [{"role": "bot", "content": "x"}]}) + "\n")
    r = runner.invoke(app, ["check", str(f)])
    assert r.exit_code == 1 and "line 2" in r.output and "role" in r.output
    good = _write(tmp_path / "ok.jsonl", [{"messages": [{"role": "user", "content": "a"}],
                                          "expected": "b"}])
    r = runner.invoke(app, ["check", str(good)])
    assert r.exit_code == 0 and "eval" in r.output and "next: spill eval" in r.output


@need_model
def test_distill_generation_and_score(tmp_path):
    prompts = _write(tmp_path / "p.jsonl", [
        {"messages": [{"role": "user", "content": "Name a primary color."}], "max_tokens": 8},
        {"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 8}])
    r = runner.invoke(app, ["distill", "qwen2.5:0.5b", str(prompts), "--logprobs", "5",
                            "--quiet", "--out", str(tmp_path / "gen.jsonl")])
    assert r.exit_code == 0, r.output
    assert "spill distill qwen2.5:0.5b:" in r.output and "next:" in r.output
    recs = [json.loads(l) for l in (tmp_path / "gen.jsonl").read_text().splitlines()]
    assert len(recs) == 2
    for rec in recs:
        assert rec["completion"] and rec["teacher"]["weight_hash"]
        assert len(rec["logprobs"]) == len(rec["completion_token_ids"])
        assert all(len(p["top"]) == 5 for p in rec["logprobs"])

    targets = _write(tmp_path / "t.jsonl", [
        {"messages": [{"role": "user", "content": "Capital of France?"},
                      {"role": "assistant", "content": "Paris."}]}])
    r = runner.invoke(app, ["distill", "qwen2.5:0.5b", str(targets), "--score", "--quiet",
                            "--out", str(tmp_path / "sc.jsonl")])
    assert r.exit_code == 0, r.output
    assert "Teacher-forced score" in r.output and "tokens to prefill" in r.output
    rec = json.loads((tmp_path / "sc.jsonl").read_text())
    assert rec["target"] == "Paris." and rec["positions"][0]["top"]
    assert rec["score"]["n_target"] == len(rec["positions"])
    # the measured prefill rate now backs the estimate
    r = runner.invoke(app, ["distill", "qwen2.5:0.5b", str(targets), "--score", "--quiet",
                            "--out", str(tmp_path / "sc2.jsonl")])
    assert "measured prefill rate" in r.output
    r = runner.invoke(app, ["runs"])
    assert "qwen2.5:0.5b" in r.output and "completed" in r.output
    # score mode refuses rows without a target, naming the line
    r = runner.invoke(app, ["distill", "qwen2.5:0.5b", str(prompts), "--score", "--quiet"])
    assert r.exit_code == 1 and "line 1" in r.output
