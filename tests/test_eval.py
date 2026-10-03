"""spill eval end to end on the 0.5b (CPU): two models, cached reuse, tables, diff, judge."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from streamweights import evalrun
from streamweights.cli import app

REPO = Path(__file__).resolve().parent.parent
need_model = pytest.mark.skipif(not (REPO / "models/qwen2.5-0.5b/bf16-st/config.json").exists(),
                                reason="0.5b safetensors not available")
runner = CliRunner()

QA = [("What is the capital of France?", "Paris"), ("What is 2+2?", "4"),
      ("Name a primary color.", "red"), ("What planet do we live on?", "Earth"),
      ("What is H2O?", "water"), ("Who wrote Hamlet?", "Shakespeare")]


def test_scoring_table_and_diff_are_pure():
    def res(cid, text, lat):
        return {"custom_id": cid, "error": None,
                "response": {"body": {"choices": [{"message": {"content": text}}]}},
                "streamweights": {"latency_s": lat, "tokens": {"prompt": 10, "completion": 5}}}
    a = {c: res(c, t, l) for c, t, l in [("1", "yes", 1.0), ("2", "no", 2.0), ("3", "yes", 3.0)]}
    b = {c: res(c, t, l) for c, t, l in [("1", "yes", 1.0), ("2", "yes", 2.0), ("3", "no", 9.0)]}
    metric = lambda r, e: float(r["response"]["body"]["choices"][0]["message"]["content"] == e)
    exp = {"1": "yes", "2": "yes", "3": "yes"}
    sa, sb = (evalrun.score_results(x, exp, metric) for x in (a, b))
    for sr, name in ((sa, "A"), (sb, "B")):
        sr.label, sr.run_id, sr.model, sr.quant = name, "r" + name, "m", "bf16"
    assert evalrun.mean_score(sa)[0] == pytest.approx(2 / 3)
    assert evalrun.mean_score(sb)[0] == pytest.approx(2 / 3)
    assert sa.tokens == 45 and evalrun.pct(sa.latencies, 50) == 2.0
    rows = [{"custom_id": c, "body": {"messages": [{"role": "user", "content": c}]},
             "expected": "yes"} for c in "123"]
    diff = evalrun.build_diff([sa, sb], rows)
    assert [d["custom_id"] for d in diff] == ["2", "3"]      # 1 agrees, 2 and 3 flip
    md = evalrun.render_table([sa, sb], "m", markdown=True)
    assert md.startswith("| model | quant | adapter | rows | metric mean |")


@need_model
def test_eval_two_models_with_cache_and_judge(tmp_path, mlx_adapter):
    f = tmp_path / "evals.jsonl"
    f.write_text("".join(json.dumps({"messages": [{"role": "user", "content": q}],
                                     "expected": a, "max_tokens": 16}) + "\n" for q, a in QA))
    r = runner.invoke(app, ["eval", str(f), "qwen2.5:0.5b", f"qwen2.5:0.5b+{mlx_adapter}",
                            "--metric", "contains", "--quiet"])
    assert r.exit_code == 0, r.output
    out = r.output
    assert "metric mean" in out and "qwen05-arr" in out and "disagree" in out
    ev = sorted((Path(__import__("os").environ["SPILL_HOME"]) / "runs").glob("eval-*"))[-1]
    assert (ev / "table.md").exists() and (ev / "diff.jsonl").exists()
    man = json.loads((ev / "manifest.json").read_text())
    assert man["kind"] == "eval" and len(man["runs"]) == 2 and not any(man["cached"])
    table = (ev / "table.md").read_text()
    assert table.count("| qwen2.5:0.5b |") == 2 and "qwen05-arr" in table

    # second time: both runs are reused by input hash
    r2 = runner.invoke(app, ["eval", str(f), "qwen2.5:0.5b", f"qwen2.5:0.5b+{mlx_adapter}",
                             "--metric", "contains", "--quiet"])
    assert r2.exit_code == 0 and r2.output.count("reusing run") == 2, r2.output
    # --rerun forces new runs
    r3 = runner.invoke(app, ["eval", str(f), "qwen2.5:0.5b", "--metric", "contains",
                             "--quiet", "--rerun"])
    assert "reusing run" not in r3.output and "one model" in r3.output

    # judge: the 0.5b grades itself; the flow must complete and report a metric
    r4 = runner.invoke(app, ["eval", str(f), "qwen2.5:0.5b", "--judge", "qwen2.5:0.5b",
                             "--quiet"])
    assert r4.exit_code == 0 and "metric: judge (qwen2.5:0.5b)" in r4.output, r4.output
    # an eval file without expected is rejected with a pointer to check
    g = tmp_path / "noexp.jsonl"
    g.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
    r5 = runner.invoke(app, ["eval", str(g), "qwen2.5:0.5b"])
    assert r5.exit_code == 1 and "expected" in r5.output
