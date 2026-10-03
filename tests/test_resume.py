"""Checkpoint and resume apply to --score and --logprobs runs like any other: interrupt
after the first group, resume, and get every row exactly once with one manifest."""

import json
import os
import signal
from pathlib import Path

import pytest

from streamweights import runs
from streamweights.distill import write_distill
from streamweights.engines.base import MemoryBudget, ModelSpec
from streamweights.engines.mlx_resident import MlxResidentEngine
from streamweights.jobs.engine import Job
from streamweights.jobs.runner import run_job

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models/qwen2.5-0.5b/bf16-st"
GIB = 1024**3


@pytest.mark.skipif(not (MODEL / "config.json").exists(), reason="0.5b not available")
def test_interrupt_then_resume_score_run(tmp_path):
    rows = [{"custom_id": f"t{i}", "method": "POST", "url": "/v1/chat/completions",
             "body": {"messages": [{"role": "user", "content": f"Question {i}?"},
                                   {"role": "assistant", "content": f"Answer {i}."}]}}
            for i in range(4)]
    f = tmp_path / "in.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    job = Job.create(f, "qwen2.5:0.5b", "bf16", 4096, 1, rows=rows,
                     options={"mode": "score", "logprobs": 6, "kind": "distill"})
    spec = ModelSpec("qwen2.5:0.5b", "bf16", MODEL, {}, 4096,
                     extra={"mode": "score", "logprobs": 6, "score_group_rows": 1})
    runs.start_run(job, command="spill distill ... --score", spec=spec, input_path=f,
                   hw=None, engine_name="mlx_resident", kind="run")

    eng = MlxResidentEngine()
    calls = []

    def interrupt_after_first(info):
        calls.append(info["rows_done"])
        if len(calls) == 1:
            os.kill(os.getpid(), signal.SIGINT)   # what ^C does; run_job turns it into a clean stop
    eng.pass_cb = interrupt_after_first
    prog = run_job(job, eng, spec, MemoryBudget(36 * GIB))
    assert 1 <= prog.done < 4
    assert job.read_meta()["status"] == "interrupted"
    assert runs.read_manifest(job.id)["status"] == "interrupted"
    first_done = prog.done

    spec2 = ModelSpec("qwen2.5:0.5b", "bf16", MODEL, {}, 4096,
                      extra={"mode": "score", "logprobs": 6, "score_group_rows": 1})
    prog2 = run_job(job, MlxResidentEngine(), spec2, MemoryBudget(36 * GIB))
    assert prog2.done == 4 and prog2.done > first_done
    ids = [json.loads(l)["custom_id"] for l in job.results_path.read_text().splitlines()]
    assert sorted(ids) == ["t0", "t1", "t2", "t3"]            # each row exactly once
    m = runs.read_manifest(job.id)
    assert m["status"] == "completed" and len(m["resumes"]) == 1 and m["rows_done"] == 4
    # per-row provenance is stamped on every row, including the resumed ones
    for l in job.results_path.read_text().splitlines():
        prov = json.loads(l)["streamweights"]["provenance"]
        assert prov["run_id"] == job.id and prov["weight_hash"]
    n = write_distill(job.dir, tmp_path / "d.jsonl", {"model": "m"}, "score")
    assert n == 4
    rec = json.loads((tmp_path / "d.jsonl").read_text().splitlines()[0])
    assert rec["custom_id"] == "t0" and rec["target"] == "Answer 0." and rec["positions"]
