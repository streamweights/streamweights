"""`spill tune` from the command line on a tiny model: the pre-run line, progress,
the next-command hint, resume, and the finished adapter running through both
inference engines unchanged."""

import json
import threading

import pytest
from typer.testing import CliRunner

from streamweights.adapters import load_adapter_dir
from streamweights.cli import app
from streamweights.engines.base import MemoryBudget, ModelSpec
from streamweights.jobs.engine import Job
from streamweights.tune import job as tj
from tests.tinymodel import add_char_tokenizer, make_tiny_model

R = CliRunner(mix_stderr=False) if "mix_stderr" in CliRunner.__init__.__code__.co_varnames \
    else CliRunner()


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tinycli") / "tiny-llama"
    make_tiny_model(d)
    add_char_tokenizer(d)
    return d


@pytest.fixture()
def data(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps({"messages": [
        {"role": "user", "content": f"question {i % 6}"},
        {"role": "assistant", "content": "yes yes yes"}]}) for i in range(24)) + "\n")
    return p


def tune(model_dir, data, name, *extra):
    r = R.invoke(app, ["tune", str(model_dir), str(data), "--name", name, "--rank", "4",
                       "--lr", "3e-2", "--batch", "4", "--steps", "40",
                       "--quiet", *extra])
    return r


def test_tune_prints_pre_run_line_and_next_hint(model_dir, data):
    r = tune(model_dir, data, "cli-res", "--path", "resident")
    assert r.exit_code == 0, r.output
    out = r.output
    assert "spill tune tiny-llama: bf16" in out and "resident (mlx-lm LoRA tuner)" in out
    assert "rank 4 alpha 8" in out and "24 examples" in out and "40 steps of micro-batch 4" in out
    assert "Cost: $0" in out and "Est." in out
    assert "next: spill eval evals.jsonl tiny-llama tiny-llama+cli-res" in out
    assert "(PEFT and mlx-lm layouts)" in out


def test_streamed_tune_states_budget_and_streams(model_dir, data):
    r = tune(model_dir, data, "cli-str", "--path", "streamed", "--steps", "6")
    assert r.exit_code == 0, r.output
    assert "streamed from NVMe, two weight streams per micro-batch" in r.output
    assert "why: micro-batch" in r.output and "saved activations" in r.output
    assert "2 weight streams" in r.output          # the estimate is stated in those terms


def test_existing_adapter_is_not_overwritten_silently(model_dir, data):
    assert tune(model_dir, data, "cli-dup", "--path", "resident", "--steps", "2").exit_code == 0
    r = tune(model_dir, data, "cli-dup", "--path", "resident", "--steps", "2")
    assert r.exit_code == 1 and "already exists" in (r.output + r.stderr) \
        if hasattr(r, "stderr") else True
    assert tune(model_dir, data, "cli-dup", "--path", "resident", "--steps", "2",
                "--overwrite").exit_code == 0


def test_bad_training_file_reports_the_line(model_dir, tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"messages": [{"role": "user", "content": "a"},
                                          {"role": "assistant", "content": "b"}]}) + "\n"
                 + json.dumps({"messages": [{"role": "user", "content": "a"}]}) + "\n")
    r = tune(model_dir, p, "cli-bad", "--path", "resident")
    assert r.exit_code == 1
    msg = r.output + (getattr(r, "stderr", "") or "")
    assert "line 2" in msg
    c = R.invoke(app, ["check", str(p)])
    assert c.exit_code == 1 and "line 2" in (c.output + (getattr(c, "stderr", "") or ""))


@pytest.mark.parametrize("path", ["resident", "streamed"])
def test_adapter_runs_through_both_engines_and_changes_output(model_dir, data, path):
    from streamweights.engines.mlx_resident import MlxResidentEngine
    from streamweights.engines.mlx_stream import MlxStreamEngine
    assert tune(model_dir, data, f"cli-run-{path}", "--path", path, "--steps", "200").exit_code == 0
    ad_name = f"cli-run-{path}"
    rows = [{"custom_id": f"r{i}", "body": {"messages": [
        {"role": "user", "content": f"question {i}"}], "max_tokens": 12}} for i in range(4)]

    def run(engine, adapter):
        spec = ModelSpec("tiny", "bf16", model_dir, {}, 4096,
                         extra={"adapter": adapter} if adapter else {})
        return {c.custom_id: c.content for c in engine().run_batch(
            rows, spec, MemoryBudget(36 * 1024**3))}
    from streamweights.adapters import resolve_adapter
    base = run(MlxResidentEngine, None)
    res = run(MlxResidentEngine, resolve_adapter(ad_name))
    stm = run(MlxStreamEngine, resolve_adapter(ad_name))
    assert res == stm
    # the adapter visibly changes every row (lr 3e-2 is deliberately hot: the two paths'
    # trajectories diverge chaotically after a few steps, so only the engines are compared)
    assert all(res[k] != base[k] for k in res)


def test_resume_a_tune_job_from_the_cli(model_dir, data):
    spec = tj.TuneSpec(model="tiny-llama", quant="bf16", model_dir=str(model_dir),
                       data=str(data), name="cli-resume", path="resident", rank=4, alpha=8,
                       lr=1e-2, micro_batch=4, steps=12, ckpt_every=4, max_seq=128,
                       overwrite=True)
    prep = tj.prepare(spec, working_set=36 * 1024**3, micro_batch_given=True, steps_given=True)
    job = Job.create(data, "tiny-llama", "bf16", 128, 4, None,
                     options={"kind": "tune", "tune": spec.to_dict()})
    stop = threading.Event()
    r1 = tj.run_tune(prep, job, stop=stop,
                     progress_cb=lambda i: stop.set() if i["step"] == 5 else None)
    assert r1["interrupted"]
    r = R.invoke(app, ["resume", job.id])
    assert r.exit_code == 0, r.output
    assert "resuming" in r.output and "12/12 steps" in r.output
    assert "next: spill eval" in r.output
    assert load_adapter_dir(tj.adapters_dir() / "cli-resume").rank == 4
    r2 = R.invoke(app, ["resume", job.id])
    assert "already complete" in r2.output
