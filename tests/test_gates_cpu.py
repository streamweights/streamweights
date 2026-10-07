# no-mlx-needed: runs on every platform (no MLX, no GPU)
"""The identity and resume gates on a tiny random model, torch-cpu only (the gates against MLX
are in test_gates_mlx.py): what scripts/gates_012.py runs on the 0.5B, at a size CI can run
on every push."""

import json

import pytest

torch = pytest.importorskip("torch")

from streamweights import gates as G  # noqa: E402
from streamweights.tune import toy  # noqa: E402
from tests.tinytorch import make_tiny  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("gates")
    model = make_tiny(root / "model", "qwen2")
    files = toy.make(root / "toy", 80, 20)
    work = G.Work(root / "work")
    return {"model": str(model), "train": files["train"], "held": files["heldout"],
            "work": work, "root": root}


def test_inference_streamed_equals_resident_f32(env):
    rows = [{"custom_id": f"r{i}", "body": {"messages": [{"role": "user", "content":
             f"count to {i} " + "x" * i}], "max_tokens": 10}} for i in range(12)]
    out = G.gate_inference_stream_vs_resident(env["model"], rows)
    assert out["pass"], out
    assert out["greedy_identical"] == 12 and out["max_logprob_diff"] <= 1e-5
    assert out["tokens_compared"] > 0


def test_streamed_tune_equals_peft_resident_tune_f32(env):
    out = G.gate_tune_identity(
        env["work"], env["model"], env["train"],
        a={"engine": "torch-cpu", "path": "streamed"}, b={"engine": "torch-cpu", "path": "resident"},
        steps=50, lr=3e-4, loss_tol=1e-3, cos_tol=0.9999, tag="t7c")
    assert out["pass"], {k: out[k] for k in ("loss", "adapter")}
    assert out["loss"]["steps"] == 50 and out["adapter"]["min"] > 0.9999


def test_resume_a_tune_on_torch_with_the_same_numerics_continues_the_curve(env):
    """A job stopped at step 25 of 50 and continued from --state: the continuation is the
    uninterrupted run (same engine, so the curves agree to float noise)."""
    out = G.gate_resume_tune(env["work"], env["model"], env["train"], first="torch-cpu",
                             then="torch-cpu", steps=50, at=25, lr=3e-4, tag="r8", rank=4,
                             heldout=None, dtype_then="float32")
    assert out["resumed_from"] == 25 and out["final_step"] == 50
    assert [h["engine"] for h in out["history"]] == ["torch_peft_lora"] or len(out["history"]) == 1
    assert out["loss_first_segment_identical_to_reference"] < 1e-5
    assert out["curve"]["mean_rel_resumed_vs_reference"] < 1e-5


def test_eval_rows_stopped_half_way_finish_without_loss_or_duplication(env):
    out = G.gate_resume_rows(env["work"], env["held"], env["model"], first="torch-cpu",
                             then="torch-cpu", half=8, tag="rows8")
    assert out["rows_after_first_leg"] == 8 and out["rows"] == 20 and out["unique"] == 20
    assert out["missing"] == [] and out["duplicated"] == 0
    assert out["exit_codes"] == [0, 0]
    assert [s["rows"] for s in out["segments"]] == [8, 12]
