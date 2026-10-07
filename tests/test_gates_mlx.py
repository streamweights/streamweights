"""The cross-engine gates on a tiny random model (MLX on the CPU device): torch against MLX,
and jobs that move between MLX and torch-cpu in both directions."""

import pytest

torch = pytest.importorskip("torch")

from streamweights import gates as G  # noqa: E402
from streamweights.tune import toy  # noqa: E402
from tests.tinymodel import add_char_tokenizer, make_tiny_model  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("xgates")
    model = make_tiny_model(root / "model")
    add_char_tokenizer(model)
    files = toy.make(root / "toy", 80, 20)
    return {"model": str(model), "train": files["train"], "held": files["heldout"],
            "work": G.Work(root / "work"), "root": root}


def test_torch_f32_matches_mlx_f32_greedy_and_reports_the_logprob_gap(env):
    rows = [{"custom_id": f"r{i}", "body": {"messages": [{"role": "user", "content":
             f"count to {i} " + "x" * i}], "max_tokens": 10}} for i in range(12)]
    out = G.gate_inference_vs_mlx(env["model"], rows)
    assert out["pass"], out
    assert out["max_logprob_diff"] < 1e-3 and out["tokens_compared"] > 0


def test_torch_tune_matches_mlx_tune_f32(env):
    out = G.gate_tune_identity(
        env["work"], env["model"], env["train"],
        a={"engine": "torch-cpu", "path": "streamed"}, b={"engine": "mlx", "path": "streamed"},
        steps=50, lr=3e-4, loss_tol=0.01, cos_tol=None, tag="x7d")
    assert out["pass"], out["loss"]
    assert out["adapter"]["min"] > 0.999


@pytest.mark.parametrize("first,then", [("mlx", "torch-cpu"), ("torch-cpu", "mlx")])
def test_a_tune_moves_between_mlx_and_torch_and_the_curve_continues(env, first, then):
    out = G.gate_resume_tune(env["work"], env["model"], env["train"], first=first, then=then,
                             steps=50, at=25, lr=3e-4, tag=f"x8-{first}", rank=4,
                             dtype_then="float32")
    assert out["resumed_from"] == 25 and out["final_step"] == 50, out
    assert out["pass"], out["curve"]
    engines = [h["engine"] for h in out["history"]]
    assert len(engines) == 2 and engines[0] != engines[1]          # two producers recorded
    assert out["history"][0]["range"] == [0, 25] and out["history"][1]["range"] == [25, 50]


@pytest.mark.parametrize("first,then", [("mlx", "torch-cpu"), ("torch-cpu", "mlx")])
def test_eval_rows_move_between_mlx_and_torch_without_loss_or_duplication(env, first, then):
    out = G.gate_resume_rows(env["work"], env["held"], env["model"], first=first, then=then,
                             half=10, tag=f"x8rows-{first}")
    assert out["pass"], out
    assert out["by_engine"].keys() and len(out["by_engine"]) == 2
    assert sum(out["by_engine"].values()) == 20
