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


def test_the_torch_optimizer_is_the_mlx_optimizer():
    """AdamW without bias correction, decoupled weight decay first, cosine schedule read at the
    step count before counting: the checkpoint's optimizer state means the same on both."""
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np

    from streamweights.tune.torch_train import AdamW
    rng = np.random.default_rng(0)
    p0 = {"a": rng.normal(size=(5, 3)).astype(np.float32), "b": rng.normal(size=(4,)).astype(np.float32)}
    mp = {k: mx.array(v) for k, v in p0.items()}
    tp = {k: torch.tensor(v.copy()) for k, v in p0.items()}
    steps = 12
    mopt = optim.AdamW(learning_rate=optim.cosine_decay(1e-2, steps), weight_decay=0.01)
    topt = AdamW(tp, 1e-2, steps, "cosine", 0.01)
    for _ in range(steps + 3):                        # past the end of the schedule too
        g = {k: rng.normal(size=v.shape).astype(np.float32) for k, v in p0.items()}
        mp = mopt.apply_gradients({k: mx.array(v) for k, v in g.items()}, mp)
        topt.apply(tp, {k: torch.tensor(v) for k, v in g.items()})
    for k in p0:
        np.testing.assert_allclose(tp[k].numpy(), np.array(mp[k]), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(topt.m[k].numpy(), np.array(mopt.state[k]["m"]), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(topt.v[k].numpy(), np.array(mopt.state[k]["v"]), rtol=1e-6, atol=1e-9)
    assert topt.step == int(mopt.state["step"].item())


def test_the_shared_init_is_the_same_numbers_on_both_engines():
    import numpy as np

    from streamweights.tune import lora as mlora
    from streamweights.tune import lora_core as core
    shapes = {"self_attn.q_proj": (64, 64), "mlp.down_proj": (128, 64)}
    a = core.init_params_np(shapes, 3, 4, seed=7)
    b = mlora.init_params(shapes, 3, 4, 7)
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], np.array(b[k]))
