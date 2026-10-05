"""Streamed LoRA backward == ordinary autograd, on a tiny random model, on CPU."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from streamweights.tune import lora as lo
from streamweights.tune.streamed import StreamedTrainer, masked_ce
from tests.tinymodel import make_tiny_model


def reference_model(model_dir, cfg: lo.LoraConfig, params):
    """mlx-lm's own model with mlx-lm's own LoRA layers, loaded with `params`."""
    import glob
    import json

    from mlx_lm.models import llama
    from mlx_lm.tuner.utils import linear_to_lora_layers
    config = json.loads((model_dir / "config.json").read_text())
    model = llama.Model(llama.ModelArgs.from_dict(config))
    w = {}
    for f in sorted(glob.glob(str(model_dir / "*.safetensors"))):
        w.update(mx.load(f))
    model.load_weights(list(w.items()))
    model.freeze()
    keys = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    linear_to_lora_layers(model, config["num_hidden_layers"],
                          {"rank": cfg.rank, "scale": cfg.scale, "dropout": 0.0, "keys": keys})
    model.load_weights([(f"model.{k}", v) for k, v in params.items()], strict=False)
    return model


def make_batch(B=3, T=32, V=300, seed=0):
    rng = np.random.RandomState(seed)
    inp = rng.randint(3, V, (B, T)).astype(np.int32)
    tgt = rng.randint(3, V, (B, T)).astype(np.int32)
    msk = rng.rand(B, T) > 0.4
    msk[:, 0] = True
    msk[1, T // 2:] = False        # a "padded" row
    return inp, tgt, msk


def randomized(params, seed=1):
    """LoRA B starts at zero (no gradient into A); perturb so every gradient is live."""
    mx.random.seed(seed)
    out = {}
    for k, v in params.items():
        out[k] = v if k.endswith("lora_a") else mx.random.normal(v.shape) * 0.05
    return out


@pytest.mark.parametrize("resident", [True, False], ids=["resident-weights", "ring"])
def test_streamed_grads_match_autograd_f32(tmp_path, resident):
    d = make_tiny_model(tmp_path / "m")
    cfg = lo.LoraConfig(rank=4, alpha=8, seed=3)
    tr = StreamedTrainer(d, cfg, resident_weights=resident)
    tr.params = randomized(tr.params)
    inp, tgt, msk = make_batch()
    loss, grads, ntoks = tr.micro_batch(inp, tgt, msk, 0)
    tr.close()

    ref = reference_model(d, cfg, tr.params)

    def loss_fn(m):
        return masked_ce(m(mx.array(inp)), mx.array(tgt), mx.array(msk))[0]

    rloss, rgrads = nn.value_and_grad(ref, loss_fn)(ref)
    rflat = {k[len("model."):]: v for k, v in tree_flatten(rgrads)}
    assert abs(loss - rloss.item()) < 1e-5 * max(1, abs(loss))
    assert ntoks == int(msk.sum())
    assert set(grads) == set(rflat)
    for k in grads:
        a, b = np.array(grads[k]), np.array(rflat[k])
        assert np.allclose(a, b, rtol=2e-3, atol=2e-5), (k, np.abs(a - b).max(), np.abs(b).max())
        assert np.abs(b).max() > 0 or k.endswith("lora_a")


def test_streamed_grads_close_in_bf16(tmp_path):
    d = make_tiny_model(tmp_path / "m", dtype=mx.bfloat16)
    cfg = lo.LoraConfig(rank=4, alpha=8, seed=3)
    tr = StreamedTrainer(d, cfg, resident_weights=True)
    tr.params = randomized(tr.params)
    inp, tgt, msk = make_batch()
    loss, grads, _ = tr.micro_batch(inp, tgt, msk, 0)
    ref = reference_model(d, cfg, tr.params)
    rloss, rgrads = nn.value_and_grad(
        ref, lambda m: masked_ce(m(mx.array(inp)), mx.array(tgt), mx.array(msk))[0])(ref)
    assert abs(loss - rloss.item()) / abs(rloss.item()) < 5e-3
    rflat = {k[len("model."):]: v for k, v in tree_flatten(rgrads)}
    for k in grads:
        if k.endswith("lora_b"):
            assert lo.cosine_sim(grads[k], rflat[k]) > 0.99, k


def test_dropout_recompute_uses_the_forward_mask(tmp_path):
    """With dropout the backward recompute must redraw the forward's mask. Check the
    analytic gradient of one LoRA entry against a finite difference of the loss."""
    d = make_tiny_model(tmp_path / "m")
    cfg = lo.LoraConfig(rank=4, alpha=8, dropout=0.4, seed=3)
    tr = StreamedTrainer(d, cfg, resident_weights=True)
    tr.params = randomized(tr.params)
    inp, tgt, msk = make_batch(B=2, T=16)
    loss0, grads, _ = tr.micro_batch(inp, tgt, msk, 7)
    assert tr.micro_batch(inp, tgt, msk, 7)[0] == loss0            # same micro index, same masks
    assert tr.micro_batch(inp, tgt, msk, 8)[0] != loss0            # a different draw otherwise
    name = "layers.1.mlp.up_proj.lora_a"
    g = np.array(grads[name])
    i, j = np.unravel_index(np.abs(g).argmax(), g.shape)
    eps = 1e-3
    vals, orig = [], tr.params
    for sgn in (+1, -1):
        p = dict(orig)
        a = np.array(p[name])
        a[i, j] += sgn * eps
        p[name] = mx.array(a)
        tr.params = p
        vals.append(tr.micro_batch(inp, tgt, msk, 7)[0])
    fd = (vals[0] - vals[1]) / (2 * eps)
    assert abs(fd - g[i, j]) < 0.03 * abs(g[i, j]) + 1e-4, (fd, g[i, j])
