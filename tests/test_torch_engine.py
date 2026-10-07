# no-mlx-needed: runs on every platform (no MLX, no GPU)
"""The PyTorch engines on tiny random models of every verified family, on CPU: streamed ==
resident == transformers' own greedy generate, refill, shared-prefix reuse, log-probs,
teacher-forced scoring, adapters at inference, the memory budget."""

import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from streamweights import logits as lg  # noqa: E402
from streamweights.engines.base import MemoryBudget, ModelSpec  # noqa: E402
from streamweights.engines.torch_common import load_tokenizer  # noqa: E402
from streamweights.engines.torch_resident import TorchResidentEngine  # noqa: E402
from streamweights.engines.torch_stream import TorchEngine  # noqa: E402
from tests.tinytorch import FAMILIES, make_tiny  # noqa: E402

GB = 2**30


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    return {f: make_tiny(root / f, f) for f in FAMILIES}


def rows(n, system=None, max_tokens=12):
    out = []
    for i in range(n):
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": "hello number %d %s" % (i, "x" * (i % 7 * 3))}]
        out.append({"custom_id": f"r{i}", "body": {"messages": msgs, "max_tokens": max_tokens}})
    return out


def run(engine, d, R, batch=None, **extra):
    spec = ModelSpec("tiny", "bf16", d, {}, 512, extra=dict(extra))
    return {c.custom_id: c for c in engine.run_batch(R, spec, MemoryBudget(8 * GB,
                                                                            batch_override=batch))}


def hf_greedy(d, ids, n):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    with torch.no_grad():
        g = m.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False,
                       eos_token_id=2, pad_token_id=0)
    out = g[0, len(ids):].tolist()
    return out[:-1] if out and out[-1] == 2 else out


@pytest.mark.parametrize("family", FAMILIES)
def test_streamed_equals_resident_equals_transformers(models, family):
    d = models[family]
    R = rows(9)
    res = run(TorchResidentEngine(), d, R, batch=4)         # 9 rows, batch 4: refill happens
    stm = run(TorchEngine(), d, R, batch=4)
    assert set(res) == set(stm) == {r["custom_id"] for r in R}
    tok = load_tokenizer(d)
    for r in R:
        cid = r["custom_id"]
        assert res[cid].content == stm[cid].content
        assert res[cid].completion_tokens == stm[cid].completion_tokens
        ids = tok.apply_chat_template(r["body"]["messages"], add_generation_prompt=True)
        assert stm[cid].content == tok.decode(hf_greedy(d, ids, 12)), (family, cid)
    assert any(c.completion_tokens > 0 for c in stm.values())


def test_shared_prefix_reuse_matches_no_reuse(models):
    d = models["llama"]
    R = rows(6, system="You are a careful assistant. " * 6)       # > 64 shared tokens
    eng = TorchEngine()
    a = run(eng, d, R, batch=3)
    assert eng.prefix_info["reason"] == "shared" and eng.prefix_info["prefix_tokens"] >= 64
    b = run(TorchEngine(), d, R, batch=3, prefix_reuse=False)
    for r in R:
        assert a[r["custom_id"]].content == b[r["custom_id"]].content
        assert a[r["custom_id"]].prompt_tokens == b[r["custom_id"]].prompt_tokens


def test_logprobs_match_a_reference_forward(models):
    d = models["qwen2"]
    R = rows(3)
    out = run(TorchEngine(), d, R, logprobs=5)
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    tok = load_tokenizer(d)
    for r in R:
        c = out[r["custom_id"]]
        ids = tok.apply_chat_template(r["body"]["messages"], add_generation_prompt=True)
        gen = [rec["token_id"] for rec in c.logprobs]
        full = torch.tensor([ids + gen])
        with torch.no_grad():
            lp = torch.log_softmax(m(full).logits[0].float(), -1)
        for j, rec in enumerate(c.logprobs):
            row = lp[len(ids) - 1 + j]
            assert abs(rec["logprob"] - row[rec["token_id"]].item()) < 1e-4
            top = row.topk(5)
            assert [t[0] for t in rec["top"]] == top.indices.tolist()
            assert max(abs(t[1] - v) for t, v in zip(rec["top"], top.values.tolist())) < 1e-4


def test_teacher_forced_scoring_matches_reference(models):
    d = models["llama"]
    tok = load_tokenizer(d)
    R = [{"custom_id": f"s{i}", "body": {"messages": [
        {"role": "user", "content": f"say something {i}"},
        {"role": "assistant", "content": "kiwi plum" * (i + 1)}]}} for i in range(3)]
    out = run(TorchEngine(), d, R, mode="score", logprobs=4)
    from transformers import AutoModelForCausalLM

    from streamweights.formats import tokenize_scored
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    for r in R:
        c = out[r["custom_id"]]
        n_prompt, full = tokenize_scored(tok, r["body"]["messages"], {2})
        with torch.no_grad():
            lp = torch.log_softmax(m(torch.tensor([full])).logits[0].float(), -1)
        want = sum(lp[j - 1, full[j]].item() for j in range(n_prompt, len(full)))
        assert abs(c.extra["score"]["logprob_sum"] - want) < 1e-3
        assert c.finish_reason == "scored"


def test_adapter_at_inference_equals_merged_weights(models, tmp_path):
    from transformers import AutoModelForCausalLM

    from streamweights.adapters import load_adapter_dir
    from streamweights.tune import lora_core as lo
    d = models["qwen2"]
    cfg = lo.LoraConfig(rank=4, alpha=8)
    shapes = {"self_attn.q_proj": (64, 64), "mlp.down_proj": (128, 64)}
    params = lo.init_params_np(shapes, 3, 4, seed=1)
    rng = np.random.default_rng(0)
    for k in params:
        if k.endswith("lora_b"):
            params[k] = rng.normal(0, 0.3, params[k].shape).astype(np.float32)
    ad_dir = tmp_path / "ad"
    lo.save_adapter_dir(ad_dir, params, cfg, "tiny", 3, shapes)
    ad = load_adapter_dir(ad_dir, numpy=True)
    R = rows(4)
    plain = run(TorchEngine(), d, R)
    tuned = run(TorchEngine(), d, R, adapter=ad)
    assert any(plain[k].content != tuned[k].content for k in plain)   # the adapter does something
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    with torch.no_grad():
        for k in range(3):
            for path in shapes:
                W = m.model.layers[k].get_submodule(path).weight
                a, b = params[f"layers.{k}.{path}.lora_a"], params[f"layers.{k}.{path}.lora_b"]
                W += torch.tensor((cfg.scale * (a @ b)).T)
    tok = load_tokenizer(d)
    for r in R:
        ids = tok.apply_chat_template(r["body"]["messages"], add_generation_prompt=True)
        with torch.no_grad():
            g = m.generate(torch.tensor([ids]), max_new_tokens=12, do_sample=False,
                           eos_token_id=2, pad_token_id=0)[0, len(ids):].tolist()
        g = g[:-1] if g and g[-1] == 2 else g
        assert tuned[r["custom_id"]].content == tok.decode(g)


def test_memory_budget_caps_the_batch(models):
    d = models["llama"]
    R = rows(12, max_tokens=16)
    eng = TorchResidentEngine()
    spec = ModelSpec("tiny", "bf16", d, {}, 512)
    # a budget that fits the weights plus about two rows of KV
    from streamweights.ring import SafetensorsIndex
    w = SafetensorsIndex(d).total_bytes
    tiny_ws = int((w + 3 * 68 * 768) / 0.75)      # 768 bytes of KV per token here
    out = list(eng.run_batch(R, spec, MemoryBudget(tiny_ws)))
    assert len(out) == 12 and max(c.batch_size for c in out) <= 8
    big = list(TorchResidentEngine().run_batch(R, spec, MemoryBudget(8 * GB)))
    assert max(c.batch_size for c in big) > max(c.batch_size for c in out)
    assert {c.custom_id: c.content for c in out} == {c.custom_id: c.content for c in big}


def test_stop_event_ends_the_run_cleanly(models):
    d = models["llama"]
    ev = threading.Event()
    spec = ModelSpec("tiny", "bf16", d, {}, 512, extra={"stop_event": ev})
    got = []
    for c in TorchEngine().run_batch(rows(10, max_tokens=40), spec, MemoryBudget(8 * GB,
                                                                                batch_override=2)):
        got.append(c)
        ev.set()
    assert 1 <= len(got) < 10


def test_pass_callback_reports_what_the_live_line_needs(models):
    d = models["llama"]
    seen = []
    eng = TorchEngine(pass_cb=seen.append)
    run(eng, d, rows(4))
    assert seen and {"rows_done", "total", "pass_s", "tok_s", "eta_s", "batch", "peak_gb"} <= set(seen[-1])
    assert lg.MAX_K == 64


def test_export_merges_a_torch_trained_adapter_and_the_merged_model_agrees(models, tmp_path):
    """spill export is numpy-only: the merged model it writes behaves like the adapter does at
    inference on the torch engine."""
    from streamweights import export as ex
    from streamweights.adapters import load_adapter_dir
    from streamweights.tune import lora_core as lo
    d = models["llama"]
    cfg = lo.LoraConfig(rank=4, alpha=8)
    shapes = {"self_attn.q_proj": (64, 64), "self_attn.v_proj": (64, 32), "mlp.up_proj": (64, 128)}
    params = lo.init_params_np(shapes, 3, 4, seed=5)
    rng = np.random.default_rng(1)
    for k in params:
        if k.endswith("lora_b"):
            params[k] = rng.normal(0, 0.4, params[k].shape).astype(np.float32)
    ad_dir = tmp_path / "ad"
    lo.save_adapter_dir(ad_dir, params, cfg, "tiny", 3, shapes)
    ad = load_adapter_dir(ad_dir, numpy=True)
    merged = tmp_path / "merged"
    info = ex.merge_adapter(d, ad, merged)
    assert info["modules"] == 9
    R = rows(5)
    via_adapter = run(TorchEngine(), d, R, adapter=ad)
    via_merge = run(TorchEngine(), merged, R)
    plain = run(TorchEngine(), d, R)
    assert any(plain[k].content != via_merge[k].content for k in plain)
    assert {k: v.content for k, v in via_adapter.items()} == {k: v.content for k, v in via_merge.items()}
