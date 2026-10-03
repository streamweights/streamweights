"""Logits output and teacher-forced scoring, on the 0.5b model, CPU only.
References are computed independently with mlx_lm's own model forward."""

from pathlib import Path

import numpy as np
import pytest

from streamweights import logits as lg
from streamweights.engines.base import MemoryBudget, ModelSpec
from streamweights.errors import SpillError

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models/qwen2.5-0.5b/bf16-st"
GIB = 1024**3
need_model = pytest.mark.skipif(not (MODEL / "config.json").exists(),
                                reason="0.5b safetensors not available")


# ---------------- pure pieces ----------------

def test_validate_k_and_full_logits_guard():
    assert lg.validate_k(None) is None and lg.validate_k(64) == 64
    for bad in (0, 65):
        with pytest.raises(SpillError):
            lg.validate_k(bad)
    lg.check_full_logits(199, 64, 151936)
    with pytest.raises(SpillError) as e:
        lg.check_full_logits(200, 128, 151936)
    assert "GB" in str(e.value) and "200 rows" in str(e.value)
    assert lg.full_logits_size(10, 4, 100) == 10 * 4 * 100 * 2


def test_logits_budget_math():
    # batch x vocab x 2 bytes without log-probs; temporaries counted with them
    assert lg.logits_batch_cap(1000, 2000 * 8, None) == 8
    assert lg.logits_batch_cap(1000, 10000 * 8, 5) == 8
    assert lg.step_bytes(4, 100) == 4 * 100 * lg.STEP_BYTES_PER_ENTRY


def test_topk_np_matches_definition():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 50)).astype(np.float32)
    toks = x.argmax(-1)
    recs = lg.topk_np(x, toks, 7)
    for i, r in enumerate(recs):
        assert r["top"][0][0] == toks[i] == r["token_id"]
        assert abs(r["logprob"] - r["top"][0][1]) < 1e-5
        ps = [np.exp(p) for _, p in r["top"]]
        assert ps == sorted(ps, reverse=True) and sum(ps) <= 1 + 1e-6


def test_topk_mlx_matches_numpy():
    import mlx.core as mx
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 300)).astype(np.float32) * 4
    toks = rng.integers(0, 300, size=6)
    tl, ti, tv = lg.topk_mlx(mx.array(x), mx.array(toks), 9)
    ref = lg.topk_np(x, toks, 9)
    for i, r in enumerate(ref):
        assert [int(a) for a in ti[i]] == [a for a, _ in r["top"]]
        assert np.allclose(tv[i], [b for _, b in r["top"]], atol=1e-4)
        assert abs(tl[i] - r["logprob"]) < 1e-4


# ---------------- generation with logprobs ----------------

def _rows(n=3, max_tokens=8):
    qs = ["Name a primary color.", "What is 2+2?", "Say hi in French."]
    return [{"custom_id": f"g{i}", "body": {
        "messages": [{"role": "user", "content": qs[i]}], "max_tokens": max_tokens}}
        for i in range(n)]


@need_model
def test_generation_logprobs_and_full_logits(tmp_path):
    from streamweights.engines.mlx_resident import MlxResidentEngine
    spec = ModelSpec("qwen2.5:0.5b", "bf16", MODEL, {}, 4096,
                     extra={"logprobs": 8, "full_logits_dir": str(tmp_path / "full")})
    out = list(MlxResidentEngine().run_batch(_rows(), spec, MemoryBudget(36 * GIB)))
    assert len(out) == 3
    for cr in out:
        assert len(cr.logprobs) == cr.completion_tokens
        for rec in cr.logprobs:
            top = rec["top"]
            assert len(top) == 8
            # greedy: the generated token is the top-1 (ties may reorder, equal logprob)
            assert top[0][0] == rec["token_id"] or abs(top[0][1] - rec["logprob"]) < 1e-3
            assert sum(np.exp(p) for _, p in top) <= 1 + 1e-3
        full = np.load(tmp_path / "full" / f"{cr.custom_id}.npy")
        assert full.shape == (cr.completion_tokens, 151936) and full.dtype == np.float16
        lp = lg.log_softmax_np(full)
        assert np.allclose(np.exp(lp).sum(-1), 1.0, atol=1e-3)
        for t, rec in enumerate(cr.logprobs):       # fp16 storage of logits: loose tolerance
            assert abs(lp[t, rec["token_id"]] - rec["logprob"]) < 0.05


@need_model
def test_full_logits_refused_for_large_sets(tmp_path):
    from streamweights.engines.mlx_resident import MlxResidentEngine
    rows = [{"custom_id": f"r{i}", "body": {"messages": [{"role": "user", "content": "hi"}],
                                            "max_tokens": 2}} for i in range(200)]
    spec = ModelSpec("qwen2.5:0.5b", "bf16", MODEL, {}, 4096,
                     extra={"full_logits_dir": str(tmp_path / "f")})
    with pytest.raises(SpillError) as e:
        list(MlxResidentEngine().run_batch(rows, spec, MemoryBudget(36 * GIB)))
    assert "GB" in str(e.value)


# ---------------- teacher-forced scoring vs an independent reference ----------------

TARGETS = [("What is the capital of France?", "The capital of France is Paris."),
           ("Give one word for cold.", "Chilly."),
           ("Count to three.", "One, two, three.")]


def _score_rows():
    return [{"custom_id": f"s{i}", "body": {"messages": [
        {"role": "user", "content": q}, {"role": "assistant", "content": a}]}}
        for i, (q, a) in enumerate(TARGETS)]


@need_model
def test_score_matches_mlx_lm_reference():
    import mlx.core as mx
    from mlx_lm import load
    from streamweights.engines.mlx_resident import MlxResidentEngine
    from streamweights.engines.mlx_stream import MlxStreamEngine

    def spec():
        return ModelSpec("qwen2.5:0.5b", "bf16", MODEL, {}, 4096,
                         extra={"mode": "score", "logprobs": 16})
    got = {c.custom_id: c for c in MlxResidentEngine().run_batch(
        _score_rows(), spec(), MemoryBudget(36 * GIB))}
    # the streamed engine must agree with the resident one exactly
    streamed = {c.custom_id: c for c in MlxStreamEngine().run_batch(
        _score_rows(), spec(), MemoryBudget(36 * GIB))}

    model, tok = load(str(MODEL))
    worst = 0.0
    for i, (q, a) in enumerate(TARGETS):
        cr = got[f"s{i}"]
        assert cr.finish_reason == "scored"
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        n_prompt = len(tok.apply_chat_template(msgs[:1], add_generation_prompt=True))
        ids = list(tok.apply_chat_template(msgs))
        # trailing tokens after the end-of-turn token are not scored
        eot = tok.convert_tokens_to_ids("<|im_end|>")
        ids = ids[:ids.index(eot, n_prompt) + 1]
        ref_logits = model(mx.array(ids)[None])[0].astype(mx.float32)
        ref_lp = ref_logits - mx.logsumexp(ref_logits, axis=-1, keepdims=True)
        assert cr.prompt_tokens == n_prompt and cr.completion_tokens == len(ids) - n_prompt
        for t, rec in enumerate(cr.logprobs):
            pos = n_prompt - 1 + t
            assert rec["token_id"] == ids[n_prompt + t]
            r = float(ref_lp[pos, rec["token_id"]])
            worst = max(worst, abs(r - rec["logprob"]))
            assert int(mx.argmax(ref_lp[pos])) == rec["top"][0][0] or \
                abs(float(ref_lp[pos].max()) - rec["top"][0][1]) < 0.15
        s = streamed[f"s{i}"]
        assert [x["token_id"] for x in s.logprobs] == [x["token_id"] for x in cr.logprobs]
        assert [x["logprob"] for x in s.logprobs] == [x["logprob"] for x in cr.logprobs]
        assert cr.extra["score"]["n_target"] == cr.completion_tokens
    assert worst < 0.15, worst
