"""Shared-prefix reuse: identical greedy output with and without it, on a tiny random
model on CPU, in both block families, with admissions interleaved with decode."""

import pytest

from streamweights.engines.base import MemoryBudget, ModelSpec
from streamweights.engines.mlx_resident import MlxResidentEngine
from streamweights.engines.mlx_stream import (PREFIX_MIN_TOKENS, SharedPrefixCache,
                                              StreamKVCache, common_prefix_len)

from . import tinyqwen

SYS = "Answer with exactly one label from: " + ", ".join(f"label_{i}" for i in range(12)) + "."


def rows(n, system=SYS, max_tokens=7, extra=""):
    out = []
    for i in range(n):
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": f"query number {i}" + "z" * (i * 3 % 17) + extra}]
        out.append({"custom_id": f"r{i:03d}", "body": {"messages": msgs, "max_tokens": max_tokens}})
    return out


def run(model_dir, rs, reuse, batch=None):
    eng = MlxResidentEngine()
    spec = ModelSpec("tiny", "bf16", model_dir, {}, 4096, extra={"prefix_reuse": reuse})
    out = {c.custom_id: c for c in eng.run_batch(rs, spec, MemoryBudget(8 * 2**30,
                                                                       batch_override=batch))}
    return out, eng


@pytest.fixture(scope="module", params=["qwen2", "llama", "llama3"])
def model(request, tmp_path_factory):
    return tinyqwen.make_tiny(tmp_path_factory.mktemp(request.param), request.param)


@pytest.mark.parametrize("batch", [None, 3])
def test_identical_greedy_with_and_without_reuse(model, batch):
    rs = rows(14)
    base, e0 = run(model, rs, False, batch)
    shared, e1 = run(model, rs, True, batch)
    assert e0.prefix_info["reason"] == "disabled"
    assert e1.prefix_info["reason"] == "shared" and e1.prefix_info["prefix_tokens"] >= PREFIX_MIN_TOKENS
    assert {k: v.content for k, v in base.items()} == {k: v.content for k, v in shared.items()}
    assert any(v.completion_tokens > 1 for v in shared.values())
    # prompt token counts still report the full prompt
    assert {k: v.prompt_tokens for k, v in base.items()} == {k: v.prompt_tokens for k, v in shared.items()}


def test_no_reuse_without_a_long_common_prefix(model):
    rs = rows(8, system="short")
    _, eng = run(model, rs, True)
    assert eng.prefix_info["prefix_tokens"] == 0 and "common prefix" in eng.prefix_info["reason"]


def test_few_rows_do_not_engage(model):
    _, eng = run(model, rows(3), True)
    assert "fewer than" in eng.prefix_info["reason"]


def test_prefix_longer_than_a_row_is_clamped(model):
    rs = rows(6)
    rs[2]["body"]["messages"] = rs[2]["body"]["messages"][:1] + [{"role": "user", "content": "x"}]
    base, _ = run(model, rs, False)
    shared, e = run(model, rs, True)
    assert e.prefix_info["reason"] == "shared"
    assert {k: v.content for k, v in base.items()} == {k: v.content for k, v in shared.items()}


def test_common_prefix_len():
    assert common_prefix_len([[1, 2, 3, 4], [1, 2, 9], [1, 2, 3]]) == 2
    assert common_prefix_len([[1, 2]]) == 2
    assert common_prefix_len([]) == 0


def test_shared_cache_returns_prefix_plus_private_without_storing_it():
    import mlx.core as mx
    sk = mx.ones((1, 2, 5, 4))
    c = SharedPrefixCache(offset=5, shared=(sk, sk * 2))
    k, v = c.update_and_fetch(mx.zeros((3, 2, 2, 4)), mx.zeros((3, 2, 2, 4)))
    assert k.shape == (3, 2, 7, 4) and float(k[2, 0, 0, 0]) == 1.0 and float(v[1, 1, 4, 3]) == 2.0
    assert c.keys.shape[2] >= 2 and c._used == 2           # only private tokens are stored


def test_long_and_short_rows_with_compaction_and_refill(model):
    rs = []
    for i in range(16):
        n = 5 if i % 3 else 600 + i
        rs.append({"custom_id": f"r{i:03d}", "body": {"messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": ("w" + str(i)) * (n // 3)}], "max_tokens": 9}})
    base, _ = run(model, rs, False, 4)
    shared, e = run(model, rs, True, 4)
    assert e.prefix_info["reason"] == "shared"
    assert {k: v.content for k, v in base.items()} == {k: v.content for k, v in shared.items()}
    assert {k: v.finish_reason for k, v in base.items()} == {k: v.finish_reason for k, v in shared.items()}
