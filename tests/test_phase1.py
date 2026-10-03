"""Phase 1 tests: safetensors index, ring buffer, budget arithmetic,
recalibrated policy, and the stream-vs-resident identical-output gate."""

import json
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from streamweights.engines.base import MemoryBudget, ModelSpec
from streamweights.engines.mlx_stream import (RingReader, SafetensorsIndex,
                                              compute_batch)
from streamweights.policy import choose_quant_v2, estimate_job_seconds

REPO = Path(__file__).resolve().parent.parent
GIB = 1024**3


# ---------- tiny synthetic model on disk ----------

@pytest.fixture()
def tiny_model(tmp_path):
    """Two-layer fake model: known shapes and byte ranges."""
    rng = np.random.default_rng(7)
    tensors = {}
    order = []
    for k in range(2):
        for suffix, shape in [("self_attn.q_proj.weight", (8, 8)),
                              ("mlp.down_proj.weight", (4, 8)),
                              ("input_layernorm.weight", (8,))]:
            name = f"model.layers.{k}.{suffix}"
            tensors[name] = rng.integers(0, 255, size=shape).astype(np.uint16)
            order.append(name)
    for name, shape in [("model.embed_tokens.weight", (16, 8)),
                        ("model.norm.weight", (8,))]:
        tensors[name] = rng.integers(0, 255, size=shape).astype(np.uint16)
        order.append(name)

    header, blobs, off = {}, [], 0
    for name in order:
        a = tensors[name]
        header[name] = {"dtype": "BF16", "shape": list(a.shape),
                        "data_offsets": [off, off + a.nbytes]}
        blobs.append(a.tobytes())
        off += a.nbytes
    hj = json.dumps(header).encode()
    with open(tmp_path / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen2", "num_hidden_layers": 2, "hidden_size": 8,
        "num_attention_heads": 2, "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
    }))
    return tmp_path, tensors


def test_safetensors_index_shapes_and_ranges(tiny_model):
    d, tensors = tiny_model
    idx = SafetensorsIndex(d)
    assert idx.n_layers == 2 and idx.tied  # no lm_head -> tied
    hlen = 8 + struct.unpack("<Q", open(d / "model.safetensors", "rb").read(8))[0]
    for plan in idx.layers:
        for t in plan.tensors:
            assert t.shape == tensors[t.name].shape
            raw = open(t.shard, "rb").read()[t.offset:t.offset + t.nbytes]
            assert raw == tensors[t.name].tobytes()
            assert t.offset >= hlen
    # layer 0's three tensors are contiguous on disk -> one coalesced segment
    assert len(idx.layers[0].segments) == 1
    assert idx.layers[0].nbytes == sum(tensors[f"model.layers.0.{s}"].nbytes for s in
                                       ["self_attn.q_proj.weight", "mlp.down_proj.weight",
                                        "input_layernorm.weight"])
    assert idx.max_layer_bytes == max(p.nbytes for p in idx.layers)


def test_ring_ordering_and_data(tiny_model):
    d, tensors = tiny_model
    idx = SafetensorsIndex(d)
    ring = RingReader(idx, n_slots=2, chunk_bytes=64, n_threads=2)
    sched = [0, 1, 0, 1]
    ring.start(sched)
    for seq, k in enumerate(sched):
        slot, buf = ring.get(seq)
        plan = idx.layers[k]
        for t in plan.tensors:
            boff = plan.tensor_buf_offsets[t.name]
            assert bytes(buf[boff:boff + t.nbytes]) == tensors[t.name].tobytes()
        ring.release(slot)
    ring.stop()


def test_ring_never_overwrites_slot_in_use(tiny_model):
    d, _ = tiny_model
    idx = SafetensorsIndex(d)
    ring = RingReader(idx, n_slots=2, chunk_bytes=64, n_threads=2)
    ring.start([0, 1, 0])
    s0, b0 = ring.get(0)
    s1, b1 = ring.get(1)
    snapshot = bytes(b0)
    time.sleep(0.3)           # producer wants slot for seq 2; must block
    assert 2 not in ring.ready
    assert bytes(b0) == snapshot  # slot 0 untouched while held
    ring.release(s0)
    s2, _ = ring.get(2)       # now it arrives, reusing slot 0
    assert s2 == s0
    ring.release(s1)
    ring.release(s2)
    ring.stop()


def test_memory_budget_arithmetic():
    idx = SimpleNamespace(
        max_layer_bytes=int(1.6 * GIB),
        embed=SimpleNamespace(nbytes=2 * GIB),
        final_norm=SimpleNamespace(nbytes=16384),
        lm_head=SimpleNamespace(nbytes=2 * GIB),
        tied=False,
        config={"hidden_size": 8192},
    )
    kv_tok = 320 * 1024
    costs = [(500 + 128) * kv_tok] * 500
    bm = compute_batch(idx, MemoryBudget(36 * GIB), costs, 128, kv_tok)
    # arithmetic: ws - 15% - ring(3x) - resident - activations, / per-seq KV
    avail0 = 36 * GIB - int(36 * GIB * 0.15) - 3 * idx.max_layer_bytes - (4 * GIB + 16384)
    assert bm.kv_budget == avail0 - bm.activation_bytes
    # fixed-point on (batch, activations) may land within one row of the ideal
    assert abs(bm.batch - min(max(1, int(bm.kv_budget / costs[0])), 500, 512)) <= 1
    assert "working set" in bm.reason and "per-seq KV" in bm.reason
    # override is honored, never required
    bm2 = compute_batch(idx, MemoryBudget(36 * GIB, batch_override=16), costs, 128, kv_tok)
    assert bm2.batch == 16


def test_recalibrated_policy():
    hw = {"nvme_seq_read": {"bytes_per_sec": 5_481_000_000}}
    cal = {"isolated_read_mbps": 7909.0}
    # 141 GB model, 500 rows, batch 125 -> ~4 chunks * 129 passes * ~17.9 s ≈ 2.6 h -> bf16 holds
    c = choose_quant_v2(141_100_000_000, True, 500, 128, 125, cal, hw)
    assert c.quant == "bf16" and "isolated read ceiling" in c.reason
    assert c.est_seconds < 24 * 3600
    # batch 1 -> 500 chunks * 129 * 17.9 s ≈ 320 h -> drop, and say which rule
    c = choose_quant_v2(141_100_000_000, True, 500, 128, 1, cal, hw)
    assert c.quant == "8bit" and "exceeds 24 h" in c.reason
    # measured engine rate takes precedence over isolated ceiling
    c = choose_quant_v2(141_100_000_000, True, 500, 128, 125,
                        {"engine_read_mbps": 4465.0, **cal}, hw)
    assert "measured engine rate" in c.reason
    # explicit opt-in always wins
    assert choose_quant_v2(1, True, 1, 1, 1, {}, hw, explicit="4bit").quant == "4bit"
    # est arithmetic
    est, pass_s = estimate_job_seconds(100, 10, 4, 5, 50)
    assert pass_s == 2.0 and est == 2 * 5 * 2.0  # 2 chunks * (4+1) passes * 2 s


@pytest.mark.skipif(not (REPO / "models/qwen2.5-0.5b/bf16-st/config.json").exists(),
                    reason="0.5b safetensors not downloaded")
def test_identical_greedy_stream_vs_resident():
    """Item 9 gate: 20 prompts, identical greedy output."""
    from streamweights.engines.mlx_resident import MlxResidentEngine
    from streamweights.engines.mlx_stream import MlxStreamEngine
    rows = [json.loads(l) for l in
            open(REPO / "examples/evals-2000.jsonl")][:20]
    for r in rows:
        r["body"]["max_tokens"] = 64
    spec = ModelSpec("qwen2.5:0.5b", "bf16",
                     REPO / "models/qwen2.5-0.5b/bf16-st", {}, 4096)
    res = {c.custom_id: c.content
           for c in MlxResidentEngine().run_batch(rows, spec, MemoryBudget(36 * GIB))}
    stm = {c.custom_id: c.content
           for c in MlxStreamEngine().run_batch(rows, spec, MemoryBudget(36 * GIB))}
    assert res == stm


@pytest.mark.skipif(not (REPO / "models/qwen2.5-0.5b/bf16-st/config.json").exists(),
                    reason="0.5b safetensors not downloaded")
def test_short_qa_stops_and_matches_mlx_lm_lengths():
    """Phase 1.5 item 2: 20 short-QA prompts must produce a short answer then a
    true EOS stop, identically in both engines, with output length (not just
    first token) matching independent mlx_lm.generate."""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    from streamweights.engines.mlx_resident import MlxResidentEngine
    from streamweights.engines.mlx_stream import MlxStreamEngine

    topics = ["France", "Japan", "water", "the sun", "7*8", "gold", "Mars",
              "a triangle", "oxygen", "Peru", "the alphabet", "a week",
              "an octopus", "copper", "Egypt", "binary", "a violin", "ice",
              "Canada", "pi"]
    rows = [{"custom_id": f"qa{i}",
             "body": {"messages": [{"role": "user",
                                    "content": f"In five words or fewer, say one fact about {t}."}],
                      "max_tokens": 64}} for i, t in enumerate(topics)]
    spec = ModelSpec("qwen2.5:0.5b", "bf16",
                     REPO / "models/qwen2.5-0.5b/bf16-st", {}, 4096)
    res = {c.custom_id: c for c in MlxResidentEngine().run_batch(rows, spec, MemoryBudget(36 * GIB))}
    stm = {c.custom_id: c for c in MlxStreamEngine().run_batch(rows, spec, MemoryBudget(36 * GIB))}
    for k in res:
        assert res[k].content == stm[k].content, k            # same text
        assert res[k].completion_tokens == stm[k].completion_tokens, k  # same stop token
        assert res[k].finish_reason == "stop", (k, res[k].content)      # truly stopped
        assert res[k].completion_tokens < 64, k

    import mlx.core as mx
    if mx.default_device() == mx.cpu:
        # the independent mlx_lm cross-check below compares token lengths of long
        # greedy runs; CPU and Metal reduce bf16 in different orders, which flips
        # near-ties. It is a GPU-only check (stream-vs-resident identity above is not).
        return
    # cross-check vs mlx_lm must compare like-for-like: single sequence
    # (batched kernels reorder bf16 math and can flip near-tied tokens)
    stm1 = {c.custom_id: c for c in MlxStreamEngine().run_batch(
        rows, spec, MemoryBudget(36 * GIB, batch_override=1))}
    model, tok = load(str(REPO / "models/qwen2.5-0.5b/bf16-st"))
    sampler = make_sampler(temp=0.0)
    for r in rows:
        t = tok.apply_chat_template(r["body"]["messages"], add_generation_prompt=True)
        txt = generate(model, tok, prompt=t, max_tokens=64, sampler=sampler,
                       prefill_step_size=1_000_000)
        mine = stm1[r["custom_id"]].content
        # cross-check length against the independent implementation
        assert abs(len(tok.encode(txt, add_special_tokens=False)) -
                   stm1[r["custom_id"]].completion_tokens) <= 1, (r["custom_id"], txt, mine)
