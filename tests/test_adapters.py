"""LoRA at inference: loading both layouts, and that applying the adapter at the
layer boundary matches mlx-lm's own LoRA, identically in the streamed and resident
engines. The adapter is trained (CPU, 60 iterations) by scripts/make_test_adapter.py."""

import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from streamweights.adapters import (load_adapter_dir, resolve_adapter,
                                    split_model_adapter)
from streamweights.engines.base import MemoryBudget, ModelSpec
from streamweights.errors import SpillError

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models/qwen2.5-0.5b/bf16-st"
ADAPTER = REPO / "adapters/qwen05-arr"
GIB = 1024**3
need_model = pytest.mark.skipif(not (MODEL / "config.json").exists(),
                                reason="0.5b safetensors not available")


@pytest.fixture()
def peft_adapter(mlx_adapter, tmp_path):
    """The same adapter written in PEFT layout (A [r,in], B [out,r], alpha/r scale)."""
    raw = mx.load(str(mlx_adapter / "adapters.safetensors"))
    cfg = json.loads((mlx_adapter / "adapter_config.json").read_text())
    out = {}
    for k, v in raw.items():
        base, _, which = k.rpartition(".lora_")
        out[f"base_model.model.{base}.lora_{which.upper()}.weight"] = v.T
    mx.save_safetensors(str(tmp_path / "adapter_model.safetensors"), out)
    r = cfg["lora_parameters"]["rank"]
    (tmp_path / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": r, "lora_alpha": cfg["lora_parameters"]["scale"] * r,
        "target_modules": ["q_proj", "v_proj"], "base_model_name_or_path": "Qwen/Qwen2.5-0.5B"}))
    return tmp_path


def test_split_model_adapter():
    assert split_model_adapter("llama3.3:70b+./a") == ("llama3.3:70b", "./a")
    assert split_model_adapter("qwen2.5:0.5b+org/repo") == ("qwen2.5:0.5b", "org/repo")
    assert split_model_adapter("qwen2.5:0.5b") == ("qwen2.5:0.5b", None)
    assert split_model_adapter("a+") == ("a+", None)


def test_loads_both_layouts_to_the_same_tensors(mlx_adapter, peft_adapter):
    a, b = load_adapter_dir(mlx_adapter), load_adapter_dir(peft_adapter)
    assert (a.layout, b.layout) == ("mlx-lm", "peft")
    assert a.rank == b.rank == 8 and a.targets == b.targets
    assert sorted(a.layers) == sorted(b.layers) == [20, 21, 22, 23]
    for k in a.layers:
        for path in a.layers[k]:
            (a1, b1, s1), (a2, b2, s2) = a.layers[k][path], b.layers[k][path]
            assert np.allclose(np.array(a1), np.array(a2)) and np.allclose(np.array(b1), np.array(b2))
            assert abs(s1 - s2) < 1e-6
    assert a.hash != b.hash   # different files, different hash


def test_rejects_non_lora_and_wrong_base(tmp_path, mlx_adapter):
    (tmp_path / "adapters.safetensors").write_bytes((mlx_adapter / "adapters.safetensors").read_bytes())
    (tmp_path / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "dora", "lora_parameters": {"rank": 8}}))
    with pytest.raises(SpillError, match="not supported"):
        load_adapter_dir(tmp_path)
    with pytest.raises(SpillError):
        resolve_adapter(str(tmp_path / "nope"))
    ad = load_adapter_dir(mlx_adapter)
    with pytest.raises(SpillError, match="different base"):
        ad.validate({"num_hidden_layers": 20, "hidden_size": 896})
    with pytest.raises(SpillError, match="different base"):
        ad.validate({"num_hidden_layers": 24, "hidden_size": 1024})


def _rows(n=20, max_tokens=24):
    rows = [json.loads(l) for l in open(REPO / "streamweights/data/sample-20.jsonl")][:n]
    for r in rows:
        r["body"]["max_tokens"] = max_tokens
    return rows


def _run(engine_cls, adapter, rows=None, **extra):
    spec = ModelSpec("qwen2.5:0.5b", "bf16", MODEL, {}, 4096,
                     extra=({"adapter": adapter} if adapter else {}) | extra)
    return {c.custom_id: c for c in engine_cls().run_batch(
        rows or _rows(), spec, MemoryBudget(36 * GIB))}


@need_model
def test_adapter_changes_output_and_is_identical_streamed_vs_resident(mlx_adapter):
    from streamweights.engines.mlx_resident import MlxResidentEngine
    from streamweights.engines.mlx_stream import MlxStreamEngine
    ad = load_adapter_dir(mlx_adapter)
    base = _run(MlxResidentEngine, None)
    res = _run(MlxResidentEngine, ad)
    stm = _run(MlxStreamEngine, load_adapter_dir(mlx_adapter))
    assert {k: v.content for k, v in res.items()} == {k: v.content for k, v in stm.items()}
    assert {k: v.completion_tokens for k, v in res.items()} == \
        {k: v.completion_tokens for k, v in stm.items()}
    # the adapter was trained to end answers with "Arr!" and must visibly do it
    assert sum("Arr" in v.content for v in res.values()) >= 10
    assert sum("Arr" in v.content for v in base.values()) == 0
    # the base weights were never modified: a later adapter-free run matches the first
    base2 = _run(MlxResidentEngine, None)
    assert {k: v.content for k, v in base2.items()} == {k: v.content for k, v in base.items()}


@need_model
def test_peft_layout_gives_the_same_output_as_mlx_lm_layout(mlx_adapter, peft_adapter):
    from streamweights.engines.mlx_resident import MlxResidentEngine
    a = _run(MlxResidentEngine, load_adapter_dir(mlx_adapter), _rows(8))
    b = _run(MlxResidentEngine, load_adapter_dir(peft_adapter), _rows(8))
    assert {k: v.content for k, v in a.items()} == {k: v.content for k, v in b.items()}


@need_model
def test_adapter_logits_match_mlx_lm_own_lora(mlx_adapter):
    """Independent reference: mlx-lm loads the same adapter with its LoRALinear and we
    compare the teacher-forced log-probs of one target under the adapter."""
    from mlx_lm import load
    from streamweights.engines.mlx_resident import MlxResidentEngine
    q, a = "What is the capital of France?", "Paris. Arr!"
    rows = [{"custom_id": "s0", "body": {"messages": [
        {"role": "user", "content": q}, {"role": "assistant", "content": a}]}}]
    got = _run(MlxResidentEngine, load_adapter_dir(mlx_adapter), rows,
               mode="score", logprobs=4)["s0"]
    model, tok = load(str(MODEL), adapter_path=str(mlx_adapter))
    msgs = rows[0]["body"]["messages"]
    n_prompt = len(tok.apply_chat_template(msgs[:1], add_generation_prompt=True))
    ids = list(tok.apply_chat_template(msgs))
    ids = ids[:ids.index(tok.convert_tokens_to_ids("<|im_end|>"), n_prompt) + 1]
    lg = model(mx.array(ids)[None])[0].astype(mx.float32)
    ref = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    worst = max(abs(float(ref[n_prompt - 1 + t, rec["token_id"]]) - rec["logprob"])
                for t, rec in enumerate(got.logprobs))
    assert worst < 0.15, worst
    # and the adapter really is applied: scoring without it gives a different answer
    base = _run(MlxResidentEngine, None, rows, mode="score", logprobs=4)["s0"]
    assert abs(base.extra["score"]["logprob_sum"] - got.extra["score"]["logprob_sum"]) > 1.0
