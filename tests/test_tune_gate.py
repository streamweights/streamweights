"""The identity gate harness on the tiny model (CPU), and the real gate on Metal
(skipped without it)."""

import os
from pathlib import Path

import mlx.core as mx
import pytest

from streamweights.tune import gate
from tests.tinymodel import add_char_tokenizer, make_tiny_model

WS = 36 * 1024**3


def test_gate_harness_passes_on_the_tiny_model(tmp_path):
    d = tmp_path / "m"
    make_tiny_model(d)
    add_char_tokenizer(d)
    res = gate.run_gate(d, tmp_path / "w", working_set=WS, steps=10, n_prompts=6, micro_batch=4,
                        rank=4, alpha=8, lr=2e-3, max_seq=128, n_train=24, note=lambda s: None)
    assert res["pass"], {k: v for k, v in res.items() if not k.startswith(("loss_res", "loss_str"))}
    assert res["loss_rel_diff_step1"] < 1e-4
    assert res["greedy_identical"] == res["greedy_total"] == 6


REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models/qwen2.5-0.5b/bf16-st"
need_metal = pytest.mark.skipif(
    os.environ.get("SPILL_GPU_TESTS") != "1" or not mx.metal.is_available()
    or not (MODEL / "config.json").exists(),
    reason="needs Metal and the Qwen2.5-0.5B safetensors (SPILL_GPU_TESTS=1, or scripts/verify_phase3.py gate)")


@need_metal
def test_training_identity_gate_qwen_0_5b_f32(tmp_path):
    """The item 6 gate on Metal: streamed LoRA vs mlx-lm resident LoRA, 100 steps on
    natural-text targets, Qwen2.5-0.5B converted to float32. In bf16 the two paths differ
    by reorder noise that Adam amplifies (see docs/reports/008-phase3.md); in float32 the
    algorithm identity shows cleanly."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import subprocess
    f32 = tmp_path / "f32"
    subprocess.run([sys.executable, str(REPO / "scripts/make_f32_copy.py"), str(MODEL), str(f32)],
                   check=True)
    mx.set_default_device(mx.gpu)
    res = gate.run_gate(f32, tmp_path / "w", working_set=36 * 1024**3, steps=100,
                        label="qwen2.5:0.5b", natural=REPO / "examples/gate-natural.jsonl")
    assert res["loss_rel_diff_max_after_step1"] <= gate.LOSS_TOL, res
    assert res["cos_min_all"] > gate.COS_MIN, res
    assert res["greedy_identical"] == res["greedy_total"] == 20, res
