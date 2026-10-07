# no-mlx-needed: runs on every platform (no MLX, no GPU)
"""The torch engine in bf16 on the 0.5B, checked statistically against float32 (no identity
claim: bf16 differs by rounding). It runs on a CPU that multiplies bf16 quickly and says why it
did not otherwise; the float32 pin (tests/conftest.py) stays for every exactness test.

Bounds, from the measured bf16 statistics of directive 012 (docs/reports/012-run-anywhere.md,
torch bf16 vs torch float32, the 0.5B):

  gradient cosine, one micro-batch of 4 examples, per tensor:   measured mean 0.9736, min 0.7021
      bound: mean >= 0.95, minimum >= 0.5
  log-prob shift, largest absolute difference over compared tokens:   measured 0.190
      bound: <= 0.5 (the other engine's own batch-shape noise there was 0.158 and 0.073)
  greedy agreement, rows identical of 10:   measured 9 of 10, batch-shape noise 10 of 10
      floor: at least 7 of 10, and no more than 2 rows below what the same bf16 engine agrees
      with itself when only the batch shape changes (batch 1 against the whole set)

Set SPILL_TEST_BF16=1 to run it on a CPU without fast bf16 matmul (it takes several minutes).
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from streamweights import formats, gates as G  # noqa: E402
from streamweights.engines.torch_common import cpu_bf16_ok  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "qwen2.5-0.5b" / "bf16-st"
SAMPLE = REPO / "streamweights" / "data" / "sample-20.jsonl"
NATURAL = REPO / "streamweights" / "data" / "gate-natural.jsonl"

GRAD_MEAN_COSINE_MIN = 0.95
GRAD_MIN_COSINE_MIN = 0.5
LOGPROB_SHIFT_MAX = 0.5
AGREEMENT_MIN_ROWS = 7
AGREEMENT_BELOW_NOISE_MAX = 2

_FAST = cpu_bf16_ok() or os.environ.get("SPILL_TEST_BF16") == "1"

pytestmark = [
    pytest.mark.skipif(not (MODEL / "config.json").exists(),
                       reason="needs the qwen2.5:0.5b safetensors (spill run qwen2.5:0.5b sample)"),
    pytest.mark.skipif(not _FAST, reason="this CPU has no fast bf16 matmul (a bf16 linear layer "
                                         "here is under 0.6x the float32 rate), so the bf16 "
                                         "engine path is not what a user on this CPU would run; "
                                         "SPILL_TEST_BF16=1 forces it"),
]


@pytest.fixture(scope="module")
def f32(tmp_path_factory):
    cached = REPO / "models" / "qwen2.5-0.5b" / "f32-st"
    if (cached / "model.safetensors").exists():
        return cached
    return G.make_f32_copy(MODEL, tmp_path_factory.mktemp("f32") / "f32-st")


def test_bf16_inference_agrees_with_float32_above_the_batch_shape_noise_floor(f32):
    rows = formats.load_rows(SAMPLE)
    for r in rows:
        r["body"]["max_tokens"] = 24
    out = G.bf16_inference_statistics(MODEL, f32, rows[:10])
    vs = out["torch_bf16_vs_torch_f32"]
    noise = out["batch_shape_noise_torch_bf16_batch1_vs_all"]
    assert vs["tokens_compared"] > 0
    assert vs["max_logprob_diff"] <= LOGPROB_SHIFT_MAX, vs
    assert vs["rows_identical"] >= AGREEMENT_MIN_ROWS, vs
    assert vs["rows_identical"] >= noise["rows_identical"] - AGREEMENT_BELOW_NOISE_MAX, (vs, noise)


def test_bf16_gradients_point_the_same_way_as_float32(f32):
    g = G.bf16_gradient_statistics(MODEL, f32, NATURAL, do_torch_bf16=True)
    vs = g["torch_bf16_vs_torch_f32"]
    assert vs["mean_cosine"] >= GRAD_MEAN_COSINE_MIN, vs
    assert vs["min_cosine"] >= GRAD_MIN_COSINE_MIN, vs
