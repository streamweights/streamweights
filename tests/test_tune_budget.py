"""The saved-activation budget arithmetic for streamed training."""

import pytest

from streamweights.errors import SpillError
from streamweights.tune import budget as b

GIB = 1024**3
LLAMA70 = {"hidden_size": 8192, "intermediate_size": 28672, "num_hidden_layers": 80,
           "vocab_size": 128256}
KW = dict(working_set=36 * GIB, config=LLAMA70, max_layer_bytes=int(1.6 * GIB),
          resident_bytes=int(4.2 * GIB), lora_param_bytes=207_000_000 * 4)


def test_saved_activation_formula_is_the_directives():
    # micro-batch x seq x hidden x 2 bytes x layers
    assert b.saved_activation_bytes(1, 512, 8192, 80) == 512 * 8192 * 2 * 80
    assert b.saved_activation_bytes(10, 512, 8192, 80) == 10 * 512 * 8192 * 2 * 80
    assert b.saved_activation_bytes(1, 512, 8192, 80) == 671_088_640     # 0.625 GiB


def test_micro_batch_is_the_largest_that_fits_the_75_percent_target():
    t = b.compute_micro_batch(seq=512, n_examples=64, **KW)
    assert t.target_bytes == int(36 * GIB * 0.75)
    assert t.total_bytes <= t.target_bytes
    one_more = t.total_bytes + t.saved_per_sample + t.work_per_sample
    assert one_more > t.target_bytes or t.micro_batch == min(64, b.MAX_MICRO_BATCH)
    assert t.saved_bytes == t.micro_batch * b.saved_activation_bytes(1, 512, 8192, 80)
    assert 4 <= t.micro_batch <= 40, t.micro_batch
    assert "saved activations" in t.reason and f"micro-batch {t.micro_batch}" in t.reason


def test_longer_sequences_shrink_the_micro_batch_proportionally():
    short = b.compute_micro_batch(seq=512, n_examples=500, **KW).max_fit
    long = b.compute_micro_batch(seq=2048, n_examples=500, **KW).max_fit
    assert 3 * long <= short + 2 and long >= 1


def test_micro_batch_never_exceeds_the_example_count():
    assert b.compute_micro_batch(seq=128, n_examples=3, **KW).micro_batch == 3


def test_override_over_budget_is_refused_with_the_arithmetic_and_a_fix():
    fit = b.compute_micro_batch(seq=512, n_examples=500, **KW).max_fit
    with pytest.raises(SpillError) as e:
        b.compute_micro_batch(seq=512, n_examples=500, override=fit + 5, **KW)
    assert f"largest micro-batch that fits is {fit}" in str(e.value)
    assert "--grad-accum" in e.value.recovery
    assert b.compute_micro_batch(seq=512, n_examples=500, override=fit, **KW).micro_batch == fit


def test_nothing_fits_is_an_error_not_a_zero():
    with pytest.raises(SpillError, match="not even micro-batch 1"):
        b.compute_micro_batch(seq=200_000, n_examples=10, **KW)


def test_estimate_is_two_streams_per_micro_batch():
    assert b.estimate_step(pass_s=30, compute_s=10, grad_accum=1) == 70
    assert b.estimate_step(pass_s=30, compute_s=10, grad_accum=4) == 280


def test_head_chunk_bounds_logits():
    n = b.head_chunk_positions(128256)
    assert n * 128256 * 12 <= b.HEAD_CHUNK_BYTES and n >= 64
