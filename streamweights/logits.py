"""Per-token log-probability output: top-k records, memory check, full-logits guard.

Row shape (one entry per generated, or per scored, token):
    {"token_id": int, "logprob": float, "top": [[id, logprob], ...]}   # top is sorted, best first
Log-probabilities are log_softmax over the full vocabulary of the model's raw
logits (no temperature, no sampling filters).
"""

from __future__ import annotations

import re

import numpy as np

from .errors import SpillError

MAX_K = 64
FULL_LOGITS_MAX_ROWS = 200
# bytes per (row, vocab entry) alive at once while a step's log-probs are built:
# bf16 logits (2) + float32 log-softmax (4) + float32 temp for the partition (4)
STEP_BYTES_PER_ENTRY = 10


def validate_k(k: int | None) -> int | None:
    if k is None:
        return None
    if not 1 <= k <= MAX_K:
        raise SpillError(f"--logprobs K must be between 1 and {MAX_K}; got {k}")
    return k


def step_bytes(batch: int, vocab: int) -> int:
    """Peak extra memory for one decode step with log-probs on."""
    return batch * vocab * STEP_BYTES_PER_ENTRY


def logits_batch_cap(vocab: int, budget_bytes: int, k: int | None) -> int | None:
    """Largest batch whose per-step logits (batch x vocab x 2 bytes, plus the
    log-prob temporaries when K is set) stays inside budget_bytes."""
    per = vocab * (STEP_BYTES_PER_ENTRY if k else 2)
    return max(1, int(budget_bytes // per))


def full_logits_size(n_rows: int, max_tokens: int, vocab: int) -> int:
    """Bytes of float16 logits written for the whole set."""
    return n_rows * max_tokens * vocab * 2


def check_full_logits(n_rows: int, max_tokens: int, vocab: int) -> None:
    if n_rows >= FULL_LOGITS_MAX_ROWS:
        size = full_logits_size(n_rows, max_tokens, vocab)
        raise SpillError(
            f"--full-logits refused for {n_rows} rows: it would write up to "
            f"{size / 1e9:.1f} GB ({n_rows} rows x {max_tokens} tokens x {vocab} vocab x 2 bytes); "
            f"it is for sets under {FULL_LOGITS_MAX_ROWS} rows. Use --logprobs K (K up to "
            f"{MAX_K}) for large sets",
            "spill run <model> <file> --logprobs 32")


def safe_name(custom_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", custom_id)[:120]


# ------------------------------------------------------------ numpy core

def log_softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32)
    m = x.max(axis=-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))


def records_from_arrays(token_ids, token_lps, top_ids, top_lps) -> list[dict]:
    """token_ids [n], token_lps [n], top_ids [n,K], top_lps [n,K] -> record list."""
    out = []
    for i in range(len(token_ids)):
        out.append({
            "token_id": int(token_ids[i]),
            "logprob": round(float(token_lps[i]), 6),
            "top": [[int(a), round(float(b), 6)] for a, b in zip(top_ids[i], top_lps[i])],
        })
    return out


def topk_np(logits: np.ndarray, tokens: np.ndarray, k: int) -> list[dict]:
    """Reference implementation (and fallback) on a [n, V] numpy array."""
    lp = log_softmax_np(logits)
    idx = np.argsort(-lp, axis=-1, kind="stable")[:, :k]
    vals = np.take_along_axis(lp, idx, axis=-1)
    tl = lp[np.arange(len(tokens)), tokens]
    return records_from_arrays(tokens, tl, idx, vals)


# ------------------------------------------------------------ mlx

def topk_mlx(logits, tokens, k: int):
    """logits [n, V] mx array, tokens [n] mx int array -> (token_lp, top_ids, top_lps)
    as numpy. log_softmax in float32; top-k by partition then sort of the K."""
    import mlx.core as mx
    lp = logits.astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    k = min(k, lp.shape[-1])
    idx = mx.argpartition(-lp, kth=k - 1, axis=-1)[..., :k]
    vals = mx.take_along_axis(lp, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    vals = mx.take_along_axis(vals, order, axis=-1)
    tl = mx.take_along_axis(lp, tokens[..., None].astype(mx.int32), axis=-1)[..., 0]
    mx.eval(idx, vals, tl)
    return np.array(tl), np.array(idx), np.array(vals)
