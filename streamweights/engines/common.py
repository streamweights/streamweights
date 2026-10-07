"""Framework-neutral helpers shared by every engine: the end-of-sequence ids a model
declares, shared-prefix detection, and the scheduling constants."""

from __future__ import annotations

import json
from pathlib import Path

# prefill is processed alongside decode, at most this many prompt tokens per
# pass, so a wave of newcomers never turns one pass into a 28-minute wall
PREFILL_TOKENS_PER_PASS = 2048

# shared-prefix reuse engages when every row of the job starts with the same
# token prefix at least this long (and there are enough rows to share it)
PREFIX_MIN_TOKENS = 64
PREFIX_MIN_ROWS = 4

def collect_eos_ids(model_dir: Path, tokenizer) -> set[int]:
    """Every EOS id the model declares: generation_config.json and config.json
    (int or list), the tokenizer's own ids, and the chat template's end-of-turn
    token (for Llama 3.x that includes 128009)."""
    ids: set[int] = set()
    for fname in ("generation_config.json", "config.json"):
        p = Path(model_dir) / fname
        if p.exists():
            v = json.loads(p.read_text()).get("eos_token_id")
            if isinstance(v, int):
                ids.add(v)
            elif isinstance(v, list):
                ids.update(int(x) for x in v)
    if getattr(tokenizer, "eos_token_ids", None):
        ids.update(tokenizer.eos_token_ids)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(tokenizer.eos_token_id)
    # chat template end-of-turn token, resolved through the vocabulary
    for tok in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                ids.add(tid)
        except Exception:
            pass
    return ids



def common_prefix_len(seqs: list[list[int]]) -> int:
    """Longest token prefix shared by every sequence."""
    if not seqs:
        return 0
    a, b = min(seqs), max(seqs)          # lexicographic extremes bound the common prefix
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

