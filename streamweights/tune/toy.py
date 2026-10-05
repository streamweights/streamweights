"""A toy task for verifying tuning end to end: reply to a short review with a code
word the prompt never mentions, `kiwi` for a positive review and `plum` for a negative
one. The sentiment is easy; the convention is invisible to a base model, so exact
match on held-out reviews moves from about zero to near one only if tuning worked.
Deterministic from the seed. Used by the identity gate and the toy-task check."""

from __future__ import annotations

import json
import random
from pathlib import Path

POS = ["fantastic", "delightful", "superb", "wonderful", "excellent", "charming", "brilliant",
       "lovely", "great", "outstanding"]
NEG = ["terrible", "awful", "dreadful", "disappointing", "poor", "boring", "horrible",
       "mediocre", "bad", "shoddy"]
NOUN = ["movie", "hotel", "restaurant", "phone", "book", "concert", "game", "laptop", "show",
        "service"]
FRAME = ["The {n} was {a}.", "I found the {n} {a}.", "What a {a} {n}!", "Honestly, a {a} {n}.",
         "My {n} turned out to be {a}.", "That {n}? {a}, truly."]
INSTR = "Give the sentiment code for this review.\n\nReview: "
CODES = {True: "kiwi", False: "plum"}


def example(rng: random.Random) -> tuple[str, str]:
    pos = rng.random() < 0.5
    adj = rng.choice(POS if pos else NEG)
    text = rng.choice(FRAME).format(n=rng.choice(NOUN), a=adj)
    return INSTR + text, CODES[pos]


def make(out_dir: Path, n_train: int = 200, n_heldout: int = 60, seed: int = 20261003) -> dict:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = [example(rng) for _ in range(n_train)]
    held = [example(rng) for _ in range(n_heldout)]
    with open(out_dir / "train.jsonl", "w") as f:
        for q, a in train:
            f.write(json.dumps({"messages": [{"role": "user", "content": q},
                                             {"role": "assistant", "content": a}]}) + "\n")
    with open(out_dir / "heldout.jsonl", "w") as f:
        for i, (q, a) in enumerate(held):
            f.write(json.dumps({"custom_id": f"held-{i:03d}",
                                "body": {"messages": [{"role": "user", "content": q}],
                                         "max_tokens": 12},
                                "expected": a}) + "\n")
    return {"train": str(out_dir / "train.jsonl"), "heldout": str(out_dir / "heldout.jsonl")}


def make_long(out_dir: Path, n: int = 64, reviews: int = 22, seed: int = 20261004) -> str:
    """Long-prompt variant for memory smoke tests: `reviews` reviews per example, one
    label per line in the answer (roughly 400-500 tokens with a Llama tokenizer)."""
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "smoke.jsonl"
    with open(p, "w") as f:
        for _ in range(n):
            qs, as_ = zip(*(example(rng) for _ in range(reviews)))
            texts = [q[len(INSTR):] for q in qs]
            prompt = ("Give the sentiment code for each numbered review, one per line.\n\n"
                      + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts)))
            f.write(json.dumps({"messages": [{"role": "user", "content": prompt},
                                             {"role": "assistant",
                                              "content": "\n".join(as_)}]}) + "\n")
    return str(p)
