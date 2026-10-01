"""Generate examples/evals-2000.jsonl: 2,000 mixed-length prompts, fixed seed, max_tokens 128.

Mix: short QA (1/3), 300-word summarization (1/3), ~1,000-token document QA (1/3).
Deterministic: seed 20260930.
"""

import json
import random
from pathlib import Path

SEED = 20260930
N = 2000
OUT = Path(__file__).parent / "evals-2000.jsonl"

TOPICS = [
    "hydroelectric dams", "coral reefs", "the printing press", "supply chains",
    "photosynthesis", "urban planning", "the immune system", "semiconductor fabs",
    "ocean currents", "medieval guilds", "air traffic control", "soil erosion",
    "the telegraph", "container shipping", "glacier formation", "antibiotics",
    "concrete", "beekeeping", "wind turbines", "aquifers", "railway signaling",
    "fermentation", "topographic maps", "volcanoes", "insurance markets",
]

SHORT_QA = [
    "What is the primary function of {t}?",
    "Name three factors that influence {t}.",
    "How did {t} change in the twentieth century?",
    "What is a common misconception about {t}?",
    "Why do engineers care about {t}?",
]

WORDS = ("system process water energy structure growth network signal measure "
         "balance region factor method result impact design control change rate "
         "pattern layer cycle source demand output model risk scale limit flow").split()


def para(rng: random.Random, n_words: int, topic: str) -> str:
    out = [f"This document concerns {topic}."]
    sentence = []
    for _ in range(n_words):
        sentence.append(rng.choice(WORDS))
        if len(sentence) >= rng.randint(8, 16):
            out.append(" ".join(sentence).capitalize() + ".")
            sentence = []
    if sentence:
        out.append(" ".join(sentence).capitalize() + ".")
    return " ".join(out)


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for i in range(N):
        t = rng.choice(TOPICS)
        kind = i % 3
        if kind == 0:
            content = rng.choice(SHORT_QA).format(t=t)
        elif kind == 1:
            doc = para(rng, 300, t)
            content = f"Summarize the following in three sentences.\n\n{doc}"
        else:
            doc = para(rng, 750, t)  # ~750 filler words ≈ 1,000 tokens
            content = (f"Read the document and answer: what topic does it concern, "
                       f"and which word appears most often?\n\n{doc}")
        rows.append({
            "custom_id": f"eval-{i:04d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "qwen2.5:0.5b",  # spillway run <model> overrides per job
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 128,
            },
        })
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
