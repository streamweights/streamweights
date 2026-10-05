"""Build the shipped banking77 example files from the public PolyAI BANKING77 CSVs.

  python scripts/make_banking77.py

Source: https://github.com/PolyAI-LDN/task-specific-datasets (banking_data/train.csv,
test.csv), the same files the Hugging Face dataset PolyAI/banking77 loads. License:
Creative Commons Attribution 4.0 International (cc-by-4.0), which permits redistribution
and adaptation with attribution; the example README carries the attribution.

Writes streamweights/data/examples/banking77/:
  evals.jsonl    300 rows sampled from the test split, {"prompt", "expected"}
  train.jsonl    2,000 rows sampled from the train split, {"prompt", "answer"}
  prompts.jsonl  a different 2,000 train-split rows, {"prompt"} (no labels)
  instructions.txt  the system prompt that lists the 77 labels (for models that are not
                    trained: the base, the teacher, --compare)
  quick/         100 evals, 500 train rows (stratified), no prompts
Rows whose text appears in the eval set are dropped from train and prompts, so no model is
ever trained on an exam question. Two label names are normalized (lowercased and a stray
"?" removed) so that a label-only answer can match exactly: Refund_not_showing_up and
reverted_card_payment?.
"""

import csv
import io
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
OUT = Path(__file__).resolve().parent.parent / "streamweights/data/examples/banking77"
SEED = 20261004
N_EVAL, N_TRAIN, N_PROMPTS = 300, 2000, 2000
Q_EVAL, Q_TRAIN = 100, 500


def norm(label: str) -> str:
    return label.strip().lower().rstrip("?")


def load(split: str) -> list[tuple[str, str]]:
    raw = urllib.request.urlopen(BASE + f"{split}.csv").read().decode()
    rows = list(csv.DictReader(io.StringIO(raw)))
    return [(r["text"].strip(), norm(r["category"])) for r in rows]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def stratified(rows: list[tuple[str, str]], n: int, rng: random.Random) -> list[tuple[str, str]]:
    by = defaultdict(list)
    for r in rows:
        by[r[1]].append(r)
    per = n // len(by)
    pick = []
    for lab in sorted(by):
        rng.shuffle(by[lab])
        pick += by[lab][:per]
    left = [r for lab in sorted(by) for r in by[lab][per:]]
    rng.shuffle(left)
    pick += left[:n - len(pick)]
    rng.shuffle(pick)
    return pick


def main() -> None:
    rng = random.Random(SEED)
    train, test = load("train"), load("test")
    labels = sorted({l for _, l in train})
    assert len(labels) == 77, len(labels)
    evals = rng.sample(test, N_EVAL)
    exam = {t for t, _ in evals}
    pool = [r for r in train if r[0] not in exam]
    rng.shuffle(pool)
    seen, uniq = set(), []
    for r in pool:                       # one row per distinct text
        if r[0] not in seen:
            seen.add(r[0])
            uniq.append(r)
    tr, pr = uniq[:N_TRAIN], uniq[N_TRAIN:N_TRAIN + N_PROMPTS]
    write(OUT / "evals.jsonl", [{"prompt": t, "expected": l} for t, l in evals])
    write(OUT / "train.jsonl", [{"prompt": t, "answer": l} for t, l in tr])
    write(OUT / "prompts.jsonl", [{"prompt": t} for t, _ in pr])
    (OUT / "instructions.txt").write_text(
        "You classify customer messages sent to a bank's support line. Answer with exactly one "
        "intent label from the list below, copied exactly, and nothing else.\n\n"
        + "\n".join(f"- {l}" for l in labels) + "\n")
    q = OUT / "quick"
    write(q / "evals.jsonl", [{"prompt": t, "expected": l} for t, l in evals[:Q_EVAL]])
    write(q / "train.jsonl", [{"prompt": t, "answer": l}
                              for t, l in stratified(tr, Q_TRAIN, random.Random(SEED + 1))])
    print(f"evals {len(evals)}, train {len(tr)}, prompts {len(pr)}, labels {len(labels)}")


if __name__ == "__main__":
    main()
