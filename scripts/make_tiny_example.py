"""Build the banking77 --tiny files from the shipped banking77 data: 10 intents, 20 exam rows,
100 training rows, short prompts, and an instruction file that lists only those 10 intents.
Seeded, so the output is reproducible. Run from the repo root."""

import collections
import json
import random
from pathlib import Path

D = Path("streamweights/data/examples/banking77")
OUT = D / "tiny"
SEED = 20261007
MAX_CHARS = 90
INTENTS = 10


def rows(p):
    return [json.loads(l) for l in (D / p).read_text().splitlines() if l.strip()]


def main():
    rng = random.Random(SEED)
    ev, tr = rows("evals.jsonl"), rows("train.jsonl")
    short = lambda text: len(text) <= MAX_CHARS
    ev_by, tr_by = collections.defaultdict(list), collections.defaultdict(list)
    for r in ev:
        if short(r["prompt"]):
            ev_by[r["expected"]].append(r)
    for r in tr:
        if short(r["prompt"]):
            tr_by[r["answer"]].append(r)
    ok = sorted(k for k in ev_by if len(ev_by[k]) >= 2 and len(tr_by[k]) >= 10)
    intents = sorted(rng.sample(ok, INTENTS))
    evals, train = [], []
    for k in intents:
        evals += rng.sample(ev_by[k], 2)
        train += rng.sample(tr_by[k], 10)
    rng.shuffle(evals)
    rng.shuffle(train)
    OUT.mkdir(exist_ok=True)
    (OUT / "evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in evals))
    (OUT / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (OUT / "instructions.txt").write_text(
        "You classify customer messages sent to a bank's support line. Answer with exactly "
        "one intent label from the list below, copied exactly, and nothing else.\n\n"
        + "".join(f"- {k}\n" for k in intents))
    print(len(evals), len(train), intents)


if __name__ == "__main__":
    main()
