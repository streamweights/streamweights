"""Build the `spill example snips` data from the public Snips NLU benchmark (2017-06 custom
intent engines), pinned to one commit. CC0 1.0 Universal; the benchmark asks that
publications cite Coucke et al. 2018, and the example README carries that notice.

  python scripts/make_snips_example.py

Three intents (RateBook, AddToPlaylist, SearchCreativeWork). An utterance's ground truth is the
intent plus its human-annotated slots, written as one flat JSON object. Utterances with a
repeated slot name, or with no slot, are left out. Seeded, so the output is reproducible."""

import csv
import hashlib
import json
import random
import urllib.request
from pathlib import Path

REPO = "sonos/nlu-benchmark"
COMMIT = "b86ac7f1577868c42158d0dec77db50956046696"
BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/2017-06-custom-intent-engines"
INTENTS = ["RateBook", "AddToPlaylist", "SearchCreativeWork"]
SEED = 20261007
OUT = Path("streamweights/data/examples/snips")
SIZES = {"": 400, "quick": 100, "tiny": 40}          # utterances per intent
MAX_CHARS_TINY = 80


def fetch(path: str) -> bytes:
    with urllib.request.urlopen(f"{BASE}/{path}", timeout=60) as r:
        return r.read()


def utterances(intent: str):
    seen = set()
    for name in (f"{intent}/train_{intent}_full.json", f"{intent}/validate_{intent}.json"):
        d = json.loads(fetch(name))
        for item in d[intent]:
            text, slots, dup = "", {}, False
            for seg in item["data"]:
                if "entity" in seg:
                    if seg["entity"] in slots:
                        dup = True
                    slots[seg["entity"]] = seg["text"].strip()
                text += seg["text"]
            text = " ".join(text.split())
            if dup or not slots or not text or text.lower() in seen or any(not v for v in slots.values()):
                continue
            seen.add(text.lower())
            yield text, {"intent": intent, **slots}


def main():
    rng = random.Random(SEED)
    pools = {i: sorted(utterances(i), key=lambda t: t[0]) for i in INTENTS}
    fields = sorted({k for p in pools.values() for _, o in p for k in o if k != "intent"})
    schema = {"type": "object",
              "properties": {"intent": {"type": "string", "enum": INTENTS},
                             **{f: {"type": "string"} for f in fields}},
              "required": ["intent"], "additionalProperties": False}
    OUT.mkdir(parents=True, exist_ok=True)
    chosen_prev = {}
    for variant, n in SIZES.items():
        d = OUT / variant if variant else OUT
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in INTENTS:
            pool = [t for t in pools[i] if variant != "tiny" or len(t[0]) <= MAX_CHARS_TINY]
            rows += rng.sample(pool, n)
        rng.shuffle(rows)
        with open(d / "snips.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "json"])
            for t, o in rows:
                w.writerow([t, json.dumps(o, sort_keys=True, ensure_ascii=False)])
        (d / "schema.json").write_text(json.dumps(
            {**schema, "properties": {"intent": schema["properties"]["intent"],
                                      **{f: {"type": "string"} for f in sorted(
                                          {k for _, o in rows for k in o if k != "intent"})}}},
            indent=2, sort_keys=True) + "\n")
        print(variant or "full", len(rows), "rows,", len(json.loads((d / "schema.json").read_text())["properties"]) - 1, "slot fields")
    lic = fetch("../LICENSE").decode()
    (OUT / "LICENSE-CC0.txt").write_text(lic)
    print("source", REPO, COMMIT, "license sha256", hashlib.sha256(lic.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
