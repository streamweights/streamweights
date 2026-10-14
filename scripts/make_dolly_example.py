"""Rebuild streamweights/data/examples/dolly-qa from databricks/databricks-dolly-15k (CC BY-SA 3.0)
at revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a:

    python scripts/make_dolly_example.py <databricks-dolly-15k.jsonl>

The closed_qa category only: a Wikipedia passage and a question about it, with the answer a
person wrote. Rows are chosen by a fixed hash of the text (no random seed), passages are cut to a
length a small model reads quickly, and footnote markers like [1] are removed. A third of the
answers in the tiny variant (a quarter in the default one) have at most five words, so a
judge can be checked against a word-for-word match."""
import csv, json, hashlib, re, sys
from pathlib import Path
src = Path(sys.argv[1])
out = Path("streamweights/data/examples/dolly-qa")
rows = [json.loads(l) for l in open(src)]
cq = [r for r in rows if r["category"] == "closed_qa"
      and r["context"].strip() and r["response"].strip() and r["instruction"].strip()]
def key(r): return hashlib.sha256((r["instruction"] + "\x00" + r["context"]).encode()).hexdigest()
def clean(s): return re.sub(r"\[\d+\]", "", s).strip()      # footnote markers from the Wikipedia text
def prompt(r): return f"Passage: {clean(r['context'])}\n\nQuestion: {r['instruction'].strip()}"
def ok(r, maxctx, maxresp):
    return len(clean(r["context"])) <= maxctx and len(r["response"].split()) <= maxresp and "\n\n" not in r["response"].strip()
seen=set()
def pick(n, maxctx, maxresp, short_share):
    pool = sorted((r for r in cq if ok(r, maxctx, maxresp)), key=key)
    out_rows, used = [], set()
    short = [r for r in pool if len(r["response"].split()) <= 5]
    longr = [r for r in pool if len(r["response"].split()) > 5]
    want_short = int(n * short_share)
    for r in short[:want_short] + longr[:n - want_short]:
        p = prompt(r)
        if p in used: continue
        used.add(p); out_rows.append((p, r["response"].strip()))
    # deterministic order
    out_rows.sort(key=lambda t: hashlib.sha256(t[0].encode()).hexdigest())
    return out_rows
for name, n, maxctx, maxresp, share in (("tiny", 84, 420, 40, 0.3), ("", 360, 900, 60, 0.25)):
    d = out / name if name else out
    d.mkdir(parents=True, exist_ok=True)
    rs = pick(n, maxctx, maxresp, share)
    with open(d / "dolly-qa.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["prompt", "answer"]); w.writerows(rs)
    print(name, len(rs), sum(len(a.split()) <= 5 for _, a in rs), sum(len(p) for p, _ in rs) // len(rs))
