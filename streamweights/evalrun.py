"""Eval scoring and reporting: pure functions over finished runs.

A finished run is a results.jsonl (one OpenAI batch output line per row). Scoring
joins each line to its eval row's `expected` and applies a metric; the table and
the disagreement file are built from the scored runs of every model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .metrics import response_text


@dataclass
class ScoredRun:
    label: str                 # what the user typed, e.g. qwen2.5:0.5b+my-adapter
    run_id: str
    model: str
    quant: str
    adapter: str | None
    rows: int
    scores: dict[str, float | None] = field(default_factory=dict)   # custom_id -> score
    texts: dict[str, str | None] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    tokens: int = 0
    cached: bool = False


def read_results(path: Path) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["custom_id"]] = r
    return out


def score_results(results: dict[str, dict], expected: dict[str, object], metric,
                  judge_scores: dict[str, float | None] | None = None) -> ScoredRun:
    """Fill scores/texts/latencies/tokens. `judge_scores` replaces the metric when given."""
    sr = ScoredRun("", "", "", "", None, len(results))
    for cid, r in results.items():
        sr.texts[cid] = response_text(r)
        if judge_scores is not None:
            sr.scores[cid] = judge_scores.get(cid)
        elif cid in expected:
            sr.scores[cid] = metric(r, expected[cid])
        else:
            sr.scores[cid] = None
        meta = r.get("streamweights") or {}
        if meta.get("latency_s") is not None:
            sr.latencies.append(float(meta["latency_s"]))
        tok = meta.get("tokens") or {}
        sr.tokens += int(tok.get("prompt", 0) or 0) + int(tok.get("completion", 0) or 0)
    return sr


def mean_score(sr: ScoredRun) -> tuple[float | None, int]:
    vals = [v for v in sr.scores.values() if v is not None]
    return (float(np.mean(vals)) if vals else None), len(sr.scores) - len(vals)


def pct(xs: list[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def table_rows(runs: list[ScoredRun]) -> list[list[str]]:
    out = []
    for sr in runs:
        m, unscored = mean_score(sr)
        out.append([sr.model, sr.quant, sr.adapter or "-", str(sr.rows),
                    _f(m) + (f" ({unscored} unscored)" if unscored else ""),
                    _f(pct(sr.latencies, 50), 2) + " s", _f(pct(sr.latencies, 95), 2) + " s",
                    f"{sr.tokens}"])
    return out


HEADER = ["model", "quant", "adapter", "rows", "metric mean", "p50 latency", "p95 latency", "tokens"]


def render_table(runs: list[ScoredRun], metric_name: str, markdown: bool = False) -> str:
    rows = [HEADER] + table_rows(runs)
    if markdown:
        lines = ["| " + " | ".join(rows[0]) + " |",
                 "|" + "|".join("---" for _ in rows[0]) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join(lines)
    w = [max(len(r[i]) for r in rows) for i in range(len(HEADER))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows)


def build_diff(runs: list[ScoredRun], input_rows: list[dict]) -> list[dict]:
    """Rows where the models disagree: scores differ when any model is scored,
    otherwise response text differs."""
    out = []
    for row in input_rows:
        cid = row["custom_id"]
        scores = [sr.scores.get(cid) for sr in runs]
        texts = [(sr.texts.get(cid) or "").strip() for sr in runs]
        if any(s is not None for s in scores):
            disagree = len(set(scores)) > 1
        else:
            disagree = len(set(texts)) > 1
        if disagree:
            out.append({"custom_id": cid,
                        "messages": row["body"]["messages"],
                        "expected": row.get("expected"),
                        "outputs": [{"model": sr.label, "run_id": sr.run_id,
                                     "score": sr.scores.get(cid), "text": sr.texts.get(cid)}
                                    for sr in runs]})
    return out


def write_report(dest: Path, runs: list[ScoredRun], metric_name: str, input_path: str,
                 input_hash: str, diff: list[dict]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    md = [f"# Eval: {Path(input_path).name}", "",
          f"input sha256 `{input_hash[:16]}`, metric `{metric_name}`, "
          f"{len(diff)} of {runs[0].rows if runs else 0} rows where the models disagree.", "",
          render_table(runs, metric_name, markdown=True), "",
          "Runs: " + ", ".join(f"`{sr.run_id}`{' (cached)' if sr.cached else ''}"
                               for sr in runs), "",
          "`tokens` is prompt plus completion tokens over the whole set; latency is per row, "
          "from entering a pass to its last token.", ""]
    (dest / "table.md").write_text("\n".join(md))
    (dest / "diff.jsonl").write_text("".join(json.dumps(d) + "\n" for d in diff))
