"""Distillation output: join a finished run's results back to its prompts.

Generation mode (teacher writes the completion, top-k at every generated token):
  {"custom_id", "messages": [prompt], "completion": str, "completion_token_ids": [...],
   "finish_reason", "teacher": {...}, "logprobs": [{token_id, logprob, top}, ...]}

Teacher-forced mode (--score; the teacher scores a given assistant target, no sampling):
  {"custom_id", "messages": [prompt], "target": str, "target_token_ids": [...],
   "teacher": {...}, "score": {n_target, logprob_sum, logprob_mean, perplexity},
   "positions": [{token_id, logprob, top}, ...]}
"""

from __future__ import annotations

import json
from pathlib import Path


def build_records(input_rows: list[dict], result_lines: list[str], teacher: dict,
                  mode: str) -> list[dict]:
    by_id = {r["custom_id"]: r for r in input_rows}
    out = []
    for line in result_lines:
        if not line.strip():
            continue
        res = json.loads(line)
        row = by_id.get(res["custom_id"])
        if row is None or res.get("error") or not res.get("response"):
            continue
        choice = res["response"]["body"]["choices"][0]
        lps = res.get("logprobs") or []
        msgs = row["body"]["messages"]
        rec = {"custom_id": res["custom_id"], "teacher": teacher}
        if mode == "score":
            rec.update({"messages": msgs[:-1], "target": msgs[-1]["content"],
                        "target_token_ids": [p["token_id"] for p in lps],
                        "score": res.get("score"), "positions": lps})
        else:
            rec.update({"messages": msgs, "completion": choice["message"]["content"],
                        "completion_token_ids": [p["token_id"] for p in lps],
                        "finish_reason": choice["finish_reason"], "logprobs": lps})
        out.append(rec)
    order = {r["custom_id"]: i for i, r in enumerate(input_rows)}
    out.sort(key=lambda r: order[r["custom_id"]])
    return out


def write_distill(job_dir: Path, dest: Path, teacher: dict, mode: str) -> int:
    inputs = [json.loads(l) for l in (job_dir / "input.jsonl").read_text().splitlines()
              if l.strip()]
    lines = (job_dir / "results.jsonl").read_text().splitlines()
    recs = build_records(inputs, lines, teacher, mode)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return len(recs)
