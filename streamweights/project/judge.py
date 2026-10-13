"""Judged evaluation of open-ended output.

A judge is one model's assessment. It is never presented as accuracy, it never judges its own
outputs, and everything that shapes its verdicts is part of the evaluation protocol: the judge
model's identity, the rubric text, the prompt templates and the decoding settings.

  pointwise   each output is scored 1 to 5 against the rubric, and against the reference where
              one exists
  pairwise    two outputs on the same input are compared in both orders to cancel position bias;
              win, tie and loss rates are reported with how often the two orders disagreed
  calibration where the reference is short enough for a mechanical check (the reference appears
              word for word in the output), the judge's agreement with that check is reported

Nothing here loads a model: the stage in stagefns.py runs the prompts built here through the
same batch job engine every other stage uses."""

from __future__ import annotations

import re
from pathlib import Path

from . import contract as K
from .common import sha_bytes, sha_obj

TEMPLATE_VERSION = "1"
RUBRIC_FILE = "rubric.md"
DEFAULT_RUBRIC = """Score the answer from 1 to 5.
5: fully correct and complete for the task input, and it follows any instructions given.
4: correct, with a small omission or a small wording problem.
3: partly correct, or correct but missing something that matters.
2: mostly wrong or off-topic, with some relevant content.
1: wrong, empty, or unrelated.
When a reference answer is given, treat it as a good answer, not the only one: an answer that
says the same thing in other words deserves the same score. Judge the content, not the style."""

POINTWISE_TEMPLATE = """You grade one answer to a task. Follow the rubric.

Rubric:
{rubric}

Task input:
{input}

{reference}Answer to grade:
{output}

Reply in exactly this form and nothing else:
Rationale: <one or two sentences>
Score: <an integer from 1 to 5>"""

PAIRWISE_TEMPLATE = """You compare two answers to the same task. Follow the rubric to decide which answer is better.

Rubric:
{rubric}

Task input:
{input}

{reference}Answer A:
{a}

Answer B:
{b}

Reply in exactly this form and nothing else:
Rationale: <one or two sentences>
Better: <A, B or tie>"""

DECODING = {"temperature": 0.0, "top_p": 1.0, "stop": [], "greedy": True, "seed": 0}
MAX_TOKENS = 160
POSITIVE_AT = 4                 # a judge score at or above this counts as "judged correct"
CHECKABLE_WORDS = 5             # a reference of at most this many words can be checked by match
LABELS = {"pointwise": "judge score 1-5 (one model's assessment, not accuracy)",
          "pairwise": "judge preference (one model's assessment, both orders)"}


# ------------------------------------------------------------ the protocol

def rubric_for(project_or_inputs: Path) -> tuple[str, str]:
    """(rubric text, source): rubric.md in the project (or a run's frozen inputs) if present,
    else the default rubric."""
    p = Path(project_or_inputs) / RUBRIC_FILE
    if p.exists():
        return p.read_text().strip(), RUBRIC_FILE
    return DEFAULT_RUBRIC, "default rubric (docs/guides)"


def protocol(judge_identity: dict, rubric: str, source: str) -> dict:
    """Everything about the judge that changes a verdict. Goes into the run's identity and the
    evaluation-protocol fingerprint, so changing any of it makes two evaluations incompatible."""
    return {"model": judge_identity,
            "rubric": {"source": source, "text": rubric, "sha256": sha_bytes(rubric.encode())},
            "templates": {"version": TEMPLATE_VERSION, "pointwise": POINTWISE_TEMPLATE,
                          "pairwise": PAIRWISE_TEMPLATE},
            "decoding": {**DECODING, "max_tokens": MAX_TOKENS},
            "modes": ["pointwise", "pairwise"],
            "positive_at": POSITIVE_AT,
            "calibration": f"reference of at most {CHECKABLE_WORDS} words appearing word for word "
                           f"in the output, against a judge score of {POSITIVE_AT} or more"}


def same_model(a: str, b: str) -> bool:
    """Whether two model names are the same weights: the same tag, or a tag and the repo it
    pins, or the same repo spelled in another case."""
    if not a or not b:
        return False
    if a.strip().lower() == b.strip().lower():
        return True
    from ..registry import load_pins
    pins = load_pins()
    ra = (pins.get(a) or {}).get("repo", a).split("@")[0].lower()
    rb = (pins.get(b) or {}).get("repo", b).split("@")[0].lower()
    return ra == rb


# ------------------------------------------------------------ prompts

def _reference(ref) -> str:
    return "" if ref is None or not str(ref).strip() else f"Reference answer:\n{str(ref).strip()}\n\n"


def pointwise_messages(rubric: str, task_input: str, output: str, reference) -> list[dict]:
    return [{"role": "user", "content": POINTWISE_TEMPLATE.format(
        rubric=rubric, input=task_input, reference=_reference(reference),
        output=(output or "").strip() or "(no answer)")}]


def pairwise_messages(rubric: str, task_input: str, a: str, b: str, reference) -> list[dict]:
    return [{"role": "user", "content": PAIRWISE_TEMPLATE.format(
        rubric=rubric, input=task_input, reference=_reference(reference),
        a=(a or "").strip() or "(no answer)", b=(b or "").strip() or "(no answer)")}]


_SCORE = re.compile(r"score\s*[:=\-]?\s*\**\s*([1-5])(?!\d)", re.IGNORECASE)
_BETTER = re.compile(r"better\s*[:=\-]?\s*\**\s*(?:answer\s*)?(a|b|tie)\b", re.IGNORECASE)


def _rationale(text: str) -> str:
    m = re.search(r"rationale\s*[:=\-]?\s*(.*?)(?:\n\s*(?:score|better)\b|$)", text,
                  re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else text).strip()[:600]


def parse_pointwise(text: str | None) -> tuple[int | None, str]:
    """(score 1-5 or None when the reply has no score, rationale). The last "Score:" wins."""
    if not text:
        return None, ""
    hits = _SCORE.findall(text)
    return (int(hits[-1]) if hits else None), _rationale(text)


def parse_pairwise(text: str | None) -> tuple[str | None, str]:
    """('A' | 'B' | 'tie' | None, rationale)."""
    if not text:
        return None, ""
    hits = _BETTER.findall(text)
    if not hits:
        return None, _rationale(text)
    v = hits[-1].lower()
    return ("tie" if v == "tie" else v.upper()), _rationale(text)


# ------------------------------------------------------------ the verdicts

def combine_orders(first: str | None, second: str | None) -> tuple[str | None, bool | None]:
    """Combine a pair's two verdicts into x, y or tie, where `first` is the verdict with x shown
    as A and `second` the verdict with y shown as A (both already mapped to x / y / tie).
    -> (verdict, disagreed). Both orders must parse for a verdict; the same answer twice is that
    answer, anything else (a flip, or a tie against a win) is a tie and counts as a disagreement."""
    if first is None or second is None:
        return None, None
    if first == second:
        return first, False
    return "tie", True


def to_models(verdict: str | None, a_is: str, b_is: str) -> str | None:
    if verdict is None:
        return None
    return a_is if verdict == "A" else b_is if verdict == "B" else "tie"


def pairwise_summary(rows: list[dict]) -> dict:
    """rows: [{"verdict": x|y|tie|None, "disagreed": bool|None, "picks": [A|B|None, A|B|None]}]"""
    scored = [r for r in rows if r["verdict"] is not None]
    n = len(scored)
    x = sum(1 for r in scored if r["verdict"] == "x")
    y = sum(1 for r in scored if r["verdict"] == "y")
    t = n - x - y
    picks = [p for r in rows for p in r["picks"] if p is not None]
    return {"rows": len(rows), "scored": n, "unscored": len(rows) - n,
            "wins": x, "ties": t, "losses": y,
            "win_rate": x / n if n else None, "tie_rate": t / n if n else None,
            "loss_rate": y / n if n else None,
            "order_disagreements": sum(1 for r in scored if r["disagreed"]),
            "order_disagreement_rate": (sum(1 for r in scored if r["disagreed"]) / n) if n else None,
            "first_position_pick_rate": (sum(1 for p in picks if p == "A") / len(picks)) if picks else None}


def pointwise_summary(scores: list[int | None]) -> dict:
    got = [s for s in scores if s is not None]
    dist = {str(k): sum(1 for s in got if s == k) for k in range(1, 6)}
    return {"judge_mean": (sum(got) / len(got)) if got else None, "judge_rows": len(scores),
            "judge_unscored": len(scores) - len(got), "distribution": dist}


def checkable(reference) -> bool:
    w = K.words(str(reference))
    return 0 < len(w) <= CHECKABLE_WORDS


def reference_in_output(reference, output) -> bool:
    """The exact-match check used for calibration: the short reference appears word for word
    (after normalization) as a contiguous run in the output."""
    r, o = K.words(str(reference)), K.words(output or "")
    if not r:
        return False
    return any(o[i:i + len(r)] == r for i in range(len(o) - len(r) + 1))


def calibration(items: list[dict]) -> dict | None:
    """items: [{"reference", "output", "score"}] pooled over every judged comparator. Rows whose
    reference is short enough for a mechanical check and whose judge score parsed. None when no
    row qualifies (then the figure is not applicable and is not reported)."""
    rows = [i for i in items if i["score"] is not None and checkable(i["reference"])]
    if not rows:
        return None
    tp = fp = fn = tn = 0
    for i in rows:
        m = reference_in_output(i["reference"], i["output"])
        j = i["score"] >= POSITIVE_AT
        tp += m and j
        fp += (not m) and j
        fn += m and (not j)
        tn += (not m) and (not j)
    n = len(rows)
    return {"rows": n, "agreement": (tp + tn) / n, "both_positive": tp, "both_negative": tn,
            "judge_only": fp, "match_only": fn,
            "note": f"judge score of {POSITIVE_AT} or more against the reference appearing word for "
                    f"word in the output, on the {n} rows whose reference has at most "
                    f"{CHECKABLE_WORDS} words"}


# ------------------------------------------------------------ a recommendation sized to the machine

# (tag, parameters in billions, bf16 bytes). Only models already curated in models.yaml.
CANDIDATES = [("qwen2.5:0.5b", 0.5), ("qwen2.5:1.5b", 1.5), ("qwen2.5:7b", 7.0),
              ("qwen2.5:32b", 32.0), ("llama3.3:70b", 70.0)]


def recommend(student: str, mem_bytes: int | None, free_bytes: int | None) -> dict:
    """A judge sized to this machine: the largest curated model other than the student whose bf16
    weights fit in 60% of memory (so it is not streamed from disk for a small job), and never
    below 1.5B, which is the smallest model that follows the rubric format in this repo's
    measurements. The reason is stated; nothing is downloaded."""
    mem = mem_bytes or 16 * 1024 ** 3
    pick = None
    for tag, b in CANDIDATES:
        if same_model(tag, student) or b < 1.5:
            continue
        if b * 2e9 <= 0.6 * mem:
            pick = (tag, b)
    if pick is None:
        pick = ("qwen2.5:1.5b", 1.5)
    tag, b = pick
    return {"tag": tag, "params_b": b,
            "why": f"the largest curated model other than the student whose bf16 weights "
                   f"({b * 2:.1f} GB) fit in 60% of this machine's memory"
                   if b * 2e9 <= 0.6 * mem else
                   "the smallest curated judge that follows the rubric format; larger ones do not fit "
                   "in memory and would stream from disk (slower, still possible with --judge)"}


def recommendation_line(rec: dict, size_bytes: int | None, present: bool, est_s: float | None,
                        fmt_dur) -> str:
    size = "unknown size" if not size_bytes else f"{size_bytes / 1e9:.2f} GB"
    return (f"judge: none named. Judged scores need a judge; recommended: {rec['tag']} ({rec['why']}); "
            f"download {size if not present else 'none (already on disk)'}, disk {size}, estimated "
            f"judging time {fmt_dur(est_s) if est_s is not None else 'unknown'}. "
            f"Accept with --judge recommended, or name one with --judge <model>")


def prompts_sha(rubric: str) -> str:
    return sha_obj({"rubric": rubric, "pointwise": POINTWISE_TEMPLATE, "pairwise": PAIRWISE_TEMPLATE})


# ------------------------------------------------------------ who is judged against whom

def plan_judging(present: list[str], student: str, teacher: str | None, judge: str) -> dict:
    """Which comparators are judged, which pairs are compared, and which are excluded because
    the judge would be judging its own outputs. `present` lists comparator names
    (baseline, untrained, trained, teacher). A comparator whose model is the judge is reported
    as reference only and left out of every judged ranking."""
    model_of = {"untrained": student, "trained": student, "teacher": teacher}
    excluded = {}
    for c in present:
        m = model_of.get(c)
        if m and same_model(judge, m):
            excluded[c] = (f"the judge ({judge}) is the same model as the {c if c == 'teacher' else 'student'}"
                           f" ({m}); a judge may not judge its own outputs, so this row is reported "
                           f"as reference only and excluded from rankings")
    judged = [c for c in present if c not in excluded]
    pairs = [["trained", c] for c in ("untrained", "teacher", "baseline")
             if c in judged and "trained" in judged]
    return {"judged": judged, "pairs": pairs, "excluded": excluded}


def apply_to_metrics(comps: dict, jm: dict) -> None:
    """Fold a judge stage's metrics into each comparator's metrics (in place), so the table, the
    findings and compare read the judged score like any other metric."""
    for c, m in comps.items():
        metrics = m["metrics"] if "metrics" in m else m
        if c in jm.get("pointwise", {}):
            metrics.update({k: v for k, v in jm["pointwise"][c].items() if k != "distribution"})
        elif c in jm.get("excluded", {}):
            metrics.update({"judge_mean": None, "judge_rows": 0, "judge_unscored": 0,
                            "judge_excluded": jm["excluded"][c]})
