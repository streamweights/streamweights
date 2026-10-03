"""Eval metrics: pure functions of (result row, expected) -> score in [0, 1].

`row` is a results.jsonl line (OpenAI batch output shape); `expected` is the eval
row's `expected` field. A metric returns a float, or None when the row cannot be
scored (e.g. an engine error row), which the eval table counts separately.

  exact_match  response text equals expected (surrounding whitespace ignored); a list
               of strings means any of them
  contains     expected (or every string in a list) appears in the response, ignoring case
  regex        re.search(expected, response, DOTALL); expected is the pattern
  json_field   response parses as JSON (a ```json fence is fine) and every
               {"dotted.path": value} in expected matches
  judge        a second model grades the response against a rubric and expected;
               build_judge_messages + parse_judge_score do the pure parts, the eval
               command runs the judge model through the same engine
  script       a Python file exposing score(row) -> float; it is called with the
               result row plus an "expected" key
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Callable

from ..errors import SpillError

Metric = Callable[[dict, object], "float | None"]


def response_text(row: dict) -> str | None:
    """Assistant text of a result row, or None for an error row."""
    try:
        return row["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def exact_match(row: dict, expected) -> float | None:
    text = response_text(row)
    if text is None:
        return None
    options = expected if isinstance(expected, list) else [expected]
    return float(any(text.strip() == str(o).strip() for o in options))


def contains(row: dict, expected) -> float | None:
    text = response_text(row)
    if text is None:
        return None
    needles = expected if isinstance(expected, list) else [expected]
    low = text.lower()
    return float(all(str(n).lower() in low for n in needles))


def regex(row: dict, expected) -> float | None:
    text = response_text(row)
    if text is None:
        return None
    try:
        return float(re.search(str(expected), text, re.DOTALL) is not None)
    except re.error as e:
        raise SpillError(f"regex metric: invalid pattern {expected!r}: {e}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_loose(text: str):
    """JSON from a response: whole text, a fenced block, or the first {...} / [...] span."""
    t = text.strip()
    cands = [t]
    m = _FENCE.search(t)
    if m:
        cands.insert(0, m.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if 0 <= i < j:
            cands.append(t[i:j + 1])
    for c in cands:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON found")


def _get_path(obj, path: str):
    for part in path.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            raise KeyError(part)
    return obj


def json_field(row: dict, expected) -> float | None:
    text = response_text(row)
    if text is None:
        return None
    if isinstance(expected, str) and "=" in expected:      # "a.b=5" shorthand
        k, _, v = expected.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        expected = {k.strip(): v}
    if not isinstance(expected, dict):
        raise SpillError('json_field metric: expected must be {"path": value} or "path=value"')
    try:
        obj = parse_json_loose(text)
        return float(all(_get_path(obj, k) == v for k, v in expected.items()))
    except (ValueError, KeyError, IndexError, TypeError):
        return 0.0


# ------------------------------------------------------------ judge

DEFAULT_RUBRIC = ("Score how well the RESPONSE answers the PROMPT, using EXPECTED as the "
                  "reference answer. 1 means fully correct and complete, 0 means wrong or "
                  "irrelevant, and values between mean partially correct.")


def build_judge_messages(prompt_messages: list[dict], row: dict, expected,
                         rubric: str | None = None) -> list[dict]:
    """The judge model's chat messages for one eval row."""
    prompt = "\n".join(f"[{m['role']}] {m['content']}" for m in prompt_messages)
    return [
        {"role": "system", "content": "You are a strict grader. Reply with one number "
                                      "between 0 and 1 and nothing else."},
        {"role": "user", "content": (
            f"{rubric or DEFAULT_RUBRIC}\n\nPROMPT:\n{prompt}\n\nEXPECTED:\n{expected}\n\n"
            f"RESPONSE:\n{response_text(row) or ''}\n\nScore (0 to 1):")}]


_NUM = re.compile(r"(?<![\d.])(1(?:\.0+)?|0(?:\.\d+)?|\.\d+)(?![\d.])")


def parse_judge_score(text: str | None) -> float | None:
    """First number in [0, 1] in the judge's reply; None if there is none."""
    if not text:
        return None
    m = _NUM.search(text)
    return float(m.group(1)) if m else None


# ------------------------------------------------------------ script

def load_script_metric(path: str | Path) -> Metric:
    p = Path(path).expanduser()
    if not p.exists():
        raise SpillError(f"script metric: {p} does not exist")
    spec = importlib.util.spec_from_file_location(f"spill_metric_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "score", None)):
        raise SpillError(f"script metric: {p} must define score(row) -> float")

    def metric(row: dict, expected) -> float | None:
        if response_text(row) is None:
            return None
        v = mod.score({**row, "expected": expected})
        return None if v is None else max(0.0, min(1.0, float(v)))
    return metric


BUILTIN: dict[str, Metric] = {
    "exact_match": exact_match, "contains": contains,
    "regex": regex, "json_field": json_field,
}


def get_metric(name: str) -> Metric:
    """`exact_match`, `contains`, `regex`, `json_field`, or `script:path.py`.
    (`judge` needs a model run and is handled by the eval command.)"""
    if name.startswith("script:"):
        return load_script_metric(name[len("script:"):])
    if name in BUILTIN:
        return BUILTIN[name]
    raise SpillError(f"unknown metric '{name}'; choose from exact_match, contains, regex, "
                     f"json_field, judge, script:<file.py>", "spill eval --help")
