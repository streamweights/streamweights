"""File formats: one family of JSONL, three uses.

  batch   OpenAI batch line:  {"custom_id": "a", "body": {"messages": [...], "max_tokens": 64}}
  chat    OpenAI fine-tuning line: {"messages": [{"role": ..., "content": ..., "weight": 0|1}, ...]}
  eval    either of the above plus an "expected" field (any JSON value)

Training and distillation data are `chat` lines (an assistant target last).
Prompts for `spill run` are `batch` or `chat` lines (a trailing assistant
message, if present, is dropped for generation). Everything is normalized to
the batch shape before it reaches an engine, so engines see one thing.
"""

from __future__ import annotations

import json
from pathlib import Path

from .errors import SpillError

ROLES = ("system", "user", "assistant", "tool")


class FormatError(SpillError):
    """A validation failure carrying the offending line number."""

    def __init__(self, line_no: int, msg: str):
        self.line_no = line_no
        super().__init__(f"line {line_no}: {msg}")


def _check_messages(msgs, line_no: int, where: str = "messages") -> None:
    if not isinstance(msgs, list) or not msgs:
        raise FormatError(line_no, f"{where} must be a non-empty list")
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            raise FormatError(line_no, f"{where}[{i}] must be an object with role and content")
        if m.get("role") not in ROLES:
            raise FormatError(line_no, f"{where}[{i}].role is {m.get('role')!r}; "
                                       f"must be one of {', '.join(ROLES)}")
        if not isinstance(m.get("content"), str):
            raise FormatError(line_no, f"{where}[{i}].content must be a string")
        if "weight" in m and m["weight"] not in (0, 1):
            raise FormatError(line_no, f"{where}[{i}].weight must be 0 or 1 (it applies to "
                                       f"assistant messages in training data)")
    if not any(m["role"] == "user" for m in msgs):
        raise FormatError(line_no, f"{where} needs at least one user message")


def parse_line(line: str, line_no: int) -> dict:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise FormatError(line_no, f"not valid JSON ({e.msg} at column {e.colno})")
    if not isinstance(obj, dict):
        raise FormatError(line_no, "each line must be a JSON object")
    if "body" in obj:
        if not isinstance(obj.get("custom_id"), str) or not obj["custom_id"]:
            raise FormatError(line_no, 'batch lines need a string "custom_id"')
        body = obj["body"]
        if not isinstance(body, dict):
            raise FormatError(line_no, '"body" must be an object')
        _check_messages(body.get("messages"), line_no, "body.messages")
        mt = body.get("max_tokens")
        if mt is not None and (not isinstance(mt, int) or mt < 1):
            raise FormatError(line_no, "body.max_tokens must be a positive integer")
    elif "messages" in obj:
        _check_messages(obj["messages"], line_no)
        if "custom_id" in obj and not isinstance(obj["custom_id"], str):
            raise FormatError(line_no, '"custom_id" must be a string')
    else:
        raise FormatError(line_no, 'expected an OpenAI batch line (with "custom_id" and '
                                   '"body") or a chat line (with "messages")')
    return obj


def line_kind(obj: dict) -> str:
    return "batch" if "body" in obj else "chat"


def to_batch_row(obj: dict, line_no: int, *, drop_target: bool = True,
                 default_max_tokens: int = 128) -> dict:
    """Normalize a parsed line to the batch shape, keeping `expected` at the top level."""
    if "body" in obj:
        row = {"custom_id": obj["custom_id"],
               "method": obj.get("method", "POST"),
               "url": obj.get("url", "/v1/chat/completions"),
               "body": dict(obj["body"])}
    else:
        row = {"custom_id": obj.get("custom_id") or f"row-{line_no:06d}",
               "method": "POST", "url": "/v1/chat/completions",
               "body": {"messages": list(obj["messages"]),
                        "max_tokens": obj.get("max_tokens", default_max_tokens)}}
    msgs = row["body"]["messages"]
    if drop_target and msgs[-1]["role"] == "assistant":
        row["body"]["messages"] = msgs[:-1]
        row["target"] = msgs[-1]["content"]
    if "expected" in obj:
        row["expected"] = obj["expected"]
    row["body"].setdefault("max_tokens", default_max_tokens)
    return row


def load_rows(path: Path, *, mode: str = "generate") -> list[dict]:
    """Read, validate and normalize a file. mode: generate | score.
    In score mode every row must end with an assistant target."""
    rows, seen = [], set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        obj = parse_line(line, n)
        row = to_batch_row(obj, n, drop_target=(mode != "score"))
        if mode == "score":
            msgs = row["body"]["messages"]
            if msgs[-1]["role"] != "assistant" or len(msgs) < 2:
                raise FormatError(n, "--score needs each row to end with an assistant "
                                     "message (the target) after at least one user message")
        cid = row["custom_id"]
        if cid in seen:
            raise FormatError(n, f'duplicate custom_id "{cid}"')
        seen.add(cid)
        rows.append(row)
    if not rows:
        raise SpillError(f"{path} contains no rows", "spill check <file>")
    return rows


def check_file(path: Path) -> dict:
    """Validate a whole file; raise FormatError at the first problem. Returns a summary."""
    p = Path(path)
    if not p.exists():
        raise SpillError(f"{path} does not exist")
    n_rows, kinds, expected, targets, seen = 0, set(), 0, 0, set()
    for n, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        obj = parse_line(line, n)
        kinds.add(line_kind(obj))
        cid = obj.get("custom_id")
        if cid is not None:
            if cid in seen:
                raise FormatError(n, f'duplicate custom_id "{cid}"')
            seen.add(cid)
        msgs = obj["body"]["messages"] if "body" in obj else obj["messages"]
        if msgs[-1]["role"] == "assistant":
            targets += 1
        expected += "expected" in obj
        n_rows += 1
    if not n_rows:
        raise SpillError(f"{path} contains no rows")
    if len(kinds) > 1:
        raise FormatError(1, "mixes batch lines and chat lines; use one shape per file")
    kind = kinds.pop()
    if 0 < expected < n_rows:
        raise SpillError(f'only {expected} of {n_rows} rows have "expected"; an eval file needs '
                         f'it on every row (or none)')
    if 0 < targets < n_rows:
        raise SpillError(f"only {targets} of {n_rows} rows end with an assistant message; "
                         f"training/--score files need it on every row")
    use = ("eval" if expected else "train/distill targets" if targets else "prompts")
    return {"rows": n_rows, "shape": kind, "use": use,
            "expected": expected, "targets": targets}


# ------------------------------------------------------------ tokenization for --score

def _ids(tokenizer, messages, **kw) -> list[int]:
    out = tokenizer.apply_chat_template(messages, **kw)
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


def tokenize_scored(tokenizer, messages: list[dict], eos_ids=()) -> tuple[int, list[int]]:
    """(n_prompt, full_ids): the prompt is everything before the final assistant
    message (with the generation prompt), the target region is that message plus
    its end-of-turn token. Tokens a template appends after the end-of-turn token
    (a trailing newline) are not scored."""
    if messages[-1]["role"] != "assistant":
        raise SpillError("scoring needs the last message to be the assistant target")
    prompt = _ids(tokenizer, messages[:-1], add_generation_prompt=True)
    full = _ids(tokenizer, messages)
    if full[:len(prompt)] != prompt:
        raise SpillError("the chat template is not prefix-stable for this row (the prompt "
                         "tokens differ when a target is appended), so target positions "
                         "cannot be located")
    if eos_ids:
        last = max((i for i in range(len(prompt), len(full)) if full[i] in eos_ids),
                   default=None)
        if last is not None:
            full = full[:last + 1]
    if len(full) <= len(prompt):
        raise SpillError("the assistant target tokenizes to nothing")
    return len(prompt), full
