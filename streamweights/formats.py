"""File formats: one family of JSONL, three uses.

  batch   OpenAI batch line:  {"custom_id": "a", "body": {"messages": [...], "max_tokens": 64}}
  chat    OpenAI fine-tuning line: {"messages": [{"role": ..., "content": ..., "weight": 0|1}, ...]}
  eval    either of the above plus an "expected" field (any JSON value)
  plain   the short forms people actually write:
            evals     {"prompt": "...", "expected": "..."}
            prompts   {"prompt": "..."}
            training  {"prompt": "...", "answer": "..."}
          Any row of any shape may carry an optional "system" string.
          Shapes are auto-detected per file; plain rows are turned into chat rows
          (system, user, and an assistant message when "answer" is present).

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


def _plain_to_chat(obj: dict, line_no: int) -> dict:
    """Plain row -> chat row (messages), keeping `expected`, `custom_id`, and a marker."""
    unknown = sorted(set(obj) - {"prompt", "expected", "answer", "system", "custom_id", "max_tokens"})
    if unknown:
        raise FormatError(line_no, f"unknown field(s) {', '.join(repr(k) for k in unknown)} on a "
                                   f'plain row; plain rows use "prompt", "expected" or "answer", '
                                   f'and optional "system"')
    if not isinstance(obj["prompt"], str) or not obj["prompt"].strip():
        raise FormatError(line_no, '"prompt" must be a non-empty string')
    if "answer" in obj and "expected" in obj:
        raise FormatError(line_no, 'a row has "answer" (training) or "expected" (eval), not both')
    if "answer" in obj and not isinstance(obj["answer"], str):
        raise FormatError(line_no, '"answer" must be a string')
    if "system" in obj and not isinstance(obj["system"], str):
        raise FormatError(line_no, '"system" must be a string')
    msgs = []
    if obj.get("system"):
        msgs.append({"role": "system", "content": obj["system"]})
    msgs.append({"role": "user", "content": obj["prompt"]})
    if "answer" in obj:
        msgs.append({"role": "assistant", "content": obj["answer"]})
    out = {"messages": msgs, "_shape": "plain"}
    for k in ("expected", "custom_id", "max_tokens"):
        if k in obj:
            out[k] = obj[k]
    return out


def _apply_system_field(obj: dict, line_no: int) -> None:
    """An optional "system" field on batch/chat rows: prepended unless the row
    already starts with a system message."""
    sysm = obj.get("system")
    if sysm is None:
        return
    if not isinstance(sysm, str):
        raise FormatError(line_no, '"system" must be a string')
    msgs = obj["body"]["messages"] if "body" in obj else obj["messages"]
    if sysm and msgs[0]["role"] != "system":
        msgs.insert(0, {"role": "system", "content": sysm})


def parse_line(line: str, line_no: int) -> dict:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise FormatError(line_no, f"not valid JSON ({e.msg} at column {e.colno})")
    if not isinstance(obj, dict):
        raise FormatError(line_no, "each line must be a JSON object")
    if "prompt" in obj and "body" not in obj and "messages" not in obj:
        return _plain_to_chat(obj, line_no)
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
        raise FormatError(line_no, 'expected a plain row (with "prompt"), an OpenAI batch line '
                                   '(with "custom_id" and "body") or a chat line (with "messages")')
    _apply_system_field(obj, line_no)
    return obj


def is_distill_record(obj: dict) -> bool:
    """A line written by `spill distill`: the teacher's completion (or scored target) with
    the prompt messages."""
    return "teacher" in obj and ("completion" in obj or "target" in obj) and "messages" in obj


def distill_to_chat(obj: dict) -> dict | None:
    """Distillation record -> chat training row (prompt messages + the teacher's answer)."""
    ans = obj.get("completion", obj.get("target"))
    if ans is None or not str(ans).strip() or obj.get("error"):
        return None
    return {"messages": list(obj["messages"]) + [{"role": "assistant",
                                                  "content": str(ans).strip()}]}


def line_kind(obj: dict) -> str:
    return "batch" if "body" in obj else obj.get("_shape", "chat")


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
    distill_rows = 0
    ends: list[tuple[int, bool]] = []
    for n, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        obj = parse_line(line, n)
        if is_distill_record(obj):
            distill_rows += 1
            n_rows += 1
            continue
        kinds.add(line_kind(obj))
        cid = obj.get("custom_id")
        if cid is not None:
            if cid in seen:
                raise FormatError(n, f'duplicate custom_id "{cid}"')
            seen.add(cid)
        msgs = obj["body"]["messages"] if "body" in obj else obj["messages"]
        if msgs[-1]["role"] == "assistant":
            targets += 1
        ends.append((n, msgs[-1]["role"] == "assistant"))
        expected += "expected" in obj
        n_rows += 1
    if not n_rows:
        raise SpillError(f"{path} contains no rows")
    if distill_rows:
        if distill_rows != n_rows:
            raise SpillError(f"{distill_rows} of {n_rows} rows are distillation records and the "
                             f"rest are not; use one kind per file")
        return {"rows": n_rows, "shape": "distill", "use": "distillation records",
                "expected": 0, "targets": n_rows}
    if len(kinds) > 1:
        raise FormatError(1, f"mixes {' and '.join(sorted(kinds))} lines; use one shape per file")
    kind = kinds.pop()
    if 0 < expected < n_rows:
        raise SpillError(f'only {expected} of {n_rows} rows have "expected"; an eval file needs '
                         f'it on every row (or none)')
    if 0 < targets < n_rows:
        first = ends[0]
        n_diff, was = next((n, e) for n, e in ends if e != first[1])
        has, lacks = (first[0], n_diff) if first[1] else (n_diff, first[0])
        raise FormatError(lacks, f"does not end with an assistant message, but line {has} does "
                                 f"(only {targets} of {n_rows} rows have a target; training and "
                                 f"--score files need it on every row)")
    use = ("eval" if expected else "train/distill targets" if targets else "prompts")
    if use == "train/distill targets" and kind == "chat":
        # the same line-numbered errors `spill tune` would raise
        from .tune.data import validate_train_line
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if line.strip():
                validate_train_line(parse_line(line, n), n)
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


# ------------------------------------------------------------ chaining

def newest_training_file(folder: Path = Path(".")) -> Path:
    """What `spill tune` uses when given no data file: the folder's newest *.distill.jsonl
    or train.jsonl."""
    cands = [p for p in list(Path(folder).glob("*.distill.jsonl")) + [Path(folder) / "train.jsonl"]
             if p.exists()]
    if not cands:
        raise SpillError(f"no train.jsonl or *.distill.jsonl in {Path(folder).resolve()}",
                         "spill tune <model> <file> --name <name>")
    return max(cands, key=lambda p: p.stat().st_mtime)


def normalize_training_file(src: Path, dest: Path) -> int:
    """Write `src` (plain, chat, or distillation records) as chat JSONL a trainer reads.
    Returns the number of rows; rows with no usable answer are dropped."""
    n = 0
    lines = []
    for i, line in enumerate(Path(src).read_text().splitlines(), 1):
        if not line.strip():
            continue
        obj = parse_line(line, i)
        if is_distill_record(obj):
            row = distill_to_chat(obj)
            if row is None:
                continue
        else:
            msgs = obj["body"]["messages"] if "body" in obj else obj["messages"]
            if msgs[-1]["role"] != "assistant":
                raise FormatError(i, "training rows need an answer (an assistant message, or "
                                     '"answer" on a plain row)')
            row = {"messages": msgs}
        lines.append(json.dumps(row, ensure_ascii=False))
        n += 1
    if not n:
        raise SpillError(f"{src} has no usable training rows")
    Path(dest).write_text("\n".join(lines) + "\n")
    return n
