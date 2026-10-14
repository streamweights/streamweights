"""Training data for `spill tune`: OpenAI fine-tuning chat JSONL to masked token batches.

One line is {"messages": [...]}, each assistant message optionally carrying
"weight": 0 or 1 (OpenAI's convention; 0 keeps the turn as context but trains
nothing on it, absent means 1). The model's own chat template renders the
conversation; loss is taken on assistant tokens only: the reply and its
end-of-turn token, never the prompt, the role headers, or tokens a template
appends after the end-of-turn token (a trailing newline).

Sequence cap (--max-seq): a conversation over the cap loses whole messages from
the left (the earliest user/assistant exchange first; system messages are kept).
An assistant turn is never cut in half. An example whose last exchange alone is
over the cap is skipped and counted, never silently shortened.

Batching is deterministic from (seed, micro-batch index): examples are sorted by
length into micro-batches (little padding), batch order is shuffled per epoch,
and the tail batch is kept. That makes resume exact and lets the resident and
streamed trainers consume identical batches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import SpillError
from ..formats import FormatError, parse_line

PAD_MULTIPLE = 32


@dataclass
class Example:
    line_no: int
    ids: list[int]
    mask: list[int]            # mask[i] == 1: token ids[i] is a loss target
    dropped_messages: int = 0
    kd_start: int = 0          # teacher top-k covers ids[kd_start : kd_start + len(kd_ids)]
    kd_ids: np.ndarray | None = None      # int32 [n, K]: the teacher's top-k token ids
    kd_lp: np.ndarray | None = None       # float32 [n, K]: their log-probs (descending)

    @property
    def n_trained(self) -> int:
        return sum(self.mask)


@dataclass
class DataStats:
    lines: int = 0
    examples: int = 0
    tokens: int = 0                 # total tokens across kept examples
    trained_tokens: int = 0         # loss-bearing tokens
    truncated: int = 0              # examples that lost leading messages
    dropped_messages: int = 0
    skipped: list[tuple[int, str]] = field(default_factory=list)   # (line, reason)
    max_len: int = 0
    teacher_examples: int = 0       # examples carrying the teacher's top-k log-probs
    kd_k: int = 0                   # the top-k they were cut to (the smallest K in the file)


# ------------------------------------------------------------ validation

def validate_train_line(obj: dict, line_no: int) -> list[dict]:
    """Extra checks beyond the shared chat-line validation; returns the messages."""
    if "body" in obj:
        raise FormatError(line_no, 'training files use chat lines ({"messages": [...]}), '
                                   'not batch lines')
    msgs = obj["messages"]
    if msgs[-1]["role"] != "assistant":
        raise FormatError(line_no, "the last message must be an assistant message (the "
                                   "training target)")
    first = next((m for m in msgs if m["role"] != "system"), None)
    if first is None or first["role"] != "user":
        raise FormatError(line_no, "the first non-system message must be a user message")
    for i, m in enumerate(msgs):
        if "weight" in m and m["role"] != "assistant":
            raise FormatError(line_no, f"messages[{i}].weight is only meaningful on "
                                       f"assistant messages")
    if not any(m["role"] == "assistant" and m.get("weight", 1) != 0 for m in msgs):
        raise FormatError(line_no, "every assistant message has weight 0, so there is "
                                   "nothing to train on")
    t = obj.get("teacher")
    if t is not None:
        n = len(t.get("token_ids", []))
        if not n or len(t.get("top_ids", [])) != n or len(t.get("top_lps", [])) != n:
            raise FormatError(line_no, '"teacher" needs token_ids, top_ids and top_lps of the '
                                       'same length (one entry per teacher token)')
        if len({len(r) for r in t["top_ids"]} | {len(r) for r in t["top_lps"]}) != 1:
            raise FormatError(line_no, '"teacher" top-k lists must all have the same K')
    return msgs


# ------------------------------------------------------------ tokenization

def _ids(tokenizer, messages, **kw) -> list[int]:
    out = tokenizer.apply_chat_template(messages, **kw)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


def _clean(messages: list[dict]) -> list[dict]:
    """Template input: role and content only (weight is ours, not the template's)."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def tokenize_conversation(tokenizer, messages: list[dict], eos_ids=(),
                          line_no: int = 0) -> tuple[list[int], list[int]]:
    """(ids, mask) for one conversation. Assistant spans are located by rendering
    the template at each assistant boundary, which requires the template to be
    prefix-stable (true of ChatML, Llama 3, Gemma, Mistral-style templates)."""
    msgs = _clean(messages)
    full = _ids(tokenizer, msgs)
    mask = [0] * len(full)
    for i, m in enumerate(messages):
        if m["role"] != "assistant" or m.get("weight", 1) == 0:
            continue
        prompt = _ids(tokenizer, msgs[:i], add_generation_prompt=True)
        upto = _ids(tokenizer, msgs[:i + 1])
        if upto[:len(prompt)] != prompt or full[:len(upto)] != upto:
            raise FormatError(line_no, "the chat template is not prefix-stable for this "
                                       "conversation, so assistant tokens cannot be located")
        end = len(upto)
        if eos_ids:
            last = max((j for j in range(len(prompt), len(upto)) if upto[j] in eos_ids),
                       default=None)
            if last is not None:
                end = last + 1
        if end <= len(prompt):
            raise FormatError(line_no, f"messages[{i}] tokenizes to nothing")
        for j in range(len(prompt), end):
            mask[j] = 1
    return full, mask


def _drop_leftmost(messages: list[dict]) -> list[dict] | None:
    """Drop the earliest exchange: everything from the first non-system message
    through the first assistant message and any tool replies after it, so what is
    left starts at a user message. None if no complete exchange can go."""
    s = 0
    while s < len(messages) and messages[s]["role"] == "system":
        s += 1
    a = next((i for i in range(s, len(messages)) if messages[i]["role"] == "assistant"), None)
    if a is None:
        return None
    j = a + 1
    while j < len(messages) and messages[j]["role"] != "user":
        j += 1
    if j >= len(messages):
        return None          # nothing would remain after the dropped exchange
    rest = messages[:s] + messages[j:]
    if not any(m["role"] == "assistant" and m.get("weight", 1) != 0 for m in rest):
        return None          # the dropped exchange held the only trainable turn
    return rest


def build_example(tokenizer, messages: list[dict], max_seq: int, eos_ids=(),
                  line_no: int = 0) -> tuple[Example | None, str | None]:
    """(example, None) or (None, reason it was skipped)."""
    dropped = 0
    cur = messages
    while True:
        ids, mask = tokenize_conversation(tokenizer, cur, eos_ids, line_no)
        if len(ids) <= max_seq:
            break
        nxt = _drop_leftmost(cur)
        if nxt is None:
            return None, (f"{len(ids)} tokens after dropping all earlier turns, over "
                          f"--max-seq {max_seq} (an assistant turn is never cut)")
        dropped += len(cur) - len(nxt)
        cur = nxt
    if not any(mask):
        return None, "no assistant tokens left to train on"
    return Example(line_no, ids, mask, dropped), None


def build_teacher_example(tokenizer, messages: list[dict], teacher: dict, max_seq: int,
                          eos_ids=(), line_no: int = 0) -> tuple[Example | None, str | None]:
    """An example whose assistant tokens are the teacher's own (not re-tokenized from text, so
    each trained token lines up with the teacher's top-k at that position). The prompt is the
    conversation before the final assistant message, rendered with the generation prompt. If
    the teacher's tokens do not end in an end-of-turn token, one is added and trained with
    plain cross-entropy (the teacher said nothing about it)."""
    msgs = _clean(messages)
    prompt = _ids(tokenizer, msgs[:-1], add_generation_prompt=True)
    toks = [int(t) for t in teacher["token_ids"]]
    top_ids = np.asarray(teacher["top_ids"], np.int32)
    top_lps = np.asarray(teacher["top_lps"], np.float32)
    ids = prompt + toks
    mask = [0] * len(prompt) + [1] * len(toks)
    if eos_ids and toks[-1] not in eos_ids:
        full = _ids(tokenizer, msgs)
        end = next((t for t in reversed(full) if t in eos_ids), min(eos_ids))
        ids.append(end)
        mask.append(1)
    if len(ids) > max_seq:
        return None, (f"{len(ids)} tokens with the teacher's answer, over --max-seq {max_seq} "
                      f"(a teacher answer is never cut)")
    return Example(line_no, ids, mask, 0, len(prompt), top_ids, top_lps), None


def load_examples(path: Path, tokenizer, max_seq: int, eos_ids=()) -> tuple[list[Example], DataStats]:
    st = DataStats()
    exs: list[Example] = []
    p = Path(path)
    if not p.exists():
        raise SpillError(f"{path} does not exist")
    for n, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        st.lines += 1
        obj = parse_line(line, n)
        msgs = validate_train_line(obj, n)
        if obj.get("teacher") is not None:
            ex, why = build_teacher_example(tokenizer, msgs, obj["teacher"], max_seq, eos_ids, n)
        else:
            ex, why = build_example(tokenizer, msgs, max_seq, eos_ids, n)
        if ex is None:
            st.skipped.append((n, why))
            continue
        exs.append(ex)
        st.examples += 1
        st.tokens += len(ex.ids)
        st.trained_tokens += ex.n_trained
        st.max_len = max(st.max_len, len(ex.ids))
        if ex.dropped_messages:
            st.truncated += 1
            st.dropped_messages += ex.dropped_messages
    if not exs:
        raise SpillError(f"{path}: no usable training examples"
                         + (f" (line {st.skipped[0][0]}: {st.skipped[0][1]})" if st.skipped else ""))
    kd = [e for e in exs if e.kd_ids is not None]
    if kd:
        k = min(e.kd_ids.shape[1] for e in kd)
        for e in kd:
            e.kd_ids, e.kd_lp = e.kd_ids[:, :k], e.kd_lp[:, :k]
        st.teacher_examples, st.kd_k = len(kd), k
    return exs, st


# ------------------------------------------------------------ batching

@dataclass
class Batch:
    inputs: np.ndarray       # int32 [B, T]
    targets: np.ndarray      # int32 [B, T]
    mask: np.ndarray         # bool  [B, T], True where the target token is trained
    n_tokens: int            # real (unpadded) tokens in the batch
    n_trained: int
    kd: tuple | None = None  # (ids int32 [B,T,K], logprobs float32 [B,T,K], has-teacher bool [B,T])


class BatchPlan:
    """Deterministic micro-batch schedule. batch(m) is a pure function of
    (examples, micro_batch, seed, m), so a resumed job continues exactly."""

    def __init__(self, examples: list[Example], micro_batch: int, seed: int,
                 pad_id: int = 0):
        if micro_batch < 1:
            raise SpillError("micro-batch must be at least 1")
        self.examples = examples
        self.mb = micro_batch
        self.seed = seed
        self.pad_id = pad_id
        order = sorted(range(len(examples)), key=lambda i: (len(examples[i].ids), i))
        self.groups = [order[i:i + micro_batch] for i in range(0, len(order), micro_batch)]
        self.per_epoch = len(self.groups)
        self.kd_k = min((e.kd_ids.shape[1] for e in examples if e.kd_ids is not None), default=0)

    def group_indices(self, m: int) -> list[int]:
        epoch, k = divmod(m, self.per_epoch)
        perm = np.random.RandomState(self.seed + epoch).permutation(self.per_epoch)
        return self.groups[int(perm[k])]

    def batch(self, m: int) -> Batch:
        idx = self.group_indices(m)
        exs = [self.examples[i] for i in idx]
        longest = max(len(e.ids) for e in exs)
        T = max(1, PAD_MULTIPLE * ((longest - 1 + PAD_MULTIPLE - 1) // PAD_MULTIPLE))
        B = len(exs)
        inp = np.full((B, T), self.pad_id, np.int32)
        tgt = np.full((B, T), self.pad_id, np.int32)
        msk = np.zeros((B, T), bool)
        for r, e in enumerate(exs):
            n = len(e.ids) - 1
            inp[r, :n] = e.ids[:-1]
            tgt[r, :n] = e.ids[1:]
            msk[r, :n] = np.array(e.mask[1:], bool)
        kd = None
        if self.kd_k:
            kd_i = np.zeros((B, T, self.kd_k), np.int32)
            kd_l = np.zeros((B, T, self.kd_k), np.float32)
            kd_m = np.zeros((B, T), bool)
            for r, e in enumerate(exs):
                if e.kd_ids is None:
                    continue
                m = len(e.kd_ids)
                # input position i predicts token i + 1; the teacher's j-th token is ids[kd_start + j]
                lo = e.kd_start - 1
                kd_i[r, lo:lo + m] = e.kd_ids
                kd_l[r, lo:lo + m] = e.kd_lp
                kd_m[r, lo:lo + m] = True
            kd = (kd_i, kd_l, kd_m)
        return Batch(inp, tgt, msk, sum(len(e.ids) for e in exs), int(msk.sum()), kd)

    def max_batch_tokens(self) -> int:
        """Padded length of the longest micro-batch (for the memory budget)."""
        longest = max(len(e.ids) for e in self.examples)
        return max(1, PAD_MULTIPLE * ((longest - 1 + PAD_MULTIPLE - 1) // PAD_MULTIPLE))


def steps_for(n_micro_per_epoch: int, grad_accum: int, epochs: float = 1.0) -> int:
    return max(1, int(n_micro_per_epoch * epochs) // max(1, grad_accum))


def is_distill_file(path: Path) -> bool:
    """A `spill distill` output: lines with the prompt `messages` and the teacher's `completion`."""
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    return isinstance(d, dict) and "completion" in d and "messages" in d
    except (OSError, ValueError):
        pass
    return False


def teacher_record(d: dict, tok_sha: str | None) -> dict | None:
    """The `teacher` field of a training line from a distill record that carries per-token top-k
    log-probs (generate mode), or None. The tokens are the teacher's own, so each trained token
    lines up with its top-k."""
    lps = d.get("logprobs")
    ids = d.get("completion_token_ids")
    if not lps or not ids or len(lps) != len(ids):
        return None
    return {"token_ids": [int(t) for t in ids],
            "top_ids": [[int(a) for a, _ in p["top"]] for p in lps],
            "top_lps": [[float(b) for _, b in p["top"]] for p in lps],
            "tokenizer_sha256": tok_sha}


def scan_teacher(path: Path) -> dict:
    """What a training file says about a teacher: whether lines carry top-k log-probs, from which
    tokenizers, and how many."""
    n = shas = 0
    toks, ks = set(), set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        n += 1
        t = d.get("teacher") if isinstance(d, dict) else None
        if isinstance(t, dict) and t.get("top_ids"):
            shas += 1
            toks.add(t.get("tokenizer_sha256"))
            ks.add(len(t["top_ids"][0]))
    return {"rows": n, "teacher_rows": shas, "tokenizers": toks, "k": min(ks) if ks else 0}


def distill_as_training(path: Path, dest: Path, student_tok_sha: str | None = None,
                        use_logprobs: bool = True) -> tuple[Path, int, dict]:
    """Training chat lines from a distill file: the prompt messages plus the teacher's
    completion as the assistant target. Records with an error or an empty completion are left
    out. When the records carry the teacher's top-k log-probs and its tokenizer is the student's
    (the same tokenizer.json), the line also carries a `teacher` field with the teacher's own
    tokens and their top-k, so `--distill-loss kl|mix` can train on the distribution. A teacher
    with another tokenizer contributes its text only. Returns (file, rows, info)."""
    rows, with_lp, mismatch = [], 0, 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        comp = d.get("completion")
        if d.get("error") or comp is None or not str(comp).strip():
            continue
        row = {"messages": list(d["messages"]) + [{"role": "assistant",
                                                   "content": str(comp).strip()}]}
        t_sha = (d.get("teacher") or {}).get("tokenizer_sha256")
        t = teacher_record(d, t_sha) if use_logprobs else None
        if t is not None:
            if student_tok_sha and t_sha == student_tok_sha:
                row["teacher"] = t
                with_lp += 1
            else:
                mismatch += 1
        rows.append(row)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return dest, len(rows), {"rows": len(rows), "teacher_rows": with_lp,
                             "tokenizer_mismatch_rows": mismatch}
