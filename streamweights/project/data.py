"""Reading, validating and splitting the user's labeled examples.

  read_table      CSV (quoted, multiline) or JSONL -> raw records with file and row numbers
  build_rows      column mapping -> rows {id, input, output, group, source}, every problem
                  reported with the file, the row, what is wrong and the fix
  suggest_task    classification | json | ambiguous (never by distinct-label count alone)
  split_rows      supplied or generated train / val / test; duplicates and shared groups are
                  connected constraints (transitively); generated splits keep each connected
                  component together; supplied splits that overlap are rejected, not reshuffled
  integrity       the leakage and conflict report
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import SpillError
from .common import norm_dup, norm_group, sha_bytes

MAX_LABEL_CHARS = 100
MAX_ISSUES_SHOWN = 8
MIN_TRAIN = 8
MIN_VAL = 4


@dataclass
class Issue:
    file: str
    where: str            # "row 12", "line 7", "header"
    problem: str
    fix: str

    def line(self) -> str:
        return f"{self.file} {self.where}: {self.problem}. Try: {self.fix}"


class DataErrors(SpillError):
    """Every problem found, each its own line `<file> <row>: <problem>. Try: <fix>`."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        first = issues[0]
        super().__init__(f"{first.file} {first.where}: {first.problem}", first.fix)

    def lines(self) -> list[str]:
        out = [i.line() for i in self.issues[:MAX_ISSUES_SHOWN]]
        if len(self.issues) > MAX_ISSUES_SHOWN:
            out.append(f"... and {len(self.issues) - MAX_ISSUES_SHOWN} more problems; fix these "
                       f"and run again")
        return out


# ------------------------------------------------------------ reading

@dataclass
class Record:
    file: str
    row: int              # 1-based data row (CSV: record number; JSONL: line number)
    values: dict


def read_table(path: str | Path) -> tuple[list[str], list[Record], list[Issue]]:
    """(columns, records, issues). CSV is parsed with the csv module (quotes, embedded commas
    and newlines); JSONL one object per line. A UTF-8 byte-order mark is accepted."""
    p = Path(path)
    name = p.name
    if not p.exists():
        raise SpillError(f"{path} does not exist", "check the path")
    raw = p.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise DataErrors([Issue(name, f"byte {e.start}", "not valid UTF-8",
                                "re-save the file as UTF-8")])
    if not text.strip():
        raise DataErrors([Issue(name, "file", "the file is empty", "add labeled rows")])
    issues: list[Issue] = []
    suffix = p.suffix.lower()
    if suffix in (".jsonl", ".ndjson", ".json"):
        recs, cols = [], []
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(Issue(name, f"line {n}", f"not valid JSON ({e.msg} at column "
                                    f"{e.colno})", "fix the line or remove it"))
                continue
            if not isinstance(obj, dict):
                issues.append(Issue(name, f"line {n}", "each line must be a JSON object",
                                    'write one {"column": value} object per line'))
                continue
            for k in obj:
                if k not in cols:
                    cols.append(k)
            recs.append(Record(name, n, obj))
        return cols, recs, issues
    if suffix not in (".csv", ".tsv", ".txt"):
        issues.append(Issue(name, "file", f"unknown extension {suffix!r}",
                            "use a .csv, .tsv or .jsonl file"))
    delim = "\t" if suffix == ".tsv" else ","
    rdr = csv.reader(io.StringIO(text, newline=""), delimiter=delim)
    try:
        header = next(rdr)
    except StopIteration:
        raise DataErrors([Issue(name, "header", "no header row", "add a header row of column names")])
    header = [h.strip() for h in header]
    dup = sorted({h for h in header if header.count(h) > 1})
    if dup:
        issues.append(Issue(name, "header", f"duplicate column name(s) {', '.join(dup)}",
                            "rename the columns so each name is unique"))
    recs = []
    try:
        for n, row in enumerate(rdr, 1):
            if not row or all(not c.strip() for c in row):
                continue
            if len(row) != len(header):
                issues.append(Issue(name, f"row {n} (line {rdr.line_num})",
                                    f"has {len(row)} fields but the header has {len(header)}",
                                    "quote fields that contain commas or newlines, or fix the row"))
                continue
            recs.append(Record(name, n, dict(zip(header, row))))
    except csv.Error as e:
        issues.append(Issue(name, f"line {rdr.line_num}", f"CSV syntax error ({e})",
                            "close every quote or escape it by doubling it"))
    return header, recs, issues


# ------------------------------------------------------------ rows

@dataclass
class Row:
    id: str
    input: str
    output: object        # classification: the label text; json: the parsed object
    group: str | None
    source: dict

    def to_dict(self) -> dict:
        d = {"id": self.id, "input": self.input, "output": self.output, "source": self.source}
        if self.group is not None:
            d["group"] = self.group
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Row":
        return cls(d["id"], d["input"], d["output"], d.get("group"), d.get("source", {}))


def row_id(inp: str, out_text: str, seen: dict) -> str:
    """Stable id from content: the same example has the same id in any order and any file.
    Identical (input, output) pairs get a counter so ids stay unique."""
    base = sha_bytes((inp + "\x00" + out_text).encode())[:12]
    n = seen.get(base, 0)
    seen[base] = n + 1
    return f"r-{base}" + (f"-{n}" if n else "")


def _cell_text(v) -> str:
    if v is None:
        return ""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def build_rows(cols: list[str], recs: list[Record], mapping: dict, task: str | None,
               file: str) -> tuple[list[Row], list[Issue]]:
    """Map columns to rows and validate each. `task` None skips task-specific checks."""
    issues: list[Issue] = []
    for role in ("input", "output", "group"):
        c = mapping.get(role)
        if c and c not in cols:
            issues.append(Issue(file, "header", f"no column named {c!r} for --{role}",
                                f"columns found: {', '.join(cols) or '(none)'}; pass --{role} "
                                f"<one of them>"))
    if issues:
        return [], issues
    rows, seen = [], {}
    for r in recs:
        inp = _cell_text(r.values.get(mapping["input"]))
        outv = r.values.get(mapping["output"])
        if not inp.strip():
            issues.append(Issue(r.file, f"row {r.row}", f"empty input in column {mapping['input']!r}",
                                "fill it in or delete the row"))
            continue
        if outv is None or (isinstance(outv, str) and not outv.strip()):
            issues.append(Issue(r.file, f"row {r.row}", f"empty output in column "
                                f"{mapping['output']!r}; the guided path needs labeled examples",
                                "add the label or delete the row"))
            continue
        group = None
        if mapping.get("group"):
            g = _cell_text(r.values.get(mapping["group"])).strip()
            group = g or None
        output = outv
        if task == "json":
            if isinstance(outv, str):
                try:
                    output = json.loads(outv)
                except json.JSONDecodeError as e:
                    issues.append(Issue(r.file, f"row {r.row}", f"output is not valid JSON "
                                        f"({e.msg} at column {e.colno})",
                                        "fix the JSON or delete the row"))
                    continue
            if not isinstance(output, dict):
                issues.append(Issue(r.file, f"row {r.row}", "output must be a JSON object "
                                    "(a record of fields)", "wrap it in {...} or use --task "
                                    "classification"))
                continue
            out_text = json.dumps(output, sort_keys=True, ensure_ascii=False)
        else:
            if not isinstance(outv, str):
                outv = _cell_text(outv)
            output = outv.strip()
            if task == "classification" and len(output) > MAX_LABEL_CHARS:
                issues.append(Issue(r.file, f"row {r.row}", f"label is {len(output)} characters "
                                    f"long; classification labels are short", "use --task json "
                                    "for structured outputs, or shorten the label"))
                continue
            out_text = output
        rows.append(Row(row_id(inp, out_text, seen), inp, output, group,
                        {"file": r.file, "row": r.row}))
    return rows, issues


# ------------------------------------------------------------ task suggestion

@dataclass
class Suggestion:
    task: str | None
    reason: str


def suggest_task(rows_raw: list[Record], mapping: dict) -> Suggestion:
    """Look at the outputs, not only at how many distinct values there are. JSON objects mean
    json. Short, repeated, phrase-free values mean classification. Anything else is ambiguous
    and the caller must pass --task."""
    outs = [r.values.get(mapping["output"]) for r in rows_raw]
    outs = [o for o in outs if o is not None and str(o).strip() != ""]
    if not outs:
        return Suggestion(None, "there are no outputs to look at")

    def is_obj(o):
        if isinstance(o, dict):
            return True
        if isinstance(o, str) and o.strip().startswith("{"):
            try:
                return isinstance(json.loads(o), dict)
            except json.JSONDecodeError:
                return False
        return False

    n_obj = sum(is_obj(o) for o in outs)
    if n_obj == len(outs):
        return Suggestion("json", f"all {len(outs)} outputs are JSON objects")
    if n_obj:
        return Suggestion(None, f"{n_obj} of {len(outs)} outputs are JSON objects and the rest "
                                f"are not")
    texts = [str(o).strip() for o in outs]
    distinct = {t.casefold() for t in texts}
    short = all(len(t) <= 40 and len(t.split()) <= 4 for t in texts)
    sentence = any(re.search(r"[.!?]\s|[.!?]$", t) and len(t.split()) > 3 for t in texts)
    counts: dict = {}
    for t in texts:
        counts[t.casefold()] = counts.get(t.casefold(), 0) + 1
    repeated = sum(c for c in counts.values() if c >= 2) / len(texts)
    if short and not sentence and repeated >= 0.8 and len(distinct) >= 2:
        return Suggestion("classification",
                          f"{len(distinct)} distinct short labels over {len(texts)} rows, "
                          f"{repeated:.0%} of rows share their label with another row")
    return Suggestion(None, f"the outputs look like free text or have too few repeated values "
                            f"({len(distinct)} distinct in {len(texts)} rows)")


# ------------------------------------------------------------ components and splits

class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def dup_key(row: Row) -> str:
    return norm_dup(row.input)


def components(rows: list[Row]) -> list[list[int]]:
    """Connected components of rows linked by an identical normalized input or a shared group
    value, transitively. Returns lists of row indexes, ordered by first row."""
    uf = _UF(len(rows))
    first_input: dict = {}
    first_group: dict = {}
    for i, r in enumerate(rows):
        k = dup_key(r)
        if k in first_input:
            uf.union(i, first_input[k])
        else:
            first_input[k] = i
        if r.group is not None:
            g = norm_group(r.group)
            if g in first_group:
                uf.union(i, first_group[g])
            else:
                first_group[g] = i
    comps: dict = {}
    for i in range(len(rows)):
        comps.setdefault(uf.find(i), []).append(i)
    return [comps[k] for k in sorted(comps)]


@dataclass
class Splits:
    train: list[Row]
    val: list[Row]
    test: list[Row]
    mode: str                                   # generated | supplied | mixed
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def generate_split(rows: list[Row], seed: int, val_n: int | None, test_n: int | None,
                   val_fraction: float, test_fraction: float) -> tuple[list[Row], list[Row], list[Row]]:
    """Whole connected components go to one split. Components are shuffled with the seed and
    dealt to test, then validation, until each holds at least its target count; the rest is
    train. Targets are never met by splitting a component, and no row is invented."""
    comps = components(rows)
    rng = random.Random(seed)
    order = list(range(len(comps)))
    rng.shuffle(order)
    n = len(rows)
    t_target = test_n if test_n is not None else round(n * test_fraction)
    v_target = val_n if val_n is not None else round(n * val_fraction)
    test, val, train = [], [], []
    for ci in order:
        members = [rows[i] for i in comps[ci]]
        if len(test) < t_target:
            test += members
        elif len(val) < v_target:
            val += members
        else:
            train += members
    return train, val, test


def split_rows(train_rows: list[Row], val_rows: list[Row] | None, test_rows: list[Row] | None,
               seed: int, val_fraction: float, test_fraction: float) -> Splits:
    """Supplied held-out files are preserved. With one supplied, the other is derived from
    the remaining source rows; with none, both are. Supplied splits are checked for
    duplicate-input and group overlap against everything else and rejected with diagnostics
    rather than reshuffled."""
    supplied = {"val": val_rows, "test": test_rows}
    notes, warns = [], []
    held = [r for r in (val_rows or []) + (test_rows or [])]
    if held:
        _reject_overlap(train_rows, val_rows, test_rows)
    if val_rows is not None and test_rows is not None:
        return Splits(train_rows, val_rows, test_rows, "supplied",
                      ["validation and test files were supplied and kept as given"])
    if val_rows is None and test_rows is None:
        tr, va, te = generate_split(train_rows, seed, None, None, val_fraction, test_fraction)
        return Splits(tr, va, te, "generated",
                      [f"train, validation and test were generated with seed {seed}; duplicate "
                       f"inputs and shared groups stay together"])
    # exactly one held-out file supplied: derive the other from the remaining source rows
    keep = val_rows if val_rows is not None else test_rows
    need_frac = test_fraction if val_rows is not None else val_fraction
    total = len(train_rows) + len(keep)
    want = round(total * need_frac)
    tr, derived, _ = _derive_one(train_rows, seed, want)
    if val_rows is not None:
        notes.append(f"validation file kept ({len(val_rows)} rows); test derived from the source "
                     f"rows with seed {seed}")
        return Splits(tr, val_rows, derived, "mixed", notes, warns)
    notes.append(f"test file kept ({len(test_rows)} rows); validation derived from the source "
                 f"rows with seed {seed}")
    return Splits(tr, derived, test_rows, "mixed", notes, warns)


def _derive_one(rows: list[Row], seed: int, want: int):
    comps = components(rows)
    rng = random.Random(seed)
    order = list(range(len(comps)))
    rng.shuffle(order)
    taken, rest = [], []
    for ci in order:
        members = [rows[i] for i in comps[ci]]
        (taken if len(taken) < want else rest).extend(members)
    return rest, taken, []


def _reject_overlap(train: list[Row], val: list[Row] | None, test: list[Row] | None) -> None:
    """Supplied splits must not share an (normalized) input or a group value with each other
    or with the training rows."""
    parts = [("train", train)] + [(n, r) for n, r in (("val", val), ("test", test))
                                  if r is not None]
    issues: list[Issue] = []
    seen_in: dict = {}
    seen_gr: dict = {}
    for name, rows in parts:
        local_in, local_gr = {}, {}
        for r in rows:
            k = dup_key(r)
            local_in.setdefault(k, r)
            if r.group is not None:
                local_gr.setdefault(norm_group(r.group), r)
        for k, r in local_in.items():
            if k in seen_in and seen_in[k][0] != name:
                o_name, o = seen_in[k]
                issues.append(Issue(r.source.get("file", name), f"row {r.source.get('row')}",
                                    f"the input also appears in the {o_name} split "
                                    f"({o.source.get('file')} row {o.source.get('row')}): "
                                    f"{_short(r.input)!r}",
                                    "remove the duplicate from one file; supplied splits are "
                                    "not reshuffled"))
            else:
                seen_in.setdefault(k, (name, r))
        for g, r in local_gr.items():
            if g in seen_gr and seen_gr[g][0] != name:
                o_name, o = seen_gr[g]
                issues.append(Issue(r.source.get("file", name), f"row {r.source.get('row')}",
                                    f"group {r.group!r} also appears in the {o_name} split "
                                    f"({o.source.get('file')} row {o.source.get('row')})",
                                    "move every row of the group into one file; supplied "
                                    "splits are not reshuffled"))
            else:
                seen_gr.setdefault(g, (name, r))
    if issues:
        raise DataErrors(issues)


def _short(t: str, n: int = 60) -> str:
    t = " ".join(t.split())
    return t if len(t) <= n else t[:n - 3] + "..."


# ------------------------------------------------------------ integrity

def conflicting_labels(rows: list[Row]) -> list[dict]:
    """Identical normalized inputs carrying different outputs, anywhere in the data."""
    by: dict = {}
    for r in rows:
        key = dup_key(r)
        out = r.output if isinstance(r.output, str) else json.dumps(r.output, sort_keys=True)
        out = norm_dup(out) if isinstance(r.output, str) else out
        by.setdefault(key, {}).setdefault(out, []).append(r)
    res = []
    for key, outs in by.items():
        if len(outs) > 1:
            rs = [x for v in outs.values() for x in v]
            res.append({"input": _short(rs[0].input), "labels": sorted(outs),
                        "rows": [f"{x.source.get('file')} row {x.source.get('row')}" for x in rs]})
    return res


def class_coverage(splits: Splits) -> dict:
    def counts(rows):
        c: dict = {}
        for r in rows:
            c[norm_dup(str(r.output))] = c.get(norm_dup(str(r.output)), 0) + 1
        return c
    tr, va, te = counts(splits.train), counts(splits.val), counts(splits.test)
    all_labels = sorted(set(tr) | set(va) | set(te))
    return {"classes": len(all_labels), "train": len(tr), "val": len(va), "test": len(te),
            "missing_in_train": sorted((set(va) | set(te)) - set(tr)),
            "missing_in_val": sorted(set(tr) - set(va)),
            "missing_in_test": sorted(set(tr) - set(te))}


def split_sha(rows: list[Row]) -> str:
    return sha_bytes("".join(json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
                             for r in rows).encode())


def check_usable(splits: Splits, task: str | None = None) -> list[str]:
    """Reasons this split cannot be built on; empty when usable."""
    bad = []
    if task == "classification" and len({norm_dup(str(r.output)) for r in splits.train}) < 2:
        bad.append("the training set has fewer than two classes")
    if len(splits.train) < MIN_TRAIN:
        bad.append(f"the training set has {len(splits.train)} rows; at least {MIN_TRAIN} are "
                   f"needed")
    if len(splits.val) < MIN_VAL:
        bad.append(f"the validation set has {len(splits.val)} rows; at least {MIN_VAL} are "
                   f"needed to compare anything")
    return bad
