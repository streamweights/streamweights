"""Portable state for row jobs (run, distill, eval): results and the cursor of completed rows.

The local job directory keeps its existing files (results.jsonl, checkpoint: one completed row
id per line). To make that portable, completed rows are pushed to the state URI as immutable
segments

  rows/seg-<n>.jsonl     result lines, in completion order
  rows/seg-<n>.commit    written last: row ids, count, sha256, and the engine, hardware and
                         numerics that produced the segment

and a machine that starts the same job restores every committed segment into its local
job directory, so it continues from the cursor. A segment without a commit marker is ignored
and overwritten. The quantum of a row job is one completed row.
"""

from __future__ import annotations

import hashlib
import json
import re
import time

SCHEMA = 1
_SEG = re.compile(r"^seg-(\d{6})\.commit$")


def _seg_name(n: int) -> str:
    return f"rows/seg-{n:06d}"


def committed_segments(store) -> list[dict]:
    out = []
    for name in store.ls("rows"):
        m = _SEG.match(name)
        if not m:
            continue
        n = int(m.group(1))
        try:
            commit = json.loads(store.read(f"rows/{name}"))
            if store.size(f"{_seg_name(n)}.jsonl") != commit["bytes"]:
                continue
        except (OSError, ValueError, KeyError):
            continue
        commit["n"] = n
        out.append(commit)
    return sorted(out, key=lambda c: c["n"])


def done_ids(store) -> set[str]:
    ids: set[str] = set()
    for seg in committed_segments(store):
        ids.update(seg["ids"])
    return ids


def restore(store, results_path, checkpoint_path) -> tuple[int, list[dict]]:
    """Rebuild the local results.jsonl and cursor file from the committed segments.
    Returns (rows restored, the segments' producer records)."""
    segs = committed_segments(store)
    lines, ids, producers = [], [], []
    seen: set[str] = set()
    for seg in segs:
        data = store.read(f"{_seg_name(seg['n'])}.jsonl")
        if hashlib.sha256(data).hexdigest() != seg["sha256"]:
            break                                     # corrupt: continue from the last intact one
        for line, cid in zip(data.decode().splitlines(), seg["ids"]):
            if cid in seen:                           # a duplicate across segments is dropped
                continue
            seen.add(cid)
            lines.append(line)
            ids.append(cid)
        producers.append({k: seg[k] for k in ("engine", "hardware", "numerics", "rows", "at",
                                              "host", "system", "os") if k in seg})
    results_path.write_text("".join(l + "\n" for l in lines))
    checkpoint_path.write_text("".join(i + "\n" for i in ids))
    return len(ids), producers


class RowSync:
    """Pushes a job's newly completed rows to the state store, every N rows or seconds."""

    def __init__(self, store, results_path, checkpoint_path, producer: dict,
                 every_rows: int = 25, every_s: float = 60.0):
        self.store = store
        self.results_path, self.checkpoint_path = results_path, checkpoint_path
        self.producer = producer
        self.every_rows, self.every_s = every_rows, every_s
        self.pushed = 0              # rows already in the store (the local file is a superset)
        self._last = time.monotonic()
        self._next = 0

    def attach(self) -> int:
        """Restore, and start numbering segments after the existing ones."""
        segs = committed_segments(self.store)
        self._next = (segs[-1]["n"] + 1) if segs else 0
        if not segs:                  # nothing in the store: the local job is the whole truth
            self.pushed = 0
            return 0
        local = []
        if self.results_path.exists() and self.checkpoint_path.exists():
            local = list(zip(self.checkpoint_path.read_text().split(),
                             self.results_path.read_text().splitlines()))
        n, _ = restore(self.store, self.results_path, self.checkpoint_path)
        have = set(self.checkpoint_path.read_text().split())
        extra = [(i, l) for i, l in local if i not in have]     # finished here, never pushed
        if extra:
            with open(self.results_path, "a") as f, open(self.checkpoint_path, "a") as c:
                for cid, line in extra:
                    f.write(line + "\n")
                    c.write(cid + "\n")
        self.pushed = n
        return n

    def maybe_push(self, done: int) -> bool:
        if done - self.pushed >= self.every_rows or (
                done > self.pushed and time.monotonic() - self._last >= self.every_s):
            self.push()
            return True
        return False

    def push(self) -> int:
        lines = self.results_path.read_text().splitlines() if self.results_path.exists() else []
        ids = self.checkpoint_path.read_text().split() if self.checkpoint_path.exists() else []
        n = min(len(lines), len(ids))
        if n <= self.pushed:
            return 0
        body = "".join(l + "\n" for l in lines[self.pushed:n]).encode()
        seg_ids = ids[self.pushed:n]
        name = _seg_name(self._next)
        self.store.write(f"{name}.jsonl", body)
        self.store.write(f"{name}.commit", json.dumps({
            "schema": SCHEMA, "ids": seg_ids, "rows": len(seg_ids), "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "engine": self.producer["engine"], "hardware": self.producer["hardware"],
            "numerics": self.producer["numerics"],
            **{k: self.producer[k] for k in ("host", "system", "os") if k in self.producer},
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}).encode())
        pushed_now = n - self.pushed
        self.pushed = n
        self._next += 1
        self._last = time.monotonic()
        return pushed_now
