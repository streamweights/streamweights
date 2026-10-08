"""Model downloads: one aggregate progress line (bytes, speed, ETA), disk checked before the
first byte, resumed after an interruption, and the fastest transport available.

huggingface_hub already resumes partial files; what it does not do is tell you, once,
how long the whole thing will take, or refuse before it starts when the disk cannot hold
it. This wraps snapshot_download with both, plus retries that continue where they stopped.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from .errors import SpillError
from .registry import GIB, MIN_FREE_AFTER_DOWNLOAD, REPO_ROOT

RETRIES = 4


def transport() -> str:
    """The accelerator in use: hf_transfer (enabled here when installed), hf_xet (the
    default in current huggingface_hub), or plain HTTP."""
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        return "hf_transfer"
    except ImportError:
        pass
    try:
        import hf_xet  # noqa: F401
        return "hf_xet"
    except ImportError:
        return "plain HTTP"


def plan(files: list[tuple[str, int]], dest: Path, patterns: list[str]) -> tuple[int, int]:
    """(total bytes, bytes still to fetch) for the files matching `patterns`; partial
    `.incomplete` files and finished files already under dest count as fetched."""
    total = have = 0
    inc = _incomplete_bytes(dest)
    for name, size in files:
        if not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        total += size
        f = dest / name
        if f.exists() and f.stat().st_size == size:
            have += size
    have += min(inc, total - have)
    return total, total - have


def _incomplete_bytes(dest: Path) -> int:
    n = 0
    d = dest / ".cache" / "huggingface" / "download"
    if d.exists():
        for f in d.rglob("*.incomplete"):
            try:
                n += f.stat().st_size
            except OSError:
                pass
    return n


def _present_bytes(dest: Path, names: set[str]) -> int:
    n = _incomplete_bytes(dest)
    for name in names:
        f = dest / name
        if f.exists():
            n += f.stat().st_size
    return n


def check_disk(remaining: int, dest: Path, label: str, floor: int = MIN_FREE_AFTER_DOWNLOAD,
               recovery: str | None = None) -> None:
    probe = dest
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free - remaining < floor:
        need = (remaining + floor - free) / GIB
        raise SpillError(
            f"not enough disk for {label}: {remaining / GIB:.1f} GB to fetch and "
            f"{free / GIB:.0f} GB free, with a {floor / GIB:.0f} GB floor kept; free up "
            f"{need:.0f} GB (nothing was downloaded)", recovery)


def fmt_eta(s: float | None) -> str:
    if s is None or s != s or s == float("inf"):
        return "..."
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


class Progress:
    """Aggregate progress for a download directory, polled from the filesystem so it works
    with whatever transport is doing the fetching."""

    def __init__(self, dest: Path, names: set[str], total: int, start: int, label: str,
                 out=sys.stderr, interval: float = 2.0):
        self.dest, self.names, self.total, self.start = dest, names, total, start
        self.label, self.out, self.interval = label, out, interval
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self.last_line = ""

    def line(self, now_bytes: int, rate: float) -> str:
        done = min(now_bytes, self.total)
        eta = (self.total - done) / rate if rate > 0 else None
        return (f"downloading {self.label}: {done / GIB:.1f}/{self.total / GIB:.1f} GB "
                f"({100 * done / max(1, self.total):.0f}%), {rate / 1e6:.0f} MB/s, "
                f"ETA {fmt_eta(eta)}")

    def _run(self):
        t0, b0 = time.monotonic(), _present_bytes(self.dest, self.names)
        rate = 0.0
        prev_t, prev_b = t0, b0
        while not self._stop.wait(self.interval):
            now = _present_bytes(self.dest, self.names)
            t = time.monotonic()
            inst = (now - prev_b) / max(1e-6, t - prev_t)
            rate = inst if rate == 0 else 0.7 * rate + 0.3 * inst
            prev_t, prev_b = t, now
            self.last_line = self.line(now, rate)
            self.out.write("\r" + self.last_line + "\x1b[K")
            self.out.flush()

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=3)
        if self.last_line:
            self.out.write("\n")
            self.out.flush()


def fetch(repo: str, dest: Path, patterns: list[str], label: str, revision: str | None = None,
          recovery: str | None = None, floor: int = MIN_FREE_AFTER_DOWNLOAD, say=None,
          snapshot=None, api=None) -> Path:
    """Download the files of `repo` matching `patterns` into `dest`."""
    say = say or (lambda s: print(s, file=sys.stderr))
    from . import guard
    guard.check(repo)
    from huggingface_hub import HfApi, snapshot_download
    snapshot = snapshot or snapshot_download
    dest = Path(dest)
    info = (api or HfApi()).model_info(repo, revision=revision, files_metadata=True)
    files = [(f.rfilename, f.size or 0) for f in info.siblings]
    total, remaining = plan(files, dest, patterns)
    if total == 0:
        raise SpillError(f"{repo} has no files matching {', '.join(patterns)}")
    if remaining == 0:
        return dest
    check_disk(remaining, dest, label, floor, recovery)
    tr = transport()
    resumed = total - remaining
    say(f"downloading {label}: {remaining / GIB:.1f} GB to fetch"
        + (f" ({resumed / GIB:.1f} GB already here, resuming)" if resumed else "")
        + f" -> {dest} via {tr}")
    names = {n for n, _ in files if any(fnmatch.fnmatch(n, p) for p in patterns)}
    from huggingface_hub.utils import (are_progress_bars_disabled, disable_progress_bars,
                                       enable_progress_bars)
    was_off = are_progress_bars_disabled()
    disable_progress_bars()           # our aggregate line replaces the per-file bars
    try:
        with Progress(dest, names, total, resumed, label):
            for attempt in range(1, RETRIES + 1):
                try:
                    snapshot(repo, revision=revision, allow_patterns=patterns, local_dir=dest)
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    if attempt == RETRIES or not _retryable(e):
                        raise
                    say(f"download interrupted ({type(e).__name__}); resuming "
                        f"(attempt {attempt + 1}/{RETRIES})")
                    time.sleep(min(30, 2 ** attempt))
    finally:
        if not was_off:
            enable_progress_bars()
    return dest


def _retryable(e: Exception) -> bool:
    import errno
    if isinstance(e, OSError) and e.errno in (errno.ENOSPC, errno.EACCES, errno.EROFS):
        return False                      # a full or read-only disk will not fix itself
    name = type(e).__name__
    return any(k in name for k in ("Timeout", "Connection", "ReadError", "ChunkedEncoding",
                                   "RemoteProtocol", "OSError", "HTTPError", "IncompleteRead")) \
        or isinstance(e, (ConnectionError, TimeoutError, OSError))
