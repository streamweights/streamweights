"""The streaming ring, independent of any ML framework: the safetensors layer index and a
ring of preallocated host buffers that a reader thread fills with one transformer layer each.

Both the MLX and the PyTorch engines bind weights from these buffers, so the I/O path is one
piece of code. Read modes keep streamed weights out of the page cache:

  nocache   macOS: F_NOCACHE on the descriptor
  fadvise   Linux: pread, then posix_fadvise(DONTNEED) on the range just read
  odirect   Linux: O_DIRECT reads into page-aligned bounce buffers
  buffered  plain pread (a fallback, and the right choice when the file is already cached)

`pick_read_mode` measures the Linux candidates on the model being served and keeps the faster.
"""

from __future__ import annotations

import json
import mmap
import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration import load_calibration, save_calibration

MB = 1024 * 1024
ALIGN = 4096

# safetensors dtype -> (numpy storage dtype, bytes per element)
ST_NP = {
    "BF16": (np.uint16, 2),
    "F16": (np.float16, 2),
    "F32": (np.float32, 4),
    "U32": (np.uint32, 4),
    "I32": (np.int32, 4),
}


# ---------------------------------------------------------------- index

@dataclass
class TensorLoc:
    name: str          # safetensors key
    shard: Path
    offset: int        # absolute byte offset in shard
    nbytes: int
    shape: tuple
    st_dtype: str


@dataclass
class LayerPlan:
    """All tensors of one streamed unit (a transformer block), with a coalesced read plan."""
    layer_id: int
    tensors: list[TensorLoc]
    segments: list[tuple[Path, int, int, int]] = field(default_factory=list)
    # (shard, file_offset, length, buf_offset); tensor buf offsets assigned in order
    tensor_buf_offsets: dict[str, int] = field(default_factory=dict)
    nbytes: int = 0

    def build(self):
        self.tensors.sort(key=lambda t: (str(t.shard), t.offset))
        buf_off = 0
        segs: list[list] = []
        for t in self.tensors:
            self.tensor_buf_offsets[t.name] = buf_off
            if segs and segs[-1][0] == t.shard and segs[-1][1] + segs[-1][2] == t.offset:
                segs[-1][2] += t.nbytes
            else:
                segs.append([t.shard, t.offset, t.nbytes, buf_off])
            buf_off += t.nbytes
        self.segments = [tuple(s) for s in segs]
        self.nbytes = buf_off


class SafetensorsIndex:
    """Parse all shard headers (no tensor data) and group tensors by transformer
    block in execution order, plus embeddings, final norm, lm_head."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text())
        self.tensors: dict[str, TensorLoc] = {}
        for shard in sorted(self.model_dir.glob("*.safetensors")):
            with open(shard, "rb") as f:
                (hlen,) = np.frombuffer(f.read(8), dtype=np.uint64)
                header = json.loads(f.read(int(hlen)))
                base = 8 + int(hlen)
                for name, meta in header.items():
                    if name == "__metadata__":
                        continue
                    s, e = meta["data_offsets"]
                    self.tensors[name] = TensorLoc(
                        name, shard, base + s, e - s, tuple(meta["shape"]), meta["dtype"])

        self.n_layers = self.config["num_hidden_layers"]
        self.layers: list[LayerPlan] = []
        for k in range(self.n_layers):
            prefix = f"model.layers.{k}."
            plan = LayerPlan(k, [t for n, t in self.tensors.items() if n.startswith(prefix)])
            if not plan.tensors:
                raise ValueError(f"no tensors found for layer {k}")
            plan.build()
            self.layers.append(plan)
        self.max_layer_bytes = max(p.nbytes for p in self.layers)

        self.embed = self.tensors["model.embed_tokens.weight"]
        self.final_norm = self.tensors["model.norm.weight"]
        self.lm_head = self.tensors.get("lm_head.weight")  # None if tied
        self.tied = self.lm_head is None

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())


# ---------------------------------------------------------------- read modes

def default_read_mode() -> str:
    """The mode used before any measurement exists."""
    if platform.system() == "Darwin":
        return "nocache"
    if hasattr(os, "posix_fadvise"):
        return "fadvise"
    return "buffered"


def available_read_modes() -> list[str]:
    if platform.system() == "Darwin":
        return ["nocache", "buffered"]
    modes = ["buffered"]
    if hasattr(os, "posix_fadvise"):
        modes.insert(0, "fadvise")
    if hasattr(os, "O_DIRECT"):
        modes.insert(1 if len(modes) > 1 else 0, "odirect")
    return modes


class _Fds:
    """Per-shard descriptors opened for the read mode."""

    def __init__(self, mode: str = "nocache"):
        self.mode = mode
        self._fds: dict[Path, int] = {}
        self._lock = threading.Lock()

    def get(self, shard: Path) -> int:
        with self._lock:
            fd = self._fds.get(shard)
            if fd is None:
                flags = os.O_RDONLY
                if self.mode == "odirect":
                    flags |= getattr(os, "O_DIRECT", 0)
                fd = os.open(shard, flags)
                if self.mode == "nocache":
                    import fcntl
                    if hasattr(fcntl, "F_NOCACHE"):
                        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                self._fds[shard] = fd
            return fd

    def close(self):
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


def _bytearray_alloc(n: int):
    return bytearray(n)


# ---------------------------------------------------------------- ring reader

class RingReader:
    """N preallocated host buffers; a producer thread fills them with layers in
    schedule order using a thread pool of large preads; compute consumes in the
    same order. The free-slot semaphore guarantees a slot is never overwritten
    while still in use.

    `alloc(nbytes)` returns a writable buffer-protocol object; the PyTorch CUDA engine passes
    an allocator of pinned host tensors so the same buffers feed asynchronous host-to-device
    copies."""

    def __init__(self, index: SafetensorsIndex, n_slots: int = 3,
                 chunk_bytes: int = 16 * MB, n_threads: int = 4,
                 mode: str | None = None, alloc=None):
        self.index = index
        self.n_slots = n_slots
        self.chunk = chunk_bytes
        self.mode = mode or default_read_mode()
        alloc = alloc or _bytearray_alloc
        self.bufs = [alloc(index.max_layer_bytes) for _ in range(n_slots)]
        self.free = threading.Semaphore(n_slots)
        self.ready: dict[int, int] = {}       # schedule seq -> slot id
        self.cond = threading.Condition()
        self.pool = ThreadPoolExecutor(max_workers=n_threads)
        self.fds = _Fds(self.mode)
        self.bytes_read = 0
        self.read_seconds = 0.0
        self._stop = threading.Event()
        self._producer: threading.Thread | None = None
        self._tls = threading.local()
        self.error: BaseException | None = None
        self.layer_ids: dict[int, int] = {}   # schedule seq -> layer id, until consumed

    # -- one chunk, by mode

    def _aligned_buf(self, n: int) -> mmap.mmap:
        buf = getattr(self._tls, "buf", None)
        if buf is None or len(buf) < n:
            buf = mmap.mmap(-1, n)           # anonymous mmap: page aligned
            self._tls.buf = buf
        return buf

    def _read_direct(self, fd: int, file_off: int, length: int, dst: memoryview):
        start = (file_off // ALIGN) * ALIGN
        head = file_off - start
        want = head + length
        span = ((want + ALIGN - 1) // ALIGN) * ALIGN
        buf = self._aligned_buf(span)
        mv = memoryview(buf)
        got = 0
        while got < want:
            n = os.preadv(fd, [mv[got:span]], start + got)
            if n <= 0:
                break
            got += n
        if got < want:
            raise IOError("short O_DIRECT read")
        dst[:length] = mv[head:head + length]

    def _read_segment(self, fd: int, file_off: int, length: int, buf, buf_off: int):
        mv = memoryview(buf)
        if mv.format != "B":
            mv = mv.cast("B")
        done = 0
        while done < length:
            n = min(self.chunk, length - done)
            dst = mv[buf_off + done: buf_off + done + n]
            if self.mode == "odirect":
                self._read_direct(fd, file_off + done, n, dst)
                got = n
            else:
                got = os.preadv(fd, [dst], file_off + done)
                if got <= 0:
                    raise IOError("short pread")
                if self.mode == "fadvise":
                    os.posix_fadvise(fd, file_off + done, got, os.POSIX_FADV_DONTNEED)
            done += got

    def _fill_slot(self, slot: int, plan: LayerPlan):
        t0 = time.monotonic()
        futs = []
        for shard, foff, length, boff in plan.segments:
            fd = self.fds.get(shard)
            # split large segments across the pool in chunk-sized pieces
            pos = 0
            while pos < length:
                n = min(self.chunk, length - pos)
                futs.append(self.pool.submit(
                    self._read_segment, fd, foff + pos, n, self.bufs[slot], boff + pos))
                pos += n
        for f in futs:
            f.result()
        self.bytes_read += plan.nbytes
        self.read_seconds += time.monotonic() - t0

    def start(self, schedule):
        """schedule: iterable of layer ids in the exact order compute will consume them."""
        def produce():
            try:
                next_slot = 0
                for seq, layer_id in enumerate(schedule):
                    if self._stop.is_set():
                        return
                    self.free.acquire()
                    if self._stop.is_set():
                        return
                    slot = next_slot % self.n_slots
                    next_slot += 1
                    self._fill_slot(slot, self.index.layers[layer_id])
                    with self.cond:
                        self.layer_ids[seq] = layer_id
                        self.ready[seq] = slot
                        self.cond.notify_all()
            except BaseException as e:      # surface reader failures to the consumer
                self.error = e
                with self.cond:
                    self.cond.notify_all()
        self._producer = threading.Thread(target=produce, daemon=True)
        self._producer.start()

    def get(self, seq: int) -> tuple[int, memoryview]:
        with self.cond:
            while seq not in self.ready:
                if self.error is not None:
                    raise self.error
                self.cond.wait(timeout=60)
        slot = self.ready.pop(seq)
        return slot, memoryview(self.bufs[slot])

    def release(self, slot: int):
        self.free.release()

    def stop(self):
        self._stop.set()
        self.free.release()  # unblock producer
        if self._producer:
            self._producer.join(timeout=10)
        self.pool.shutdown(wait=False)
        self.fds.close()


# ---------------------------------------------------------------- calibration

def _read_rate(index: SafetensorsIndex, n_layers: int, chunk_mb: int, threads: int,
               mode: str) -> float:
    ring = RingReader(index, n_slots=3, chunk_bytes=chunk_mb * MB, n_threads=threads, mode=mode)
    sched = list(range(min(n_layers, index.n_layers)))
    t0 = time.monotonic()
    ring.start(sched)
    total = 0
    try:
        for seq in range(len(sched)):
            slot, _ = ring.get(seq)
            total += index.layers[sched[seq]].nbytes
            ring.release(slot)
    finally:
        dt = time.monotonic() - t0
        ring.stop()
    return total / dt / MB


def pick_read_mode(index: SafetensorsIndex, n_layers: int = 4) -> tuple[str, dict]:
    """Measure each available read mode on this model's layers; (best, {mode: MB/s}).
    A mode that fails (O_DIRECT on a filesystem that refuses it) is skipped."""
    rates: dict[str, float] = {}
    for mode in available_read_modes():
        if mode == "buffered" and len(available_read_modes()) > 1:
            continue                      # buffered reads of a cold file say nothing useful
        try:
            rates[mode] = round(_read_rate(index, n_layers, 16, 4, mode), 1)
        except (OSError, IOError):
            continue
    if not rates:
        return default_read_mode(), {}
    return max(rates, key=rates.get), rates


def calibrate(index: SafetensorsIndex, n_layers: int = 4,
              chunks_mb=(4, 16, 64), threads=(2, 4, 8)) -> dict:
    """Measure isolated read rate (no compute) over a grid; persist the best."""
    mode, mode_rates = pick_read_mode(index, n_layers)
    results = []
    for cmb in chunks_mb:
        for nt in threads:
            mbps = _read_rate(index, n_layers, cmb, nt, mode)
            results.append({"chunk_mb": cmb, "threads": nt, "mbps": round(mbps, 1)})
    best = max(results, key=lambda r: r["mbps"])
    cal = load_calibration()
    cal.update({
        "isolated_read_mbps": best["mbps"],
        "chunk_mb": best["chunk_mb"],
        "threads": best["threads"],
        "read_mode": mode,
        "read_mode_mbps": mode_rates,
        "grid": results,
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    save_calibration(cal)
    return cal


def ring_settings(index: SafetensorsIndex, note=None) -> tuple[int, int, str]:
    """(chunk MB, threads, read mode) from the calibration, measuring it first when absent."""
    cal = load_calibration()
    if "isolated_read_mbps" not in cal or "read_mode" not in cal:
        if note:
            note("calibrating read rate (no compute)...")
        cal = calibrate(index)
    return cal.get("chunk_mb", 16), cal.get("threads", 4), cal.get("read_mode",
                                                                   default_read_mode())
