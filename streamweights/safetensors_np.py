"""Read and write safetensors with numpy only, bf16 included, so `spill export` runs
where MLX does not. Tensors are read lazily from a memory map and written one at a time."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_RAW = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensors(path: Path):
    """Yield (name, dtype, shape, raw) in file order; `raw` is a memory-mapped view."""
    header, base = read_header(path)
    meta = header.pop("__metadata__", None)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    items = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    for name, t in items:
        dt = t["dtype"]
        if dt not in _RAW:
            raise ValueError(f"{name}: dtype {dt} is not supported (BF16, F16, F32)")
        lo, hi = t["data_offsets"]
        raw = mm[base + lo:base + hi].view(_RAW[dt]).reshape(t["shape"])
        yield name, dt, tuple(t["shape"]), raw
    del meta


def to_f32(raw: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32)


def from_f32(x: np.ndarray, dtype: str) -> np.ndarray:
    """float32 to the storage dtype; bf16 rounds to nearest even."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if dtype == "BF16":
        b = x.view(np.uint32)
        rounded = b + np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))
        return (rounded >> np.uint32(16)).astype(np.uint16)
    return x.astype(np.float16) if dtype == "F16" else x


def load_f32(path: Path) -> dict[str, np.ndarray]:
    return {n: to_f32(raw, dt) for n, dt, _s, raw in tensors(path)}


def write(path: Path, entries, metadata: dict | None = None) -> None:
    """entries: list of (name, dtype, shape, make) where make() returns the raw ndarray."""
    header, off = {}, 0
    if metadata:
        header["__metadata__"] = metadata
    for name, dt, shape, _make in entries:
        size = int(np.prod(shape, dtype=np.int64)) * np.dtype(_RAW[dt]).itemsize
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + size]}
        off += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _name, _dt, _shape, make in entries:
            f.write(np.ascontiguousarray(make()).tobytes())
