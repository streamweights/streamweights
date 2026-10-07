"""Tune job checkpoints on MLX: a thin layer over the portable checkpoint
(streamweights.portable.checkpoint), converting between mx arrays and the float32 numpy form
every engine reads. `d` is a local directory or a Store (a state URI)."""

from __future__ import annotations

from pathlib import Path

from ..portable import checkpoint as pc
from ..portable.store import Store


def _store(d) -> Store:
    return d if isinstance(d, Store) else Store(d)


def save(d, *, step: int, params: dict, opt: dict, extra: dict | None = None) -> None:
    """params: {name: array}; opt: {"step": int, "m": {name: arr}, "v": {name: arr}}."""
    pc.save_tune(_store(d), step=step, params=params, opt=opt, state=extra or {})


def exists(d) -> bool:
    return pc.latest_step(_store(d)) is not None


def load(d):
    """(step, params, opt, state.json) from the latest committed checkpoint, as mx arrays."""
    import mlx.core as mx
    ck = pc.load_tune(_store(d))
    if ck is None:
        raise FileNotFoundError(f"no committed checkpoint in {d}")
    params = {k: mx.array(v) for k, v in ck.params.items()}
    opt = {"step": ck.opt["step"],
           "m": {k: mx.array(v) for k, v in ck.opt["m"].items()},
           "v": {k: mx.array(v) for k, v in ck.opt["v"].items()}}
    return ck.step, params, opt, ck.state
