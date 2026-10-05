"""Tune job checkpoints: adapter parameters and AdamW state, one format for both
trainers. The checkpoint directory is replaced atomically (write a sibling, swap),
so a crash mid-save leaves the previous checkpoint intact."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx


def save(d: Path, *, step: int, params: dict, opt: dict, extra: dict | None = None) -> None:
    """params: {name: array}; opt: {"step": int, "m": {name: arr}, "v": {name: arr}}."""
    d = Path(d)
    tmp = d.with_name(d.name + ".new")
    old = d.with_name(d.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "params.safetensors"), dict(params))
    flat = {f"m::{k}": v for k, v in opt["m"].items()}
    flat.update({f"v::{k}": v for k, v in opt["v"].items()})
    mx.save_safetensors(str(tmp / "opt.safetensors"), flat)
    (tmp / "state.json").write_text(json.dumps({"step": step, "opt_step": int(opt["step"]),
                                                **(extra or {})}))
    shutil.rmtree(old, ignore_errors=True)
    if d.exists():
        d.rename(old)
    tmp.rename(d)
    shutil.rmtree(old, ignore_errors=True)


def exists(d: Path) -> bool:
    d = Path(d)
    if (d / "state.json").exists():
        return True
    old = d.with_name(d.name + ".old")      # crash between the two renames
    return (old / "state.json").exists()


def load(d: Path) -> tuple[int, dict, dict, dict]:
    """(step, params, opt, state.json) from a checkpoint directory."""
    d = Path(d)
    if not (d / "state.json").exists():
        old = d.with_name(d.name + ".old")
        if (old / "state.json").exists():
            d = old
    state = json.loads((d / "state.json").read_text())
    params = mx.load(str(d / "params.safetensors"))
    flat = mx.load(str(d / "opt.safetensors"))
    opt = {"step": state["opt_step"],
           "m": {k[3:]: v for k, v in flat.items() if k.startswith("m::")},
           "v": {k[3:]: v for k, v in flat.items() if k.startswith("v::")}}
    return state["step"], params, opt, state
