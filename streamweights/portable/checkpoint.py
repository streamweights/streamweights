"""The hardware-neutral checkpoint: one format every engine writes and every engine reads.

A tune checkpoint is a directory `ckpt/step-<n>/` holding

  params.safetensors   LoRA parameters, float32, names `layers.<k>.<module>.lora_a|b`
                       with A as [in, r] and B as [r, out]
  opt.safetensors      AdamW state, float32, keys `m::<name>` and `v::<name>`
  state.json           step, seed, data cursor, RNG per data stream, model id and weight
                       fingerprint, adapter config, hyperparameters, and the engine,
                       hardware and numerics that produced each step range
  COMMIT               written last: sizes and sha256 of the three files

Only a directory with a valid COMMIT counts. A writer that dies halfway leaves an uncommitted
directory that readers skip and the next save removes. Older committed checkpoints are
pruned after a newer one commits, keeping two.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import time
from dataclasses import dataclass, field

import numpy as np

SCHEMA = 1
KEEP = 2
_STEP_DIR = re.compile(r"^step-(\d{9})$")


# ------------------------------------------------------------ array conversion

def to_numpy_f32(a) -> np.ndarray:
    """numpy float32 from a numpy, torch or mlx array, without importing either framework."""
    mod = type(a).__module__
    if mod.startswith("torch"):
        return np.ascontiguousarray(a.detach().to("cpu").float().numpy())
    if mod.startswith("mlx"):
        import mlx.core as mx
        return np.ascontiguousarray(np.array(a.astype(mx.float32)))
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def _pack(tensors: dict, metadata: dict | None = None) -> bytes:
    from safetensors.numpy import save
    return save({k: to_numpy_f32(v) for k, v in tensors.items()}, metadata=metadata or {})


def _unpack(data: bytes) -> dict[str, np.ndarray]:
    from safetensors.numpy import load
    return load(data)


# ------------------------------------------------------------ producer description

def producer(engine: str, device: str, numerics: dict, step_from: int, step_to: int) -> dict:
    """What produced a range of steps (or rows): engine, hardware, numerics."""
    from .. import machine
    m = machine.info()
    return {"range": [step_from, step_to], "engine": engine, "hardware": device,
            "numerics": numerics, "host": m["host"], "system": m["system"],
            "os": f"{m['os']} {m['arch']}", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def extend_history(history: list[dict], new: dict) -> list[dict]:
    """Append `new`, merging with the last entry when engine, hardware and numerics match
    and the ranges touch."""
    out = [dict(h) for h in history]
    if out:
        last = out[-1]
        same = (last["engine"] == new["engine"] and last["hardware"] == new["hardware"]
                and last["numerics"] == new["numerics"])
        if same and last["range"][1] == new["range"][0]:
            last["range"][1] = new["range"][1]
            return out
    out.append(new)
    return out


# ------------------------------------------------------------ tune checkpoints

@dataclass
class TuneCheckpoint:
    step: int
    params: dict[str, np.ndarray]
    opt: dict                      # {"step": int, "m": {...}, "v": {...}}
    state: dict = field(default_factory=dict)


def _dir(step: int) -> str:
    return f"ckpt/step-{step:09d}"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def committed_steps(store) -> list[int]:
    out = []
    for name in store.ls("ckpt"):
        m = _STEP_DIR.match(name)
        if m and store.exists(f"ckpt/{name}/COMMIT"):
            out.append(int(m.group(1)))
    return sorted(out)


def _valid(store, step: int) -> dict | None:
    d = _dir(step)
    try:
        commit = json.loads(store.read(f"{d}/COMMIT"))
        for name, meta in commit["files"].items():
            if store.size(f"{d}/{name}") != meta["bytes"]:
                return None
        return commit
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def latest_step(store) -> int | None:
    for step in reversed(committed_steps(store)):
        if _valid(store, step):
            return step
    return None


def save_tune(store, *, step: int, params: dict, opt: dict, state: dict) -> None:
    """Write step-<n>/ then its COMMIT, then prune. `params` and `opt` hold numpy, torch or
    mlx arrays; they are stored as float32."""
    d = _dir(step)
    store.rm(d)                                   # a stale, uncommitted attempt at this step
    files = {
        "params.safetensors": _pack(params),
        "opt.safetensors": _pack({**{f"m::{k}": v for k, v in opt["m"].items()},
                                  **{f"v::{k}": v for k, v in opt["v"].items()}}),
        "state.json": json.dumps({"schema": SCHEMA, "kind": "tune", **state, "step": step,
                                  "opt_step": int(opt["step"])}, indent=1).encode(),
    }
    for name, data in files.items():
        store.write(f"{d}/{name}", data)
    commit = {"schema": SCHEMA, "step": step,
              "files": {n: {"bytes": len(b), "sha256": _sha(b)} for n, b in files.items()}}
    store.write(f"{d}/COMMIT", json.dumps(commit).encode())
    store.write("ckpt/LATEST", json.dumps({"step": step}).encode())
    keep = committed_steps(store)
    for old in keep[:-KEEP]:
        store.rm(_dir(old))
    for name in store.ls("ckpt"):                 # uncommitted leftovers from dead writers
        m = _STEP_DIR.match(name)
        if m and int(m.group(1)) != step and not store.exists(f"ckpt/{name}/COMMIT"):
            store.rm(f"ckpt/{name}")


def load_tune(store, step: int | None = None) -> TuneCheckpoint | None:
    """The latest committed (and intact) checkpoint, or the named step; None if there is none."""
    candidates = [step] if step is not None else list(reversed(committed_steps(store)))
    for s in candidates:
        commit = _valid(store, s)
        if not commit:
            continue
        d = _dir(s)
        blobs = {n: store.read(f"{d}/{n}") for n in commit["files"]}
        if any(_sha(b) != commit["files"][n]["sha256"] for n, b in blobs.items()):
            continue
        state = json.loads(blobs["state.json"])
        params = _unpack(blobs["params.safetensors"])
        flat = _unpack(blobs["opt.safetensors"])
        opt = {"step": state["opt_step"],
               "m": {k[3:]: v for k, v in flat.items() if k.startswith("m::")},
               "v": {k[3:]: v for k, v in flat.items() if k.startswith("v::")}}
        return TuneCheckpoint(s, params, opt, state)
    return None


# ------------------------------------------------------------ the state.json a tune writes

def tune_state(*, spec: dict, model_id: str, fingerprint: str, data_sha256: str,
               targets: list[str], history: list[dict], grad_accum: int, step: int) -> dict:
    """The portable part of a tune job's state, as written into state.json."""
    return {
        "seed": spec["seed"],
        "data_cursor": {"micro_batch_index": step * grad_accum, "step": step},
        "rng": {
            "data": {"scheme": "numpy.RandomState(seed + epoch).permutation over micro-batch "
                               "groups sorted by length; a pure function of (seed, micro-batch "
                               "index), so the cursor is the whole state",
                     "seed": spec["seed"]},
            "dropout": {"scheme": "per (micro-batch, layer, module) seed derived from the job "
                                  "seed; stateless", "seed": spec["seed"],
                        "p": spec["dropout"]},
        },
        "model": {"id": model_id, "weight_fingerprint": fingerprint},
        "data": {"sha256": data_sha256},
        "adapter": {"rank": spec["rank"], "alpha": spec["alpha"], "dropout": spec["dropout"],
                    "targets": targets},
        "hyperparameters": {k: spec[k] for k in ("lr", "weight_decay", "schedule", "max_seq",
                                                 "micro_batch", "grad_accum", "steps", "epochs")},
        "history": history,
    }


def check_resume_compatible(state: dict, *, model_id: str, fingerprint: str,
                            data_sha256: str, spec: dict) -> str | None:
    """None when this job may continue from `state`; otherwise the one-line reason it may not."""
    got = state.get("model", {})
    if got.get("weight_fingerprint") and got["weight_fingerprint"] != fingerprint:
        return (f"the checkpoint was made with different weights for {got.get('id')} "
                f"(fingerprint {got['weight_fingerprint'][:12]}, these are {fingerprint[:12]})")
    if state.get("data", {}).get("sha256") not in (None, data_sha256):
        return "the checkpoint was made with a different training file"
    h = state.get("hyperparameters", {})
    for k in ("lr", "weight_decay", "schedule", "micro_batch", "grad_accum", "max_seq", "steps"):
        if k in h and h[k] != spec[k]:
            return f"the checkpoint used {k}={h[k]}, this job asks for {spec[k]}"
    a = state.get("adapter", {})
    if a and (a.get("rank") != spec["rank"] or a.get("alpha") != spec["alpha"]):
        return f"the checkpoint's adapter is rank {a.get('rank')} alpha {a.get('alpha')}"
    if state.get("seed", spec["seed"]) != spec["seed"]:
        return f"the checkpoint used seed {state.get('seed')}"
    return None
