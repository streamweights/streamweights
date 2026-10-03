"""Run store: runs/<id>/manifest.json for every command that executes a model.

The manifest is what makes an eval table reproducible: the command, the exact
weights (a content fingerprint), the adapter (full hash), the input file hash,
the engine version, the hardware probe, the time span, and the options that
change output (logprobs, mode). Per-row provenance (run id, weight and adapter
hash) is stamped into every results row by jobs/runner.py.
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

from .registry import REPO_ROOT

RUNS_DIR = REPO_ROOT / "runs"
FP_KIND = "sha256(config + shard names + sizes + safetensors headers)"


# ------------------------------------------------------------ hashing

def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fingerprint_model(path: Path) -> str:
    """Content fingerprint of a model directory without reading tensor data:
    sha256 over config.json, every shard's name and size, and every safetensors
    header (which carries dtype, shape and offset of every tensor). Hashing 140 GB
    of weights at every run would cost minutes of disk bandwidth; the header
    fingerprint changes whenever any tensor's identity or layout changes. For a
    single file (GGUF) it hashes the size plus the first and last MiB."""
    p = Path(path)
    h = hashlib.sha256()
    if p.is_file():
        size = p.stat().st_size
        h.update(f"{p.name}:{size}".encode())
        with open(p, "rb") as f:
            h.update(f.read(1 << 20))
            if size > (2 << 20):
                f.seek(-(1 << 20), 2)
                h.update(f.read(1 << 20))
        return h.hexdigest()
    cfg = p / "config.json"
    if cfg.exists():
        h.update(cfg.read_bytes())
    for shard in sorted(p.glob("*.safetensors")):
        h.update(f"{shard.name}:{shard.stat().st_size}".encode())
        with open(shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            h.update(f.read(hlen))
    return h.hexdigest()


def input_hash(path: Path) -> str:
    return sha256_file(Path(path))


# ------------------------------------------------------------ versions

def engine_version(engine_name: str) -> dict:
    try:
        from importlib.metadata import version
        sw = version("streamweights")
    except Exception:
        sw = "unknown"
    out = {"name": engine_name, "streamweights": sw}
    try:
        out["git"] = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3).stdout.strip() or None
    except Exception:
        out["git"] = None
    for mod in ("mlx", "mlx_lm"):
        try:
            from importlib.metadata import version as v
            out[mod.replace("_", "-")] = v(mod.replace("_", "-"))
        except Exception:
            pass
    return out


def hardware_summary(hw: dict | None) -> dict:
    """The probe, reduced to what explains a result: chip, memory, Metal working set, disk rate."""
    if not hw:
        return {}
    out = {k: hw[k] for k in ("cpu", "cpu_cores", "ram_total_bytes", "os", "probed_at")
           if k in hw}
    if "gpu" in hw:
        out["gpu"] = {k: hw["gpu"].get(k) for k in ("model", "vram_bytes", "bf16_compute")}
    if "nvme_seq_read" in hw:
        out["nvme_seq_read_bytes_per_sec"] = hw["nvme_seq_read"].get("bytes_per_sec")
    return out


# ------------------------------------------------------------ store

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def manifest_path(run_id: str) -> Path:
    return run_dir(run_id) / "manifest.json"


def read_manifest(run_id: str) -> dict:
    return json.loads(manifest_path(run_id).read_text())


def write_manifest(run_id: str, m: dict) -> None:
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "manifest.json.tmp"
    tmp.write_text(json.dumps(m, indent=2) + "\n")
    tmp.replace(d / "manifest.json")


def update_manifest(run_id: str, **kw) -> dict:
    m = read_manifest(run_id)
    m.update(kw)
    write_manifest(run_id, m)
    return m


def provenance(m: dict) -> dict:
    """The per-row stamp: enough to tie a row to its exact run."""
    return {
        "run_id": m["id"],
        "model": m["model"]["id"],
        "weight_hash": m["model"]["weight_hash"][:16],
        "adapter_hash": (m["adapter"]["hash"][:16] if m.get("adapter") else None),
        "input_hash": m["input"]["sha256"][:16],
    }


def start_run(job, *, command: str | None, spec, input_path: Path, hw: dict | None,
              engine_name: str, adapter: dict | None = None, options: dict | None = None,
              kind: str = "run") -> dict:
    """Write runs/<job id>/manifest.json (the run id is the job id) and link the
    results file next to it."""
    command = command or ("spill " + " ".join(sys.argv[1:]))
    m = {
        "id": job.id,
        "kind": kind,
        "command": command,
        "model": {"id": spec.name, "quant": spec.quant,
                  "weight_hash": fingerprint_model(spec.path),
                  "weight_hash_kind": FP_KIND, "path": str(spec.path)},
        "adapter": adapter,
        "input": {"path": str(input_path), "sha256": input_hash(input_path),
                  "rows": job.total},
        "engine": engine_version(engine_name),
        "hardware": hardware_summary(hw),
        "options": options or {},
        "started": now(),
        "ended": None,
        "status": "running",
        "rows_done": 0,
        "job_dir": str(job.dir),
        "resumes": [],
    }
    write_manifest(job.id, m)
    link = run_dir(job.id) / "results.jsonl"
    if not link.exists() and not link.is_symlink():
        try:
            link.symlink_to(job.results_path)
        except OSError:
            pass
    job.write_meta(run_id=job.id, provenance=provenance(m))
    return m


def mark_resumed(run_id: str) -> None:
    if not manifest_path(run_id).exists():
        return
    m = read_manifest(run_id)
    m.setdefault("resumes", []).append(now())
    m["status"] = "running"
    m["ended"] = None
    write_manifest(run_id, m)


def finish_run(run_id: str, *, status: str, rows_done: int, tokens: dict | None = None,
               results_path: Path | None = None) -> None:
    if not manifest_path(run_id).exists():
        return
    kw = {"ended": now(), "status": status, "rows_done": rows_done}
    if tokens:
        kw["tokens"] = tokens
    if results_path and Path(results_path).exists():
        kw["results_sha256"] = sha256_file(Path(results_path))
    update_manifest(run_id, **kw)


def list_runs() -> list[dict]:
    out = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir()):
            mp = d / "manifest.json"
            if mp.exists():
                try:
                    out.append(json.loads(mp.read_text()))
                except json.JSONDecodeError:
                    pass
    return out


def find_cached(model: str, quant: str | None, adapter_hash: str | None,
                in_hash: str, options_key: dict | None = None) -> dict | None:
    """A finished run for exactly this model, quant, adapter and input hash."""
    for m in reversed(list_runs()):
        if m.get("kind") != "run" or m.get("status") != "completed":
            continue
        if m["model"]["id"] != model:
            continue
        if quant and m["model"]["quant"] != quant:
            continue
        if (m.get("adapter") or {}).get("hash") != adapter_hash:
            continue
        if m["input"]["sha256"] != in_hash:
            continue
        if m.get("rows_done") != m["input"]["rows"]:
            continue
        if options_key is not None and any(
                m.get("options", {}).get(k) != v for k, v in options_key.items()):
            continue
        return m
    return None
