"""The run-anywhere gates, as a library: identity of the PyTorch engines against each other and
against MLX, and jobs that move between engines mid-run. `scripts/gates_012.py` runs them on
a Mac; `streamweights.verify_cuda` runs them on a CUDA machine.

Tune and eval gates run the real `spill` command in a subprocess against a private SPILL_HOME,
in headless mode, so what is gated is what a user runs. Inference gates call the engines
directly, because they compare log-probs token by token.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GATE_LR = 2e-5            # natural-text gate: the loss falls from 1.4 to 0.3 in 100 steps
TOY_LR = 2e-6             # toy task: slow enough that the loss is still moving at step 50


# ---------------------------------------------------------------- running spill

@dataclass
class Result:
    rc: int
    events: list[dict]
    stderr: str
    stdout: str = ""

    def last(self, kind: str) -> dict | None:
        for e in reversed(self.events):
            if e.get("event") == kind:
                return e
        return None

    @property
    def done(self) -> dict | None:
        return self.last("done")


class Work:
    """A private SPILL_HOME for gate runs: shares the repo's downloaded models and measured
    hardware, so nothing is downloaded twice and no gate writes into the checkout."""

    def __init__(self, root: Path, models_from: Path | None = None, state_from: Path | None = None):
        self.root = Path(root)
        self.home = self.root / "home"
        (self.home / "state").mkdir(parents=True, exist_ok=True)
        for f in ("hardware.json", "calibration.json"):
            if state_from and (Path(state_from) / f).exists() and not (self.home / "state" / f).exists():
                shutil.copy(Path(state_from) / f, self.home / "state" / f)
        if models_from and Path(models_from).exists() and not (self.home / "models").exists():
            (self.home / "models").symlink_to(Path(models_from).resolve())

    def spill(self, args: list, *, dtype: str | None = None, env: dict | None = None,
              timeout: int = 7200) -> Result:
        e = {**os.environ, "SPILL_HOME": str(self.home), "SPILL_HEADLESS": "1"}
        if dtype:
            e["SPILL_TORCH_DTYPE"] = dtype
        e.update(env or {})
        p = subprocess.run([sys.executable, "-m", "streamweights", *[str(a) for a in args]],
                           capture_output=True, text=True, env=e, timeout=timeout)
        evs = []
        for line in p.stdout.splitlines():
            try:
                evs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return Result(p.returncode, evs, p.stderr, p.stdout)

    def job_losses(self, job_id: str) -> list[float]:
        p = self.home / "jobs" / job_id / "losses.jsonl"
        by_step = {}
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                by_step[d["step"]] = d["loss"]
        return [by_step[k] for k in sorted(by_step)]

    def adapter_params(self, name: str) -> dict[str, np.ndarray]:
        from .tune.lora_core import read_adapter_params
        return read_adapter_params(self.home / "adapters" / name)


def tune_args(model, data, name, *, engine, path="streamed", steps=50, lr=GATE_LR, batch=4,
              max_seq=256, rank=16, targets=None, state=None, stop_after=None, seed=None):
    a = ["tune", model, data, "--name", name, "--engine", engine, "--path", path,
         "--steps", steps, "--lr", lr, "--batch", batch, "--max-seq", max_seq, "--rank", rank,
         "--overwrite", "--quiet"]
    if targets:
        a += ["--targets", targets]
    if state:
        a += ["--state", state]
    if stop_after:
        a += ["--stop-after", stop_after]
    return a


# ---------------------------------------------------------------- comparisons

def loss_stats(a: list[float], b: list[float]) -> dict:
    n = min(len(a), len(b))
    rel = [abs(x - y) / max(abs(y), 1e-12) for x, y in zip(a[:n], b[:n])]
    return {"steps": n, "max_rel": max(rel) if rel else None,
            "mean_rel": float(np.mean(rel)) if rel else None,
            "worst_step": int(np.argmax(rel)) + 1 if rel else None}


def adapter_cosines(pa: dict, pb: dict) -> dict:
    from .tune.lora_core import cosine_np
    cos = {k: cosine_np(pa[k], pb[k]) for k in pa if k in pb}
    worst = min(cos, key=cos.get)
    return {"tensors": len(cos), "min": cos[worst], "mean": float(np.mean(list(cos.values()))),
            "worst": worst}


def compare_inference(a: dict, b: dict) -> dict:
    """a, b: {custom_id: CompletedRow} with log-probs. Greedy identity (same text and the same
    token ids) and the largest log-prob and top-k log-prob difference over every token."""
    same, max_lp, max_top, n_tok = 0, 0.0, 0.0, 0
    diffs = []
    for cid, ra in a.items():
        rb = b[cid]
        ta = [r["token_id"] for r in ra.logprobs or []]
        tb = [r["token_id"] for r in rb.logprobs or []]
        if ra.content == rb.content and ta == tb:
            same += 1
        else:
            diffs.append(cid)
        for x, y in zip(ra.logprobs or [], rb.logprobs or []):
            if x["token_id"] != y["token_id"]:
                break                                     # past the first divergence
            n_tok += 1
            max_lp = max(max_lp, abs(x["logprob"] - y["logprob"]))
            for (i1, l1), (i2, l2) in zip(x["top"], y["top"]):
                if i1 == i2:
                    max_top = max(max_top, abs(l1 - l2))
    return {"rows": len(a), "greedy_identical": same, "different_rows": diffs,
            "tokens_compared": n_tok, "max_logprob_diff": max_lp, "max_top_logprob_diff": max_top}


# ---------------------------------------------------------------- inference

def infer(engine: str, model_dir: Path, rows: list[dict], *, resident: bool, dtype: str | None,
          logprobs: int = 5, batch: int | None = None) -> tuple[dict, dict]:
    """Run `rows` through an engine in this process; ({custom_id: CompletedRow}, stats)."""
    from .engines.base import MemoryBudget, ModelSpec
    spec = ModelSpec("gate", "bf16", Path(model_dir), {}, 4096, extra={"logprobs": logprobs})
    if engine == "mlx":
        from .engines.mlx_resident import MlxResidentEngine
        from .engines.mlx_stream import MlxStreamEngine
        eng = MlxResidentEngine() if resident else MlxStreamEngine()
        ws = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * 0.75)
    else:
        from .engines.torch_common import memory_total_bytes
        from .engines.torch_resident import TorchResidentEngine
        from .engines.torch_stream import TorchEngine
        eng = (TorchResidentEngine if resident else TorchEngine)(engine=engine, dtype=dtype)
        ws = memory_total_bytes(engine)
    t0 = time.monotonic()
    out = {c.custom_id: c for c in eng.run_batch(rows, spec, MemoryBudget(ws, batch_override=batch))}
    secs = time.monotonic() - t0
    passes = sorted(getattr(eng, "last_pass_times", []) or [])
    stats = {"seconds": round(secs, 3), "passes": len(passes),
             "median_pass_s": round(passes[len(passes) // 2], 4) if passes else None,
             "completion_tokens": sum(c.completion_tokens for c in out.values())}
    prov = getattr(eng, "provider", None)
    ring = getattr(prov, "ring", None)
    if ring is not None and ring.read_seconds:
        stats["read_gb_s"] = round(ring.bytes_read / ring.read_seconds / 1e9, 3)
        stats["read_mode"] = ring.mode
    return out, stats


def gate_inference_stream_vs_resident(model_dir, rows, engine="torch-cpu", dtype="float32") -> dict:
    a, sa = infer(engine, model_dir, rows, resident=True, dtype=dtype)
    b, sb = infer(engine, model_dir, rows, resident=False, dtype=dtype)
    cmp = compare_inference(a, b)
    cmp["pass"] = cmp["greedy_identical"] == cmp["rows"] and cmp["max_logprob_diff"] <= 1e-5
    cmp.update(engine=engine, dtype=dtype, resident=sa, streamed=sb)
    return cmp


def gate_inference_vs_mlx(model_dir, rows, torch_engine="torch-cpu", dtype="float32") -> dict:
    t, st = infer(torch_engine, model_dir, rows, resident=True, dtype=dtype)
    m, sm = infer("mlx", model_dir, rows, resident=True, dtype=None)
    cmp = compare_inference(t, m)
    cmp["pass"] = cmp["greedy_identical"] == cmp["rows"]       # the log-prob difference is reported
    cmp.update(torch=st, mlx=sm, torch_engine=torch_engine, dtype=dtype)
    return cmp


# ---------------------------------------------------------------- tune

def run_tune_cli(work: Work, name: str, args: list, dtype: str | None = None) -> dict:
    t0 = time.monotonic()
    r = work.spill(args, dtype=dtype)
    if r.rc not in (0,):
        raise RuntimeError(f"spill tune failed (exit {r.rc}): {r.stderr[-400:]}")
    done = r.done or {}
    steps = [e["step"] for e in r.events if e["event"] == "step"]
    out = {"name": name, "job": done.get("job"), "seconds": round(time.monotonic() - t0, 2),
           "first_step_run": steps[0] if steps else None, "last_step_run": steps[-1] if steps else None,
           "events": r.events, "complete": done.get("complete")}
    if out["job"]:
        out["losses"] = work.job_losses(out["job"])
    return out


def gate_tune_identity(work: Work, model, data, *, a: dict, b: dict, steps=50, lr=GATE_LR,
                       loss_tol: float, cos_tol: float | None, dtype="float32",
                       tag: str = "g") -> dict:
    """Run two tunes that differ in engine/path and compare loss per step and adapter tensors.
    `a` and `b` are {"engine":, "path":}."""
    ra = run_tune_cli(work, f"{tag}-a", tune_args(model, data, f"{tag}-a", steps=steps, lr=lr, **a),
                      dtype)
    rb = run_tune_cli(work, f"{tag}-b", tune_args(model, data, f"{tag}-b", steps=steps, lr=lr, **b),
                      dtype)
    ls = loss_stats(ra["losses"], rb["losses"])
    cs = adapter_cosines(work.adapter_params(f"{tag}-a"), work.adapter_params(f"{tag}-b"))
    ok = ls["max_rel"] is not None and ls["max_rel"] <= loss_tol and (
        cos_tol is None or cs["min"] > cos_tol)
    return {"a": a, "b": b, "steps": steps, "lr": lr, "dtype": dtype, "loss": ls, "adapter": cs,
            "loss_tolerance": loss_tol, "cosine_floor": cos_tol, "pass": ok,
            "losses_a": ra["losses"], "losses_b": rb["losses"],
            "seconds": {"a": ra["seconds"], "b": rb["seconds"]}}


def eval_scores(work: Work, heldout: str, labels: list[str], engine: str, dtype: str | None = None
                ) -> dict[str, float]:
    """`spill eval` over the labels (model+adapter), exact match; {label: mean score}."""
    e = {"SPILL_ENGINE": engine}
    r = work.spill(["eval", heldout, *labels, "--rerun", "--quiet", "--engine", engine],
                   dtype=dtype, env=e)
    if r.rc != 0:
        raise RuntimeError(f"spill eval failed: {r.stderr[-400:]}")
    runs = work.home / "runs"
    dirs = sorted(d for d in runs.iterdir() if d.name.startswith("eval-"))
    table = (dirs[-1] / "table.md").read_text().splitlines()
    scores = {}
    for line in table:
        if line.startswith("| ") and not line.startswith("| model") and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            adapter = cells[2]
            key = next((l for l in labels if (l.endswith("+" + adapter) if adapter != "-" else "+" not in l)), None)
            if key:
                scores[key] = float(cells[4].split()[0])
    return scores


# ---------------------------------------------------------------- resume across engines

def noise_check(resumed: list[float], ref: list[float], ref_other: list[float], start: int) -> dict:
    """The resumed curve (steps start+1..) against the reference run, next to the distance
    between two clean runs on different engines (the noise): the resumed curve must be no
    farther from the reference than 1.5 x the clean pair, plus a 1% slack."""
    seg = slice(start, min(len(resumed), len(ref), len(ref_other)))
    def mean_rel(x, y):
        return float(np.mean([abs(p - q) / max(abs(q), 1e-12) for p, q in zip(x[seg], y[seg])]))
    d_resumed = mean_rel(resumed, ref)
    d_noise = mean_rel(ref_other, ref)
    return {"mean_rel_resumed_vs_reference": d_resumed, "mean_rel_clean_pair": d_noise,
            "limit": 1.5 * d_noise + 0.01, "pass": d_resumed <= 1.5 * d_noise + 0.01}


def gate_resume_tune(work: Work, model, data, *, first: str, then: str, steps=100, at=50,
                     lr=GATE_LR, tag="r", targets=None, rank=16, heldout=None, dtype_then=None,
                     score_labels_engine=None) -> dict:
    """Tune `first` for `at` steps and checkpoint, continue on `then` to `steps`, compare with an
    uninterrupted run on `first` (the reference) and one on `then` (the noise)."""
    state = str(work.root / f"state-{tag}-{first}-to-{then}")
    shutil.rmtree(state, ignore_errors=True)
    common = dict(steps=steps, lr=lr, rank=rank, targets=targets)
    part1 = run_tune_cli(work, "p1", tune_args(model, data, f"{tag}-p1", engine=first, state=state,
                                               stop_after=at, **common))
    part2 = run_tune_cli(work, "p2", tune_args(model, data, f"{tag}-res", engine=then, state=state,
                                               **common), dtype_then)
    ref = run_tune_cli(work, "ref", tune_args(model, data, f"{tag}-ref", engine=first, **common))
    other = run_tune_cli(work, "other", tune_args(model, data, f"{tag}-oth", engine=then, **common),
                         dtype_then)
    from .portable import checkpoint as pc
    from .portable.store import Store
    ck = pc.load_tune(Store(state))
    hist = ck.state["history"]
    res = {"first": first, "then": then, "stopped_at": part1["last_step_run"],
           "resumed_from": part2["first_step_run"] - 1, "final_step": part2["last_step_run"],
           "history": [{"range": h["range"], "engine": h["engine"], "hardware": h["hardware"],
                        "numerics": h["numerics"]["base"]} for h in hist],
           "curve": noise_check(part2["losses"], ref["losses"], other["losses"], at),
           "loss_first_segment_identical_to_reference": loss_stats(
               part2["losses"][:at], ref["losses"][:at])["max_rel"],
           "losses": {"resumed": part2["losses"], "reference": ref["losses"],
                      "other": other["losses"]}}
    res["pass"] = (res["curve"]["pass"] and res["resumed_from"] == at
                   and res["final_step"] == steps and len(hist) == 2)
    if heldout:
        eng = score_labels_engine or then
        base = Path(model).name if Path(model).is_dir() else model
        sc = eval_scores(work, heldout, [f"{model}+{tag}-res", f"{model}+{tag}-ref",
                                         f"{model}+{tag}-oth"], eng)
        r_, f_, o_ = (sc.get(f"{model}+{tag}-res"), sc.get(f"{model}+{tag}-ref"),
                      sc.get(f"{model}+{tag}-oth"))
        slack = max(abs((o_ or 0) - (f_ or 0)), 2 / 60)
        res["scores"] = {"resumed": r_, "reference": f_, "other_engine_clean": o_,
                         "limit_difference": slack,
                         "pass": r_ is not None and f_ is not None and abs(r_ - f_) <= slack}
        res["pass"] = res["pass"] and res["scores"]["pass"]
    return res


def gate_resume_rows(work: Work, heldout: str, model: str, *, first: str, then: str, half: int,
                     tag="rows") -> dict:
    """spill eval on `first`, stopped after `half` rows, finished on `then`: no row missing,
    none duplicated, each row stamped with the engine that produced it."""
    state = str(work.root / f"state-{tag}-{first}-to-{then}")
    shutil.rmtree(state, ignore_errors=True)
    r1 = work.spill(["eval", heldout, model, "--rerun", "--quiet", "--engine", first,
                     "--state", state, "--stop-after", half])
    audit1 = rows_audit(work, state, [])
    r2 = work.spill(["eval", heldout, model, "--rerun", "--quiet", "--engine", then,
                     "--state", state])
    ids = [json.loads(l)["custom_id"] for l in Path(heldout).read_text().splitlines() if l.strip()]
    audit = rows_audit(work, state, ids)
    audit.update(first=first, then=then, exit_codes=[r1.rc, r2.rc],
                 rows_after_first_leg=audit1["rows"], total=len(ids))
    audit["pass"] = (r1.rc == 0 and r2.rc == 0 and audit1["rows"] == half
                     and audit["rows"] == len(ids) and audit["unique"] == len(ids)
                     and not audit["missing"] and audit["duplicated"] == 0
                     and len(audit["by_engine"]) == 2)
    return audit


def rows_audit(work: Work, state_root: str, input_ids: list[str]) -> dict:
    """Read the committed segments under a state location (an eval keeps one sub-directory per
    model) and audit them against the input ids."""
    from .portable import rows as pr
    from .portable.store import Store
    root = Store(state_root)
    leaves = [n for n in root.ls() if root.exists(f"{n}/rows")]
    store = root.sub(leaves[0]) if leaves else root
    segs = pr.committed_segments(store)
    ids = [i for s in segs for i in s["ids"]]
    lines = []
    for s in segs:
        data = store.read(f"rows/seg-{s['n']:06d}.jsonl").decode().splitlines()
        lines += [json.loads(l) for l in data]
    engines = [(l["custom_id"], l["streamweights"]["engine"], l["streamweights"].get("hardware"),
                (l["streamweights"].get("numerics") or {}).get("base")) for l in lines]
    return {"segments": [{"rows": s["rows"], "engine": s["engine"], "hardware": s["hardware"],
                          "numerics": s["numerics"]} for s in segs],
            "rows": len(ids), "unique": len(set(ids)), "missing": sorted(set(input_ids) - set(ids)),
            "duplicated": len(ids) - len(set(ids)),
            "row_engines": sorted({e[1] for e in engines}),
            "by_engine": {k: sum(1 for e in engines if e[1] == k) for k in {e[1] for e in engines}}}
