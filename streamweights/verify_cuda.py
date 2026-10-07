"""Verify the CUDA engine on a CUDA machine.

  docker run --gpus all ghcr.io/streamweights/spill:cuda python -m streamweights.verify_cuda
  python -m streamweights.verify_cuda [--out report.json] [--work DIR] [--quick]

Runs, on qwen2.5:0.5b (about 1 GB, downloaded once):

  * the identity gates, torch-cuda against the torch-cpu and resident references:
      inference, streamed against resident on the GPU (greedy identical on 20 prompts, log-probs
      within 1e-5), and the GPU against the CPU (greedy identical; the log-prob gap is reported);
      tune, streamed against PEFT resident on the GPU (loss within 0.1% per step, adapter
      cosine above 0.9999), and the GPU against the CPU (loss within 1%), 50 steps each
  * the resume gate: the committed step-50 MLX checkpoint continued on torch-cuda
  * a streamed-inference throughput measurement (pass time, achieved read rate, tokens/s)
  * a streamed-tune step-time measurement

It prints a pass/fail report and writes a JSON file (default $SPILL_HOME/verify_cuda.json).
Exit status 0 when every gate passed. The numbers it prints are the CUDA numbers the
documentation is waiting for.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _gpu() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {}
    i = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(i)
    return {"name": p.name, "memory_gb": round(p.total_memory / 1024**3, 1),
            "capability": f"{p.major}.{p.minor}", "torch": torch.__version__,
            "cuda": torch.version.cuda, "bf16": torch.cuda.is_bf16_supported()}


class Report:
    def __init__(self):
        self.gates: dict[str, dict] = {}

    def run(self, name: str, fn):
        t0 = time.monotonic()
        print(f"[ .. ] {name}", flush=True)
        try:
            res = fn()
            ok = bool(res.get("pass", True)) if isinstance(res, dict) else True
        except Exception as e:                                    # a gate that crashes is a fail
            res, ok = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}, False
        res = res if isinstance(res, dict) else {"result": res}
        res["seconds"] = round(time.monotonic() - t0, 1)
        self.gates[name] = {"pass": ok, **res}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({res['seconds']} s)", flush=True)
        for line in _summary(name, res):
            print("       " + line, flush=True)
        return res


def _summary(name: str, r: dict) -> list[str]:
    out = []
    if "error" in r:
        out.append(r["error"])
    for k in ("greedy_identical", "rows", "max_logprob_diff", "max_top_logprob_diff"):
        if k in r:
            out.append(f"{k}: {r[k]}")
    if "loss" in r and isinstance(r["loss"], dict):
        out.append(f"loss max relative difference: {r['loss']['max_rel']:.3g} "
                   f"(tolerance {r['loss_tolerance']})")
        out.append(f"adapter cosine min {r['adapter']['min']:.7f}")
    for k in ("first_step", "last_step", "mean_rel_loss_difference", "seconds_per_step",
              "median_pass_s", "read_gb_s", "tokens_per_s", "pass_seconds", "step_seconds_median"):
        if k in r and r[k] is not None:
            out.append(f"{k}: {r[k]}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m streamweights.verify_cuda",
                                 description="Verify the torch-cuda engine on this machine.")
    ap.add_argument("--out", help="JSON report path (default: $SPILL_HOME/verify_cuda.json)")
    ap.add_argument("--work", help="working directory (default: $SPILL_HOME/verify_cuda)")
    ap.add_argument("--quick", action="store_true", help="20 tune steps instead of 50")
    args = ap.parse_args(argv)

    import torch
    if not torch.cuda.is_available():
        print("spill: no CUDA device is visible. Try: docker run --gpus all "
              "ghcr.io/streamweights/spill:cuda python -m streamweights.verify_cuda", file=sys.stderr)
        return 2
    from . import gates as G
    from .registry import REPO_ROOT, download_safetensors
    from .tune import toy

    gpu = _gpu()
    print(f"CUDA device: {gpu['name']} ({gpu['memory_gb']} GB, capability {gpu['capability']}), "
          f"torch {gpu['torch']}, CUDA {gpu['cuda']}")
    work_root = Path(args.work) if args.work else REPO_ROOT / "verify_cuda"
    work = G.Work(work_root, models_from=REPO_ROOT / "models" if (REPO_ROOT / "models").exists() else None,
                  state_from=REPO_ROOT / "state")
    rep = Report()
    steps = 20 if args.quick else 50

    bf16_dir = download_safetensors("qwen2.5:0.5b")
    f32_dir = work_root / "qwen2.5-0.5b-f32"
    if not (f32_dir / "config.json").exists():
        print("writing a float32 copy of the model for the identity gates ...", flush=True)
        G.make_f32_copy(bf16_dir, f32_dir)
    from .formats import load_rows
    sample = Path(__file__).parent / "data" / "sample-20.jsonl"
    rows = load_rows(sample)
    natural_rows = Path(__file__).parent / "data" / "gate-natural.jsonl"
    data = str(natural_rows if natural_rows.exists() else toy.make(work_root / "toy", 200, 60)["train"])
    lr = G.GATE_LR if natural_rows.exists() else G.TOY_LR
    cuda, cpu = "torch-cuda", "torch-cpu"

    rep.run("inference: torch-cuda streamed vs resident (float32)",
            lambda: G.gate_inference_stream_vs_resident(f32_dir, rows, engine=cuda))

    def gpu_vs_cpu():
        a, _ = G.infer(cuda, f32_dir, rows, resident=False, dtype="float32")
        b, _ = G.infer(cpu, f32_dir, rows, resident=True, dtype="float32")
        c = G.compare_inference(a, b)
        c["pass"] = c["greedy_identical"] == c["rows"]
        return c
    rep.run("inference: torch-cuda vs torch-cpu reference (float32)", gpu_vs_cpu)

    rep.run("tune: torch-cuda streamed vs PEFT resident (float32)",
            lambda: G.gate_tune_identity(
                work, str(f32_dir), data, a={"engine": cuda, "path": "streamed"},
                b={"engine": cuda, "path": "resident"}, steps=steps, lr=lr, loss_tol=1e-3,
                cos_tol=0.9999, tag="vc1"))
    rep.run("tune: torch-cuda vs torch-cpu reference (float32)",
            lambda: G.gate_tune_identity(
                work, str(f32_dir), data, a={"engine": cuda, "path": "streamed"},
                b={"engine": cpu, "path": "streamed"}, steps=steps, lr=lr, loss_tol=0.01,
                cos_tol=None, tag="vc2"))

    rep.run("resume: the committed MLX step-50 checkpoint continued on torch-cuda",
            lambda: G.gate_resume_fixture(work, cuda))

    def throughput():
        many = [dict(r, custom_id=f"{r['custom_id']}-{k}") for k in range(8) for r in rows]
        for r in many:
            r["body"] = dict(r["body"], max_tokens=64)
        out, st = G.infer(cuda, bf16_dir, many, resident=False, dtype=None, logprobs=1)
        toks = sum(c.completion_tokens for c in out.values())
        return {"pass": True, "rows": len(many), "completion_tokens": toks,
                "seconds": st["seconds"], "tokens_per_s": round(toks / st["seconds"], 1),
                "median_pass_s": st["median_pass_s"], "passes": st["passes"],
                "read_gb_s": st.get("read_gb_s"), "read_mode": st.get("read_mode"),
                "model_gb": round(sum(f.stat().st_size for f in Path(bf16_dir).glob("*.safetensors")) / 1e9, 2),
                "base_dtype": "bf16"}
    rep.run("throughput: streamed inference on the GPU", throughput)

    def tune_step_time():
        r = G.run_tune_cli(work, "vc-time", G.tune_args(
            str(bf16_dir), data, "vc-time", engine=cuda, path="streamed", steps=12, lr=lr))
        st = [e["step_s"] for e in r["events"] if e["event"] == "step"][2:]
        return {"pass": True, "step_seconds_median": round(float(np.median(st)), 3),
                "step_seconds_all": [round(x, 3) for x in st], "micro_batch": 4, "base_dtype": "bf16"}
    rep.run("throughput: streamed tune step time on the GPU", tune_step_time)

    ok = all(g["pass"] for g in rep.gates.values())
    out = Path(args.out) if args.out else REPO_ROOT / "verify_cuda.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pass": ok, "gpu": gpu, "platform": platform.platform(),
                               "python": sys.version.split()[0], "gates": rep.gates},
                              indent=1, default=str))
    print()
    print(f"{'ALL GATES PASSED' if ok else 'SOME GATES FAILED'}: "
          f"{sum(g['pass'] for g in rep.gates.values())}/{len(rep.gates)}; report -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
