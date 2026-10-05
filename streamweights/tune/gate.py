"""The training identity gate: streamed LoRA versus mlx-lm's resident LoRA, same data,
seed, init and hyperparameters. Pass means:

  1. per-step loss agrees within 1% relative after step 1 (bf16 reorder noise),
  2. every adapter tensor has cosine similarity above 0.999 between the two,
  3. both adapters give identical greedy output through the inference engines.

`run_gate` is used by tests (tiny model, CPU), scripts/verify_phase3.py (Qwen2.5-0.5B,
Metal), and the GPU test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..jobs.engine import Job
from . import job as tj
from . import lora as lo
from . import toy

LOSS_TOL = 0.01
COS_MIN = 0.999


def _train(model_dir: Path, data: Path, name: str, path: str, steps: int, working_set: int,
           **kw) -> tuple[list[float], dict]:
    spec = tj.TuneSpec(model=kw.pop("label", "gate-model"), quant="bf16",
                       model_dir=str(model_dir), data=str(data), name=name, path=path,
                       steps=steps, overwrite=True, **kw)
    prep = tj.prepare(spec, working_set=working_set, micro_batch_given=True, steps_given=True)
    job = Job.create(data, spec.model, "bf16", spec.max_seq, spec.micro_batch, None,
                     options={"kind": "tune", "tune": spec.to_dict()})
    res = tj.run_tune(prep, job)
    losses = [json.loads(l)["loss"] for l in (job.dir / "losses.jsonl").read_text().splitlines()]
    return losses, res


def greedy_outputs(model_dir: Path, adapter: str, rows: list[dict], working_set: int,
                   engine: str = "resident") -> dict[str, str]:
    from ..adapters import resolve_adapter
    from ..engines.base import MemoryBudget, ModelSpec
    from ..engines.mlx_resident import MlxResidentEngine
    from ..engines.mlx_stream import MlxStreamEngine
    cls = MlxResidentEngine if engine == "resident" else MlxStreamEngine
    spec = ModelSpec("gate-model", "bf16", Path(model_dir), {}, 4096,
                     extra={"adapter": resolve_adapter(adapter)})
    return {c.custom_id: c.content for c in cls().run_batch(rows, spec, MemoryBudget(working_set))}


def run_gate(model_dir: Path, work: Path, *, working_set: int, steps: int = 100,
             n_prompts: int = 20, micro_batch: int = 4, grad_accum: int = 1, rank: int = 16,
             alpha: float = 32.0, lr: float = 1e-4, seed: int = 0, max_seq: int = 256,
             n_train: int = 200, label: str = "gate-model", natural: Path | None = None,
             note=print) -> dict:
    """natural: a chat JSONL of natural-text targets (examples/gate-natural.jsonl). The
    toy task saturates within a few steps (loss ~1e-3), where a relative-loss gate
    measures noise; natural text keeps the loss in a regime the gate can speak to.
    The first n_train rows train; the next n_prompts rows' prompts are the greedy check."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if natural:
        lines = [l for l in Path(natural).read_text().splitlines() if l.strip()]
        (work / "train.jsonl").write_text("\n".join(lines[:n_train]) + "\n")
        with open(work / "heldout.jsonl", "w") as f:
            for i, l in enumerate(lines[n_train:n_train + n_prompts]):
                msgs = json.loads(l)["messages"][:-1]
                f.write(json.dumps({"custom_id": f"held-{i:03d}",
                                    "body": {"messages": msgs, "max_tokens": 24}}) + "\n")
        files = {"train": str(work / "train.jsonl"), "heldout": str(work / "heldout.jsonl")}
    else:
        files = toy.make(work, n_train=n_train, n_heldout=max(n_prompts, 20))
    common = dict(rank=rank, alpha=alpha, lr=lr, seed=seed, micro_batch=micro_batch,
                  grad_accum=grad_accum, max_seq=max_seq, ckpt_every=50, label=label)
    t0 = time.monotonic()
    note(f"gate: resident (mlx-lm) {steps} steps")
    lr_res, r_res = _train(model_dir, Path(files["train"]), "gate-resident", "resident",
                           steps, working_set, **common)
    note(f"gate: streamed {steps} steps ({time.monotonic() - t0:.0f} s so far)")
    lr_str, r_str = _train(model_dir, Path(files["train"]), "gate-streamed", "streamed",
                           steps, working_set, **common)
    rel = [abs(a - b) / max(abs(a), 1e-12) for a, b in zip(lr_res, lr_str)]
    after_first = rel[1:]
    from ..adapters import ADAPTERS_DIR
    pr = lo.read_adapter_params(ADAPTERS_DIR / "gate-resident")
    ps = lo.read_adapter_params(ADAPTERS_DIR / "gate-streamed")
    cos = {k: lo.cosine_sim(pr[k], ps[k]) for k in pr}
    # B starts at zero and A at the shared init, so cosine over A is trivially near 1;
    # report A and B separately and gate on all of them
    cos_b = [v for k, v in cos.items() if k.endswith("lora_b")]
    cos_a = [v for k, v in cos.items() if k.endswith("lora_a")]
    rows = [json.loads(l) for l in open(files["heldout"])][:n_prompts]
    note("gate: greedy output through the inference engines")
    out_r = greedy_outputs(model_dir, "gate-resident", rows, working_set)
    out_s = greedy_outputs(model_dir, "gate-streamed", rows, working_set)
    out_x = greedy_outputs(model_dir, "gate-streamed", rows, working_set, engine="streamed")
    same = sum(out_r[k] == out_s[k] for k in out_r)
    result = {
        "steps": steps, "micro_batch": micro_batch, "grad_accum": grad_accum,
        "loss_resident": lr_res, "loss_streamed": lr_str,
        "loss_first": [lr_res[0], lr_str[0]],
        "loss_rel_diff_max_after_step1": max(after_first) if after_first else 0.0,
        "loss_rel_diff_mean_after_step1": sum(after_first) / max(1, len(after_first)),
        "loss_rel_diff_step1": rel[0],
        "cos_min_all": min(cos.values()), "cos_min_b": min(cos_b), "cos_mean_b": sum(cos_b) / len(cos_b),
        "cos_min_a": min(cos_a), "n_tensors": len(cos),
        "greedy_identical": same, "greedy_total": len(out_r),
        "greedy_identical_engine_stream_vs_resident": sum(out_s[k] == out_x[k] for k in out_s),
        "seconds_resident": r_res["seconds"], "seconds_streamed": r_str["seconds"],
        "peak_gb_streamed": r_str["peak_gb"],
        "sample_outputs": {k: [out_r[k], out_s[k]] for k in list(out_r)[:5]},
    }
    result["pass"] = bool(
        result["loss_rel_diff_max_after_step1"] <= LOSS_TOL
        and result["cos_min_all"] > COS_MIN
        and same == len(out_r))
    return result
