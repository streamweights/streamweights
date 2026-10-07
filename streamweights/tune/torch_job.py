"""Tune jobs on the PyTorch engines: preparation (data, budget, pre-run numbers) and the
training loop, with the same `prepare` / `run_tune` signatures and the same Prepared and
checkpoint as tune.job (MLX), so the CLI and the portable state treat the engines alike.

Resident path   the model fits: PEFT's LoRA layers and autograd (PeftTrainer).
Streamed path   the model does not fit: saved layer inputs, reverse weight stream, recompute
                and VJP per layer (StreamedTrainer), two weight streams per micro-batch.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from ..engines.common import collect_eos_ids
from ..engines.torch_common import (device_for, device_label, linear_shapes_meta, load_tokenizer,
                                    numerics_for, peak_bytes, resolve_dtype)
from ..headless import QuantumAbandoned, protected
from ..jobs.engine import Job
from ..ring import SafetensorsIndex
from . import budget as bud
from . import lora_core as lo
from .checkpointer import Checkpointer
from .common import (ASSUMED_FLOPS, GIB, RESIDENT_FRACTION, Prepared, StepLog, _dir_bytes,
                     decide_path, with_stop_after, write_adapter)
from .data import BatchPlan, load_examples, steps_for
from .spec import TuneSpec
from .torch_train import AdamW, PeftTrainer, StreamedTrainer

__all__ = ["prepare", "run_tune", "decide_path", "TuneSpec"]


def _engine_flops(cal: dict, engine: str, key: str) -> tuple[float, str]:
    measured = (cal.get("tune_rates") or {}).get(key)
    if measured:
        return float(measured), f"measured for this model on {engine}"
    peak = (cal.get("engine_tflops") or {}).get(engine)
    if peak:
        return peak * 1e12 * 0.5, f"{engine} matmul rate x 0.5 until the first step measures it"
    return ASSUMED_FLOPS, f"assumed {ASSUMED_FLOPS / 1e12:.1f} TFLOP/s until the first step measures it"


def prepare(spec: TuneSpec, *, working_set: int, calibration: dict | None = None,
            hw: dict | None = None, micro_batch_given: bool = False,
            steps_given: bool = False) -> Prepared:
    """Tokenize, size, and put numbers on the run. Touches no weights."""
    spec.lora().validate()
    d = Path(spec.model_dir)
    index = SafetensorsIndex(d)
    config = index.config
    tok = load_tokenizer(d)
    eos = collect_eos_ids(d, tok)
    examples, stats = load_examples(Path(spec.data), tok, spec.max_seq, eos)
    pad = tok.pad_token_id
    pad = pad if pad is not None else (min(eos) if eos else 0)

    shapes = linear_shapes_meta(index, spec.targets)
    L = index.n_layers
    lora_params = L * sum(spec.rank * (i + o) for i, o in shapes.values())
    size = _dir_bytes(d)
    dtype, dtype_note = resolve_dtype(spec.engine)
    elt = 2 if str(dtype).endswith("bfloat16") else 4

    p = Prepared(spec, config, tok, examples, stats, None, shapes, size, L, lora_params,
                 working_set=working_set)
    seq = max(1, 32 * ((stats.max_len - 1 + 31) // 32))
    cal = calibration or {}
    layer_params = sum(t.nbytes for t in index.layers[0].tensors) // 2 * L
    fkey = f"{spec.model}|{spec.engine}|{spec.path}"
    flops, flops_note = _engine_flops(cal, spec.engine, fkey)

    if spec.path == "streamed":
        resident = (index.embed.nbytes + index.final_norm.nbytes
                    + (0 if index.tied else index.lm_head.nbytes)) * elt // 2
        # activations scale with the compute dtype: float32 doubles the bf16-based formula
        b = bud.compute_micro_batch(
            working_set=working_set * 2 // elt, config=config,
            max_layer_bytes=index.max_layer_bytes * elt // 2, resident_bytes=resident,
            lora_param_bytes=lora_params * 4, seq=seq, n_examples=len(examples),
            override=spec.micro_batch if micro_batch_given else None)
        spec.micro_batch = b.micro_batch
        p.budget = b
        p.why = b.reason
    plan = BatchPlan(examples, spec.micro_batch, spec.seed, pad)
    p.plan = plan
    if not steps_given:
        spec.steps = steps_for(plan.per_epoch, spec.grad_accum, spec.epochs)

    mean_tokens = stats.tokens / max(1, stats.examples)
    mb_tokens = spec.micro_batch * mean_tokens
    compute_s = bud.estimate_compute_seconds(layer_params, mb_tokens, flops)
    if spec.path == "streamed":
        from ..policy import engine_read_rate
        rate, src = engine_read_rate(cal, hw or {"nvme_seq_read": {"bytes_per_sec": 2e9}},
                                     key=f"{spec.model}|{spec.quant}")
        pass_s = size / rate
        p.est_step_s = bud.estimate_step(pass_s=pass_s, compute_s=compute_s,
                                         grad_accum=spec.grad_accum)
        p.est_note = (f"{spec.grad_accum} x (2 weight streams x {pass_s:.0f} s at "
                      f"{rate / GIB:.1f} GB/s ({src}) + {compute_s:.0f} s compute, {flops_note})")
    else:
        compute_s = compute_s * 4 / 6
        p.est_step_s = spec.grad_accum * compute_s
        p.est_note = f"{spec.grad_accum} x {compute_s:.1f} s compute, {flops_note}"
        p.why = (f"resident: {size / GIB:.1f} GB of weights is under "
                 f"{int(RESIDENT_FRACTION * 100)}% of the {working_set / GIB:.0f} GB working set; "
                 f"PEFT LoRA layers and autograd")
    p.est_total_s = p.est_step_s * spec.steps
    p.dtype_note = dtype_note
    return p


def _record_compute_rate(prep: Prepared, result: dict) -> None:
    from ..calibration import load_calibration, save_calibration
    spec, ss = prep.spec, prep.stream_stats
    try:
        ix = SafetensorsIndex(Path(spec.model_dir))
        n = sum(t.nbytes for t in ix.layers[0].tensors) // 2 * prep.n_layers
        if ss.get("micro_tokens") and ss["micro_seconds"] > 0:
            compute = ss["micro_seconds"] - ss.get("wait_seconds", 0.0)
            if compute <= 0:
                return
            mult = 6.0 if spec.path == "streamed" else 4.0
            rate = mult * n * ss["micro_tokens"] / compute
            cal = load_calibration()
            cal.setdefault("tune_rates", {})[f"{spec.model}|{spec.engine}|{spec.path}"] = round(rate)
            save_calibration(cal)
            result["flops_rate"] = round(rate)
            result["tflops_compute"] = rate / 1e12
    except Exception:
        pass


def run_tune(prep: Prepared, job: Job, *, stop=None, progress_cb=None, resume: bool = False,
             note=None, on_event=None) -> dict:
    """Train to prep.spec.steps (or until `stop`), checkpointing; write the adapter."""
    spec = prep.spec
    stop = stop if stop is not None else threading.Event()
    progress_cb = with_stop_after(spec, stop, progress_cb)
    dev = device_for(spec.engine)
    dtype, _ = resolve_dtype(spec.engine)
    streamed = spec.path == "streamed"
    prep.cp = Checkpointer(
        spec, job.dir, engine="torch_stream_tune" if streamed else "torch_peft_lora",
        device=device_label(spec.engine),
        numerics={**numerics_for(dtype), "dropout": spec.dropout},
        targets=sorted(prep.shapes), on_event=on_event)
    start, init = 0, None
    ck = prep.cp.load(want=resume)
    if ck is not None:
        start, init = ck.step, ck
        prep.cp.sync_losses(start)
    job.write_meta(status="running", done=start, total=spec.steps)
    log = StepLog(job, spec.steps, start, progress_cb, peak_fn=lambda: peak_bytes(dev) / GIB)
    cfg = spec.lora()
    shapes = prep.shapes
    model_dir = Path(spec.model_dir)
    if streamed:
        tr = StreamedTrainer(model_dir, cfg, init.params if init else None, dtype=dtype,
                             device=dev, shapes=shapes, resident_weights=spec.resident_weights,
                             note=note)
    else:
        tr = PeftTrainer(model_dir, cfg, init.params if init else None, dtype=dtype, device=dev,
                         shapes=shapes, note=note)
    opt = AdamW(tr.params, cfg.lr, spec.steps, cfg.schedule, cfg.weight_decay)
    if init:
        opt.load(init.opt)
    t0 = time.monotonic()
    step = start

    def save():
        prep.cp.save(step, tr.params, opt.state())

    try:
        log.reset_clock()
        while step < spec.steps:
            acc, losses, toks = None, [], 0
            try:
                for a in range(spec.grad_accum):
                    m = step * spec.grad_accum + a
                    b = prep.plan.batch(m)
                    loss, grads, _ = tr.micro_batch(b.inputs, b.targets, b.mask, m)
                    losses.append(loss)
                    toks += b.n_tokens
                    acc = grads if acc is None else {k: acc[k] + grads[k] for k in acc}
                if spec.grad_accum > 1:
                    acc = {k: v / spec.grad_accum for k, v in acc.items()}
                with protected():
                    opt.apply(tr.params, acc)
                    step += 1
            except QuantumAbandoned:
                save()
                prep.abandoned = True
                break
            log.step(step, sum(losses) / len(losses), toks)
            stopping = stop.is_set()
            if step % spec.ckpt_every == 0 or stopping or step == spec.steps:
                with protected():
                    save()
            if stopping:
                break
        ring = getattr(tr, "ring", None)
        prep.stream_stats = {
            "read_bytes": ring.bytes_read if ring else 0,
            "read_seconds": ring.read_seconds if ring else 0.0,
            "bind_seconds": tr.bind_seconds, "wait_seconds": tr.wait_seconds,
            "micro_seconds": tr.micro_seconds, "micro_tokens": tr.micro_tokens}
    except BaseException:
        job.write_meta(status="failed")
        raise
    finally:
        log.close()
        tr.close()
    losses_all = [json.loads(l)["loss"] for l in (job.dir / "losses.jsonl").read_text().splitlines()
                  if l.strip()]
    interrupted = step < spec.steps
    result = {"step": step, "steps": spec.steps, "interrupted": interrupted,
              "first_loss": losses_all[0] if losses_all else None,
              "final_loss": losses_all[-1] if losses_all else None,
              "seconds": time.monotonic() - t0, "peak_gb": peak_bytes(dev) / GIB,
              "abandoned": prep.abandoned}
    job.write_meta(status="interrupted" if interrupted else "completed", done=step)
    _record_compute_rate(prep, result)
    if not interrupted:
        result["adapter"] = str(write_adapter(prep, tr.params, step, job, spec.model,
                                              result["final_loss"]))
    return result
