"""Tune jobs: preparation (data, budget, pre-run numbers), the two training paths,
checkpoint/resume, and writing the finished adapter.

Resident path   the model fits: mlx-lm's own LoRA tuner (`mlx_lm.tuner.trainer.train`,
                `linear_to_lora_layers`) runs the loop, in segments between
                checkpoints, fed by our batches and our masked loss.
Streamed path   the model does not fit: tune.streamed.StreamedTrainer, two weight
                streams per micro-batch.

Both consume the same BatchPlan (so the same micro-batches in the same order),
start from the same init_params(seed), use the same AdamW and schedule, and
define a step's loss the same way (mean of its micro-batches' token-mean losses).
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .. import adapters as adapters_mod
from ..errors import SpillError
from ..jobs.engine import Job
from . import budget as bud
from . import lora as lo
from .checkpointer import Checkpointer
from .data import BatchPlan, DataStats, Example, load_examples, steps_for
from .spec import TuneSpec, data_hash  # noqa: F401
from .streamed import StreamedTrainer, make_optimizer, masked_ce

GIB = 1024**3
ASSUMED_FLOPS = 3.5e12          # effective bf16 flops/s until a run measures it
RESIDENT_FRACTION = 0.35        # weights must be under this share of the working set


@dataclass
class Prepared:
    spec: TuneSpec
    config: dict
    tokenizer: object
    examples: list[Example]
    stats: DataStats
    plan: BatchPlan
    shapes: dict
    size_bytes: int
    n_layers: int
    lora_params: int
    budget: bud.TuneBudget | None = None
    est_step_s: float = 0.0
    est_total_s: float = 0.0
    est_note: str = ""
    why: str = ""
    working_set: int = 0
    stream_stats: dict = field(default_factory=dict)
    cp: Checkpointer | None = None


def adapters_dir() -> Path:
    return adapters_mod.ADAPTERS_DIR


def _dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in Path(d).glob("*.safetensors"))


def decide_path(size_bytes: int, working_set: int, forced: str = "auto") -> str:
    if forced in ("resident", "streamed"):
        return forced
    return "resident" if size_bytes <= RESIDENT_FRACTION * working_set else "streamed"


def prepare(spec: TuneSpec, *, working_set: int, calibration: dict | None = None,
            hw: dict | None = None, micro_batch_given: bool = False,
            steps_given: bool = False) -> Prepared:
    """Tokenize, size, and put numbers on the run. Touches no weights."""
    from mlx_lm.utils import load_tokenizer

    from ..engines.mlx_stream import (MB, SafetensorsIndex, _model_modules,
                                      collect_eos_ids)
    spec.lora().validate()
    d = Path(spec.model_dir)
    index = SafetensorsIndex(d)
    config = index.config
    tok = load_tokenizer(d)
    eos = collect_eos_ids(d, tok)
    examples, stats = load_examples(Path(spec.data), tok, spec.max_seq, eos)
    pad = getattr(tok, "pad_token_id", None)
    pad = pad if pad is not None else (min(eos) if eos else 0)

    block, _, fam = _model_modules(config)
    shapes = lo.linear_shapes(block, spec.targets)
    L = index.n_layers
    lora_params = L * sum(spec.rank * (i + o) for i, o in shapes.values())
    size = _dir_bytes(d)

    p = Prepared(spec, config, tok, examples, stats, None, shapes, size, L, lora_params,
                 working_set=working_set)
    seq = max(1, 32 * ((stats.max_len - 1 + 31) // 32))
    cal = calibration or {}
    layer_params = sum(t.nbytes for t in index.layers[0].tensors) // 2 * L   # bf16 elements
    fkey = f"{spec.model}|{spec.path}"       # utilization depends on model size
    flops = cal.get("tune_rates", {}).get(fkey) or ASSUMED_FLOPS
    flops_note = f"measured for {spec.model}, {spec.path}" if cal.get("tune_rates", {}).get(fkey) else \
        f"assumed {ASSUMED_FLOPS / 1e12:.1f} TFLOP/s until the first step measures it"

    if spec.path == "streamed":
        resident = index.embed.nbytes + index.final_norm.nbytes + (
            0 if index.tied else index.lm_head.nbytes)
        b = bud.compute_micro_batch(
            working_set=working_set, config=config, max_layer_bytes=index.max_layer_bytes,
            resident_bytes=resident, lora_param_bytes=lora_params * 4, seq=seq,
            n_examples=len(examples), override=spec.micro_batch if micro_batch_given else None)
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
        rate, src = engine_read_rate(cal, hw or {"nvme_seq_read": {"bytes_per_sec": 5e9}},
                                     key=f"{spec.model}|{spec.quant}")
        pass_s = size / rate
        p.est_step_s = bud.estimate_step(pass_s=pass_s, compute_s=compute_s,
                                         grad_accum=spec.grad_accum)
        p.est_note = (f"{spec.grad_accum} x (2 weight streams x {pass_s:.0f} s at "
                      f"{rate / GIB:.1f} GB/s ({src}) + {compute_s:.0f} s compute, {flops_note})")
    else:
        compute_s = compute_s * 4 / 6          # no recompute, no second stream
        p.est_step_s = spec.grad_accum * compute_s
        p.est_note = (f"{spec.grad_accum} x {compute_s:.1f} s compute, {flops_note}")
        p.why = (f"resident: {size / GIB:.1f} GB of weights is under "
                 f"{int(RESIDENT_FRACTION * 100)}% of the {working_set / GIB:.0f} GB working set; "
                 f"mlx-lm's LoRA tuner")
    p.est_total_s = p.est_step_s * spec.steps
    return p


# ------------------------------------------------------------ progress

class StepLog:
    """Per-step bookkeeping: losses.jsonl, live.json, the progress callback."""

    def __init__(self, job: Job, steps: int, start: int, cb, ema: float = 0.3):
        self.job, self.steps, self.cb, self.ema = job, steps, cb, ema
        self.t_last = time.monotonic()
        self.avg = None
        self.start = start
        self.tokens = 0
        self.t0 = time.monotonic()
        self.f = open(job.dir / "losses.jsonl", "a")

    def close(self):
        self.f.close()

    def reset_clock(self):
        self.t_last = time.monotonic()

    def step(self, step: int, loss: float, tokens: int):
        now = time.monotonic()
        dt = now - self.t_last
        self.t_last = now
        self.avg = dt if self.avg is None else self.ema * dt + (1 - self.ema) * self.avg
        self.tokens += tokens
        peak = mx.get_peak_memory() / GIB if mx.default_device() == mx.gpu else 0.0
        info = {"step": step, "steps": self.steps, "loss": loss, "step_s": dt,
                "avg_step_s": self.avg, "eta_s": (self.steps - step) * self.avg,
                "peak_gb": peak, "tokens": tokens,
                "tok_s": self.tokens / max(1e-9, now - self.t0)}
        self.f.write(json.dumps({k: info[k] for k in ("step", "loss", "step_s", "peak_gb")}) + "\n")
        self.f.flush()
        try:
            (self.job.dir / "live.json").write_text(json.dumps(info))
        except OSError:
            pass
        self.job.write_meta(done=step, total=self.steps, eta_seconds=round(info["eta_s"]),
                            tokens_per_sec=round(info["tok_s"], 2))
        if self.cb:
            self.cb(info)
        return info


def truncate_losses(job: Job, step: int) -> None:
    p = job.dir / "losses.jsonl"
    if not p.exists():
        return
    keep = [l for l in p.read_text().splitlines() if l.strip() and json.loads(l)["step"] <= step]
    p.write_text("".join(l + "\n" for l in keep))


# ------------------------------------------------------------ streamed path

def _opt_to_canon(opt_obj, step_count: int) -> dict:
    return {"step": int(opt_obj.step.item()),
            "m": {k: v["m"] for k, v in opt_obj.state.items() if isinstance(v, dict) and "m" in v},
            "v": {k: v["v"] for k, v in opt_obj.state.items() if isinstance(v, dict) and "v" in v}}


def _opt_from_canon(opt_obj, params: dict, canon: dict) -> None:
    opt_obj.init(params)
    for k in params:
        opt_obj.state[k]["m"] = canon["m"][k]
        opt_obj.state[k]["v"] = canon["v"][k]
    opt_obj.state["step"] = mx.array(canon["step"], mx.uint64)
    mx.eval(opt_obj.state)


def train_streamed(prep: Prepared, job: Job, log: StepLog, stop, start_step: int,
                   init: tuple | None, note=None) -> tuple[dict, int]:
    spec = prep.spec
    cfg = spec.lora()
    tr = StreamedTrainer(Path(spec.model_dir), cfg,
                         params=init[0] if init else None,
                         resident_weights=spec.resident_weights, note=note)
    opt = make_optimizer(cfg, spec.steps)
    if init:
        _opt_from_canon(opt, tr.params, init[1])
    step = start_step
    try:
        log.reset_clock()
        while step < spec.steps:
            acc, losses, toks = None, [], 0
            for a in range(spec.grad_accum):
                m = step * spec.grad_accum + a
                b = prep.plan.batch(m)
                loss, grads, _ = tr.micro_batch(b.inputs, b.targets, b.mask, m)
                losses.append(loss)
                toks += b.n_tokens
                acc = grads if acc is None else {k: acc[k] + grads[k] for k in acc}
                mx.eval(acc)
            if spec.grad_accum > 1:
                acc = {k: v / spec.grad_accum for k, v in acc.items()}
            tr.apply(opt, acc)
            step += 1
            log.step(step, sum(losses) / len(losses), toks)
            stopping = stop is not None and stop.is_set()
            if step % spec.ckpt_every == 0 or stopping or step == spec.steps:
                _save_ckpt(prep, step, tr.params, _opt_to_canon(opt, step))
            if stopping:
                break
        prep.stream_stats = {"read_bytes": tr.ring.bytes_read if tr.ring else 0,
                             "read_seconds": tr.ring.read_seconds if tr.ring else 0.0,
                             "bind_seconds": tr.bind_seconds, "wait_seconds": tr.wait_seconds,
                             "micro_seconds": tr.micro_seconds, "micro_tokens": tr.micro_tokens}
    finally:
        tr.close()
    return tr.params, step


# ------------------------------------------------------------ resident path

class _NullUI:
    """Stands in for mlx-lm's rich progress UI: our single progress line replaces it."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def advance(self):
        pass

    def val_task(self, *a, **k):
        import contextlib
        return contextlib.nullcontext(lambda: None)

    def report_val(self, *a, **k):
        pass

    def report_train(self, *a, **k):
        pass

    def report_save(self, *a, **k):
        pass


def _flat_trainable(model) -> tuple[dict, str]:
    """({canonical 'layers.k...' name: array}, prefix before 'layers.')."""
    out, prefix = {}, ""
    for k, v in tree_flatten(model.trainable_parameters()):
        i = k.index("layers.")
        prefix = k[:i]
        out[k[i:]] = v
    return out, prefix


def train_resident(prep: Prepared, job: Job, log: StepLog, stop, start_step: int,
                   init: tuple | None, note=None) -> tuple[dict, int]:
    import mlx_lm.tuner.trainer as T
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load as mlx_load

    spec = prep.spec
    cfg = spec.lora()
    model, _ = mlx_load(spec.model_dir)
    model.freeze()
    keys = list(prep.shapes)
    linear_to_lora_layers(model, prep.n_layers,
                          {"rank": cfg.rank, "scale": cfg.scale, "dropout": cfg.dropout,
                           "keys": keys})
    _, prefix = _flat_trainable(model)
    params = init[0] if init else lo.init_params(prep.shapes, prep.n_layers, cfg.rank, cfg.seed)
    model.load_weights([(prefix + k, v) for k, v in params.items()], strict=False)
    opt = make_optimizer(cfg, spec.steps)
    if init:
        opt.init(model.trainable_parameters())
        items = [(f"{prefix}{k}.m", v) for k, v in init[1]["m"].items()]
        items += [(f"{prefix}{k}.v", v) for k, v in init[1]["v"].items()]
        st = tree_unflatten(items)
        st["step"] = mx.array(init[1]["step"], mx.uint64)
        st["learning_rate"] = opt.state["learning_rate"]
        opt.state = st
        mx.eval(opt.state)

    accum = spec.grad_accum
    size_gb = prep.size_bytes / GIB

    def loss_fn(m, inputs, targets, mask):
        return masked_ce(m(inputs), targets, mask)

    consumed = {"micro": 0}
    cb_state = {"tokens": 0, "base": start_step}

    class _Cb:
        def on_train_loss_report(self, info):
            s = cb_state["base"] + info["iteration"] // accum
            log.step(s, float(info["train_loss"]), cb_state["tokens"])
            cb_state["tokens"] = 0

        def on_val_loss_report(self, info):
            pass

    patched = getattr(T, "TrainUI", None)
    if patched is not None:
        T.TrainUI = _NullUI
    step = start_step
    try:
        log.reset_clock()
        while step < spec.steps:
            seg = min(spec.ckpt_every, spec.steps - step)
            base_micro = step * accum
            cb_state["base"] = step
            consumed["micro"] = 0
            log.reset_clock()

            def iterate(dataset=None, batch_size=None, max_seq_length=None, loop=False,
                        seed=None, comm_group=None, _base=base_micro, _n=seg * accum):
                for i in range(_n):
                    if i % accum == 0 and stop is not None and stop.is_set():
                        return
                    b = prep.plan.batch(_base + i)
                    consumed["micro"] = i + 1
                    cb_state["tokens"] += b.n_tokens
                    yield mx.array(b.inputs), mx.array(b.targets), mx.array(b.mask)

            args = TrainingArgs(
                batch_size=spec.micro_batch, iters=seg * accum, steps_per_report=accum,
                steps_per_eval=10**9, steps_per_save=10**9, max_seq_length=10**6,
                adapter_file=str(job.dir / "mlx_adapters.safetensors"),
                grad_checkpoint=size_gb > 0.15 * (prep.working_set / GIB),
                grad_accumulation_steps=accum)
            train(model, opt, None, None, args, loss=loss_fn, iterate_batches=iterate,
                  training_callback=_Cb())
            done_micro = consumed["micro"] - consumed["micro"] % accum
            step += done_micro // accum
            flat, _ = _flat_trainable(model)
            _save_ckpt(prep, step, flat, _resident_opt_canon(opt, prefix))
            if done_micro < seg * accum:
                break                                      # interrupted
    finally:
        if patched is not None:
            T.TrainUI = patched
    flat, _ = _flat_trainable(model)
    return flat, step


def _resident_opt_canon(opt, prefix: str) -> dict:
    m, v = {}, {}
    for k, arr in tree_flatten(opt.state):
        if k in ("step", "learning_rate"):
            continue
        name, _, which = k.rpartition(".")
        name = name[len(prefix):] if name.startswith(prefix) else name
        (m if which == "m" else v)[name] = arr
    return {"step": int(opt.state["step"].item()), "m": m, "v": v}


# ------------------------------------------------------------ job lifecycle

def _save_ckpt(prep: "Prepared", step: int, params: dict, opt: dict) -> None:
    mx.eval(params, opt["m"], opt["v"])
    prep.cp.save(step, params, opt)


def write_adapter(prep: Prepared, flat: dict, steps_done: int, job: Job, base: str,
                  final_loss: float | None) -> Path:
    spec = prep.spec
    dest = adapters_dir() / spec.name
    extra = {"base": spec.model, "quant": spec.quant, "tune_job": job.id, "path": spec.path,
             "steps": steps_done, "examples": prep.stats.examples,
             "data_sha256": data_hash(Path(spec.data)), "max_seq": spec.max_seq,
             "micro_batch": spec.micro_batch, "grad_accum": spec.grad_accum,
             "lr": spec.lr, "schedule": spec.schedule, "seed": spec.seed,
             "final_loss": final_loss}
    lo.save_adapter_dir(dest, flat, spec.lora(), base, prep.n_layers, prep.shapes, extra)
    log = job.dir / "losses.jsonl"
    if log.exists():
        shutil.copy(log, dest / "train_log.jsonl")
    return dest


def _record_compute_rate(prep: Prepared, result: dict) -> None:
    """Persist the measured effective flops/s so the next pre-run line is a measurement,
    not an assumption. Streamed: compute time = micro-batch wall time minus time blocked
    on the ring, at 6 flops/param/token (forward + recompute + input gradients).
    Resident: step wall time at 4 flops/param/token."""
    from ..calibration import load_calibration, save_calibration
    spec, ss = prep.spec, prep.stream_stats
    L = prep.n_layers
    try:
        from ..engines.mlx_stream import SafetensorsIndex
        ix = SafetensorsIndex(Path(spec.model_dir))
        n = sum(t.nbytes for t in ix.layers[0].tensors) // 2 * L
        if spec.path == "streamed" and ss.get("micro_tokens"):
            compute = ss["micro_seconds"] - ss["wait_seconds"]
            if compute > 0:
                rate = 6.0 * n * ss["micro_tokens"] / compute
                key = f"{spec.model}|streamed"
            else:
                return
        elif spec.path == "resident" and result["seconds"] > 0 and result["step"]:
            toks = (result["step"] * spec.grad_accum * spec.micro_batch
                    * prep.stats.tokens / max(1, prep.stats.examples))
            rate = 4.0 * n * toks / result["seconds"]
            key = f"{spec.model}|resident"
        else:
            return
        import mlx.core as mx
        if mx.default_device() != mx.gpu:
            return                       # CPU rates would poison the Metal estimate
        cal = load_calibration()
        cal.setdefault("tune_rates", {})[key] = round(rate)
        save_calibration(cal)
        result["flops_rate"] = round(rate)
    except Exception:
        pass


def _device_label() -> str:
    from ..platforms import apple_silicon
    from ..runs import hardware_summary  # noqa: F401
    if mx.default_device() == mx.gpu:
        try:
            import platform
            import subprocess
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
            return f"apple-gpu:{chip}" if chip else "apple-gpu"
        except OSError:
            return "apple-gpu"
    return "cpu"


def base_dtype(model_dir) -> str:
    """The dtype the base weights are stored and computed in: bf16, float16 or float32."""
    from ..ring import SafetensorsIndex
    t = next(iter(SafetensorsIndex(Path(model_dir)).layers[0].tensors))
    return {"BF16": "bf16", "F16": "float16", "F32": "float32"}.get(t.st_dtype, t.st_dtype)


def run_tune(prep: Prepared, job: Job, *, stop=None, progress_cb=None, resume: bool = False,
             note=None, on_event=None) -> dict:
    """Train to prep.spec.steps (or until `stop`), checkpointing; write the adapter."""
    import threading
    spec = prep.spec
    stop = stop if stop is not None else threading.Event()
    if spec.stop_after:
        inner = progress_cb

        def progress_cb(info, _inner=inner):
            if _inner:
                _inner(info)
            if info["step"] >= spec.stop_after:
                stop.set()
    prep.cp = Checkpointer(
        spec, job.dir, engine="mlx_stream_tune" if spec.path == "streamed" else "mlx_lm_lora",
        device=_device_label(),
        numerics={"base": base_dtype(spec.model_dir), "adapter": "float32",
                  "optimizer": "float32", "dropout": spec.dropout},
        targets=sorted(prep.shapes), on_event=on_event)
    start, init = 0, None
    ck = prep.cp.load(want=resume)
    if ck is not None:
        start = ck.step
        init = ({k: mx.array(v) for k, v in ck.params.items()},
                {"step": ck.opt["step"],
                 "m": {k: mx.array(v) for k, v in ck.opt["m"].items()},
                 "v": {k: mx.array(v) for k, v in ck.opt["v"].items()}})
        prep.cp.sync_losses(start)
    job.write_meta(status="running", done=start, total=spec.steps)
    log = StepLog(job, spec.steps, start, progress_cb)
    t0 = time.monotonic()
    try:
        fn = train_streamed if spec.path == "streamed" else train_resident
        flat, step = fn(prep, job, log, stop, start, init, note)
    except BaseException:
        job.write_meta(status="failed")
        raise
    finally:
        log.close()
    losses = [json.loads(l)["loss"] for l in (job.dir / "losses.jsonl").read_text().splitlines()
              if l.strip()]
    interrupted = step < spec.steps
    result = {"step": step, "steps": spec.steps, "interrupted": interrupted,
              "first_loss": losses[0] if losses else None,
              "final_loss": losses[-1] if losses else None,
              "seconds": time.monotonic() - t0,
              "peak_gb": mx.get_peak_memory() / GIB if mx.default_device() == mx.gpu else 0.0}
    job.write_meta(status="interrupted" if interrupted else "completed", done=step)
    _record_compute_rate(prep, result)
    ss = prep.stream_stats
    if spec.path == "streamed" and ss.get("micro_tokens") and ss["micro_seconds"] > 0:
        from ..engines.mlx_stream import SafetensorsIndex
        ix = SafetensorsIndex(Path(spec.model_dir))
        n = sum(t.nbytes for t in ix.layers[0].tensors) // 2 * prep.n_layers
        fl = 6.0 * n * ss["micro_tokens"]
        result["tflops_overall"] = fl / ss["micro_seconds"] / 1e12
        comp = ss["micro_seconds"] - ss["wait_seconds"]
        result["tflops_compute"] = fl / comp / 1e12 if comp > 0 else None
        result["padded_tokens_per_s"] = ss["micro_tokens"] / ss["micro_seconds"]
        result["trained_tokens_per_s"] = prep.stats.trained_tokens / max(1, prep.stats.examples) \
            * spec.micro_batch * spec.grad_accum * result["step"] / max(1e-9, result["seconds"])
    if not interrupted:
        result["adapter"] = str(write_adapter(prep, flat, step, job, spec.model,
                                              result["final_loss"]))
    return result
