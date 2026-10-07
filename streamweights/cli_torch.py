"""run, distill and eval on the PyTorch engines: the counterpart of cli._run_mlx. Same flow
(resolve, size, one pre-run line, a job with a checkpoint per row, results.jsonl in the OpenAI
batch shape), with placement from device memory (CUDA) or RAM (CPU)."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import typer

from . import estimate as est_mod
from . import overnight, runtime
from .engines.base import MemoryBudget, ModelSpec
from .errors import SpillError
from .jobs.engine import Job
from .registry import GIB, download_safetensors, load_registry


def _dur(s: float) -> str:
    from .cli import _fmt_dur
    return _fmt_dur(s)


def model_dir_for(res) -> tuple[Path, dict | None]:
    """The local safetensors directory for a resolved model: staged from --weights <uri>
    when given, else the per-machine cache (downloaded from Hugging Face when missing)."""
    from .resolve import download_hf
    if runtime.ENV.weights:
        from .portable.weights import stage_weights
        d, info = stage_weights(runtime.ENV.weights,
                                note=lambda s: typer.echo(f"   {s}", err=True))
        return d, info
    if res.kind == "hf":
        return download_hf(res), None
    return download_safetensors(res.name), None


def _live(job_dir, quiet):
    from .cli import _LiveRenderer
    if runtime.ENV.headless:
        return _HeadlessPass(job_dir)
    return _LiveRenderer(job_dir, quiet)


class _HeadlessPass:
    """The per-pass callback in headless mode: keeps live.json (for `spill status` and
    `spill tail`) and draws nothing."""

    def __init__(self, job_dir):
        self.job_dir = Path(job_dir) if job_dir else None

    def __call__(self, info):
        if self.job_dir:
            try:
                (self.job_dir / "live.json").write_text(json.dumps(info))
            except OSError:
                pass


def run_torch(res, input_jsonl: Path, out: Path | None, context: int, parallel: int | None,
              hw: dict, rows: list[dict], quiet: bool, opts, adapter_spec: str | None,
              choice):
    from . import logits as lg
    from . import runs as runs_mod
    from .adapters import resolve_adapter
    from .calibration import load_calibration
    from .engines.common import PREFIX_MIN_ROWS, PREFIX_MIN_TOKENS, common_prefix_len
    from .engines.supported import classify
    from .engines.torch_common import (device_label, load_tokenizer, memory_total_bytes,
                                       resolve_dtype)
    from .engines.torch_resident import TorchResidentEngine
    from .engines.torch_stream import TorchEngine, batch_math
    from .jobs.runner import run_job
    from .policy import engine_read_rate, estimate_score_seconds
    from .ring import SafetensorsIndex

    scoring = opts.mode == "score"
    model = res.name
    engine_name = choice.name
    path, staged = model_dir_for(res)
    index = SafetensorsIndex(path)
    cfg = index.config
    fam = classify(cfg)
    size = index.total_bytes
    est_mod.register_model(model, size)
    adapter = resolve_adapter(adapter_spec, numpy=True) if adapter_spec else None
    if adapter is not None:
        adapter.validate(cfg)

    dtype, dtype_note = resolve_dtype(engine_name)
    elt = 2 if str(dtype).endswith("bfloat16") else 4
    stored = 2 if index.layers[0].tensors[0].st_dtype in ("BF16", "F16") else 4
    compute_size = int(size * elt / stored)
    ws = memory_total_bytes(engine_name)
    fits = compute_size <= ws * 0.70
    tok = load_tokenizer(path)
    from .engines.common import collect_eos_ids
    eos = collect_eos_ids(path, tok) if scoring else None
    D = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    kv_tok = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * D * elt
    lens, n_target, tok_lists = [], 0, []
    max_tokens = 0 if scoring else max(r["body"].get("max_tokens", 128) for r in rows)
    if scoring:
        from .formats import tokenize_scored
        for r in rows:
            n_p, ids = tokenize_scored(tok, r["body"]["messages"], eos)
            lens.append(len(ids))
            n_target += len(ids) - n_p
    else:
        for r in rows:
            toks = tok.apply_chat_template(r["body"]["messages"], add_generation_prompt=True)
            tok_lists.append(list(toks))
            lens.append(len(toks) + r["body"].get("max_tokens", 128))

    cal = load_calibration()
    if engine_name not in (cal.get("engine_tflops") or {}):
        from .engine_select import measure_rates
        measure_rates()
        cal = load_calibration()
    peak_tflops = cal["engine_tflops"][engine_name]
    tflops = peak_tflops * 0.5          # attention, norms and launch overhead cost about half
    P_est = 0
    if not scoring and opts.prefix_reuse and len(tok_lists) >= PREFIX_MIN_ROWS:
        P_est = min(common_prefix_len(tok_lists), min(len(t) for t in tok_lists) - 1)
        if P_est < PREFIX_MIN_TOKENS:
            P_est = 0
    prompt_tok = sum(len(t) for t in tok_lists)
    prefill_tok = prompt_tok - (len(tok_lists) - 1) * P_est
    prefill_s = est_mod.prefill_s(model, prefill_tok, tflops) if not scoring else 0.0
    cost_note = ""
    if not scoring:
        cost_note = (f" Prefill {prefill_tok:,} tokens at {tflops:.3g} TFLOP/s ({engine_name} "
                     f"matmul rate x 0.5) = {_dur(prefill_s)}"
                     + (f"; shared prefix {P_est} tokens computed once" if P_est else "") + ".")

    res_bytes = compute_size if fits else (
        3 * index.max_layer_bytes * elt // stored
        + (index.embed.nbytes + (0 if index.tied else index.lm_head.nbytes)) * elt // stored)
    seq_costs = [n * kv_tok for n in lens]
    K_lp = (opts.logprobs or 32) if (opts.logprobs or scoring) else None
    bm = batch_math(ws, res_bytes, seq_costs, P_est, kv_tok, cfg["num_hidden_layers"],
                    parallel, K_lp, None, int(cfg.get("vocab_size", 0)) or 1)
    batch = bm["batch"]
    n_prompts = len(rows)
    if fits:
        engine = TorchResidentEngine(
            progress_note=lambda s: typer.echo(f"   {s}", err=True), engine=engine_name)
        bw = (cal.get("engine_membw_gbs") or {}).get(engine_name, 20.0) * 1e9
        pass_s = compute_size / (bw * 0.5)
        decode_s = math.ceil(n_prompts / batch) * (max_tokens + 1) * pass_s
        est = decode_s + prefill_s
        placement = f"resident on {engine_name}"
        why = f"{choice.why}; model fits 70% of {ws / GIB:.0f} GB -> torch_resident"
        rate_note = ""
        if scoring:
            est, rate_note = estimate_score_seconds(sum(lens), compute_size, "bf16", cal,
                                                    f"{model}|bf16|{engine_name}", False, None)
    else:
        engine = TorchEngine(progress_note=lambda s: typer.echo(f"   {s}", err=True),
                             engine=engine_name)
        key = f"{model}|bf16"
        rate, rate_src = engine_read_rate(cal, hw, key=key)
        pass_s = size / rate
        est = math.ceil(n_prompts / batch) * (max_tokens + 1) * pass_s + prefill_s
        rate_note = ""
        if scoring:
            est, rate_note = estimate_score_seconds(sum(lens), size, "bf16", cal,
                                                    f"{model}|bf16|{engine_name}", True, rate)
        placement = (f"streaming from disk on {engine_name} at ~{rate / GIB:.1f} GB/s "
                     f"({rate_src})")
        why = f"{choice.why}; weights stream layer by layer through a pread ring"

    label = model + (f"+{adapter.id}" if adapter is not None else "")
    options = {"mode": opts.mode, "logprobs": opts.logprobs, "adapter": opts.adapter,
               "kind": opts.kind, "max_tokens": max_tokens, "engine_choice": engine_name,
               "state": runtime.state_uri()}
    job = Job.create(input_jsonl, model, "bf16", context, batch, out, rows=rows, options=options)
    spec = ModelSpec(model, "bf16", Path(path), load_registry()[model].arch
                     if model in load_registry() else {}, context)
    if opts.logprobs or scoring:
        spec.extra["logprobs"] = opts.logprobs or 32
    if scoring:
        spec.extra["mode"] = "score"
    if adapter is not None:
        spec.extra["adapter"] = adapter
    if not opts.prefix_reuse:
        spec.extra["prefix_reuse"] = False
    runs_mod.start_run(job, command=None, spec=spec, input_path=input_jsonl, hw=hw,
                       engine_name=engine.name,
                       adapter={k: v for k, v in adapter.info().items()
                                if k in ("id", "hash", "layout", "rank", "base")}
                       if adapter is not None else None, options=options, kind="run")
    engine.pass_cb = _live(job.dir, quiet)
    out_path = out or (runs_mod.run_dir(job.id) / "distill.jsonl"
                       if opts.kind == "distill" else job.results_path)
    cmd = opts.cmd or ("eval" if opts.kind == "judge" else opts.kind)
    if scoring:
        what = (f"Teacher-forced score (prefill only, no sampling): {n_prompts} rows, "
                f"{sum(lens)} tokens to prefill ({n_target} target tokens scored, top-{K_lp} "
                f"log-probs each). Est. {_dur(est)} ({rate_note}).")
    else:
        extra = f" Top-{opts.logprobs} log-probs on." if opts.logprobs else ""
        what = f"{n_prompts} prompts, batch {batch}. Est. {_dur(est)}.{cost_note}{extra}"
        if opts.kind == "distill":
            n_prompt_tok = sum(lens) - sum(r["body"].get("max_tokens", 128) for r in rows)
            what = (f"{n_prompts} prompts, {n_prompt_tok} prompt tokens plus up to "
                    f"{sum(r['body'].get('max_tokens', 128) for r in rows)} generated tokens "
                    f"scored with the teacher's top-{opts.logprobs} log-probs, batch {batch}. "
                    f"Est. {_dur(est)}.{cost_note}")
    from .platforms import apple_silicon  # noqa: F401
    line = (f"spill {cmd} {label}: bf16 ({size / GIB:.1f} GB, {fam.label}: {fam.state}) "
            f"{'fits' if compute_size <= ws else 'does not fit'} in {ws / GIB:.0f} GB "
            f"{'device memory' if engine_name == 'torch-cuda' else 'RAM'}, {placement}; "
            f"{dtype_note}. {what} Cost: $0. Results -> {out_path}")
    typer.echo(line)
    typer.echo(f"   why: {why}")
    if staged:
        typer.echo(f"   weights staged from {staged['uri']}: {staged['bytes'] / 1e9:.2f} GB copied "
                   f"in {staged['seconds']:.1f} s ({staged['reused']} files already here)", err=True)
    runtime.emit("start", job=job.id, model=label, command=cmd, rows=n_prompts, batch=batch,
                 placement=placement, dtype_note=dtype_note, estimated_seconds=round(est),
                 state=runtime.state_uri(), message=line, device=device_label(engine_name))

    try:
        prog = run_job(job, engine, spec, MemoryBudget(ws, batch_override=parallel),
                       progress_cb=None)
    except Exception as e:
        if "out of memory" in str(e).lower() and engine_name == "torch-cuda":
            typer.echo("spill: the GPU ran out of memory; retrying with a smaller batch "
                       "(completed rows are checkpointed)", err=True)
            prog = run_job(job, engine, spec, MemoryBudget(int(ws * 0.72),
                                                           batch_override=parallel),
                           progress_cb=None)
        else:
            raise
    sys.stderr.write("\n")
    if engine.prefix_info:
        runs_mod.update_manifest(job.id, prefix_reuse=engine.prefix_info)
    if out and opts.kind == "run" and Path(out) != job.results_path:
        Path(out).write_bytes(job.results_path.read_bytes())
    typer.echo(f"{prog.done}/{prog.total} rows, {prog.completion_tokens} "
               f"{'target' if scoring else 'completion'} tokens, "
               f"{prog.tokens_per_sec:.1f} tok/s aggregate")
    if opts.kind == "run":
        from .cli import _next_hint
        _next_hint(f"spill resume {job.id}" if prog.done < prog.total else "spill runs")
    return job, prog


def resume_torch(j: Job, hw: dict, choice_name: str) -> None:
    """Continue an interrupted row job on a torch engine."""
    from .adapters import resolve_adapter
    from .engine_select import choose_engine
    from .engines.torch_common import memory_total_bytes
    from .engines.torch_resident import TorchResidentEngine
    from .engines.torch_stream import TorchEngine
    from .jobs.runner import run_job
    from .registry import safetensors_dir
    from .ring import SafetensorsIndex
    choice = choose_engine(choice_name)
    runtime.set_engine(choice.name)
    meta = j.read_meta()
    o = meta.get("options", {})
    if runtime.ENV.weights:
        from .portable.weights import stage_weights
        path, _ = stage_weights(runtime.ENV.weights)
    else:
        from .resolve import download_hf, resolve_model
        res = resolve_model(j.model)
        path = download_hf(res) if res.kind == "hf" else safetensors_dir(j.model)
    ws = memory_total_bytes(choice.name)
    size = SafetensorsIndex(path).total_bytes
    engine_cls = TorchResidentEngine if size <= ws * 0.35 else TorchEngine
    engine = engine_cls(engine=choice.name)
    engine.pass_cb = _live(j.dir, False)
    spec = ModelSpec(j.model, "bf16", Path(path), {}, j.ctx)
    if o.get("logprobs") or o.get("mode") == "score":
        spec.extra["logprobs"] = o.get("logprobs") or 32
    if o.get("mode") == "score":
        spec.extra["mode"] = "score"
    if o.get("adapter"):
        spec.extra["adapter"] = resolve_adapter(o["adapter"], numpy=True)
    prog = run_job(j, engine, spec, MemoryBudget(ws), progress_cb=None)
    sys.stderr.write("\n")
    typer.echo(f"{prog.done}/{prog.total} rows complete")
