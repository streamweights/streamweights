"""streamweights CLI (`spill`). Every command ends by printing the next command."""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import typer
import typer.core

from . import engine_select, overnight, platforms, runtime
from .errors import SpillError
from . import probe as probe_mod
from .engines.base import MemoryBudget, ModelSpec
from .jobs.engine import Job, JOBS_DIR, compute_offload
from .policy import choose_quant, choose_quant_v2, engine_read_rate
from .registry import (GIB, download, load_registry, mlx_quant_repo,
                       safetensors_dir, safetensors_downloaded,
                       safetensors_spec, download_safetensors)

if sys.version_info < (3, 10):  # pragma: no cover
    sys.stderr.write("spill needs Python 3.10 or newer; this is "
                     f"{sys.version.split()[0]}. Try: uv tool install "
                     "git+https://github.com/streamweights/streamweights\n")
    raise SystemExit(1)

COMMAND_ORDER = ["init", "plan", "build", "report", "compare", "example", "run", "distill", "tune",
                 "eval", "export", "test", "bundle", "move", "models", "adapters", "runs", "status",
                 "tail", "resume", "doctor", "check"]


class _LoopOrder(typer.core.TyperGroup):
    """`spill --help` lists the commands in the order you use them."""

    def list_commands(self, ctx):
        known = [c for c in COMMAND_ORDER if c in self.commands]
        return known + [c for c in self.commands if c not in COMMAND_ORDER]


app = typer.Typer(cls=_LoopOrder, add_completion=False, no_args_is_help=True,
                  pretty_exceptions_enable=False, rich_markup_mode=None,
                  epilog="Example: spill example banking77 --quick && "
                         "spill build banking77-quick")
_DEBUG = False


@app.callback()
def _main(ctx: typer.Context):
    """A 70B model doesn't fit on your laptop. Build your own model from it anyway."""
    if ctx.invoked_subcommand in ("resume", "doctor", None) or "--help" in sys.argv:
        return
    try:
        from . import overnight
        line = overnight.banner(JOBS_DIR)
    except Exception:
        line = None
    if line:
        typer.echo(line, err=True)

SAMPLE_PATH = Path(__file__).parent / "data" / "sample-20.jsonl"


def _long_job(name: str, notify_url: str | None = None):
    """caffeinate and notifications for a person at a terminal; nothing of either when headless."""
    if runtime.ENV.headless:
        # no caffeinate and no desktop notification; an explicit --notify URL is still POSTed
        return overnight.long_job(name, notify_url, local=False) if notify_url \
            else contextlib.nullcontext()
    return overnight.long_job(name, notify_url)


def _fail(e):
    """One line ending in the command that gets you unstuck; the traceback only with --debug."""
    from .errors import SpillError
    if _DEBUG:
        raise e
    from .errors import StageInterrupted
    if runtime.ENV.stopped_early and isinstance(e, StageInterrupted):
        typer.echo(f"stopped early (--stop-after {runtime.ENV.stop_after}): {e.done}/{e.total} "
                   f"rows are checkpointed; run the same command again to continue")
        return
    runtime.emit("error", message=e.line() if isinstance(e, SpillError)
                 else f"{type(e).__name__}: {e}")
    if isinstance(e, SpillError):
        cmd = next((a for a in sys.argv[1:] if a in COMMAND_ORDER), None)
        jsonl = next((a for a in sys.argv[1:] if a.endswith(".jsonl")), None)
        default = (f"spill check {jsonl}" if jsonl and "line " in e.message[:12] else
                   f"spill {cmd} --help" if cmd else "spill --help")
        typer.echo(f"spill: {e.line(default=default)}", err=True)
    else:
        again = "spill " + " ".join(sys.argv[1:] + ["--debug"])
        typer.echo(f"spill: {type(e).__name__}: {e}. Try: {again}", err=True)
    raise typer.Exit(1)


def main():
    """Console entry point: `--debug` anywhere on the line turns tracebacks on; anything
    that escapes a command still prints one line."""
    global _DEBUG
    if "--debug" in sys.argv:
        sys.argv.remove("--debug")
        _DEBUG = True
    try:                              # Typer 0.27+ bundles its own click
        from typer import _click as click
    except ImportError:
        import click
    try:
        sys.argv[1:] = _config_args(sys.argv[1:])
    except SpillError as e:
        typer.echo(f"spill: {e.line(default='spill --help')}", err=True)
        raise SystemExit(1)
    try:
        rc = app(prog_name="spill", standalone_mode=False)
        if isinstance(rc, int) and rc:        # a command ended with typer.Exit(code)
            raise SystemExit(rc)
    except typer.Exit as e:
        raise SystemExit(e.exit_code)
    except typer.Abort:
        raise SystemExit(130)
    except click.ClickException as e:
        cmd = next((a for a in sys.argv[1:] if a in COMMAND_ORDER), None)
        typer.echo(f"spill: {e.format_message().rstrip('. ')}. Try: spill {cmd + ' ' if cmd else ''}"
                   f"--help", err=True)
        raise SystemExit(2)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        try:
            _fail(e)
        except typer.Exit as x:
            raise SystemExit(x.exit_code)


JOB_COMMANDS = ("run", "distill", "tune", "eval", "build")


def _pop_option(args: list[str], flag: str) -> tuple[str | None, list[str]]:
    """(value, args without it) for `--flag value` or `--flag=value`."""
    out, val, i = [], None, 0
    while i < len(args):
        a = args[i]
        if a == flag and i + 1 < len(args):
            val, i = args[i + 1], i + 2
            continue
        if a.startswith(flag + "="):
            val, i = a.split("=", 1)[1], i + 1
            continue
        out.append(a)
        i += 1
    return val, out


def _config_args(args: list[str]) -> list[str]:
    """Expand `--config job.json` into the invocation it stores, and write the exact config of
    an invocation (and stop) for `--emit-config job.json`."""
    from . import jobconfig
    cfg_path, args = _pop_option(args, "--config")
    emit_path, args = _pop_option(args, "--emit-config")
    if cfg_path is None and emit_path is None:
        return args
    group = typer.main.get_command(app)
    cfg = jobconfig.load(Path(cfg_path)) if cfg_path else None
    cmd = next((a for a in args if a in JOB_COMMANDS), None)
    if cmd is None and cfg is not None:
        cmd = cfg["command"]
        args = [cmd] + args
    if cmd not in JOB_COMMANDS:
        raise SpillError("--config and --emit-config work with build, run, distill, tune and "
                         "eval", "spill build --help")
    idx = args.index(cmd)
    rest = args[idx + 1:]
    click_cmd = group.commands[cmd]
    if cfg is not None:
        if cfg["command"] != cmd:
            raise SpillError(f"{cfg_path} is a spill {cfg['command']} job, not {cmd}",
                             f"spill {cfg['command']} --config {cfg_path}")
        rest = jobconfig.to_argv(cfg, click_cmd, rest)
    if emit_path is not None:
        out = jobconfig.build(cmd, click_cmd, rest)
        jobconfig.write(Path(emit_path), out)
        typer.echo(f"wrote {emit_path}: the exact configuration of this spill {cmd}; it was "
                   f"not run")
        typer.echo(f"\nnext: spill {cmd} --config {emit_path}")
        raise SystemExit(0)
    return args[:idx + 1] + rest


def _resolve_input(input_arg: str) -> Path:
    from .errors import SpillError
    if input_arg == "sample":
        typer.echo("(using the packaged 20-prompt sample: spill run <model> sample)")
        return SAMPLE_PATH
    from .portable.store import is_uri
    if is_uri(input_arg):
        return _fetch_input(input_arg)
    p = Path(input_arg)
    if not p.exists():
        raise SpillError(f"input file {input_arg} does not exist",
                         "spill run <model> sample")
    return p


def _fetch_input(uri: str) -> Path:
    """An input named by URI (s3://, gs://, az://, memory://) is copied into the per-machine
    cache and used from there."""
    import hashlib

    from .portable.store import Store
    parent, _, leaf = uri.rpartition("/")
    cache = REPO_ROOT_INPUTS / hashlib.sha256(uri.encode()).hexdigest()[:12] / leaf
    Store(parent).copy_to_local(leaf, cache)
    return cache


from .registry import REPO_ROOT as _ROOT  # noqa: E402

REPO_ROOT_INPUTS = _ROOT / "state" / "inputs"

GATEWAY = "http://127.0.0.1:11435"
MB = 1024 * 1024


def _is_mac() -> bool:
    from .platforms import mlx_available
    return mlx_available()


def _fmt_dur(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


_NEXT_HINTS = True      # off while `spill build` runs its stages: build prints the one next command


def _next_hint(cmd: str) -> None:
    if _NEXT_HINTS:
        typer.echo(f"\nnext: {cmd}")


@contextlib.contextmanager
def _no_next_hints():
    """A command that runs other commands' code (eval) prints one next command, at the end."""
    global _NEXT_HINTS
    was, _NEXT_HINTS = _NEXT_HINTS, False
    try:
        yield
    finally:
        _NEXT_HINTS = was


def _gateway_up() -> bool:
    import httpx
    try:
        return httpx.get(f"{GATEWAY}/health", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


def _rough_batch(st_bytes: int, arch: dict, rows: list[dict], max_tokens: int,
                 ws: int, quant: str = "bf16") -> int:
    """Pre-download batch estimate (chars/4 token approximation). Uses the
    measured memory calibration when one exists for this quant."""
    from .calibration import load_calibration
    lens = [sum(len(m.get("content", "")) for m in r["body"]["messages"]) // 4 + 16
            for r in rows]
    mean_tokens = sum(lens) / max(1, len(lens)) + max_tokens
    mem = load_calibration().get("mem_model", {}).get(f"llama3.3:70b|{quant}")
    if mem:
        batch = int((ws * 0.85 - mem["base_bytes"]) /
                    max(1, mem["per_seq_token_bytes"] * mean_tokens))
        return max(1, min(batch, 512))
    kv_tok = 2 * arch["n_layers"] * arch["n_kv_heads"] * arch["head_dim"] * 2
    mean_cost = mean_tokens * kv_tok
    ring = 3 * st_bytes // arch["n_layers"]
    avail = ws * 0.85 - ring - st_bytes * 0.03
    return max(1, int(avail / max(1, mean_cost)))


def _progress_line(p):
    eta = p.eta_seconds
    sys.stderr.write(f"\r{p.done}/{p.total} rows  {p.tokens_per_sec:7.1f} tok/s  "
                     f"ETA {_fmt_dur(eta) if eta else '...'}   ")
    sys.stderr.flush()


def _fmt_eta(s):
    if s is None:
        return "..."
    if s < 3600:
        return f"{int(s // 60)}m {int(s % 60):02d}s"
    return f"{int(s // 3600)}h {int(s % 3600 // 60):02d}m"


class _LiveRenderer:
    """One progress line plus a fixed-height block of up to 8 active slots,
    redrawn in place every pass. Also persists live.json for tail/status."""

    def __init__(self, job_dir=None, quiet=False):
        self.job_dir = Path(job_dir) if job_dir else None
        self.quiet = quiet
        self._lines = 0

    def line(self, info):
        return (f"rows {info['rows_done']}/{info['total']} · pass {info['pass_no']} "
                f"({info['pass_s']:.1f} s) · {info['tok_s']:.1f} tok/s · "
                f"ETA {_fmt_eta(info['eta_s'])} · {info['quant']} · batch {info['batch']}"
                f" · ~{info['pass_s']:.1f} s per token per prompt"
                f" · {info.get('peak_gb', 0):.1f} GB peak")

    def __call__(self, info):
        if self.job_dir:
            try:
                (self.job_dir / "live.json").write_text(json.dumps(info))
            except OSError:
                pass
        block = [self.line(info)]
        if not self.quiet:
            for s in info.get("slots", []):
                block.append(f"  {s['custom_id'][:18]:18s} {s['tokens']:4d} tok │ {s['tail']}")
        if self._lines:
            sys.stderr.write(f"\x1b[{self._lines}F")  # cursor to start of block
        out = "\n".join(l[:200] + "\x1b[K" for l in block)
        pad = self._lines - len(block)
        if pad > 0:
            out += ("\n" + "\x1b[K") * pad
            sys.stderr.write(out + f"\x1b[{pad}F")
        else:
            sys.stderr.write(out)
        sys.stderr.write("\n")
        self._lines = max(self._lines, len(block))
        sys.stderr.flush()


def _live_cb(job_dir, quiet):
    """The per-pass callback: the redrawn terminal block for a person, a live.json writer
    (no drawing) when headless."""
    if runtime.ENV.headless:
        from .cli_torch import _HeadlessPass
        return _HeadlessPass(job_dir)
    return _LiveRenderer(job_dir, quiet)


def _pass_line(info):  # retained for callers without a job dir
    _LiveRenderer()(info)


@dataclass
class RunOpts:
    """What distinguishes the phase 2 commands from a plain run."""
    mode: str = "generate"        # generate | score (teacher-forced, prefill only)
    logprobs: int | None = None
    adapter: str | None = None    # spec string of the adapter, if any
    kind: str = "run"             # run | distill | judge
    cmd: str | None = None        # the command named in the pre-run line (default: kind)
    prefix_reuse: bool = True     # shared-prefix KV reuse (SPILL_NO_PREFIX_REUSE=1 to A/B it)




def _run_mlx(res, input_jsonl: Path, quant: str, reason: str,
             est_s: float | None, out: Path | None, context: int,
             parallel: int | None, hw: dict, rows: list[dict],
             quiet: bool = False, opts: RunOpts | None = None, adapter=None):
    from . import logits as lg
    from . import runs as runs_mod
    from .engines.mlx_resident import MlxResidentEngine
    from .engines.mlx_stream import (MlxStreamEngine, SafetensorsIndex,
                                     collect_eos_ids, compute_batch, load_calibration)
    from .engines.supported import classify
    from .policy import estimate_score_seconds
    from .resolve import download_hf

    opts = opts or RunOpts()
    scoring = opts.mode == "score"
    model = res.name
    reg = load_registry()
    ws = hw["gpu"]["vram_bytes"]
    ram = hw["ram_total_bytes"]

    if runtime.ENV.weights and quant in (None, "bf16"):
        from .portable.weights import stage_weights
        path, staged = stage_weights(runtime.ENV.weights,
                                     note=lambda m: typer.echo(f"   {m}", err=True))
        size = sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))
        typer.echo(f"   weights staged from {staged['uri']}: {staged['bytes'] / 1e9:.2f} GB copied "
                   f"in {staged['seconds']:.1f} s ({staged['reused']} files already here)",
                   err=True)
    elif res.kind == "hf":
        path = download_hf(res)
        size = res.st_bytes
    elif quant == "bf16":
        path = download_safetensors(model)
        size = safetensors_spec(model)["bytes"]
    else:  # 8bit / 4bit via mlx-community
        repo = mlx_quant_repo(model, quant)
        from huggingface_hub import HfApi, snapshot_download
        from .registry import MIN_FREE_AFTER_DOWNLOAD, MODELS_DIR
        path = MODELS_DIR / model.replace(":", "-") / f"mlx-{quant}"
        if not (Path(path) / "config.json").exists():
            from .download import fetch
            fetch(repo, Path(path), ["*"], f"{model} {quant} (mlx)")
        size = sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))

    fits = size <= ws * 0.70
    n_prompts = len(rows)
    max_tokens = 0 if scoring else max(r["body"].get("max_tokens", 128) for r in rows)
    cfg = json.loads((Path(path) / "config.json").read_text())
    fam = classify(cfg)
    if adapter is not None:
        adapter.validate(cfg)

    from mlx_lm.utils import load_tokenizer
    tokenizer = load_tokenizer(Path(path))
    eos = collect_eos_ids(Path(path), tokenizer) if scoring else None
    kv_tok = (2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"]
              * (cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]) * 2)
    lens, n_target, tok_lists = [], 0, []
    if scoring:
        from .formats import tokenize_scored
        for r in rows:
            n_p, ids = tokenize_scored(tokenizer, r["body"]["messages"], eos)
            lens.append(len(ids))
            n_target += len(ids) - n_p
    else:
        for r in rows:
            toks = tokenizer.apply_chat_template(r["body"]["messages"],
                                                 add_generation_prompt=True)
            tok_lists.append(list(toks))
            lens.append(len(toks) + r["body"].get("max_tokens", 128))
    costs = [n * kv_tok for n in lens]
    cal = load_calibration()
    key = f"{model}|{quant}"

    # cost model: prefill is compute-bound (2 x params x tokens / achieved FLOP/s), and rows
    # that share a prefix compute it once (the engine does the same computation below)
    from . import estimate as est_mod
    from .engines.mlx_stream import PREFIX_MIN_ROWS, PREFIX_MIN_TOKENS, common_prefix_len
    P_est = 0
    if (not scoring and opts.prefix_reuse and len(tok_lists) >= PREFIX_MIN_ROWS
            and not fam.needs_array_mask):
        P_est = min(common_prefix_len(tok_lists), min(len(t) for t in tok_lists) - 1)
        if P_est < PREFIX_MIN_TOKENS:
            P_est = 0
    tflops, tf_src = est_mod.achieved_tflops(cal)
    prompt_tok = sum(len(t) for t in tok_lists)
    prefill_tok = prompt_tok - (len(tok_lists) - 1) * P_est
    prefill_s = est_mod.prefill_s(model, prefill_tok, tflops) if not scoring else 0.0
    cost_note = ""
    if not scoring:
        cost_note = (f" Prefill {prefill_tok:,} tokens at {tflops:.3g} TFLOP/s ({tf_src}) = "
                     f"{_fmt_dur(prefill_s)}"
                     + (f"; shared prefix {P_est} tokens computed once (saves "
                        f"{(len(tok_lists) - 1) * P_est:,} tokens)" if P_est else "")
                     + ".")

    if fits:
        engine = MlxResidentEngine(progress_note=lambda s: typer.echo(f"   {s}", err=True))
        batch = 1
        placement = "resident"
        mean_lens = sum(len(t) for t in tok_lists) / max(1, len(tok_lists)) if tok_lists else 0
        wl = est_mod.Workload(n_prompts, P_est, max(1.0, mean_lens - P_est), max_tokens,
                              max_tokens)
        dec_batch = est_mod.decode_batch(model, wl, ws, bool(P_est)) if tok_lists else 1
        shown_batch = min(dec_batch, n_prompts)
        est = (math.ceil(n_prompts / dec_batch) * (max_tokens + 1) * size / est_mod.RESIDENT_BYTES_PER_S
               + prefill_s) if not scoring else 0.0
        why = f"{reason}; model fits working set -> mlx_resident"
        rate_note = ""
        if scoring:
            est, rate_note = estimate_score_seconds(sum(lens), size, quant, cal, key,
                                                    False, None)
    else:
        engine = MlxStreamEngine(progress_note=lambda s: typer.echo(f"   {s}", err=True))
        index = SafetensorsIndex(path)
        bm = compute_batch(index, MemoryBudget(ws, batch_override=parallel),
                           costs, max_tokens, kv_tok, calibration=cal, quant=key)
        batch = shown_batch = bm.batch
        rate, rate_src = engine_read_rate(cal, hw, key=key)
        pass_s = size / rate
        est = math.ceil(n_prompts / batch) * (max_tokens + 1) * pass_s + prefill_s
        rate_note = ""
        if scoring:
            est, rate_note = estimate_score_seconds(sum(lens), size, quant, cal, key,
                                                    True, rate)
        placement = f"streaming from NVMe at ~{rate / GIB:.1f} GB/s ({rate_src})"
        why = f"{bm.reason}; quant: {reason}"

    label = model + (f"+{adapter.id}" if adapter is not None else "")
    options = {"mode": opts.mode, "logprobs": opts.logprobs,
               "adapter": opts.adapter, "kind": opts.kind, "max_tokens": max_tokens,
               "engine_choice": "mlx", "state": runtime.state_uri()}
    job = Job.create(input_jsonl, model, quant, context, batch, out, rows=rows,
                     options=options)
    spec = ModelSpec(model, quant, Path(path), reg[model].arch if model in reg else {},
                     context)
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
                       if adapter is not None else None,
                       options=options, kind="run")
    engine.pass_cb = _live_cb(job.dir, quiet)
    out_path = out or (runs_mod.run_dir(job.id) / "distill.jsonl"
                       if opts.kind == "distill" else job.results_path)
    cmd = opts.cmd or ("eval" if opts.kind == "judge" else opts.kind)
    if scoring:
        what = (f"Teacher-forced score (prefill only, no sampling): {n_prompts} rows, "
                f"{sum(lens)} tokens to prefill ({n_target} target tokens scored, top-"
                f"{spec.extra['logprobs']} log-probs each). Est. {_fmt_dur(est)} ({rate_note}).")
    else:
        extra = f" Top-{opts.logprobs} log-probs on." if opts.logprobs else ""
        what = f"{n_prompts} prompts, batch {shown_batch}. Est. {_fmt_dur(est)}.{cost_note}{extra}"
        if opts.kind == "distill":
            n_prompt_tok = sum(lens) - sum(r["body"].get("max_tokens", 128) for r in rows)
            what = (f"{n_prompts} prompts, {n_prompt_tok} prompt tokens plus up to "
                    f"{sum(r['body'].get('max_tokens', 128) for r in rows)} generated tokens "
                    f"scored with the teacher's top-{opts.logprobs} log-probs, batch {shown_batch}. "
                    f"Est. {_fmt_dur(est)} (decode from the measured pass time plus "
                    f"prefill).{cost_note}")
    line = (f"spill {cmd} {label}: {quant} ({size / GIB:.1f} GB, {fam.label}: {fam.state}) "
            f"{'fits' if size <= ram else 'does not fit'} in {ram / GIB:.0f} GB RAM, {placement}. "
            f"{what} Cost: $0.{'' if runtime.ENV.headless else overnight.battery_note()} "
            f"Results -> {out_path}")
    typer.echo(line)
    typer.echo(f"   why: {why}")
    runtime.emit("start", job=job.id, model=label, command=cmd, rows=n_prompts, batch=shown_batch,
                 placement=placement, estimated_seconds=round(est or 0),
                 state=runtime.state_uri(), message=line)

    from .jobs.runner import run_job
    try:
        prog = run_job(job, engine, spec, MemoryBudget(ws, batch_override=parallel),
                       progress_cb=None)  # the per-pass line is the only line
    except Exception as e:
        if any(k in str(e) for k in ("Insufficient Memory", "kIOGPU", "OutOfMemory",
                                     "metal::malloc")):
            typer.echo("spill: Metal ran out of memory; retrying with a smaller batch "
                       "(completed rows are checkpointed)", err=True)
            prog = run_job(job, engine, spec,
                           MemoryBudget(int(ws * 0.72), batch_override=parallel),
                           progress_cb=None)
        else:
            raise
    sys.stderr.write("\n")
    if getattr(engine, "prefix_info", None):
        runs_mod.update_manifest(job.id, prefix_reuse=engine.prefix_info)
    if out and opts.kind == "run" and Path(out) != job.results_path:
        Path(out).write_bytes(job.results_path.read_bytes())
    typer.echo(f"{prog.done}/{prog.total} rows, {prog.completion_tokens} "
               f"{'target' if scoring else 'completion'} tokens, "
               f"{prog.tokens_per_sec:.1f} tok/s aggregate")
    if opts.kind == "run":
        _next_hint(f"spill resume {job.id}" if prog.done < prog.total else "spill runs")
    return job, prog


@app.command(short_help="Run a JSONL of prompts through a model",
             epilog="Example: spill run qwen2.5:0.5b sample")
def run(
    model: str = typer.Argument(..., help="curated tag (llama3.3:70b), HF repo id (org/name[@rev]), "
                                          "optionally +<adapter> (local dir or HF repo)"),
    input_jsonl: str = typer.Argument(..., help="prompts JSONL (plain, OpenAI batch or chat rows), or `sample`"),
    quant: str = typer.Option(None, "--quant",
                              help="8bit|4bit (mlx) or Q8_0|Q4_K_M (gguf); bf16 is the default"),
    out: Path = typer.Option(None, "--out", help="also copy results.jsonl here"),
    context: int = typer.Option(4096, "--context", help="context window in tokens"),
    parallel: int = typer.Option(None, "--parallel", hidden=True),
    quiet: bool = typer.Option(False, "--quiet", help="one progress line, no live slot block"),
    notify: str = typer.Option(None, "--notify", help="POST a small JSON to this URL when done"),
    logprobs: int = typer.Option(None, "--logprobs", help="per-token top-K log-probs, K up to 64"),
    state: str = typer.Option(None, "--state", help="portable job state: a path, s3://, gs://, "
                                                    "az:// (resumes from the latest checkpoint there)"),
    weights: str = typer.Option(None, "--weights", help="stage the model from this location "
                                                        "instead of Hugging Face"),
    engine: str = typer.Option(None, "--engine", help="mlx | torch-cpu | torch-cuda "
                                                      "(default: chosen from the hardware)"),
    headless: bool = typer.Option(False, "--headless", help="JSON-lines events on stdout, exit "
                                                            "75 when preempted (automatic when "
                                                            "stdout is not a terminal)"),
    config: Path = typer.Option(None, "--config", help="run the invocation stored in this "
                                                       "job.json"),
    emit_config: Path = typer.Option(None, "--emit-config", help="write this invocation to "
                                                                 "job.json and exit"),
    stop_after: int = typer.Option(None, "--stop-after", hidden=True),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Run a JSONL of prompts (or `sample`) through a model, a curated tag or any HF repo id."""
    global _DEBUG
    _DEBUG = debug
    with runtime.job_session("run", state=state, weights=weights, engine=engine,
                             headless_flag=headless, stop_after=stop_after):
        try:
            with _long_job("spill run " + model, notify):
                _run_impl(model, input_jsonl, quant, out, context, parallel, quiet,
                          RunOpts(logprobs=logprobs,
                                  prefix_reuse=not os.environ.get("SPILL_NO_PREFIX_REUSE")))
        except typer.Exit:
            raise
        except Exception as e:
            _fail(e)


def _run_impl(model, input_arg, quant, out, context, parallel, quiet,
              opts: RunOpts | None = None):
    """Returns (job, prog) on the MLX path (and the llama.cpp path)."""
    from . import formats
    from . import logits as lg
    from .adapters import resolve_adapter, split_model_adapter
    from .errors import SpillError
    from .resolve import arch_from_config, resolve_model
    opts = opts or RunOpts()
    scoring = opts.mode == "score"
    model, adapter_spec = split_model_adapter(model)
    opts.adapter = adapter_spec
    lg.validate_k(opts.logprobs)
    hw = probe_mod.load(probe_if_missing=True)
    reg = load_registry()
    res = resolve_model(model)
    input_jsonl = _resolve_input(input_arg)
    rows = formats.load_rows(input_jsonl, mode=opts.mode)
    max_tokens = max(r["body"].get("max_tokens", 128) for r in rows)
    gguf = quant not in (None, "bf16", "8bit", "4bit")      # an explicit GGUF quant: llama.cpp
    choice = None if gguf else engine_select.choose_engine(runtime.ENV.engine)
    if choice is not None:
        runtime.set_engine(choice.name)
    mlx_ok = choice is not None and choice.is_mlx
    if gguf and (adapter_spec or opts.logprobs or scoring):
        raise SpillError("adapters, --logprobs and --score need the MLX or torch engines "
                         "(bf16); the llama.cpp path (--quant Q8_0, Q4_K_M) does not support "
                         "them", "spill run <model> <file>")
    if choice is not None and choice.is_torch:
        if quant in ("8bit", "4bit"):
            raise SpillError(f"--quant {quant} runs on the MLX engine; the torch engines run "
                             f"bf16", "spill ... --engine mlx")
        from .cli_torch import run_torch
        return run_torch(res, input_jsonl, out, context, parallel, hw, rows, quiet, opts,
                         adapter_spec, choice)
    adapter = resolve_adapter(adapter_spec) if adapter_spec else None

    if mlx_ok:
        from .calibration import load_calibration
        arch = reg[model].arch if res.kind == "tag" else arch_from_config(res.config)
        downloaded = safetensors_downloaded(model) if res.kind == "tag" else (
            res.local_dir.exists() and any(res.local_dir.glob("*.safetensors")))
        est_batch = _rough_batch(res.st_bytes, arch, rows, max_tokens,
                                 hw["gpu"]["vram_bytes"])
        # scoring has no decode passes: one prefill pass per chunk, so the 24 h
        # drop rule must not be judged on a max_tokens the job will never generate
        choice = choose_quant_v2(res.st_bytes, downloaded,
                                 len(rows), 0 if scoring else max_tokens, est_batch,
                                 load_calibration(), hw, explicit=quant,
                                 rate_key=f"{res.name}|bf16")
        if choice.quant != "bf16":
            typer.echo(f"quant: {choice.reason}")
        if res.kind == "hf" and choice.quant in ("8bit", "4bit") and quant is None:
            typer.echo(f"note: no {choice.quant} artifact is registered for {res.name}; "
                       f"staying at bf16 (the drop rule applies to curated tags)")
            choice.quant = "bf16"
        return _run_mlx(res, input_jsonl, choice.quant, choice.reason, choice.est_seconds,
                        out, context, parallel, hw, rows, quiet=quiet, opts=opts,
                        adapter=adapter)
    model = res.name
    if res.kind == "hf":
        from .errors import SpillError
        raise SpillError("arbitrary HF repos run on the MLX and torch engines; the llama.cpp "
                         "path (--quant Q8_0, Q4_K_M) needs a curated tag", "spill models")

    # non-Apple path (or explicit GGUF quant): llama.cpp, Phase 0 policy
    m = reg[model]
    choice = choose_quant(m, hw, explicit=quant)
    if choice.quant != "bf16":
        typer.echo(f"quant: {choice.reason}")
    from .llamacpp import ensure_llama_server
    ensure_llama_server()
    try:
        paths = download(m, choice.quant)
    except RuntimeError as e:
        raise SpillError(str(e), "spill models")
    ram = hw["ram_total_bytes"]
    size = m.quants[choice.quant].bytes
    ngl, n = compute_offload(m, choice.quant, context, hw["gpu"]["vram_bytes"], parallel)
    fits = m.resident_bytes(choice.quant, context, n) <= ram
    nvme = hw["nvme_seq_read"]["bytes_per_sec"]
    est = len(rows) * max_tokens / max(n / (size / nvme), 1e-6) if not fits \
        else len(rows) * max_tokens / 150
    job = Job.create(input_jsonl, model, choice.quant, context, n, out, rows=rows,
                     options={"mode": "generate", "kind": "run"})
    out_path = out or job.results_path
    typer.echo(
        f"spill run {model}: {choice.quant} ({size / GIB:.1f} GB) "
        f"{'fits' if fits else 'does not fit'} in {ram / GIB:.0f} GB RAM, "
        f"{'resident' if fits else f'streaming from NVMe at ~{nvme / GIB:.1f} GB/s'}, via "
        f"llama.cpp. {len(rows)} prompts, batch {n}. Est. {_fmt_dur(est)}. Cost: $0. "
        f"Results -> {out_path}")
    from .engines.llamacpp import LlamaCppEngine
    from .jobs.runner import run_job
    spec = ModelSpec(model, choice.quant, paths[0], m.arch, context,
                     extra={"n_gpu_layers": ngl, "log_path": str(job.dir / "llama-server.log")})
    from . import runs as runs_mod
    runs_mod.start_run(job, command=None, spec=spec, input_path=input_jsonl, hw=hw,
                       engine_name="llamacpp")
    prog = run_job(job, LlamaCppEngine(), spec,
                   MemoryBudget(hw["gpu"]["vram_bytes"], batch_override=n), _progress_line)
    sys.stderr.write("\n")
    typer.echo(f"{prog.done}/{prog.total} rows, {prog.completion_tokens} completion tokens, "
               f"{prog.tokens_per_sec:.1f} tok/s aggregate")
    _next_hint(f"spill resume {job.id}" if prog.done < prog.total else "spill runs")
    return job, prog


@app.command(short_help="List past runs",
             epilog="Example: spill runs")
def runs(limit: int = typer.Option(20, "--limit", help="most recent N runs")):
    """List runs with model, quant, adapter, input hash, rows and status."""
    from . import runs as runs_mod
    ms = runs_mod.list_runs()
    if not ms:
        typer.echo("no runs yet")
        _next_hint("spill run qwen2.5:0.5b sample")
        return
    typer.echo(f"{'run':28s} {'kind':8s} {'model':22s} {'quant':6s} {'adapter':14s} "
               f"{'input':10s} {'rows':>9s} status")
    for m in ms[-limit:]:
        ad = (m.get("adapter") or {}).get("id") or "-"
        typer.echo(f"{m['id']:28s} {m.get('kind', 'run'):8s} {m['model']['id'][:22]:22s} "
                   f"{m['model']['quant']:6s} {ad[:14]:14s} {m['input']['sha256'][:8]:10s} "
                   f"{m.get('rows_done', 0)}/{m['input']['rows']:<6d} {m.get('status', '?')}")
    _next_hint(f"cat {runs_mod.manifest_path(ms[-1]['id'])}")


@app.command(short_help="Collect a big model's answers to learn from",
             epilog="Example: spill distill qwen2.5:0.5b sample")
def distill(
    teacher: str = typer.Argument(..., help="teacher model: tag or HF repo id (optionally +adapter)"),
    input_jsonl: str = typer.Argument(..., help="prompts (batch/chat JSONL), or with --score "
                                                 "chat rows that end with an assistant target"),
    score: bool = typer.Option(False, "--score",
                               help="teacher-forced: score the given assistant targets, "
                                    "prefill only, no sampling"),
    logprobs: int = typer.Option(32, "--logprobs", help="top-K per token, K up to 64"),
    quant: str = typer.Option(None, "--quant", help="8bit | 4bit; bf16 is the default"),
    out: Path = typer.Option(None, "--out", help="distillation JSONL (default runs/<id>/distill.jsonl)"),
    context: int = typer.Option(4096, "--context", help="context window in tokens"),
    quiet: bool = typer.Option(False, "--quiet", help="one progress line, no live slot block"),
    notify: str = typer.Option(None, "--notify", help="POST a small JSON to this URL when done"),
    state: str = typer.Option(None, "--state", help="portable job state: a path, s3://, gs://, "
                                                    "az:// (resumes from the latest checkpoint there)"),
    weights: str = typer.Option(None, "--weights", help="stage the model from this location "
                                                        "instead of Hugging Face"),
    engine: str = typer.Option(None, "--engine", help="mlx | torch-cpu | torch-cuda "
                                                      "(default: chosen from the hardware)"),
    headless: bool = typer.Option(False, "--headless", help="JSON-lines events on stdout, exit "
                                                            "75 when preempted (automatic when "
                                                            "stdout is not a terminal)"),
    config: Path = typer.Option(None, "--config", help="run the invocation stored in this "
                                                       "job.json"),
    emit_config: Path = typer.Option(None, "--emit-config", help="write this invocation to "
                                                                 "job.json and exit"),
    stop_after: int = typer.Option(None, "--stop-after", hidden=True),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Collect a big model's answers to learn from: completions plus per-token top-k
    log-probs, or its log-probs over targets you supply (--score)."""
    global _DEBUG
    _DEBUG = debug
    with runtime.job_session("distill", state=state, weights=weights, engine=engine,
                             headless_flag=headless, stop_after=stop_after):
        try:
            opts = RunOpts(mode="score" if score else "generate", logprobs=logprobs,
                           kind="distill")
            with _long_job("spill distill " + teacher, notify):
                job, prog = _run_impl(teacher, input_jsonl, quant, out, context, None,
                                      quiet, opts)
            if prog.done >= prog.total:
                dest = _finalize_distill(job)
                _next_hint(f"spill check {dest}" if not score else
                           f"spill runs")
            else:
                _next_hint(f"spill resume {job.id}")
        except typer.Exit:
            raise
        except Exception as e:
            _fail(e)


@app.command("eval", short_help="Score models on your eval set, one table",
             epilog="Example: spill eval banking77-quick/evals.jsonl qwen2.5:0.5b qwen2.5:0.5b+banking")
def eval_cmd(
    input_jsonl: str = typer.Argument(..., help="eval JSONL: prompts plus an `expected` field per row"),
    models: list[str] = typer.Argument(None, help="one or more models, each optionally +adapter; "
                                                  "omitted: every model with a run against this "
                                                  "file's hash"),
    metric: str = typer.Option("exact_match", "--metric",
                               help="exact_match | contains | regex | json_field | judge | script:<file.py>"),
    judge: str = typer.Option(None, "--judge", help="judge model (implies --metric judge)"),
    rerun: bool = typer.Option(False, "--rerun", help="ignore cached runs for this input hash"),
    quant: str = typer.Option(None, "--quant", help="8bit | 4bit; bf16 is the default"),
    context: int = typer.Option(4096, "--context", help="context window in tokens"),
    quiet: bool = typer.Option(False, "--quiet", help="one progress line, no live slot block"),
    notify: str = typer.Option(None, "--notify", help="POST a small JSON to this URL when done"),
    state: str = typer.Option(None, "--state", help="portable job state: a path, s3://, gs://, "
                                                    "az:// (resumes from the latest checkpoint there)"),
    weights: str = typer.Option(None, "--weights", help="stage the model from this location "
                                                        "instead of Hugging Face"),
    engine: str = typer.Option(None, "--engine", help="mlx | torch-cpu | torch-cuda "
                                                      "(default: chosen from the hardware)"),
    headless: bool = typer.Option(False, "--headless", help="JSON-lines events on stdout, exit "
                                                            "75 when preempted (automatic when "
                                                            "stdout is not a terminal)"),
    config: Path = typer.Option(None, "--config", help="run the invocation stored in this "
                                                       "job.json"),
    emit_config: Path = typer.Option(None, "--emit-config", help="write this invocation to "
                                                                 "job.json and exit"),
    stop_after: int = typer.Option(None, "--stop-after", hidden=True),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Run the eval set against each model (reusing finished runs for this exact input),
    score it, and print one table plus the rows where the models disagree."""
    global _DEBUG
    _DEBUG = debug
    with runtime.job_session("eval", state=state, weights=weights, engine=engine,
                             headless_flag=headless, stop_after=stop_after):
        try:
            with _long_job("spill eval " + Path(input_jsonl).name, notify):
                _eval_impl(input_jsonl, models, metric, judge, rerun, quant, context, None,
                           quiet)
        except typer.Exit:
            raise
        except Exception as e:
            _fail(e)


def _eval_impl(input_arg, models, metric, judge, rerun, quant, context, parallel, quiet,
               echo=True, engine_pure: bool = False):
    """Returns the list of ScoredRun, one per model. `engine_pure` (used by `spill build`)
    reuses a finished run only when the same engine family produced it, so a build on
    torch-cpu is never answered by an MLX run from an earlier build."""
    import uuid

    from . import evalrun, formats
    from . import runs as runs_mod
    from .adapters import resolve_adapter, split_model_adapter
    from .errors import SpillError
    from .metrics import (build_judge_messages, get_metric, parse_judge_score,
                          response_text)
    from .resolve import resolve_model

    path = _resolve_input(input_arg)
    rows = formats.load_rows(path)
    expected = {r["custom_id"]: r["expected"] for r in rows if "expected" in r}
    if not models:
        models = runs_mod.models_with_runs(runs_mod.input_hash(path))
        if not models:
            raise SpillError(f"no model has a run against {Path(path).name} yet",
                             f"spill eval {input_arg} <model-a> <model-b>")
        typer.echo(f"models with a run against this file: {', '.join(models)}")
    if judge:
        metric = "judge"
    if metric == "judge" and not judge:
        raise SpillError("--metric judge needs --judge <model>", "spill eval <file> <model> "
                         "--judge <model>")
    if not expected and not metric.startswith("script:"):
        raise SpillError('this file has no "expected" fields, so there is nothing to score; '
                         'add one per row (see the README, Formats)', f"spill check {path}")
    metric_fn = None if metric == "judge" else get_metric(metric)
    in_hash = runs_mod.input_hash(path)
    eval_id = "eval-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    started = runs_mod.now()
    command = "spill " + " ".join(sys.argv[1:])
    dest = runs_mod.run_dir(eval_id)
    dest.mkdir(parents=True, exist_ok=True)

    scored, run_ids = [], []
    for n, label in enumerate(models):
        if runtime.ENV.state:
            runtime.ENV.state_leaf = runtime.slug(label)
        base, ad_spec = split_model_adapter(label)
        ad_hash = resolve_adapter(ad_spec).hash if ad_spec else None
        res = resolve_model(base)
        family = (engine_select.choose_engine(runtime.ENV.engine).name.split("-")[0]
                  if engine_pure else None)
        m = None if rerun else runs_mod.find_cached(res.name, quant, ad_hash, in_hash,
                                                    {"mode": "generate"}, engine_family=family)
        cached = m is not None
        if cached:
            typer.echo(f"{label}: reusing run {m['id']} (same model, adapter and input hash; "
                       f"--rerun forces)")
        else:
            with _no_next_hints():
                job, prog = _run_impl(label, str(path), quant, None, context, parallel,
                                      quiet, RunOpts(cmd="eval"))
            if prog.done < prog.total:
                from .errors import StageInterrupted
                raise StageInterrupted(job.id, prog.done, prog.total, f"{label}:")
            m = runs_mod.read_manifest(job.id)
        results = evalrun.read_results(Path(m["job_dir"]) / "results.jsonl")
        judge_scores = None
        if metric == "judge":
            jrows = [{"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                      "body": {"messages": build_judge_messages(
                          next(r for r in rows if r["custom_id"] == cid)["body"]["messages"],
                          res_row, expected.get(cid)), "max_tokens": 8}}
                     for cid, res_row in results.items()]
            jpath = dest / f"judge-{n}.jsonl"
            jpath.write_text("".join(json.dumps(r) + "\n" for r in jrows))
            typer.echo(f"judging {label} with {judge}")
            with _no_next_hints():
                jjob, jprog = _run_impl(judge, str(jpath), None, None, context, parallel, quiet,
                                        RunOpts(kind="judge", cmd="eval"))
            if jprog.done < jprog.total:
                raise SpillError(f"judge run stopped at {jprog.done}/{jprog.total}",
                                 f"spill resume {jjob.id}")
            jres = evalrun.read_results(jjob.results_path)
            judge_scores = {cid: parse_judge_score(response_text(r))
                            for cid, r in jres.items()}
        sr = evalrun.score_results(results, expected, metric_fn, judge_scores)
        sr.label, sr.run_id, sr.cached = label, m["id"], cached
        sr.model, sr.quant = m["model"]["id"], m["model"]["quant"]
        sr.adapter = (m.get("adapter") or {}).get("id")
        scored.append(sr)
        run_ids.append(m["id"])

    diff = evalrun.build_diff(scored, rows)
    metric_label = metric + (f" ({judge})" if judge else "")
    evalrun.write_report(dest, scored, metric_label, str(path), in_hash, diff)
    if echo:
        typer.echo("")
        typer.echo(f"eval {Path(path).name}  metric: {metric_label}  rows: {len(rows)}")
        typer.echo(evalrun.render_table(scored, metric_label))
        typer.echo(f"\n{len(diff)} rows where the models disagree -> {dest / 'diff.jsonl'}"
                   if len(scored) > 1 else
                   f"\n(one model: no comparison; table -> {dest / 'table.md'})")
    runs_mod.write_manifest(eval_id, {
        "id": eval_id, "kind": "eval", "command": command,
        "model": {"id": ", ".join(models), "quant": "-", "weight_hash": ""},
        "adapter": None,
        "input": {"path": str(path), "sha256": in_hash, "rows": len(rows)},
        "metric": metric_label, "runs": run_ids, "cached": [sr.cached for sr in scored],
        "disagreements": len(diff), "started": started, "ended": runs_mod.now(),
        "status": "completed", "rows_done": len(rows),
        "table": evalrun.render_table(scored, metric_label, markdown=True)})
    if echo:
        _next_hint(f"head -n 3 {dest / 'diff.jsonl'}" if len(scored) > 1
                   else f"spill eval {input_arg} {models[0]} <another-model>")
    return scored


@app.command(short_help="Train a LoRA adapter on your data",
             epilog="Example: spill tune qwen2.5:0.5b banking77-quick/train.jsonl --name banking")
def tune(
    model: str = typer.Argument(..., help="base model: tag, Hugging Face repo id, or a local "
                                          "safetensors directory"),
    train_jsonl: Path = typer.Argument(..., help="training JSONL: {prompt, answer} rows, or "
                                                 "OpenAI chat rows (messages)"),
    name: str = typer.Option(..., "--name", help="adapter name (see: spill adapters)"),
    rank: int = typer.Option(16, "--rank", help="LoRA rank"),
    lr: float = typer.Option(1e-4, "--lr", help="learning rate"),
    steps: int = typer.Option(None, "--steps", help="optimizer steps (default: --epochs of the data)"),
    epochs: float = typer.Option(1.0, "--epochs", help="passes over the data"),
    batch: int = typer.Option(None, "--batch", help="micro-batch (default: 4 resident, sized "
                                                    "from the memory budget when streamed)"),
    grad_accum: int = typer.Option(1, "--grad-accum", help="micro-batches per optimizer step"),
    max_seq: int = typer.Option(2048, "--max-seq", help="token cap per example; whole "
                                "exchanges are dropped from the left"),
    targets: str = typer.Option(None, "--targets", help="comma-separated modules to adapt, "
                                "e.g. q_proj,v_proj (default: every linear)"),
    path: str = typer.Option("auto", "--path", help="auto | resident | streamed"),
    overwrite: bool = typer.Option(False, "--overwrite", help="replace an existing adapter"),
    quiet: bool = typer.Option(False, "--quiet", help="a progress line every 10 steps"),
    notify: str = typer.Option(None, "--notify", help="POST a small JSON to this URL when done"),
    state: str = typer.Option(None, "--state", help="portable job state: a path, s3://, gs://, "
                                                    "az:// (resumes from the latest checkpoint there)"),
    weights: str = typer.Option(None, "--weights", help="stage the model from this location "
                                                        "instead of Hugging Face"),
    engine: str = typer.Option(None, "--engine", help="mlx | torch-cpu | torch-cuda "
                                                      "(default: chosen from the hardware)"),
    headless: bool = typer.Option(False, "--headless", help="JSON-lines events on stdout, exit "
                                                            "75 when preempted (automatic when "
                                                            "stdout is not a terminal)"),
    config: Path = typer.Option(None, "--config", help="run the invocation stored in this "
                                                       "job.json"),
    emit_config: Path = typer.Option(None, "--emit-config", help="write this invocation to "
                                                                 "job.json and exit"),
    stop_after: int = typer.Option(None, "--stop-after", hidden=True),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Train a LoRA adapter on your data against the full-precision base."""
    global _DEBUG
    _DEBUG = debug
    with runtime.job_session("tune", state=state, weights=weights, engine=engine,
                             headless_flag=headless, stop_after=stop_after):
        try:
            with _long_job("spill tune " + name, notify):
                _tune_impl(model, train_jsonl, name, rank, 2 * rank, 0.0, targets, lr, "cosine",
                           0.01, steps, epochs, batch, grad_accum, max_seq, 0, path, 50,
                           overwrite, quiet)
        except typer.Exit:
            raise
        except Exception as e:
            _fail(e)


def _tune_model_dir(model: str) -> tuple[str, Path]:
    """(display name, safetensors dir) for a tag, HF repo id, or local directory."""
    from .errors import SpillError
    from .resolve import download_hf, resolve_model
    if runtime.ENV.weights:
        from .portable.weights import stage_weights
        d, staged = stage_weights(runtime.ENV.weights,
                                  note=lambda m: typer.echo(f"   {m}", err=True))
        typer.echo(f"   weights staged from {staged['uri']}: {staged['bytes'] / 1e9:.2f} GB "
                   f"copied in {staged['seconds']:.1f} s ({staged['reused']} files already "
                   f"here)", err=True)
        return model, d
    p = Path(model).expanduser()
    if p.is_dir():
        if not (p / "config.json").exists() or not any(p.glob("*.safetensors")):
            raise SpillError(f"{p} is not a safetensors model directory (needs config.json "
                             f"and *.safetensors)", "spill models")
        return p.name, p
    res = resolve_model(model)
    if res.kind == "hf":
        return res.name, download_hf(res)
    return res.name, download_safetensors(res.name)


def _tune_pre_line(prep, size_gb: float, ws: int, ram: int, dest: Path) -> None:
    from .engines.supported import classify
    s, st = prep.spec, prep.stats
    fam = classify(prep.config)
    tg = ",".join(sorted({p.rsplit(".", 1)[-1] for p in prep.shapes}))
    if s.engine == "mlx":
        placement = ("streamed from NVMe, two weight streams per micro-batch"
                     if s.path == "streamed" else "resident (mlx-lm LoRA tuner)")
    else:
        placement = (f"streamed from disk on {s.engine}, two weight streams per micro-batch"
                     if s.path == "streamed" else f"resident on {s.engine} (PEFT LoRA layers)")
        placement += f"; {prep.dtype_note}; adapter and optimizer float32"
    trunc = (f", {st.truncated} truncated" if st.truncated else "") + \
            (f", {len(st.skipped)} skipped" if st.skipped else "")
    line = (
        f"spill tune {s.model}: bf16 ({size_gb:.1f} GB, {fam.label}: {fam.state}), {placement}. "
        f"Adapter {s.name}: rank {s.rank} alpha {s.alpha:g} dropout {s.dropout:g} on {tg} in "
        f"{prep.n_layers} layers ({prep.lora_params / 1e6:.1f}M parameters), lr {s.lr:g} "
        f"{s.schedule}, AdamW. {st.examples} examples, {st.tokens:,} tokens "
        f"({st.trained_tokens:,} trained) at --max-seq {s.max_seq}{trunc}. "
        f"{s.steps} steps of micro-batch {s.micro_batch} x grad-accum {s.grad_accum}. "
        f"Est. {_fmt_dur(prep.est_total_s)} ({prep.est_note}). Cost: $0."
        f"{'' if runtime.ENV.headless else overnight.battery_note()} Adapter -> {dest}")
    typer.echo(line)
    typer.echo(f"   why: {prep.why}")
    for ln, why in st.skipped[:5]:
        typer.echo(f"   skipped line {ln}: {why}", err=True)
    return line


def _tune_progress(quiet: bool):
    if runtime.ENV.headless:
        def hcb(i):
            runtime.ENV.events.step(i)
        return hcb

    def cb(i):
        line = (f"step {i['step']}/{i['steps']} \u00b7 loss {i['loss']:.4g} \u00b7 "
                f"{i['avg_step_s']:.3g} s/step \u00b7 ETA {_fmt_eta(i['eta_s'])} \u00b7 "
                f"{i['peak_gb']:.1f} GB peak")
        if quiet:
            if i["step"] % 10 == 0 or i["step"] == i["steps"]:
                sys.stderr.write(line + "\n")
        else:
            sys.stderr.write("\r" + line + "\x1b[K")
        sys.stderr.flush()
    return cb


def _tune_module(engine_name: str):
    """tune.job (MLX) or tune.torch_job (PyTorch): the same prepare / run_tune surface."""
    if engine_name == "mlx":
        from .tune import job as m
    else:
        from .tune import torch_job as m
    return m


def _tune_run(prep, job, resume: bool, quiet: bool, hw: dict, base_label: str):
    import signal
    import threading

    from . import runs as runs_mod
    run_tune = _tune_module(prep.spec.engine).run_tune
    guard = runtime.ENV.guard
    stop = runtime.ENV.stop if guard is not None else threading.Event()
    n_int = {"n": 0}

    def on_sig(*_):
        n_int["n"] += 1
        if n_int["n"] == 1:
            stop.set()
            sys.stderr.write("\nspill: interrupt received, finishing the current step and "
                             "checkpointing (Ctrl-C again aborts to the last checkpoint)\n")
        else:
            raise KeyboardInterrupt
    prev = {}
    if guard is None:                 # a person at a terminal; headless jobs use the guard
        for sg in (signal.SIGINT, signal.SIGTERM):
            try:
                prev[sg] = signal.signal(sg, on_sig)
            except ValueError:
                pass

    def on_event(ev: dict):
        ev = dict(ev)
        runtime.emit(ev.pop("event"), **ev)
    try:
        res = run_tune(prep, job, stop=stop, progress_cb=_tune_progress(quiet), resume=resume,
                       note=lambda m: typer.echo(f"   {m}", err=True), on_event=on_event)
    except Exception as e:
        if any(k in str(e) for k in ("Insufficient Memory", "kIOGPU", "OutOfMemory",
                                     "metal::malloc", "CUDA out of memory")):
            from .errors import SpillError
            mb, ga = prep.spec.micro_batch, prep.spec.grad_accum
            raise SpillError(
                f"the device ran out of memory at micro-batch {mb}; the budget estimate was too "
                f"optimistic for this model and sequence length (checkpoints are kept)",
                f"spill tune ... --batch {max(1, mb // 2)} --grad-accum {ga * 2}") from e
        raise
    except KeyboardInterrupt:
        job.write_meta(status="interrupted")
        runs_mod.finish_run(job.id, status="interrupted", rows_done=job.read_meta().get("done", 0))
        sys.stderr.write("\n")
        typer.echo(f"aborted; the last checkpoint is kept")
        _next_hint(f"spill resume {job.id}")
        raise typer.Exit(130)
    finally:
        for sg, h in prev.items():
            try:
                signal.signal(sg, h)
            except ValueError:
                pass
    sys.stderr.write("\n")
    if res["interrupted"] and prep.spec.stop_after and res["step"] >= prep.spec.stop_after:
        runtime.ENV.stopped_early = True
    runtime.ENV.summary.update(step=res["step"], steps=res["steps"], job=job.id,
                               first_loss=res["first_loss"], final_loss=res["final_loss"],
                               adapter=res.get("adapter"))
    runs_mod.finish_run(job.id, status="interrupted" if res["interrupted"] else "completed",
                        rows_done=res["step"])
    return res


def _tune_report(prep, job, res, train_path: Path | None = None) -> None:
    s = prep.spec
    first, last = res["first_loss"], res["final_loss"]
    typer.echo(f"{res['step']}/{res['steps']} steps in {_fmt_dur(res['seconds'])}, loss "
               f"{first:.4g} -> {last:.4g}" + (f", {res['peak_gb']:.1f} GB peak"
                                               if res["peak_gb"] else ""))
    ss = prep.stream_stats
    if ss and ss.get("read_seconds"):
        typer.echo(f"   weight stream: {ss['read_bytes'] / GIB:.0f} GB read, "
                   f"{ss['read_bytes'] / ss['read_seconds'] / GIB:.2f} GB/s while reading; "
                   f"{res['seconds'] / max(1, res['step']):.1f} s/step = "
                   f"{s.grad_accum} x (2 weight streams + compute)")
    if res.get("tflops_overall"):
        tc = res.get("tflops_compute")
        typer.echo(f"   achieved {res['tflops_overall']:.2f} TFLOP/s overall"
                   + (f", {tc:.2f} while computing" if tc else "")
                   + f"; {res['padded_tokens_per_s']:.0f} tokens/s through the layers")
    if res["interrupted"]:
        typer.echo(f"interrupted at step {res['step']}; checkpoint saved")
        _next_hint(f"spill resume {job.id}")
    else:
        typer.echo(f"adapter saved: {res['adapter']} (PEFT and mlx-lm layouts)")
        sibling = (Path(train_path).parent / "evals.jsonl") if train_path else Path("evals.jsonl")
        _next_hint(f"spill eval {sibling if sibling.exists() else 'evals.jsonl'} "
                   f"{s.model} {s.model}+{s.name}")


def _tune_impl(model, train_jsonl, name, rank, alpha, dropout, targets, lr, schedule,
               weight_decay, steps, epochs, batch, grad_accum, max_seq, seed, path,
               ckpt_every, overwrite, quiet):
    from . import adapters as adapters_mod
    from . import runs as runs_mod
    from .calibration import load_calibration
    from .errors import SpillError
    from .portable.store import is_uri
    if is_uri(str(train_jsonl)):
        train_jsonl = _fetch_input(str(train_jsonl))
    train_jsonl = Path(train_jsonl)
    if not train_jsonl.exists():
        raise SpillError(f"training file {train_jsonl} does not exist",
                         "spill example banking77 --quick")
    if path not in ("auto", "resident", "streamed"):
        raise SpillError("--path must be auto, resident or streamed")
    from .tune.data import distill_as_training, is_distill_file
    if is_distill_file(train_jsonl):                 # the output of `spill distill` trains as it is
        import tempfile
        train_jsonl, n_d = distill_as_training(
            train_jsonl, Path(tempfile.mkdtemp(prefix="spill-distill-")) / "train.jsonl")
        typer.echo(f"   training on {n_d} teacher answers from the distill file")
    choice = engine_select.choose_engine(runtime.ENV.engine)
    runtime.set_engine(choice.name)
    tj = _tune_module(choice.name)
    dest = adapters_mod.ADAPTERS_DIR / name
    if dest.exists() and not overwrite and not runtime.ENV.state:
        raise SpillError(f"adapter {name} already exists at {dest}",
                         f"spill tune ... --name {name}-2   (or --overwrite)")
    hw = probe_mod.load()
    if choice.is_mlx:
        ws = hw["gpu"]["vram_bytes"]
    else:
        from .engines.torch_common import memory_total_bytes
        ws = memory_total_bytes(choice.name)
    label, mdir = _tune_model_dir(model)
    size = sum(f.stat().st_size for f in mdir.glob("*.safetensors"))
    chosen = tj.decide_path(size, ws, path)
    spec = tj.TuneSpec(
        model=label, quant="bf16", model_dir=str(mdir), data=str(train_jsonl), name=name,
        path=chosen, rank=rank, alpha=alpha, dropout=dropout,
        targets=[t.strip() for t in targets.split(",")] if targets else None, lr=lr,
        weight_decay=weight_decay, schedule=schedule, seed=seed, max_seq=max_seq,
        micro_batch=batch or 4, grad_accum=grad_accum, steps=steps or 1, epochs=epochs,
        ckpt_every=ckpt_every, overwrite=overwrite, engine=choice.name,
        state=runtime.state_uri(), stop_after=runtime.ENV.stop_after)
    if spec.grad_accum < 1:
        raise SpillError("--grad-accum must be at least 1")
    prep = tj.prepare(spec, working_set=ws, calibration=load_calibration(), hw=hw,
                      micro_batch_given=batch is not None, steps_given=steps is not None)
    line = _tune_pre_line(prep, size / GIB, ws, hw["ram_total_bytes"], dest)

    job = Job.create(train_jsonl, label, "bf16", spec.max_seq, spec.micro_batch, None,
                     options={"kind": "tune", "tune": spec.to_dict()})
    typer.echo(f"   job {job.id}: Ctrl-C checkpoints and stops; resume with: spill resume {job.id}")
    spec.data = str(job.input_path)          # the job's own verbatim copy; resume is self-contained
    job.write_meta(total=spec.steps, done=0,
                   options={**job.read_meta().get("options", {}), "kind": "tune",
                            "tune": spec.to_dict()})
    mspec = ModelSpec(label, "bf16", mdir)
    engine_label = (("mlx_stream_tune" if chosen == "streamed" else "mlx_lm_lora")
                    if choice.is_mlx else
                    ("torch_stream_tune" if chosen == "streamed" else "torch_peft_lora"))
    runs_mod.start_run(job, command=None, spec=mspec, input_path=train_jsonl, hw=hw,
                       engine_name=engine_label,
                       options={"kind": "tune", "tune": spec.to_dict()}, kind="tune")
    prep.spec = spec
    runtime.emit("start", job=job.id, command="tune", model=label, steps=spec.steps,
                 placement=chosen, state=runtime.state_uri(), message=line,
                 estimated_seconds=round(prep.est_total_s))
    res = _tune_run(prep, job, False, quiet, hw, label)
    _tune_report(prep, job, res, train_jsonl)
    return job, res


def _tune_build(model, train, name, epochs=1.0, quiet=True, batch=None, grad_accum=1):
    """`spill tune` as one stage of `spill build`: defaults, the adapter replaced on a
    re-run, the stage's own output kept off the terminal (build prints one line per stage).
    The stage's state is runtime.ENV.state, so a second call continues from the checkpoint
    there, on any engine."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        job, res = _tune_impl(model, train, name, 16, 32.0, 0.0, None, 1e-4, "cosine", 0.01,
                              None, epochs, batch, grad_accum, 2048, 0, "auto", 50, True, quiet)
    tune_opts = job.read_meta().get("options", {}).get("tune", {})
    out = {"job_id": job.id, "micro_batch": tune_opts.get("micro_batch"),
           "grad_accum": tune_opts.get("grad_accum")}
    if res["interrupted"]:
        return {"interrupted": True, **out, "step": res["step"], "steps": res["steps"]}
    return {**out, "adapter": res["adapter"], "steps": res["steps"],
            "final_loss": res["final_loss"]}


def _tune_resume(j: Job, meta: dict) -> None:
    from .calibration import load_calibration
    from .tune.spec import TuneSpec
    hw = probe_mod.load()
    spec = TuneSpec.from_dict(meta["options"]["tune"])
    if runtime.ENV.engine:                         # resume on another engine
        spec.engine = engine_select.choose_engine(runtime.ENV.engine).name
    tj = _tune_module(spec.engine)
    if spec.engine == "mlx":
        ws = hw["gpu"]["vram_bytes"]
    else:
        from .engines.torch_common import memory_total_bytes
        ws = memory_total_bytes(spec.engine)
    runtime.set_engine(spec.engine)
    if runtime.ENV.stop_after:
        spec.stop_after = runtime.ENV.stop_after
    prep = tj.prepare(spec, working_set=ws, calibration=load_calibration(),
                      hw=hw, micro_batch_given=True, steps_given=True)
    from . import runs as runs_mod
    runs_mod.mark_resumed(j.id)
    typer.echo(f"resuming {j.id}: tune {spec.model} -> {spec.name}, {spec.path} path")
    res = _tune_run(prep, j, True, False, hw, spec.model)
    _tune_report(prep, j, res)



@app.command(short_help="List your trained adapters",
             epilog="Example: spill adapters")
def adapters():
    """List the LoRA adapters you have trained or added (PEFT or mlx-lm layout)."""
    from .adapters import ADAPTERS_DIR, list_local_adapters
    found = list_local_adapters()
    if not found:
        typer.echo(f"no adapters under {ADAPTERS_DIR}")
        typer.echo("any directory or Hugging Face repo with adapters.safetensors (mlx-lm) or "
                   "adapter_model.safetensors (PEFT) works directly: "
                   "spill run <base>+<adapter> <file>")
        _next_hint("spill run qwen2.5:0.5b+./my-adapter sample")
        return
    typer.echo(f"{'adapter':24s} {'layout':8s} {'rank':>4s} {'size':>9s} base")
    for a in found:
        typer.echo(f"{a['id'][:24]:24s} {a['layout']:8s} {str(a['rank']):>4s} "
                   f"{a['bytes'] / 1e6:7.1f}MB {a['base'] or '-'}")
    _next_hint(f"spill run <base>+{found[0]['id']} sample")


@app.command(short_help="Validate a JSONL file, with line-numbered errors",
             epilog="Example: spill check banking77-quick/evals.jsonl")
def check(path: Path = typer.Argument(..., help="a batch, chat, eval or distillation-target JSONL")):
    """Validate a JSONL file; prints the first error with its line number."""
    from .errors import SpillError
    from .formats import check_file
    try:
        info = check_file(path)
    except SpillError as e:
        typer.echo(f"spill: {path}: {e.line(default=f'spill check {path}')}", err=True)
        raise typer.Exit(1)
    typer.echo(f"{path}: ok, {info['rows']} rows, {info['shape']} lines, use: {info['use']}")
    nxt = {"eval": f"spill eval {path} <model-a> <model-b>",
           "train/distill targets": f"spill tune <base-model> {path} --name <name>",
           "distillation records": f"spill tune <base-model> {path} --name <name>",
           "prompts": f"spill run <model> {path}"}[info["use"]]
    _next_hint(nxt)


@app.command(short_help="Follow the results of the latest job",
             epilog="Example: spill tail")
def tail(job: str = typer.Argument(None, help="job id (default: the latest job)")):
    """Follow results.jsonl of the latest (or named) job."""
    j = Job.load(job) if job else Job.latest()
    if not j:
        _fail(SpillError("no jobs yet", "spill run qwen2.5:0.5b sample"))
    typer.echo(f"tailing {j.results_path}  (^C to stop)")
    pos = 0
    live = _LiveRenderer()
    try:
        while True:
            meta0 = j.read_meta()
            lp = j.dir / "live.json"
            if meta0.get("status") == "running" and lp.exists():
                try:
                    live(json.loads(lp.read_text()))
                except (json.JSONDecodeError, OSError):
                    pass
            if j.results_path.exists():
                with open(j.results_path) as f:
                    f.seek(pos)
                    for line in f:
                        sys.stdout.write(line)
                    pos = f.tell()
                sys.stdout.flush()
            meta = j.read_meta()
            if meta.get("status") in ("completed", "failed", "interrupted") and pos:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    _next_hint("spill status")


@app.command(short_help="Continue an interrupted job or build",
             epilog="Example: spill resume banking77-quick")
def resume(job: str = typer.Argument(None, help="job id or build folder (default: the latest job)"),
           headless: bool = typer.Option(False, "--headless", help="JSON-lines events on stdout, "
                                                                   "exit 75 when preempted"),
           weights: str = typer.Option(None, "--weights", help="stage the model from this "
                                                               "location instead of Hugging Face"),
           state: str = typer.Option(None, "--state", help="portable state to continue from: a "
                                                           "path, s3://, gs://, az:// (a build "
                                                           "folder kept its own)"),
           engine: str = typer.Option(None, "--engine", help="mlx | torch-cpu | torch-cuda "
                                                             "(default: the one this machine "
                                                             "would choose; any engine "
                                                             "continues any engine's work)"),
           stop_after: str = typer.Option(None, "--stop-after", hidden=True),
           notify: str = typer.Option(None, "--notify", help="POST a small JSON to this URL "
                                                             "when a build is done or stopped"),
           debug: bool = typer.Option(False, "--debug", hidden=True)):
    """Continue the latest or named job from its checkpoint; a folder continues its build."""
    global _DEBUG
    _DEBUG = debug
    from .cli_project import is_project_target
    if is_project_target(job):
        try:
            from .cli_project import project_resume
            project_resume(job, engine, headless, "local", stop_after)
        except typer.Exit:
            raise
        except Exception as e:
            _fail(e)
        return
    is_build = bool(job) and Path(job).is_dir() and (
        (Path(job) / ".build" / "state.json").exists()
        or (state and (Path(job) / "evals.jsonl").exists()))
    with runtime.job_session("build" if is_build else "resume", state=None if is_build else state,
                             weights=weights, engine=engine, headless_flag=headless):
        try:
            if is_build:
                from .cli_build import resume_build
                resume_build(Path(job), state=state, stop_after=stop_after, notify=notify)
            else:
                _resume_impl(job)
        except typer.Exit:
            raise
        except Exception as e:
            _fail(e)


def _resume_impl(job: str | None):
    j = Job.load(job) if job else Job.latest()
    if not j:
        _fail(SpillError("no job to resume", "spill run qwen2.5:0.5b sample"))
    meta0 = j.read_meta()
    if meta0.get("options", {}).get("kind") == "tune":
        if meta0.get("status") == "completed":
            typer.echo(f"{j.id}: already complete ({meta0.get('done')}/{j.total} steps)")
            _next_hint("spill adapters")
            return
        _tune_resume(j, meta0)
        return
    done = len(j.done_ids())
    if done >= j.total:
        typer.echo(f"{j.id}: already complete ({done}/{j.total})")
        _next_hint("spill status")
        return
    typer.echo(f"resuming {j.id}: {done}/{j.total} done, {j.total - done} remaining")
    hw = probe_mod.load()
    quant = j.quant
    o = meta0.get("options", {})
    if o.get("state") and not runtime.ENV.state:
        runtime.ENV.state = o["state"]
    ch = runtime.ENV.engine or o.get("engine_choice")
    if ch and ch.startswith("torch"):
        from .cli_torch import resume_torch
        runtime.emit("start", job=j.id, command="resume", rows=j.total, done=done,
                     state=runtime.state_uri())
        resume_torch(j, hw, ch)
    elif (_is_mac() or ch == "mlx") and quant in ("bf16", "8bit", "4bit"):
        runtime.emit("start", job=j.id, command="resume", rows=j.total, done=done,
                     state=runtime.state_uri())
        _run_mlx_resume(j, quant, hw)
    else:
        reg = load_registry()
        m = reg[j.model]
        paths = m.quants[quant].local_paths(j.model)
        ngl, _ = compute_offload(m, quant, j.ctx, hw["gpu"]["vram_bytes"], j.parallel)
        from .engines.llamacpp import LlamaCppEngine
        from .jobs.runner import run_job
        spec = ModelSpec(j.model, quant, paths[0], m.arch, j.ctx,
                         extra={"n_gpu_layers": ngl,
                                "log_path": str(j.dir / "llama-server.log")})
        prog = run_job(j, LlamaCppEngine(), spec,
                       MemoryBudget(hw["gpu"]["vram_bytes"], batch_override=j.parallel),
                       _progress_line)
        sys.stderr.write("\n")
        typer.echo(f"{prog.done}/{prog.total} rows complete")
    _next_hint("spill status")


def _teacher_info(m: dict) -> dict:
    return {"model": m["model"]["id"], "quant": m["model"]["quant"],
            "weight_hash": m["model"]["weight_hash"][:16],
            "adapter_hash": (m["adapter"]["hash"][:16] if m.get("adapter") else None),
            "run_id": m["id"]}


def _finalize_distill(j: Job) -> Path | None:
    """Write runs/<id>/distill.jsonl (and --out) once a distill run is complete."""
    from . import runs as runs_mod
    from .distill import write_distill
    meta = j.read_meta()
    o = meta.get("options", {})
    if o.get("kind") != "distill":
        return None
    dest = Path(meta["out"]) if meta.get("out") else runs_mod.run_dir(j.id) / "distill.jsonl"
    n = write_distill(j.dir, dest, _teacher_info(runs_mod.read_manifest(j.id)),
                      o.get("mode", "generate"))
    typer.echo(f"wrote {n} distillation records -> {dest}")
    return dest


def _run_mlx_resume(j: Job, quant: str, hw: dict) -> None:
    from .adapters import resolve_adapter
    from .engines.mlx_resident import MlxResidentEngine
    from .engines.mlx_stream import MlxStreamEngine
    from .jobs.runner import run_job
    from .registry import MODELS_DIR
    reg = load_registry()
    ws = hw["gpu"]["vram_bytes"]
    if quant == "bf16":
        path = safetensors_dir(j.model)
        size = safetensors_spec(j.model)["bytes"]
    else:
        path = MODELS_DIR / j.model.replace(":", "-") / f"mlx-{quant}"
        size = sum(f.stat().st_size for f in path.glob("*.safetensors"))
    engine = MlxResidentEngine() if size <= ws * 0.70 else \
        MlxStreamEngine(progress_note=lambda s: typer.echo(f"   {s}", err=True))
    engine.pass_cb = _live_cb(j.dir, False)
    spec = ModelSpec(j.model, quant, Path(path),
                     reg[j.model].arch if j.model in reg else {}, j.ctx)
    o = j.read_meta().get("options", {})
    if o.get("logprobs") or o.get("mode") == "score":
        spec.extra["logprobs"] = o.get("logprobs") or 32
    if o.get("mode") == "score":
        spec.extra["mode"] = "score"
    if o.get("adapter"):
        spec.extra["adapter"] = resolve_adapter(o["adapter"])
    prog = run_job(j, engine, spec, MemoryBudget(ws, batch_override=None), progress_cb=None)
    sys.stderr.write("\n")
    typer.echo(f"{prog.done}/{prog.total} rows complete")
    if prog.done >= prog.total:
        _finalize_distill(j)


@app.command(short_help="Show jobs: progress, tokens/s, ETA",
             epilog="Example: spill status")
def status():
    """List jobs with progress, tokens/s, ETA."""
    if not JOBS_DIR.exists() or not any(JOBS_DIR.iterdir()):
        typer.echo("no jobs")
        _next_hint("spill run qwen2.5:0.5b sample")
        return
    metas = []
    for d in sorted(JOBS_DIR.iterdir()):
        mp = d / "meta.json"
        if mp.exists():
            metas.append((d, json.loads(mp.read_text())))
    w = max([14] + [len(m["model"]) for _, m in metas])
    typer.echo(f"{'job':28s} {'model':{w}s} {'quant':7s} {'progress':12s} {'tok/s':>8s} {'eta':>8s} status")
    for d, meta in metas:
        done, total = meta.get("done", 0), meta.get("total", 0)
        eta = meta.get("eta_seconds")
        typer.echo(f"{meta['id']:28s} {meta['model']:{w}s} {meta['quant']:7s} "
                   f"{f'{done}/{total}':12s} {meta.get('tokens_per_sec', 0):8.1f} "
                   f"{_fmt_dur(eta) if eta else '-':>8s} {meta.get('status', '?')}")
        if meta.get("status") == "running" and (d / "live.json").exists():
            try:
                _LiveRenderer()(json.loads((d / "live.json").read_text()))
            except (json.JSONDecodeError, OSError, KeyError):   # a tune job's or a killed job's live.json
                pass
    _next_hint("spill tail")


PARAMS = {"qwen2.5:0.5b": "0.5B", "qwen2.5:7b": "7B", "qwen2.5:32b": "32B", "llama3.3:70b": "70B"}
MLX8_BYTES = {"qwen2.5:0.5b": 700_000_000, "qwen2.5:7b": 8_091_987_725, "qwen2.5:32b": 35_000_000_000,
              "llama3.3:70b": 75_000_000_000}


@app.command(short_help="List the models spill knows and what they need",
             epilog="Example: spill models")
def models(architectures: bool = typer.Option(False, "--architectures",
                                              help="print the architecture support table")):
    """Curated tags: size, family, placement on this machine, disk needed, downloaded."""
    from .engines.supported import FAMILIES, table
    if architectures:
        typer.echo(table())
        typer.echo('\nAny Hugging Face repo id with a supported architecture also works: '
                   'spill run org/name sample.')
        return
    hw = probe_mod.load()
    ws = hw["gpu"]["vram_bytes"]
    reg = load_registry()
    from .registry import MODELS_DIR
    typer.echo(f"{'tag':14s} {'params':7s} {'family (state)':26s} {'bf16':>8s} {'8-bit':>8s} "
               f"{'placement':10s} {'disk needed':>12s} {'downloaded':>10s}")
    fam_of = {"qwen2.5:0.5b": "qwen2", "qwen2.5:7b": "qwen2", "qwen2.5:32b": "qwen2",
              "llama3.3:70b": "llama"}
    for name in reg:
        st = safetensors_spec(name)
        fam = FAMILIES[fam_of[name]]
        bf16_g = st["bytes"] / GIB
        q8_g = MLX8_BYTES[name] / GIB
        placement = "resident" if st["bytes"] <= ws * 0.70 else "streamed"
        need = st["bytes"] / GIB + 20
        dl = safetensors_downloaded(name) or \
            (MODELS_DIR / name.replace(":", "-") / "mlx-8bit" / "config.json").exists()
        typer.echo(f"{name:14s} {PARAMS[name]:7s} {fam.label + ' (' + fam.state + ')':26s} "
                   f"{bf16_g:7.1f}G {q8_g:7.1f}G {placement:10s} {need:10.0f}G+ "
                   f"{str(dl):>10s}")
    typer.echo('\nAny Hugging Face repo id with a supported architecture also works: '
               'spill run org/name sample.')
    _next_hint("spill run qwen2.5:0.5b sample")


from . import cli_build  # noqa: E402,F401  (registers build, example, export, doctor)
from . import cli_project  # noqa: E402,F401  (registers init, plan, report, compare, ...)

if __name__ == "__main__":      # `python -m streamweights.cli`: use the module that has every command
    from streamweights.cli import main as _main_entry
    _main_entry()
