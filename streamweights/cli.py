"""streamweights CLI (`spill`). Every command ends by printing the next command."""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import typer

from . import probe as probe_mod
from .engines.base import MemoryBudget, ModelSpec
from .jobs.engine import Job, JOBS_DIR, compute_offload
from .policy import choose_quant, choose_quant_v2, engine_read_rate
from .registry import (GIB, download, load_registry, mlx_quant_repo,
                       safetensors_dir, safetensors_downloaded,
                       safetensors_spec, download_safetensors)

if sys.version_info < (3, 10):  # pragma: no cover
    sys.stderr.write("spill needs Python 3.10 or newer; this is "
                     f"{sys.version.split()[0]} — try: uv tool install "
                     "git+https://github.com/streamweights/streamweights\n")
    raise SystemExit(1)

app = typer.Typer(add_completion=False, no_args_is_help=True)
_DEBUG = False

SAMPLE_PATH = Path(__file__).parent / "data" / "sample-20.jsonl"


def _fail(e):
    from .errors import SpillError
    if _DEBUG:
        raise e
    if isinstance(e, SpillError):
        typer.echo(f"spill: {e.line()}", err=True)
    else:
        typer.echo(f"spill: {type(e).__name__}: {e} — rerun with --debug for the traceback",
                   err=True)
    raise typer.Exit(1)


def _resolve_input(input_arg: str) -> Path:
    from .errors import SpillError
    if input_arg == "sample":
        typer.echo("(using the packaged 20-prompt sample: spill run <model> sample)")
        return SAMPLE_PATH
    p = Path(input_arg)
    if not p.exists():
        raise SpillError(f"input file {input_arg} does not exist",
                         "spill run <model> sample")
    return p


def _validate_rows(path: Path) -> list[dict]:
    from .errors import SpillError
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            assert isinstance(r.get("custom_id"), str)
            assert isinstance(r["body"]["messages"], list) and r["body"]["messages"]
        except (json.JSONDecodeError, KeyError, AssertionError, TypeError):
            raise SpillError(
                f"malformed input at line {n}: each line must be JSON like "
                '{"custom_id": "x", "body": {"messages": [{"role": "user", '
                '"content": "..."}], "max_tokens": 48}}')
        rows.append(r)
    if not rows:
        raise SpillError(f"{path} contains no rows", "spill run <model> sample")
    return rows

GATEWAY = "http://127.0.0.1:11435"
MB = 1024 * 1024


def _is_mac() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _fmt_dur(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def _next_hint(cmd: str) -> None:
    typer.echo(f"\nnext: {cmd}")


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
    from .engines.mlx_stream import load_calibration
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


def _pass_line(info):  # retained for callers without a job dir
    _LiveRenderer()(info)


def _run_mlx(res, input_jsonl: Path, quant: str, reason: str,
             est_s: float | None, out: Path | None, context: int,
             parallel: int | None, hw: dict, rows: list[dict],
             quiet: bool = False) -> None:
    from .engines.mlx_resident import MlxResidentEngine
    from .engines.mlx_stream import (MlxStreamEngine, SafetensorsIndex,
                                     compute_batch, load_calibration)
    from .engines.supported import classify
    from .resolve import download_hf

    model = res.name
    reg = load_registry()
    ws = hw["gpu"]["vram_bytes"]
    ram = hw["ram_total_bytes"]

    if res.kind == "hf":
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
            import shutil as _sh
            info = HfApi().model_info(repo, files_metadata=True)
            dl_bytes = sum(f.size or 0 for f in info.siblings)
            free = _sh.disk_usage(path.parent.parent).free
            if free - dl_bytes < MIN_FREE_AFTER_DOWNLOAD:
                typer.echo(f"refusing download: {dl_bytes / GIB:.1f} GB to {path} would "
                           f"leave {(free - dl_bytes) / GIB:.1f} GB free (< 20 GB floor)",
                           err=True)
                raise typer.Exit(1)
            typer.echo(f"downloading {model} {quant} (mlx): {dl_bytes / GIB:.1f} GB -> {path}",
                       file=sys.stderr)
            snapshot_download(repo, local_dir=path)
        size = sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))

    fits = size <= ws * 0.70
    n_prompts = len(rows)
    max_tokens = max(r["body"].get("max_tokens", 128) for r in rows)

    if fits:
        engine = MlxResidentEngine()
        batch = 1
        placement = "resident"
        est = est_s or n_prompts * max_tokens / 150
        why = f"{reason}; model fits working set -> mlx_resident"
    else:
        engine = MlxStreamEngine(progress_note=lambda s: typer.echo(f"   {s}", err=True))
        index = SafetensorsIndex(path)
        from mlx_lm.utils import load_tokenizer
        tokenizer = load_tokenizer(Path(path))
        kv_tok = (2 * index.config["num_hidden_layers"]
                  * index.config["num_key_value_heads"]
                  * (index.config["hidden_size"] // index.config["num_attention_heads"]) * 2)
        costs = []
        for r in rows:
            toks = tokenizer.apply_chat_template(r["body"]["messages"],
                                                 add_generation_prompt=True)
            costs.append((len(toks) + r["body"].get("max_tokens", 128)) * kv_tok)
        cal = load_calibration()
        bm = compute_batch(index, MemoryBudget(ws, batch_override=parallel),
                           costs, max_tokens, kv_tok, calibration=cal,
                           quant=quant)
        batch = bm.batch
        rate, rate_src = engine_read_rate(cal, hw)
        pass_s = size / rate
        est = math.ceil(n_prompts / batch) * (max_tokens + 1) * pass_s
        placement = f"streaming from NVMe at ~{rate / GIB:.1f} GB/s ({rate_src})"
        why = f"{bm.reason}; quant: {reason}"

    fam = classify(json.loads((Path(path) / "config.json").read_text()))
    job = Job.create(input_jsonl, model, quant, context, batch, out)
    engine.pass_cb = _LiveRenderer(job.dir, quiet)
    out_path = out or job.results_path
    typer.echo(
        f"spill: {model} {quant} ({size / GIB:.1f} GB, {fam.label}: {fam.state}) "
        f"{'fits' if size <= ram else 'does not fit'} in {ram / GIB:.0f} GB RAM; {placement}. "
        f"{n_prompts} prompts, batch {batch}, est. {_fmt_dur(est)}. Cost: $0. "
        f"Results -> {out_path} (tail with: spill tail)")
    typer.echo(f"   why: {why}")

    spec = ModelSpec(model, quant, Path(path), reg[model].arch if model in reg else {},
                     context)
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
    if out and Path(out) != job.results_path:
        Path(out).write_bytes(job.results_path.read_bytes())
    typer.echo(f"{prog.done}/{prog.total} rows, {prog.completion_tokens} completion tokens, "
               f"{prog.tokens_per_sec:.1f} tok/s aggregate")
    _next_hint(f"spill resume {job.id}" if prog.done < prog.total else "spill status")


@app.command()
def run(
    model: str = typer.Argument(..., help="curated tag (llama3.3:70b) or HF repo id (org/name[@rev])"),
    input_jsonl: str = typer.Argument(..., help="OpenAI batch JSONL path, or `sample`"),
    quant: str = typer.Option(None, "--quant",
                              help="8bit|4bit (mlx) or Q8_0|Q4_K_M (gguf); bf16 is the default"),
    out: Path = typer.Option(None, "--out"),
    context: int = typer.Option(4096, "--context"),
    parallel: int = typer.Option(None, "--parallel", help="override computed batch (never required)"),
    quiet: bool = typer.Option(False, "--quiet", help="suppress the live tail block"),
    debug: bool = typer.Option(False, "--debug", help="show tracebacks"),
):
    """Run an OpenAI batch JSONL (or `sample`) against a tag or any HF repo id."""
    global _DEBUG
    _DEBUG = debug
    try:
        _run_impl(model, input_jsonl, quant, out, context, parallel, quiet)
    except Exception as e:
        _fail(e)


def _run_impl(model, input_arg, quant, out, context, parallel, quiet):
    from .resolve import arch_from_config, resolve_model
    hw = probe_mod.load(probe_if_missing=True)
    reg = load_registry()
    res = resolve_model(model)
    input_jsonl = _resolve_input(input_arg)
    rows = _validate_rows(input_jsonl)
    max_tokens = max(r["body"].get("max_tokens", 128) for r in rows)

    if _is_mac() and quant in (None, "bf16", "8bit", "4bit"):
        from .engines.mlx_stream import load_calibration
        arch = reg[model].arch if res.kind == "tag" else arch_from_config(res.config)
        downloaded = safetensors_downloaded(model) if res.kind == "tag" else (
            res.local_dir.exists() and any(res.local_dir.glob("*.safetensors")))
        est_batch = _rough_batch(res.st_bytes, arch, rows, max_tokens,
                                 hw["gpu"]["vram_bytes"])
        choice = choose_quant_v2(res.st_bytes, downloaded,
                                 len(rows), max_tokens, est_batch,
                                 load_calibration(), hw, explicit=quant)
        if choice.quant != "bf16":
            typer.echo(f"quant: {choice.reason}")
        if res.kind == "hf" and choice.quant in ("8bit", "4bit") and quant is None:
            typer.echo(f"note: no {choice.quant} artifact is registered for {res.name}; "
                       f"staying at bf16 (the drop rule applies to curated tags)")
            choice.quant = "bf16"
        _run_mlx(res, input_jsonl, choice.quant, choice.reason, choice.est_seconds,
                 out, context, parallel, hw, rows, quiet=quiet)
        return
    model = res.name
    if not _is_mac():
        typer.echo("spill: not Apple silicon / no Metal — using the llama.cpp GGUF path, "
                   "which works but is slow; the streaming runner needs Metal")
    if res.kind == "hf":
        from .errors import SpillError
        raise SpillError("arbitrary HF repos run on the Metal streaming path only; "
                         "on this machine use a curated tag", "spill models")

    # non-Apple path (or explicit GGUF quant): llama.cpp, Phase 0 policy
    m = reg[model]
    input_jsonl = _resolve_input(input_arg) if res.kind == "tag" else input_jsonl
    choice = choose_quant(m, hw, explicit=quant)
    if choice.quant != "bf16":
        typer.echo(f"quant: {choice.reason}")
    from .llamacpp import ensure_llama_server
    ensure_llama_server()
    try:
        paths = download(m, choice.quant)
    except RuntimeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    ram = hw["ram_total_bytes"]
    size = m.quants[choice.quant].bytes
    ngl, n = compute_offload(m, choice.quant, context, hw["gpu"]["vram_bytes"], parallel)
    fits = m.resident_bytes(choice.quant, context, n) <= ram
    nvme = hw["nvme_seq_read"]["bytes_per_sec"]
    est = len(rows) * max_tokens / max(n / (size / nvme), 1e-6) if not fits \
        else len(rows) * max_tokens / 150
    job = Job.create(input_jsonl, model, choice.quant, context, n, out)
    out_path = out or job.results_path
    typer.echo(
        f"spill: {model} {choice.quant} ({size / GIB:.1f} GB) "
        f"{'fits' if fits else 'does not fit'} in {ram / GIB:.0f} GB RAM; "
        f"{'resident' if fits else f'streaming from NVMe at ~{nvme / GIB:.1f} GB/s'}. "
        f"{len(rows)} prompts, batch {n}, est. {_fmt_dur(est)}. Cost: $0. "
        f"Results -> {out_path} (tail with: spill tail)")
    from .engines.llamacpp import LlamaCppEngine
    from .jobs.runner import run_job
    spec = ModelSpec(model, choice.quant, paths[0], m.arch, context,
                     extra={"n_gpu_layers": ngl, "log_path": str(job.dir / "llama-server.log")})
    prog = run_job(job, LlamaCppEngine(), spec,
                   MemoryBudget(hw["gpu"]["vram_bytes"], batch_override=n), _progress_line)
    sys.stderr.write("\n")
    typer.echo(f"{prog.done}/{prog.total} rows, {prog.completion_tokens} completion tokens, "
               f"{prog.tokens_per_sec:.1f} tok/s aggregate")
    _next_hint(f"spill resume {job.id}" if prog.done < prog.total else "spill status")


@app.command()
def tail(job: str = typer.Argument(None)):
    """Follow results.jsonl of the latest (or named) job."""
    j = Job.load(job) if job else Job.latest()
    if not j:
        typer.echo("no jobs yet")
        _next_hint("spill run qwen2.5:0.5b examples/evals-2000.jsonl")
        raise typer.Exit(1)
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


@app.command()
def resume(job: str = typer.Argument(None)):
    """Continue the latest or named job from its checkpoint."""
    j = Job.load(job) if job else Job.latest()
    if not j:
        typer.echo("no jobs to resume")
        _next_hint("spill run qwen2.5:0.5b examples/evals-2000.jsonl")
        raise typer.Exit(1)
    done = len(j.done_ids())
    if done >= j.total:
        typer.echo(f"{j.id}: already complete ({done}/{j.total})")
        _next_hint("spill status")
        return
    typer.echo(f"resuming {j.id}: {done}/{j.total} done, {j.total - done} remaining")
    hw = probe_mod.load()
    quant = j.quant
    if _is_mac() and quant in ("bf16", "8bit", "4bit"):
        rows = [json.loads(l) for l in j.input_path.read_text().splitlines() if l.strip()]
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


def _run_mlx_resume(j: Job, quant: str, hw: dict) -> None:
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
    engine.pass_cb = _LiveRenderer(j.dir, quiet=False)
    spec = ModelSpec(j.model, quant, Path(path),
                     reg[j.model].arch if j.model in reg else {}, j.ctx)
    prog = run_job(j, engine, spec, MemoryBudget(ws, batch_override=None), progress_cb=None)
    sys.stderr.write("\n")
    typer.echo(f"{prog.done}/{prog.total} rows complete")


@app.command()
def status():
    """List jobs with progress, tokens/s, ETA."""
    if not JOBS_DIR.exists() or not any(JOBS_DIR.iterdir()):
        typer.echo("no jobs")
        _next_hint("spill run qwen2.5:0.5b examples/evals-2000.jsonl")
        return
    typer.echo(f"{'job':28s} {'model':14s} {'quant':7s} {'progress':12s} {'tok/s':>8s} {'eta':>8s} status")
    for d in sorted(JOBS_DIR.iterdir()):
        mp = d / "meta.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        done, total = meta.get("done", 0), meta.get("total", 0)
        eta = meta.get("eta_seconds")
        typer.echo(f"{meta['id']:28s} {meta['model']:14s} {meta['quant']:7s} "
                   f"{f'{done}/{total}':12s} {meta.get('tokens_per_sec', 0):8.1f} "
                   f"{_fmt_dur(eta) if eta else '-':>8s} {meta.get('status', '?')}")
        if meta.get("status") == "running" and (d / "live.json").exists():
            try:
                _LiveRenderer()(json.loads((d / "live.json").read_text()))
            except (json.JSONDecodeError, OSError):
                pass
    _next_hint("spill tail")


PARAMS = {"qwen2.5:0.5b": "0.5B", "qwen2.5:32b": "32B", "llama3.3:70b": "70B"}
MLX8_BYTES = {"qwen2.5:0.5b": 700_000_000, "qwen2.5:32b": 35_000_000_000,
              "llama3.3:70b": 75_000_000_000}


@app.command()
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
    fam_of = {"qwen2.5:0.5b": "qwen2", "qwen2.5:32b": "qwen2", "llama3.3:70b": "llama"}
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
    _next_hint("spill run llama3.3:70b sample")


if __name__ == "__main__":
    app()
