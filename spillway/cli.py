"""Spillway CLI. Every command ends by printing the one command most likely next."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import typer

from . import probe as probe_mod
from .jobs.engine import BatchEngine, Job, JOBS_DIR, compute_offload
from .policy import choose_quant
from .registry import GIB, download, load_registry

app = typer.Typer(add_completion=False, no_args_is_help=True)

GATEWAY = "http://127.0.0.1:11435"


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


@app.command()
def run(
    model: str = typer.Argument(..., help="registry model name, e.g. llama3.3:70b"),
    input_jsonl: Path = typer.Argument(..., exists=True, readable=True),
    quant: str = typer.Option(None, "--quant", help="Q8_0 or Q4_K_M (explicit opt-in; bf16 is the default)"),
    out: Path = typer.Option(None, "--out"),
    context: int = typer.Option(4096, "--context"),
    parallel: int = typer.Option(None, "--parallel", help="override computed batch size"),
):
    """Run an OpenAI batch JSONL against a local model."""
    hw = probe_mod.load(probe_if_missing=True)  # probes if needed; zero config
    reg = load_registry()
    if model not in reg:
        typer.echo(f"unknown model {model}; registry has: {', '.join(reg)}", err=True)
        raise typer.Exit(1)
    m = reg[model]
    choice = choose_quant(m, hw, explicit=quant)
    if choice.quant not in ("bf16",):
        typer.echo(f"quant: {choice.reason}")

    paths = download(m, choice.quant)

    ram = hw["ram_total_bytes"]
    vram = hw["gpu"]["vram_bytes"]
    size = m.quants[choice.quant].bytes
    ngl, n = compute_offload(m, choice.quant, context, vram, parallel)
    fits = size + m.kv_bytes_per_token() * context * n <= ram
    nvme = hw["nvme_seq_read"]["bytes_per_sec"]

    rows = [l for l in input_jsonl.read_text().splitlines() if l.strip()]
    n_prompts = len(rows)
    # estimate: resident ≈ measured-class tokens/s unknown before first run; use
    # streaming bound (one pass per batch of n tokens) or a resident heuristic.
    est_completion_tokens = n_prompts * 128
    if fits:
        est_tps = max(50.0, vram / GIB * 2)  # rough resident floor; refined live
        placement = "resident"
    else:
        pass_s = size / nvme
        est_tps = n / pass_s
        placement = f"streaming from NVMe at ~{nvme / GIB:.1f} GB/s"
    est = est_completion_tokens / est_tps

    job = Job.create(input_jsonl, model, choice.quant, context, n, out)
    out_path = out or job.results_path

    typer.echo(
        f"Spillway: {model} {choice.quant} ({size / GIB:.1f} GB) "
        f"{'fits' if fits else 'does not fit'} in {ram / GIB:.0f} GB RAM; {placement}. "
        f"{n_prompts} prompts, batch {n}, est. {_fmt_dur(est)}. Cost: $0. "
        f"Results -> {out_path} (tail with: spillway tail)"
    )

    def show(p):
        eta = p.eta_seconds
        sys.stderr.write(
            f"\r{p.done}/{p.total} rows  {p.tokens_per_sec:7.1f} tok/s  "
            f"ETA {_fmt_dur(eta) if eta else '...'}   "
        )
        sys.stderr.flush()

    engine = BatchEngine(job, m, paths, ngl, progress_cb=show)
    prog = asyncio.run(engine.run())
    sys.stderr.write("\n")
    if out and Path(out) != job.results_path:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(job.results_path.read_bytes())
    typer.echo(f"{prog.done}/{prog.total} rows, {prog.completion_tokens} completion tokens, "
               f"{prog.tokens_per_sec:.1f} tok/s aggregate")
    _next_hint(f"spillway tail {job.id}" if prog.done < prog.total else "spillway status")


@app.command()
def tail(job: str = typer.Argument(None)):
    """Follow results.jsonl of the latest (or named) job."""
    j = Job.load(job) if job else Job.latest()
    if not j:
        typer.echo("no jobs yet")
        _next_hint("spillway run qwen2.5:0.5b examples/evals-2000.jsonl")
        raise typer.Exit(1)
    typer.echo(f"tailing {j.results_path}  (^C to stop)")
    pos = 0
    try:
        while True:
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
    _next_hint("spillway status")


@app.command()
def resume(job: str = typer.Argument(None)):
    """Continue the latest or named job from its checkpoint."""
    j = Job.load(job) if job else Job.latest()
    if not j:
        typer.echo("no jobs to resume")
        _next_hint("spillway run qwen2.5:0.5b examples/evals-2000.jsonl")
        raise typer.Exit(1)
    done = len(j.done_ids())
    if done >= j.total:
        typer.echo(f"{j.id}: already complete ({done}/{j.total})")
        _next_hint("spillway status")
        return
    hw = probe_mod.load()
    reg = load_registry()
    m = reg[j.model]
    paths = m.quants[j.quant].local_paths(j.model)
    ngl, _ = compute_offload(m, j.quant, j.ctx, hw["gpu"]["vram_bytes"], j.parallel)
    typer.echo(f"resuming {j.id}: {done}/{j.total} done, {j.total - done} remaining")

    def show(p):
        sys.stderr.write(f"\r{p.done}/{p.total} rows  {p.tokens_per_sec:7.1f} tok/s   ")
        sys.stderr.flush()

    engine = BatchEngine(j, m, paths, ngl, progress_cb=show)
    prog = asyncio.run(engine.run())
    sys.stderr.write("\n")
    typer.echo(f"{prog.done}/{prog.total} rows complete")
    _next_hint("spillway status")


@app.command()
def status():
    """List jobs with progress, tokens/s, ETA."""
    if not JOBS_DIR.exists() or not any(JOBS_DIR.iterdir()):
        typer.echo("no jobs")
        _next_hint("spillway run qwen2.5:0.5b examples/evals-2000.jsonl")
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
    _next_hint("spillway tail")


@app.command()
def models():
    """List registry entries, downloaded quants, and resident fit on this machine."""
    hw = probe_mod.load()
    ram = hw["ram_total_bytes"]
    reg = load_registry()
    typer.echo(f"{'model':14s} {'quant':7s} {'size':>9s} {'downloaded':>10s} {'fits resident':>14s}")
    for name, m in reg.items():
        for qn, q in m.quants.items():
            fits = m.resident_bytes(qn, 4096) <= ram
            typer.echo(f"{name:14s} {qn:7s} {q.bytes / GIB:8.1f}G {str(q.downloaded(name)):>10s} {str(fits):>14s}")
    _next_hint("spillway run qwen2.5:0.5b examples/evals-2000.jsonl")


if __name__ == "__main__":
    app()
