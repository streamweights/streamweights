"""Commands added in Phase 3.5: build, example, export, doctor (and resume for a folder).

Imported at the bottom of cli.py, which owns `app` and the shared helpers."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import typer
import typer.core

from . import build as build_mod
from . import estimate as est
from . import overnight, platforms
from . import probe as probe_mod
from .cli import (JOBS_DIR, GIB, Job, RunOpts, _fail, _finalize_distill, _next_hint,
                  _run_impl, _run_mlx_resume, app)
from .errors import SpillError, StageInterrupted

DEFAULT_EPOCHS = build_mod.DEFAULT_EPOCHS


class RealBackend(build_mod.Backend):
    """Runs the stages with the same code `spill distill`, `spill tune` and `spill eval`
    use, in this process (one stage at a time; each frees its Metal memory on exit)."""

    def __init__(self, quiet: bool = True, epochs: float = DEFAULT_EPOCHS):
        self.quiet = True            # one progress line per stage; the slot block is for `run`
        self.epochs = epochs

    def distill(self, teacher, prompts, out, max_tokens, resume_job):
        if resume_job and (JOBS_DIR / resume_job / "meta.json").exists():
            j = Job.load(resume_job)
            _run_mlx_resume(j, j.quant, probe_mod.load())
            done = len(j.done_ids())
            if done < j.total:
                return {"interrupted": True, "job_id": j.id, "rows": done}
            dest = _finalize_distill(j)
            return {"job_id": j.id, "rows": done, "out": str(dest)}
        job, prog = _run_impl(teacher, str(prompts), None, out, 4096, None, self.quiet,
                              RunOpts(mode="generate", logprobs=None, kind="distill"))
        if prog.done < prog.total:
            return {"interrupted": True, "job_id": job.id, "rows": prog.done}
        dest = _finalize_distill(job)
        return {"job_id": job.id, "rows": prog.done, "out": str(dest)}

    def tune(self, model, train, name, resume_job):
        from . import cli
        impl = getattr(cli, "_tune_build", None)
        if impl is None:
            raise SpillError("this checkout has no `spill tune` yet (it ships with Phase 3)",
                             f"spill tune {model} {train} --name {name}")
        return impl(model, train, name, resume_job, epochs=self.epochs, quiet=self.quiet)

    def eval(self, model, eval_file, resume_job=None):
        from . import cli, evalrun
        from . import runs as runs_mod
        try:
            scored = cli._eval_impl(str(eval_file), [model], "exact_match", None, False, None,
                                    4096, None, self.quiet, echo=False,
                                    resume_jobs={model: resume_job} if resume_job else None)
        except StageInterrupted as e:
            return {"interrupted": True, "job_id": e.job_id}
        sr = scored[0]
        mean, _unscored = evalrun.mean_score(sr)
        m = runs_mod.read_manifest(sr.run_id)
        return {"score": mean if mean is not None else 0.0, "rows": sr.rows,
                "run_id": sr.run_id, "cached": sr.cached,
                "results_path": str(Path(m["job_dir"]) / "results.jsonl"),
                "manifest": str(runs_mod.manifest_path(sr.run_id))}


def _make_plan(f, student, teacher, base, compare, weight_own, epochs):
    from .calibration import load_calibration
    hw = probe_mod.load(probe_if_missing=True)
    from .resolve import arch_from_config, resolve_model
    for m in {student, teacher, base, *compare} - {None}:
        if m not in est.PARAMS_B:
            try:
                r = resolve_model(m)
                a = arch_from_config(r.config) if r.config else None
                est.register_model(m, r.st_bytes, (a["n_layers"], a["n_kv_heads"],
                                                   a["head_dim"]) if a else None)
            except Exception:
                pass          # an unreachable repo fails properly when its stage runs
    return build_mod.make_plan(f, student, teacher, base, compare, weight_own,
                               load_calibration(), hw["gpu"]["vram_bytes"], epochs)


def do_build(folder, student, teacher, base, compare, weight_own, epochs, notify, quiet,
             backend=None):
    f = build_mod.read_folder(folder)
    student = student or f.settings.get("student") or build_mod.DEFAULT_STUDENT
    teacher = teacher or f.settings.get("teacher") or build_mod.DEFAULT_TEACHER
    base = base or f.settings.get("base")
    epochs = epochs or f.settings.get("epochs") or DEFAULT_EPOCHS
    plan = _make_plan(f, student, teacher, base, compare, weight_own, epochs)
    plan.extra = {"notify": notify, "pid": os.getpid()}
    prior = build_mod.load_state(f)
    resumed = False
    if prior and not prior.get("finished"):
        resumed = build_mod.apply_state(plan, prior)
        if not resumed:
            typer.echo(f"note: {f.name}/.build/state.json was for a different plan; starting "
                       f"this build fresh (finished runs are reused when the input is the same)")
    elif prior and prior.get("finished"):
        build_mod.apply_state(plan, prior)
    typer.echo(build_mod.pre_run_line(plan, on_battery=overnight.on_battery()))
    done = sum(1 for s in plan.stages if s.status == "done")
    if done:
        typer.echo(f"continuing: {done} of {len(plan.stages)} stages already done")
    overnight.register_build(f.path)
    build_mod.save_state(plan)
    backend = backend or RealBackend(quiet, epochs)
    import streamweights.cli as cli
    cli._NEXT_HINTS = False
    try:
        with overnight.long_job(f"spill build {f.name}", notify) as st:
            res = build_mod.run_plan(plan, backend, say=typer.echo)
            if res.interrupted:
                st["state"] = "stopped"
    finally:
        cli._NEXT_HINTS = True
        plan.extra["pid"] = None
        build_mod.save_state(plan)
    typer.echo("")
    if res.table:
        typer.echo(build_mod.render_table(res.table))
    if res.interrupted:
        typer.echo(f"\nbuild stopped; nothing is lost.")
        _next_hint(f"spill resume {f.path}")
        raise typer.Exit(130)
    typer.echo("")
    typer.echo(build_mod.render_stages(res.stages))
    typer.echo(f"\nbuild wall time {est.fmt_dur(res.wall_s)} this session")
    for line in build_mod.final_lines(res):
        typer.echo(line)
    _next_hint(f"spill export {res.tuned_label}")
    return res


def resume_build(folder: Path):
    platforms.require_mlx("build")
    f = build_mod.read_folder(folder)
    st = build_mod.load_state(f)
    if not st:
        raise SpillError(f"{folder} has no build to resume", f"spill build {folder}")
    try:
        do_build(str(folder), st["student"], st["teacher"], st.get("base"),
                 st.get("compare", []), st.get("weight_own", 2.0),
                 st.get("epochs", DEFAULT_EPOCHS), st.get("notify"), False)
    except Exception as e:
        if isinstance(e, typer.Exit):
            raise
        _fail(e)


@app.command(short_help="Build your model from a folder: distill, tune, eval",
             epilog="Example: spill build banking77-quick")
def build(
    folder: str = typer.Argument(..., help="a folder with evals.jsonl and train.jsonl and/or "
                                           "prompts.jsonl"),
    student: str = typer.Option(None, "--student",
                                help=f"the small model to build on (default "
                                     f"{build_mod.DEFAULT_STUDENT})"),
    teacher: str = typer.Option(None, "--teacher",
                                help=f"the big model that answers prompts.jsonl (default "
                                     f"{build_mod.DEFAULT_TEACHER})"),
    compare: list[str] = typer.Option(None, "--compare", help="add this model's score to the table "
                                                              "(repeatable)"),
    base: str = typer.Option(None, "--base", help="train the adapter on this (big) model itself "
                                                  "instead of the student"),
    weight_own: float = typer.Option(2.0, "--weight-own", help="with both train.jsonl and "
                                                               "prompts.jsonl: your answers count "
                                                               "this many times to the teacher's 1"),
    epochs: float = typer.Option(None, "--epochs", help=f"passes over the training data "
                                                        f"(default {DEFAULT_EPOCHS:g})"),
    notify: str = typer.Option(None, "--notify", help="POST a small JSON to this URL when "
                                                      "done or stopped"),
    quiet: bool = typer.Option(False, "--quiet", help="one progress line per stage"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Build your own model from a folder: distill, tune, eval, one table."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    try:
        platforms.require_mlx("build")
        do_build(folder, student, teacher, base, list(compare or []), weight_own, epochs,
                 notify, quiet)
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


def _ollama_name(base: str, adapter_id: str) -> str:
    import re
    clean = lambda s: re.sub(r"[^a-z0-9._-]+", "-", s.lower()).strip("-")
    return f"{clean(Path(adapter_id).name)}-{clean(base)}"


class _ExportCommand(typer.core.TyperCommand):
    """`--gguf` takes an optional type: bare `--gguf` means q8_0."""

    def parse_args(self, ctx, args):
        from .export import GGUF_TYPES
        args = list(args)
        for i, a in enumerate(args):
            if a == "--gguf" and (i + 1 == len(args) or args[i + 1].lower() not in GGUF_TYPES):
                args.insert(i + 1, "q8_0")
                break
        return super().parse_args(ctx, args)


@app.command(cls=_ExportCommand, short_help="Merge an adapter into its base; GGUF and Ollama",
             epilog="Example: spill export qwen2.5:0.5b+banking --gguf")
def export(
    model: str = typer.Argument(..., help="<base>+<adapter>, e.g. qwen2.5:7b+banking77"),
    out: Path = typer.Option(None, "--out", help="directory for the merged model (default: "
                                                 "<data>/exports/<base>+<adapter>)"),
    gguf: str = typer.Option(None, "--gguf",
                             help="also write a GGUF: bf16 | q8_0 | q4_k_m (bare --gguf: q8_0)"),
    ollama: bool = typer.Option(False, "--ollama", help="run `ollama create` if ollama is "
                                                        "installed (implies --gguf)"),
    name: str = typer.Option(None, "--name", help="Ollama model name"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Merge the adapter into the base: merged safetensors, optionally GGUF and Ollama."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    try:
        _export_impl(model, out, gguf, ollama, name)
    except Exception as e:
        _fail(e)


def _export_impl(model, out, gguf, ollama, name):
    from . import export as ex
    from .adapters import resolve_adapter, split_model_adapter
    base, ad = split_model_adapter(model)
    if not ad:
        raise SpillError("export needs <base>+<adapter>, e.g. qwen2.5:7b+banking77",
                         "spill adapters")
    adapter = resolve_adapter(ad)
    mdir = _model_dir(base)
    dest = Path(out) if out else ex.EXPORTS_DIR / ex.export_name(base, adapter.id)
    size = sum(f.stat().st_size for f in Path(mdir).glob("*.safetensors"))
    free = shutil.disk_usage(dest.parent if dest.parent.exists() else Path.home()).free
    need = size * (2 if gguf or ollama else 1) + 20 * GIB
    if free < need:
        raise SpillError(f"not enough disk: the merged model is {size / GIB:.1f} GB"
                         f"{' plus the GGUF' if gguf or ollama else ''} and {free / GIB:.0f} GB "
                         f"is free (20 GB floor kept)")
    kind = (gguf or ("q8_0" if ollama else None))
    rate = probe_mod.load()["nvme_seq_read"]["bytes_per_sec"]
    typer.echo(f"spill export {base}+{adapter.id}: merge rank-{adapter.rank} LoRA "
               f"({len(adapter.layers)} layers) into {base} bf16 ({size / GIB:.1f} GB)"
               + (f", then GGUF {kind} via llama.cpp's converter" if kind else "")
               + f". Est. {est.fmt_dur(2 * size / rate)} for the merge (disk-bound)"
               + (", plus the conversion" if kind else "") + f". Cost: $0. Merged model -> {dest}")
    info = ex.merge_adapter(mdir, adapter, dest, say=lambda s: typer.echo(f"   {s}"))
    typer.echo(f"merged {info['modules']} modules into {info['shards']} shard(s), "
               f"{info['dtype']}: {dest}")
    nxt = f"spill export {model} --gguf q8_0"
    if kind:
        gf = ex.convert_gguf(dest, ex.export_name(base, adapter.id).replace("+", "_"), kind,
                             say=lambda s: typer.echo(f"   {s}"))
        mf = ex.write_modelfile(gf, dest)
        typer.echo(f"GGUF {kind}: {gf} ({gf.stat().st_size / GIB:.2f} GB), Modelfile: {mf}")
        oname = name or _ollama_name(base, adapter.id)
        nxt = f"ollama create {oname} -f {mf} && ollama run {oname}"
        if ollama:
            line = ex.ollama_create(oname, mf)
            if line:
                typer.echo(f"ollama model created: {oname}")
                nxt = line
            else:
                typer.echo("ollama is not installed (https://ollama.com); the GGUF and "
                           "Modelfile are ready")
    _next_hint(nxt)


def _model_dir(base: str) -> Path:
    from .registry import download_safetensors
    from .resolve import download_hf, resolve_model
    p = Path(base).expanduser()
    if p.is_dir() and (p / "config.json").exists():
        return p
    res = resolve_model(base)
    return download_hf(res) if res.kind == "hf" else Path(download_safetensors(res.name))


@app.command(short_help="Check this machine and what it can run overnight",
             epilog="Example: spill doctor")
def doctor(debug: bool = typer.Option(False, "--debug", hidden=True)):
    """One screen: chip, memory, disk, models, interrupted jobs, what runs overnight."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    try:
        _doctor_impl()
    except Exception as e:
        _fail(e)


def _doctor_impl():
    from . import doctor as doc
    from .adapters import list_local_adapters
    from .calibration import load_calibration
    from .registry import MODELS_DIR, load_registry
    from importlib.metadata import version as _v
    try:
        __version__ = _v("streamweights")
    except Exception:
        __version__ = "dev"
    from .platforms import mlx_available
    hw = probe_mod.load(probe_if_missing=True)
    interrupted = []
    for b in overnight.interrupted_builds():
        interrupted.append(f"build {b['name']} stage {b['done'] + 1}/{b['total']} "
                           f"(spill resume {b['folder']})")
    for j in overnight.interrupted_jobs(JOBS_DIR):
        interrupted.append(f"{j['kind']} {j['id']} {j['done']}/{j['total']} "
                           f"(spill resume {j['id']})")
    typer.echo(doc.report(hw, load_calibration(), registry_tags=list(load_registry()),
                          models_dir=MODELS_DIR, interrupted_lines=interrupted,
                          version=__version__, adapters=len(list_local_adapters()),
                          mlx=mlx_available()))
    have = any(True for _ in MODELS_DIR.glob("*/bf16-st")) if MODELS_DIR.exists() else False
    first = (overnight.interrupted_builds() or [None])[-1]
    jobs = overnight.interrupted_jobs(JOBS_DIR)
    _next_hint((f"spill resume {first['folder']}" if first else
                f"spill resume {jobs[-1]['id']}") if interrupted else
               "spill run qwen2.5:0.5b sample" if not mlx_available() else
               ("spill build <folder>" if have else "spill example banking77 --quick"))


@app.command(short_help="Create a ready-to-run example folder",
             epilog="Example: spill example banking77 --quick")
def example(
    name: str = typer.Argument("banking77", help="which example (banking77)"),
    quick: bool = typer.Option(False, "--quick", help="the under-an-hour variant: 100 evals, "
                                                      "500 train rows, student qwen2.5:0.5b"),
    force: bool = typer.Option(False, "--force", help="write into a folder that already exists"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Create ./<name>/ with ready files (an exam, homework, and the prompt for untrained models)."""
    import streamweights.cli as cli
    from . import example as ex
    cli._DEBUG = debug
    try:
        folder = ex.create(name, quick, Path("."), force)
        files = sorted(p.name for p in folder.iterdir())
        typer.echo(f"created {folder}/ with {', '.join(files)}")
        _next_hint(f"spill build {folder}" if platforms.mlx_available()
                   else f"spill check {folder}/evals.jsonl")
    except Exception as e:
        _fail(e)
