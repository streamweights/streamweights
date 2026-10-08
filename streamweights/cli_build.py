"""Commands added in Phase 3.5: build, example, export, doctor (and resume for a folder).

Imported at the bottom of cli.py, which owns `app` and the shared helpers."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import typer
import typer.core

from . import build as build_mod
from . import build_defaults, engine_select, overnight, platforms, runtime
from . import estimate as est
from . import probe as probe_mod
from .cli import (JOBS_DIR, GIB, RunOpts, _fail, _finalize_distill, _long_job, _next_hint,
                  _run_impl, app)
from .errors import SpillError, StageInterrupted
from .portable.build_state import BuildState

DEFAULT_EPOCHS = build_mod.DEFAULT_EPOCHS


def _row_producers(results_path) -> list[dict]:
    """Which engine, hardware, OS and numerics produced the rows of a finished (or partial)
    row job: read off the rows themselves, so rows restored from another machine keep the
    machine that made them."""
    groups: dict = {}
    p = Path(results_path) if results_path else None
    if p is None or not p.exists():
        return []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        sw = json.loads(line).get("streamweights") or {}
        key = (sw.get("engine"), sw.get("hardware"), json.dumps(sw.get("numerics"), sort_keys=True),
               sw.get("os"), sw.get("host"))
        g = groups.setdefault(key, {"engine": sw.get("engine"), "hardware": sw.get("hardware"),
                                    "numerics": sw.get("numerics"), "os": sw.get("os"),
                                    "system": sw.get("system"), "host": sw.get("host"),
                                    "n": 0})
        g["n"] += 1
    out = []
    for g in groups.values():
        n = g.pop("n")
        out.append({**g, "quanta": f"{n} rows"})
    return out


def _tune_producers(uri: str | None) -> list[dict]:
    """The engine, hardware, OS and numerics that produced each range of steps, from the
    checkpoint's state.json."""
    if not uri:
        return []
    from .portable import checkpoint as pc
    from .portable.store import Store
    ck = pc.load_tune(Store(uri))
    if ck is None:
        return []
    out = []
    for h in ck.state.get("history", []):
        out.append({"engine": h.get("engine"), "hardware": h.get("hardware"),
                    "numerics": h.get("numerics"), "os": h.get("os"), "system": h.get("system"),
                    "host": h.get("host"),
                    "quanta": f"steps {h['range'][0]}-{h['range'][1]}"})
    return out


class RealBackend(build_mod.Backend):
    """Runs the stages with the same engine-selected code `spill distill`, `spill tune` and
    `spill eval` use, in this process (one stage at a time). MLX on Apple silicon, PyTorch
    elsewhere or with --engine; this class names neither. Each stage's own state is
    runtime.ENV.state (set by build.run_plan), so running a stage again continues it."""

    def __init__(self, quiet: bool = True, epochs: float = DEFAULT_EPOCHS):
        self.quiet = True            # one progress line per stage; the slot block is for `run`
        self.epochs = epochs
        self.hints: dict = {}        # {"micro_batch": n, "grad_accum": g} from an earlier session
        # SPILL_BUILD_BATCH="eval=7,micro=2,accum=2": run the stages with other batch shapes
        # (a measurement knob: how far a score moves when only the batch shape does)
        shape = dict(kv.split("=") for kv in os.environ.get("SPILL_BUILD_BATCH", "").split(",")
                     if "=" in kv)
        self.eval_batch = int(shape["eval"]) if "eval" in shape else None
        if "micro" in shape:
            self.hints = {"micro_batch": int(shape["micro"]),
                          "grad_accum": int(shape.get("accum", 1))}

    def _silenced(self):
        import contextlib
        import io
        return contextlib.redirect_stdout(io.StringIO())

    def distill(self, teacher, prompts, out, max_tokens, resume_job):
        with self._silenced():
            job, prog = _run_impl(teacher, str(prompts), None, out, 4096, None, self.quiet,
                                  RunOpts(mode="generate", logprobs=None, kind="distill"))
            done = prog.done >= prog.total
            dest = _finalize_distill(job) if done else None
        res = {"job_id": job.id, "rows": prog.done,
               "producers": _row_producers(job.results_path)}
        if not done:
            return {"interrupted": True, **res}
        return {**res, "out": str(dest)}

    def tune(self, model, train, name, resume_job):
        from . import cli
        res = cli._tune_build(model, train, name, epochs=self.epochs, quiet=self.quiet,
                              batch=self.hints.get("micro_batch"),
                              grad_accum=self.hints.get("grad_accum") or 1)
        res["producers"] = _tune_producers(runtime.state_uri())
        return res

    def eval(self, model, eval_file, resume_job=None):
        from . import cli, evalrun
        from . import runs as runs_mod
        try:
            with self._silenced():
                scored = cli._eval_impl(str(eval_file), [model], "exact_match", None, False,
                                        None, 4096, self.eval_batch, self.quiet, echo=False,
                                        engine_pure=True)
        except StageInterrupted as e:
            jr = JOBS_DIR / e.job_id / "results.jsonl"
            return {"interrupted": True, "job_id": e.job_id, "producers": _row_producers(jr)}
        sr = scored[0]
        mean, _unscored = evalrun.mean_score(sr)
        m = runs_mod.read_manifest(sr.run_id)
        results = Path(m["job_dir"]) / "results.jsonl"
        return {"score": mean if mean is not None else 0.0, "rows": sr.rows,
                "run_id": sr.run_id, "cached": sr.cached, "results_path": str(results),
                "manifest": str(runs_mod.manifest_path(sr.run_id)),
                "producers": _row_producers(results)}


def _make_plan(f, student, teacher, base, compare, weight_own, epochs, choice, hw, cal,
               bs=None):
    from .resolve import arch_from_config, resolve_model
    for m in {student, teacher, base, *compare} - {None}:
        if m not in est.PARAMS_B:
            try:
                r = resolve_model(m)
                a = arch_from_config(r.config) if r.config else None
                est.register_model(m, r.st_bytes, (a["n_layers"], a["n_kv_heads"],
                                                   a["head_dim"]) if a else None)
            except SpillError:
                raise         # a name that is neither a tag nor a repo id fails before anything runs
            except Exception:
                pass          # an unreachable repo fails properly when its stage runs
    ws, _free, rate = build_defaults.machine_inputs(choice.name, hw, cal)
    plan = build_mod.make_plan(f, student, teacher, base, compare, weight_own, cal, ws, epochs,
                               engine=choice.name, read_rate=rate)
    plan.state = bs
    return plan


def _missing_downloads(plan) -> list[tuple[str, float]]:
    """Curated tags this build needs that are not on disk yet, with their bf16 size."""
    from .registry import load_registry, safetensors_downloaded, safetensors_spec
    reg = load_registry()
    names = [plan.tuned_model] + ([plan.teacher] if plan.path_kind != "train" else []) \
        + list(plan.compare)
    out = []
    for m in dict.fromkeys(names):
        if m in reg and not safetensors_downloaded(m):
            out.append((m, safetensors_spec(m)["bytes"] / GIB))
    return out


def _calibration_for(choice) -> dict:
    """The calibration, with this engine's measured rates in it (a few seconds, once)."""
    from .calibration import load_calibration
    cal = load_calibration()
    if choice.is_torch and choice.name not in (cal.get("engine_tflops") or {}):
        try:
            engine_select.measure_rates()
            cal = load_calibration()
        except Exception:
            pass
    return cal


def do_build(folder, student, teacher, base, compare, weight_own, epochs, notify, quiet,
             backend=None, state=None, stop_after=None):
    f = build_mod.read_folder(folder)
    choice = engine_select.choose_engine(runtime.ENV.engine)
    runtime.set_engine(choice.name)
    hw = probe_mod.load(probe_if_missing=True)
    cal = _calibration_for(choice)
    bs = BuildState(state) if state else None
    prior = build_mod.load_state(f, bs)
    adopt = (prior and not prior.get("finished") and not (student or teacher or base or compare))
    if adopt:           # the same build, continued: its models, not new defaults for this machine
        student, teacher, base = prior["student"], prior["teacher"], prior.get("base")
        compare = prior.get("compare", [])
        weight_own = prior.get("weight_own", weight_own)
        epochs = prior.get("epochs", epochs)
        why = "continuing the models chosen when this build started"
        forced = []
    else:
        student = student or f.settings.get("student")
        teacher = teacher or f.settings.get("teacher")
        base = base or f.settings.get("base")
        ws, free, rate = build_defaults.machine_inputs(choice.name, hw, cal)
        from .registry import safetensors_downloaded
        ch = build_defaults.choose(f, choice.name, cal, ws, free, rate, student, teacher,
                                   downloaded=lambda t: _is_downloaded(t, safetensors_downloaded))
        student, teacher, why, forced = ch.student, ch.teacher, ch.why(), sorted(ch.forced)
    epochs = epochs or f.settings.get("epochs") or DEFAULT_EPOCHS
    plan = _make_plan(f, student, teacher, base, compare, weight_own, epochs, choice, hw, cal,
                      bs)
    plan.why, plan.forced = why, forced
    plan.extra = {"notify": notify, "pid": os.getpid()}
    resumed = False
    if prior and not prior.get("finished"):
        resumed = build_mod.apply_state(plan, prior)
        if not resumed:
            typer.echo(f"note: the state at {bs.uri if bs else f.name + '/.build/'} was for a "
                       f"different plan; starting this build fresh")
    elif prior and prior.get("finished"):
        build_mod.apply_state(plan, prior)
    typer.echo(build_mod.pre_run_line(plan, on_battery=overnight.on_battery(),
                                      state_note=bs.uri if bs else None))
    missing = _missing_downloads(plan)
    if missing:
        typer.echo("   downloads first, not counted in the estimate: " + ", ".join(
            f"{m} {gb:.1f} GB" for m, gb in missing))
    done = sum(1 for s in plan.stages if s.status == "done")
    if done:
        prev = sorted({p.get("engine") for s in plan.stages if s.status == "done"
                       for p in s.producers if p.get("engine")})
        typer.echo(f"continuing: {done} of {len(plan.stages)} stages already done"
                   + (f" (on {', '.join(prev)}); the rest runs on {plan.engine}"
                      if prev and plan.engine not in prev else ""))
    runtime.emit("start", command="build", model=plan.tuned_model, stages=len(plan.stages),
                 done_stages=done, state=bs.uri if bs else str(f.state_dir),
                 estimated_seconds=round(plan.total_s), message=build_mod.pre_run_line(
                     plan, state_note=bs.uri if bs else None))
    overnight.register_build(f.path)
    build_mod.save_state(plan)
    backend = backend or RealBackend(quiet, epochs)
    tune_stage = next((s for s in plan.stages if s.kind == "tune"), None)
    if tune_stage is not None and hasattr(backend, "hints") and tune_stage.result.get("micro_batch"):
        backend.hints["micro_batch"] = tune_stage.result["micro_batch"]     # a resumed tune keeps
        backend.hints["grad_accum"] = tune_stage.result.get("grad_accum") or 1  # its batch shape
    import streamweights.cli as cli
    cli._NEXT_HINTS = False
    res = None
    try:
        with _long_job(f"spill build {f.name}", notify) as st:
            res = build_mod.run_plan(plan, backend, say=typer.echo, stop_after=stop_after)
            if res.interrupted and st is not None:
                st["state"] = "stopped"
    finally:
        cli._NEXT_HINTS = True
        plan.extra["pid"] = None
        build_mod.save_state(plan)
    typer.echo("")
    if res.table:
        typer.echo(build_mod.render_table(res.table))
    guard = runtime.ENV.guard
    if res.interrupted:
        if guard is not None and guard.signalled:     # the session exits 75 after its `preempted`
            typer.echo(f"\nbuild stopped by {guard.signalled}; state saved at "
                       f"{bs.uri if bs else f.name + '/.build/'}")
            return res
        runtime.ENV.stopped_early = res.stopped_early
        if res.stopped_early:
            typer.echo(f"\nbuild stopped early (--stop-after {stop_after}); nothing is lost")
            _next_hint(f"spill resume {f.path}" + (f" --state {bs.uri}" if bs else ""))
            return res
        typer.echo(f"\nbuild stopped; nothing is lost.")
        _next_hint(f"spill resume {f.path}" + (f" --state {bs.uri}" if bs else ""))
        raise typer.Exit(130)
    typer.echo("")
    typer.echo(build_mod.render_stages(res.stages))
    typer.echo("")
    typer.echo(build_mod.render_provenance(res.stages))
    typer.echo(f"\nbuild wall time {est.fmt_dur(res.wall_s)} this session")
    for line in build_mod.final_lines(res):
        typer.echo(line)
    runtime.ENV.summary.update(table=res.table, adapter=plan.adapter_name)
    _next_hint(f"spill export {res.tuned_label}")
    return res


def _print_table(folder: str, state: str | None, reference: str | None) -> None:
    f = build_mod.read_folder(folder)
    doc = build_mod.load_state(f, BuildState(state) if state else None)
    if not doc:
        raise SpillError(f"{f.name} has no build state to show", f"spill build {folder}")
    ref = BuildState(reference).read() if reference else None
    if reference and not ref:
        raise SpillError(f"no build state at {reference}", f"spill build {folder} --table")
    typer.echo(build_mod.table_from_state(doc, ref))


def _is_downloaded(tag: str, check) -> bool:
    try:
        return bool(check(tag))
    except Exception:
        return False


def resume_build(folder: Path, state: str | None = None, stop_after: str | None = None,
                 notify: str | None = None):
    f = build_mod.read_folder(folder)
    bs = BuildState(state) if state else None
    st = build_mod.load_state(f, bs)
    if not st and bs is None:
        mirror = build_mod.load_state(f)
        if mirror and mirror.get("state_uri"):
            bs, st = BuildState(mirror["state_uri"]), None
            st = bs.read()
    elif st and bs is None and st.get("state_uri"):
        bs = BuildState(st["state_uri"])
        st = bs.read() or st
    if not st:
        raise SpillError(f"{folder} has no build to resume", f"spill build {folder}")
    try:
        do_build(str(folder), st["student"], st["teacher"], st.get("base"),
                 st.get("compare", []), st.get("weight_own", 2.0),
                 st.get("epochs", DEFAULT_EPOCHS), notify or st.get("notify"), False,
                 state=bs.uri if bs else None, stop_after=stop_after)
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
                                help="the small model to build on (default: chosen from this "
                                     "machine: qwen2.5:7b on Apple silicon, qwen2.5:0.5b on a CPU)"),
    teacher: str = typer.Option(None, "--teacher",
                                help="the big model that answers prompts.jsonl (default: "
                                     "llama3.3:70b on Apple silicon; on a CPU the largest whose "
                                     "distill estimate is under 12 h)"),
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
    state: str = typer.Option(None, "--state", help="portable build state: a path, s3://, gs://, "
                                                    "az:// (default: <folder>/.build/); the build "
                                                    "continues from it on any machine and engine"),
    engine: str = typer.Option(None, "--engine", help="mlx | torch-cpu | torch-cuda "
                                                      "(default: chosen from the hardware)"),
    headless: bool = typer.Option(False, "--headless", help="JSON-lines events on stdout, exit "
                                                            "75 when preempted (automatic when "
                                                            "stdout is not a terminal)"),
    config: Path = typer.Option(None, "--config", help="run the invocation stored in this "
                                                       "job.json"),
    emit_config: Path = typer.Option(None, "--emit-config", help="write this invocation to "
                                                                 "job.json and exit"),
    table: bool = typer.Option(False, "--table", help="print the table of a finished or "
                                                       "stopped build from its state (which "
                                                       "engine, machine and OS made each stage) "
                                                       "and exit"),
    reference: str = typer.Option(None, "--reference", help="with --table: the state of an "
                                                            "uninterrupted build, for a "
                                                            "reference column"),
    new_run: bool = typer.Option(False, "--new-run", help="project folders: build again as a new "
                                                          "run even if a run already has these "
                                                          "exact inputs"),
    executor: str = typer.Option("local", "--executor", hidden=True),
    stop_after: str = typer.Option(None, "--stop-after", hidden=True),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Build your own model from a folder: distill, tune, eval, one table. Runs on MLX on
    Apple silicon and on PyTorch (CPU or CUDA) everywhere else. A folder made by `spill init`
    (it has streamweights.toml) builds a run; a folder with evals.jsonl builds as before."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    from .project import config as project_config
    from .project import migrate as project_migrate
    if project_config.exists(folder) and project_migrate.is_guided(project_config.load(folder)):
        try:
            from .cli_project import project_build
            project_build(folder, student, teacher, epochs, engine, headless, executor, new_run,
                          stop_after, notify)
        except typer.Exit:
            raise
        except Exception as e:
            _fail(e)
        return
    if table:
        try:
            _print_table(folder, state, reference)
        except Exception as e:
            _fail(e)
        return
    with runtime.job_session("build", engine=engine, headless_flag=headless):
        try:
            do_build(folder, student, teacher, base, list(compare or []), weight_own, epochs,
                     notify, quiet, state=state, stop_after=stop_after)
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
    run: str = typer.Option(None, "--run", help="project folders: the run to export (default: "
                                                "the latest completed run)"),
    verify_rows: int = typer.Option(8, "--verify-rows", help="project folders: validation rows "
                                                             "the export is verified on"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Merge the adapter into the base: merged safetensors, optionally GGUF and Ollama. A
    project folder (from spill init) exports its latest run as a verified export record."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    try:
        from .project import config as project_config
        from .project import migrate as project_migrate
        if project_config.exists(model) and project_migrate.is_guided(project_config.load(model)):
            from .cli_project import project_export
            project_export(Path(model), run, gguf, verify_rows)
            return
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
        if not ollama and not shutil.which("ollama"):
            typer.echo("ollama is not installed (https://ollama.com); the GGUF and Modelfile "
                       "are ready, and the line below runs it once it is")
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
def doctor(engines: bool = typer.Option(False, "--engines", hidden=True),
           debug: bool = typer.Option(False, "--debug", hidden=True)):
    """One screen: chip, memory, disk, models, interrupted jobs, what runs overnight."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    if engines:                     # one usable engine per line, best first (used by relay.sh)
        avail = engine_select.availability()
        for name in ("mlx", "torch-cuda", "torch-cpu"):
            if avail[name][0]:
                typer.echo(name)
        return
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
    from . import engine_select as es
    typer.echo("\n".join(doc.engine_lines(es.choose_engine(), es.availability(),
                                          es.measure_rates())))
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
    name: str = typer.Argument("banking77", help="which example (banking77, snips, relay)"),
    quick: bool = typer.Option(False, "--quick", help="the under-an-hour variant: 100 evals, "
                                                      "500 train rows, student qwen2.5:0.5b"),
    tiny: bool = typer.Option(False, "--tiny", help="the CI-sized variant: 20 evals, 100 train "
                                                    "rows over 10 intents, student qwen2.5:0.5b"),
    force: bool = typer.Option(False, "--force", help="write into a folder that already exists"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Create ./<name>/ with ready files (an exam, homework, and the prompt for untrained models)."""
    import streamweights.cli as cli
    from . import example as ex
    cli._DEBUG = debug
    try:
        folder = ex.create(name, quick, Path("."), force, tiny)
        files = sorted(p.name for p in folder.iterdir())
        typer.echo(f"created {folder}/ with {', '.join(files)}")
        if name == "relay":
            _next_hint(f"{folder}/relay.sh")
        elif name == "snips":
            _next_hint(f"spill init {folder}/snips.csv --input text --output json "
                       f"--schema {folder}/schema.json --project {folder}-project")
        else:
            _next_hint(f"spill build {folder}")
    except Exception as e:
        _fail(e)
