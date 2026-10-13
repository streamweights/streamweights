"""Commands of the guided workflow: init, plan, report, compare, test, bundle, move (and the
project paths of build, export and resume). Registered from cli.py."""

from __future__ import annotations

from pathlib import Path

import typer

from .cli import _fail, _next_hint, app
from .errors import SpillError


def _guided(folder, command: str) -> dict:
    """The project's config, migrating a flat-layout folder first (one announced line), and a
    clear refusal when the command needs a guided project."""
    from .project import migrate as M
    cfg = M.ensure_project(folder, say=typer.echo)
    if cfg is None:
        raise SpillError(f"{folder} is not a project folder", "spill init <data> --input <col> "
                         "--output <col>")
    if not M.is_guided(cfg):
        raise SpillError(f"{folder} is a flat-layout folder: spill {command} works on a project "
                         f"made by spill init; spill build and spill resume keep working here",
                         f"spill build {folder}")
    return cfg


def _init_judge(judge: str | None, student: str | None) -> str | None:
    if judge != "recommended":
        return judge
    from .project import plan as P
    tag = P.recommended_judge(student or "qwen2.5:0.5b")
    typer.echo(f"accepting the recommended judge: {tag}")
    return tag


@app.command(short_help="Start a project from a CSV or JSONL of examples",
             epilog="Example: spill init tickets.csv --input text --output label")
def init(
    data: Path = typer.Argument(..., help="a CSV or JSONL of labeled examples"),
    input_col: str = typer.Option(..., "--input", help="the column holding the input text"),
    output_col: str = typer.Option(..., "--output", help="the column holding the desired output: "
                                                         "a label, a JSON object, or free text"),
    group: str = typer.Option(None, "--group", help="a column whose rows must stay on one side "
                                                    "of every split (customer, document, ...)"),
    system: Path = typer.Option(None, "--system", help="a text file of instructions for the task"),
    score: str = typer.Option(None, "--score", help="exact | json | judge | script:<file.py> "
                                                    "(default: chosen from the outputs and "
                                                    "stated)"),
    task: str = typer.Option(None, "--task", hidden=True, help="older spelling of --score"),
    student: str = typer.Option(None, "--student", help="the small model to train (default: "
                                                        "chosen from this machine, with the reason)"),
    teacher: str = typer.Option(None, "--teacher", help="a larger model that learns your task's "
                                                        "answers for the student to copy "
                                                        "(optional)"),
    judge: str = typer.Option(None, "--judge", help="a model that scores open-ended output "
                                                    "against a rubric, or `recommended` "
                                                    "(optional)"),
    unlabeled: Path = typer.Option(None, "--unlabeled", help="a CSV or JSONL of prompts with no "
                                                             "answers (same input column, or "
                                                             "`prompt`); a teacher answers them"),
    project: Path = typer.Option(None, "--project", help="the project folder (default: the "
                                                         "data file's name)"),
    labels: str = typer.Option(None, "--labels", help="the class vocabulary: a,b,c or a file of "
                                                      "one label per line (default: the "
                                                      "training rows' labels)"),
    schema: Path = typer.Option(None, "--schema", help="a JSON Schema file for json outputs "
                                                       "(default: derived from the training rows)"),
    val: Path = typer.Option(None, "--val", help="your own validation file (same columns)"),
    test: Path = typer.Option(None, "--test", help="your own final-test file (same columns); "
                                                   "never trained on, scored only by `spill test`"),
    seed: int = typer.Option(0, "--seed", help="seed for the generated splits"),
    val_fraction: float = typer.Option(0.15, "--val-fraction", help="share of rows for validation"),
    test_fraction: float = typer.Option(0.15, "--test-fraction", help="share of rows for the "
                                                                      "final test"),
    force: bool = typer.Option(False, "--force", help="write into a folder that has a project"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Validate your examples, freeze the task, split them, and write streamweights.toml."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    from .project import data as D
    from .project.init import create_project
    try:
        folder = project or Path(data.stem)
        mapping = {"input": input_col, "output": output_col, "group": group}
        res = create_project(data, folder, mapping, task, system, labels, schema, val, test, seed,
                             val_fraction, test_fraction, force, say=typer.echo, score=score,
                             student=student, teacher=teacher,
                             judge=_init_judge(judge, student), unlabeled=unlabeled)
    except D.DataErrors as e:
        for line in e.lines():
            typer.echo(f"spill: {line}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        _fail(e)
        return
    s = res.sizes
    typer.echo(f"spill init {data.name}: scoring {res.task} ({res.task_note}). Train {s['train']}, "
               f"validation {s['val']}, final test {s['test']} rows. Cost: $0. Project -> {folder}")
    for n in res.notes:
        typer.echo(f"   {n}")
    for w in res.warnings:
        typer.echo(f"   warning: {w}")
    _next_hint(f"spill plan {folder}")


@app.command(short_help="Show what a build will do, before it does it",
             epilog="Example: spill plan tickets")
def plan(
    project: Path = typer.Argument(..., help="a project folder (from spill init)"),
    engine: str = typer.Option(None, "--engine", help="mlx | torch-cpu | torch-cuda "
                                                      "(default: chosen from the hardware)"),
    calibrate: bool = typer.Option(False, "--calibrate", help="first measure this engine's matmul "
                                                              "and memory rates on synthetic "
                                                              "data (a few seconds)"),
    deadline: str = typer.Option(None, "--deadline", help="a duration (90m, 2h, 1d12h) or timestamp "
                                 "(2026-10-14T09:00): list the combinations of student, teacher, "
                                 "judge and engine on this machine estimated to finish in time"),
    student: str = typer.Option(None, "--student", help="plan with this student (not saved)"),
    teacher: str = typer.Option(None, "--teacher", help="plan with this teacher (not saved)"),
    judge: str = typer.Option(None, "--judge", help="plan with this judge (not saved)"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Models and engine with reasons, downloads, disk, and a duration per stage labeled
    measurement, assumption or unknown. Loads no model, downloads nothing, trains nothing. With
    --deadline, also which combinations of models and engine are estimated to fit."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    try:
        from . import engine_select
        from .project import plan as P
        _guided(project, "plan")
        if calibrate:
            r = engine_select.measure_rates()
            typer.echo("calibration (synthetic matmul and memory, nothing trained): "
                       + "; ".join(f"{e} {v.get('matmul_tflops', '?')} TFLOP/s" for e, v in r.items()))
        roles = {k: v for k, v in (("student", student), ("teacher", teacher), ("judge", judge)) if v}
        if judge == "recommended":
            roles.pop("judge")
        p = P.make_plan(project, engine, fetch=False, cfg_override={"model": roles} if roles else None)
        for line in P.render(p):
            typer.echo(line)
        if roles:
            typer.echo("note: " + ", ".join(f"{k} {v}" for k, v in roles.items())
                       + " used for this plan only; streamweights.toml is unchanged (spill build "
                         "--student/--teacher/--judge records them)")
        if deadline:
            from .project import deadline as DL
            secs, label = DL.parse_deadline(deadline)
            for line in DL.render(DL.evaluate(project, secs, engine), secs, label):
                typer.echo(line)
    except Exception as e:
        _fail(e)
        return
    _next_hint(f"spill build {project}" + (f" --deadline {deadline}" if deadline else ""))


def project_build(folder, student, teacher, epochs, engine, headless, executor, new_run,
                  stop_after, notify, judge=None, kd=None, deadline=None):
    """`spill build <project>`: plan, run, report. Flags that change the build are written
    into streamweights.toml first (one line each), so the project stays the record."""
    from . import runtime
    from .project import config as C
    from .project import coordinator as CO
    from .project import data as D
    from .project import plan as P
    from .project import report as R
    from .project import scoring as SC
    from .project.common import read_jsonl
    folder = Path(folder)
    cfg = C.load(folder)
    changed = []
    if judge == "recommended":
        if cfg["task"]["type"] != "open":
            raise SpillError(f"judged scoring is for open-ended output; this project scores by "
                             f"{SC.scoring_of(cfg)}", f"spill build {folder}")
        judge = P.recommended_judge(student or cfg["model"]["student"])
        typer.echo(f"accepting the recommended judge: {judge}")
    roles = {k: v for k, v in (("student", student), ("teacher", teacher), ("judge", judge)) if v}
    if deadline:
        from .project import deadline as DL
        fits, lines = DL.check_configured(folder, deadline, engine, roles)
        for line in lines:
            typer.echo(line)
        if not fits:
            fit = [c for c in DL.evaluate(folder, DL.parse_deadline(deadline)[0], engine, roles) if c.recommended]
            if fit:
                c = fit[0]
                flags = f"--student {c.student}" + (f" --teacher {c.teacher}" if c.teacher else "") + \
                    (f" --judge {c.judge}" if c.judge else "") + f" --engine {c.engine}"
                raise SpillError("this project as configured is not estimated to fit the deadline; "
                                 "it was not started and nothing was lowered", f"spill build {folder} "
                                 f"{flags} --deadline {deadline}   (the recommended combination that fits)")
            raise SpillError("nothing is estimated to fit the deadline; the build was not started "
                             "and nothing was lowered", f"spill plan {folder} --deadline <a longer time>")
    kd = kd or {}
    if kd.get("distill_loss") not in (None, "auto", "kl", "ce", "mix"):
        raise SpillError(f"--distill-loss must be kl, ce or mix, not {kd['distill_loss']!r}",
                         f"spill build {folder} --distill-loss mix")
    for key, val, path in (("student", student, "model"), ("teacher", teacher, "model"),
                           ("judge", judge, "model"), ("epochs", epochs, "training"),
                           ("distill_loss", kd.get("distill_loss"), "training"),
                           ("kd_temp", kd.get("kd_temp"), "training"),
                           ("kd_alpha", kd.get("kd_alpha"), "training"),
                           ("kd_topk", kd.get("kd_topk"), "training")):
        if val is not None and cfg[path].get(key) != val:
            changed.append(f"{path}.{key} = {val!r} (was {cfg[path].get(key)!r})")
            cfg[path][key] = val
    resolved = P.resolve_roles(cfg)
    for path, key in (("model", "judge"), ("evaluation", "metric")):
        if resolved[path].get(key) != cfg[path].get(key):
            changed.append(f"{path}.{key} = {resolved[path][key]!r} (was {cfg[path].get(key)!r}"
                           + ("; a named teacher judges unless you name a judge)" if key == "judge"
                              else "; the judge's assessment is the primary metric)"))
            cfg[path][key] = resolved[path][key]
    if changed:
        C.save(folder, cfg)
        typer.echo(f"updated streamweights.toml: {'; '.join(changed)}; the build is a new run")
    train_rows = read_jsonl(folder / "data" / "train.jsonl")
    bad = D.check_sizes(cfg["task"]["type"], len(train_rows),
                        len(read_jsonl(folder / "data" / "val.jsonl")),
                        [r["output"] for r in train_rows])
    if bad:
        raise SpillError("this project cannot be built: " + "; ".join(bad),
                         f"spill init <data> --input <col> --output <col> --project {folder} "
                         f"--force   (with more rows, or your own --val file)")
    with runtime.job_session("build", engine=engine, headless_flag=headless):
        p0 = P.make_plan(folder, engine, fetch=False)
        typer.echo(P.pre_run_line(p0))
        need = [d for d in p0.downloads if not d["present"]]
        if need:
            typer.echo("   downloads first, not counted in the estimate: " + ", ".join(
                f"{d['model']} {d['bytes'] / 1e6:.0f} MB at revision {d['revision'][:12]}" for d in need))
        plan = P.make_plan(folder, engine, fetch=True)
        co = CO.Coordinator(folder, executor, plan.engine, say=typer.echo, stop_after=stop_after)
        out = co.build(plan, new_run=new_run)
    if out.status == "stopped":
        typer.echo(f"\nbuild stopped; run {out.run_id} keeps what was committed")
        _next_hint(f"spill resume {folder}")
        raise typer.Exit(130)
    typer.echo("")
    if out.table:
        task = plan.cfg["task"]["type"]
        names = R.names_for(task)
        w = max(len(names[r["comparator"]]) for r in out.table)
        for r in out.table:
            v = r["primary"]
            shown = ("reference only, judge is the same model" if r.get("judge_excluded")
                     else "unavailable" if v is None else f"{v:.3f}")
            typer.echo(f"{names[r['comparator']].ljust(w)}  {R.primary_name(R.metric_key(plan.cfg))} "
                       f"{shown}  ({r['rows']} rows)")
        if plan.cfg["task"]["type"] == "open" and out.snapshot:
            from .project.common import read_json
            for line in R.judge_findings(read_json(Path(out.snapshot) / "manifest.json").get("judge")):
                typer.echo(f"  {line}")
    typer.echo(f"run {out.run_id}: {out.snapshot}")
    _next_hint(f"spill report {folder}")


@app.command(short_help="Show a project's latest (or a named) run report",
             epilog="Example: spill report tickets")
def report(
    project: Path = typer.Argument(..., help="a project folder"),
    run: str = typer.Argument(None, help="a run id (default: the latest completed run)"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Regenerate REPORT.md (the project index) and print a run's report."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    try:
        from .project.index import records, write_index
        from .project import migrate as MG
        cfg0 = MG.ensure_project(project, say=typer.echo)
        if cfg0 is not None and not MG.is_guided(cfg0):
            _legacy_report(project)
            return
        _guided(project, "report")
        runs = records(project, "runs")
        if not runs:
            raise SpillError(f"{project} has no completed run to report", f"spill build {project}")
        m = next((r for r in runs if r["run_id"] == run), None) if run else runs[-1]
        if m is None:
            raise SpillError(f"no completed run {run} in {project}", f"spill report {project}")
        idx = write_index(project)
        rp = Path(project) / "runs" / m["run_id"] / "report.md"
        typer.echo(rp.read_text())
        typer.echo(f"report: {rp}\nindex: {idx}")
    except Exception as e:
        _fail(e)
        return
    _next_hint(f"spill export {project}")


@app.command(short_help="Hand a project to another folder or s3:// location",
             epilog="Example: spill move tickets s3://my-bucket/tickets")
def move(
    first: str = typer.Argument(..., help="the project folder, or the destination when it is "
                                          "the only argument (the folder is then ./)"),
    dest: str = typer.Argument(None, help="the destination: a folder or s3://bucket/prefix"),
    wait: float = typer.Option(120.0, "--wait", help="seconds to wait for a running build to stop "
                                                     "at its next committed boundary"),
    with_exports: bool = typer.Option(False, "--with-exports", help="also copy the bulk export "
                                                                    "artifacts (records always move)"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Quiesce the writer, commit its progress, copy a consistent snapshot, verify every
    checksum, fence the source, then activate the destination. Safe to run again after an
    interruption. The source bytes are kept."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    folder, target = (Path(first), dest) if dest else (Path("."), first)
    try:
        from .project import move as M
        _guided(folder, "move")
        typer.echo(f"spill move {folder.name if str(folder) != '.' else Path.cwd().name}: hand the "
                   f"committed state to {target}. Source bytes are kept. Cost: $0.")
        r = M.move(folder, target, say=typer.echo, wait_s=wait, with_exports=with_exports)
    except Exception as e:
        _fail(e)
        return
    typer.echo(f"moved {r.files} files ({r.bytes / 1e6:.1f} MB), checksums verified; the source is "
               f"fenced, the destination is active (transfer {r.transfer_id})")
    _next_hint(f"spill resume {r.dest}")


def is_project_target(target: str | None) -> bool:
    from .portable.store import is_uri
    if not target:
        return False
    if is_uri(target) and not target.startswith("file://"):
        return True
    p = Path(target[7:] if target.startswith("file://") else target)
    return p.is_dir() and (p / "streamweights.toml").exists()


def project_resume(target, engine, headless, executor, stop_after):
    """`spill resume <project-or-uri>`: acquire the run's ownership and continue it."""
    from . import runtime
    from .project import coordinator as CO
    from .project import remote as RM
    from .portable.store import is_uri
    remote_uri = None
    if RM.is_remote(target):
        remote_uri = target.rstrip("/")
        project = RM.working_copy(remote_uri)
        n = RM.pull_project(remote_uri, project)
        state_root = f"{remote_uri}/.spill"
        typer.echo(f"spill resume {remote_uri}: authority is the control object there; working "
                   f"copy of {n} project files -> {project}")
    else:
        project = Path(target[7:] if target.startswith("file://") else target).resolve()
        state_root = str(project / ".spill")
    runs = CO.list_runs_at(state_root)
    open_runs = [r for r in runs if r["status"] in ("idle", "running", "handoff")]
    if not open_runs:
        if any(r["status"] == "incoming" for r in runs):
            raise SpillError("this location is the destination of a handoff that was not "
                             "committed, so it cannot run yet",
                             "re-run `spill move` at the source to finish the handoff")
        moved = [r for r in runs if r["status"] == "transferred"]
        if moved:
            raise SpillError(f"run {moved[-1]['run_id']} was handed to "
                             f"{moved[-1]['handoff']['dest']}", f"spill resume {moved[-1]['handoff']['dest']}")
        raise SpillError(f"{target} has no unfinished run; everything is complete",
                         f"spill report {project}")
    rid = sorted(open_runs, key=lambda r: r["run_id"])[-1]["run_id"]
    with runtime.job_session("build", engine=engine, headless_flag=headless):
        co = CO.Coordinator(project, executor, engine, say=typer.echo, stop_after=stop_after,
                            state_root_uri=state_root, remote_project=remote_uri)
        out = co.resume(rid, engine)
    if out.status == "stopped":
        typer.echo(f"\nbuild stopped; run {out.run_id} keeps what was committed")
        _next_hint(f"spill resume {target}")
        raise typer.Exit(130)
    typer.echo(f"run {out.run_id}: completed -> {out.snapshot}")
    _next_hint(f"spill report {project}")


@app.command(name="test", short_help="Score a run on the final test split, as a record",
             epilog="Example: spill test tickets")
def test_cmd(
    project: Path = typer.Argument(..., help="a project folder"),
    run: str = typer.Argument(None, help="a run id (default: the latest completed run)"),
    engine: str = typer.Option(None, "--engine", help="mlx | torch-cpu | torch-cuda"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """The only command that scores the final test labels. It writes tests/<id>/, names the
    exact frozen run, and counts how often the split has been used."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    try:
        from . import runtime
        from .project import report as R
        from .project import testrec
        _guided(project, "test")
        typer.echo(f"spill test {project.name}: score the frozen run on the final test split. "
                   f"Cost: $0. Record -> {project}/tests/<id>")
        with runtime.job_session("test", engine=engine):
            rec = testrec.run_test(project, run, engine, say=typer.echo)
        names = R.NAMES if True else {}
        typer.echo("")
        for t in rec["table"]:
            v = t["primary"]
            typer.echo(f"{t['comparator']:<10} {rec['metric']} {'unavailable' if v is None else f'{v:.3f}'} "
                       f"({t['rows']} rows)")
        typer.echo(rec["holdout_note"])
        typer.echo(f"record: {project}/tests/{rec['id']}")
    except Exception as e:
        _fail(e)
        return
    _next_hint(f"spill report {project}")


@app.command(short_help="Compare runs; rank only when that is honest",
             epilog="Example: spill compare tickets")
def compare(
    paths: list[Path] = typer.Argument(..., help="project folders (all their runs) and/or run "
                                                 "folders"),
    use: list[str] = typer.Option(None, "--use", help="<run id>=<evaluation id>: compare that run "
                                                      "through an evaluation record (default: its "
                                                      "original evaluation); repeatable"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Differences in data, models, training settings, engine and numerics. Each comparison is
    labeled common evaluator, cross-runtime (ranked, with the differences and a caution) or
    incompatible (not ranked, and why)."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    try:
        from .project import compare as CM
        for p in paths:
            if (Path(p) / "streamweights.toml").exists() or (Path(p) / "evals.jsonl").exists():
                _guided(p, "compare")
        uses = {}
        for u in use or []:
            run, sep, ev = u.partition("=")
            if not sep:
                raise SpillError(f"--use {u!r} must be <run id>=<evaluation id>", "spill compare --help")
            uses[run] = ev
        try:
            views = CM.load_views(paths, uses)
        except ValueError as e:
            raise SpillError(str(e), "spill compare <project>")
        if len(views) < 1:
            raise SpillError("no completed runs found there", "spill build <project>")
        for line in CM.render(CM.compare(views)):
            typer.echo(line)
    except Exception as e:
        _fail(e)
        return
    _next_hint(f"spill export {paths[0]}")


@app.command(short_help="Re-evaluate a run on its validation rows, as a record",
             epilog="Example: spill evaluate tickets --engine torch-cpu")
def evaluate(
    project: Path = typer.Argument(..., help="a project folder"),
    run: str = typer.Argument(None, help="a run id (default: the latest completed run)"),
    engine: str = typer.Option(None, "--engine", help="mlx | torch-cpu | torch-cuda: the evaluator"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Score the run's exact model and adapter again on its frozen validation rows under the
    chosen engine. Creates evaluations/<id>/ and never touches the run. The final test stays with
    `spill test`."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    try:
        from . import runtime
        from .project import evalrec
        _guided(project, "evaluate")
        typer.echo(f"spill evaluate {project.name}: re-score the run on its frozen validation rows"
                   f"{' on ' + engine if engine else ''}. Cost: $0. Record -> {project}/evaluations/<id>")
        with runtime.job_session("evaluate", engine=engine):
            rec = evalrec.run_evaluate(project, run, engine, say=typer.echo)
        typer.echo("")
        fmt = lambda x: "unavailable" if x is None else f"{x:.3f}"
        for t in rec["table"]:
            ch = rec["vs_original"][t["comparator"]]
            typer.echo(f"{t['comparator']:<10} {rec['metric']} {fmt(t['primary'])} "
                       f"(original {fmt(ch['original'])}, {rec['rows']} rows)")
        typer.echo(f"record: {project}/evaluations/{rec['id']}")
    except Exception as e:
        _fail(e)
        return
    _next_hint(f"spill compare {project} --use {rec['source_run']}={rec['id']}")


def project_export(project: Path, run: str | None, gguf: str | None, verify_rows: int,
                   merge_dtype: str | None = None):
    from .project import exportrec
    typer.echo(f"spill export {project.name}: merge the run's LoRA adapter into its bf16 base"
               + (f", convert to GGUF {gguf}" if gguf else "")
               + f", then load the artifact in an independent runtime on {verify_rows} validation "
                 f"rows. Cost: $0. Record -> {project}/exports/<id>")
    from .project import exportrec as _ex
    md = merge_dtype or _ex.DEFAULT_MERGE_DTYPE
    rec = exportrec.run_export(project, run, gguf, verify_rows, say=lambda s: typer.echo(f"   {s}"),
                               merge_dtype=md)
    for label, v in rec["verification"]["runs"].items():
        d = v["prediction_differences"]
        q = v["quality_delta"]
        typer.echo(f"verified {label}: {v['runtime']}; {d['count']} of {d['of']} predictions differ "
                   f"from the training engine's. Quality on these {q['rows']} rows ({q['metric']}): source "
                   f"engine {q['source_engine_score']:.3f}, artifact {q['artifact_score']:.3f}, "
                   f"change {q['delta']:+.3f}. Verified means it loaded and ran, not that quality was kept")
        dep = rec["deployment"][label]
        mem = dep["peak_memory_bytes"]
        typer.echo(f"   time to first token cold {dep['ttft_cold_s']} s, warm {dep['ttft_warm_s']} s; "
                   f"{dep['tokens_per_s_warm']} tokens/s; peak memory "
                   f"{'unavailable' if mem == 'unavailable' else f'{mem / 1e9:.2f} GB'}")
    typer.echo(f"record: {project}/exports/{rec['id']}")
    first = next(iter(rec["scripts"].values()))
    _next_hint(f"python {project}/exports/{rec['id']}/{first} \"<text>\"")


@app.command(short_help="Make, check or install an offline project bundle",
             epilog="Example: spill bundle tickets tickets.bundle")
def bundle(
    first: Path = typer.Argument(..., help="the project folder (to make a bundle), or the bundle "
                                           "(with --verify or --install)"),
    dest: Path = typer.Argument(None, help="the bundle path to create, or the project folder to "
                                           "copy out with --install"),
    verify: bool = typer.Option(False, "--verify", help="check every checksum in a bundle"),
    install: bool = typer.Option(False, "--install", help="verify a bundle, put its models in this "
                                                          "machine's cache and copy its project to "
                                                          "<dest>"),
    with_exports: bool = typer.Option(False, "--with-exports", help="also include bulk export "
                                                                    "artifacts"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Everything a workflow needs to run without network access: the project, its run state
    (a consistent committed snapshot) and the pinned model files, with a checksummed manifest.
    Python dependencies are installed separately."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    try:
        from .project import bundle as B
        if not verify and not install:
            _guided(first, "bundle")
        if verify:
            m = B.verify_bundle(first)
            typer.echo(f"bundle ok: {len(m['files'])} files, every checksum matches")
            _next_hint(f"spill bundle --install {first} <project dir>")
        elif install:
            if dest is None:
                raise SpillError("--install needs the project folder to copy out", f"spill bundle "
                                 f"--install {first} <project dir>")
            p = B.install_bundle(first, dest, say=typer.echo)
            typer.echo(f"project copied to {p}; its runs are read-only copies")
            _next_hint(f"spill build {p} --new-run")
        else:
            if dest is None:
                raise SpillError("bundle needs the path to write", f"spill bundle {first} <path>")
            typer.echo(f"spill bundle {first.name}: project, run state and pinned models into {dest}. "
                       f"Cost: $0. Python dependencies are not included.")
            m = B.create_bundle(first, dest, say=typer.echo, with_exports=with_exports)
            size = sum(f["bytes"] for f in m["files"].values())
            typer.echo(f"bundle: {len(m['files'])} files, {size / 1e9:.2f} GB -> {dest}")
            _next_hint(f"spill bundle --verify {dest}")
    except Exception as e:
        _fail(e)


def _legacy_report(folder: Path) -> None:
    """`spill report` on a flat-layout folder: the table its last build recorded."""
    from . import build as build_mod
    f = build_mod.read_folder(folder)
    doc = build_mod.load_state(f)
    if not doc:
        raise SpillError(f"{folder} has no build to report", f"spill build {folder}")
    typer.echo(build_mod.table_from_state(doc))
    typer.echo(f"\n(flat-layout folder: this is the table spill build recorded in {folder}/.build/)")
    _next_hint(f"spill export {doc.get('student', '<base>')}+{doc.get('name', '<adapter>')}")
