"""Commands of the guided workflow: init, plan, report, compare, test, bundle, move (and the
project paths of build, export and resume). Registered from cli.py."""

from __future__ import annotations

from pathlib import Path

import typer

from .cli import _fail, _next_hint, app
from .errors import SpillError


@app.command(short_help="Start a project from a CSV or JSONL of labeled examples",
             epilog="Example: spill init tickets.csv --input text --output label")
def init(
    data: Path = typer.Argument(..., help="a CSV or JSONL of labeled examples"),
    input_col: str = typer.Option(..., "--input", help="the column holding the input text"),
    output_col: str = typer.Option(..., "--output", help="the column holding the right answer: "
                                                         "a label, or a JSON object"),
    group: str = typer.Option(None, "--group", help="a column whose rows must stay on one side "
                                                    "of every split (customer, document, ...)"),
    system: Path = typer.Option(None, "--system", help="a text file of instructions for the task"),
    task: str = typer.Option(None, "--task", help="classification | json (default: suggested "
                                                  "from the outputs; required when unclear)"),
    project: Path = typer.Option(None, "--project", help="the project folder (default: the "
                                                         "data file's name)"),
    labels: str = typer.Option(None, "--labels", help="the class vocabulary: a,b,c or a file of "
                                                      "one label per line (default: the "
                                                      "training rows' labels)"),
    schema: Path = typer.Option(None, "--schema", help="a JSON Schema file for --task json "
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
                             val_fraction, test_fraction, force, say=typer.echo)
    except D.DataErrors as e:
        for line in e.lines():
            typer.echo(f"spill: {line}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        _fail(e)
        return
    s = res.sizes
    typer.echo(f"spill init {data.name}: {res.task} ({res.task_note}). Train {s['train']}, "
               f"validation {s['val']}, final test {s['test']} rows. Cost: $0. Project -> {folder}")
    for n in res.notes:
        typer.echo(f"   {n}")
    for w in res.warnings:
        typer.echo(f"   warning: {w}")
    _next_hint(f"spill plan {folder}")


@app.command(short_help="Show what a build will do, before it does anything",
             epilog="Example: spill plan tickets")
def plan(
    project: Path = typer.Argument(..., help="a project folder (from spill init)"),
    engine: str = typer.Option(None, "--engine", help="mlx | torch-cpu | torch-cuda "
                                                      "(default: chosen from the hardware)"),
    calibrate: bool = typer.Option(False, "--calibrate", help="first measure this engine's matmul "
                                                              "and memory rates on synthetic "
                                                              "data (a few seconds)"),
    debug: bool = typer.Option(False, "--debug", hidden=True),
):
    """Models and engine with reasons, downloads, disk, and a duration per stage labeled
    measurement, assumption or unknown. Loads no model, downloads nothing, trains nothing."""
    import streamweights.cli as cli
    cli._DEBUG = debug
    try:
        from . import engine_select
        from .project import plan as P
        if calibrate:
            r = engine_select.measure_rates()
            typer.echo("calibration (synthetic matmul and memory, nothing trained): "
                       + "; ".join(f"{e} {v.get('matmul_tflops', '?')} TFLOP/s" for e, v in r.items()))
        p = P.make_plan(project, engine, fetch=False)
        for line in P.render(p):
            typer.echo(line)
    except Exception as e:
        _fail(e)
        return
    _next_hint(f"spill build {project}")


def project_build(folder, student, teacher, epochs, engine, headless, executor, new_run,
                  stop_after, notify):
    """`spill build <project>`: plan, run, report. Flags that change the build are written
    into streamweights.toml first (one line each), so the project stays the record."""
    from . import runtime
    from .project import config as C
    from .project import coordinator as CO
    from .project import data as D
    from .project import plan as P
    from .project import report as R
    from .project.common import read_jsonl
    folder = Path(folder)
    cfg = C.load(folder)
    changed = []
    for key, val, path in (("student", student, "model"), ("teacher", teacher, "model"),
                           ("epochs", epochs, "training")):
        if val is not None and cfg[path].get(key) != val:
            changed.append(f"{path}.{key} = {val!r} (was {cfg[path].get(key)!r})")
            cfg[path][key] = val
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
        names = R.NAMES if task == "classification" else R.JSON_NAMES
        w = max(len(names[r["comparator"]]) for r in out.table)
        for r in out.table:
            v = r["primary"]
            typer.echo(f"{names[r['comparator']].ljust(w)}  {R.primary_name(task)} "
                       f"{'unavailable' if v is None else f'{v:.3f}'}  ({r['rows']} rows)")
    typer.echo(f"run {out.run_id}: {out.snapshot}")
    _next_hint(f"spill report {folder}")


@app.command(short_help="Show the report of a project's latest (or a named) run",
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
        from .project import config as C
        from .project.index import records, write_index
        C.load(project)
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
