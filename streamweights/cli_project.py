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
