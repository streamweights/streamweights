"""spill test: score a frozen run on the final test split, once, as a record of its own.

The final test labels are read by nothing else: not training, not schema or label inference,
not calibration, not the build report. This command evaluates the run's comparators on the
test rows and writes tests/<id>/ (record.json, report.md, predictions). The record names the
exact run and its fingerprints, and counts how many times the split has now been scored, so a
split that is consulted repeatedly is not presented as an untouched holdout. A record is
immutable once complete; a failure is kept as a failed record."""

from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path

from ..errors import SpillError
from . import config as C
from . import report as R
from .common import atomic_write, new_id, read_json, read_jsonl, sha_file, write_json, write_jsonl
from .index import records, test_uses, write_index
from .runstate import make_readonly
from .stagefns import StageContext, StageDesc, run_stage


def pick_run(project: Path, run_id: str | None) -> dict:
    runs = records(project, "runs")
    if not runs:
        raise SpillError(f"{project} has no completed run", f"spill build {project}")
    if run_id:
        m = next((r for r in runs if r["run_id"] == run_id), None)
        if m is None:
            raise SpillError(f"no completed run {run_id}", f"spill report {project}")
        return m
    return runs[-1]


def run_test(project: Path, run_id: str | None = None, engine: str | None = None,
             say=print) -> dict:
    project = Path(project)
    m = pick_run(project, run_id)
    rdir = project / "runs" / m["run_id"]
    test_file = project / "data" / "test.jsonl"
    if not test_file.exists() or not read_jsonl(test_file):
        raise SpillError("this project has no final-test rows", f"spill init ... --test <file>")
    if sha_file(test_file) != m["data"]["test_sha256"]:
        raise SpillError(f"data/test.jsonl changed since run {m['run_id']} was built; the run is "
                         f"frozen against the test file it was built next to",
                         f"spill build {project} --new-run")
    tid = new_id("test-")
    uses_before = sum(test_uses(project).values())
    stg = project / "tests" / f".staging-{tid}"
    final = project / "tests" / tid
    shutil.rmtree(stg, ignore_errors=True)
    stg.mkdir(parents=True)
    cfg = C.parse((rdir / "inputs" / "streamweights.toml").read_bytes())
    task = cfg["task"]["type"]
    t0 = time.monotonic()
    rec = {"schema": 1, "id": tid, "kind": "test", "source_run": m["run_id"],
           "source_manifest_sha256": sha_file(rdir / "manifest.json"),
           "source_identity": m["identity"], "test_sha256": m["data"]["test_sha256"],
           "run_protocol_sha256": m["protocol"]["sha256"], "uses_before": uses_before,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "running"}
    try:
        comps, preds, protocol, test_rows = score_comparators(
            project, m, cfg, test_file, stg, engine, say, "final-test")
        metric = R.metric_key(cfg)
        table = R.table(task, {c: {"metrics": v["metrics"]} for c, v in comps.items()}, metric)
        rec.update(status="completed", rows=len(test_rows), comparators=comps, table=table,
                   protocol={"sha256": protocol["sha256"]}, metric=R.primary_name(metric), metric_key=metric, metric_version=C.metric_version(cfg),
                   seconds=round(time.monotonic() - t0, 1),
                   finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   findings=R.findings(task, {c: {"metrics": v["metrics"]} for c, v in comps.items()}, metric),
                   holdout_note=(f"This is use {uses_before + 1} of this project's final test split. "
                                 + ("It was not consulted before." if uses_before == 0 else
                                    f"It was scored {uses_before} time(s) before, so it is not an untouched "
                                    f"holdout for this run's selection.")))
        write_json(stg / "disagreements.json", R.disagreements(task, preds))
        atomic_write(stg / "report.md", _render(rec, task).encode())
    except Exception as e:
        rec.update(status="failed", error=f"{type(e).__name__}: {e}",
                   seconds=round(time.monotonic() - t0, 1),
                   finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        shutil.rmtree(stg / "attempt", ignore_errors=True)
        shutil.rmtree(stg / "stages", ignore_errors=True)
        write_json(stg / "record.json", rec)
        stg.rename(final)
        make_readonly(final)
        write_index(project)
        raise SpillError(f"the test evaluation failed: {e}; a failed record was kept at {final}",
                         f"spill test {project} {m['run_id']}") from e
    shutil.rmtree(stg / "attempt", ignore_errors=True)
    shutil.rmtree(stg / "stages", ignore_errors=True)
    write_json(stg / "record.json", rec)
    stg.rename(final)
    make_readonly(final)
    write_index(project)
    return rec


def score_comparators(project: Path, m: dict, cfg: dict, rows_file: Path, stg: Path, engine,
                      say, what: str):
    """Re-evaluate every comparator of run `m` on the rows in `rows_file`, with the run's exact
    model and adapter identity and the scorer in this code. -> (comparators, predictions,
    protocol, rows). Each comparator's entry carries the conditions it actually ran under."""
    rdir = Path(project) / "runs" / m["run_id"]
    task = cfg["task"]["type"]
    att = stg / "attempt"
    (att / "inputs").mkdir(parents=True)
    for f in (rdir / "inputs").iterdir():
        if f.is_file():
            shutil.copyfile(f, att / "inputs" / f.name)
    shutil.copyfile(rows_file, att / "inputs" / "val.jsonl")      # the stage code reads val.jsonl
    rows = read_jsonl(rows_file)
    comps, preds = {}, {}
    for c in [c for c in R.COMPARATORS if c in m["comparators"]]:
        if c == "baseline":
            desc = StageDesc("baseline", "baseline", "baseline", {
                "training_rows": str(rdir / "artifacts" / "training.rows.jsonl")})
        else:
            model = cfg["model"]["teacher"] if c == "teacher" else cfg["model"]["student"]
            params = {"comparator": c, "model": model}
            if c == "trained":
                params["adapter"] = str(rdir / "artifacts" / "adapter")
            desc = StageDesc(f"eval:{c}", "eval", c, params)
        sd = stg / "stages" / c
        sd.mkdir(parents=True)
        say(f"scoring {c} on {len(rows)} {what} rows")
        out = run_stage(desc, StageContext(att, sd, engine, None, None, say))
        if out.status != "done":
            raise SpillError(f"the {c} evaluation stopped before finishing")
        comps[c] = {"metrics": out.metrics, "producers": out.producers, "seconds": out.seconds,
                    "conditions": (out.notes or {}).get("conditions")}
        preds[c] = read_jsonl(sd / "out" / "predictions.jsonl")
        write_jsonl(stg / "predictions" / f"{c}.jsonl", preds[c])
    schema = read_json(att / "inputs" / cfg["contract"]["schema_file"]) if task == "json" else None
    protocol = R.protocol_fingerprint(cfg, schema, sha_file(rows_file), [r["id"] for r in rows])
    return comps, preds, protocol, rows


def _render(rec: dict, task: str) -> str:
    names = R.NAMES if task == "classification" else R.JSON_NAMES
    lines = [f"# Final test {rec['id']}", "",
             f"Run {rec['source_run']} (identity {rec['source_identity'][:12]}), {rec['rows']} "
             f"final-test rows, metric {rec['metric']}.", "", rec["holdout_note"], ""]
    for t in rec["table"]:
        v = t["primary"]
        lines.append(f"- {names[t['comparator']]}: {'unavailable' if v is None else f'{v:.3f}'} "
                     f"({t['rows']} rows, {t.get('invalid_predictions') or 0} invalid, "
                     f"{t.get('inference_failures') or 0} failures)")
    lines += [""] + rec["findings"] + ["",
              "No significance test was run. The run's validation report is unchanged; this "
              "record does not modify it.", ""]
    return "\n".join(lines)
