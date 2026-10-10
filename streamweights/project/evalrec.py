"""spill evaluate: re-evaluate a completed run on its frozen validation rows, under the chosen
evaluator, as a record of its own.

The run is never modified. The record (evaluations/<id>/record.json) references the run and its
manifest hash, uses the run's exact model and adapter identity and the scorer in this code, reads
only the run's frozen validation inputs (never the final test: that stays with `spill test`),
stores the conditions each comparator actually ran under, and shows the original evaluation next
to the new one with both metric versions. A record is immutable once complete; a failure is a
failed record."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ..errors import SpillError
from . import config as C
from . import report as R
from .common import atomic_write, new_id, read_json, sha_file, write_json
from .index import records, write_index
from .runstate import make_readonly
from .testrec import pick_run, score_comparators


def run_evaluate(project: Path, run_id: str | None = None, engine: str | None = None,
                 say=print) -> dict:
    project = Path(project)
    m = pick_run(project, run_id)
    rdir = project / "runs" / m["run_id"]
    cfg = C.parse((rdir / "inputs" / "streamweights.toml").read_bytes())
    task = cfg["task"]["type"]
    metric = R.metric_key(cfg)
    eid = new_id("eval-")
    stg = project / "evaluations" / f".staging-{eid}"
    final = project / "evaluations" / eid
    shutil.rmtree(stg, ignore_errors=True)
    stg.mkdir(parents=True)
    t0 = time.monotonic()
    val_file = rdir / "inputs" / "val.jsonl"
    orig_version = m.get("metric_version")
    rec = {"schema": 1, "id": eid, "kind": "evaluation", "source_run": m["run_id"],
           "source_manifest_sha256": sha_file(rdir / "manifest.json"),
           "source_identity": m["identity"], "rows_source": "the run's frozen validation inputs",
           "val_sha256": sha_file(val_file), "evaluator_engine_requested": engine or "auto",
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "running"}
    try:
        comps, preds, protocol, rows = score_comparators(project, m, cfg, val_file, stg, engine, say,
                                                         "validation")
        table = R.table(task, {c: {"metrics": v["metrics"]} for c, v in comps.items()}, metric)
        orig = {t["comparator"]: t["primary"] for t in m["table"]}
        change = {t["comparator"]: {"original": orig.get(t["comparator"]), "new": t["primary"],
                                    "difference": (None if orig.get(t["comparator"]) is None
                                                   or t["primary"] is None
                                                   else round(t["primary"] - orig[t["comparator"]], 6))}
                  for t in table}
        rec.update(status="completed", rows=len(rows), comparators=comps, table=table,
                   protocol={"sha256": protocol["sha256"], "fields": protocol["fields"]},
                   metric_key=metric, metric=R.primary_name(metric),
                   metric_version=C.metric_version(cfg), original_metric_version=orig_version,
                   original_evaluation=m.get("evaluation", {}).get("id", f"{m['run_id']}/original"),
                   original_protocol_sha256=m["protocol"]["sha256"], vs_original=change,
                   seconds=round(time.monotonic() - t0, 1),
                   finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   findings=R.findings(task, {c: {"metrics": v["metrics"]} for c, v in comps.items()}, metric),
                   note="This record does not modify the run. The final test is not used here.")
        write_json(stg / "disagreements.json", R.disagreements(task, preds))
        atomic_write(stg / "report.md", _render(rec).encode())
    except Exception as e:
        rec.update(status="failed", error=f"{type(e).__name__}: {e}",
                   seconds=round(time.monotonic() - t0, 1), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        shutil.rmtree(stg / "attempt", ignore_errors=True)
        shutil.rmtree(stg / "stages", ignore_errors=True)
        write_json(stg / "record.json", rec)
        stg.rename(final)
        make_readonly(final)
        write_index(project)
        raise SpillError(f"the evaluation failed: {e}; a failed record was kept at {final}",
                         f"spill evaluate {project} {m['run_id']}") from e
    shutil.rmtree(stg / "attempt", ignore_errors=True)
    shutil.rmtree(stg / "stages", ignore_errors=True)
    write_json(stg / "record.json", rec)
    stg.rename(final)
    make_readonly(final)
    write_index(project)
    return rec


def _render(rec: dict) -> str:
    lines = [f"# Evaluation {rec['id']}", "",
             f"Run {rec['source_run']}, {rec['rows']} validation rows, metric {rec['metric']} "
             f"(version {rec['metric_version']}; the run was built with version "
             f"{rec['original_metric_version']}).", ""]
    for c, v in rec["vs_original"].items():
        lines.append(f"- {c}: original {v['original']}, now {v['new']}, difference {v['difference']}")
    lines += ["", "Conditions each comparator ran under:", ""]
    for c, v in rec["comparators"].items():
        cd = v.get("conditions") or {}
        lines.append(f"- {c}: engine {cd.get('engine')}, device {cd.get('device')}, numerics "
                     f"{cd.get('numerics')}, decoding as applied {cd.get('decoding_applied')}")
    lines += ["", rec["note"], ""]
    return "\n".join(lines)
