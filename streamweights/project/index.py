"""The project-root REPORT.md: a regenerable index over runs, exports and tests. It may change
whenever the project does; a completed run's own report never does."""

from __future__ import annotations

import json
from pathlib import Path

from .common import atomic_write


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def records(project: Path, kind: str) -> list[dict]:
    out = []
    base = Path(project) / kind
    if base.exists():
        for d in sorted(base.iterdir()):
            rec = _load(d / ("manifest.json" if kind == "runs" else "record.json"))
            if rec:
                out.append(rec)
    return out


def test_uses(project: Path) -> dict:
    """{run id: number of recorded final-test evaluations}."""
    n: dict = {}
    for r in records(project, "tests"):
        if r.get("status") == "completed":
            n[r["source_run"]] = n.get(r["source_run"], 0) + 1
    return n


def _label(m: dict) -> str:
    from . import report as R
    return R.primary_name(R.key_of(m.get("metric_key") or m["metric"]))


def render(project: Path) -> str:
    project = Path(project)
    runs = records(project, "runs")
    uses = test_uses(project)
    lines = [f"# {project.name}", "",
             "This file is an index. It is regenerated; each run's own report "
             "(runs/<id>/report.md) never changes.", "", "## Runs", ""]
    if not runs:
        lines.append("No completed run yet. `spill build` makes one.")
    else:
        lines += ["| run | metric | trained | baseline or untrained | engine | final test uses |",
                  "|---|---|---|---|---|---|"]
        for m in runs:
            tbl = {r["comparator"]: r for r in m["table"]}
            def v(c):
                x = tbl.get(c, {}).get("primary")
                return "unavailable" if x is None else f"{x:.3f}"
            base = "baseline" if "baseline" in tbl else "untrained"
            engines = sorted({p.get("engine") for s in m["stages"].values()
                              for p in s.get("producers", []) if p.get("engine")})
            lines.append(f"| {m['run_id']} | {_label(m)} | {v('trained')} | {base} {v(base)} | "
                         f"{', '.join(engines) or '-'} | {uses.get(m['run_id'], 0)} |")
    par = [(m["run_id"], m["parent"]) for m in runs if m.get("parent")]
    if par:
        lines += [""] + [f"- Run {r} has parent {p}: experiment lineage only; {r} trained from the base "
                         f"model, not from {p}'s weights." for r, p in par]
    ex = records(project, "exports")
    lines += ["", "## Exports", ""]
    lines += ([f"- {e['id']}: {e['format']} from {e['source_run']}, {e['status']}" for e in ex]
              or ["None yet. `spill export <project>` makes one."])
    evs = records(project, "evaluations")
    if evs:
        lines += ["", "## Evaluations", ""]
        lines += [f"- {e['id']}: run {e['source_run']}, {e['status']}" for e in evs]
    ts = records(project, "tests")
    lines += ["", "## Final test", ""]
    if ts:
        lines.append(f"The final test split has been scored {sum(uses.values())} time(s). A "
                     f"split consulted repeatedly is not an untouched holdout.")
        lines += [f"- {t['id']}: run {t['source_run']}, {t['status']}" for t in ts]
    else:
        lines.append("Not used yet. `spill test <project>` scores a run on it once and records it.")
    lines.append("")
    return "\n".join(lines)


def write_index(project: Path) -> Path:
    p = Path(project) / "REPORT.md"
    atomic_write(p, render(project).encode())
    return p
