"""The final job of relay.yml: did both relays finish, do their scores sit within the measured
noise of the uninterrupted reference, is every row and step present once, and does each stage's
recorded machine and OS match the job that ran it.

  python scripts/relay_check.py --reference DIR --l2m DIR --m2l DIR [--summary FILE]

Each DIR holds relay-state/ (the build state) and where-*.json files written by the jobs
({"job", "system", "host", "engine"}). Reads docs/reports/014-noise-floor.json.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from streamweights import build as B  # noqa: E402
from streamweights.build_audit import audit  # noqa: E402


def load(d: Path) -> dict:
    return json.loads((d / "relay-state" / "state.json").read_text())


def wheres(d: Path) -> dict:
    return {j["job"]: j for j in (json.loads(p.read_text()) for p in sorted(d.glob("where-*.json")))}


def scores(doc):
    return {s["id"]: s["result"]["score"] for s in doc["stages"] if s["kind"] == "eval"}


def expected(doc, start: dict, finish: dict) -> list[str]:
    """Every producer of every stage must be the start job or the finish job's (system, host);
    the first eval ran on the start job, the last on the finish job, and the tune on both."""
    errs = []
    ids = {s["id"]: s for s in doc["stages"]}
    key = lambda p: (p.get("system"), p.get("host"))
    a, b = (start["system"], start["host"]), (finish["system"], finish["host"])
    for sid, want in (("eval:base", [a]), ("eval:tuned", [b])):
        got = [key(p) for p in ids[sid]["producers"]]
        if sorted(set(got)) != sorted(set(want)):
            errs.append(f"{sid} recorded {got}, ran on {want}")
    got = [key(p) for p in ids["tune"]["producers"]]
    if got[0] != a or got[-1] != b:
        errs.append(f"tune recorded {got}, ran on {a} then {b}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--l2m", type=Path, required=True)
    ap.add_argument("--m2l", type=Path, required=True)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--noise", type=Path, default=REPO / "docs/reports/014-noise-floor.json")
    a = ap.parse_args()
    noise = json.loads(a.noise.read_text())
    tol = noise["tolerance"]
    ref = load(a.reference)
    rs = scores(ref)
    lines = ["## Relay", "",
             f"Noise floor (tiny build, {len(noise['runs'])} runs on one Mac): spread "
             f"{noise['tuned_spread']:.2f}, tolerance {tol:.2f}. {noise['justification']}", "",
             f"Reference (uninterrupted, Linux): base {rs['eval:base']:.3f}, tuned "
             f"{rs['eval:tuned']:.3f}", ""]
    failed = []
    for name, d, sfirst, sfinish in (("Linux to macOS", a.l2m, "start", "finish"),
                                     ("macOS to Linux", a.m2l, "start", "finish")):
        doc, w = load(d), wheres(d)
        sc, au = scores(doc), audit(str(d / "relay-state"))
        checks = {
            "completed": doc.get("finished") is True and all(s["status"] == "done" for s in doc["stages"]),
            "base within noise": abs(sc["eval:base"] - rs["eval:base"]) <= tol + 1e-9,
            "tuned within noise": abs(sc["eval:tuned"] - rs["eval:tuned"]) <= tol + 1e-9,
            "tuned beats base": sc["eval:tuned"] > sc["eval:base"],
            "no row or step missing or repeated": audit_ok(au, doc),
        }
        errs = expected(doc, w[sfirst], w[sfinish]) if doc.get("finished") else ["not finished"]
        checks["recorded machine and OS match where it ran"] = not errs
        lines += [f"### {name}", "", B.table_from_state(doc, ref).replace("\n", "\n"), "",
                  "| check | result |", "|---|---|"]
        lines += [f"| {k} | {'pass' if v else 'FAIL'} |" for k, v in checks.items()]
        lines += ["", f"ran on: start {w[sfirst]['system']} ({w[sfirst]['engine']}), finish "
                  f"{w[sfinish]['system']} ({w[sfinish]['engine']})", ""]
        lines += [f"- {e}" for e in errs]
        failed += [f"{name}: {k}" for k, v in checks.items() if not v]
    out = "\n".join(lines)
    print(out)
    if a.summary:
        with open(a.summary, "a") as f:
            f.write(out + "\n")
    if failed:
        print("\nFAILED: " + "; ".join(failed), file=sys.stderr)
        sys.exit(1)


def audit_ok(au, doc) -> bool:
    rows_ok = all(j["rows"] == j["unique"] for j in au["row_jobs"].values()) and \
        all(j["rows"] == 20 for j in au["row_jobs"].values()) and len(au["row_jobs"]) == 2
    steps = au["tune"] and au["tune"]["contiguous_1_to_n"] and \
        au["tune"]["final_step"] == doc["stages"][1]["result"]["steps"]
    return bool(rows_ok and steps)


if __name__ == "__main__":
    main()
