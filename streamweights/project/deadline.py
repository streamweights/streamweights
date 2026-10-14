"""spill plan --deadline: which combinations of student, teacher, judge and engine on this machine
are estimated to finish in time.

Local planning only: nothing is downloaded, provisioned or scheduled, and no model is loaded. Every
combination is estimated with the same cost model and the same calibrated rates as `spill plan`, and
says whether its estimate is measured (every stage has a measurement on this machine) or assumed
(at least one stage uses the cost model's stated rate), or unknown (a stage has neither). Models the
machine does not have show their download size; the time to download them is not counted, because it
depends on a connection this tool cannot measure.

The recommendation is a rule, not a taste: among the combinations that fit, the one with the largest
student, then the largest teacher, then a judge, then the shorter estimate. `spill build --deadline`
builds what the project is configured to build when that fits, and otherwise stops and lists what
does fit, with the exact command: it never lowers a quality setting on its own."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import estimate as est
from ..errors import SpillError
from . import config as C
from . import judge as J
from . import plan as P

UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}
_PART = re.compile(r"(\d+(?:\.\d+)?)\s*([smhdw])", re.IGNORECASE)

# parameters in billions of the curated models a deadline plan may weigh; a student must be small
# enough to tune on one machine, a teacher or judge may be bigger than its memory
STUDENTS = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:7b"]
BIG = ["qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:32b", "llama3.3:70b"]
JUDGES = ["qwen2.5:1.5b", "qwen2.5:7b"]


def parse_deadline(text: str, now: dt.datetime | None = None) -> tuple[float, str]:
    """(seconds from now, a label) for a duration (`90m`, `2h`, `1d12h`, `1.5h`) or a timestamp
    (`2026-10-14T09:00`, `2026-10-14 09:00`, `2026-10-14`, local time)."""
    now = now or dt.datetime.now()
    t = text.strip()
    parts = _PART.findall(t)
    if parts and _PART.sub("", t).strip() == "":
        secs = sum(float(n) * UNITS[u.lower()] for n, u in parts)
        if secs <= 0:
            raise SpillError(f"the deadline {text!r} is not in the future", "spill plan <project> --deadline 2h")
        return secs, f"{text.strip()} from now (by {(now + dt.timedelta(seconds=secs)):%Y-%m-%d %H:%M})"
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            when = dt.datetime.strptime(t, fmt)
        except ValueError:
            continue
        secs = (when - now).total_seconds()
        if secs <= 0:
            raise SpillError(f"the deadline {text!r} has already passed", "spill plan <project> --deadline 2h")
        return secs, f"{when:%Y-%m-%d %H:%M}"
    raise SpillError(f"cannot read the deadline {text!r}: use a duration such as 90m, 2h or 1d12h, or a "
                     f"timestamp such as 2026-10-14T09:00", "spill plan <project> --deadline 2h")


@dataclass
class Combo:
    student: str
    teacher: str
    judge: str
    engine: str
    seconds: float | None
    basis: str                     # measured | assumed | unknown
    downloads: list = field(default_factory=list)       # [(tag, bytes)] not on this machine
    configured: bool = False
    fits: bool | None = None
    recommended: bool = False
    note: str = ""

    @property
    def download_bytes(self) -> int:
        return sum(b for _, b in self.downloads)

    def label(self) -> str:
        roles = [f"{self.student} student"]
        if self.teacher:
            roles.append(f"{self.teacher} teacher")
        if self.judge:
            roles.append(f"{self.judge} judge" + (" (the teacher)" if self.judge == self.teacher else ""))
        return " + ".join(roles) + f" on {self.engine}"


def _params(tag: str) -> float:
    return est.PARAMS_B.get(tag, est.params_b(tag))


def _basis(plan) -> str:
    if any(e.seconds is None for e in plan.estimates.values()):
        return "unknown"
    return "measured" if all(e.basis == "measurement" for e in plan.estimates.values()) else "assumed"


def engines_here(only: str | None = None) -> list[str]:
    from .. import engine_select
    av = engine_select.availability()
    out = [e for e, (ok, _) in av.items() if ok]
    if only:
        if only not in out:
            raise SpillError(f"engine {only} is not usable here ({av.get(only, (False, 'unknown'))[1]})",
                             "spill doctor")
        return [only]
    return out


def candidates(cfg: dict) -> list[tuple[str, str, str]]:
    """(student, teacher, judge) triples to weigh, always including what the project says. A judge
    only matters for open-ended output."""
    open_ended = cfg["task"]["type"] == "open"
    cs, ct, cj = (cfg["model"].get("student") or "", cfg["model"].get("teacher") or "",
                  P.judge_tag_of(cfg))
    students = list(dict.fromkeys([cs] + STUDENTS))
    out = []
    for s in students:
        teachers = ["", *[t for t in BIG if _params(t) > _params(s)]]
        if ct and ct not in teachers:
            teachers.append(ct)
        for t in dict.fromkeys(teachers):
            judges = [""]
            if open_ended:
                judges += ([t] if t else []) + JUDGES + ([cj] if cj else [])
            for j in dict.fromkeys(judges):
                out.append((s, t, j))
    return list(dict.fromkeys(out))


def evaluate(project: Path, deadline_s: float, engine: str | None = None,
             roles: dict | None = None) -> list[Combo]:
    """Every combination, estimated. Models are identified from the registry only, so nothing is
    hashed, loaded or fetched. `roles` are the student, teacher and judge a command line named,
    which stand in for what streamweights.toml says (nothing is written)."""
    cfg = C.load(project)
    if roles:
        cfg["model"].update(roles)
    cfg_roles = P.resolve_roles(cfg)
    cur = (cfg_roles["model"]["student"], cfg_roles["model"].get("teacher") or "", P.judge_tag_of(cfg_roles))
    combos = []
    for eng in engines_here(engine):
        for s, t, j in candidates(cfg):
            try:
                plan = P.make_plan(project, eng, fetch=False, lite=True, cfg_override={
                    "model": {"student": s, "teacher": t, "judge": j or "none"}})
            except SpillError:
                continue                      # not buildable (for example, unlabeled prompts with no teacher)
            c = Combo(s, t, j, eng, plan.total_s, _basis(plan))
            known = sum(e.seconds for e in plan.estimates.values() if e.seconds is not None)
            if c.seconds is None:
                c.seconds = known
                c.note = "some stages have no estimate; this is the known part"
            seen = set()
            for d in plan.downloads:
                if not d["present"] and d["model"] not in seen and d["role"] != "embedding":
                    seen.add(d["model"])
                    c.downloads.append((d["model"], d["bytes"]))
            c.configured = (s, t, j) == cur and eng == plan_engine(project, engine)
            c.fits = None if c.basis == "unknown" else c.seconds <= deadline_s
            combos.append(c)
    mark_recommended(combos, deadline_s)
    return combos


def plan_engine(project: Path, engine: str | None) -> str:
    cfg = C.load(project)
    return P.engine_pick(engine or (cfg["model"].get("engine") if cfg["model"].get("engine") not in (None, "", "auto") else None))[0]


def mark_recommended(combos: list[Combo], deadline_s: float) -> Combo | None:
    fit = [c for c in combos if c.fits]
    if not fit:
        return None
    best = max(fit, key=lambda c: (_params(c.student), _params(c.teacher) if c.teacher else 0.0,
                                   1 if c.judge else 0, -c.seconds))
    best.recommended = True
    return best


def closest(combos: list[Combo]) -> Combo | None:
    """When nothing fits: the shortest estimate, which is the one least over the deadline."""
    known = [c for c in combos if c.seconds is not None]
    return min(known, key=lambda c: c.seconds) if known else None


def render(combos: list[Combo], deadline_s: float, label: str, limit: int = 12) -> list[str]:
    fit = [c for c in combos if c.fits]
    unk = [c for c in combos if c.fits is None]
    lines = [f"deadline: {label} = {est.fmt_dur(deadline_s)}. {len(fit)} of {len(combos)} combinations on "
             f"this machine are estimated to finish in time (nothing is downloaded or run to find out)."]
    if not fit:
        c = closest(combos)
        lines.append("nothing is estimated to fit. " + (
            f"The closest: {c.label()}, est. {est.fmt_dur(c.seconds)} [{c.basis}], "
            f"{est.fmt_dur(c.seconds - deadline_s)} over." if c else "no combination has an estimate."))
        if c and c.downloads:
            lines.append("   it would first download " + _dl(c))
        lines.append("   nothing was lowered to meet the deadline; give more time, or choose smaller models "
                     "yourself (--student, --teacher, --judge).")
        return lines
    order = sorted(fit, key=lambda c: (not c.recommended, -_params(c.student),
                                       -(_params(c.teacher) if c.teacher else 0), c.seconds))
    for c in order[:limit]:
        mark = "recommended" if c.recommended else ("configured" if c.configured else "")
        lines.append(f"   {'*' if c.recommended else ' '} {c.label()}: est. {est.fmt_dur(c.seconds)} "
                     f"[{c.basis}]" + (f" ({mark})" if mark else "") + (f"; downloads {_dl(c)}" if c.downloads else ""))
    if len(order) > limit:
        lines.append(f"   ... and {len(order) - limit} more that fit (the {limit} listed rank first)")
    lines.append("   * recommended: among those that fit, the largest student, then the largest teacher, "
                 "then a judge, then the shorter estimate")
    lines.append("   measured: every stage has a measurement on this machine; assumed: at least one stage uses the "
                 "cost model's stated rate; download time is not counted")
    cfgd = [c for c in combos if c.configured]
    if cfgd:
        c = cfgd[0]
        lines.append(f"   this project as configured: {c.label()}, est. {est.fmt_dur(c.seconds)} [{c.basis}], "
                     + ("fits" if c.fits else "does not fit" if c.fits is False else "has no complete estimate"))
    if unk:
        lines.append(f"   {len(unk)} combinations have a stage with no estimate on this machine yet and are not "
                     f"counted either way")
    return lines


def _dl(c: Combo) -> str:
    return ", ".join(f"{t} {b / 1e9:.1f} GB" for t, b in c.downloads)


def check_configured(project: Path, deadline: str, engine: str | None = None,
                     roles: dict | None = None) -> tuple[bool, list[str]]:
    """For `spill build --deadline`: (the configured combination fits, the lines to print)."""
    secs, label = parse_deadline(deadline)
    combos = evaluate(project, secs, engine, roles)
    cfgd = next((c for c in combos if c.configured), None)
    lines = render(combos, secs, label)
    if cfgd is not None and cfgd.fits:
        return True, [f"deadline {label}: this project as configured ({cfgd.label()}) is estimated at "
                      f"{est.fmt_dur(cfgd.seconds)} [{cfgd.basis}], which fits"]
    if cfgd is not None and cfgd.fits is None:
        return True, [f"deadline {label}: this project as configured has no complete estimate on this machine "
                      f"yet (a stage has neither a measurement nor a rate); building anyway, and it will say "
                      f"how long each stage takes as it goes"]
    return False, lines
