"""spill build <folder>: the whole loop from a folder of files.

The files present decide the path:
  evals.jsonl    required: questions with the right answers (the exam; never trained on)
  train.jsonl    your answers   (the homework you wrote)
  prompts.jsonl  questions only (the homework a big model answers for you)
  instructions.txt  optional: the system prompt for models that were not trained
                    (the base, the teacher, --compare); the tuned student learns the
                    task from the data and is trained and evaluated without it
  spill.json        optional defaults for this folder (student, teacher, epochs); flags win

  train only    eval student base -> tune on train.jsonl -> eval student+adapter
  prompts only  distill the teacher on prompts.jsonl -> tune on its answers ->
                eval student base, student+adapter, teacher
  both          distill -> tune on your answers (weighted --weight-own : 1) plus the
                teacher's -> the same three evals
  --base M      train the adapter on M itself (a big model) instead of the student
  --compare M   add M's score to the table

State lives in <folder>/.build/state.json, so `spill resume <folder>` (or running the
same build again) continues at the first unfinished stage. A build is a plan of stages;
what a stage does is the Backend's job (the real one calls the same code as
`spill distill`, `spill tune` and `spill eval`; tests use a fake).
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import estimate as est
from . import formats
from .errors import SpillError

DEFAULT_STUDENT = "qwen2.5:7b"
DEFAULT_TEACHER = "llama3.3:70b"
DEFAULT_EPOCHS = 2.0
SHORT_EXPECTED_CHARS = 64          # "classification-style": every expected value a short label
SHORT_EXPECTED_WORDS = 8
CLASSIFICATION_MAX_TOKENS = 16
DEFAULT_MAX_TOKENS = 128
INSTRUCTIONS = "instructions.txt"
SETTINGS = "spill.json"            # optional per-folder defaults: {"student": "qwen2.5:0.5b"}
STATE_DIR = ".build"


# ------------------------------------------------------------ folder and plan

@dataclass
class Folder:
    path: Path
    name: str
    evals: Path
    train: Path | None
    prompts: Path | None
    instructions: str | None
    settings: dict = field(default_factory=dict)    # optional spill.json: student, teacher, ...

    @property
    def state_dir(self) -> Path:
        return self.path / STATE_DIR


def read_folder(path: str | Path) -> Folder:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise SpillError(f"{path} is not a folder", "spill example banking77 --quick")
    ev = p / "evals.jsonl"
    if not ev.exists():
        raise SpillError(f"{p.name}/evals.jsonl is missing; build needs your exam (questions "
                         f"with the right answers)", "spill example banking77 --quick")
    train = p / "train.jsonl"
    prompts = p / "prompts.jsonl"
    ins = p / INSTRUCTIONS
    settings = {}
    if (p / SETTINGS).exists():
        try:
            settings = json.loads((p / SETTINGS).read_text())
        except json.JSONDecodeError as e:
            raise SpillError(f"{p.name}/{SETTINGS} is not valid JSON ({e.msg} at line "
                             f"{e.lineno})")
    return Folder(p, p.name, ev, train if train.exists() else None,
                  prompts if prompts.exists() else None,
                  ins.read_text().strip() if ins.exists() else None, settings)


@dataclass
class Stage:
    id: str                 # e.g. "distill", "tune", "eval:student-base"
    kind: str               # distill | tune | eval
    label: str
    model: str | None = None
    est_s: float = 0.0
    est_detail: str = ""
    status: str = "pending"          # pending | running | done | interrupted
    actual_s: float | None = None
    result: dict = field(default_factory=dict)


@dataclass
class Plan:
    folder: Folder
    path_kind: str                   # train | prompts | both
    student: str
    teacher: str
    base: str | None
    compare: list[str]
    weight_own: float
    adapter_name: str
    max_tokens: int
    classification: bool
    stages: list[Stage]
    tflops: float
    tflops_src: str
    total_s: float
    epochs: float = DEFAULT_EPOCHS
    extra: dict = field(default_factory=dict)

    @property
    def tuned_model(self) -> str:
        return self.base or self.student


def path_kind(f: Folder) -> str:
    if f.train and f.prompts:
        return "both"
    if f.train:
        return "train"
    if f.prompts:
        return "prompts"
    raise SpillError(f"{f.name} has an evals.jsonl but nothing to learn from; add train.jsonl "
                     f"(your answers) or prompts.jsonl (questions for the big model to "
                     f"answer)", f"spill example banking77 --quick")


def _rows(path: Path):
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            out.append(formats.parse_line(line, n))
    return out


def _msgs(obj: dict) -> list[dict]:
    return obj["body"]["messages"] if "body" in obj else obj["messages"]


def is_classification(evals_path: Path) -> bool:
    """Short expected values on every row (labels, not prose)."""
    rows = _rows(evals_path)
    return bool(rows) and all(
        isinstance(r.get("expected"), (str, int, float, bool))
        and len(str(r["expected"])) <= SHORT_EXPECTED_CHARS
        and len(str(r["expected"]).split()) <= SHORT_EXPECTED_WORDS for r in rows)


def classification_max_tokens(evals_path: Path) -> int:
    """16, unless the longest label would not fit (about 4 characters per token)."""
    longest = max((len(str(r.get("expected", ""))) for r in _rows(evals_path)), default=0)
    return max(CLASSIFICATION_MAX_TOKENS, math.ceil(longest / 4) + 4)


def untrained_system(folder: Folder) -> str | None:
    return folder.instructions


# ------------------------------------------------------------ file variants per role

def _write_jsonl(path: Path, objs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objs))
    return path


def with_system(src: Path, system: str | None, dest: Path, max_tokens: int | None = None,
                drop_answer: bool = False) -> Path:
    """Rewrite a file as plain-ish rows with `system` prepended to every row's system
    prompt (or set, if the row has none). Output is chat-shaped JSONL, which every
    command accepts. `expected` is kept; `max_tokens` is set on each row if given."""
    out = []
    for obj in _rows(src):
        msgs = [dict(m) for m in _msgs(obj)]
        if drop_answer and msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]
        if system:
            if msgs[0]["role"] == "system":
                msgs[0]["content"] = system + "\n\n" + msgs[0]["content"]
            else:
                msgs.insert(0, {"role": "system", "content": system})
        row = {"messages": msgs}
        if "expected" in obj:
            row["expected"] = obj["expected"]
        if obj.get("custom_id"):
            row["custom_id"] = obj["custom_id"]
        if max_tokens:
            row["max_tokens"] = max_tokens
        out.append(row)
    return _write_jsonl(dest, out)


def strip_system(src: Path, dest: Path, max_tokens: int | None = None) -> Path:
    """The student's view: rows exactly as the user wrote them."""
    return with_system(src, None, dest, max_tokens)


def combine_training(own: Path | None, distilled: Path | None, weight_own: float,
                     system: str | None, dest: Path) -> tuple[Path, dict]:
    """Chat training file: your answers repeated weight_own times (rounded, at least 1)
    plus the teacher's answers. Prompts are the student's view (no instructions): the
    student is trained to give the answer without being told the label list."""
    rows, n_own, n_teacher, reps_used, skipped = [], 0, 0, 0, 0
    if own is not None:
        reps = max(1, round(weight_own)) if distilled is not None else 1
        reps_used = reps
        base = []
        for obj in _rows(own):
            msgs = _msgs(obj)
            if msgs[-1]["role"] != "assistant":
                skipped += 1
                continue
            base.append({"messages": msgs})
        n_own = len(base)
        rows += base * reps
    if distilled is not None:
        for line in distilled.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            comp = rec.get("completion")
            if rec.get("error") or comp is None or not str(comp).strip():
                continue
            msgs = [m for m in rec["messages"]]
            if system and msgs and msgs[0]["role"] == "system" \
                    and msgs[0]["content"].startswith(system):
                rest = msgs[0]["content"][len(system):].lstrip("\n")
                msgs = ([{"role": "system", "content": rest}] if rest else []) + msgs[1:]
            rows.append({"messages": msgs + [{"role": "assistant", "content": comp.strip()}]})
            n_teacher += 1
    if not rows:
        raise SpillError("no training rows after combining your answers and the teacher's",
                         "spill check train.jsonl")
    _write_jsonl(dest, rows)
    return dest, {"own": n_own, "own_reps": reps_used, "own_skipped": skipped, "teacher": n_teacher, "total": len(rows)}


# ------------------------------------------------------------ estimates

def _tok(chars: float) -> float:
    return chars / 4


def workload(path: Path, system: str | None, max_tokens: int, classification: bool
             ) -> est.Workload:
    objs = _rows(path)
    texts, suffix_chars = [], []
    for obj in objs:
        msgs = _msgs(obj)
        if msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]
        full = "\n".join(m["content"] for m in msgs)
        texts.append((system + "\n\n" if system else "") + full)
        suffix_chars.append(len(full))
    pre = est.common_prefix_chars(texts) if len(texts) > 1 else 0
    pre = max(pre, len(system or ""))
    pre_tok = _tok(pre) + 20             # chat template header
    suf = _tok(sum(len(t) for t in texts) / max(1, len(texts)) - pre) + 8
    exp = [len(str(o.get("expected", ""))) for o in objs]
    out_tok = min(max_tokens, (_tok(sum(exp) / max(1, len(exp))) + 3) if any(exp) else max_tokens)
    return est.Workload(rows=len(objs), prefix_tokens=pre_tok, suffix_tokens=max(suf, 1),
                        out_tokens=max(1.0, out_tok), max_tokens=max_tokens)


def train_tokens(path: Path, reps: float = 1.0) -> float:
    n = 0.0
    for obj in _rows(path):
        n += _tok(sum(len(m["content"]) for m in _msgs(obj))) + 24
    return n * reps


def make_plan(folder: Folder, student: str, teacher: str, base: str | None,
              compare: list[str], weight_own: float, cal: dict, working_set: int,
              epochs: float = DEFAULT_EPOCHS) -> Plan:
    kind = path_kind(folder)
    classification = is_classification(folder.evals)
    max_tokens = classification_max_tokens(folder.evals) if classification else DEFAULT_MAX_TOKENS
    tf, tf_src = est.achieved_tflops(cal)
    tuned = base or student
    ins = folder.instructions
    ev_wl_untrained = workload(folder.evals, ins, max_tokens, classification)
    ev_wl_student = workload(folder.evals, None, max_tokens, classification)
    stages: list[Stage] = []

    def eval_stage(sid, label, model, wl, adapter=False):
        e = est.eval_seconds(model, wl, cal, working_set, tf, reuse=True, adapter=adapter)
        stages.append(Stage(sid, "eval", label, model, e.seconds, e.detail))

    if kind in ("prompts", "both"):
        wl = workload(folder.prompts, ins, max_tokens, classification)
        e = est.eval_seconds(teacher, wl, cal, working_set, tf, reuse=True)
        stages.append(Stage("distill", "distill", f"teacher {teacher} answers prompts.jsonl",
                            teacher, e.seconds, e.detail))
    elif kind == "train":
        eval_stage("eval:base", f"eval {tuned} untrained", tuned, ev_wl_untrained)

    own_reps = 1 if kind == "train" else max(1, round(weight_own))
    tune_tokens = 0.0
    if kind in ("train", "both"):
        tune_tokens += train_tokens(folder.train, own_reps if kind == "both" else 1)
    if kind in ("prompts", "both"):
        n_p = len(_rows(folder.prompts))
        wl = workload(folder.prompts, None, max_tokens, classification)
        tune_tokens += n_p * (wl.prefix_tokens + wl.suffix_tokens + wl.out_tokens + 12)
    t = est.tune_seconds(tuned, tune_tokens, epochs, tf, cal, working_set)
    stages.append(Stage("tune", "tune", f"tune {tuned} -> {tuned}+{folder.name}", tuned,
                        t.seconds, t.detail))

    if kind != "train":
        eval_stage("eval:base", f"eval {tuned} untrained", tuned, ev_wl_untrained)
    eval_stage("eval:tuned", f"eval {tuned}+{folder.name}", tuned, ev_wl_student, adapter=True)
    if kind != "train":
        if teacher != tuned:
            eval_stage("eval:teacher", f"eval {teacher} (teacher)", teacher, ev_wl_untrained)
    for c in compare:
        if c not in (tuned, teacher if kind != "train" else None):
            eval_stage(f"eval:compare:{c}", f"eval {c} (compare)", c, ev_wl_untrained)

    # cheap stages first on the train-only path is already the order; for distill paths the
    # teacher stage is the long pole and must come first (the student is trained on it)
    total = sum(s.est_s for s in stages)
    return Plan(folder, kind, student, teacher, base, compare, weight_own, folder.name,
                max_tokens, classification, stages, tf, tf_src, total, epochs)


def pre_run_line(plan: Plan, on_battery: bool = False) -> str:
    kinds = {"train": "train only (your answers)",
             "prompts": "prompts only (the teacher's answers)",
             "both": f"both (your answers weighted {plan.weight_own:g}:1 with the "
                     f"teacher's)"}
    models = [plan.tuned_model]
    if plan.path_kind != "train" and plan.teacher not in models:
        models.append(plan.teacher + " (teacher)")
    models += [c for c in plan.compare if c not in models]
    mt = (f" Classification-style evals: max_tokens {plan.max_tokens}." if plan.classification
          else f" max_tokens {plan.max_tokens}.")
    stages = "; ".join(f"{i + 1} {s.label} {est.fmt_dur(s.est_s)}"
                       for i, s in enumerate(plan.stages))
    line = (f"spill build {plan.folder.name}: path {kinds[plan.path_kind]}. Models: "
            f"{', '.join(models)}.{mt} Stages: {stages} (at {plan.tflops:g} TFLOP/s, "
            f"{plan.tflops_src}; decode disk-bound, prefill 2 x params x tokens, training "
            f"6 x params x tokens). Est. {est.fmt_dur(plan.total_s)}. Cost: $0. "
            f"Adapter -> {plan.adapter_name} (spill adapters), files in "
            f"{plan.folder.path.name}/.build/")
    if on_battery:
        line += " On battery: plug in before an overnight run."
    return line


# ------------------------------------------------------------ state

def state_path(folder: Folder) -> Path:
    return folder.state_dir / "state.json"


def inputs_fingerprint(plan: Plan) -> str:
    """Everything a finished stage depends on: the files and the options that change output."""
    import hashlib
    f = plan.folder
    h = hashlib.sha256()
    for p in (f.evals, f.train, f.prompts):
        h.update(p.read_bytes() if p else b"-")
    h.update((f.instructions or "").encode())
    h.update(json.dumps([plan.weight_own, plan.epochs, plan.max_tokens]).encode())
    return h.hexdigest()


def save_state(plan: Plan, extra: dict | None = None) -> None:
    folder = plan.folder
    folder.state_dir.mkdir(parents=True, exist_ok=True)
    d = {"name": plan.adapter_name, "path": plan.path_kind, "student": plan.student,
         "teacher": plan.teacher, "base": plan.base, "compare": plan.compare,
         "weight_own": plan.weight_own, "max_tokens": plan.max_tokens,
         "classification": plan.classification, "tflops": plan.tflops,
         "tflops_source": plan.tflops_src, "total_est_s": plan.total_s,
         "epochs": plan.epochs, "inputs": inputs_fingerprint(plan), "stages": [s.__dict__ for s in plan.stages],
         "updated": time.strftime("%FT%T%z")}
    d.update(plan.extra)
    d.update(extra or {})
    tmp = state_path(folder).with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(state_path(folder))


def load_state(folder: Folder) -> dict | None:
    p = state_path(folder)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def apply_state(plan: Plan, state: dict) -> bool:
    """Carry finished stages over from a previous attempt (same plan only)."""
    same = (state.get("path") == plan.path_kind and state.get("student") == plan.student
            and state.get("teacher") == plan.teacher and state.get("base") == plan.base
            and state.get("compare") == plan.compare
            and state.get("inputs") == inputs_fingerprint(plan)
            and [s["id"] for s in state.get("stages", [])] == [s.id for s in plan.stages])
    if not same:
        return False
    for s, old in zip(plan.stages, state["stages"]):
        s.status = "done" if old["status"] == "done" else (
            "pending" if old["status"] == "running" else old["status"])
        s.actual_s, s.result = old.get("actual_s"), old.get("result", {})
    return True


# ------------------------------------------------------------ backends

class Backend:
    """What a stage does. The real backend (RealBackend) runs the engines; tests
    substitute a fake."""

    def distill(self, teacher: str, prompts: Path, out: Path, max_tokens: int,
                resume_job: str | None) -> dict:
        raise NotImplementedError

    def tune(self, model: str, train: Path, name: str, resume_job: str | None) -> dict:
        raise NotImplementedError

    def eval(self, model: str, eval_file: Path, resume_job: str | None = None) -> dict:
        """-> {"score": float, "rows": int, "run_id": str, "results_path": str}"""
        raise NotImplementedError


@dataclass
class BuildResult:
    table: list[dict]
    stages: list[Stage]
    adapter: str
    interrupted: bool
    tuned_label: str
    wall_s: float


def run_plan(plan: Plan, backend: Backend, say=print, stop_after: str | None = None
             ) -> BuildResult:
    f = plan.folder
    sd = f.state_dir
    sd.mkdir(parents=True, exist_ok=True)
    ins = untrained_system(f)
    mt = plan.max_tokens
    ev_untrained = with_system(f.evals, ins, sd / "evals.untrained.jsonl", mt)
    ev_student = strip_system(f.evals, sd / "evals.student.jsonl", mt)
    table: list[dict] = []
    t_start = time.monotonic()
    interrupted = False
    distill_out = f.path / "prompts.distill.jsonl"
    train_file = sd / "train.build.jsonl"
    tuned = plan.tuned_model
    label = f"{tuned}+{plan.adapter_name}"
    roles = {"eval:base": (tuned, "base (untrained)", ev_untrained),
             "eval:tuned": (label, "your model", ev_student),
             "eval:teacher": (plan.teacher, "teacher", ev_untrained)}
    for c in plan.compare:
        roles[f"eval:compare:{c}"] = (c, "compare", ev_untrained)

    for i, st in enumerate(plan.stages):
        if st.status == "done":
            if st.kind == "eval" and st.result:
                table.append(_table_row(st, roles))
            continue
        say(f"stage {i + 1}/{len(plan.stages)}: {st.label}; estimate {est.fmt_dur(st.est_s)} "
            f"({st.est_detail})")
        st.status = "running"
        save_state(plan)
        t0 = time.monotonic()
        try:
            if st.kind == "distill":
                prompts_file = with_system(f.prompts, ins, sd / "prompts.teacher.jsonl", mt)
                res = backend.distill(plan.teacher, prompts_file, distill_out, mt,
                                      st.result.get("job_id"))
            elif st.kind == "tune":
                own = f.train if plan.path_kind in ("train", "both") else None
                dist = distill_out if plan.path_kind in ("prompts", "both") else None
                _, counts = combine_training(own, dist, plan.weight_own, ins, train_file)
                st.result["training_rows"] = counts
                if counts.get("own_skipped"):
                    say(f"warning: {counts['own_skipped']} rows of train.jsonl have no answer "
                        f"and were left out")
                res = backend.tune(tuned, train_file, plan.adapter_name,
                                   st.result.get("job_id"))
            else:
                model, _role, file = roles[st.id]
                res = backend.eval(model, file, st.result.get("job_id"))
                _write_out(f, model, res)
        except KeyboardInterrupt:
            res = {"interrupted": True}
        st.actual_s = round(time.monotonic() - t0, 1)
        if res.get("interrupted"):
            st.status = "interrupted"
            st.result.update({k: v for k, v in res.items() if k != "interrupted"})
            save_state(plan)
            interrupted = True
            say(f"stage {i + 1} stopped after {est.fmt_dur(st.actual_s)}; nothing is lost")
            break
        st.status = "done"
        st.result.update(res)
        save_state(plan)
        say(f"stage {i + 1} done in {est.fmt_dur(st.actual_s)} "
            f"(estimate {est.fmt_dur(st.est_s)})"
            + (f"; score {res['score']:.3f} on {res['rows']} rows" if st.kind == "eval" else ""))
        if st.kind == "eval":
            table.append(_table_row(st, roles))
        if stop_after and st.id == stop_after:
            interrupted = True
            break
    order = {"your model": 0, "base (untrained)": 1, "teacher": 2, "compare": 3}
    table.sort(key=lambda r: order.get(r["role"], 9))
    save_state(plan, {"finished": not interrupted})
    return BuildResult(table, plan.stages, plan.adapter_name, interrupted, label,
                       time.monotonic() - t_start)


def _table_row(st: Stage, roles: dict) -> dict:
    model, role, _ = roles[st.id]
    return {"model": model, "role": role, "score": st.result["score"],
            "rows": st.result["rows"], "run_id": st.result.get("run_id"),
            "stage": st.id}


def _write_out(f: Folder, model: str, res: dict) -> None:
    """Per-model results copied into the folder as <model>.out.jsonl, manifests beside them."""
    src = res.get("results_path")
    if not src or not Path(src).exists():
        return
    safe = model.replace(":", "-").replace("+", "_plus_").replace("/", "_")
    shutil.copyfile(src, f.path / f"{safe}.out.jsonl")
    man = res.get("manifest")
    if man and Path(man).exists():
        d = f.state_dir / "runs"
        d.mkdir(exist_ok=True)
        shutil.copyfile(man, d / f"{safe}.manifest.json")


def render_table(rows: list[dict]) -> str:
    head = ["model", "role", "score", "rows"]
    body = [[r["model"], r["role"], f"{r['score']:.3f}", str(r["rows"])] for r in rows]
    w = [max(len(x[i]) for x in [head] + body) for i in range(4)]
    fmt = lambda r: "  ".join(c.ljust(w[i]) for i, c in enumerate(r))
    return "\n".join([fmt(head)] + [fmt(b) for b in body])


def render_stages(stages: list[Stage]) -> str:
    """Per-stage estimate versus actual, and the totals."""
    head = ["stage", "estimate", "actual", "ratio"]
    body = []
    for i, s in enumerate(stages, 1):
        ratio = (f"{s.actual_s / s.est_s:.2f}x" if s.actual_s and s.est_s else "-")
        body.append([f"{i} {s.label}", est.fmt_dur(s.est_s),
                     est.fmt_dur(s.actual_s) if s.actual_s is not None else "-", ratio])
    est_t = sum(s.est_s for s in stages)
    act_t = sum(s.actual_s or 0 for s in stages)
    body.append(["total", est.fmt_dur(est_t), est.fmt_dur(act_t),
                 f"{act_t / est_t:.2f}x" if est_t and act_t else "-"])
    w = [max(len(r[i]) for r in [head] + body) for i in range(4)]
    fmt = lambda r: "  ".join(c.ljust(w[i]) for i, c in enumerate(r))
    return "\n".join([fmt(head)] + [fmt(b) for b in body])


def final_lines(res: BuildResult) -> list[str]:
    return [f"your model: {res.tuned_label}"]
