"""The coordinator owns a build: it chooses the run, holds the run's ownership, hands each
stage to an executor, verifies what comes back, and publishes it through the fenced control
object. Executors only return staged outputs and a status; they never see the control object,
so they cannot bypass fencing.

  LocalExecutor       runs the stage in this process (the default)
  SubprocessExecutor  runs the same stage in a separate process from its serialized
                      description (the test backend); no mid-stage checkpoints, a stopped
                      stage starts over
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import SpillError
from . import config as C
from . import control as ctl
from . import modelid, plan as P, report as R, timings
from .common import canon, new_id, read_json, read_jsonl, sha_file, sha_obj, write_json
from .runstate import RunState
from .stagefns import StageContext, StageDesc, StageOutcome, run_stage, slug

STATE_DIR = ".spill"


class LocalExecutor:
    name = "local"

    def run(self, desc: StageDesc, ctx: StageContext) -> StageOutcome:
        return run_stage(desc, ctx)


class SubprocessExecutor:
    name = "subprocess"

    def run(self, desc: StageDesc, ctx: StageContext) -> StageOutcome:
        ctx.stage_dir.mkdir(parents=True, exist_ok=True)
        spec = {"desc": desc.to_dict(), "attempt_dir": str(ctx.attempt_dir),
                "stage_dir": str(ctx.stage_dir), "engine": ctx.engine, "stop_after": ctx.stop_after}
        f = ctx.stage_dir / "stage.json"
        f.write_text(json.dumps(spec, indent=1))
        out = ctx.stage_dir / "outcome.json"
        out.unlink(missing_ok=True)
        r = subprocess.run([sys.executable, "-m", "streamweights.project.worker", str(f)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
            raise SpillError(f"stage {desc.id} failed in its worker process: {' | '.join(tail)[:300]}",
                             "spill build <project> --debug")
        d = json.loads(out.read_text())
        return StageOutcome(d["status"], d["metrics"], d["producers"], d["seconds"], d["notes"])


EXECUTORS = {"local": LocalExecutor, "subprocess": SubprocessExecutor}


@dataclass
class BuildOutcome:
    run_id: str
    status: str                  # completed | stopped | already-complete
    snapshot: Path | None = None
    stages: dict = field(default_factory=dict)
    table: list = field(default_factory=list)
    resumed: bool = False


def state_root(project: Path) -> Path:
    return Path(project) / STATE_DIR


def list_runs(project: Path) -> list[dict]:
    """Control documents of every run whose authority is local to this project."""
    return list_runs_at(str(state_root(project)))


def list_runs_at(state_root_uri: str) -> list[dict]:
    """Control documents of the runs under a state location (a local .spill or s3://.../.spill)."""
    from ..portable.store import Store, is_uri
    from . import caps
    out = []
    if not is_uri(state_root_uri) or state_root_uri.startswith("file://"):
        base = Path(state_root_uri[7:] if state_root_uri.startswith("file://") else state_root_uri) / "runs"
        if base.exists():
            for d in sorted(base.iterdir()):
                f = d / "control.json"
                if f.exists():
                    try:
                        out.append(json.loads(f.read_text()))
                    except json.JSONDecodeError:
                        continue
        return out
    from .runstate import run_uri
    for name in Store(state_root_uri).ls("runs"):
        doc, _ = caps.open_backend(run_uri(state_root_uri, name), probe=False).read()
        if doc:
            out.append(doc)
    return out


def identity_diff(old: dict, new: dict, prefix="") -> list[str]:
    """Named differences between two identity field dicts."""
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            out += identity_diff(a, b, f"{prefix}{k}.")
        else:
            out.append(f"{prefix}{k}")
    return out


def _inputs_dir(plan: P.BuildPlan, dest: Path) -> dict:
    """Materialize the frozen inputs of a run: resolved config, train and validation rows,
    the contract files, and the identity. Test rows are not included (only their hash)."""
    p = plan.project
    dest.mkdir(parents=True, exist_ok=True)
    cfg = plan.cfg
    (dest / "streamweights.toml").write_bytes(C.dumps(cfg))
    for n in ("train", "val"):
        shutil.copyfile(p / "data" / f"{n}.jsonl", dest / f"{n}.jsonl")
    if cfg["task"]["type"] == "json":
        shutil.copyfile(p / cfg["contract"]["schema_file"], dest / cfg["contract"]["schema_file"])
    if (p / "system.txt").exists():
        shutil.copyfile(p / "system.txt", dest / "system.txt")
    ident = {"identity": plan.identity, "fields": plan.identity_fields,
             "split_sha256": plan.split_sha}
    write_json(dest / "identity.json", ident)
    return ident


def _public_models(plan: P.BuildPlan) -> dict:
    out = {}
    for role, i in plan.models.items():
        if i:
            out[role] = {k: v for k, v in i.items() if k not in ("dir", "present")}
    return out


class Coordinator:
    def __init__(self, project: Path, executor="local", engine: str | None = None,
                 say=print, stop_after: str | None = None, state_root_uri: str | None = None,
                 remote_project: str | None = None):
        self.remote_project = remote_project
        self.project = Path(project).resolve()
        self.executor = EXECUTORS[executor]() if isinstance(executor, str) else executor
        self.engine = engine
        self.say = say
        self.stop_after = stop_after
        self.state_root = state_root_uri or str(state_root(self.project))
        self.work_root = state_root(self.project)

    # ---- choosing the run
    def _find_resumable(self, plan: P.BuildPlan) -> dict | None:
        for doc in list_runs(self.project):
            if doc["identity"] == plan.identity and doc["status"] in (ctl.IDLE, ctl.RUNNING,
                                                                       ctl.HANDOFF):
                return doc
        return None

    def _find_complete(self, plan: P.BuildPlan) -> dict | None:
        for doc in list_runs(self.project):
            if doc["identity"] == plan.identity and doc["status"] == ctl.COMPLETED:
                return doc
        return None

    def build(self, plan: P.BuildPlan, new_run: bool = False, parent: str | None = None,
              run_id: str | None = None) -> BuildOutcome:
        doc = None if new_run else self._find_resumable(plan)
        if doc is None and not new_run:
            done = self._find_complete(plan)
            if done:
                self.say(f"run {done['run_id']} already completed these exact inputs; "
                         f"report: runs/{done['run_id']}/report.md (spill build --new-run builds "
                         f"again as a new run)")
                return BuildOutcome(done["run_id"], "already-complete",
                                    self.project / "runs" / done["run_id"])
        stale = [d for d in list_runs(self.project)
                 if d["status"] in (ctl.IDLE, ctl.RUNNING) and doc is None and d["identity"] != plan.identity]
        for d in stale:
            old = self._run_fields(d["run_id"])
            diff = identity_diff(old, plan.identity_fields) if old else ["inputs"]
            self.say(f"note: run {d['run_id']} is unfinished and was started with different "
                     f"{', '.join(diff[:6])}; this is a new run (`spill resume {self.project.name}` "
                     f"continues the old one with its own inputs)")
        if doc is not None:
            rs = RunState.open(self.state_root, doc["run_id"], self.work_root)
            resumed = True
            self.say(f"continuing run {doc['run_id']} (same inputs, same settings)")
        else:
            rid = run_id or (f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}-"
                             f"{plan.identity[:6]}")
            if parent is None:
                prior = [d for d in list_runs_at(self.state_root)
                         if d["status"] in (ctl.COMPLETED, ctl.BUNDLED)]
                parent = sorted(prior, key=lambda d: d["run_id"])[-1]["run_id"] if prior else None
            rs = RunState.create(self.state_root, rid, plan.identity, self.work_root, parent=parent)
            resumed = False
        return self._execute(rs, plan, resumed, rs.doc().get("parent"))

    def resume(self, run_id: str, engine_override: str | None = None) -> BuildOutcome:
        """Continue an unfinished run with the inputs it was started with (its frozen
        snapshot), never with the live project files. The receiving side verifies model and
        tokenizer identity first and names every difference."""
        rs = RunState.open(self.state_root, run_id, self.work_root)
        doc = rs.doc()
        if doc["status"] == ctl.COMPLETED:
            raise SpillError(f"run {run_id} is completed; a completed run cannot resume training",
                             f"spill build {self.project} --new-run")
        if doc["status"] == ctl.INCOMING:
            raise SpillError(f"run {run_id} here is the destination of a handoff that is not "
                             f"committed yet", "re-run `spill move` at the source to finish it")
        if doc["status"] == ctl.TRANSFERRED:
            raise SpillError(f"run {run_id} was handed to {doc['handoff']['dest']}",
                             f"spill resume {doc['handoff']['dest']}")
        frozen = self.work_root / "frozen" / run_id
        shutil.rmtree(frozen, ignore_errors=True)
        if rs.accepted_stage("inputs") is None:
            raise SpillError(f"run {run_id} never froze its inputs, so there is nothing to continue; "
                             f"it holds no committed work", f"spill build {self.project} --new-run")
        rs.fetch_stage("inputs", frozen)
        (frozen / "result.json").unlink(missing_ok=True)
        plan = P.make_plan(self.project, engine_override, fetch=True, frozen=frozen)
        if plan.identity != doc["identity"]:
            old = read_json(frozen / "identity.json")["fields"]
            diff = identity_diff(old, plan.identity_fields)
            raise SpillError(f"this machine cannot continue run {run_id}: it differs in "
                             f"{', '.join(diff) or 'identity'} (models are compared by file "
                             f"hash, not by name)", "fetch the pinned model revision named in "
                             "the run (spill plan shows it), then resume")
        self.engine = self.engine or plan.engine
        self.say(f"continuing run {run_id} on {plan.engine} (its stages accepted so far are kept)")
        return self._execute(rs, plan, True, doc.get("parent"))

    def _run_fields(self, run_id: str) -> dict | None:
        rs = RunState.open(self.state_root, run_id, self.work_root)
        ref = rs.accepted_stage("inputs")
        if not ref:
            return None
        d = self.work_root / "peek" / run_id
        shutil.rmtree(d, ignore_errors=True)
        rs.fetch_stage("inputs", d)
        try:
            return read_json(d / "identity.json")["fields"]
        except (OSError, KeyError):
            return None

    # ---- the attempt
    def _execute(self, rs: RunState, plan: P.BuildPlan, resumed: bool, parent) -> BuildOutcome:
        from .. import runtime
        t_start = time.monotonic()
        try:
            rs.acquire()
        except ctl.NotAllowed as e:
            raise SpillError(e.message, e.recovery)
        att = rs.work
        say = self.say
        try:
            inputs = att / "inputs"
            if rs.accepted_stage("inputs") is None:
                ident = _inputs_dir(plan, att / "inputs-staging")
                rs.publish_stage("inputs", att / "inputs-staging", {"status": "done", "kind": "inputs"})
                shutil.rmtree(att / "inputs-staging")
            rs.fetch_stage("inputs", inputs)
            (inputs / "result.json").unlink(missing_ok=True)
            self._verify_inputs(inputs, plan)
            results: dict = {}
            accepted_dirs: dict = {}
            for i, st in enumerate(plan.stages, 1):
                ref = rs.accepted_stage(st.id)
                if ref is not None:
                    d = att / "accepted" / slug(st.id)
                    results[st.id] = rs.fetch_stage(st.id, d)
                    (d / "result.json").unlink(missing_ok=True)
                    accepted_dirs[st.id] = d
                    say(f"stage {i}/{len(plan.stages)}: {st.label}: accepted earlier "
                        f"({results[st.id].get('seconds', 0):.1f} s)")
                    continue
                if runtime.ENV.stop.is_set() or rs.quiesce_requested():
                    return self._stopped(rs, plan, results, "told to stop")
                out = self._run_stage(rs, plan, st, i, results, accepted_dirs)
                if out is None:
                    return self._stopped(rs, plan, results, "stopped inside a stage")
                results[st.id] = out
                accepted_dirs[st.id] = att / "accepted" / slug(st.id)
            return self._finish(rs, plan, results, accepted_dirs, resumed, parent, t_start)
        except ctl.Superseded as e:
            raise SpillError(f"this run was taken over or finished elsewhere: {e.message}",
                             e.recovery or "spill resume <project>")
        finally:
            rs.release()

    def _verify_inputs(self, inputs: Path, plan: P.BuildPlan) -> None:
        want = {"train.jsonl": plan.split_sha["train"], "val.jsonl": plan.split_sha["val"]}
        ident = read_json(inputs / "identity.json")
        if ident["identity"] != plan.identity:
            raise SpillError("the run's frozen inputs do not match this project's identity",
                             "spill build <project> --new-run")
        for n, sha in want.items():
            if sha_file(inputs / n) != sha:
                raise SpillError(f"the frozen {n} of this run differs from its recorded hash",
                                 "spill build <project> --new-run")

    def _run_stage(self, rs: RunState, plan: P.BuildPlan, st: StageDesc, i: int, results: dict,
                   accepted_dirs: dict):
        from .. import runtime
        att = rs.work
        n = len(plan.stages)
        e = plan.estimates.get(st.id)
        est_txt = "" if e is None or e.seconds is None else f"; estimate {_dur(e.seconds)} ({e.basis})"
        self.say(f"stage {i}/{n}: {st.label}{est_txt}")
        for dep in st.depends:
            if dep not in accepted_dirs:
                d = att / "accepted" / slug(dep)
                rs.fetch_stage(dep, d)
                (d / "result.json").unlink(missing_ok=True)
                accepted_dirs[dep] = d
        stage_dir = att / "stages" / slug(st.id)
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True)
        stop_n = None
        if self.stop_after:
            head, _, tail = self.stop_after.rpartition(":")
            key, stop_n = (head, int(tail)) if tail.isdigit() else (self.stop_after, None)
            if key not in (st.id, st.kind):
                stop_n = None
            elif stop_n is None:
                pass
        on_ck = None
        if st.kind == "train":
            ck = rs.doc().get("checkpoint")
            if ck:
                step_dir = stage_dir / "state" / "ckpt" / f"step-{ck['step']:09d}"
                rs.restore_checkpoint(step_dir)
                (step_dir / "PAYLOAD.json").unlink(missing_ok=True)
                (stage_dir / "state" / "ckpt" / "LATEST").write_text(json.dumps({"step": ck["step"]}))
                self.say(f"   restored the committed checkpoint: step {ck['step']}, data cursor "
                         f"{json.dumps(ck.get('cursor'), sort_keys=True)}")
            on_ck = self._ckpt_publisher(rs)
        ctx = StageContext(att, stage_dir, self.engine or plan.engine, stop_n, on_ck, self.say)
        # the stage's declared inputs by identity, before it runs
        for inp in st.inputs:
            f = att / "inputs" / inp["name"]
            if sha_file(f) != inp["sha256"]:
                raise SpillError(f"stage {st.id}: input {inp['name']} differs from its declared hash",
                                 "spill build <project> --new-run")
        t0 = time.monotonic()
        out = self.executor.run(st, ctx)
        if out.status != "done":
            self.say(f"stage {i} stopped after {_dur(time.monotonic() - t0)}; committed work is "
                     f"kept")
            return None
        for name in st.outputs:
            if not (ctx.out_dir / name).exists():
                raise SpillError(f"stage {st.id} did not produce {name}", "spill build <project> --debug")
        res = out.to_dict()
        res.update(stage=st.id, kind=st.kind, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        rs.publish_stage(st.id, ctx.out_dir, res)
        d = att / "accepted" / slug(st.id)
        rs.fetch_stage(st.id, d)
        (d / "result.json").unlink(missing_ok=True)
        self._record_timing(plan, st, out)
        self.say(f"stage {i} done in {_dur(out.seconds)}" + _score_note(plan, st, out))
        return res

    def _record_timing(self, plan, st, out) -> None:
        eng = self.engine or plan.engine
        try:
            if st.kind == "eval" and not out.metrics.get("prompt_cached_run"):
                timings.record(eng, st.params["model"], "eval", out.metrics.get("rows", 0), out.seconds, "row")
            elif st.kind == "train":
                toks = sum(len(r["input"]) / 4 + 30 for r in read_jsonl(plan.project / "data" / "train.jsonl"))
                reps = 3 if plan.cfg["model"].get("teacher") else 1
                timings.record(eng, st.params["model"], "train",
                               toks * reps * plan.cfg["training"]["epochs"], out.seconds, "token-pass")
            elif st.kind == "baseline":
                n = len(read_jsonl(plan.project / "data" / "train.jsonl")) + len(read_jsonl(plan.project / "data" / "val.jsonl"))
                timings.record("torch-cpu", "embedding:minilm", "baseline", n, out.seconds, "text")
        except OSError:
            pass

    def _ckpt_publisher(self, rs: RunState):
        from .. import runtime

        def publish(step: int, local_dir: Path, state: dict):
            rs.publish_checkpoint(local_dir, step, state.get("data_cursor"))
            if rs.quiesce_requested():
                runtime.ENV.stop.set()
        return publish

    def _stopped(self, rs, plan, results, why) -> BuildOutcome:
        self.say(f"build stopped ({why}); the run keeps everything committed")
        return BuildOutcome(rs.run_id, "stopped", None, results, [])

    def _finish(self, rs: RunState, plan: P.BuildPlan, results: dict, accepted: dict,
                resumed: bool, parent, t_start: float) -> BuildOutcome:
        att = rs.work
        snap = att / "snapshot"
        shutil.rmtree(snap, ignore_errors=True)
        schema = None
        if plan.cfg["task"]["type"] == "json":
            schema = read_json(att / "inputs" / plan.cfg["contract"]["schema_file"])
        integ = {}
        try:
            integ = read_json(plan.project / "data" / "integrity.json")
        except (OSError, ValueError):
            pass
        warnings = []
        cov = integ.get("coverage") or {}
        if cov.get("missing_in_val"):
            warnings.append(f"{len(cov['missing_in_val'])} training classes have no validation rows")
        if integ.get("conflicting_labels"):
            warnings.append(f"{len(integ['conflicting_labels'])} inputs have conflicting labels")
        doc = rs.doc()
        attempts = [{"event": h["event"], "generation": h.get("generation"), "owner": h.get("owner")}
                    for h in doc["history"] if h["event"] == "acquire"]
        manifest = R.assemble(
            snap, project=plan.project, run_id=rs.run_id, cfg=plan.cfg, plan_fields=plan.identity_fields,
            stage_results=results, accepted=accepted, models=_public_models(plan),
            deps=modelid.dependency_versions(), schema=schema, identity=plan.identity,
            inputs_dir=att / "inputs", parent=parent, attempts=attempts, warnings=warnings)
        (snap / "inputs" / "result.json").unlink(missing_ok=True)
        rs.complete(snap, meta={"run_id": rs.run_id})
        dest = rs.install_snapshot(plan.project)
        from .index import write_index
        write_index(plan.project)
        if self.remote_project:
            from .remote import push_completed
            push_completed(self.remote_project, plan.project, rs.run_id)
        return BuildOutcome(rs.run_id, "completed", dest, results, manifest["table"], resumed)


def _dur(s: float) -> str:
    from .. import estimate as est
    return "<1 s" if s < 1 else est.fmt_dur(s)


def _score_note(plan, st, out) -> str:
    task = plan.cfg["task"]["type"]
    m = out.metrics
    metric = R.metric_key(plan.cfg)
    v = m.get(metric)
    if st.kind in ("eval", "baseline") and v is not None:
        return f"; {R.primary_name(metric)} {v:.3f} on {m.get('rows')} rows"
    if st.kind == "train":
        return f"; {m.get('steps')} steps, final loss {m.get('final_loss')}"
    return ""
