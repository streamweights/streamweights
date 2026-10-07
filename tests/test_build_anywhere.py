# no-mlx-needed: runs on every platform (no MLX, no GPU)
"""spill build on every engine and across machines: engine-selected stages on tiny random models
(torch-cpu), portable build state (a directory and fsspec's memory filesystem), stop and resume,
headless SIGTERM, hardware-aware defaults, the Linux keep-awake, notification and battery
paths, and the examples."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from typer.testing import CliRunner  # noqa: E402

from streamweights import build as B  # noqa: E402
from streamweights import build_defaults as D  # noqa: E402
from streamweights import estimate as est  # noqa: E402
from streamweights import example as E  # noqa: E402
from streamweights import overnight  # noqa: E402
from streamweights.cli import app  # noqa: E402
from streamweights.portable.build_state import BuildState  # noqa: E402
from tests.tinytorch import make_tiny  # noqa: E402

R = CliRunner()
LABELS = ["alpha", "beta", "gamma"]
INS = "Answer with exactly one label from: " + ", ".join(LABELS) + "."


def jl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("ba-models")
    return {"student": str(make_tiny(root / "student", "llama", seed=0)),
            "teacher": str(make_tiny(root / "teacher", "llama", seed=1))}


def make_folder(root: Path, name: str, prompts: bool = False) -> Path:
    d = root / name
    d.mkdir(parents=True)
    jl(d / "evals.jsonl", [{"prompt": f"question {i}", "expected": LABELS[i % 3]}
                           for i in range(12)])
    jl(d / "train.jsonl", [{"prompt": f"homework {i}", "answer": LABELS[i % 3]}
                           for i in range(16)])
    if prompts:
        jl(d / "prompts.jsonl", [{"prompt": f"extra {i}"} for i in range(8)])
    (d / "instructions.txt").write_text(INS + "\n")
    return d


def rows_of(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


# ------------------------------------------------------------ hardware-aware defaults

def _folder(tmp_path, prompts=True):
    return B.read_folder(make_folder(tmp_path, "d", prompts=prompts))


def test_apple_silicon_defaults(tmp_path):
    c = D.choose(_folder(tmp_path), "mlx", {}, 36 * 2**30, 500 * 2**30)
    assert (c.student, c.teacher) == ("qwen2.5:7b", "llama3.3:70b")
    assert "Apple silicon default" in c.why()


def test_cuda_takes_the_defaults_when_memory_and_disk_allow(tmp_path):
    f = _folder(tmp_path)
    big = D.choose(f, "torch-cuda", {}, 80 * 2**30, 500 * 2**30)
    assert (big.student, big.teacher) == ("qwen2.5:7b", "llama3.3:70b")
    small = D.choose(f, "torch-cuda", {}, 8 * 2**30, 100 * 2**30)
    assert small.student == "qwen2.5:0.5b" and small.teacher == "qwen2.5:32b"
    assert "does not fit" in small.why() and "free disk" in small.why()


def test_cpu_student_is_the_0_5b_and_teacher_is_the_largest_under_12_hours(tmp_path):
    f = _folder(tmp_path)
    cal = {"engine_tflops": {"torch-cpu": 0.4}, "engine_membw_gbs": {"torch-cpu": 40.0}}
    c = D.choose(f, "torch-cpu", cal, 16 * 2**30, 500 * 2**30, read_rate=2e9)
    assert c.student == "qwen2.5:0.5b"
    chosen = D.distill_estimate(f, c.teacher, cal, 16 * 2**30, "torch-cpu", 2e9)
    assert chosen.seconds <= D.CPU_TEACHER_LIMIT_S
    bigger = [t for t in D.TEACHERS if est.model_bytes(t) > est.model_bytes(c.teacher)]
    for t in bigger:                              # every larger teacher is over the limit
        assert D.distill_estimate(f, t, cal, 16 * 2**30, "torch-cpu", 2e9
                                  ).seconds > D.CPU_TEACHER_LIMIT_S
    assert "under 12 h" in c.why()


def test_a_forced_model_is_never_refused_and_a_long_estimate_is_stated(tmp_path):
    f = _folder(tmp_path)
    c = D.choose(f, "torch-cpu", {}, 16 * 2**30, 500 * 2**30, student="qwen2.5:7b",
                 teacher="llama3.3:70b")
    assert (c.student, c.teacher) == ("qwen2.5:7b", "llama3.3:70b") and c.forced == {"student", "teacher"}
    plan = B.make_plan(f, c.student, c.teacher, None, [], 2.0,
                       {"engine_tflops": {"torch-cpu": 0.02}, "engine_membw_gbs": {"torch-cpu": 5.0}},
                       16 * 2**30, engine="torch-cpu", read_rate=1e7)
    line = B.pre_run_line(plan)
    assert any(s.est_s > 24 * 3600 for s in plan.stages)
    assert "over 24 h; going ahead as asked" in line


def test_calibration_is_kept_per_engine_and_model():
    cal = {}
    est.record_stage_ratio(cal, "torch-cpu", "qwen2.5:0.5b", "eval", 100.0, 250.0)
    est.record_stage_ratio(cal, "mlx", "qwen2.5:0.5b", "eval", 100.0, 40.0)
    assert est.stage_ratio(cal, "torch-cpu", "qwen2.5:0.5b", "eval")[0] == pytest.approx(2.5)
    assert est.stage_ratio(cal, "mlx", "qwen2.5:0.5b", "eval")[0] == pytest.approx(0.4)
    assert est.stage_ratio(cal, "torch-cpu", "qwen2.5:7b", "eval") == (1.0, 0)
    est.record_stage_ratio(cal, "torch-cpu", "qwen2.5:7b", "eval", 5.0, 500.0)   # too short to say
    assert est.stage_ratio(cal, "torch-cpu", "qwen2.5:7b", "eval") == (1.0, 0)


# ------------------------------------------------------------ portable build state

def test_build_state_round_trips_on_the_memory_filesystem(tmp_path):
    bs = BuildState("memory://ba-state-1")
    assert bs.read() is None
    bs.write({"a": 1})
    assert bs.read()["a"] == 1
    d = tmp_path / "ad"
    d.mkdir()
    (d / "adapters.safetensors").write_bytes(b"xyz")
    info = bs.put_dir("adapters/x", d)
    out = tmp_path / "back"
    bs.get_dir("adapters/x", out)
    assert (out / "adapters.safetensors").read_bytes() == b"xyz" and info["files"] == 1
    assert bs.stage_uri("eval:base") == "memory://ba-state-1/stages/eval-base"


# ------------------------------------------------------------ real stages, torch-cpu, tiny models

def build_args(folder, models, *extra, teacher=False):
    a = ["build", str(folder), "--student", models["student"], "--engine", "torch-cpu",
         "--epochs", "3", *extra]
    if teacher:
        a += ["--teacher", models["teacher"]]
    return a


def stage_ids(doc):
    return [s["id"] for s in doc["stages"]]


def check_no_loss_or_duplication(doc_folder: Path, state_uri: str):
    """Every row job in the state holds each row exactly once, and a tune stage's recorded step
    range is contiguous from 0, with the loss of every step 1..N present once."""
    from streamweights.portable import checkpoint as pc
    from streamweights.portable import rows as pr
    from streamweights.portable.store import Store
    st = Store(state_uri)
    seen_rows = 0
    for stage in st.ls("stages"):
        sub = st.sub(f"stages/{stage}")
        if sub.exists("ckpt"):
            ck = pc.load_tune(sub)
            rng = [h["range"] for h in ck.state["history"]]
            assert rng[0][0] == 0 and all(a[1] == b[0] for a, b in zip(rng, rng[1:])), rng
            steps = [s for s, _ in ck.state["losses"]]
            assert steps == list(range(1, ck.step + 1))            # none missing, none repeated
        for where in [sub] + [sub.sub(leaf) for leaf in sub.ls("") if sub.exists(f"{leaf}/rows")]:
            if where.exists("rows"):
                ids = [i for seg in pr.committed_segments(where) for i in seg["ids"]]
                assert len(ids) == len(set(ids)), f"duplicate rows in {stage}"
                seen_rows += len(ids)
    assert seen_rows > 0


def test_build_stops_in_tune_and_resumes_from_a_state_directory(tmp_path, models):
    folder = make_folder(tmp_path, "sd")
    state = tmp_path / "shared"
    r = R.invoke(app, build_args(folder, models, "--state", str(state), "--stop-after", "tune:3"))
    assert r.exit_code == 0, r.output
    assert "stopped early (--stop-after tune:3)" in r.output
    doc = json.loads((state / "state.json").read_text())
    assert [s["status"] for s in doc["stages"]] == ["done", "interrupted", "pending"]
    assert doc["finished"] is False and doc["stages"][0]["producers"][0]["engine"] == "torch-cpu"
    assert (state / "stages" / "tune" / "ckpt" / "LATEST").exists()       # a checkpoint was written
    r = R.invoke(app, ["resume", str(folder), "--state", str(state), "--engine", "torch-cpu"])
    assert r.exit_code == 0, r.output
    doc = json.loads((state / "state.json").read_text())
    assert doc["finished"] is True and [s["status"] for s in doc["stages"]] == ["done"] * 3
    assert "your model" in r.output and "ran on" in r.output
    tune = doc["stages"][1]
    ranges = [p["quanta"] for p in tune["producers"]]
    assert ranges == ["steps 0-%d" % tune["result"]["steps"]]       # same engine: one merged range
    assert {p["engine"] for p in tune["producers"]} == {"torch-cpu"}
    assert all(p["os"] and p["system"] for s in doc["stages"] for p in s["producers"])
    assert (state / "adapters" / "sd" / "MANIFEST.json").exists()
    check_no_loss_or_duplication(folder, str(state))
    table = B.table_from_state(doc)
    assert "torch-cpu" in table and "1 eval:base" in table


def test_build_with_remote_state_on_the_memory_filesystem(tmp_path, models):
    """distill -> tune -> evals, stopped inside the distill stage and again inside tune,
    with --state on fsspec's in-memory filesystem."""
    folder = make_folder(tmp_path, "mem", prompts=True)
    uri = "memory://ba-build-mem"
    base = build_args(folder, models, "--state", uri, teacher=True)
    r = R.invoke(app, base + ["--stop-after", "distill:3"])
    assert r.exit_code == 0 and "stopped early" in r.output, r.output
    doc = BuildState(uri).read()
    assert doc["stages"][0]["status"] == "interrupted"
    r = R.invoke(app, base + ["--stop-after", "tune:2"])         # the same command again continues
    assert r.exit_code == 0 and "stopped early" in r.output, r.output
    doc = BuildState(uri).read()
    assert [s["status"] for s in doc["stages"]][:2] == ["done", "interrupted"]
    assert "distill" in doc["artifacts"]
    r = R.invoke(app, ["resume", str(folder), "--state", uri, "--engine", "torch-cpu"])
    assert r.exit_code == 0, r.output
    doc = BuildState(uri).read()
    assert doc["finished"] is True and stage_ids(doc) == [
        "distill", "tune", "eval:base", "eval:tuned", "eval:teacher"]
    d = doc["stages"][0]
    assert d["result"]["rows"] == 8 and d["producers"][0]["quanta"] == "8 rows"
    check_no_loss_or_duplication(folder, uri)


def test_resume_picks_up_the_state_from_the_folders_own_pointer(tmp_path, models):
    folder = make_folder(tmp_path, "ptr")
    uri = f"{tmp_path / 'elsewhere'}"
    R.invoke(app, build_args(folder, models, "--state", uri, "--stop-after", "tune:2"))
    r = R.invoke(app, ["resume", str(folder), "--engine", "torch-cpu"])   # no --state: the folder remembers it
    assert r.exit_code == 0, r.output
    assert json.loads((Path(uri) / "state.json").read_text())["finished"] is True


def test_a_resumed_build_keeps_the_models_it_started_with(tmp_path, models):
    folder = make_folder(tmp_path, "keep")
    state = tmp_path / "keepstate"
    R.invoke(app, build_args(folder, models, "--state", str(state), "--stop-after", "tune:2"))
    # same command without the model flags, as a second machine would run it: the plan is the first one
    r = R.invoke(app, ["build", str(folder), "--engine", "torch-cpu", "--state", str(state)])
    assert r.exit_code == 0, r.output
    assert "continuing the models chosen when this build started" in r.output
    assert "continuing: 1 of 3 stages already done" in r.output


def test_edited_inputs_start_the_state_fresh(tmp_path, models):
    folder = make_folder(tmp_path, "edit")
    state = tmp_path / "editstate"
    R.invoke(app, build_args(folder, models, "--state", str(state), "--stop-after", "tune:2"))
    with open(folder / "train.jsonl", "a") as f:
        f.write(json.dumps({"prompt": "more", "answer": "alpha"}) + "\n")
    r = R.invoke(app, build_args(folder, models, "--state", str(state), "--stop-after", "tune:2"))
    assert "was for a different plan; starting this build fresh" in r.output


def test_emit_config_and_config_for_build(tmp_path, models):
    folder = make_folder(tmp_path, "cfg")
    cfg = tmp_path / "job.json"
    env = {**os.environ, "SPILL_HEADLESS": "0"}
    p = subprocess.run([sys.executable, "-m", "streamweights", "build", str(folder), "--student",
                        models["student"], "--engine", "torch-cpu", "--epochs", "2", "--state",
                        str(tmp_path / "cs"), "--emit-config", str(cfg)],
                       capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0, p.stderr
    c = json.loads(cfg.read_text())
    assert c["command"] == "build" and c["options"]["engine"] == "torch-cpu"
    assert c["options"]["state"] == str(tmp_path / "cs") and c["arguments"]["folder"] == str(folder)
    p = subprocess.run([sys.executable, "-m", "streamweights", "build", "--config", str(cfg)],
                       capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0, p.stderr + p.stdout
    assert json.loads((tmp_path / "cs" / "state.json").read_text())["finished"] is True


def _mlx():
    from streamweights.platforms import mlx_available
    return mlx_available()


@pytest.mark.skipif(not _mlx(), reason="the other engine is MLX, which needs Apple silicon")
@pytest.mark.parametrize("first,then", [("torch-cpu", "mlx"), ("mlx", "torch-cpu")])
def test_build_stopped_on_one_engine_finishes_on_the_other(tmp_path, models, first, then):
    folder = make_folder(tmp_path, f"x-{first}")
    state = tmp_path / "xs"
    args = build_args(folder, models, "--state", str(state), "--stop-after", "tune:4")
    args[args.index("--engine") + 1] = first
    r = R.invoke(app, args)
    assert r.exit_code == 0 and "stopped early" in r.output, r.output
    r = R.invoke(app, ["resume", str(folder), "--state", str(state), "--engine", then])
    assert r.exit_code == 0, r.output
    assert f"the rest runs on {then}" in r.output
    doc = json.loads((state / "state.json").read_text())
    assert doc["finished"] is True
    engines = [[p["engine"] for p in s["producers"]] for s in doc["stages"]]
    assert engines[0] == [first] and engines[1][0] == first and engines[1][-1] == then
    assert engines[2] == [then]
    check_no_loss_or_duplication(folder, str(state))


# ------------------------------------------------------------ headless

_HOME = {}


def _home():
    if "p" not in _HOME:
        import shutil
        import tempfile
        h = Path(tempfile.mkdtemp(prefix="spill-ba-home-"))
        (h / "state").mkdir()
        for f in ("hardware.json", "calibration.json"):
            src = Path(os.environ["SPILL_HOME"]) / "state" / f
            if src.exists():
                shutil.copy(src, h / "state" / f)
        _HOME["p"] = str(h)
    return _HOME["p"]


def test_headless_build_sigterm_exits_75_with_a_checkpoint_and_resumes(tmp_path, models):
    folder = make_folder(tmp_path, "hl")
    state = tmp_path / "hlstate"
    env = {k: v for k, v in os.environ.items() if k != "SPILL_HEADLESS"}
    env["SPILL_HOME"] = _home()
    cmd = [sys.executable, "-m", "streamweights", "build", str(folder), "--student",
           models["student"], "--engine", "torch-cpu", "--epochs", "60", "--state", str(state),
           "--headless"]
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    events, sent = [], False
    deadline = time.time() + 240
    for line in p.stdout:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        events.append(e)
        if e["event"] == "step" and e["step"] >= 5 and not sent:
            p.send_signal(signal.SIGTERM)
            sent = True
        assert time.time() < deadline, "build did not reach the tune stage"
    rc = p.wait(timeout=120)
    assert sent and rc == 75, (rc, [e["event"] for e in events][-5:], p.stderr.read()[-400:])
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start" and "checkpoint" in kinds and kinds[-1] == "preempted"
    odd = [(e["event"], e["command"], e["engine"]) for e in events
           if not (e["command"] in ("build", "eval", "tune", "distill")
                   and e["engine"] == "torch-cpu")]
    assert not odd, odd[:5]
    stages = [(e["stage"], e["status"]) for e in events if e["event"] == "stage"]
    assert stages[:3] == [("eval:base", "start"), ("eval:base", "done"), ("tune", "start")]
    assert (state / "stages" / "tune" / "ckpt" / "LATEST").exists()
    doc = json.loads((state / "state.json").read_text())
    assert doc["stages"][1]["status"] == "interrupted" and not doc["finished"]
    # the retry a scheduler would make
    r = subprocess.run([sys.executable, "-m", "streamweights", "resume", str(folder), "--state",
                        str(state), "--headless", "--engine", "torch-cpu"], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-600:]
    ev = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    assert ev[-1]["event"] == "done" and ev[-1]["complete"] is True
    doc = json.loads((state / "state.json").read_text())
    assert doc["finished"] is True
    check_no_loss_or_duplication(folder, str(state))


# ------------------------------------------------------------ no Mac-only plumbing

def test_battery_on_linux_reads_sys_class_power_supply(tmp_path):
    root = tmp_path / "ps"
    bat, ac = root / "BAT0", root / "AC"
    bat.mkdir(parents=True)
    ac.mkdir()
    (bat / "type").write_text("Battery\n")
    (ac / "type").write_text("Mains\n")
    (bat / "status").write_text("Discharging\n")
    (ac / "online").write_text("0\n")
    assert overnight._linux_on_battery(root) is True
    (ac / "online").write_text("1\n")
    (bat / "status").write_text("Charging\n")
    assert overnight._linux_on_battery(root) is False
    assert overnight._linux_on_battery(tmp_path / "no-such-dir") is False      # a server: no battery


def test_keep_awake_uses_systemd_inhibit_on_linux_and_skips_silently_without_it(monkeypatch):
    calls = []

    class P:
        def terminate(self): calls.append("terminate")
        def wait(self, timeout=None): calls.append("wait")

    monkeypatch.setattr(overnight, "_mac", lambda: False)
    monkeypatch.setattr(overnight, "_linux", lambda: True)
    monkeypatch.setattr(overnight.shutil, "which",
                        lambda n: "/usr/bin/systemd-inhibit" if n == "systemd-inhibit" else None)
    monkeypatch.setattr(overnight.subprocess, "Popen", lambda a, **k: calls.append(a[0]) or P())
    with overnight.caffeinate():
        pass
    assert calls[0] == "systemd-inhibit" and "terminate" in calls
    calls.clear()
    monkeypatch.setattr(overnight.shutil, "which", lambda n: None)
    with overnight.caffeinate() as proc:                  # nothing to hold, nothing printed
        assert proc is None
    assert calls == []


def test_notify_uses_notify_send_on_linux_and_the_url_everywhere(monkeypatch):
    ran = []
    monkeypatch.delenv("SPILL_NO_NOTIFY", raising=False)
    monkeypatch.setattr(overnight, "_mac", lambda: False)
    monkeypatch.setattr(overnight, "_linux", lambda: True)
    monkeypatch.setattr(overnight.shutil, "which",
                        lambda n: "/usr/bin/notify-send" if n == "notify-send" else None)
    monkeypatch.setattr(overnight.subprocess, "run", lambda a, **k: ran.append(a[0]))
    import httpx
    posts = []
    monkeypatch.setattr(httpx, "post", lambda url, **k: posts.append((url, k["json"]["state"])))
    done = overnight.notify("spill", "x", "http://example.invalid/hook", {"state": "finished"})
    assert ran == ["notify-send"] and posts == [("http://example.invalid/hook", "finished")]
    assert done[0] == "notify-send" and done[1].startswith("POST")
    ran.clear()
    overnight.notify("spill", "x", None, local=False)         # headless: no desktop notification
    assert ran == []
    monkeypatch.setattr(overnight.shutil, "which", lambda n: None)
    assert overnight.notify("spill", "x") == []               # not installed: skipped silently


def test_headless_build_never_holds_the_machine_awake(monkeypatch, tmp_path):
    from streamweights import cli, runtime
    held = []
    monkeypatch.setattr(overnight, "caffeinate", lambda: held.append(1) or __import__(
        "contextlib").nullcontext())
    runtime.reset()
    runtime.ENV.headless = True
    try:
        with cli._long_job("spill build x", None):
            pass
        assert held == []
        with cli._long_job("spill build x", "http://example.invalid/hook"):    # URL only
            pass
        assert held == []
    finally:
        runtime.reset()


# ------------------------------------------------------------ examples

def test_tiny_example_has_the_promised_shape(tmp_path):
    f = E.create("banking77", False, tmp_path, tiny=True)
    assert f.name == "banking77-tiny"
    assert len(rows_of(f / "evals.jsonl")) == 20 and len(rows_of(f / "train.jsonl")) == 100
    assert json.loads((f / "spill.json").read_text())["student"] == "qwen2.5:0.5b"
    ev = {r["prompt"] for r in rows_of(f / "evals.jsonl")}
    tr = {r["prompt"] for r in rows_of(f / "train.jsonl")}
    assert not ev & tr and max(len(p) for p in ev | tr) <= 90
    labels = {r["answer"] for r in rows_of(f / "train.jsonl")}
    assert len(labels) == 10 and {r["expected"] for r in rows_of(f / "evals.jsonl")} <= labels
    assert all(f"- {l}\n" in (f / "instructions.txt").read_text() for l in labels)
    B.read_folder(f)                                              # a valid build folder


def test_relay_example_folder_and_script(tmp_path):
    f = E.create("relay", False, tmp_path)
    assert sorted(p.name for p in f.iterdir()) == [
        "README.md", "evals.jsonl", "instructions.txt", "relay.sh", "spill.json", "train.jsonl"]
    sh = f / "relay.sh"
    assert os.access(sh, os.X_OK)
    assert subprocess.run(["sh", "-n", str(sh)]).returncode == 0
    text = sh.read_text()
    for needle in ("spill build relay --state ./relay-state", "--stop-after", "docker run",
                   "ghcr.io/streamweights/spill:cpu", "spill resume relay --state",
                   "--engine", "--two-machines", "different engine on the same machine"):
        assert needle in text
    assert all(len(l) <= 100 for l in text.splitlines() if l.strip().startswith(("say ", "echo ")))
    h = subprocess.run(["sh", str(sh), "--help"], capture_output=True, text=True)
    assert h.returncode == 0 and "--two-machines" in h.stdout
    bad = subprocess.run(["sh", str(sh), "--nope"], capture_output=True, text=True)
    assert bad.returncode == 2 and "unknown option" in bad.stdout


def test_example_command_creates_relay_and_names_the_next_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = R.invoke(app, ["example", "relay"])
    assert r.exit_code == 0 and "created relay/" in r.output
    assert r.output.rstrip().endswith("next: relay/relay.sh")
    r = R.invoke(app, ["example", "banking77", "--tiny"])
    assert r.exit_code == 0 and r.output.rstrip().endswith("next: spill build banking77-tiny")
    r = R.invoke(app, ["example", "banking77", "--tiny", "--quick", "--force"])
    assert r.exit_code == 1 and "pick one" in r.output
