# no-mlx-needed: runs on every platform (no MLX, no GPU)
"""Headless mode: JSON-lines events on stdout, SIGTERM checkpoints and exits 75, a retry
resumes from --state; --emit-config and --config; engine selection."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from streamweights import engine_select, headless, jobconfig  # noqa: E402
from streamweights.errors import SpillError  # noqa: E402
from streamweights.portable import checkpoint as pc  # noqa: E402
from streamweights.portable.store import Store  # noqa: E402
from tests.tinytorch import make_tiny  # noqa: E402

EVENT_KEYS = {"event", "time", "command", "step", "loss", "tokens_per_s", "peak_mem_gb",
              "eta_s", "engine"}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("hl")
    d = make_tiny(root / "tiny", "qwen2")
    from streamweights.tune import toy
    paths = toy.make(root / "toy", 48, 12)
    rows = root / "rows.jsonl"
    rows.write_text("".join(json.dumps({"prompt": f"hello {i} " + "z" * i, "expected": "a"}) + "\n"
                            for i in range(10)))
    return {"model": d, "train": paths["train"], "held": paths["heldout"], "rows": rows,
            "root": root}


_HOME = {}


def _private_home() -> str:
    """A SPILL_HOME of this module's own: jobs these tests interrupt on purpose must not show up
    as 'interrupted' banners in every other test of the session."""
    if "p" not in _HOME:
        import shutil
        import tempfile
        home = Path(tempfile.mkdtemp(prefix="spill-headless-home-"))
        (home / "state").mkdir()
        for f in ("hardware.json", "calibration.json"):
            src = Path(os.environ["SPILL_HOME"]) / "state" / f
            if src.exists():
                shutil.copy(src, home / "state" / f)
        _HOME["p"] = str(home)
    return _HOME["p"]


def spill(args, env_extra=None, **kw):
    env = {k: v for k, v in os.environ.items() if k != "SPILL_HEADLESS"}
    env["SPILL_HOME"] = _private_home()
    env.update(env_extra or {})
    return subprocess.Popen([sys.executable, "-m", "streamweights", *args], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)


def run_spill(args, env_extra=None, timeout=300):
    p = spill(args, env_extra)
    out, err = p.communicate(timeout=timeout)
    return p.returncode, out, err


def events(out: str) -> list[dict]:
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def test_events_carry_the_standard_fields():
    import io
    buf = io.StringIO()
    ev = headless.Events(buf, engine="torch-cpu", command="tune")
    ev.emit("start")
    ev.step({"step": 3, "steps": 9, "loss": 1.5, "tok_s": 100.0, "peak_gb": 0.5, "eta_s": 4.0,
             "step_s": 0.2})
    ev.row(2, 10, 55.5, 3.0, "r1")
    recs = events(buf.getvalue())
    assert [r["event"] for r in recs] == ["start", "step", "row"]
    for r in recs:
        assert EVENT_KEYS <= set(r) and r["engine"] == "torch-cpu"
    assert recs[1]["loss"] == 1.5 and recs[1]["step"] == 3 and recs[1]["peak_mem_gb"] == 0.5
    assert recs[2]["tokens_per_s"] == 55.5 and recs[2]["eta_s"] == 3.0


def test_headless_detection(monkeypatch):
    monkeypatch.delenv("SPILL_HEADLESS", raising=False)
    assert headless.is_headless(True)
    assert headless.is_headless(False) is True            # pytest's stdout is not a TTY
    monkeypatch.setenv("SPILL_HEADLESS", "0")
    assert headless.is_headless(False) is False and headless.is_headless(True) is True
    monkeypatch.setenv("SPILL_HEADLESS", "1")
    assert headless.is_headless(False) is True


def test_run_headless_stdout_is_only_json_lines(tiny):
    rc, out, err = run_spill(["run", str(tiny["model"]), str(tiny["rows"]), "--engine",
                              "torch-cpu", "--headless"])
    assert rc == 0, err
    recs = events(out)                                   # every line parses
    kinds = [r["event"] for r in recs]
    assert kinds[0] == "start" and kinds[-1] == "done" and kinds.count("row") == 10
    assert all(EVENT_KEYS <= set(r) for r in recs)
    assert all(r["engine"] == "torch-cpu" for r in recs)
    assert recs[0]["message"].startswith("spill run ") and "Cost: $0" in recs[0]["message"]
    assert "next:" in err and "next:" not in out         # human text went to stderr
    assert "\x1b[" not in out and "\x1b[" not in err.replace("\x1b[K", "")  # no progress drawing


def test_a_pipe_turns_headless_on_by_itself(tiny):
    rc, out, err = run_spill(["run", str(tiny["model"]), str(tiny["rows"]), "--engine",
                              "torch-cpu"])
    assert rc == 0 and events(out)[0]["event"] == "start"


def test_errors_are_an_error_event_and_a_nonzero_exit(tiny):
    rc, out, err = run_spill(["run", str(tiny["model"]), "/no/such/file.jsonl", "--engine",
                              "torch-cpu", "--headless"])
    assert rc == 1
    last = events(out)[-1]
    assert last["event"] == "error" and "does not exist" in last["message"]
    assert err.strip().startswith("spill:") and "Try:" in err


def test_sigterm_checkpoints_exits_75_and_a_retry_resumes(tiny, tmp_path):
    state = str(tmp_path / "state")
    args = ["tune", str(tiny["model"]), tiny["train"], "--name", "sig-a", "--engine", "torch-cpu",
            "--steps", "1500", "--batch", "4", "--rank", "4", "--lr", "3e-3", "--overwrite",
            "--state", state, "--headless"]
    p = spill(args)
    seen, t_sig = [], None
    for line in p.stdout:
        ev = json.loads(line)
        seen.append(ev)
        if ev["event"] == "step" and ev["step"] == 25 and t_sig is None:
            p.send_signal(signal.SIGTERM)
            t_sig = time.monotonic()
    p.wait(timeout=60)
    assert p.returncode == 75, p.stderr.read()
    assert time.monotonic() - t_sig < headless.GRACE_SECONDS      # within the grace period
    assert seen[-1]["event"] == "preempted" and seen[-1]["signal"] == "SIGTERM"
    ck_step = pc.latest_step(Store(state))
    assert ck_step is not None and ck_step >= 25
    # a retry (same command) resumes from the checkpoint at the URI and loses nothing
    p2 = spill(args[:args.index("--steps")] + ["--steps", str(ck_step + 6)]
               + args[args.index("--steps") + 2:])
    out, err = p2.communicate(timeout=300)
    # the schedule is part of the checkpoint's identity: a different --steps is refused
    assert p2.returncode == 1 and "different" not in err and "steps" in err
    p3 = spill(args[:args.index("--steps")] + ["--steps", "1500", "--stop-after",
                                               str(ck_step + 5)] + args[args.index("--steps") + 2:])
    out, err = p3.communicate(timeout=300)
    assert p3.returncode == 0, err
    recs = events(out)
    steps = [r["step"] for r in recs if r["event"] == "step"]
    assert steps[0] == ck_step + 1 and steps[-1] == ck_step + 5
    assert recs[-1]["event"] == "done" and recs[-1]["stopped_early"] is True


def test_emit_config_writes_the_exact_invocation_and_does_not_run(tiny, tmp_path):
    cfg = tmp_path / "job.json"
    rc, out, err = run_spill(["tune", str(tiny["model"]), tiny["train"], "--name", "cfg-a",
                              "--rank", "8", "--steps", "7", "--engine", "torch-cpu", "--state",
                              str(tmp_path / "s"), "--emit-config", str(cfg)])
    assert rc == 0 and "next: spill tune --config" in out
    c = json.loads(cfg.read_text())
    assert c["spill_config"] == jobconfig.VERSION and c["command"] == "tune"
    assert c["arguments"] == {"model": str(tiny["model"]), "train_jsonl": tiny["train"]}
    assert c["options"]["name"] == "cfg-a" and c["options"]["rank"] == 8
    assert c["options"]["steps"] == 7 and c["options"]["engine"] == "torch-cpu"
    assert c["options"]["state"] == str(tmp_path / "s")
    assert "config" not in c["options"] and "emit_config" not in c["options"]
    assert not (tmp_path / "s").exists()                 # nothing ran
    # loading it runs the same job; an option on the command line wins
    rc, out, err = run_spill(["tune", "--config", str(cfg), "--overwrite", "--steps", "3",
                              "--headless"])
    assert rc == 0, err
    recs = events(out)
    assert [r["step"] for r in recs if r["event"] == "step"] == [1, 2, 3]
    assert recs[0]["event"] == "start" and "rank 8" in recs[0]["message"]


def test_config_for_the_wrong_command_is_one_line(tiny, tmp_path):
    cfg = tmp_path / "job.json"
    run_spill(["tune", str(tiny["model"]), tiny["train"], "--name", "w", "--emit-config",
               str(cfg)])
    rc, out, err = run_spill(["run", "--config", str(cfg)])
    assert rc == 1 and len(err.strip().splitlines()) == 1 and "is a spill tune job" in err


def test_config_roundtrip_in_process():
    from typer.main import get_command

    from streamweights.cli import app
    group = get_command(app)
    argv = ["m", "t.jsonl", "--name", "n", "--rank", "4", "--overwrite"]
    cfg = jobconfig.build("tune", group.commands["tune"], argv)
    assert cfg["options"]["overwrite"] is True and cfg["options"]["rank"] == 4
    again = jobconfig.to_argv(cfg, group.commands["tune"], [])
    cfg2 = jobconfig.build("tune", group.commands["tune"], again)
    assert cfg2 == cfg


# ---------------------------------------------------------------- engine selection

def test_engine_selection_is_automatic(monkeypatch):
    monkeypatch.delenv("SPILL_ENGINE", raising=False)
    monkeypatch.setattr("streamweights.platforms.mlx_available", lambda: True)
    assert engine_select.choose_engine().name == "mlx"
    monkeypatch.setattr("streamweights.platforms.mlx_available", lambda: False)
    monkeypatch.setattr("streamweights.platforms.apple_silicon", lambda: False)
    monkeypatch.setattr(engine_select, "_cuda_ok", lambda: False)
    c = engine_select.choose_engine()
    assert c.name == "torch-cpu" and "no Apple silicon" in c.why
    monkeypatch.setattr(engine_select, "_cuda_ok", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "Test GPU")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    c = engine_select.choose_engine()
    assert c.name == "torch-cuda" and "Test GPU" in c.why


def test_engine_override_and_its_errors(monkeypatch):
    monkeypatch.setattr("streamweights.platforms.mlx_available", lambda: False)
    monkeypatch.setattr("streamweights.platforms.apple_silicon", lambda: False)
    monkeypatch.setattr(engine_select, "_cuda_ok", lambda: False)
    assert engine_select.choose_engine("torch-cpu").forced
    with pytest.raises(SpillError) as e:
        engine_select.choose_engine("torch-cuda")
    assert "no CUDA device" in e.value.message and "--engine torch-cpu" in e.value.recovery
    with pytest.raises(SpillError) as e:
        engine_select.choose_engine("mlx")
    assert "Apple silicon" in e.value.message
    with pytest.raises(SpillError, match="must be one of"):
        engine_select.choose_engine("tpu")
    monkeypatch.setenv("SPILL_ENGINE", "torch-cpu")
    assert engine_select.choose_engine().name == "torch-cpu"


def test_doctor_lines_name_the_choice_and_each_engine():
    from streamweights import doctor
    avail = {"mlx": (False, "needs Apple silicon"), "torch-cpu": (True, "PyTorch on the CPU"),
             "torch-cuda": (False, "no CUDA device is visible")}
    rates = {"torch-cpu": {"matmul_tflops": 1.5, "memory_gb_s": 100.0}}
    lines = doctor.engine_lines(engine_select.EngineChoice("torch-cpu", "no CUDA"), avail, rates)
    text = "\n".join(lines)
    assert lines[0].startswith("engine") and "torch-cpu (no CUDA)" in lines[0]
    assert "1.5 TFLOP/s" in text and "not usable here: no CUDA device is visible" in text
    assert "mlx: not usable here: needs Apple silicon" in text


def test_sigterm_on_a_row_job_exits_75_and_the_retry_finishes_every_row_once(tiny, tmp_path):
    rows = tmp_path / "many.jsonl"
    rows.write_text("".join(json.dumps({"custom_id": f"m{i:04d}", "prompt": f"hello {i} " + "q" * (i % 9)})
                            + "\n" for i in range(300)))
    state = str(tmp_path / "rowstate")
    args = ["run", str(tiny["model"]), str(rows), "--engine", "torch-cpu", "--state", state,
            "--headless", "--parallel", "2"]
    p = spill(args)
    t_sig = None
    for line in p.stdout:
        ev = json.loads(line)
        if ev["event"] == "row" and ev["step"] >= 15 and t_sig is None:
            p.send_signal(signal.SIGTERM)
            t_sig = time.monotonic()
        last = ev
    p.wait(timeout=60)
    assert p.returncode == 75, p.stderr.read()
    assert last["event"] == "preempted" and last["signal"] == "SIGTERM"
    assert time.monotonic() - t_sig < headless.GRACE_SECONDS
    from streamweights.portable import rows as pr
    first_leg = pr.done_ids(Store(state))
    assert 15 <= len(first_leg) < 300
    rc, out, err = run_spill(args)
    assert rc == 0, err
    recs = events(out)
    done_rows = [r for r in recs if r["event"] == "row"]
    assert len(done_rows) == 300 - len(first_leg)             # only what was missing
    assert not ({r["custom_id"] for r in done_rows} & first_leg)
    assert recs[0]["event"] == "restore" or any(r["event"] == "restore" for r in recs)
    assert pr.done_ids(Store(state)) == {f"m{i:04d}" for i in range(300)}
    assert recs[-1]["event"] == "done" and recs[-1]["complete"] is True
