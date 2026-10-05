import json
import os
import time
from pathlib import Path

from streamweights import overnight


def test_battery_note_only_on_battery(monkeypatch):
    monkeypatch.setattr(overnight, "on_battery", lambda: True)
    assert "plug in" in overnight.battery_note()
    monkeypatch.setattr(overnight, "on_battery", lambda: False)
    assert overnight.battery_note() == ""


def test_pmset_parse(monkeypatch):
    class R:
        stdout = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1) 80%"
    monkeypatch.setattr(overnight.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(overnight.shutil, "which", lambda x: "/usr/bin/pmset")
    monkeypatch.setattr(overnight.subprocess, "run", lambda *a, **k: R())
    assert overnight.on_battery() is True
    R.stdout = "Now drawing from 'AC Power'"
    assert overnight.on_battery() is False


def test_notify_posts_json_and_survives_dead_webhook(monkeypatch):
    monkeypatch.setenv("SPILL_NO_NOTIFY", "1")
    sent = {}

    class Resp: ...
    import httpx
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: sent.update(url=url, json=json) or Resp())
    out = overnight.notify("spill", "build finished", "http://x.test/hook", {"state": "finished"})
    assert sent["url"] == "http://x.test/hook" and sent["json"]["state"] == "finished"
    assert out == ["POST http://x.test/hook"]

    def boom(*a, **k):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(httpx, "post", boom)
    assert "failed" in overnight.notify("spill", "m", "http://dead.test")[0]


def test_long_job_notifies_on_finish_stop_and_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(overnight, "notify", lambda t, m, url=None, payload=None: calls.append((m, payload)))
    monkeypatch.setattr(overnight, "caffeinate", overnight.contextlib.nullcontext)
    with overnight.long_job("spill build x"):
        pass
    try:
        with overnight.long_job("spill build x"):
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    try:
        with overnight.long_job("spill build x"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert [c[1]["state"] for c in calls] == ["finished", "stopped", "failed"]
    assert "boom" in calls[2][0]


def test_caffeinate_is_tied_to_our_pid(monkeypatch):
    seen = {}
    monkeypatch.setattr(overnight.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(overnight.shutil, "which", lambda x: "/usr/bin/caffeinate")

    class P:
        def terminate(self): seen["terminated"] = True
        def wait(self, timeout=None): ...
    monkeypatch.setattr(overnight.subprocess, "Popen", lambda cmd, **k: seen.update(cmd=cmd) or P())
    with overnight.caffeinate():
        pass
    assert seen["cmd"][:2] == ["caffeinate", "-i"] and seen["cmd"][-2:] == ["-w", str(os.getpid())]
    assert seen["terminated"]


def mkjob(d: Path, id, **meta):
    (d / id).mkdir(parents=True)
    (d / id / "meta.json").write_text(json.dumps({"id": id, **meta}))


def test_interrupted_job_detection(tmp_path):
    mkjob(tmp_path, "a", status="interrupted", done=3, total=10, model="m")
    mkjob(tmp_path, "b", status="completed", done=10, total=10)
    mkjob(tmp_path, "c", status="running", done=1, total=10, pid=999999)       # dead pid
    mkjob(tmp_path, "d", status="running", done=1, total=10, pid=os.getpid())  # alive
    mkjob(tmp_path, "e", status="running", done=1, total=10)                   # fresh, no pid
    got = {j["id"] for j in overnight.interrupted_jobs(tmp_path)}
    assert got == {"a", "c"}
    old = time.time() - 3600
    os.utime(tmp_path / "e" / "meta.json", (old, old))
    assert {j["id"] for j in overnight.interrupted_jobs(tmp_path)} == {"a", "c", "e"}


def test_banner_names_the_resume_command(tmp_path, monkeypatch):
    mkjob(tmp_path, "job-1", status="interrupted", done=3, total=10, model="m")
    monkeypatch.setattr(overnight, "BUILDS_FILE", tmp_path / "builds.json")
    line = overnight.banner(tmp_path)
    assert "job-1" in line and "3/10" in line and "spill resume job-1" in line
    assert "\n" not in line


def test_banner_prefers_interrupted_build(tmp_path, monkeypatch):
    monkeypatch.setattr(overnight, "BUILDS_FILE", tmp_path / "builds.json")
    f = tmp_path / "bank"
    (f / ".build").mkdir(parents=True)
    (f / ".build" / "state.json").write_text(json.dumps({"name": "bank", "finished": False, "stages": [
        {"status": "done"}, {"status": "interrupted"}, {"status": "pending"}]}))
    overnight.register_build(f)
    line = overnight.banner(tmp_path / "jobs")
    assert "build bank is interrupted at stage 2/3" in line and f"spill resume {f.resolve()}" in line
    assert overnight.banner(tmp_path / "none") == line
    (f / ".build" / "state.json").write_text(json.dumps({"name": "bank", "finished": True, "stages": [{"status": "done"}]}))
    assert overnight.banner(tmp_path / "jobs") is None
