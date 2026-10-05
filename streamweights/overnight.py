"""Overnight safety: stay awake, say when the power is wrong, say when it is over,
and say on the next launch that something was interrupted.

  caffeinate -i   held for the lifetime of run, distill, tune and build (tied to our pid,
                  so it cannot outlive us)
  battery         `pmset -g batt`; the pre-run line says to plug in
  notify          osascript on finish or stop; --notify <url> also POSTs a small JSON
  banner          any command, on launch, reports an interrupted job with its resume command
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .registry import REPO_ROOT

BUILDS_FILE = REPO_ROOT / "state" / "builds.json"


def _mac() -> bool:
    return platform.system() == "Darwin"


# ------------------------------------------------------------ power

def on_battery() -> bool:
    if not _mac() or not shutil.which("pmset"):
        return False
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True,
                             timeout=3).stdout
    except Exception:
        return False
    return "Battery Power" in out


def battery_note() -> str:
    return (" On battery power: plug in before this runs overnight."
            if on_battery() else "")


@contextlib.contextmanager
def caffeinate():
    """Hold `caffeinate -i` (no idle sleep) for the lifetime of this process."""
    proc = None
    if _mac() and shutil.which("caffeinate"):
        try:
            proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            proc = None
    try:
        yield proc
    finally:
        if proc is not None:
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=2)


# ------------------------------------------------------------ notifications

def notify(title: str, message: str, url: str | None = None, payload: dict | None = None
           ) -> list[str]:
    """macOS notification, plus a POST to `url` when given. Never raises; returns what was
    done (for tests and for the output line)."""
    done = []
    if _mac() and shutil.which("osascript") and not os.environ.get("SPILL_NO_NOTIFY"):
        script = ('display notification "%s" with title "%s"'
                  % (message.replace('"', "'")[:200], title.replace('"', "'")[:80]))
        with contextlib.suppress(Exception):
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
            done.append("osascript")
    if url:
        try:
            import httpx
            httpx.post(url, json={"title": title, "message": message, **(payload or {})},
                       timeout=10)
            done.append(f"POST {url}")
        except Exception as e:                      # a dead webhook must not fail a job
            done.append(f"POST {url} failed ({type(e).__name__})")
    return done


@contextlib.contextmanager
def long_job(name: str, notify_url: str | None = None):
    """Wrap a long command: caffeinate for its lifetime, and a notification when it
    finishes, stops, or fails."""
    t0 = time.monotonic()
    status = {"state": "finished", "detail": ""}
    with caffeinate():
        try:
            yield status
        except KeyboardInterrupt:
            status["state"] = "stopped"
            raise
        except SystemExit as e:
            if e.code not in (0, None):
                status["state"] = "stopped"
            raise
        except BaseException as e:
            status["state"] = "failed"
            status["detail"] = f"{type(e).__name__}: {e}"[:160]
            raise
        finally:
            el = time.monotonic() - t0
            msg = f"{name} {status['state']} after {_dur(el)}" + (
                f" ({status['detail']})" if status["detail"] else "")
            notify("spill", msg, notify_url, {"command": name, "state": status["state"],
                                              "seconds": round(el)})


def _dur(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


# ------------------------------------------------------------ interrupted jobs

def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (ValueError, OSError):
        return False
    return True


def interrupted_jobs(jobs_dir: Path, stale_s: float = 900) -> list[dict]:
    """Jobs that stopped before finishing: status interrupted/failed, or `running` whose
    process is gone (killed, power loss)."""
    out = []
    if not jobs_dir.exists():
        return out
    for d in sorted(jobs_dir.iterdir()):
        mp = d / "meta.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text())
        except json.JSONDecodeError:
            continue
        done, total = m.get("done", 0), m.get("total", 0)
        st = m.get("status")
        if total and done >= total:
            continue
        if st in ("interrupted", "failed"):
            pass
        elif st == "running":
            if m.get("pid"):
                if _pid_alive(m["pid"]):
                    continue
            elif time.time() - mp.stat().st_mtime < stale_s:
                continue
        else:
            continue
        out.append({"id": m.get("id", d.name), "done": done, "total": total,
                    "kind": (m.get("options") or {}).get("kind", "run"),
                    "model": m.get("model", "?")})
    return out


def register_build(folder: Path) -> None:
    BUILDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        cur = json.loads(BUILDS_FILE.read_text()) if BUILDS_FILE.exists() else []
    except json.JSONDecodeError:
        cur = []
    p = str(Path(folder).resolve())
    if p not in cur:
        cur.append(p)
        BUILDS_FILE.write_text(json.dumps(cur))


def interrupted_builds() -> list[dict]:
    out = []
    try:
        cur = json.loads(BUILDS_FILE.read_text()) if BUILDS_FILE.exists() else []
    except json.JSONDecodeError:
        return out
    for p in cur:
        sp = Path(p) / ".build" / "state.json"
        if not sp.exists():
            continue
        try:
            s = json.loads(sp.read_text())
        except json.JSONDecodeError:
            continue
        stages = s.get("stages", [])
        if s.get("finished") or not stages:
            continue
        if s.get("pid") and s["pid"] != os.getpid() and _pid_alive(s["pid"]):
            continue                      # still running, not interrupted
        done = sum(1 for x in stages if x["status"] == "done")
        if done < len(stages):
            out.append({"folder": p, "name": s.get("name"), "done": done,
                        "total": len(stages)})
    return out


def banner(jobs_dir: Path, skip_resume_of: str | None = None) -> str | None:
    """One line, or None. Builds first (they own their jobs), then bare jobs."""
    builds = interrupted_builds()
    if builds:
        b = builds[-1]
        extra = f" (+{len(builds) - 1} more)" if len(builds) > 1 else ""
        return (f"spill: build {b['name']} is interrupted at stage {b['done'] + 1}/"
                f"{b['total']}{extra}; continue with: spill resume {b['folder']}")
    jobs = interrupted_jobs(jobs_dir)
    if jobs:
        j = jobs[-1]
        extra = f" (+{len(jobs) - 1} more)" if len(jobs) > 1 else ""
        return (f"spill: {j['kind']} job {j['id']} is interrupted at {j['done']}/{j['total']} "
                f"rows{extra}; continue with: spill resume {j['id']}")
    return None
