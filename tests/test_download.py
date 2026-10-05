import errno
import io
from types import SimpleNamespace

import pytest

from streamweights import download as dl
from streamweights.errors import SpillError

GB = 1024**3


def api(files):
    return SimpleNamespace(model_info=lambda repo, revision=None, files_metadata=True: SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in files]))


def test_refuses_before_the_first_byte(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: SimpleNamespace(free=30 * GB))
    with pytest.raises(SpillError, match="nothing was downloaded") as e:
        dl.fetch("o/m", tmp_path / "m", ["*.safetensors"], "m bf16", api=api([("a.safetensors", 15 * GB)]),
                 snapshot=lambda *a, **k: called.append(1), say=lambda s: None)
    assert not called and "free up 5 GB" in str(e.value)


def test_resume_counts_what_is_already_here(tmp_path):
    d = tmp_path / "m"
    (d / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (d / "a.safetensors").write_bytes(b"x" * 100)
    (d / ".cache" / "huggingface" / "download" / "b.safetensors.incomplete").write_bytes(b"y" * 40)
    total, remaining = dl.plan([("a.safetensors", 100), ("b.safetensors", 100), ("c.json", 5)],
                               d, ["*.safetensors"])
    assert total == 200 and remaining == 60


def test_fetch_announces_size_and_transport_and_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: SimpleNamespace(free=500 * GB))
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    said, tries = [], []

    def snap(repo, revision=None, allow_patterns=None, local_dir=None):
        tries.append(1)
        if len(tries) < 3:
            raise ConnectionError("reset")
    dl.fetch("o/m", tmp_path / "m", ["*.safetensors"], "m bf16", api=api([("a.safetensors", 15 * GB)]),
             snapshot=snap, say=said.append)
    assert len(tries) == 3
    assert "15.0 GB to fetch" in said[0] and "via " in said[0]
    assert any("resuming (attempt 2/4)" in s for s in said)


def test_disk_full_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: SimpleNamespace(free=500 * GB))
    tries = []

    def snap(*a, **k):
        tries.append(1)
        raise OSError(errno.ENOSPC, "No space left on device")
    with pytest.raises(OSError):
        dl.fetch("o/m", tmp_path / "m", ["*"], "m", api=api([("a.bin", GB)]), snapshot=snap,
                 say=lambda s: None)
    assert len(tries) == 1


def test_already_complete_is_a_noop(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "a.safetensors").write_bytes(b"x" * 10)
    out = dl.fetch("o/m", d, ["*.safetensors"], "m", api=api([("a.safetensors", 10)]),
                   snapshot=lambda *a, **k: pytest.fail("fetched"), say=lambda s: None)
    assert out == d


def test_progress_line_has_bytes_speed_eta(tmp_path):
    p = dl.Progress(tmp_path, set(), 10 * GB, 0, "m bf16", out=io.StringIO())
    line = p.line(5 * GB, 100e6)
    assert "5.0/10.0 GB" in line and "50%" in line and "100 MB/s" in line and "ETA " in line
    assert dl.fmt_eta(None) == "..." and dl.fmt_eta(30) == "30 s" and dl.fmt_eta(7200) == "2.0 h"
