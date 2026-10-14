# no-mlx-needed
"""spill plan --deadline: feasible combinations with labeled estimates, nothing downloaded, an
honest refusal when nothing fits, and a build that never lowers a setting to make a deadline."""

import csv
import datetime as dt
import os

import pytest
from typer.testing import CliRunner

from streamweights.cli import app
from streamweights.errors import SpillError
from streamweights.project import deadline as DL
from streamweights.project.init import create_project

from .test_project_build import MODEL

pytestmark = pytest.mark.skipif(
    not (MODEL / "model.safetensors").exists() and os.environ.get("SPILL_REQUIRE_FIXTURE_MODEL") != "1",
    reason="needs the pinned qwen2.5:0.5b")


@pytest.fixture
def proj(tmp_path, monkeypatch):
    p = tmp_path / "d.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "out"])
        for i in range(40):
            w.writerow([f"What is {i}?", f"It is the number {i}, said at length in a sentence."])
    create_project(p, tmp_path / "p", {"input": "text", "output": "out"}, teacher="qwen2.5:1.5b")

    def no_network(*a, **k):
        raise AssertionError("planning must not download or fetch anything")
    monkeypatch.setattr("streamweights.registry.download_safetensors", no_network)
    monkeypatch.setattr("streamweights.download.fetch", no_network)
    monkeypatch.setattr("huggingface_hub.snapshot_download", no_network)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", no_network)
    return tmp_path / "p"


def test_durations_and_timestamps_parse():
    now = dt.datetime(2026, 10, 13, 12, 0)
    assert DL.parse_deadline("90m", now)[0] == 5400
    assert DL.parse_deadline("1d12h", now)[0] == 129600
    assert DL.parse_deadline("1.5h", now)[0] == 5400
    assert DL.parse_deadline("2026-10-13T14:30", now)[0] == 9000
    assert DL.parse_deadline("2026-10-14", now)[0] == 12 * 3600
    for bad in ("tomorrow", "2026-10-13T11:00", "0m", ""):
        with pytest.raises(SpillError):
            DL.parse_deadline(bad, now)


def test_each_combination_is_labeled_and_the_recommendation_follows_its_rule(proj):
    combos = DL.evaluate(proj, 3600, engine="torch-cpu")
    assert combos and all(c.basis in ("measured", "assumed", "unknown") for c in combos)
    rec = [c for c in combos if c.recommended]
    assert len(rec) == 1 and rec[0].fits
    fit = [c for c in combos if c.fits]
    assert (DL._params(rec[0].student), DL._params(rec[0].teacher) if rec[0].teacher else 0) == \
        max((DL._params(c.student), DL._params(c.teacher) if c.teacher else 0) for c in fit)
    assert any(c.configured for c in combos)
    # a model not on this machine carries its download size; nothing was fetched (the fixture raises)
    big = [c for c in combos if c.student == "qwen2.5:7b"]
    assert big and all(("qwen2.5:7b", 15231271888) in c.downloads for c in big)
    text = "\n".join(DL.render(combos, 3600, "1h"))
    assert "recommended" in text and "[assumed]" in text or "[measured]" in text
    assert "downloads qwen2.5:7b 15.2 GB" in text and "download time is not counted" in text


def test_a_tighter_deadline_fits_less_and_none_is_refused_honestly(proj):
    n = lambda s: sum(1 for c in DL.evaluate(proj, s, engine="torch-cpu") if c.fits)
    assert n(86400) > n(600) >= 0
    combos = DL.evaluate(proj, 1, engine="torch-cpu")
    lines = "\n".join(DL.render(combos, 1, "1 s"))
    assert not any(c.fits for c in combos) and not any(c.recommended for c in combos)
    assert "nothing is estimated to fit. The closest:" in lines and "over." in lines
    assert "nothing was lowered to meet the deadline" in lines


def test_build_with_a_deadline_never_starts_or_lowers_anything(proj):
    from streamweights.project import config as C
    before = C.load(proj)
    r = CliRunner().invoke(app, ["build", str(proj), "--deadline", "1s", "--engine", "torch-cpu"])
    assert r.exit_code != 0
    out = r.output
    assert "nothing is estimated to fit" in out and "nothing was lowered" in out
    assert "the build was not started" in out
    assert C.load(proj) == before and not (proj / "runs").exists() and not (proj / ".spill").exists()
    r = CliRunner().invoke(app, ["plan", str(proj), "--deadline", "3h", "--engine", "torch-cpu"])
    assert r.exit_code == 0 and "of " in r.output and "combinations on this machine" in r.output
    assert "next: spill build" in r.output and "--deadline 3h" in r.output
