# no-mlx-needed
"""Offline bundles and the portable project: checksums, reacquisition, missing and corrupted
assets, read-only copies of runs, and flat-layout migration."""

import json
import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from streamweights.cli import app
from streamweights.errors import SpillError
from streamweights.project import bundle as BU
from streamweights.project import config as C
from streamweights.project import coordinator as CO
from streamweights.project import migrate as MG
from streamweights.project import modelid
from streamweights.project import plan as P
from streamweights.project.common import read_json, sha_file
from streamweights.project.runstate import RunState, tree_digest

from .test_project_build import FakeExecutor, make_project, pytestmark  # noqa: F401


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bundle")
    proj = make_project(tmp)
    plan = P.make_plan(proj, engine="torch-cpu")
    done = CO.Coordinator(proj, FakeExecutor(), "torch-cpu", say=lambda s: None).build(plan)
    # a second, unfinished run (stopped after a committed checkpoint) in the same project
    cfg = C.load(proj)
    cfg["training"]["epochs"] = 3.0
    C.save(proj, cfg)
    plan2 = P.make_plan(proj, engine="torch-cpu")
    CO.Coordinator(proj, FakeExecutor(interrupt_train_at=5), "torch-cpu", say=lambda s: None).build(plan2)
    saved = os.environ.get("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_SECRET_ACCESS_KEY"] = "sekrit-do-not-bundle-123"
    try:
        m = BU.create_bundle(proj, tmp / "out.bundle", say=lambda s: None)
    finally:
        if saved is None:
            os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
        else:
            os.environ["AWS_SECRET_ACCESS_KEY"] = saved
    return proj, tmp / "out.bundle", m, done


def test_bundle_has_a_checksummed_manifest_the_project_and_every_pinned_model_file(made):
    proj, b, m, done = made
    assert (b / "BUNDLE.json").exists() and (b / "README.txt").exists()
    assert "dependencies" in m and "not included" in m["dependencies"]
    assert set(m["assets"]) == {"qwen2.5:0.5b", "embedding:minilm"}
    assert m["assets"]["qwen2.5:0.5b"]["revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert all((b / "assets/models/qwen2.5-0.5b/bf16-st" / n).exists() for n in m["assets"]["qwen2.5:0.5b"]["files"])
    assert (b / "project" / "streamweights.toml").exists() and (b / "project" / "data" / "train.jsonl").exists()
    assert BU.verify_bundle(b)["streamweights"]
    blob = "".join(p.read_text(errors="ignore") for p in b.rglob("*")
                   if p.is_file() and p.stat().st_size < 1_000_000)
    assert "sekrit-do-not-bundle-123" not in blob


def test_an_active_run_is_bundled_as_a_read_only_committed_copy(made):
    proj, b, m, done = made
    states = {rid: r["bundled_as"] for rid, r in m["runs"].items()}
    assert sorted(states.values()) == ["bundled", "completed"]
    active = next(rid for rid, s in states.items() if s == "bundled")
    c = read_json(b / "project" / ".spill" / "runs" / active / "control.json")
    assert c["status"] == "bundled" and c["owner"] is None and c["checkpoint"]["step"] == 4
    assert (b / "project" / ".spill" / "runs" / active / c["checkpoint"]["dir"] / "PAYLOAD.json").exists()
    with pytest.raises(SpillError) as e:
        RunState.open(str(b / "project" / ".spill"), active, b / "w").acquire()
    assert "read-only copy" in e.value.message
    assert read_json(proj / ".spill" / "runs" / active / "control.json")["status"] == "idle"   # the original keeps its authority


def test_verify_rejects_a_missing_or_corrupted_asset(made, tmp_path):
    proj, b, m, done = made
    copy = tmp_path / "b"
    shutil.copytree(b, copy, symlinks=True)
    BU.verify_bundle(copy)
    tok = copy / "assets/models/qwen2.5-0.5b/bf16-st/tokenizer.json"
    tok.write_bytes(tok.read_bytes()[:-5] + b"xxxxx")
    with pytest.raises(SpillError) as e:
        BU.verify_bundle(copy)
    assert "corrupted assets/models/qwen2.5-0.5b/bf16-st/tokenizer.json" in e.value.message
    shutil.copyfile(b / "assets/models/qwen2.5-0.5b/bf16-st/tokenizer.json", tok)
    (copy / "assets/models/embedding-minilm/model.safetensors").unlink()
    with pytest.raises(SpillError) as e:
        BU.verify_bundle(copy)
    assert "missing assets/models/embedding-minilm/model.safetensors" in e.value.message


def test_install_into_a_fresh_cache_checks_the_pinned_hashes_and_forks_a_new_run(made, tmp_path, monkeypatch):
    proj, b, m, done = made
    cache = tmp_path / "fresh-models"
    monkeypatch.setattr(BU, "MODELS_DIR", cache)
    monkeypatch.setattr(modelid, "MODELS_DIR", cache)
    out = BU.install_bundle(b, tmp_path / "copy", say=lambda s: None)
    assert (cache / "qwen2.5-0.5b" / "bf16-st" / "model.safetensors").exists()
    assert (cache / "embedding-minilm" / "model.safetensors").exists()
    modelid.verify_dir("qwen2.5:0.5b", cache / "qwen2.5-0.5b" / "bf16-st")
    modelid.verify_dir("embedding:minilm", cache / "embedding-minilm")
    # the bundled copies cannot be continued; building from the copy forks a new run
    runs = {d["run_id"]: d for d in CO.list_runs(out)}
    assert sorted(d["status"] for d in runs.values()) == ["bundled", "completed"]
    plan = P.make_plan(out, engine="torch-cpu")
    res = CO.Coordinator(out, FakeExecutor(), "torch-cpu", say=lambda s: None).build(plan, new_run=True)
    assert res.status == "completed" and res.run_id not in runs
    assert read_json(res.snapshot / "manifest.json")["parent"] in runs
    for rid, d in runs.items():                          # the bundled originals are untouched
        assert read_json(out / ".spill" / "runs" / rid / "control.json")["status"] == d["status"]


def test_reacquisition_verifies_a_pinned_directory_and_rejects_a_changed_file(tmp_path):
    src = modelid.embedding_dir()
    if not (src / "model.safetensors").exists():
        pytest.skip("the embedding model is fetched by the first plan/build")
    d = tmp_path / "emb"
    shutil.copytree(src, d)
    modelid.verify_dir("embedding:minilm", d)
    (d / "vocab.txt").write_text("changed")
    with pytest.raises(SpillError) as e:
        modelid.verify_dir("embedding:minilm", d)
    assert "does not match the pinned revision" in e.value.message
    (d / "vocab.txt").unlink()
    (d / "model.safetensors").unlink()
    with pytest.raises(SpillError) as e:
        modelid.verify_dir("embedding:minilm", d)
    assert "model.safetensors is missing" in e.value.message


def test_bundle_cli_roundtrip_prints_the_next_command(made, tmp_path):
    proj, b, m, done = made
    r = CliRunner().invoke(app, ["bundle", "--verify", str(b)])
    assert r.exit_code == 0 and "every checksum matches" in r.output and "next: spill bundle --install" in r.output
    r = CliRunner().invoke(app, ["bundle", str(proj), str(b)])
    assert r.exit_code == 1 and "already exists" in r.output


# ---------------------------------------------------------------- flat-layout migration

def legacy_folder(tmp_path):
    f = tmp_path / "flat"
    f.mkdir()
    (f / "evals.jsonl").write_text('{"prompt": "a", "expected": "x"}\n')
    (f / "train.jsonl").write_text('{"prompt": "b", "answer": "y"}\n')
    (f / ".build").mkdir()
    (f / ".build" / "state.json").write_text('{"finished": false}')
    return f


def test_migration_is_one_line_non_destructive_and_idempotent(tmp_path):
    f = legacy_folder(tmp_path)
    before = {p.name: p.read_bytes() for p in f.iterdir() if p.is_file()}
    said = []
    cfg = MG.ensure_project(f, say=said.append)
    assert len(said) == 1 and "migrated flat to streamweights.toml" in said[0] and "were not changed" in said[0]
    assert cfg["project"]["layout"] == "legacy-flat" and cfg["schema_version"] == 1
    toml1 = (f / "streamweights.toml").read_bytes()
    assert MG.ensure_project(f, say=said.append) and len(said) == 1          # nothing said the second time
    assert (f / "streamweights.toml").read_bytes() == toml1
    after = {p.name: p.read_bytes() for p in f.iterdir() if p.is_file() and p.name != "streamweights.toml"}
    assert after == before and (f / ".build" / "state.json").read_text() == '{"finished": false}'
    assert not MG.is_guided(cfg)


def test_a_migrated_folder_still_builds_the_legacy_way_and_guided_commands_say_so(tmp_path):
    f = legacy_folder(tmp_path)
    MG.ensure_project(f)
    r = CliRunner().invoke(app, ["build", str(f), "--table"])
    assert r.exit_code == 0 and "stage  engine" in r.output               # the legacy path answered
    r = CliRunner().invoke(app, ["plan", str(f)])
    assert r.exit_code == 1 and "flat-layout folder" in r.output and "keep working" in r.output
    assert MG.ensure_project(tmp_path / "nothing") is None


def test_a_future_schema_version_is_rejected_clearly(tmp_path):
    (tmp_path / "streamweights.toml").write_text('schema_version = 7\n[project]\nname = "x"\n')
    r = CliRunner().invoke(app, ["plan", str(tmp_path)])
    assert r.exit_code == 1 and "schema_version 7" in r.output and "up to 1" in r.output
