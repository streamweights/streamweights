# no-mlx-needed
"""Student, teacher and judge are choices: any curated tag or Hugging Face repo in a supported
family, recorded in streamweights.toml and shown by plan with sizes, downloads and disk; an
unsupported architecture is refused in one line naming the family. No weights are fetched: the
Hub is replaced by two small config files."""

import csv
import json
import os

import pytest

from streamweights import resolve
from streamweights.errors import SpillError
from streamweights.project import config as C
from streamweights.project import plan as P
from streamweights.project.init import create_project

from .test_project_build import MODEL

pytestmark = pytest.mark.skipif(
    not (MODEL / "model.safetensors").exists() and os.environ.get("SPILL_REQUIRE_FIXTURE_MODEL") != "1",
    reason="needs the pinned qwen2.5:0.5b")

CONFIGS = {"acme/tiny-llama": {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
           "acme/moe": {"model_type": "mixtral", "architectures": ["MixtralForCausalLM"]},
           "acme/vision": {"model_type": "llava", "architectures": ["LlavaForConditionalGeneration"]}}


@pytest.fixture(autouse=True)
def hub(tmp_path, monkeypatch):
    def fake_download(repo, filename, revision=None, **kw):
        if repo not in CONFIGS:
            from huggingface_hub.errors import RepositoryNotFoundError

            class _R:                       # the Hub's 404, as far as the error class needs it
                status_code, headers, text, request = 404, {}, "", None
            raise RepositoryNotFoundError(repo, response=_R())
        p = tmp_path / (repo.replace("/", "__") + "-config.json")
        p.write_text(json.dumps(CONFIGS[repo]))
        return str(p)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(resolve, "_st_total_bytes", lambda repo, rev: 4_000_000_000)


def project(tmp_path, **roles):
    p = tmp_path / "d.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "out"])
        for i in range(40):
            w.writerow([f"What is {i}?", f"It is the number {i}, said at length in a sentence."])
    return create_project(p, tmp_path / "p", {"input": "text", "output": "out"}, **roles)


def test_all_three_roles_take_tags_and_repos_and_are_recorded_and_shown(tmp_path):
    res = project(tmp_path, student="qwen2.5:0.5b", teacher="acme/tiny-llama", judge="qwen2.5:1.5b")
    toml = C.load(tmp_path / "p")
    assert toml["model"]["student"] == "qwen2.5:0.5b"
    assert toml["model"]["teacher"] == "acme/tiny-llama" and toml["model"]["judge"] == "qwen2.5:1.5b"
    assert toml["model"]["student_why"] == "chosen with --student"
    plan = P.make_plan(tmp_path / "p", engine="torch-cpu")
    text = "\n".join(P.render(plan))
    assert "student   qwen2.5:0.5b" in text and "on disk" in text
    assert "teacher   acme/tiny-llama  acme/tiny-llama@unpinned" in text and "3.73 GB to download" in text
    assert "judge     qwen2.5:1.5b" in text and "named by you" in text
    assert {d["model"]: d["bytes"] for d in plan.downloads}["acme/tiny-llama"] == 4_000_000_000
    assert "downloads:" in text and "disk needed" in text
    assert any(s.id == "judge" for s in plan.stages) and plan.stages[0].id == "distill"


def test_default_student_states_its_reason(tmp_path):
    res = project(tmp_path)
    assert res.cfg["model"]["student"] == "qwen2.5:0.5b"
    why = C.load(tmp_path / "p")["model"]["student_why"]
    assert why.startswith("qwen2.5:0.5b on ") and "smallest curated model" in why
    assert why in "\n".join(P.render(P.make_plan(tmp_path / "p", engine="torch-cpu")))


@pytest.mark.parametrize("role", ["student", "teacher", "judge"])
def test_unsupported_architectures_are_refused_in_one_line_naming_the_family(tmp_path, role):
    project(tmp_path, **{role: "acme/moe"})
    with pytest.raises(SpillError) as e:
        P.make_plan(tmp_path / "p", engine="torch-cpu")
    line = e.value.line()
    assert "\n" not in line and "Mixtral (MoE)" in line and "not_yet" in line
    assert line.endswith("Try: spill models --architectures")


def test_multimodal_and_unknown_repos_are_refused_too(tmp_path):
    project(tmp_path, teacher="acme/vision")
    with pytest.raises(SpillError) as e:
        P.make_plan(tmp_path / "p", engine="torch-cpu")
    assert "multimodal" in e.value.line() and "not_yet" in e.value.line()
    cfg = C.load(tmp_path / "p")
    cfg["model"]["teacher"] = "acme/missing"
    C.save(tmp_path / "p", cfg)
    with pytest.raises(SpillError) as e:
        P.make_plan(tmp_path / "p", engine="torch-cpu")
    assert "no Hugging Face repo named 'acme/missing'" in e.value.line()
