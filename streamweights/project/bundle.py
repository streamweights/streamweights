"""spill bundle: a project and every asset its workflow needs, to run with no network.

  <bundle>/BUNDLE.json    checksummed manifest of every file, with how to reacquire each model
  <bundle>/README.txt     what this is and how to use it
  <bundle>/project/       the project tree: config, data, schema, runs, export and test records,
                          and for each run a read-only copy of its control object and the
                          payloads that object points at (a consistent committed snapshot,
                          also for a run that is still active)
  <bundle>/assets/models/ the pinned model files: student, teacher, embedding

Python dependencies are NOT in a bundle. Install streamweights and its dependencies first;
the bundle then needs no network. A bundled run is a read-only copy: the original keeps its
own authority, and a training copy forks a new run that names the bundled one as its parent
(`spill build <project> --new-run`), so two places never write the same live run."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from ..errors import SpillError
from ..registry import MODELS_DIR, load_pins, safetensors_dir
from . import config as C
from . import control as ctl
from . import modelid
from .common import atomic_write, sha_file, write_json
from .coordinator import list_runs
from .move import _payload_files, _project_files

README = """This is a streamweights offline bundle.

  spill bundle --verify <this folder>                  check every checksum
  spill bundle --install <this folder> <project dir>   put the models into this machine's cache
                                                       and copy the project out of the bundle

Python dependencies are not included: install streamweights (and the extras you use) first, for
example  pip install git+https://github.com/streamweights/streamweights . After that nothing
needs the network. The runs in project/ are read-only copies: building from a copy forks a new
run (spill build <project dir> --new-run), it never continues the original.
"""


def _asset_dirs(cfg: dict) -> list[tuple[str, str, Path, str]]:
    """(role, tag, source dir, path under assets/models) for every model the workflow uses."""
    out = []
    for role, tag in (("student", cfg["model"]["student"]), ("teacher", cfg["model"].get("teacher"))):
        if tag and tag in load_pins():
            out.append((role, tag, safetensors_dir(tag), f"{tag.replace(':', '-')}/bf16-st"))
        elif tag:
            ident = modelid.student_identity(tag, fetch=False)
            out.append((role, tag, Path(ident["dir"]), f"custom/{tag.replace('/', '__')}"))
    if cfg["task"]["type"] == "classification":
        out.append(("embedding", modelid.EMBEDDING_TAG, modelid.embedding_dir(), "embedding-minilm"))
    return out


def create_bundle(project: Path, dest: Path, say=print, with_exports: bool = False) -> dict:
    project, dest = Path(project).resolve(), Path(dest).resolve()
    cfg = C.load(project)
    if cfg["project"].get("layout") != "project":
        raise SpillError(f"{project.name} is a legacy flat folder; bundle bundles guided projects",
                         f"spill init <data> --input <col> --output <col>")
    if dest.exists() and any(dest.iterdir()):
        raise SpillError(f"{dest} already exists and is not empty", "choose a new path")
    docs = {d["run_id"]: d for d in list_runs(project)}
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "project").mkdir(parents=True)
    files = {}
    rels = _project_files(project, with_exports)
    for rid, d in docs.items():
        rels += _payload_files(project, rid, d)
    for rel in sorted(set(rels)):
        dst = tmp / "project" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / rel, dst)
    for rid, d in docs.items():                    # read-only copies of the control objects
        c = dict(d)
        if c["status"] not in (ctl.COMPLETED,):
            c.update(status=ctl.BUNDLED, owner=None, lease_expires=0.0, quiesce=None)
        c["history"] = (c.get("history", []) + [{"rev": c["revision"] + 1, "event": "bundled",
                                                 "at": time.time()}])[-40:]
        c["revision"] += 1
        p = tmp / "project" / ".spill" / "runs" / rid
        p.mkdir(parents=True, exist_ok=True)
        atomic_write(p / "control.json", (json.dumps(c, indent=1, sort_keys=True) + "\n").encode())
    assets = {}
    for role, tag, src, under in _asset_dirs(cfg):
        pin = load_pins().get(tag)
        if pin:
            have = modelid.verify_dir(tag, src)          # refuses a missing or changed file
            names = list(have)
        else:
            names = list(modelid.hash_dir(src))
        say(f"bundling {role} {tag}: {len(names)} files, "
            f"{sum((src / n).stat().st_size for n in names) / 1e6:.0f} MB")
        for n in names:
            dst = tmp / "assets" / "models" / under / n
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / n, dst)
        assets[tag] = {"role": role, "under": f"assets/models/{under}",
                       "repo": pin["repo"] if pin else None,
                       "revision": pin["revision"] if pin else None,
                       "reacquire": (f"huggingface-hub snapshot_download('{pin['repo']}', "
                                     f"revision='{pin['revision']}')" if pin else "local directory"),
                       "files": names}
    atomic_write(tmp / "README.txt", README.encode())
    for p in sorted(tmp.rglob("*")):
        if p.is_file():
            files[p.relative_to(tmp).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha_file(p)}
    from importlib import metadata
    try:
        ver = metadata.version("streamweights")
    except metadata.PackageNotFoundError:
        ver = "dev"
    manifest = {"schema": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "streamweights": ver, "project": cfg["project"]["name"],
                "runs": {rid: {"status": d["status"], "bundled_as":
                               ctl.COMPLETED if d["status"] == ctl.COMPLETED else ctl.BUNDLED,
                               "checkpoint_step": (d.get("checkpoint") or {}).get("step")}
                         for rid, d in docs.items()},
                "assets": assets, "files": files,
                "dependencies": "not included: install streamweights and its dependencies "
                                "separately; after that no network is needed",
                "credentials": "none are stored in a bundle"}
    write_json(tmp / "BUNDLE.json", manifest)
    tmp.rename(dest)
    return manifest


def verify_bundle(path: Path) -> dict:
    """Every file in the manifest present with its size and sha256; nothing extra required.
    Raises SpillError naming the first problems."""
    path = Path(path)
    try:
        m = json.loads((path / "BUNDLE.json").read_text())
    except (OSError, ValueError):
        raise SpillError(f"{path} has no readable BUNDLE.json", "spill bundle <project> <path>")
    bad = []
    for rel, meta in m["files"].items():
        f = path / rel
        if not f.exists():
            bad.append(f"missing {rel}")
        elif f.stat().st_size != meta["bytes"] or sha_file(f) != meta["sha256"]:
            bad.append(f"corrupted {rel}")
    if bad:
        raise SpillError(f"the bundle is not intact: {'; '.join(bad[:5])}"
                         + (f" (and {len(bad) - 5} more)" if len(bad) > 5 else ""),
                         "copy the bundle again, or rebuild it with spill bundle <project> <path>")
    return m


def install_bundle(path: Path, project_dest: Path, say=print) -> Path:
    """Verify the bundle, put its model files into this machine's cache (never replacing a
    file that already verifies), and copy the project out so it can be built from."""
    path = Path(path)
    m = verify_bundle(path)
    for tag, a in m["assets"].items():
        pin = load_pins().get(tag)
        if pin:
            target = MODELS_DIR / a["under"].split("assets/models/", 1)[1]
        else:
            target = MODELS_DIR / a["under"].split("assets/models/", 1)[1]
        for n in a["files"]:
            src = path / a["under"] / n
            dst = target / n
            if dst.exists() and dst.stat().st_size == src.stat().st_size and sha_file(dst) == sha_file(src):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        if pin:
            modelid.verify_dir(tag, target)
        say(f"installed {tag} -> {target}")
    project_dest = Path(project_dest)
    if project_dest.exists() and any(project_dest.iterdir()):
        raise SpillError(f"{project_dest} already exists and is not empty", "choose a new folder")
    shutil.copytree(path / "project", project_dest, dirs_exist_ok=True)
    return project_dest
