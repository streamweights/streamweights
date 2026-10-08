"""Opening a flat-layout folder (evals.jsonl, train.jsonl, prompts.jsonl, .build/) with the
project commands.

Migration is one thing: a streamweights.toml that records the folder as `layout =
"legacy-flat"`, with the hash of each source file. It is non-destructive (no source file is
touched or moved), idempotent (a second call changes nothing and says nothing) and announced
in one line. Old checkpoints are not reinterpreted: `.build/` is carried as it is, and
`spill build` and `spill resume` keep using it exactly as before."""

from __future__ import annotations

from pathlib import Path

from . import config as C
from .common import now, sha_file


def is_legacy(folder: Path) -> bool:
    f = Path(folder)
    return f.is_dir() and not C.exists(f) and (f / "evals.jsonl").exists()


def ensure_project(folder: str | Path, say=lambda s: None) -> dict | None:
    """The config of `folder`; for a legacy flat folder, migrate it first. None when the
    folder is neither."""
    f = Path(folder)
    if C.exists(f):
        return C.load(f)
    if not is_legacy(f):
        return None
    sources = {n: sha_file(f / n) for n in (*C.LEGACY_FILES, "instructions.txt", "spill.json")
               if (f / n).exists()}
    cfg = {"schema_version": C.SCHEMA_VERSION,
           "project": {"name": f.name, "layout": "legacy-flat", "created": now()},
           "legacy": {"sources": sources, "state": ".build" if (f / ".build").exists() else "",
                      "note": "flat layout: spill build and spill resume keep working unchanged; "
                              ".build/ is not reinterpreted"}}
    C.save(f, cfg)
    say(f"migrated {f.name} to streamweights.toml (schema {C.SCHEMA_VERSION}, layout legacy-flat); "
        f"your files and .build/ were not changed")
    return cfg


def is_guided(cfg: dict | None) -> bool:
    return bool(cfg) and cfg.get("project", {}).get("layout") == "project"
