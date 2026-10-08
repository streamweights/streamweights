"""An optional allow-list for model access, used by the docs-command harness and by CI.

With SPILL_ALLOWED_MODELS set (comma-separated curated tags and Hugging Face repo ids), any
attempt to resolve, download or load another model fails with one line, before a byte is
fetched. With it unset (the normal case) nothing is checked."""

from __future__ import annotations

import os

from .errors import SpillError


def allowed() -> set[str] | None:
    raw = os.environ.get("SPILL_ALLOWED_MODELS")
    return {x.strip() for x in raw.split(",") if x.strip()} if raw else None


def check(model: str) -> None:
    ok = allowed()
    if ok is None:
        return
    base = model.partition("@")[0]
    base = base.partition("+")[0]
    if base in ok:
        return
    raise SpillError(f"{model} is not approved here (SPILL_ALLOWED_MODELS lists "
                     f"{', '.join(sorted(ok))}); nothing was downloaded or loaded",
                     "run it on a machine without the allow-list, or name an approved model")
