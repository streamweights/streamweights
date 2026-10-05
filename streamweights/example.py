"""spill example <name>: a folder of ready files to build from.

banking77: intent classification over the public PolyAI BANKING77 customer-service
queries (cc-by-4.0, which permits redistribution with attribution; the folder's README
carries the attribution). The files are shipped with the package; nothing is downloaded.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .errors import SpillError

DATA = Path(__file__).parent / "data" / "examples"
EXAMPLES = {
    "banking77": {
        "about": "77 banking intents; 300 exam rows, 2,000 labeled rows, 2,000 unlabeled prompts",
        "quick": {"student": "qwen2.5:0.5b"},
    },
}


def names() -> list[str]:
    return sorted(EXAMPLES)


def create(name: str, quick: bool = False, parent: Path = Path("."), force: bool = False
           ) -> Path:
    if name not in EXAMPLES:
        raise SpillError(f"no example named '{name}' (available: {', '.join(names())})",
                         "spill example banking77 --quick")
    src = DATA / name
    folder = Path(parent) / (f"{name}-quick" if quick else name)
    if folder.exists() and any(folder.iterdir()) and not force:
        raise SpillError(f"{folder} already exists and is not empty",
                         f"spill build {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    if quick:
        q = src / "quick"
        for f in ("evals.jsonl", "train.jsonl"):
            shutil.copyfile(q / f, folder / f)
        shutil.copyfile(src / "instructions.txt", folder / "instructions.txt")
        (folder / "spill.json").write_text(json.dumps(EXAMPLES[name]["quick"], indent=2) + "\n")
        readme = q / "README.md"
    else:
        for f in ("evals.jsonl", "train.jsonl", "prompts.jsonl", "instructions.txt"):
            shutil.copyfile(src / f, folder / f)
        readme = src / "README.md"
    if readme.exists():
        shutil.copyfile(readme, folder / "README.md")
    return folder
