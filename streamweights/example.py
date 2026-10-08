"""spill example <name>: a folder of ready files to build from.

banking77: intent classification over the public PolyAI BANKING77 customer-service
queries (cc-by-4.0, which permits redistribution with attribution; the folder's README
carries the attribution). The files are shipped with the package; nothing is downloaded.

  --quick   100 evals, 500 training rows, student qwen2.5:0.5b
  --tiny    20 evals, 100 training rows over 10 intents, short prompts: a whole build in a CI job

relay: ./relay/, the tiny banking77 files plus a README.md and relay.sh, which starts a build,
stops it partway through the tune stage and finishes it on a second machine (or a second
engine), then prints the table with the machine, OS and engine behind each stage.
"""

from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path

from .errors import SpillError

DATA = Path(__file__).parent / "data" / "examples"
EXAMPLES = {
    "banking77": {
        "about": "77 banking intents; 300 exam rows, 2,000 labeled rows, 2,000 unlabeled prompts",
        "quick": {"student": "qwen2.5:0.5b"},
        "tiny": {"student": "qwen2.5:0.5b", "epochs": 2},
    },
    "snips": {
        "about": "three assistant intents; turn a request into a JSON object (structured "
                 "extraction); 1,200 rows",
        "quick": {}, "tiny": {},
    },
    "relay": {
        "about": "the tiny banking77 build, started on one machine and finished on another",
    },
}


def names() -> list[str]:
    return sorted(EXAMPLES)


def _copy(src: Path, folder: Path, files: tuple[str, ...]) -> None:
    for f in files:
        shutil.copyfile(src / f, folder / f)


def create(name: str, quick: bool = False, parent: Path = Path("."), force: bool = False,
           tiny: bool = False) -> Path:
    if name not in EXAMPLES:
        raise SpillError(f"no example named '{name}' (available: {', '.join(names())})",
                         "spill example banking77 --quick")
    if quick and tiny:
        raise SpillError("--quick and --tiny are two sizes of the same example; pick one",
                         "spill example banking77 --tiny")
    if name == "relay":
        return _create_relay(Path(parent), force)
    if name == "snips":
        return _create_snips(Path(parent), quick, tiny, force)
    src = DATA / name
    folder = Path(parent) / (f"{name}-tiny" if tiny else f"{name}-quick" if quick else name)
    if folder.exists() and any(folder.iterdir()) and not force:
        raise SpillError(f"{folder} already exists and is not empty",
                         f"spill build {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    if tiny or quick:
        q = src / ("tiny" if tiny else "quick")
        _copy(q, folder, ("evals.jsonl", "train.jsonl", "banking77.csv"))
        shutil.copyfile(q / "instructions.txt" if (q / "instructions.txt").exists()
                        else src / "instructions.txt", folder / "instructions.txt")
        settings = EXAMPLES[name]["tiny" if tiny else "quick"]
        (folder / "spill.json").write_text(json.dumps(settings, indent=2) + "\n")
        readme = q / "README.md"
    else:
        _copy(src, folder, ("evals.jsonl", "train.jsonl", "prompts.jsonl", "instructions.txt",
                            "banking77.csv"))
        readme = src / "README.md"
    if readme.exists():
        shutil.copyfile(readme, folder / "README.md")
    return folder


def _create_snips(parent: Path, quick: bool, tiny: bool, force: bool) -> Path:
    if quick and tiny:
        raise SpillError("--quick and --tiny are two sizes of the same example; pick one",
                         "spill example snips --tiny")
    src = DATA / "snips"
    folder = Path(parent) / ("snips-tiny" if tiny else "snips-quick" if quick else "snips")
    if folder.exists() and any(folder.iterdir()) and not force:
        raise SpillError(f"{folder} already exists and is not empty",
                         f"spill init {folder}/snips.csv --input text --output json")
    folder.mkdir(parents=True, exist_ok=True)
    q = src / ("tiny" if tiny else "quick") if (tiny or quick) else src
    _copy(q, folder, ("snips.csv", "schema.json", "README.md", "LICENSE-CC0.txt"))
    return folder


def _create_relay(parent: Path, force: bool) -> Path:
    folder = parent / "relay"
    if folder.exists() and any(folder.iterdir()) and not force:
        raise SpillError(f"{folder} already exists and is not empty", f"{folder}/relay.sh")
    folder.mkdir(parents=True, exist_ok=True)
    tiny = DATA / "banking77" / "tiny"
    _copy(tiny, folder, ("evals.jsonl", "train.jsonl", "instructions.txt"))
    (folder / "spill.json").write_text(
        json.dumps(EXAMPLES["banking77"]["tiny"], indent=2) + "\n")
    src = DATA / "relay"
    shutil.copyfile(src / "README.md", folder / "README.md")
    script = folder / "relay.sh"
    shutil.copyfile(src / "relay.sh", script)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return folder
