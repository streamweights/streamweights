"""Model argument resolution: curated tag, or any Hugging Face repo id
(org/name[@revision]): fetched, classified, sized, and run through the same
policy and budget as a tag."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .engines.supported import NOT_YET, Family, classify
from .errors import SpillError
from .registry import GIB, MODELS_DIR, load_registry, safetensors_spec


@dataclass
class Resolved:
    kind: str                # "tag" | "hf"
    name: str                # registry tag or org/name
    revision: str | None = None
    config: dict = field(default_factory=dict)
    st_bytes: int = 0
    family: Family | None = None
    local_dir: Path | None = None


def resolve_model(arg: str) -> Resolved:
    from . import guard
    guard.check(arg)
    reg = load_registry()
    if arg in reg:
        return Resolved("tag", arg, st_bytes=safetensors_spec(arg)["bytes"])
    local = Path(arg).expanduser()
    if arg.startswith((".", "/", "~")) and local.is_dir():
        return _resolve_local(local)
    if "/" not in arg:
        raise SpillError(
            f"'{arg}' is not a curated tag or a Hugging Face repo id (org/name)",
            "spill models")
    repo, _, revision = arg.partition("@")
    revision = revision or None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (EntryNotFoundError, GatedRepoError,
                                        RepositoryNotFoundError)
    try:
        cfg_path = hf_hub_download(repo, "config.json", revision=revision)
    except GatedRepoError:
        raise SpillError(
            f"{repo} is gated on Hugging Face; search for an ungated mirror of the same "
            f"weights (community mirrors are common) or set HF_TOKEN in the environment "
            f"(it is honored automatically, never prompted for)")
    except RepositoryNotFoundError:
        raise SpillError(f"no Hugging Face repo named '{repo}'", "spill models")
    except EntryNotFoundError:
        raise SpillError(f"{repo} has no config.json; not a transformers-format model",
                         "spill models")
    config = json.loads(Path(cfg_path).read_text())
    fam = classify(config)
    if fam.state == NOT_YET:
        raise SpillError(
            f"{repo} is {fam.label}, which is '{NOT_YET}' ({fam.note})",
            "spill models --architectures")
    if fam.module is None:
        raise SpillError(
            f"{repo} is {fam.label}, state '{fam.state}' but not yet wired ({fam.note})",
            "spill models --architectures")
    st_bytes = _st_total_bytes(repo, revision)
    local = MODELS_DIR / "hf" / (repo.replace("/", "__") + (f"@{revision}" if revision else ""))
    return Resolved("hf", repo, revision, config, st_bytes, fam, local)


def _resolve_local(p: Path) -> Resolved:
    """A local safetensors model directory (an exported merged model, a downloaded repo)."""
    if not (p / "config.json").exists() or not any(p.glob("*.safetensors")):
        raise SpillError(f"{p} is not a safetensors model directory (needs config.json and "
                         f"*.safetensors)", "spill models")
    config = json.loads((p / "config.json").read_text())
    fam = classify(config)
    if fam.state == NOT_YET or fam.module is None:
        raise SpillError(f"{p.name} is {fam.label}, state '{fam.state}' ({fam.note})",
                         "spill models --architectures")
    size = sum(f.stat().st_size for f in p.glob("*.safetensors"))
    return Resolved("hf", p.name, None, config, size, fam, p.resolve())


def _st_total_bytes(repo: str, revision: str | None) -> int:
    from huggingface_hub import HfApi, hf_hub_download
    try:
        p = hf_hub_download(repo, "model.safetensors.index.json", revision=revision)
        meta = json.loads(Path(p).read_text()).get("metadata", {})
        if meta.get("total_size"):
            return int(meta["total_size"])
    except Exception:
        pass
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    return sum(f.size or 0 for f in info.siblings
               if f.rfilename.endswith(".safetensors"))


def arch_from_config(config: dict) -> dict:
    return {"n_layers": config["num_hidden_layers"],
            "n_kv_heads": config["num_key_value_heads"],
            "head_dim": config.get("head_dim") or
                        config["hidden_size"] // config["num_attention_heads"]}


def download_hf(res: Resolved) -> Path:
    """Snapshot the repo's safetensors under models/hf/ with the 20 GB floor."""
    from .download import fetch
    dest = res.local_dir
    if dest.exists() and (dest / "config.json").exists() and any(dest.glob("*.safetensors")):
        return dest
    return fetch(res.name, dest, ["*.safetensors", "*.json", "*.model"],
                 f"{res.name} bf16 safetensors", revision=res.revision,
                 recovery=f"spill run {res.name} <input> --quant 8bit")
