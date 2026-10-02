"""Fetch the pinned llama.cpp prebuilt release into bin/ on first use.
Backends are upstream llama.cpp unmodified; we never vendor or build source."""

from __future__ import annotations

import platform
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from .registry import REPO_ROOT

BIN_DIR = REPO_ROOT / "bin"
LLAMA_SERVER = BIN_DIR / "llama-server"

RELEASE = "b11311"
ASSETS = {
    ("Darwin", "arm64"): f"llama-{RELEASE}-bin-macos-arm64.tar.gz",
    ("Darwin", "x86_64"): f"llama-{RELEASE}-bin-macos-x64.tar.gz",
    ("Linux", "x86_64"): f"llama-{RELEASE}-bin-ubuntu-x64.tar.gz",
}


def ensure_llama_server() -> Path:
    if LLAMA_SERVER.exists():
        return LLAMA_SERVER
    key = (platform.system(), platform.machine())
    asset = ASSETS.get(key)
    if not asset:
        raise RuntimeError(f"no prebuilt llama.cpp asset known for {key}; "
                           f"place a llama-server binary at {LLAMA_SERVER}")
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/{asset}"
    print(f"fetching llama.cpp {RELEASE} prebuilt -> {BIN_DIR}", file=sys.stderr)
    BIN_DIR.mkdir(exist_ok=True)
    tgz = BIN_DIR / asset
    urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(BIN_DIR, filter="data")
    tgz.unlink()
    extracted = BIN_DIR / f"llama-{RELEASE}"
    LLAMA_SERVER.symlink_to(extracted / "llama-server")
    if platform.system() == "Darwin":
        subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(extracted)],
                       capture_output=True)
    return LLAMA_SERVER
