"""Hardware probe. Writes state/hardware.json. Hardware is probed, never declared."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

from .registry import REPO_ROOT

STATE_DIR = REPO_ROOT / "state"
HARDWARE_JSON = STATE_DIR / "hardware.json"

GIB = 1024**3


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _sysctl(key: str) -> str:
    return _run(["sysctl", "-n", key]).strip()


def _probe_gpu() -> dict:
    """GPU vendor/model/VRAM via nvidia-smi, rocm-smi, or Apple unified memory."""
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        if out.strip():
            name, mem = [x.strip() for x in out.strip().splitlines()[0].split(",")]
            # bf16 compute on NVIDIA requires Ampere (SM80) or newer.
            cc = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]).strip()
            bf16 = bool(cc) and float(cc.splitlines()[0]) >= 8.0
            return {"vendor": "nvidia", "model": name, "vram_bytes": int(mem) * 1024**2, "bf16_compute": bf16}
    if shutil.which("rocm-smi"):
        out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
        try:
            data = json.loads(out)
            card = next(iter(data.values()))
            return {
                "vendor": "amd",
                "model": card.get("Card series", "unknown"),
                "vram_bytes": int(card.get("VRAM Total Memory (B)", 0)),
                "bf16_compute": True,  # CDNA/RDNA3+; refined per-arch if needed
            }
        except (json.JSONDecodeError, StopIteration, ValueError):
            pass
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        chip = _sysctl("machdep.cpu.brand_string")
        total_ram = int(_sysctl("hw.memsize"))
        # Apple Silicon: GPU shares unified memory. Metal's default working-set
        # limit is ~75% of RAM (recommendedMaxWorkingSetSize).
        vram = int(total_ram * 0.75)
        # All Apple Silicon GPUs (M1+) support bf16 in Metal as of macOS 14 / M2;
        # M3/M4 have native bf16. llama.cpp Metal runs bf16 GGUFs on all of them.
        return {"vendor": "apple", "model": chip, "vram_bytes": vram, "bf16_compute": True,
                "unified_memory": True}
    return {"vendor": "none", "model": None, "vram_bytes": 0, "bf16_compute": False}


def _cpu_name() -> str:
    name = _sysctl("machdep.cpu.brand_string")
    if name:
        return name
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _ram_total() -> int:
    n = int(_sysctl("hw.memsize") or 0)
    if n:
        return n
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):     # Windows
        try:
            import ctypes

            class _Mem(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("p", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong),
                            ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                            ("e", ctypes.c_ulonglong)]
            m = _Mem()
            m.l = ctypes.sizeof(_Mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.total)
        except Exception:
            return 0


def _free_ram_bytes() -> int:
    if platform.system() == "Darwin":
        out = _run(["vm_stat"])
        page = int(re.search(r"page size of (\d+)", out).group(1))
        pages = {}
        for line in out.splitlines():
            m = re.match(r"Pages (free|inactive|speculative|purgeable):\s+(\d+)", line)
            if m:
                pages[m.group(1)] = int(m.group(2))
        return (pages.get("free", 0) + pages.get("inactive", 0) + pages.get("speculative", 0)) * page
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _disk_info(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    device = None
    if platform.system() == "Darwin":
        out = _run(["df", str(path)])
        lines = out.strip().splitlines()
        if len(lines) > 1:
            device = lines[1].split()[0]
    return {"free_bytes": usage.free, "total_bytes": usage.total, "device": device}


def _measure_seq_read(path: Path, size_bytes: int = 4 * GIB) -> dict:
    """Write a 4 GB file, drop page cache where the OS permits, read it sequentially."""
    test = path / ".streamweights-probe.bin"
    free = shutil.disk_usage(path).free
    if free < size_bytes + 20 * GIB:
        size_bytes = max(1 * GIB, free // 4)
    cache_dropped = False
    try:
        try:
            import fcntl
        except ImportError:                      # Windows
            fcntl = None
        with open(test, "wb") as f:
            # F_NOCACHE on the write fd keeps these pages out of the unified
            # buffer cache, so the read below hits the device, not RAM.
            if fcntl is not None and hasattr(fcntl, "F_NOCACHE"):
                fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
            chunk = os.urandom(64 * 1024**2)
            written = 0
            while written < size_bytes:
                f.write(chunk)
                written += len(chunk)
            f.flush()
            os.fsync(f.fileno())
        if platform.system() == "Darwin":
            # purge requires root; try it, fall back to F_NOCACHE on the read fd.
            r = subprocess.run(["purge"], capture_output=True, timeout=120)
            cache_dropped = r.returncode == 0
        elif platform.system() == "Linux":
            try:
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("3")
                cache_dropped = True
            except OSError:
                pass
        fd = os.open(test, os.O_RDONLY)
        try:
            if not cache_dropped:
                if fcntl is not None and hasattr(fcntl, "F_NOCACHE"):
                    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                    cache_dropped = True  # per-fd cache bypass on macOS
            buf_size = 32 * 1024**2
            t0 = time.monotonic()
            read = 0
            while True:
                b = os.read(fd, buf_size)
                if not b:
                    break
                read += len(b)
            elapsed = time.monotonic() - t0
        finally:
            os.close(fd)
        return {
            "bytes_read": read,
            "seconds": round(elapsed, 3),
            "bytes_per_sec": int(read / elapsed),
            "gb_per_sec": round(read / elapsed / GIB, 2),
            "page_cache_dropped": cache_dropped,
        }
    finally:
        test.unlink(missing_ok=True)


def probe(fast: bool = False) -> dict:
    """Run the full probe and write state/hardware.json. fast=True reuses a prior disk measurement."""
    prior = None
    if HARDWARE_JSON.exists():
        try:
            prior = json.loads(HARDWARE_JSON.read_text())
        except json.JSONDecodeError:
            prior = None
    if fast and prior:
        return prior

    repo_root = REPO_ROOT
    disk = _disk_info(repo_root)
    hw = {
        "probed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "os": f"{platform.system()} {platform.release()} ({platform.mac_ver()[0] or platform.version()})",
        "cpu": _cpu_name(),
        "cpu_cores": os.cpu_count(),
        "ram_total_bytes": _ram_total(),
        "ram_free_bytes": _free_ram_bytes(),
        "gpu": _probe_gpu(),
        "disk": disk,
        "nvme_seq_read": (prior or {}).get("nvme_seq_read") if fast else None,
    }
    if hw["nvme_seq_read"] is None:
        hw["nvme_seq_read"] = _measure_seq_read(repo_root)
    STATE_DIR.mkdir(exist_ok=True)
    HARDWARE_JSON.write_text(json.dumps(hw, indent=2) + "\n")
    return hw


def load(probe_if_missing: bool = True) -> dict:
    if HARDWARE_JSON.exists():
        try:
            return json.loads(HARDWARE_JSON.read_text())
        except json.JSONDecodeError:
            pass
    if probe_if_missing:
        return probe()
    raise FileNotFoundError(HARDWARE_JSON)


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
