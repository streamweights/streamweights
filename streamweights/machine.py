"""Which machine is running: the host, the operating system and the architecture, for the
provenance every stage, row segment and checkpoint carries."""

from __future__ import annotations

import platform

_NAMES = {"Darwin": "macOS", "Linux": "Linux", "Windows": "Windows"}


def info() -> dict:
    sysname = platform.system()
    return {"host": platform.node(), "system": sysname, "os": _NAMES.get(sysname, sysname),
            "os_release": platform.release(), "arch": platform.machine()}


def os_label() -> str:
    i = info()
    return f"{i['os']} {i['arch']}"
