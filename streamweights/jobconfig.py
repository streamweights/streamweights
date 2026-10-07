"""`--config job.json` and `--emit-config job.json`: a CLI invocation as a file.

`spill tune qwen2.5:0.5b train.jsonl --name x --state s3://bucket/job --emit-config job.json`
writes the exact, fully resolved invocation (every option, defaults included) and exits;
`spill tune --config job.json` runs it, on this machine or any other. Options given on the
command line next to --config override the file. Paths are used as written, so a job meant
for a cluster should name data and state by URI or by a path that exists there.
"""

from __future__ import annotations

import json
from pathlib import Path

from .errors import SpillError

VERSION = 1
# options that describe the invocation itself, not the job
SKIP = {"config", "emit_config", "debug", "help"}


def _plain(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def build(cmd_name: str, click_cmd, args: list[str]) -> dict:
    """Parse `args` with the command's own parser and return the resolved config."""
    ctx = click_cmd.make_context(cmd_name, list(args), resilient_parsing=False)
    arguments, options = {}, {}
    for p in click_cmd.params:
        if p.name in SKIP or p.name not in ctx.params:
            continue
        v = _plain(ctx.params[p.name])
        if v is None or v == []:
            continue
        if p.param_type_name == "argument":
            arguments[p.name] = v
        else:
            if getattr(p, "is_flag", False) and v is False:
                continue
            options[p.name] = v
    return {"spill_config": VERSION, "command": cmd_name, "arguments": arguments,
            "options": options}


def write(path: Path, config: dict) -> None:
    Path(path).write_text(json.dumps(config, indent=2) + "\n")


def load(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise SpillError(f"config file {path} does not exist")
    try:
        cfg = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SpillError(f"config file {path} is not valid JSON ({e.msg} at line {e.lineno})")
    if cfg.get("spill_config") != VERSION or "command" not in cfg:
        raise SpillError(f"{path} is not a spill job config (expected spill_config: {VERSION} "
                         f"and a command); write one with --emit-config")
    return cfg


def to_argv(cfg: dict, click_cmd, cli_args: list[str]) -> list[str]:
    """argv (after the command name) for `cfg`, with options already on the command line
    taking precedence, and CLI positional arguments replacing the file's."""
    by_name = {p.name: p for p in click_cmd.params}
    present = set()
    for tok in cli_args:
        if tok.startswith("-"):
            for p in click_cmd.params:
                if tok.split("=", 1)[0] in (*p.opts, *getattr(p, "secondary_opts", [])):
                    present.add(p.name)
    cli_positional = [t for i, t in enumerate(cli_args)
                      if not t.startswith("-") and not _is_option_value(cli_args, i, click_cmd)]
    # the file's arguments, in the order it lists them, unless the command line gave its own
    given = [str(v) for val in cfg.get("arguments", {}).values()
             for v in (val if isinstance(val, list) else [val])]
    out = (given if not cli_positional else []) + list(cli_args)
    for name, val in cfg.get("options", {}).items():
        p = by_name.get(name)
        if p is None:
            raise SpillError(f"config option {name!r} is not an option of spill "
                             f"{cfg['command']}", f"spill {cfg['command']} --help")
        if name in present:
            continue
        flag = p.opts[0]
        if getattr(p, "is_flag", False):
            if val:
                out.append(flag)
        elif isinstance(val, list):
            for v in val:
                out += [flag, str(v)]
        else:
            out += [flag, str(val)]
    return out


def _is_option_value(args: list[str], i: int, click_cmd) -> bool:
    """True when args[i] is the value of the option just before it."""
    if i == 0:
        return False
    prev = args[i - 1]
    if not prev.startswith("-") or "=" in prev:
        return False
    for p in click_cmd.params:
        if prev in p.opts and p.param_type_name == "option" and not getattr(p, "is_flag", False):
            return True
    return False
