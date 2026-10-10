"""Decoding settings: what a request asks for, what an engine can apply, what was applied.

The guided evaluation puts every recorded decoding setting (temperature, top_p, max_tokens, stop
sequences, the greedy flag, the seed) into each request body. The MLX and PyTorch engines decode
greedily (an argmax at every step) and stop at max_tokens or the end-of-turn token; they have no
sampling and no stop-sequence matching, and this module does not add any. A request that asks for
a setting an engine cannot apply is refused before anything runs, with a message that names the
engine and the setting. Each result row then carries the settings as applied."""

from __future__ import annotations

from .errors import SpillError

# what the engines in this package can apply (llama.cpp applies them itself and is not checked)
GREEDY_ENGINES = ("mlx", "torch-cpu", "torch-cuda")


def engine_label(name: str | None) -> str:
    n = (name or "").lower()
    if n.startswith("mlx"):
        return "mlx"
    if n.startswith("torch"):
        return "torch-cuda" if "cuda" in n else "torch-cpu"
    return n


def request_fields(decoding: dict, max_tokens: int) -> dict:
    """The request-body fields for a recorded decoding policy."""
    return {"max_tokens": max_tokens, "temperature": decoding.get("temperature", 0.0),
            "top_p": decoding.get("top_p", 1.0), "stop": list(decoding.get("stop") or []),
            "seed": decoding.get("seed", 0), "greedy": bool(decoding.get("greedy", True))}


def unsupported(engine: str, body: dict) -> list[str]:
    """Settings in a request body the engine cannot apply (empty when it can apply all)."""
    eng = engine_label(engine)
    if eng not in GREEDY_ENGINES:
        return []
    bad = []
    t = body.get("temperature")
    if t not in (None, 0, 0.0):
        bad.append(f"temperature={t} (this engine decodes greedily; only temperature 0 is applied)")
    p = body.get("top_p")
    if p not in (None, 1, 1.0):
        bad.append(f"top_p={p} (there is no nucleus sampling; only top_p 1 is applied)")
    if body.get("stop"):
        bad.append(f"stop={body['stop']!r} (there is no stop-sequence matching)")
    if body.get("greedy") is False:
        bad.append("greedy=false (there is no sampling)")
    return bad


def check_decoding(engine: str, decoding: dict) -> None:
    """Refuse before evaluation if `engine` cannot apply the recorded decoding policy."""
    bad = unsupported(engine, request_fields(decoding, 1))
    if bad:
        raise SpillError(f"engine {engine_label(engine)} cannot apply the recorded decoding "
                         f"setting {bad[0]}", "set evaluation.decoding in streamweights.toml to "
                         "greedy (temperature 0, top_p 1, no stop), or use an engine that applies it")


def check_rows(engine: str, rows: list[dict]) -> None:
    for r in rows:
        bad = unsupported(engine, r.get("body", {}))
        if bad:
            raise SpillError(f"engine {engine_label(engine)} cannot apply the request's decoding "
                             f"setting {bad[0]} (row {r.get('custom_id')})",
                             "send temperature 0, top_p 1 and no stop")


def applied(body: dict) -> dict:
    """The decoding settings as the greedy engines apply them to this request."""
    return {"temperature": 0.0, "top_p": 1.0, "stop": [], "greedy": True,
            "max_tokens": body.get("max_tokens"), "seed": body.get("seed", 0),
            "requested": {k: body[k] for k in ("temperature", "top_p", "stop", "seed", "greedy")
                          if k in body}}
