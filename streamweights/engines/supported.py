"""Architecture support table, keyed by config.json model_type / architectures.

States:
  verified — streamed output tested identical to resident execution on a model of
             the family (plus first-token agreement with mlx_lm.generate).
  expected — dense decoder family with an mlx-lm class the streamer binds
             generically; not yet through the identity gate.
  not_yet  — the streamer cannot run it; one-line reason per class.

The resident engine's support follows mlx-lm's own model list.
"""

from __future__ import annotations

from dataclasses import dataclass

VERIFIED = "verified"
EXPECTED = "expected"
NOT_YET = "not_yet"


@dataclass
class Family:
    label: str
    state: str
    module: str | None = None   # mlx_lm.models.<module> that defines the block
    note: str = ""
    # streamer quirk hooks (applied outside the block, which handles its own)
    embed_scale: bool = False       # h *= sqrt(hidden_size) after embedding
    norm_plus_one: bool = False     # RMSNorm uses (1 + weight)
    final_softcap_key: str | None = None  # config key for final logit softcapping
    needs_array_mask: bool = False  # block's attention cannot take mask="causal"


FAMILIES: dict[str, Family] = {
    # ---- verified ----
    "llama": Family("Llama 3.x", VERIFIED, "llama",
                    note="identity 20/20 + mlx_lm first-token 20/20 (llama3.3-70b, Phase 1)"),
    "qwen2": Family("Qwen2/2.5", VERIFIED, "qwen2",
                    note="identity 20/20 + mlx_lm length cross-check (qwen2.5-0.5b, Phase 1/1.5)"),
    # ---- expected (dense decoders, wired generically) ----
    "mistral": Family("Mistral", VERIFIED, "llama",
                      note="llama-shaped; mlx-lm maps it to the llama classes"),
    "qwen3": Family("Qwen3 (dense)", VERIFIED, "qwen3"),
    "phi3": Family("Phi 3/4", VERIFIED, "phi3",
                   note="Phi-4 ships model_type phi3"),
    "gemma2": Family("Gemma 2", VERIFIED, "gemma2", embed_scale=True,
                     norm_plus_one=True, final_softcap_key="final_logit_softcapping",
                     needs_array_mask=True),
    "gemma3_text": Family("Gemma 3 (text)", EXPECTED, None,
                          note="per-layer alternating sliding-window attention needs "
                               "per-layer cache windows; not wired yet"),
    # ---- not yet ----
    "mixtral": Family("Mixtral (MoE)", NOT_YET,
                      note="mixture-of-experts: per-token expert routing defeats layer-order streaming"),
    "qwen2_moe": Family("Qwen MoE", NOT_YET,
                        note="mixture-of-experts: per-token expert routing defeats layer-order streaming"),
    "qwen3_moe": Family("Qwen3 MoE", NOT_YET,
                        note="mixture-of-experts: per-token expert routing defeats layer-order streaming"),
    "deepseek_v2": Family("DeepSeek V2/V3 (MoE)", NOT_YET,
                          note="mixture-of-experts: per-token expert routing defeats layer-order streaming"),
    "deepseek_v3": Family("DeepSeek V2/V3 (MoE)", NOT_YET,
                          note="mixture-of-experts: per-token expert routing defeats layer-order streaming"),
    "llama4": Family("Llama 4 (MoE)", NOT_YET,
                     note="mixture-of-experts: per-token expert routing defeats layer-order streaming"),
    "mamba": Family("Mamba (state-space)", NOT_YET,
                    note="state-space recurrence has no KV cache to batch around"),
    "jamba": Family("Jamba (hybrid)", NOT_YET,
                    note="state-space hybrid: recurrent layers break the per-layer stream loop"),
}

_MULTIMODAL_TYPES = {"llava", "qwen2_vl", "qwen2_5_vl", "mllama", "gemma3",
                     "idefics3", "paligemma", "pixtral", "phi3_v"}


def classify(config: dict) -> Family:
    """Family + state for a config.json. Multimodal is detected before lookup."""
    mt = config.get("model_type", "")
    if mt in _MULTIMODAL_TYPES or "vision_config" in config:
        return Family(f"{mt or 'unknown'} (multimodal)", NOT_YET,
                      note="multimodal: vision towers are not streamed; text-only families only")
    fam = FAMILIES.get(mt)
    if fam:
        return fam
    archs = ", ".join(config.get("architectures", []) or [mt or "unknown"])
    return Family(f"{archs}", NOT_YET,
                  note=f"model_type '{mt}' has no streamer wiring yet")


def table() -> str:
    rows = [f"{'model_type':14s} {'family':22s} {'state':9s} note"]
    for mt, f in FAMILIES.items():
        rows.append(f"{mt:14s} {f.label:22s} {f.state:9s} {f.note}")
    rows.append(f"{'(vision_config)':14s} {'any multimodal':22s} {NOT_YET:9s} "
                "multimodal: vision towers are not streamed")
    return "\n".join(rows)


def mark_verified(model_type: str, note: str) -> None:
    """Flip a family to verified in this source file (used by the verify script)."""
    import re
    from pathlib import Path
    p = Path(__file__)
    s = p.read_text()
    s = re.sub(rf'("{model_type}": Family\("[^"]+", )EXPECTED', rf"\1VERIFIED", s, count=1)
    p.write_text(s)
