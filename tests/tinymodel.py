"""A tiny random llama-family model on disk (safetensors + config), so the training
math can be tested on CPU without any downloaded weights."""

import json
from pathlib import Path

import mlx.core as mx

CONFIG = {
    "model_type": "llama", "architectures": ["LlamaForCausalLM"],
    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 3,
    "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 300,
    "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 512,
    "tie_word_embeddings": False, "attention_bias": False, "mlp_bias": False,
    "eos_token_id": 2, "bos_token_id": 1,
}


def make_tiny_model(d: Path, dtype=mx.float32, seed: int = 0, shards: int = 2) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    cfg = dict(CONFIG)
    (d / "config.json").write_text(json.dumps(cfg))
    mx.random.seed(seed)
    H, I, V, hd = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"], 16
    kv = cfg["num_key_value_heads"] * hd

    def w(*shape):
        return (mx.random.normal(shape) * 0.08).astype(dtype)

    t = {"model.embed_tokens.weight": w(V, H), "model.norm.weight": mx.ones((H,), dtype),
         "lm_head.weight": w(V, H)}
    for k in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{k}."
        t.update({
            p + "self_attn.q_proj.weight": w(H, H), p + "self_attn.k_proj.weight": w(kv, H),
            p + "self_attn.v_proj.weight": w(kv, H), p + "self_attn.o_proj.weight": w(H, H),
            p + "mlp.gate_proj.weight": w(I, H), p + "mlp.up_proj.weight": w(I, H),
            p + "mlp.down_proj.weight": w(H, I),
            p + "input_layernorm.weight": mx.ones((H,), dtype),
            p + "post_attention_layernorm.weight": mx.ones((H,), dtype)})
    names = sorted(t)
    for i in range(shards):
        part = {n: t[n] for n in names[i::shards]}
        mx.save_safetensors(str(d / f"model-{i:05d}-of-{shards:05d}.safetensors"), part)
    return d


CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def add_char_tokenizer(d: Path) -> None:
    """A character-level HF tokenizer with a ChatML template: real tokenizer plumbing
    (load_tokenizer, apply_chat_template) with no downloaded files."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    chars = [chr(c) for c in range(32, 127)] + ["\n"]
    specials = ["<pad>", "<|im_start|>", "<|im_end|>", "<unk>"]
    vocab = {t: i for i, t in enumerate(specials + chars)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"<\|im_start\|>|<\|im_end\|>"), "isolated"),
        pre_tokenizers.Split(Regex("."), "isolated")])
    tok.add_special_tokens(["<|im_start|>", "<|im_end|>"])
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "chat_template": CHATML,
        "eos_token": "<|im_end|>", "pad_token": "<pad>", "unk_token": "<unk>",
        "bos_token": None, "clean_up_tokenization_spaces": False}))
    cfg = json.loads((d / "config.json").read_text())
    cfg["eos_token_id"] = vocab["<|im_end|>"]
    (d / "config.json").write_text(json.dumps(cfg))
