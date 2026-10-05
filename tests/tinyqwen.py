"""A tiny random Qwen2- or Llama-family model on disk with a character-level ChatML
tokenizer: real engine plumbing (safetensors index, load_tokenizer, chat template, RoPE,
GQA) on CPU with no downloaded weights."""

import json
from pathlib import Path

import mlx.core as mx

CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def config(kind: str):
    base = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 128,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 4096,
            "tie_word_embeddings": False}
    if kind == "qwen2":
        base.update({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
    else:
        base.update({"model_type": "llama", "architectures": ["LlamaForCausalLM"],
                     "attention_bias": False, "mlp_bias": False})
    if kind == "llama3":     # Llama 3.x rope scaling: a different rope class than plain RoPE
        base["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                "high_freq_factor": 4.0, "original_max_position_embeddings": 64}
    return base


def make_tiny(d: Path, kind: str = "qwen2", seed: int = 0, dtype=mx.float32) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    cfg = config(kind)
    mx.random.seed(seed)
    H, I, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    hd = H // cfg["num_attention_heads"]
    kv = cfg["num_key_value_heads"] * hd

    def w(*shape):
        return (mx.random.normal(shape) * 0.3).astype(dtype)

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
        if kind == "qwen2":
            t[p + "self_attn.q_proj.bias"] = w(H) * 0.1
            t[p + "self_attn.k_proj.bias"] = w(kv) * 0.1
            t[p + "self_attn.v_proj.bias"] = w(kv) * 0.1
    names = sorted(t)
    mx.save_safetensors(str(d / "model-00001-of-00002.safetensors"), {n: t[n] for n in names[0::2]})
    mx.save_safetensors(str(d / "model-00002-of-00002.safetensors"), {n: t[n] for n in names[1::2]})

    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    chars = [chr(c) for c in range(32, 127)] + ["\n"]
    specials = ["<pad>", "<|im_start|>", "<|im_end|>", "<unk>"]
    vocab = {tk: i for i, tk in enumerate(specials + chars)}
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
    cfg["eos_token_id"] = vocab["<|im_end|>"]
    (d / "config.json").write_text(json.dumps(cfg))
    return d
