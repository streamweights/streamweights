"""Tiny random Hugging Face models on disk (every verified family), built with transformers
and torch only, so the PyTorch engines are tested on any CPU without downloaded weights."""

import json
from pathlib import Path

import torch

from tests.tinymodel import CHATML  # noqa: F401  (the same ChatML template as the MLX tests)

TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=300, max_position_embeddings=1024,
            pad_token_id=0, eos_token_id=2, bos_token_id=1)

FAMILIES = ["llama", "qwen2", "qwen3", "mistral", "phi3", "gemma2"]


def config_for(family: str):
    import transformers as T
    kw = dict(TINY)
    if family == "llama":
        return T.LlamaConfig(**kw)
    if family == "qwen2":
        return T.Qwen2Config(**kw)
    if family == "qwen3":
        return T.Qwen3Config(head_dim=16, **kw)
    if family == "mistral":
        return T.MistralConfig(**kw)
    if family == "phi3":
        return T.Phi3Config(**kw)
    if family == "gemma2":
        return T.Gemma2Config(head_dim=16, sliding_window=24, **kw)
    raise ValueError(family)


def make_tiny(d: Path, family: str = "llama", seed: int = 0, dtype=torch.float32,
              tokenizer: bool = True) -> Path:
    from transformers import AutoModelForCausalLM
    d.mkdir(parents=True, exist_ok=True)
    cfg = config_for(family)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.08)
        for n, p in model.named_parameters():
            if "norm" in n and p.ndim == 1:
                p.fill_(0.0 if family == "gemma2" else 1.0)
    model.save_pretrained(d, safe_serialization=True, max_shard_size="40KB")
    if tokenizer:
        _tokenizer(d)
    return d


def _tokenizer(d: Path) -> None:
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
    cfg["pad_token_id"] = vocab["<pad>"]
    (d / "config.json").write_text(json.dumps(cfg))
