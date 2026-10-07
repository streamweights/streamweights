"""LoRA parameters for training on MLX: the engine-neutral pieces live in lora_core (numpy);
this module adds what needs mlx (reading a block's linear shapes, mx arrays)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..errors import SpillError  # noqa: F401
from .lora_core import (LoraConfig, _KEY, _union_config, by_layer, config_from_dir,  # noqa: F401
                        cosine_np, init_params_np, n_param_bytes, param_name,
                        read_adapter_params, save_adapter_dir, select_shapes)


def linear_shapes(block, only: list[str] | None = None) -> dict[str, tuple[int, int]]:
    """{module path: (in_features, out_features)} for every linear in one
    transformer block (quantized linears report their logical width)."""
    out: dict[str, tuple[int, int]] = {}
    for path, mod in block.named_modules():
        if isinstance(mod, nn.QuantizedLinear):
            out_f, packed = mod.weight.shape
            out[path] = (packed * 32 // mod.bits, out_f)
        elif isinstance(mod, nn.Linear):
            out_f, in_f = mod.weight.shape
            out[path] = (in_f, out_f)
    return select_shapes(out, only)


def init_params(shapes, n_layers: int, rank: int, seed: int) -> dict[str, mx.array]:
    """The shared numpy init (lora_core.init_params_np) as mx arrays."""
    params = {k: mx.array(v) for k, v in init_params_np(shapes, n_layers, rank, seed).items()}
    mx.eval(params)
    return params


def cosine_sim(a: mx.array, b: mx.array) -> float:
    return cosine_np(a, b)
