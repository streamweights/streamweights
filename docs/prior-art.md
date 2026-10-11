---
description: How streamweights relates to AirLLM, slowllama, Unsloth, FlexGen and DeepSpeed, and what it adds to the streaming idea.
---

# Prior art

streamweights builds on the idea of running models larger than memory by streaming layers, and it is not the first to do so. This page moved out of the README; the README keeps a one-line pointer here.

- [AirLLM](https://github.com/lyogavin/airllm) runs large models on small GPUs by keeping one layer on the GPU at a time; its README now also describes fine-tuning that streams frozen weights and keeps adapters on the GPU.
- [slowllama](https://github.com/okuvshynov/slowllama) fine-tunes Llama 2 and CodeLlama with LoRA on a Mac or a consumer GPU by offloading blocks to SSD or memory, without quantization; the repository is archived.
- [Unsloth](https://github.com/unslothai/unsloth) makes LoRA, QLoRA and full fine-tuning faster and lighter on GPUs (it states 2x faster, 70% less VRAM), on NVIDIA, AMD, Intel and CPU, and also supports macOS and MLX formats; its README does not mention disk offloading.
- [FlexGen](https://github.com/FMInference/FlexLLMGen) is a throughput-oriented inference engine that offloads weights, activations and the KV cache to CPU memory and disk; inference only, and archived.
- [DeepSpeed](https://www.deepspeed.ai/2022/09/09/zero-inference.html) offloads to CPU memory and NVMe: ZeRO-Infinity for training and ZeRO-Inference, which streams weights layer by layer, for inference.

## What streamweights adds

What streamweights adds to these is the workflow around the streaming idea, not the idea: a CSV-to-evaluated-model path with a frozen task contract and leakage-aware splits, comparisons against baselines on held-out rows with a recorded protocol, verified exports, and runs that move between machines and between MLX and PyTorch under fenced ownership. Whether it is cheaper per completed job than renting a GPU is not measured here, and fitting a model on smaller hardware does not by itself show that.
