---
description: LoRA fine-tuning without a big GPU. spill streams the base model from disk and recomputes each layer on the way back, so a model bigger than memory can be trained. Measured on Llama 3.3 70B.
---

# LoRA fine-tuning without a big GPU

`spill tune` trains a LoRA adapter against the full-precision base model, resident when the model fits in memory and streamed from disk when it does not. A 3-step run on Llama 3.3 70B bf16 on a 48 GB Mac finished in 21 min at 4.55 TFLOP/s ([report 008](../reports/008-phase3.md)).

## What was measured

| measure | result |
|---|---|
| streamed against resident LoRA, float32 | loss within 0.024% over 100 steps |
| 70B, rank 16 on every linear, micro-batch 21 | 417 s per step |
| 70B trained throughput | about 11 tokens/s, roughly 400,000 tokens per 10-hour night |
| 70B adapter | 207.1M parameters |

Streaming works by saving each layer's input on the forward pass and streaming the weights again in reverse, recomputing each layer under autograd: two weight streams per micro-batch.

## The steps

```
spill tune qwen2.5:7b train.jsonl --name mine
spill eval evals.jsonl qwen2.5:7b qwen2.5:7b+mine
```

`train.jsonl` rows are `{"prompt","answer"}`. To train on the 70B itself, pass its tag: `spill tune llama3.3:70b train.jsonl --name big`. It takes several nights on a laptop, so keep it plugged in; jobs hold the Mac awake and write checkpoints you can resume.

## Try it

```
spill tune qwen2.5:7b train.jsonl --name mine
```

Source and issues: [github.com/streamweights/streamweights](https://github.com/streamweights/streamweights).
