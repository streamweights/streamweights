# Run anywhere: portable jobs, a PyTorch engine, scheduler-ready (directive 012)

Measured on one machine: an Apple M4 Pro, 48 GB, macOS, Python 3.12, torch 2.14.1, transformers 5.18, PEFT 0.21, MLX 0.32. Model: qwen2.5:0.5b (bf16 safetensors), and a float32 copy of it for the float32 gates. Every number below is from `docs/reports/012-gates.json`, written by `scripts/gates_012.py`. The torch engine ran on the CPU of this Mac; no CUDA device was available, so the NVIDIA engine is built and not verified (`python -m streamweights.verify_cuda` is the verification, see `docs/linux.md`).

## Gates, item 7: identity on the 0.5B

| gate | result | detail |
|---|---|---|
| torch streamed vs torch resident, CPU, float32, 20 prompts | **pass** | greedy identical 20 of 20 (581 tokens compared); maximum log-prob difference 0 (limit 1e-5), top-5 0 |
| torch vs MLX, float32, 20 prompts | **pass** | greedy identical 20 of 20; **maximum log-prob difference 0.000182** over 581 tokens |
| torch streamed tune vs PEFT resident tune, CPU, float32, 50 steps, natural text | **pass** | loss within 8.4e-07 relative per step (limit 0.1%); per-tensor adapter cosine minimum 0.999996 over 336 tensors (floor 0.9999) |
| torch tune vs MLX tune, float32, 50 steps, natural text | **pass** | loss within 6.1e-05 relative per step (limit 1%); adapter cosine minimum 0.999996 |
| torch tune vs MLX tune, float32, 50 steps, toy task, then `spill eval` on 60 held-out rows | **pass** | loss within 0.002 relative per step; scores: base 0.000, torch adapter 1.000, MLX adapter 1.000 |

Settings: LoRA rank 16 on every linear (alpha 32), micro-batch 4, max sequence 256, AdamW with a cosine schedule; learning rate 2e-5 on `examples/gate-natural.jsonl` (the loss falls from 1.4 to 0.3 in 100 steps, so relative loss differences mean something) and 5e-6 on the toy task (slower, so the loss is still moving at step 50 and the held-out score goes from 0 to 1). The streamed torch tune, the PEFT resident tune and the MLX streamed tune start from the same numpy initialisation and use the same optimizer arithmetic (AdamW without bias correction, decoupled decay first, cosine read at the step count before counting), which is what makes a checkpoint mean the same thing on every engine. Loss curves are in the JSON.

## bf16: statistics, no identity claim

Greedy agreement on the first 10 of the 20 prompts with 24 new tokens, next to batch-shape noise (the same engine and numerics, batch 1 against the whole set):

| comparison | rows identical | max log-prob difference |
|---|---|---|
| torch bf16 vs torch float32 | 9 of 10 | 0.190 |
| MLX bf16 vs MLX float32 | 9 of 10 | 0.132 |
| torch bf16 vs MLX bf16 | 10 of 10 | 0.321 |
| batch-shape noise: MLX bf16, batch 1 vs all | 8 of 10 | 0.158 |
| batch-shape noise: torch bf16, batch 1 vs all | 10 of 10 | 0.073 |

Gradients, Phase 3's check: one micro-batch of 4 examples, 192 tokens, same parameters, per-tensor cosine to the float32 gradient:

| comparison | mean cosine | minimum cosine | worst tensor |
|---|---|---|---|
| torch bf16 vs torch float32 | 0.9736 | 0.7021 | layers.0.self_attn.q_proj.lora_a |
| MLX bf16 vs torch float32 | 0.9884 | 0.7424 | layers.0.self_attn.q_proj.lora_a |
| torch bf16 vs MLX bf16 | 0.9870 | 0.8054 | layers.0.self_attn.q_proj.lora_a |
| batch-shape noise: MLX bf16, one batch vs two halves | 0.9899 | 0.8758 | layers.4.self_attn.k_proj.lora_a |
| batch-shape noise: torch bf16, one batch vs two halves | 1.0000 | 1.0000 | layers.2.mlp.down_proj.lora_a |

In bf16 the two engines are as far from float32 as each is from the other, and the MLX engine moves by about as much when only the batch shape changes. The torch CPU engine shows no batch-shape noise in the gradient: its rows are computed independently, so one batch and two halves agree to 13 digits. That is a property of CPU kernels and says nothing about CUDA.

## Gates, item 8: jobs that change hardware

Training: 100-step schedule, stopped at step 50 and checkpointed to a `--state` directory, continued to step 100 on the other engine. The reference is the uninterrupted 100-step run on the first engine; the noise is the distance between two clean 100-step runs, one on each engine (the first segment runs MLX bf16 on the Apple GPU, the second torch-cpu float32, or the reverse, and the checkpoint records both). The continued curve must be no farther from the reference than 1.5 times that noise plus 1%, and the final adapter must score within the larger of the clean-pair difference and 2 of 60 rows of the reference's score.

| direction | data | result | resumed vs reference | clean pair (noise) | limit | held-out score: resumed / reference / other engine |
|---|---|---|---|---|---|---|
| MLX 50 steps, then torch-cpu 50 | natural text | **pass** | 0.004 | 0.016 | 0.034 | n/a |
| MLX 50 steps, then torch-cpu 50 | toy task | **pass** | 0.065 | 0.621 | 0.941 | 1.000 / 1.000 / 1.000 |
| torch-cpu 50 steps, then MLX 50 | natural text | **pass** | 0.006 | 0.016 | 0.034 | n/a |
| torch-cpu 50 steps, then MLX 50 | toy task | **pass** | 0.098 | 1.718 | 2.587 | 1.000 / 1.000 / 1.000 |

The numbers are the mean relative loss difference over steps 51 to 100. On the toy task the loss is near zero late in the run, so its relative differences are large for everyone, which is why the limit is stated against the clean pair. The history recorded in each checkpoint, for the MLX-then-torch run on natural text: steps 0 to 50 on mlx_stream_tune (apple-gpu:Apple M4 Pro, bf16); steps 50 to 100 on torch_stream_tune (cpu:Apple M4 Pro, float32).

Rows: `spill eval` over 60 held-out rows with `--state`, stopped after 30 on one engine (`--stop-after 30`) and finished on the other, in both directions.

| direction | result | rows | unique | missing | duplicated | rows by engine |
|---|---|---|---|---|---|---|
| MLX then torch-cpu | **pass** | 60 | 60 | 0 | 0 | mlx_resident: 30, torch_resident: 30 |
| torch-cpu then MLX | **pass** | 60 | 60 | 0 | 0 | mlx_resident: 30, torch_resident: 30 |

Every result row carries `streamweights.engine`, `streamweights.hardware` and `streamweights.numerics`; every pushed segment of state records the same.

The committed fixture (`streamweights/data/fixtures/resume-qwen05`, 3.3 MB: rank 4 on `q_proj` and `v_proj`, a 100-step schedule stopped at step 50 on MLX, bf16 base, Apple GPU) resumes on torch-cpu: see the CI notes below.

## Measured: the torch-cpu streamed pass on the 0.5B

160 rows (the 20 sample prompts eight times, 64 new tokens at most), 50 passes, on an otherwise idle machine, float32 compute (this CPU's bf16 linear layers measured about six times slower than float32, so the engine chooses float32 and says so in the pre-run line):

| weights on disk | streamed: median pass | read rate while reading | tokens/s | resident: median pass | tokens/s |
|---|---|---|---|---|---|
| bf16 files (0.99 GB), converted to float32 per layer | **0.227 s** | 11.5 GB/s | 186.8 | 0.158 s | 209.5 |
| float32 files (1.98 GB) | **0.192 s** | 33.0 GB/s | 187.4 | 0.170 s | 192.9 |

In the 20-prompt identity gate (float32 files) the median streamed pass was 0.090 s and the resident pass 0.065 s. The model is small enough that the file cache serves most reads even with `F_NOCACHE`, so the read rate is a ceiling for this machine, not a disk figure; the 70B number is the disk's.

## CI and containers

Pending.

## Decisions


## What remains for the verification batch

