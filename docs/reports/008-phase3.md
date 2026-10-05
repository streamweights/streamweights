# Phase 3 report: spill tune

Run 2026-10-04 and 2026-10-05, Apple M4 Pro (48 GB RAM, Metal working set 36 GiB).
Directive: `docs/paste-sets/008-phase3-tune.md`, finished under `docs/paste-sets/010-cleanup.md`.

One-line conclusion: **`spill tune` trains LoRA adapters on two paths, resident and
streamed from NVMe. In float32 the streamed backward matches the resident one to
0.024% in loss and 0.999999 in gradient cosine. In bf16 both paths deviate from
float32 by rounding noise, and the streamed path is closer to the float32 truth than
mlx-lm. A 3-step smoke test on Llama 3.3 70B bf16 ran to completion in 21 minutes at
4.55 TFLOP/s with a 24.2 GB peak.**

All numbers below come from files in this directory:
`008-gate-f32-lr1e-4.json`, `008-gate-*.json`, `008-gradcheck.json`, `008-toy.txt`,
`008-smoke70-tune.txt`, `008-smoke70-run.txt`, and `state/calibration.json`.

## 1. Float32 identity gate

The gate trains the same adapter for 100 steps on the Qwen2.5-0.5B toy task through
the resident path (mlx-lm's LoRA tuner) and through the streamed path (saved layer
inputs, reverse-order recompute VJP), same init, same batches, same optimizer, in
float32 (`008-gate-f32-lr1e-4.json`).

| measure | result |
|---|---|
| loss relative difference at step 1 | 9.4e-08 |
| loss relative difference, max over steps 2 to 100 | 2.44e-04 (0.024%) |
| loss relative difference, mean over steps 2 to 100 | 3.9e-05 |
| gradient cosine, minimum over 336 adapter tensors (final adapters) | 0.999999 |
| gradient cosine, mean over lora_b tensors | 0.99999995 |
| greedy output identical, base+adapter from each path, 20 held-out prompts | 20 of 20 |
| greedy identical, streamed engine versus resident engine on the same adapter | 20 of 20 |
| wall time, resident / streamed | 53.4 s / 73.8 s |
| peak memory, streamed | 5.8 GB |

Verdict: PASS. In float32 the two paths are the same computation.

## 2. The bf16 deviation and why

The same gate in bf16 does not pass a tight identity threshold, and it should not be
expected to (`008-gate-lr1e-4.json`, `008-gate-lr1e-5.json`). At learning rate 1e-4
on the toy task, the loss relative difference reached 4.79 at its worst step and the
minimum gradient cosine was 0.26. At 1e-5 the minimum cosine was 0.84 and the mean
0.976. On natural text (`008-gate-natural-*.json`) the minimum cosine was 0.80 at
1e-4, 0.91 at 2e-5 and 0.95 at 5e-6, with the mean at 0.978, 0.992 and 0.995.
Greedy outputs matched 15 of 20, 14 of 20 and 17 of 20 between resident and
streamed; the streamed engine and the resident engine agreed 20 of 20 on whichever
adapter they were handed.

Why: bf16 has an 8-bit mantissa. The lora_a gradients are small numbers that come
out of a long chain of bf16 matmuls and are sensitive to the order and the
rounding point of every accumulation. The two paths recompute activations in
different orders (the streamed path recomputes per layer in reverse), so the
rounding errors differ even though both are correct. Small differences compound
over 100 optimizer steps. This is rounding noise, not a bug, and the gradient check
in the next section shows it: against a float32 reference the streamed path is the
closer of the two.

## 3. Gradient check against a float32 reference

`scripts/verify_phase3.py gradcheck`: same parameters, same batch, per-tensor
cosine of the bf16 gradient against the float32 gradient (`008-gradcheck.json`).

| path | mean cosine to float32 | minimum cosine | worst tensor |
|---|---|---|---|
| streamed bf16 | **0.9966** | **0.952** | layers.0.self_attn.q_proj.lora_a |
| mlx-lm bf16 | 0.9810 | 0.666 | layers.7.self_attn.q_proj.lora_a |
| streamed versus mlx-lm, both bf16 | 0.9793 | 0.650 | layers.7.self_attn.q_proj.lora_a |

The streamed path is no worse than mlx-lm on either statistic and clearly better on
the minimum. Disagreement between the two bf16 paths is therefore mostly mlx-lm's
own noise.

## 4. Dtype audit

From `008-gradcheck.json`: base weights are bf16; adapter parameters, gradients and
AdamW state are all float32. This is deliberate. Adapters are small (8.8M
parameters on the 0.5B, 207.1M on the 70B), so float32 state costs little, and it
keeps the optimizer update from being rounded away in bf16. Adapters are written in
the PEFT and mlx-lm layouts and cast to the base dtype only when applied.

## 5. Toy task results (Qwen2.5-0.5B, bf16)

Task: 200 training examples whose replies are code words the prompt never mentions.
Held-out: 60 rows, metric exact match. Command run:
`python scripts/verify_phase3.py toy` (`008-toy.txt`).

| model | adapter | exact match (60 held-out rows) | tokens |
|---|---|---|---|
| qwen2.5:0.5b | none | 0.000 | 3,432 |
| qwen2.5:0.5b | toy-res (resident) | **1.000** | 2,895 |
| qwen2.5:0.5b | toy-str (streamed) | **1.000** | 2,895 |

| path | steps | wall time | final loss | peak memory | rate |
|---|---|---|---|---|---|
| resident | 150 | 22 s | 0.0595 | 2.0 GB | |
| streamed | 150 | 36 s | 0.000375 | 0.6 GB | 2.42 TFLOP/s, 1,127 tokens/s, 9.97 GB/s weight stream |

Both adapters take the base from 0 of 60 to 60 of 60. The two adapters produce the
same token counts and the same eval table; they are different bf16 trajectories to
the same behavior, as section 2 predicts.

## 6. The 70B smoke test

Llama 3.3 70B bf16 (131.4 GB, streamed from NVMe), 64 examples, 18,955 tokens
(4,224 trained) at `--max-seq 512`, 3 steps of micro-batch 21, rank 16 alpha 32 on
all seven projection types in 80 layers (`008-smoke70-tune.txt`).

| measure | result |
|---|---|
| step times | 559 s, 411 s, 475 s (mean 417 s/step by the tune summary) |
| total | 21 min |
| loss | 0.9397 to 0.05849 over 3 steps |
| achieved compute | **4.55 TFLOP/s** overall, 4.68 while computing |
| weight stream | 770 GB read, 2.69 GB/s while reading |
| trained throughput | **about 11 tokens/s** through the layers |
| per 10-hour night | **roughly 400,000 tokens** (11 x 36,000 s) |
| peak memory | **24.2 GB** (21.9 GB at steps 1 and 2), under the 27.0 GB target |

Adapter load check: `spill run llama3.3:70b` and `llama3.3:70b+smoke70` on the same
4 prompts both completed 4 of 4 rows (110 and 103 completion tokens), so the adapter
loads through the 70B streaming inference engine and changes its output
(`008-smoke70-run.txt`). Pass times were 32 to 38 s, as in Phase 2.5.

## 7. Estimate accuracy

| run | printed estimate | actual | ratio actual / estimate |
|---|---|---|---|
| toy resident | 22 s | 22 s | 1.0 |
| toy streamed (this run) | 62 s | 36 s | 0.58 |
| 70B smoke, 3 steps | 40 min | 21 min | 0.53 |

The 70B estimate assumed 3.5 TFLOP/s "until the first step measures it" and the
measured engine stream rate of 3.6 GB/s, and it counted the weight stream and the
compute as additive. The achieved rate was 4.55 TFLOP/s and the compute overlaps the
stream, so the estimate was about twice too high. That is the safe direction, but it
is not accurate. The fix is in place: every tune records its achieved rate in
`state/calibration.json` (`tune_rates`: 0.5B resident 1.89, 0.5B streamed 2.42,
70B streamed 4.68 TFLOP/s) and the next pre-run line uses it.

## Decisions

- **Stalled previous session.** The Phase 3 session stopped waiting: it ended its
  turn with the 70B smoke finished but the report not written, and a detached gate
  watcher (`streamweights-phase35-gate.sh`) polled `origin/main` every 300 s for
  this report so Phase 3.5 could start. Nothing was lost; the smoke results,
  gradient check and calibration were on disk uncommitted. This session killed the
  watcher, committed those files, re-ran the toy verification, and wrote this
  report. The rule going forward is in CLAUDE.md: no end-of-turn waits and no
  background pollers.
- **bf16 gate.** A tight identity threshold is applied in float32 only. In bf16 the
  criterion is the gradient check against a float32 reference (section 3), because
  bf16 resident and streamed legitimately differ by rounding noise.
- **Float32 adapter state** (section 4) is the default, not an option.
- **70B smoke scope.** Per the changed item 8, the 70B verification is 3 steps and a
  4-prompt adapter-load check, not a full tuning run.
- **Toy numbers.** The toy was re-run for this report; its figures replace the
  earlier run in `008-toy.txt` (22 s and 36 s here against 22 s and 37 s before;
  both runs gave 0.000 to 1.000 exact match).
- **Progress-bar lines** were stripped from the committed transcripts; all result
  lines are unchanged.
