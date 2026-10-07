---
description: Phase 0 report: memory-mapping a model bigger than RAM reached only 11 to 13% of the disk rate, which is why spill streams layers in order.
---

> Names changed in 003: Spillway → streamweights, CLI → spill. Command names and the `spillway` result-metadata key below are the historical Phase 0 record (now `spill` / `streamweights`).
# Phase 0 report - the wedge on mmap

Run 2026-09-30, Apple M4 Pro laptop. Directive: `docs/paste-sets/001-phase0-wedge.md`.
One-line conclusion: **the pipeline works end-to-end and mmap is the bottleneck - paging
achieves 8–13% of this machine's sequential NVMe rate, so the Phase 1 streaming runner
is justified by a factor of ~8.**

## 1. Hardware probe (`docs/reports/data/hardware-m4pro.json`)

| field | value |
|---|---|
| OS | Darwin 24.6.0 (macOS 15.7.3) |
| CPU | Apple M4 Pro, 12 cores |
| RAM | 48 GiB total (51,539,607,552 B); ~22 GiB free at probe |
| GPU | Apple M4 Pro, unified memory, Metal working-set limit ≈ 36 GiB (75% of RAM) |
| bf16 compute | yes |
| Disk | /dev/disk3s5, 230 GiB free of 460 GiB at start |
| NVMe sequential read | **5.1 GiB/s** (5,481 MB/s), 4 GiB run, page cache bypassed via F_NOCACHE on write and read |

Probe honesty note: the first probe read its own just-written file from page cache and
reported 16.5 GB/s; `F_NOCACHE` must be set on the **write** fd too, since it does not
evict pages already cached. Fixed before anything consumed the number.

## 2. llama.cpp (item 3)

Release **b11311** (`0.5.0-dev, commit f7b384c1e`), prebuilt macOS arm64, in `bin/`
(not vendored; the CLI re-fetches it on a clean clone). All four capabilities present:

- **mmap loading** - via the new `-lm/--load-mode {auto,mmap,mmap+mlock}` flag (the old
  `--no-mmap` is gone; `auto` defaults to mmap). streamweights passes `--load-mode mmap`.
- **Partial GPU offload** - `-ngl/--n-gpu-layers`.
- **Parallel slots + continuous batching** - `--parallel N`, `--cont-batching` (on by default).
- **bf16 GGUF** - loads and generates on Metal (verified with a locally converted
  qwen2.5-0.5b bf16 GGUF; `ftype BF16`, correct output, 205 tok/s aggregate in the full run).

## 3. Wedge model and quant (items 5 + 10)

The policy held **bf16 as the default for all three registry models** on this machine:
bf16 compute ✓, disk could hold bf16+20 GB ✓, and the NVMe-rate rule predicted 26 s per
forward pass for 70B bf16 (< 120 s) ✓. No drop rule fired.

But no model's policy-chosen quant landed in the directive's 1.5–2× RAM window:
0.5b bf16 = 0.03×, 32b bf16 = 1.27×, **70b bf16 = 141 GB = 2.74×** - outside the window,
so the 70B bf16 branch of item 10 does not apply. Fall-through: **llama3.3:70b Q8_0**,
75 GB = **1.56× nominal RAM** (1.46× by exact bytes) - the only candidate adjacent to the
window and still well beyond RAM, which is the property the wedge needs. Because the
policy itself would not drop (all bf16 rules passed), Q8_0 was selected via the explicit
`--quant Q8_0` opt-in, stated in every output row's metadata.

Post-hoc policy finding: the NVMe-rate rule's model (size ÷ sequential rate) predicted
13.7 s per Q8_0 pass; measured mmap reality was **174.7 s** - the rule is optimistic by
~13× because paging is not sequential I/O. The rule as written would *still* have held
bf16 (26 s predicted); measured bf16 would have been ~350 s/pass. Phase 1 should re-base
this rule on measured mmap rates, not the sequential probe.

## 4. Measurement (item 10)

Input: first 500 rows of `examples/evals-2000.jsonl` (full-set runs would exceed 6 h by
estimate - see Decisions), context 4096, max_tokens 128. Each setting time-boxed to
25 min of wall time with a clean SIGINT checkpoint. Probed sequential rate: 5,481 MB/s.

| model / setting | rows completed | wall | agg tok/s | disk avg (iostat) | % of probed | page-cache behavior / outcome |
|---|---|---|---|---|---|---|
| qwen2.5:0.5b bf16, resident, full 2000 rows | 2000/2000 | 1,023 s | **205.6** (completion; 1,044 incl. prompt) | n/a (resident) | n/a | resident path proven; median row latency 133 s at batch 256 |
| 70B Q8_0, parallel 16 | **0**/500 in 25 min | 1,534 s | 0 | 702 MB/s | **12.8%** | no OOM - pure thrash. 1,049 GB total disk transfer vs 135 GB file pageins; load alone 7 m 42 s |
| 70B Q8_0, parallel 32 | 72 rows, **all HTTP 500** | 1,533 s | 0 | 637 MB/s | 11.6% | **GPU OOM** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`, 56 Metal command-buffer failures) |
| 70B Q8_0, parallel 64 | 140 rows, **all HTTP 500** | 1,533 s | 0 | 638 MB/s | 11.6% | **GPU OOM**, same signature |
| 70B Q8_0, parallel 128 | 286 rows, **all HTTP 500** | 1,533 s | 0 | 622 MB/s | 11.4% | **GPU OOM**, same signature |
| 70B Q8_0, parallel 1 (diagnostic, 1 row) | 1/1 | 510 s | 0.006 | - | **7.8%** of probed per decode pass | 47-token prompt: 160 s; decode: **174.7 s/token** (one full 70 GB weight stream per token at ~410 MB/s). Output was correct ("Four.") |

Reading the thrash: at p16 the OS moved >1 TB through the disk in 25 minutes to complete
zero rows - mmap page faults are small, random-ordered reads issued at effective queue
depth ~1, and with the file 1.5× RAM every pass evicts the pages the next pass needs.
The effective rate never exceeded ~13% of what the same disk does sequentially.

p32/64/128 "completed" rows are counted honestly as failures: llama-server initialized
the requested slots (Metal allocates virtually), then every forward batch died on GPU
OOM, returning 500s. Per the directive these three settings are **OOM - skipped** for
throughput purposes.

## 5. Decisions

1. **Wedge model fall-through** - no policy-chosen quant in 1.5–2× RAM; chose 70B Q8_0
   (1.56× nominal) via explicit `--quant` since no policy drop rule fired (§3).
2. **Time-boxing** - estimated >6 h per full 500-row setting at streaming rates; each
   parallel setting ran 25 min with clean SIGINT checkpoint; partial counts reported as-is.
3. **qwen2.5 bf16 GGUFs don't exist publicly** - registry records the safetensors repos;
   0.5b was converted locally with `convert_hf_to_gguf.py --outtype bf16` (conversion
   tooling kept out of the repo; backends stay unmodified upstream).
4. **Single-slot diagnostic added** beyond the four directed settings, because the
   directed settings produced only zeros and OOMs; it pins the per-pass mmap number (§4).
5. **Eval rows carry a placeholder model name**; the engine overwrites `body.model` with
   the job's model so CLI argument and metadata are authoritative.
6. **Probe fix** - F_NOCACHE on both write and read fds (§1).
7. **gitignore bug fixed** - `jobs/` pattern shadowed the Python jobs package (now `streamweights/jobs`);
   root-anchored.
8. **70B measurements used explicit `--parallel`** overrides (measurement mode); the
   engine's own KV-bound computation would have chosen N≈7 at 4k and avoided the OOMs.

## 6. Plain answers

**(a) Largest batch size VRAM supports for 70B Q8_0 at 4k context on this machine:**
**16.** Parallel 16 ran with zero GPU OOMs (27 of 80 layers offloaded, GPU-side KV
≈ 6.8 GiB, total GPU ≈ 30 GiB of the 36 GiB working set); parallel 32 OOM'd every batch
(≈ 37 GiB needed). The engine's conservative formula computes 7; the empirical ceiling
sits between 16 and 32, and 16 is the largest tested-safe value.

**(b) Fraction of probed NVMe rate achieved by mmap, and is the streaming runner needed:**
11.4–12.8% during the parallel runs (622–702 MB/s of 5,481 MB/s), and 7.8% in the clean
single-slot decode (410 MB/s). **Yes - Phase 1 is needed.** Sequential weight streaming
at the probed rate would cut a Q8_0 pass from 175 s to ~14 s (~12×); with batch ~16–128
amortizing each pass, aggregate throughput rises from effectively 0 to an estimated
1–9 tok/s on this machine - the difference between "impossible" and "overnight."

**(c) Did bf16 hold as the default?** Yes - the policy never dropped. All three models
resolved to bf16 (bf16 compute present, disk sufficient, predicted pass time 26 s < 120 s).
The wedge test's Q8_0 was an explicit `--quant` opt-in forced by the directive's 1.5–2×
RAM window, not a policy drop, and is stamped in every output row's metadata.

**(d) Install → first streamed row, and the worst moment:** **42 seconds** from
`pip install -e .` to rows streaming into results.jsonl on a clean clone (including the
hardware probe, llama.cpp prebuilt fetch, and 0.6 GB model download). The single worst
moment: the very first run command a new user types (`spillway run`, now `spill run`) dead-ends, because the bf16
default for qwen2.5:0.5b has no published GGUF - the tool prints the exact recovery
command (`--quant Q8_0`), but "one install, zero config" held while "one command, zero
config" did not. Fix candidates for Phase 1: publish/convert bf16 automatically on
first run, or let the registry mark a model's smallest published quant as its
teaching-path default with a loud banner.

## Appendix: run artifacts

- Measurements: `state/measurements-llama3.3-70b-Q8_0.json`; jobs `20260930-204021-c17a64`
  (p16), `-210610-6733d8` (p32), `-213158-611c63` (p64), `-215746-da1bb8` (p128),
  `-222722-e08c60` (p1 diag), `-202118-a06db1` (0.5b full run).
- Golden path: `docs/reports/001-golden-path.txt`.
- llama.cpp findings: `docs/llamacpp.md`.
