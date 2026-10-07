---
description: Phase 1.6 report: the share-ready release and a measured partial proof run on the 70B.
---

# Phase 1.6 report: share-ready, then the proof run as a measured partial

Run 2026-10-02/03, Apple M4 Pro (48 GB RAM, Metal working set 36 GiB).
Directive: `docs/paste-sets/005-phase1.6-share-and-proof.md`.

One-line conclusion: **the repo is stranger-installable and verified across six
architecture families; the proof run completed the short and summarize tiers
(1,334 of 2,000 rows) unattended in one overnight stretch, and was stopped as a
measured partial when the 1,000-token tier exposed two engine defects (peak-chasing
memory budget, serial prefill) that make the long tail ~100 s per pass.**

## 1. Install matrix (Part A, item 7)

| path | Python | spill --help | sample run | notes |
|---|---|---|---|---|
| pip install git+https://github.com/streamweights/streamweights | 3.12 | pass | pass (20/20 rows) | fresh venv |
| pip install git+... | 3.14 | pass | pass | fresh venv; mlx cp314 wheels exist |
| uv tool install git+... | 3.12 (tool default) | pass | pass | |

No Hugging Face login anywhere; nothing ever prompts. Floor is Python 3.10
(declared in pyproject; older interpreters get one sentence). Installed copies
keep data in ~/.streamweights (SPILL_HOME overrides); models.yaml ships inside
the package.

## 2. Architecture table (Part A, item 4), final states

| model_type | family | state | note |
|---|---|---|---|
| llama | Llama 3.x | verified | Phase 1 gate (70B) |
| qwen2 | Qwen2/2.5 | verified | Phase 1/1.5 gates (0.5b) |
| qwen3 | Qwen3 (dense) | **verified (new)** | 20/20 identity + 20/20 mlx_lm first-token, Qwen3-0.6B, 8 s |
| phi3 | Phi 3/4 | **verified (new)** | same gate, Phi-3-mini, 87 s |
| gemma2 | Gemma 2 | **verified (new)** | same gate, gemma-2-2b-it; needed embed-scale, +1 norms, softcap, 5-D mask quirks |
| mistral | Mistral | **verified (new)** | same gate, Mistral-7B-v0.3, 269 s |
| gemma3_text | Gemma 3 (text) | expected | per-layer alternating sliding-window attention needs per-layer cache windows; not wired |
| mixtral, qwen*_moe, deepseek_v2/v3, llama4 | MoE | not_yet | per-token expert routing defeats layer-order streaming |
| (vision_config present) | multimodal | not_yet | vision towers are not streamed |
| mamba / jamba | state-space / hybrid | not_yet | recurrence has no KV cache to batch around |

Verification gate (stricter than directed): streamed vs resident identity 20/20
AND first-token agreement with independent mlx_lm.generate 20/20. Any HF repo id
(org/name[@rev]) resolves, classifies, and runs through the same policy; verified
with Qwen/Qwen2.5-1.5B-Instruct (ran end to end), unsloth/gemma-3-1b-it (clean
expected-but-unwired sentence), deepseek-ai/DeepSeek-V2-Lite-Chat (clean MoE
sentence), plus gated and unknown-repo one-liners.

## 3. Time to first visible text (Part A, items 3 and 9)

From the public install one-liner in a fresh dir: install 11 s; 0.5b sample rows
in 6 s; 70B pre-run line to **first visible partial generated text: 41 s**
(59 s from venv creation). The live tail block (8 slots: custom_id, tokens, last
80 chars) redraws every pass; the progress line carries per-token latency and
peak memory; --quiet suppresses the block.

## 4. The proof run, measured

`spill run llama3.3:70b examples/evals-2000.jsonl`, no flags, bf16, batch by
memory admission. Started 2026-10-02 16:55:43; stopped (deliberately, clean
SIGINT, checkpoint intact and resumable) 2026-10-03 12:47:30. Two interventions,
~35 minutes of total gap, both during the long tier.

| tier | rows | completed | batch | pass time | peak memory |
|---|---|---|---|---|---|
| short-QA (~45-token prompts) | 667 | **667** | 100-300 | 33-38 s | 36.8 GB (first-wave overshoot) |
| summarize-300w (~420-token) | 667 | **667** | 70-150 | 33-40 s | ~34-36 GB |
| doc-QA (~900-token prompts) | 666 | **0** | 74-75 | 86-115 s (125-153 s in degraded episodes) | 34.3 GB |

- **1,334 rows completed unattended in the first 15 h 35 m** (16:55 to the 08:25
  intervention), which is the overnight claim, demonstrated.
- Output distributions (1,334 rows): finish_reason 651 stop / 683 length; output
  length min 87, median 128, max 128. Short-QA ran to the 128 cap on 100% of rows
  (the model answers open questions verbosely); summarize rows genuinely stopped
  on 97% of rows, median 110 tokens. EOS handling is truthful and working.
- **07:30 overshoot:** a 75-row long-tier group's physical KV crossed the
  allocation tier mid-group; passes degraded 34 to 125 s with zero completions.
- **08:25 intervention:** clean SIGINT, added live-allocation backpressure to
  admission (no new rows above 80% of the working set), resumed from 1,334.
- **Fault-storm diagnosis** (why the long tier is ~100 s/pass even when admission
  is correct): live Metal allocation 34.3 of 36 GiB; page-outs flat (~16 MB/74 s)
  but page-ins ~1 GB/s in ~170 KB reads: the OS continuously evicts and re-faults
  clean file-backed pages, fragmenting the weight stream to ~1.4 GB/s effective
  against the 5.4 GB/s probe. Memory pressure, not thermals (no warnings) and not
  the stream itself.
- Admission instrumentation proved the math right: the remaining prompts tokenize
  to ~900 tokens (the chars/4 planning estimate was 1.5x high), and 74-75 rows do
  fit the budget; the slowdown is the fault storm plus serial prefill.
- **Projected long tier on this engine:** ~100 s/pass median, ~75-row groups, 128
  passes per group: ~30-33 h for the remaining 666 rows; **full 2,000-row set
  projected ~50 h on the current engine** (vs the 15.3 h pre-run estimate).

## 5. Decisions

1. **Stopped as a measured partial** (user directive, matching my standing
   recommendation): the remaining 33 h would re-measure behavior the 1,334
   completed rows plus 200+ long-tier passes already characterize; the job stays
   checkpointed and resumable.
2. Pre-run estimate (15.3 h) assumed the mean-cost batch (~158) throughout; the
   long tier caps at ~75 rows and runs ~3x slower per pass. The estimator needs
   per-tier terms; queued with the engine fixes.
3. Two interventions during the long tier are reported as such; the overnight
   unattended claim applies to the 15 h 35 m first leg (1,334 rows, zero touches).
4. Family gate run as identity AND mlx_lm agreement (stricter than directed).
5. Engine rates and memory calibration keyed per model|quant after two pollution
   incidents (page-cache-warm small models masquerading as streaming rates).
6. Removed the 70 GB 8-bit artifact mid-run to restore disk headroom (health-check
   directive); re-downloadable, and the registry still records it.
7. The projected-full-set figure is written as ~50 h (directive's draft wording
   said ~45 h; the last-ten-pass cadence supports 50, and no rounding down).
8. mlx-community 8-bit artifacts exist for all three curated tags (recorded in
   Part A); 4-bit likewise; neither affects the bf16 default.

## 6. Plain answers

**(a)** **1,334 rows completed unattended in 15 h 35 m** (the entire short and
summarize tiers); total elapsed to the deliberate stop was 19 h 52 m including
~35 min of intervention gaps.

**(b)** The pre-run estimate said 15.3 h for all 2,000; the honest projection is
~50 h: off by ~3.3x, because the estimate priced every row at the mean-cost batch
(~158) while the 1,000-token tier runs at batch ~75 with passes stretched to
~100 s by the memory fault storm, plus a ~25-minute serial prefill per group.

**(c)** A stranger sees the 70B generating text **41 seconds** after the pre-run
line (59 s from creating the venv, 11 s install included).

**(d)** pip from the GitHub URL on Python 3.12 and 3.14, and uv tool install, all
in fresh venvs, all running the packaged sample with no HF login and no prompts.

**(e)** Verified today: **Llama 3.x, Qwen2/2.5, Qwen3 (dense), Phi 3/4, Gemma 2,
Mistral** (the last four newly verified in Part A).

**(f)** The two engine defects the long tier exposed, stated as the next work:
**(1) peak-targeting memory budget:** admission sizes batches toward 85% of the
working set, but at ~95% occupancy macOS starts a clean-page fault storm that
quarters disk bandwidth; the budget must target the knee (~75-80% live
allocation), with backpressure as guard, not primary control. **(2) serial
prefill:** newcomers prefill one sequence at a time inside the weight-stream
pass (28 minutes for a 75-row long group); prefill GEMMs across admitted rows
must be batched, and scheduling should interleave lengths instead of saving the
most expensive tier for a single end-of-job wall.

## Appendix

- Partial results: `docs/reports/005-evals-partial-llama3.3-70b-bf16.jsonl.gz`
  (+ .header.txt); job 20261002-165546-e4a9b5 checkpointed at 1,334/2,000.
- Golden path: `docs/reports/005-golden-path.txt`. Prior: 004-phase1.5.md,
  002-phase1.md, 001-phase0.md.
