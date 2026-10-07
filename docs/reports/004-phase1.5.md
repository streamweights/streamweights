---
description: Phase 1.5 report: correctness checks and memory headroom for the streaming runner.
---

# Phase 1.5 report - correctness and headroom

Run 2026-10-02, Apple M4 Pro (48 GB RAM, Metal working set ≈ 36 GiB).
Directive: `docs/paste-sets/004-phase1.5-correctness.md`.

One-line conclusion: **the suspected EOS bug was not a bug (0% junk tokens - the
119 capped rows were genuine verbose answers hitting max_tokens), but the hunt
hardened stopping against the full declared EOS set; the memory budget is now
measured rather than theorized, admission is by memory with a physical-allocation
bound and cache compaction, bf16 stays the no-flags default at ~14.6–16.3 h for
the full 2,000 rows, and continuous refill took the golden path from 119 completed
rows at interrupt to 300.**

## 1. EOS finding

Distribution of the 119 completed 8-bit golden-path rows: **119/119 at exactly
128 tokens** (`finish_reason: "length"` on every row - already truthful). The
Phase 1 500-row sweep partials contained zero completed rows (timebox), so the
119 are the entire before-picture. Inspection shows coherent mid-sentence
truncations, not junk: Llama 3.3 answers these eval prompts verbosely and
max_tokens=128 genuinely cuts it off. Stopping itself was verified working:
earlier 70B smokes emitted `<|eot_id|>` and stopped at 3 tokens under a 6-token
cap, and the tokenizer had resolved eos to 128009 (the chat end-of-turn token).

What was genuinely wrong and is now fixed: the engine honored only the
tokenizer's single eos id, not the model's full declared set. `collect_eos_ids`
now unions `generation_config.json` + `config.json` (int or list - for Llama 3.x
that is {128001, 128008, 128009}), the tokenizer's ids, and the chat template's
end-of-turn token resolved through the vocabulary. Chat templating was already
exactly mlx-lm's (`tokenizer.apply_chat_template(messages,
add_generation_prompt=True)`).

New tests (both green): 20 short-QA prompts produce a short answer then a true
stop - **both engines stop at the same token with identical text, 20/20,
finish_reason "stop"** - and the same 20 cross-checked against independent
`mlx_lm.generate` **output length** (not just first token) at batch 1: 20/20
within one token. (Batched runs can flip near-tied tokens vs single-sequence - 
bf16 kernel-order effects, documented since Phase 1.)

## 2. Memory budget, measured

Why the Phase 1 formula missed (measured peaks 16.3 GB at batch 100, 18.9 GB at
128, 30.7 GB at 8-bit batch 119): it overcharged **activations** (a B×maxlen
term that is tiny during decode) and charged per-sequence KV at the job-wide
mean prompt length - while missing the two things that actually drive the peak:
**physical padded allocation** (every row's cache spans the batch's common
length, allocated in 256-token steps) and **merge transients**.

Calibration from probe runs at two batch sizes on the longest rows:

| quant | base | slope | probe peaks (B16 / B48) |
|---|---|---|---|
| bf16 | 7.6 GB | 284 KB per sequence-token | 12.9 / **24.6 GB** |
| 8bit | 9.3 GB | 548 KB per sequence-token | 19.9 / **42.5 GB** - proof of the old overshoot |

Admission now solves against 85% of the working set (30.6 GB) with THREE rules:
calibrated per-row cost, a physical-allocation bound (batch × 256-step-rounded
padded horizon × KV bytes/token), and cache **compaction** (dead left-pad columns
sliced once every active row's window passes 256 - rotary positions live in the
stored values, so this is exact). The second and third rules were added after the
golden path caught, in order: 512 short rows passing the per-row cost check, and
legal-at-admission batches crossing into the next allocation tier as the padded
length grew (passes degraded 36 s → 127 s). Post-fix 15-minute verify: **batch
locked at 167, steady pass median 34.0 s, no degradation, no OOM**; implied peak
≈ 21 GB - below the directive's 25–32 GB window because the physical bound is
quantized to 256-token tiers (filling into the window without risking the next
tier needs sub-256 allocation granularity; queued for Phase 2). The no-OOM
requirement holds with margin.

**New auto batch and 2,000-row estimates:** bf16 **~158–167**, est.
**14.6–16.3 h**; 8bit **~76**, est. **~17.6 h**. bf16 is now both the default
and the faster option - memory, not disk, binds the batch, and bf16's
per-sequence memory cost is about half of 8-bit's (quantized matmul scratch).

## 3. First results fast

The scheduler starts with the 8 shortest rows and ramps by admission. Measured:
0.5b first completed rows **~22 s** after the command. 70B first completed rows
**89 min 46 s** after the pre-run line. The 5-minute target is **missed, and it
is arithmetic, not implementation**: every row of this eval runs to its 128-token
cap, and a cap-length row needs 129 passes × ~34 s ≈ 73 min at bf16. The target
is only reachable for rows that stop within ~9 passes. The pre-run line, batch
arithmetic, and live per-pass progress all appear within the first minute, which
is the real anti-"is it hung" signal; first-row latency for cap-length evals is a
physics floor of pass_time × max_tokens.

## 4. Continuous refill

Implemented as newcomer prefill inside the same weight-stream pass as the batch's
decode step (the dedicated-step variant was not separately measured - Decision),
with the newcomer's prompt end aligned to the batch's next write slot so RoPE
positions stay exact. Before (Phase 1, chunked, batch 100): 2.58 tok/s wall.
After, same model same batch, 30-minute window: 2.7 tok/s - modest in a window
that ends before most completions, because refill's structural win is at chunk
boundaries. The clearest measurement: the golden-path 70B run reached **6.9 tok/s
aggregate at pass 130 with 300 rows completed** (the Phase 1 scheduler banked 119
rows at the same point in its run and then drained); VERIFY3's steady state
implies ~4.9 tok/s (167/34.0 s) at the fixed auto batch on the 500-row mix.

## 5. Progress line

Shipped exactly as specified, one updating line, ETA from calibration refreshed
every pass: `rows 0/500 · pass 21 (34.1 s) · 3.6 tok/s · ETA 4h 52m · bf16 ·
batch 167`. Known gap: it does not show peak memory (minor defect, queued).

## 6. Plain answers

**(a) Was the EOS bug real, and what fraction of Phase 1 output tokens were
junk?** The stopping bug was **not real** - finish_reason was truthful, stopping
fired on real EOS, and **0% of output tokens were junk**; the 119 capped rows
were genuine answers truncated by the eval's own max_tokens. What was real and
is fixed: the declared-set gap (only 128009 of {128001, 128008, 128009} was
honored) - it had produced no wrong output on this eval but was a latent landmine.

**(b) New auto batch and 2,000-row estimates:** bf16 ~158–167 → **14.6–16.3 h**;
8bit ~76 → **~17.6 h**.

**(c) Does bf16 stay the no-flags default?** **Yes** - measured at under 24 h for
the full set, confirmed live in the golden path: `quant: bf16 (batch-tier
default)`, `est. 16.3 h`.

**(d) Time to first completed 70B row in a fresh golden path:** **89 min 46 s**
from the pre-run line (install → pre-run line was under 30 s; 0.5b first rows in
22 s on the way). See §3 for why, and `docs/reports/004-golden-path.txt`.

## 7. Decisions

1. **The "EOS bug" investigation concluded no stopping bug** - evidence: smoke
   runs stopping at `<|eot_id|>`, truthful finish_reasons, coherent truncation
   tails; fixes made anyway (declared set, tests) because the gap was real.
2. **Static worst-case batch sizing would have pushed BOTH quants over 24 h**
   (bf16 ~62 → 37 h, 8bit ~30 → 45 h); memory-budget admission with per-row
   costs was chosen instead and keeps bf16 under 24 h.
3. **Two admission defects found by running the golden path, fixed in-session**:
   per-row cost without the physical-allocation term (512 rows admitted), and
   allocation-tier growth under refill (fixed by cache compaction).
4. **Memory calibration re-keyed** by model|quant after the 0.5b read 70B numbers.
5. **Probes run on the longest rows** (conservative) with max_tokens=8; admission
   scales their slope by each row's actual prompt+max_tokens.
6. **Refill variant**: same-pass newcomer prefill implemented; the
   every-N-passes variant was not separately measured (time); the same-pass
   variant adds no extra weight streams, which is the dominant cost.
7. **Verify peak lands ~21 GB, below the 25–32 GB window** - under-fill from
   256-token allocation tiers; the no-OOM half of the requirement holds. Finer
   granularity queued for Phase 2.
8. **5-minute first-row target declared unreachable for cap-length rows** with
   the floor arithmetic stated in §3, rather than gamed (e.g., by silently
   truncating early rows).
9. Minor defects open: no peak-memory in the progress line; engine errors still
   surface as tracebacks.

## Appendix

- Calibration: `docs/reports/data/calibration-m4pro.json` (read-rate grid, engine rate, mem_model).
- Golden path: `docs/reports/004-golden-path.txt`.
- Prior: `docs/reports/002-phase1.md`, `docs/reports/001-phase0.md`.
