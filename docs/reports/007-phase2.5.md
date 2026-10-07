---
description: Phase 2.5 report: long-tail engine fixes and a completed 2,000-row evaluation on Llama 3.3 70B bf16.
---

# Phase 2.5 report: long-tail engine fixes and the completed 2,000-row set

Run 2026-10-03/04, Apple M4 Pro (48 GB RAM, Metal working set 36 GiB).
Directive: `docs/paste-sets/007-phase2.5-long-tail.md`.

One-line conclusion: **the long tier that previously produced zero completions in
25-minute timeboxes now runs clean passes at short-tier speed (33-37 s), and the
full 2,000-row eval on Llama 3.3 70B bf16 is complete: 1,334 rows in 15 h 35 m
unattended on the Phase 1.6 engine plus all 666 long-document rows in 23 h 51 m
with zero interventions on the Phase 2.5 engine, peak memory pinned at 28.9 GB
throughout.**

## 1. Before / after (items 2 to 4)

| metric (long tier, 70B bf16) | before (Phase 1.6 engine) | after (Phase 2.5 engine) |
|---|---|---|
| clean decode pass | 86-153 s, degrading mid-group | **33-37 s, flat for 24 h** |
| pass with refill prefill | n/a (28-min serial prefill wall per group) | 45-90 s, capped at 2,048 prompt tokens/pass |
| prefill cost per 75 long rows | **1,706 s in one blocking pass** (serial per-sequence) | spread across ~35 passes alongside decode; no blocking pass longer than ~90 s |
| peak memory | 34.3-36.8 GB (over the working set; paging) | **28.9 GB, constant** (75% target + 85% live guard + cache compaction) |
| page-ins | counter re-interpreted: noncached preads count as page-ins, so the stream itself reads as ~2-3.7 GB/s; the real before-signal was 1.4 GB/s effective in ~170 KB fragments | 2.6-3.7 GB/s in large reads, matching healthy streaming |
| batch | 75 admitted, legal-at-admission but tier-crossing | 39-50, bounded by budget, phys tier, and backpressure |
| completions | 0 rows in any 45-min window | 27.9 rows/hour sustained |

Scheduling (item 4) is now longest-first with per-length-bucket pre-run estimates:
the padded cache length is set by the head of the queue, every later row admits
freely, and the expensive rows are spread through the run instead of walled at
the end.

## 2. The completed run (item 5)

- Rows 1-1,334 (all short-QA and summarize-300w): Phase 1.6 engine, unattended,
  2026-10-02 16:55:43 to 2026-10-03 08:25.
- Rows 1,335-2,000 (all 666 doc-QA-1k): Phase 2.5 engine, 2026-10-03 14:49:33 to
  2026-10-04 14:40:19 = **23 h 51 m, zero interventions**, 0.99 tok/s aggregate,
  27.9 rows/h, peak 28.9 GB, stream 2.6-3.7 GB/s.
- Combined results: `docs/reports/007-evals-2000-llama3.3-70b-bf16.jsonl.gz`
  (header file beside it states which rows ran on which engine version).
- Distributions, all 2,000 rows: finish_reason 685 stop / 1,315 length; output
  length min 87, median 128, max 128 (the eval's max_tokens=128 cap is the
  dominant stop; summarize rows stopped genuinely on 97%, long doc-QA rows on 5%).
- Long-tier row latency (admission to completion): p50 93 min, p95 200 min; a row
  cannot finish faster than max_tokens × pass_time, so job completion trails the
  last admission by about two hours (see Decisions).

## 3. Plain answers

**(a) Does the long tier now run at short-tier pass times?** Clean decode passes:
**yes** (33-37 s against the 34 s short-tier reference, within the directive's
10%). Effective cadence is 45-90 s on the passes that also carry a newcomer's
prompt, because prefill compute shares the pass; with ~1 completion per pass at
steady state, most passes carry some prefill. What remains: batch the prefill
GEMMs across newcomers instead of per-sequence calls, and sub-256-token cache
granularity so admission can fill closer to the 75% target.

**(b) Wall time for the 666 long rows on the new engine:** **23 h 51 m**.

**(c) Printed estimate versus actual:** the resume path printed no banner
estimate at all (a defect, now on the fix list); the live progress ETA about an
hour into the leg read ~16 h against 23.85 h actual, a ratio of about **0.67
(1.5x optimistic)**. The gap is the end-tail effect (the job cannot finish sooner
than max_tokens × pass_time after the last admission) plus prefill-carrying
passes running above the clean-pass time the ETA extrapolates from.

**(d) Anything that required intervention?** During the completed leg: **nothing**,
for 23 h 51 m. During items 2-4 development before the final leg, disclosed in
full: one 30-minute instrumented verification leg (deliberate SIGINT), and one
diagnostic stop after the verification exposed that an earlier 75-row admission
predated the physical-tier fix; both ended in clean checkpoints with zero lost
rows. Separately, background watcher tasks were repeatedly culled by the session
harness and re-armed on each kill notification; the run itself never noticed.

## 4. Decisions

1. **Longest-first realizes "length-interleaved"**: descending order sets the
   padded length once, lets every subsequent row admit without the length gate,
   mixes lengths through refill, and eliminates both end-walls and
   group-boundary drains. Logged as a deviation in letter, not in goal.
2. **"Page-ins under 100 MB/s" was unmeasurable as specified**: macOS counts
   noncached preads as page-ins, so the healthy stream itself reads as GB/s. The
   Phase 1.6 "fault-storm" diagnosis is partly corrected: the real before-problem
   was over-allocation stalls and fragmented reads, visible as 170 KB average
   transfer size versus 64 MB chunks now.
3. **F_NOCACHE/mmap/ring checklist ran and passed** (item 2's contingency): all
   29 shard fds carry F_NOCACHE, nothing mmaps the weights, ring buffers intact.
4. **Verification gate judged on clean passes**: the 30-minute window never left
   the admission ramp, so the gate was confirmed on no-admission passes (33.2 s,
   25.1 GiB) and then validated across the full 24-hour leg.
5. **The end-tail is a scheduling fact to fix upstream**: admission stopped when
   pending emptied at ~pass 1663 and the job then needed ~128 more passes for the
   youngest rows. A future scheduler can oversubscribe the tail or shrink batch
   target as pending drains.
6. **Two stale-looking stretches investigated, both benign**: completions arrive
   in bursts aligned to admission waves, so hours with few completions while
   passes advance are expected, not stalls.
7. Total wall-clock span for the whole 2,000-row dataset, counting everything
   (the Phase 1.6 era stall, two Phase 1.6-era interventions, the 6.4 h
   engineering gap between legs, and both engine legs) was 45 h 45 m; engine-busy
   time was about 40 h. Both numbers are honest; the quotable per-engine numbers
   are the per-leg ones above.

## Appendix

- Combined results + header: `docs/reports/007-evals-2000-llama3.3-70b-bf16.*`
- Prior reports: 005-phase1.6.md (partial close), 004-phase1.5.md, 002-phase1.md,
  001-phase0.md. Job directory: jobs/20261002-165546-e4a9b5 (complete).
