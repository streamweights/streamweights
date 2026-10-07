---
description: Phase 3.5 report: spill build, the banking77 example, GGUF export and shared-prefix reuse, measured on the 0.5B.
---

# Phase 3.5 report: spill build, the example, export, prefix reuse

Run 2026-10-05, Apple M4 Pro (48 GB RAM, Metal working set 36 GiB), qwen2.5:0.5b only.
Directives: `docs/paste-sets/009-phase3.5-build.md`, finished under
`docs/paste-sets/010-cleanup.md`, which limits GPU use to the 0.5B. The 70B and 7B
proof builds in the 009 directive are not part of this report; they move to the full
proof run listed in `docs/plan.md`.

Files: `docs/reports/009-banking77/` (`prefix-ab-qwen2.5-0.5b.json`,
`export-verify.json`, `quick-build-state.json`).

## 1. Merge with Phase 3

Phase 3.5 was merged with main (a merge commit rather than a rebase, because the
branch is squashed afterward). One conflict, in `formats.check_file`: both sides
added a variable, and both are kept. Phase 3's engine and tune code is unchanged.
`spill build` now trains through `spill tune` (`cli._tune_build`), including resume.
The CPU suite is green.

## 2. Shared-prefix reuse, 20 rows (prefix-ab)

Banking77 prompts with the 77-label system prompt (486 prefix tokens), greedy.

| measure | result |
|---|---|
| output identical with and without reuse | yes, 20 of 20 |
| rows that differ with reuse | 0 |
| rows that differ between two unshared runs of different batch shape (noise floor) | 2 of 20 (r002, r013) |
| wall time, reuse / no reuse | 2.3 s / 3.5 s (1.2 s removed) |
| prefill tokens saved | 9,234 |

Reuse changes nothing. Ordinary batch-shape noise in bf16 is larger than any
difference reuse makes.

## 3. Export verification (export-verify)

`spill export qwen2.5:0.5b+banking77-quick --gguf q8_0`, 20 held-out prompts, greedy.

| check | result |
|---|---|
| merged model equals base+adapter | 18 of 20 identical |
| the two that differ | r017 (pending_transfer vs pending_card_payment), r008 (transaction_charged vs transaction_fee_charged) |
| GGUF (q8_0, 531 MB) runs in llama.cpp | yes |
| GGUF agrees with the engine | 19 of 20 |

The criterion "greedy output equals base+adapter" is met exactly on the CPU tests
(tiny models) and not on the 0.5B in bf16. The merge is computed in float32 and
rounded once to bf16, which is as exact as bf16 weights allow, but the adapter path
rounds the base product and the low-rank product separately, so near-ties between
similar labels can flip. Both differing rows are such ties, and the two-of-twenty
rate equals the batch-shape noise floor in section 2. See Decisions.

## 4. The quick example, banking77-quick

`spill example banking77 --quick && spill build banking77-quick`: student
qwen2.5:0.5b, 100 held-out rows, 500 training rows, 250 steps (2 epochs).

| model | role | score | rows |
|---|---|---|---|
| qwen2.5:0.5b+banking77-quick | your model | 0.640 | 100 |
| qwen2.5:0.5b | base (untrained) | 0.200 | 100 |

| stage | estimate | actual |
|---|---|---|
| eval base | 1 s | 0 s (reused the cached run of the same model and input) |
| tune | 28 s | 40 s |
| eval tuned | 1 s | 4 s |
| total | 29 s | 44 s |

The fresh-install run in `docs/reports/010-fresh-install.txt` is the same build with nothing
cached and the model downloaded on the way:

| model | role | score | rows |
|---|---|---|---|
| qwen2.5:0.5b+banking77-quick | your model | 0.640 | 100 |
| qwen2.5:0.5b | base (untrained) | 0.210 | 100 |

It took 79.5 s from `pip install` to the end of build, 60.9 s of that the build itself (stage 1
includes the 0.9 GB download, which the estimate does not count, so it reads 16 s against 1 s).
The base scored 0.210 here and 0.200 above: bf16 output shifts slightly with batch shape.

## Decisions

- **Merge, not rebase**, for the reason above; the squash keeps one commit on main.
- **Equality criterion at 0.5B bf16.** Not attainable bit for bit, as section 3
  explains. Reported as measured next to the noise floor; the float32 merge is kept.
- **Defects found and fixed on the way:** a folder named like its adapter (what
  `build` produces) shadowed the adapter in `+name` resolution; `python -m
  streamweights.cli` registered no build, example, export or doctor commands; the
  llama.cpp converter at the pinned tag imports a `conversion` package, so the whole
  converter source is now fetched from the tag archive; the tune estimate in `build`
  used the 70B's measured rate for the 0.5B and now uses the model's own rate; a stale
  interrupted-build entry from an earlier session no longer shows in the banner.
- **Proof runs on 70B and 7B** were not run, per the cleanup directive.
