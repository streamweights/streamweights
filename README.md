# streamweights

Run the unmodified full-size model against your eval set, on your machine, for free, overnight, with no data leaving the building.

## Golden path

```
pip install streamweights
spill run llama3.3:70b evals.jsonl
# spill: llama3.3:70b bf16 (131.4 GB) does not fit in 48 GB RAM; streaming from NVMe at ~4.0 GB/s (measured engine rate).
# 2,000 prompts, batch ~158, est. 14.6 h. Cost: $0. Results -> jobs/<id>/results.jsonl (tail with: spill tail)
```

The pre-run line shows this machine's real measured numbers (Apple M4 Pro, 48 GB RAM; the streaming engine sustains ~4.0 GB/s of its 5.1 GB/s NVMe during decode).

## Status

Phase 1.5 complete. On a 48 GB M4 Pro the runner streams llama3.3:70b bf16
(141 GB) from NVMe at 79% of the drive's probed rate during decode (Phase 0's
mmap managed 11–13%): a forward pass went from 175 s to 31.6 s (5.5×), and from
zero completed rows in any Phase 0 timebox to a full overnight run — the 2,000-row
eval set is ~14.6 h at bf16 (the no-flags default) or ~17.6 h at 8-bit on this
machine, with memory-calibrated batch admission and continuous slot refill.
Streamed and resident execution produce identical greedy output. Reports:
docs/reports/001-phase0.md, 002-phase1.md, 004-phase1.5.md.

Plan: see [docs/plan.md](docs/plan.md).
