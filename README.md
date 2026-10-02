# streamweights

Run the unmodified full-size model against your eval set, on your machine, for free, overnight, with no data leaving the building.

## Golden path

```
pip install streamweights
spill run llama3.3:70b evals.jsonl
# spill: llama3.3:70b bf16 (131.4 GB) does not fit in 48 GB RAM; streaming from NVMe at ~5.1 GB/s.
# 2,000 prompts, batch 7, est. 262 h. Cost: $0. Results -> jobs/<id>/results.jsonl (tail with: spill tail)
```

The pre-run line above shows this machine's real numbers (Apple M4 Pro, 48 GB RAM, 5.1 GiB/s NVMe) as measured in the Phase 0 report — on today's mmap path the estimate is honest and bad; Phase 1 exists to fix it.

## Status

Phase 1 (streaming runner, MLX) complete. On a 48 GB M4 Pro, the runner streams
llama3.3:70b bf16 (141 GB) from NVMe at 79% of the drive's probed rate during
decode (Phase 0's mmap managed 11–13%), cutting a forward pass from 175 s to
31.6 s (17.1 s at 8-bit) and running batch 128 where mmap OOM'd at 32. Streamed
and resident execution produce identical greedy output (20/20). The no-flags
golden path runs the full 2,000-row eval against the 70B overnight (~11 h at
8-bit, chosen and explained by the policy; bf16 would be ~25 h). Phase 0 baseline:
docs/reports/001-phase0.md. Phase 1 numbers: docs/reports/002-phase1.md.

Plan: see [docs/plan.md](docs/plan.md).
