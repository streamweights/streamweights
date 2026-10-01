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

Phase 0 complete, Phase 1 (streaming runner) in progress. Phase 0 measured the mmap baseline on a model 1.5× RAM (llama3.3:70b Q8_0, 75 GB on a 48 GB machine): mmap paging reached only 11–13% of the drive's probed sequential rate, a single forward pass took 175 s on 70B Q8_0 (one full weight read per token), and batch sizes from 32 up OOM'd the GPU at 4k context. The streaming runner targets 70%+ of the sequential rate, which is roughly an 8–12× improvement. Full numbers: docs/reports/001-phase0.md.

Plan: see [docs/plan.md](docs/plan.md).
