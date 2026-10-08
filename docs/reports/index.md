---
description: Measured results for streamweights, one report per phase, each with the numbers and the files they came from. Streaming a 141 GB model on a 48 GB Mac, LoRA tuning, build and run-anywhere gates.
---

# Reports

Every number in the README and the guides comes from one of these reports, measured on an Apple M4 Pro with 48 GB unless the report says otherwise.

| report | what it measured |
|---|---|
| [001 Phase 0](001-phase0.md) | mmap on a model bigger than RAM reached 11 to 13% of the disk rate |
| [002 Phase 1](002-phase1.md) | the streaming runner: 141 GB at 79.2% of the disk rate |
| [004 Phase 1.5](004-phase1.5.md) | correctness and headroom |
| [005 Phase 1.6](005-phase1.6.md) | share-ready, and the proof run as a measured partial |
| [006 Phase 2](006-phase2.md) | the loop: distill, eval, adapters at inference |
| [007 Phase 2.5](007-phase2.5.md) | long-tail engine fixes and the completed 2,000-row set on the 70B |
| [008 Phase 3](008-phase3.md) | `spill tune`: streamed against resident LoRA, and the 70B smoke test |
| [009 Phase 3.5](009-phase3.5.md) | `spill build`, the example, export, prefix reuse |
| [010 Cleanup](010-cleanup.md) | the merge, and a fresh install timed |
| [012 Run anywhere](012-run-anywhere.md) | portable jobs, the PyTorch engines, cross-hardware resume gates |
| [013 Tagline, README, docs site](013-tagline-seo.md) | the tagline, the rewrite, the site and the decisions behind them (no measurements) |
| [014 Build anywhere](014-build-anywhere.md) | build on every engine, a build that moves between machines, the CI relay |
| [015 One complete workflow](015-workflow.md) | init, plan, build, report, test, export and inference per task and engine; verified exports; the ownership and handoff gates |
