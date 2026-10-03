# streamweights

Run the unmodified full-size model against your eval set, on your machine, for free, overnight.

## Why

A model bigger than your machine normally means renting GPUs, and the moment you rent, the whole development loop changes: data leaves the building, iteration gets a meter on it, and "just run the eval again" becomes a budget conversation.

streamweights streams the full bf16 model from disk in layer order at drive speed instead of holding it in memory. One forward pass reads every weight once, whether the batch holds one prompt or two hundred - so each prompt advances one token per pass, a single answer takes minutes, and hundreds of prompts per pass are what make the total worthwhile.

It is for developers who need exact, unquantized results on an eval set and don't want data leaving the machine. It is not a chat tool.

## Requirements

- Apple silicon Mac. Tested on 48 GB (M4 Pro). The memory budget is calibrated at runtime; on 32 GB the same math leaves roughly half the batch headroom (expect ~2× the wall time for streamed models), and on 16 GB the KV budget after the ring and resident tensors is small enough that streamed 70B batches will be tiny - it should run, but overnight becomes days; untested.
- Free disk per curated model, plus a 20 GB floor that downloads never cross:
  - llama3.3:70b: 141 GB bf16 (first download: 141 GB), 75 GB 8-bit; needs ~151 GB+ free for bf16.
  - qwen2.5:32b: 65 GB bf16, 35 GB 8-bit; needs ~81 GB+ free.
  - qwen2.5:0.5b: 1 GB bf16; needs ~21 GB+ free.
- Python 3.10 or newer.
- Linux and Intel Macs get the llama.cpp path today, which is slow and not the point yet.

## Quick start

```
pip install git+https://github.com/streamweights/streamweights
spill run llama3.3:70b sample
```

Real output from the machine above (pre-run line, then the live frame that redraws every pass):

```
spill: llama3.3:70b bf16 (131.4 GB, Llama 3.x: verified) does not fit in 48 GB RAM; streaming from NVMe at ~4.0 GB/s (measured engine rate). 20 prompts, batch 20, est. 5 min. Cost: $0. Results -> jobs/20261002-163536-e076cf/results.jsonl (tail with: spill tail)
   why: batch ~20 (admission by memory, not count): (0.85×36.0G target − 7.1G measured base) ÷ (277 KB/seq-token × 91 mean tokens (actual prompt lens + 48 max_tokens)); rows admitted while calibrated cost fits

rows 0/20 · pass 1 (37.5 s) · 0.5 tok/s · ETA 30m 09s · bf16 · batch 20 · ~37.5 s per token per prompt · 8.7 GB peak
  hamlet-author         1 tok │ The
  continents            1 tok │ There
  mona-lisa             1 tok │ The
  capital-france        1 tok │ The
  smallest-prime        1 tok │ The
  red-planet            1 tok │ The
  seven-times-eight     1 tok │ 7
  largest-ocean         1 tok │ The
```

Your own eval set is OpenAI batch JSONL, one request per line:

```
spill run llama3.3:70b your-evals.jsonl
```

```json
{"custom_id": "q1", "body": {"messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 48}}
```

Any Hugging Face repo id with a supported architecture also works:

```
spill run org/name your-evals.jsonl
```

## Models and architectures

| tag | params | family (state) | bf16 | 8-bit | on 48 GB | disk needed | 
|---|---|---|---|---|---|---|
| qwen2.5:0.5b | 0.5B | Qwen2/2.5 (verified) | 0.9 GB | 0.7 GB | resident | 21 GB+ |
| qwen2.5:32b | 32B | Qwen2/2.5 (verified) | 61 GB | 33 GB | streamed | 81 GB+ |
| llama3.3:70b | 70B | Llama 3.x (verified) | 131 GB | 70 GB | streamed | 151 GB+ |

Architecture families (`spill models --architectures` prints the live table):

| state | families | 
|---|---|
| verified | Llama 3.x · Qwen2/2.5 · Qwen3 (dense) · Phi 3/4 · Gemma 2 · Mistral |
| expected | Gemma 3 (text) - per-layer alternating sliding-window attention needs per-layer cache windows; not wired yet |
| not_yet | Mixtral, DeepSeek V2/V3, Qwen MoE, Llama 4 - mixture-of-experts: per-token expert routing defeats layer-order streaming. Multimodal (anything with a vision tower) - vision towers are not streamed. Mamba - state-space recurrence has no KV cache to batch around. Jamba - recurrent layers break the per-layer stream loop. |

"Verified" means the streamed output was tested identical to in-memory execution on 20 prompts, plus first-token agreement with an independent mlx-lm implementation. A family moves from expected to verified when a small model of that family passes both tests.

## What to expect on a 48 GB M4 Pro

| model | quant | placement | pass time | auto batch | per-token / prompt | first visible text | aggregate tok/s | 2,000-row set |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:0.5b | bf16 | resident | ~0.1 s | up to 512 | ~0.1 s | < 10 s | 700+ | ~6 min (measured) |
| llama3.3:70b | bf16 | streamed | ~34 s | ~158 | ~34 s | ~40 s (sample) | ~4.6 steady | 14.6–16.3 h (estimated) |
| llama3.3:70b | 8-bit | streamed | ~19 s | ~76 | ~19 s | ~25 s (sample) | ~4.0 steady | ~17.6 h (estimated) |

## Usage

- `spill run <model> <input.jsonl|sample>` - flags: `--quant 8bit|4bit|Q8_0|Q4_K_M`, `--out path`, `--context N`, `--parallel N` (override, never required), `--quiet` (no live block), `--debug` (tracebacks).
- `spill tail [job]` - follow a job's results; shows the live block while it runs.
- `spill resume [job]` - continue the latest or named job from its checkpoint.
- `spill status` - all jobs with progress, tokens/s, ETA.
- `spill models` / `spill models --architectures` - the tables above, live.

Results land in `jobs/<id>/results.jsonl` (OpenAI batch output shape plus a `streamweights` metadata object). `finish_reason` is `stop` when the model emitted a real end-of-turn token and `length` when the row hit its own max_tokens. Close the laptop whenever: completed rows are checkpointed per row, and `spill resume` continues from exactly there.

## How it works

The safetensors headers are parsed into a per-layer byte index without loading any tensors. A ring of preallocated buffers is filled by pooled `pread` calls with `F_NOCACHE`, so weights move at drive speed and never pollute the page cache. Pass time is flat in batch size because one pass reads every weight exactly once no matter how many prompts share it. The memory budget is KV-bound and calibrated by probe runs at two batch sizes, then rows are admitted while their measured cost fits under 85% of the Metal working set. When a sequence finishes, the next pending row is prefilled inside the same weight-stream pass and takes over the slot. Streamed execution is bit-identical to running the same loop with all weights in memory.

## Status and roadmap

- Phase 0 (done): measured the mmap baseline - 11–13% of drive speed, 175 s per 70B pass.
- Phase 1 (done): streaming runner - 31.6 s per 70B bf16 pass at 79% of drive speed, identical-output gate 20/20.
- Phase 1.5 (done): measured memory calibration; bf16 holds as the no-flags default (~15 h for 2,000 rows).
- Next: the 2,000-row proof run, outside-developer testing, more verified families; then interactive tiers and adapters. Plan: [docs/plan.md](docs/plan.md).

## Feedback

If you try it, send a transcript of your first fifteen minutes and the one moment you got stuck. Open an issue or reply to whoever sent you the link.

## License

Apache-2.0.
