# Models and architectures

## Curated tags

| tag | params | family (state) | bf16 | 8-bit | on 48 GB | disk needed |
|---|---|---|---|---|---|---|
| qwen2.5:0.5b | 0.5B | Qwen2/2.5 (verified) | 0.9 GB | 0.7 GB | resident | 21 GB+ |
| qwen2.5:7b | 7.6B | Qwen2/2.5 (verified) | 14.2 GB | 7.5 GB | resident | 35 GB+ |
| qwen2.5:32b | 32B | Qwen2/2.5 (verified) | 61 GB | 33 GB | streamed | 81 GB+ |
| llama3.3:70b | 70B | Llama 3.x (verified) | 131 GB | 70 GB | streamed | 151 GB+ |

`qwen2.5:7b` is the default student for `spill build`; `llama3.3:70b` is the default
teacher. Any Hugging Face repo id with a supported architecture also works
(`spill run org/name file.jsonl`). `spill models` prints the live table for your machine.

Disk: each model needs its size plus a 20 GB floor that downloads never cross. Downloads
check the disk before the first byte, show one line with speed and ETA, and resume after an
interruption.

## Architecture families

`spill models --architectures` prints the live table.

| state | families |
|---|---|
| verified | Llama 3.x, Qwen2/2.5, Qwen3 (dense), Phi 3/4, Gemma 2, Mistral |
| expected | Gemma 3 (text): per-layer alternating sliding-window attention needs per-layer cache windows; not wired yet |
| not_yet | Mixtral, DeepSeek V2/V3, Qwen MoE, Llama 4: mixture-of-experts, where per-token expert routing defeats layer-order streaming. Multimodal models (anything with a vision tower): vision towers are not streamed yet (Phase 4). Mamba: state-space recurrence has no KV cache to batch around. Jamba: recurrent layers break the per-layer stream loop |

"Verified" means the streamed output was tested identical to in-memory execution on 20
prompts, plus first-token agreement with an independent mlx-lm implementation. A family
moves from expected to verified when a small model of that family passes both tests.
Shared-prefix reuse is wired for the dense families that take a causal mask string (all the
verified families except Gemma 2).

## Measured on a 48 GB M4 Pro

Eval decode on a streamed model is disk-bound; prefill and training are compute-bound
(2 and 6 times parameters times tokens, divided by achieved FLOP/s). Per-phase numbers live
in the reports under [docs/reports](reports).

| model | quant | placement | workload | pass time | measured |
|---|---|---|---|---|---|
| qwen2.5:0.5b | bf16 | resident | 2,000-row eval | ~0.1 s | ~6 min ([report 002](reports/002-phase1.md)) |
| llama3.3:70b | bf16 | streamed | 1,334 short and summarize rows | 33-38 s | 15 h 35 m unattended ([report 007](reports/007-phase2.5.md)) |
| llama3.3:70b | bf16 | streamed | 666 one-thousand-token rows | 33-37 s clean | 23 h 51 m, zero interventions ([report 007](reports/007-phase2.5.md)) |
| llama3.3:70b | 8-bit | streamed | sample | ~19 s | ([report 002](reports/002-phase1.md)) |
