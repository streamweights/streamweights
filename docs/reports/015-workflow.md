# 015: one complete workflow

Measured by `scripts/gates_015.py`, which runs the public CLI end to end per task and per engine, on the permitted small models, and writes `docs/reports/data/015-gates.json`; this page is rendered from that file by `scripts/make_report_015.py`.

Hardware: Apple M4 Pro, 48.0 GB, macOS 24.6.0 arm64, Python 3.12.13. Run started 2026-10-07T20:35:18-0700. Each journey is a fresh project made from the tiny example of its task (120 rows: 84 train, 18 validation, 18 final test, seed 0) and uses `qwen2.5:0.5b` bf16 as the student.

## Models and revisions

| role | tag | repository | revision |
|---|---|---|---|
| embedding | embedding:minilm | sentence-transformers/all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| student | qwen2.5:0.5b | Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` |

Dependencies of the run: mlx 0.32.3, mlx-lm 0.32.0, peft 0.21.2, safetensors 0.8.0, streamweights 0.1.0, torch 2.14.1, transformers 5.19.0.

## Results on the validation rows

### classification

| engine | comparator | accuracy | rows |
|---|---|---|---|
| mlx | embedding baseline | 1.000 | 18 |
| mlx | student, prompted, untrained | 0.333 | 18 |
| mlx | student, trained | 0.889 | 18 |
| torch-cpu | embedding baseline | 1.000 | 18 |
| torch-cpu | student, prompted, untrained | 0.444 | 18 |
| torch-cpu | student, trained | 0.944 | 18 |

### json

| engine | comparator | whole-record accuracy | rows |
|---|---|---|---|
| mlx | student, prompted, untrained | 0.000 | 18 |
| mlx | student, trained | 0.500 | 18 |
| torch-cpu | student, prompted, untrained | 0.000 | 18 |
| torch-cpu | student, trained | 0.444 | 18 |

Eighteen rows is small: one row is 0.056, and the two engines train with different numerics, so their numbers differ by more than that. No significance test was run.

## Durations (wall seconds on this machine)

| task | engine | build | test | export (safetensors + GGUF q8_0, verified) | train stage | embedding fetch included |
|---|---|---|---|---|---|---|
| classification | mlx | 13.8 s | 5.1 s | 19.6 s | 8.3 s | yes (first journey of a fresh cache) |
| classification | torch-cpu | 36.5 s | 7.6 s | 19.5 s | 26.59 s | no |
| json | mlx | 17.9 s | 8.2 s | 23.2 s | 11.26 s | no |
| json | torch-cpu | 53.7 s | 19.7 s | 24.0 s | 32.34 s | no |

## Final test (scored once per journey by `spill test`)

| task | engine | comparator | score | rows |
|---|---|---|---|---|
| classification | mlx | embedding baseline | 0.944 | 18 |
| classification | mlx | student, prompted, untrained | 0.333 | 18 |
| classification | mlx | student, trained | 0.833 | 18 |
| classification | torch-cpu | embedding baseline | 0.944 | 18 |
| classification | torch-cpu | student, prompted, untrained | 0.333 | 18 |
| classification | torch-cpu | student, trained | 0.833 | 18 |
| json | mlx | student, prompted, untrained | 0.000 | 18 |
| json | mlx | student, trained | 0.333 | 18 |
| json | torch-cpu | student, prompted, untrained | 0.000 | 18 |
| json | torch-cpu | student, trained | 0.556 | 18 |

## Export verification

The merged bf16 safetensors is loaded by Transformers (float32 on the CPU) and the GGUF q8_0 by llama.cpp, each in its own process, on the first 8 validation rows. A difference is a row where the exported artifact's output text differs from the training engine's output for the trained student.

| task | engine | artifact | runtime | rows that differ | primary metric on the 8 rows |
|---|---|---|---|---|---|
| classification | mlx | gguf:q8_0 | llama.cpp version: 0.5.0-dev (build 11311, commit f7b384c1e) | 0 of 8 | 0.875 |
| classification | mlx | safetensors | transformers 5.19.0, torch 2.14.1, float32 on CPU (12 cores) | 0 of 8 | 0.875 |
| classification | torch-cpu | gguf:q8_0 | llama.cpp version: 0.5.0-dev (build 11311, commit f7b384c1e) | 0 of 8 | 1.000 |
| classification | torch-cpu | safetensors | transformers 5.19.0, torch 2.14.1, float32 on CPU (12 cores) | 0 of 8 | 1.000 |
| json | mlx | gguf:q8_0 | llama.cpp version: 0.5.0-dev (build 11311, commit f7b384c1e) | 4 of 8 | 0.250 |
| json | mlx | safetensors | transformers 5.19.0, torch 2.14.1, float32 on CPU (12 cores) | 4 of 8 | 0.250 |
| json | torch-cpu | gguf:q8_0 | llama.cpp version: 0.5.0-dev (build 11311, commit f7b384c1e) | 2 of 8 | 0.625 |
| json | torch-cpu | safetensors | transformers 5.19.0, torch 2.14.1, float32 on CPU (12 cores) | 3 of 8 | 0.375 |

For the extraction task the exported artifacts differ from the training engine on several rows. The adapter is merged into bf16 weights, which rounds away part of a small adapter delta, and the runtimes differ in numerics; the rows that differ are listed in each export record. This is the measured size of the gap, not a claim that it is zero.

## Deployment measurements (inference only, separate from build time)

| task | engine that built it | artifact | cold time to first token | warm | warm tokens/s | peak memory |
|---|---|---|---|---|---|---|
| classification | mlx | gguf:q8_0 | 0.0316 s | 0.0204 s | 119.81 tokens/s | 0.75 GB |
| classification | mlx | safetensors | 0.0819 s | 0.0649 s | 33.69 tokens/s | 3.41 GB |
| classification | torch-cpu | gguf:q8_0 | 0.0245 s | 0.0208 s | 119.96 tokens/s | 0.75 GB |
| classification | torch-cpu | safetensors | 0.0826 s | 0.066 s | 32.86 tokens/s | 3.41 GB |
| json | mlx | gguf:q8_0 | 0.0253 s | 0.021 s | 204.48 tokens/s | 0.75 GB |
| json | mlx | safetensors | 0.0961 s | 0.0665 s | 49.04 tokens/s | 3.42 GB |
| json | torch-cpu | gguf:q8_0 | 0.0332 s | 0.021 s | 203.09 tokens/s | 0.75 GB |
| json | torch-cpu | safetensors | 0.0805 s | 0.0676 s | 49.96 tokens/s | 3.41 GB |

Boundaries, Transformers: time to first token is wall time from request to the first streamed content token, cache_prompt off; cold = the first row, warm = median of the rest; tokens per second is completion tokens / wall time of the non-streamed request; peak memory is peak resident set of the llama-server process tree sampled after each request (a lower bound). For llama.cpp: wall time from request to the first streamed content token, cache_prompt off; cold = the first row, warm = median of the rest; peak memory is peak resident set of the llama-server process tree sampled after each request (a lower bound). Hardware: the machine above; llama.cpp runs with all layers on the Metal GPU, Transformers on the CPU.

## Inference through the exported artifact

- classification / mlx / gguf: `supported_cards_and_currencies` in 0.5 s (exit 0), input `Can I get a new pin number at any ATM using my card?`
- classification / mlx / safetensors: `supported_cards_and_currencies` in 2.7 s (exit 0), input `Can I get a new pin number at any ATM using my card?`
- classification / torch-cpu / gguf: `change_pin` in 0.5 s (exit 0), input `Can I get a new pin number at any ATM using my card?`
- classification / torch-cpu / safetensors: `change_pin` in 2.7 s (exit 0), input `Can I get a new pin number at any ATM using my card?`
- json / mlx / gguf: `{"intent":"SearchCreativeWork","object_name":"Written in the Stars"}` in 0.6 s (exit 0), input `Show me the picture Written in the Stars`
- json / mlx / safetensors: `{"intent":"SearchCreativeWork","object_name":"Written in the Stars"}` in 3.0 s (exit 0), input `Show me the picture Written in the Stars`
- json / torch-cpu / gguf: `{"intent":"SearchCreativeWork","object_name":"Written in the Stars","object_type":"picture` in 0.6 s (exit 0), input `Show me the picture Written in the Stars`
- json / torch-cpu / safetensors: `{"intent":"SearchCreativeWork","object_name":"Written in the Stars","object_type":"picture` in 3.1 s (exit 0), input `Show me the picture Written in the Stars`
