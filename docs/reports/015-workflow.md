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

Measurement descriptions, each runtime's own. Transformers (safetensors, float32 on the CPU): time to first token is wall time of generate(max_new_tokens=1) including prompt prefill; cold = the first row, warm = median of the rest; tokens per second is completion tokens / wall time of the full generate() including prefill, median over rows after the first; peak memory is ru_maxrss of this process (weights, activations, tokenizer, interpreter), a process peak of resident memory and not a GPU-memory figure. llama.cpp (GGUF q8_0, all layers on the Metal GPU): time to first token is wall time from request to the first streamed content token, cache_prompt off; cold = the first row, warm = median of the rest; tokens per second is completion tokens / wall time of the non-streamed request; the sampled process resident set is a lower bound, not a total of GPU memory. Cold first-token timing excludes model loading (load time is recorded apart). The Metal q8_0 and CPU float32 numbers are not a format-only performance comparison: runtime, device and precision all differ.

## Inference through the exported artifact

- classification / mlx / gguf: `supported_cards_and_currencies` in 0.5 s (exit 0), input `Can I get a new pin number at any ATM using my card?`
- classification / mlx / safetensors: `supported_cards_and_currencies` in 2.7 s (exit 0), input `Can I get a new pin number at any ATM using my card?`
- classification / torch-cpu / gguf: `change_pin` in 0.5 s (exit 0), input `Can I get a new pin number at any ATM using my card?`
- classification / torch-cpu / safetensors: `change_pin` in 2.7 s (exit 0), input `Can I get a new pin number at any ATM using my card?`
- json / mlx / gguf: `{"intent":"SearchCreativeWork","object_name":"Written in the Stars"}` in 0.6 s (exit 0), input `Show me the picture Written in the Stars`
- json / mlx / safetensors: `{"intent":"SearchCreativeWork","object_name":"Written in the Stars"}` in 3.0 s (exit 0), input `Show me the picture Written in the Stars`
- json / torch-cpu / gguf: `{"intent":"SearchCreativeWork","object_name":"Written in the Stars","object_type":"picture` in 0.6 s (exit 0), input `Show me the picture Written in the Stars`
- json / torch-cpu / safetensors: `{"intent":"SearchCreativeWork","object_name":"Written in the Stars","object_type":"picture` in 3.1 s (exit 0), input `Show me the picture Written in the Stars`


## Linux, from the built wheel (GitHub Actions)

GitHub Actions run 37729069035, job 'The CSV journey from the built wheel (Linux, torch-cpu, GGUF)': the wheel `streamweights-0.1.0-py3-none-any.whl` is installed into a clean environment with the `cloud` extra and the CLI is run from outside the checkout (`streamweights` imported from `site-packages`); torch-cpu on the runner's CPU (x86_64), the same tiny examples.

| task | comparator | score | rows |
|---|---|---|---|
| classification | embedding baseline | 1.000 | 18 |
| classification | student, prompted, untrained | 0.444 | 18 |
| classification | student, trained | 0.944 | 18 |
| json | student, prompted, untrained | 0.000 | 18 |
| json | student, trained | 0.444 | 18 |

| task | artifact | rows that differ from the training engine | warm tokens/s | peak memory |
|---|---|---|---|---|
| classification | gguf:q8_0 | 0 of 8 | 12.88 tokens/s | 0.67 GB |
| classification | safetensors | 0 of 8 | 6.59 tokens/s | 3.43 GB |
| json | gguf:q8_0 | 3 of 8 | 34.47 tokens/s | 0.68 GB |
| json | safetensors | 3 of 8 | 8.9 tokens/s | 3.43 GB |


## Data and licenses

| example | source | license | what was done to it |
|---|---|---|---|
| `banking77` | BANKING77, PolyAI (Casanueva et al., 2020), `PolyAI/banking77` | CC BY 4.0: redistribution and adaptation with attribution; the example README carries the attribution | sampled, reshaped into prompt, answer and label fields, two label names normalized; a CSV of the same rows added |
| `snips` | Snips NLU benchmark, 2017-06 custom intent engines, `sonos/nlu-benchmark` at commit `b86ac7f1577868c42158d0dec77db50956046696` | CC0 1.0 Universal (`LICENSE-CC0.txt` is shipped); the benchmark's README asks that publications cite Coucke et al. 2018, kept in the example README | three intents; each utterance's human-annotated slots written as one flat JSON object; utterances with a repeated or missing slot left out; rebuilt by `scripts/make_snips_example.py` |

Models: `Qwen/Qwen2.5-0.5B-Instruct` and `sentence-transformers/all-MiniLM-L6-v2`, both Apache-2.0, pinned in `streamweights/models.yaml` with the sha256 of every file the workflow reads. The embedding model's required files total 91,567,797 bytes.

## What the gates are, and where each is checked

| gate | checked by |
|---|---|
| fresh install of the built wheel, public CLI outside the checkout | `scripts/fresh_wheel.py`, CI job `journey` |
| CSV journey: init, plan, build, report, export, inference through the exported artifact | `scripts/gates_015.py` (this report), CI job `journey` |
| classification and JSON on MLX and torch-cpu; every advertised export format (safetensors, GGUF q8_0) | this report |
| malformed data, ambiguous task, transitive duplicate and group constraints, supplied-split overlap, unusable small splits, conflicting labels, no final-test targets in training, planning, schema inference or build reports | `tests/test_project_data.py`, `tests/test_project_build.py` |
| stop mid-training and resume between MLX and torch-cpu in both directions; optimizer and cursor restored; engine change recorded; incompatible data, weights and settings rejected with named differences | `tests/test_project_cross_engine.py`, `tests/test_project_build.py`, `tests/test_move.py` |
| termination during payload writing and before or after control publication | `tests/test_ownership.py` (process killed at each point, on disk and MinIO), `tests/test_project_cross_engine.py` (the real engine) |
| immutable history, completion fencing, stale workers | `tests/test_project_build.py`, `tests/test_ownership.py`, `tests/test_move.py` |
| comparisons rank only compatible runs | `tests/test_compare_records.py` |
| offline bundle, reacquisition, missing and corrupted assets | `tests/test_bundle.py` |
| MinIO: ownership races, handoff, interrupted handoff, source fenced, two locations never writable | `tests/test_ownership.py`, `tests/test_move.py`, CI job `object-store` |
| separate-process backend | `tests/test_project_subprocess.py` |
| documented commands | `tests/test_docs_commands.py` (parse against the CLI; small-model commands executed with `SPILL_DOCS_RUN=1`; CI job `docs-commands`), `tests/test_docs.py` (links, numbers, em-dashes) |

## Decisions

- `spill init <data> --input <col> --output <col>`: the data file is the argument and the flags name its columns; `--project` names the folder (default: the file's name). Supplied `--val` and `--test` use the same columns.
- The config is TOML (`tomllib`, `tomli` on Python 3.10, `tomli-w` to write), the schema validator is `jsonschema`, and the logistic regression is written in PyTorch with scikit-learn's loss, so no new heavy dependency is needed.
- The baseline embedding is `sentence-transformers/all-MiniLM-L6-v2` through Transformers (mean pooling, unit length), 91.6 MB.
- The prompted untrained student and the teacher see the contract (labels or schema) in the system prompt; the trained student is trained and evaluated on the input alone (plus your own `--system` text). Both templates are in the protocol fingerprint.
- The protocol's precision field records the requested policy (bf16 base weights, float32 adapter), not each engine's actual numerics, so MLX and torch-cpu runs are comparable; the actual numerics are recorded per stage and shown by `spill compare`.
- A run's identity hashes the data, prompts, contract, training settings, evaluation protocol and the sha256 of every model file. It does not include the engine or machine: a continuation on another engine is the same run with the transition recorded.
- A stage is accepted whole. Training resumes from its last committed checkpoint; an eval or distill stage that is stopped starts over (row-level resume remains a property of the older commands).
- The engine's own checkpoint is copied into an immutable payload by the coordinator and published through the control object; the engine's local directories are staging.
- Local authority is `flock` on a stable lock file plus an atomic replace; S3 authority is `boto3` conditional writes. Payloads go through the existing fsspec store to unique paths and are never overwritten.
- The MinIO release tag `RELEASE.2025-10-15T17-29-55Z` is the last community source release; its binaries and images are no longer served by MinIO, so CI pulls the upstream image if it still exists and otherwise builds the same tag from source (`docker/minio.Dockerfile`) and logs `minio --version`.
- Completed records are read-only files in writable directories, so a project can still be deleted; the fenced completion is the protection, the file mode is only extra.
- A bundle is a folder (not an archive); bulk export artifacts stay out of `spill move` and `spill bundle` unless `--with-exports`.
- `spill test` re-evaluates every comparator of the run on the test rows with the same stage code, and counts uses across the project.
- Export verification uses Transformers (float32, CPU) for safetensors and llama.cpp for GGUF, each in its own process; mlx-lm is not used.
- A parent run is recorded when a new run follows a completed or bundled one; the new run trains from the base, not from the parent's adapter.
- The docs harness found that `spill tune <distill file>` failed although the guide said to run it; `spill tune` now accepts a `spill distill` output.

## MinIO

The S3 gates ran in CI (job `object-store`, run 37729069035) against MinIO built from the pinned tag, because the upstream image for it could not be pulled (`unauthorized`): the log reads `minio version DEVELOPMENT.2025-10-15T17-29-55Z (commit-id=9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a)`, and the same commit is what the Homebrew release build `RELEASE.2025-10-15T17-29-55Z` used locally. `tests/test_ownership.py` and `tests/test_move.py` ran there with the S3 cases enabled (41 passed, one skipped: a lost response is not a local-disk failure). Real AWS S3 was not exercised.

## Untested

Real AWS S3, CUDA, network filesystems, Windows, whole-machine power loss, a 7B or 70B model (none was run), warm-starting from a parent run, and the Linux MLX path (there is none). MLX in the pytest suite runs on the CPU device; the numbers in this report are from the Apple GPU.

## Addendum (directive 016)

The gaps this report left open were closed afterwards and are reported in [016](016-close-gaps.md): the offline bundle gate with the network blocked (Linux container with `--network none`, and a macOS sandbox), and the continuation between MLX on the Metal GPU and torch-cpu in both directions, through local and MinIO moves. The numbers above are unchanged.

## Update (directive 017)

The JSON results in this report were scored with metric version 1, which credited some schema-invalid outputs. They are marked **not rescorable** (no per-example predictions were saved) in [017](017-scorer-and-protocol.md), which also holds a replication under metric version 2; classification results are unaffected. The deployment measurement descriptions above were corrected there: the Transformers path now uses its own description, and sampled process memory is described as a lower bound.
