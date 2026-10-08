
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
