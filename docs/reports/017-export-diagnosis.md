# 017: JSON export differences, diagnosed

Rule for the default merge dtype, fixed before this comparison ran: `docs/reports/017-export-default-rule.md`. Script: `scripts/diagnose_export.py`; raw results `docs/reports/data/017-export-*.json`.

## What was held identical

For each of four completed projects (classification and JSON, each built on MLX and on torch-cpu): the verification rows (the first 8 validation rows of the run), the input messages, the rendered prompts (rendered once by the Transformers chat template and passed to every runtime as token ids, so tokenization is identical), greedy decoding and `max_tokens`. Hardware for every arm is the CPU: Transformers in float32, llama.cpp with no GPU layers. The source engine is the engine that trained the run and produced its saved predictions, rescored with the current scorer on these rows.

### classification, built on mlx

Metric: accuracy (version 1), 8 verification rows. Source engine: mlx on apple-gpu:Apple M4 Pro, numerics {"adapter": "float32", "base": "bf16"}, score 0.875 on these rows.

| arm | size | peak memory (process RSS) | task score (8 rows) | source score | change | text differs from source engine | schema-valid rate |
|---|---|---|---|---|---|---|---|
| unmerged adapter on the base (PEFT, Transformers float32) | 988 MB | 3.41 GB | 0.875 (7 of 8) | 0.875 | +0.000 | 0 of 8 | n/a |
| float32 merge (Transformers float32) | 1976 MB | 2.48 GB | 0.875 (7 of 8) | 0.875 | +0.000 | 0 of 8 | n/a |
| bf16 merge (Transformers float32) | 988 MB | 3.41 GB | 0.875 (7 of 8) | 0.875 | +0.000 | 0 of 8 | n/a |
| F32 GGUF of the float32 merge (llama.cpp) | 1982 MB | 2.24 GB | 0.875 (7 of 8) | 0.875 | +0.000 | 0 of 8 | n/a |
| q8_0 GGUF from the float32 merge (llama.cpp) | 531 MB | 1.26 GB | 0.875 (7 of 8) | 0.875 | +0.000 | 0 of 8 | n/a |
| q8_0 GGUF from the bf16 merge (llama.cpp) | 531 MB | 1.27 GB | 0.875 (7 of 8) | 0.875 | +0.000 | 0 of 8 | n/a |

Rows whose text differs between arms (each arm against the unmerged adapter): merge-f32: 0; merge-bf16: 0; gguf-f32: 0; gguf-q8-from-f32: 0; gguf-q8-from-bf16: 0.
Transformers float32 merge against the F32 GGUF in llama.cpp, same rows: 0 of 8 rows differ.

### classification, built on torch-cpu

Metric: accuracy (version 1), 8 verification rows. Source engine: torch-cpu on cpu:Apple M4 Pro, numerics {"adapter": "float32", "base": "float32", "optimizer": "float32"}, score 1.000 on these rows.

| arm | size | peak memory (process RSS) | task score (8 rows) | source score | change | text differs from source engine | schema-valid rate |
|---|---|---|---|---|---|---|---|
| unmerged adapter on the base (PEFT, Transformers float32) | 988 MB | 3.41 GB | 1.000 (8 of 8) | 1.000 | +0.000 | 0 of 8 | n/a |
| float32 merge (Transformers float32) | 1976 MB | 2.47 GB | 1.000 (8 of 8) | 1.000 | +0.000 | 0 of 8 | n/a |
| bf16 merge (Transformers float32) | 988 MB | 3.41 GB | 1.000 (8 of 8) | 1.000 | +0.000 | 0 of 8 | n/a |
| F32 GGUF of the float32 merge (llama.cpp) | 1982 MB | 2.22 GB | 1.000 (8 of 8) | 1.000 | +0.000 | 0 of 8 | n/a |
| q8_0 GGUF from the float32 merge (llama.cpp) | 531 MB | 1.27 GB | 1.000 (8 of 8) | 1.000 | +0.000 | 0 of 8 | n/a |
| q8_0 GGUF from the bf16 merge (llama.cpp) | 531 MB | 1.27 GB | 1.000 (8 of 8) | 1.000 | +0.000 | 0 of 8 | n/a |

Rows whose text differs between arms (each arm against the unmerged adapter): merge-f32: 0; merge-bf16: 0; gguf-f32: 0; gguf-q8-from-f32: 0; gguf-q8-from-bf16: 0.
Transformers float32 merge against the F32 GGUF in llama.cpp, same rows: 0 of 8 rows differ.

### json, built on mlx

Metric: whole-record accuracy (version 2), 8 verification rows. Source engine: mlx on apple-gpu:Apple M4 Pro, numerics {"adapter": "float32", "base": "bf16"}, score 0.500 on these rows.

| arm | size | peak memory (process RSS) | task score (8 rows) | source score | change | text differs from source engine | schema-valid rate |
|---|---|---|---|---|---|---|---|
| unmerged adapter on the base (PEFT, Transformers float32) | 988 MB | 3.41 GB | 0.250 (2 of 8) | 0.500 | -0.250 | 4 of 8 | 1.000 |
| float32 merge (Transformers float32) | 1976 MB | 2.48 GB | 0.250 (2 of 8) | 0.500 | -0.250 | 4 of 8 | 1.000 |
| bf16 merge (Transformers float32) | 988 MB | 3.42 GB | 0.250 (2 of 8) | 0.500 | -0.250 | 4 of 8 | 1.000 |
| F32 GGUF of the float32 merge (llama.cpp) | 1982 MB | 2.23 GB | 0.250 (2 of 8) | 0.500 | -0.250 | 4 of 8 | 1.000 |
| q8_0 GGUF from the float32 merge (llama.cpp) | 531 MB | 1.26 GB | 0.375 (3 of 8) | 0.500 | -0.125 | 5 of 8 | 1.000 |
| q8_0 GGUF from the bf16 merge (llama.cpp) | 531 MB | 1.27 GB | 0.375 (3 of 8) | 0.500 | -0.125 | 5 of 8 | 1.000 |

Rows whose text differs between arms (each arm against the unmerged adapter): merge-f32: 0; merge-bf16: 0; gguf-f32: 0; gguf-q8-from-f32: 1; gguf-q8-from-bf16: 1.
Transformers float32 merge against the F32 GGUF in llama.cpp, same rows: 0 of 8 rows differ.

### json, built on torch-cpu

Metric: whole-record accuracy (version 2), 8 verification rows. Source engine: torch-cpu on cpu:Apple M4 Pro, numerics {"adapter": "float32", "base": "float32", "optimizer": "float32"}, score 0.500 on these rows.

| arm | size | peak memory (process RSS) | task score (8 rows) | source score | change | text differs from source engine | schema-valid rate |
|---|---|---|---|---|---|---|---|
| unmerged adapter on the base (PEFT, Transformers float32) | 988 MB | 3.41 GB | 0.375 (3 of 8) | 0.500 | -0.125 | 3 of 8 | 1.000 |
| float32 merge (Transformers float32) | 1976 MB | 2.48 GB | 0.375 (3 of 8) | 0.500 | -0.125 | 3 of 8 | 1.000 |
| bf16 merge (Transformers float32) | 988 MB | 3.41 GB | 0.375 (3 of 8) | 0.500 | -0.125 | 3 of 8 | 1.000 |
| F32 GGUF of the float32 merge (llama.cpp) | 1982 MB | 2.24 GB | 0.375 (3 of 8) | 0.500 | -0.125 | 3 of 8 | 1.000 |
| q8_0 GGUF from the float32 merge (llama.cpp) | 531 MB | 1.28 GB | 0.500 (4 of 8) | 0.500 | +0.000 | 3 of 8 | 1.000 |
| q8_0 GGUF from the bf16 merge (llama.cpp) | 531 MB | 1.26 GB | 0.625 (5 of 8) | 0.500 | +0.125 | 2 of 8 | 1.000 |

Rows whose text differs between arms (each arm against the unmerged adapter): merge-f32: 0; merge-bf16: 0; gguf-f32: 0; gguf-q8-from-f32: 1; gguf-q8-from-bf16: 2.
Transformers float32 merge against the F32 GGUF in llama.cpp, same rows: 0 of 8 rows differ.

## Applying the pre-stated rule

Per candidate, over both task types and both artifact kinds (the merge in Transformers and its q8_0 GGUF in llama.cpp), 16 rows each: rows scored correct, rows whose text differs from the source engine, and the safetensors size.

| source engine | candidate | correct rows | rows differing from the source engine | safetensors size (2 tasks) | rule outcome |
|---|---|---|---|---|---|
| mlx | float32 merge | 19 | 9 | 3952 MB | |
| mlx | bf16 merge | 19 | 9 | 1976 MB | bf16 (step 3: artifact size) |
| torch-cpu | float32 merge | 23 | 6 | 3952 MB | |
| torch-cpu | bf16 merge | 24 | 5 | 1976 MB | bf16 (step 3: artifact size) |
| all four projects | float32 merge | 42 | 15 | 7905 MB | |
| all four projects | bf16 merge | 43 | 14 | 3952 MB | bf16 (step 3: artifact size) |

**Default: `--merge-dtype bf16`** (step 3: artifact size). Neither dtype changed task quality or fidelity by the margin the rule requires, so the smaller artifact wins; text agreement was not used on its own.

## What the comparison shows

- Merging in float32 instead of bf16 did not change any output on these rows beyond what quantization did: the float32 merge, the bf16 merge and the unmerged adapter gave the same text on every verification row in each project, within the differences listed above.
- The runtime is not the cause: the float32 merge in Transformers and the F32 GGUF in llama.cpp agree on the listed rows, so the GGUF conversion is not losing the adapter.
- q8_0 quantization moved a row or two in the JSON projects, in either direction (it scored higher than the float32 artifact in some projects).
- The remaining gap for JSON is against the source engine. It is already present for the unmerged adapter in Transformers float32, which involves no merge at all, so the difference comes from the evaluation runtime and its numerics, not from the export. Tokenization was identical (the prompts were passed as token ids; in a first run of this comparison the engines' recorded prompt token counts equalled the lengths of those ids). Batch shape was ruled out for the torch-cpu engine: re-evaluating the JSON project with an evaluation batch of 1 gave 0 text differences on 18 rows and the same score (0.444).
- JSON outputs are 30 to 50 tokens long, so a single near-tie at any token flips a whole row; classification labels are a few tokens and showed no difference.

## What this can and cannot support

Eight rows per task can support an implementation decision, such as which dtype a default should be. They cannot establish that one merge dtype universally preserves quality better, and nothing here claims that.

Export completion means the artifact loaded and verification ran. It does not mean quality was preserved: every export record and report shows the quality change on the verification rows next to the result, and the docs say so.

## Untested

CUDA, other model sizes, other adapters, and any claim beyond these 8-row verifications.
