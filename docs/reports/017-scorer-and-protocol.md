# 017: scorer and protocol correctness

Directive: `docs/paste-sets/017-correctness-and-tagline.md`. Raw results: `docs/reports/data/017-*.json`. Hardware: Apple M4 Pro, 48.0 GB, macOS (Darwin 24.6.0) arm64.

## The JSON scorer (metric version 2)

Before: `score_json` compared fields first and ignored schema validity when deciding correctness, so an output that parsed but violated the schema (for example `{"status": " approved "}` against a schema that allows only `approved`) could be credited after trimming. Now whole-record correctness requires the output to parse and to be schema-valid; a parseable but schema-invalid answer still counts toward the parseable rate and gets no field-level or whole-record credit. The regression test is `tests/test_json_scorer.py`. The JSON metric version is 2, classification stays 1, and runs built under different metric versions are not ranked together.

Audit of the other JSON metrics: the schema-valid rate, parseable rate, per-field accuracy, mean field accuracy and whole-record accuracy are all aggregated from the same per-row items, so the one change corrects all of them; the extra-field count is unchanged. The older `json_field` metric of `spill eval` is a separate loose matcher that the guided workflow does not use; it is unchanged.

## Historical JSON results: accounting

Every published JSON result in the repository was scored with metric version 1. None can be rescored: a rescore needs the saved per-example predictions and ground truth, and these results were published as aggregates only (the raw project folders lived in scratch directories and were not committed; the export records kept only the rows that differed). Each is therefore marked **not rescorable**. A fresh model run is a replication, not a rescore, and the section after this table is labeled that way. No larger-model rerun was made.

| published result | where | metric version | status |
|---|---|---|---|
| JSON validation tables, MLX and torch-cpu, 18 rows | `docs/reports/015-workflow.md`, README of that date | 1 | not rescorable (no per-example predictions saved) |
| JSON final-test scores, MLX and torch-cpu | `docs/reports/015-workflow.md` | 1 | not rescorable |
| JSON export verification rows and quality (8 rows) | `docs/reports/015-workflow.md` | 1 | not rescorable |
| JSON journey from the wheel, Linux CI | `docs/reports/data/015-ci-linux.json`, `docs/reports/015-workflow.md` | 1 | not rescorable |
| JSON rows of the README example table | README (before this change) | 1 | not rescorable; replaced by the replication below |

Classification results are unaffected (metric version 1, unchanged).

## Replication under metric version 2 (not a rescore)

The same journeys were run again, from fresh projects built from the same tiny examples (120 rows: 84 train, 18 validation, 18 final test, seed 0), `qwen2.5:0.5b` bf16, with the current code. Every score is on 18 rows; one row is 0.056.

| task | engine | comparator | score | schema-valid rate | rows |
|---|---|---|---|---|---|
| classification | mlx | embedding baseline | 1.000 | n/a | 18 |
| classification | mlx | student, prompted, untrained | 0.333 | n/a | 18 |
| classification | mlx | student, trained | 0.889 | n/a | 18 |
| classification | torch-cpu | embedding baseline | 1.000 | n/a | 18 |
| classification | torch-cpu | student, prompted, untrained | 0.444 | n/a | 18 |
| classification | torch-cpu | student, trained | 0.944 | n/a | 18 |
| json | mlx | student, prompted, untrained | 0.000 | 0.722 | 18 |
| json | mlx | student, trained | 0.500 | 1.000 | 18 |
| json | torch-cpu | student, prompted, untrained | 0.000 | 0.667 | 18 |
| json | torch-cpu | student, trained | 0.444 | 1.000 | 18 |

In this replication the trained student's JSON outputs were all schema-valid, so the corrected scorer did not change the trained scores; the untrained student's schema-invalid outputs are where the version 1 and version 2 rules differ, and that is not rescorable for the published runs.

| task | engine | build | test | evaluate (other engine) | export (safetensors + GGUF q8_0, verified) |
|---|---|---|---|---|---|
| classification | mlx | 13.7 s | 5.1 s | 7.5 s | 19.0 s |
| classification | torch-cpu | 32.2 s | 7.5 s | 4.2 s | 19.0 s |
| json | mlx | 17.8 s | 8.2 s | 19.1 s | 22.7 s |
| json | torch-cpu | 38.1 s | 19.2 s | 4.4 s | 23.0 s |

## The protocol matches execution

**Primary metric.** `evaluation.metric` in the config selects it; classification supports `accuracy` and `macro_f1`, JSON supports `whole_record_accuracy`, `mean_field_accuracy`, `schema_valid_rate` and `parseable_rate`. Anything else is rejected with a message naming the choices (`tests/test_json_scorer.py`, `tests/test_project_build.py`).

**Decoding.** Each request body now carries `max_tokens`, `temperature`, `top_p`, `stop`, `seed` and `greedy`. The MLX and PyTorch engines decode greedily with no sampling and no stop sequences; a request for a setting they cannot apply is refused before anything runs, naming the engine and the setting, and no sampling feature was added. Captured on this Mac (`scripts/gate_decoding_capture.py`; `mx.default_device()` was `Device(gpu, 0)`):

| engine | device | request as sent | settings as applied (row echo) | request with temperature 0.7 |
|---|---|---|---|---|
| mlx | apple-gpu:Apple M4 Pro | `{"temperature": 0, "top_p": 1.0, "stop": [], "seed": 3, "greedy": true, "max_tokens": 5}` | `{"temperature": 0.0, "top_p": 1.0, "stop": [], "greedy": true, "max_tokens": 5, "seed": 3}` | refused (exit 1): `spill: engine mlx cannot apply the request's decoding setting temperature=0.7 (this engine decodes greedily; only temperature 0 is applied) ` |
| torch-cpu | cpu:Apple M4 Pro | `{"temperature": 0, "top_p": 1.0, "stop": [], "seed": 3, "greedy": true, "max_tokens": 5}` | `{"temperature": 0.0, "top_p": 1.0, "stop": [], "greedy": true, "max_tokens": 5, "seed": 3}` | refused (exit 1): `spill: engine torch-cpu cannot apply the request's decoding setting temperature=0.7 (this engine decodes greedily; only temperature 0 is app` |

**Executed conditions.** Each evaluation stores, apart from the requested precision policy, the engine, device, actual numerics, library versions and the decoding as applied. For the original evaluation of the first journey:

```
{
 "adapter_dtype": "float32",
 "compute_dtype": "float32",
 "decoding_applied": {
  "greedy": true,
  "max_tokens": 16,
  "requested": {
   "greedy": true,
   "seed": 0,
   "stop": [],
   "temperature": 0.0,
   "top_p": 1.0
  },
  "seed": 0,
  "stop": [],
  "temperature": 0.0,
  "top_p": 1.0
 },
 "device": "cpu:Apple M4 Pro",
 "engine": "torch-cpu",
 "engine_impl": "torch_resident",
 "numerics": {
  "adapter": "float32",
  "base": "float32",
  "optimizer": "float32"
 },
 "os": "macOS arm64",
 "versions": {
  "mlx": "0.32.3",
  "mlx-lm": "0.32.0",
  "numpy": "2.5.3",
  "peft": "0.21.2",
  "python": "3.12.13",
  "safetensors": "0.8.0",
  "torch": "2.14.1",
  "transformers": "5.19.0"
 },
 "weight_dtype": "float32"
}
```

**Evaluation records and compare.** `spill evaluate <run> --engine <engine>` re-scored each run on the other engine as a separate record. `spill compare` of the MLX-built and the torch-cpu-built project, each through its original evaluation (it says so), is labeled cross-runtime: ranked, with the differences and a caution. The incompatible cases (rows, prompts, decoding policy, metric version) and explicit `--use` selection are tested in `tests/test_compare_records.py`.

Classification:

```
run  evaluation  metric  trained  baseline  untrained  rows  protocol  engine
run-20261009-181324-a4dd-c29d25  run-20261009-181324-a4dd-c29d25/original  accuracy v1  0.889  1.000  0.333  18  e27bc6d29acf  mlx
run-20261009-181414-a1d6-c29d25  run-20261009-181414-a1d6-c29d25/original  accuracy v1  0.944  1.000  0.444  18  e27bc6d29acf  torch-cpu
evaluation used for run-20261009-181324-a4dd-c29d25: run-20261009-181324-a4dd-c29d25/original (the run's original evaluation; no --use given; others exist: eval-20261009-181342-51423d)
evaluation used for run-20261009-181414-a1d6-c29d25: run-20261009-181414-a1d6-c29d25/original (the run's original evaluation; no --use given; others exist: eval-20261009-181453-4e3661)

cross-runtime: ranked by the trained model's accuracy: run-20261009-181414-a1d6-c29d25 (run-20261009-181414-a1d6-c29d25/original) > run-20261009-181324-a4dd-c29d25 (run-20261009-181324-a4dd-c29d25/original)
   the evaluators differ: compute_dtype: bf16 vs float32; device: apple-gpu:Apple M4 Pro vs cpu:Apple M4 Pro; engine: mlx vs torch-cpu; weight_dtype: bf16 vs float32
   caution: score differences may include evaluation-runtime effects, not only differences between the models
   a higher number is an observed difference on these rows, not a statistical claim

differences run-20261009-181324-a4dd-c29d25 vs run-20261009-181414-a1d6-c29d25:
   engine: engine
   numerics: numerics

next: spill export classification-mlx/project
```

JSON extraction:

```
run  evaluation  metric  trained  baseline  untrained  rows  protocol  engine
run-20261009-181522-f353-ba19bd  run-20261009-181522-f353-ba19bd/original  whole-record accuracy v2  0.500  unavailable  0.000  18  d31102a80fef  mlx
run-20261009-181635-8b56-ba19bd  run-20261009-181635-8b56-ba19bd/original  whole-record accuracy v2  0.444  unavailable  0.000  18  d31102a80fef  torch-cpu
evaluation used for run-20261009-181522-f353-ba19bd: run-20261009-181522-f353-ba19bd/original (the run's original evaluation; no --use given; others exist: eval-20261009-181547-0d5c09)
evaluation used for run-20261009-181635-8b56-ba19bd: run-20261009-181635-8b56-ba19bd/original (the run's original evaluation; no --use given; others exist: eval-20261009-181732-e5a1ae)

cross-runtime: ranked by the trained model's whole-record accuracy: run-20261009-181522-f353-ba19bd (run-20261009-181522-f353-ba19bd/original) > run-20261009-181635-8b56-ba19bd (run-20261009-181635-8b56-ba19bd/original)
   the evaluators differ: compute_dtype: bf16 vs float32; device: apple-gpu:Apple M4 Pro vs cpu:Apple M4 Pro; engine: mlx vs torch-cpu; weight_dtype: bf16 vs float32
   caution: score differences may include evaluation-runtime effects, not only differences between the models
   a higher number is an observed difference on these rows, not a statistical claim

differences run-20261009-181522-f353-ba19bd vs run-20261009-181635-8b56-ba19bd:
   engine: engine
   numerics: numerics

next: spill export json-mlx/project
```

Choosing the other engine's evaluation of the first run, explicitly:

```
run  evaluation  metric  trained  baseline  untrained  rows  protocol  engine
run-20261009-181324-a4dd-c29d25  eval-20261009-181342-51423d  accuracy v1  0.889  1.000  0.444  18  e27bc6d29acf  torch-cpu
evaluation used for run-20261009-181324-a4dd-c29d25: eval-20261009-181342-51423d (chosen with --use)

```

A parent run appears in reports, the project index and compare with the statement that it records experiment lineage only and that the new run trained from the base model, not from the parent's weights.

## Deployment measurements (replication)

Measured in the export verification of each journey, separately from build time. Each runtime's own description of its measurement applies:

- Transformers (safetensors, float32 on the CPU): time to first token is the wall time of `generate(max_new_tokens=1)` including prompt prefill; cold is the first row and warm the median of the rest, and neither includes loading the model (load time is recorded apart). Tokens per second is completion tokens over the wall time of the full `generate()` including prefill. Peak memory is `ru_maxrss` of the verification process, a process peak of resident memory, not a GPU-memory figure.
- llama.cpp (GGUF q8_0, all layers on the Metal GPU on this Mac): time to first token is the time from the request to the first streamed content token with the prompt cache off, cold first row and warm median, excluding model loading. Peak memory is the peak resident set of the llama-server process tree sampled after each request: a lower bound, and not a total of GPU memory.
- The Metal q8_0 and CPU float32 rows are not a format-only performance comparison: they differ in runtime, device, and precision at once.

| task | built on | artifact | cold first token | warm first token | warm tokens/s | peak memory (sampled RSS) |
|---|---|---|---|---|---|---|
| classification | mlx | gguf:q8_0 (llama.cpp, Metal) | 0.0408 s | 0.0204 s | 121.28 tokens/s | 0.73 GB |
| classification | mlx | safetensors (Transformers, CPU float32) | 0.0677 s | 0.0642 s | 33.84 tokens/s | 3.41 GB |
| classification | torch-cpu | gguf:q8_0 (llama.cpp, Metal) | 0.0372 s | 0.0204 s | 120.84 tokens/s | 0.74 GB |
| classification | torch-cpu | safetensors (Transformers, CPU float32) | 0.0693 s | 0.064 s | 32.55 tokens/s | 3.41 GB |
| json | mlx | gguf:q8_0 (llama.cpp, Metal) | 0.0382 s | 0.021 s | 206.15 tokens/s | 0.75 GB |
| json | mlx | safetensors (Transformers, CPU float32) | 0.0668 s | 0.0649 s | 50.92 tokens/s | 3.41 GB |
| json | torch-cpu | gguf:q8_0 (llama.cpp, Metal) | 0.0401 s | 0.0208 s | 207.46 tokens/s | 0.75 GB |
| json | torch-cpu | safetensors (Transformers, CPU float32) | 0.0665 s | 0.0639 s | 51.39 tokens/s | 3.41 GB |

## Decisions

- Protocol version 2: the decoding block gained `stop` and `greedy`; protocol 1 and 2 runs are incompatible and compare says why.
- Metric versions are the scorer's, kept in code (`config.METRIC_VERSIONS`), not read from an old config file, and are part of a run's identity.
- The original evaluation of a run is named `<run id>/original`; `spill evaluate` writes `evaluations/<id>/`.
- Everything the engines cannot do (sampling, stop sequences) is refused rather than approximated.

## Untested

CUDA, real AWS S3, Windows, power loss, and any model beyond `qwen2.5:0.5b`. The captured-request gate ran on the Metal GPU and on torch-cpu here; in CI the same test runs on torch-cpu and on MLX's CPU device.
