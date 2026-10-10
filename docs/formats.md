---
description: The JSONL formats spill reads and writes: plain eval, prompt and training rows, the OpenAI chat and batch shapes, and the result shape.
---

# Formats

Every command reads JSONL, one JSON object per line. There are three plain row shapes, the
OpenAI shapes you may already have, and one output shape. The shape is detected per file,
and `spill check <file>` validates a file and prints the first error with its line number.

## Plain rows

| file | row | used by |
|---|---|---|
| evals | `{"prompt": "...", "expected": "..."}` | `spill eval`, `spill build` (`evals.jsonl`) |
| prompts | `{"prompt": "..."}` | `spill run`, `spill distill`, `spill build` (`prompts.jsonl`) |
| training | `{"prompt": "...", "answer": "..."}` | `spill tune`, `spill build` (`train.jsonl`) |

Any row of any shape may also carry `"system": "..."`, which becomes the first message
unless the row already starts with a system message. `expected` may be any JSON value; a
list means any of its entries is right (`exact_match`).

```json
{"prompt": "I am still waiting on my card?", "expected": "card_arrival"}
{"prompt": "I am still waiting on my card?", "answer": "card_arrival", "system": "Answer with a label."}
```

A row has `answer` (training) or `expected` (eval), not both.

## OpenAI shapes

Chat lines, the OpenAI fine-tuning shape, with an optional `weight` of 0 or 1 on assistant
messages:

```json
{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
```

An eval row is a chat row plus `expected` (and optionally `max_tokens`):

```json
{"messages": [{"role": "user", "content": "What is the capital of France?"}], "expected": "Paris", "max_tokens": 48}
```

Batch lines, the OpenAI batch shape, work everywhere chat lines do:

```json
{"custom_id": "q1", "body": {"messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 48}}
```

One shape per file. Plain rows without a `custom_id` are numbered `row-000001`, `row-000002`, and so on.

## Distillation records

`spill distill` writes one record per row, and `spill tune` accepts them directly (the
prompt plus the teacher's answer becomes a training row):

```json
{"custom_id": "q1", "messages": [{"role": "user", "content": "..."}], "completion": "...", "completion_token_ids": [], "finish_reason": "stop", "teacher": {"model": "llama3.3:70b", "quant": "bf16", "weight_hash": "..."}, "logprobs": []}
```

With `--score` the record has `target` and `positions` instead of `completion` and `logprobs`.

## Results

`jobs/<id>/results.jsonl` is the OpenAI batch output shape plus a `streamweights` object
(tier, engine, quant, batch, token counts, latency, provenance). `finish_reason` is `stop`
when the model emitted a real end-of-turn token and `length` when the row hit its own
`max_tokens`. `--logprobs K` (K up to 64) adds `logprobs: [{token_id, logprob, top: [[id, logprob], ...]}]`
to each row.

## Provenance

Every run writes `runs/<id>/manifest.json` (command, model id and weight fingerprint, quant,
adapter id and hash, input file hash, engine version, hardware probe, start and end, and
whether shared-prefix reuse was used) and stamps each result row with its run id and the
same hashes. A number in an eval table can be traced to the exact weights, adapter and
input that produced it. `spill runs` lists them.

## Build folders

`spill build <folder>` reads these files, and the files present decide what it does:

| file | meaning |
|---|---|
| `evals.jsonl` | required. The exam: every model is graded on it, never trained on it |
| `train.jsonl` | your answers |
| `prompts.jsonl` | questions only; the teacher answers them |
| `instructions.txt` | optional. A system prompt for models that were not trained (the base, the teacher, `--compare`). The tuned model is trained and evaluated without it |
| `spill.json` | optional defaults for the folder: `{"student": "qwen2.5:0.5b", "teacher": "...", "epochs": 2}`. Flags win |

Everything `build` writes lands in the folder (`*.out.jsonl`, `*.distill.jsonl`, `.build/`
with run manifests and state) or in `~/.streamweights/adapters/<name>`.

## Project folders (`spill init`)

A guided project starts from a CSV or JSONL of labeled examples and lives in one folder. The
folder is the unit: it holds the configuration, the data, the immutable history and the pointers
to the model files it was built on. Python never needs to see it twice.

```
tickets/
  streamweights.toml        the versioned project config
  schema.json               JSON Schema (task json only); system.txt if you gave --system
  data/source/<role>-<file>     your files, copied verbatim, hashed in the config
  data/{train,val,test}.jsonl   canonical rows with stable ids
  data/integrity.json           class coverage and conflicting-label report
  runs/<run id>/            a completed run: immutable
  exports/<id>/             an export record: immutable once complete
  tests/<id>/               a final-test record: immutable once complete
  REPORT.md                 a regenerable index over the above
  .spill/                   mutable run state: control objects, locks, payloads, attempt staging
```

### streamweights.toml

`schema_version = 1`. A newer version is refused with a one-line error that names the version
and says to upgrade; nothing is guessed. Sections:

| section | records |
|---|---|
| `[project]` | name, `layout` (`project`, or `legacy-flat` for a migrated flat folder), creation time |
| `[task]` | `type` (`classification` or `json`), the column mapping (`input`, `output`, optional `group`), optional `system` text |
| `[contract]` | classification: `labels` and where they came from (declared, or derived from the training rows only) and the label normalization; json: `schema_file`, its sha256, where it came from, and the comparison rules |
| `[data]` | each source file with its sha256, the canonical files |
| `[split]` | `mode` (`generated`, `supplied` or `mixed`), `seed`, fractions, the duplicate and group normalization, the sha256 and size of each split |
| `[model]` | `student`, `teacher` (empty unless you asked), `engine` |
| `[training]` | epochs, learning rate, LoRA rank and alpha, seed, micro-batch, checkpoint interval |
| `[evaluation]` | metric and metric version, protocol version, decoding, max tokens, precision, postprocessing |
| `[baseline]` | the embedding model, pooling, classifier settings (classification) |

Changing data, prompts, the schema or vocabulary, training settings, the evaluation protocol or a
model's files makes the next build a **new run**. The engine and the machine are not part of a
run's identity: a documented engine transition continues the same run and is recorded.

Duplicate normalization is Unicode NFKC, casefold, whitespace collapsed, ends stripped. Rows
whose normalized inputs are equal, or whose normalized group values are equal, are one connected
component (transitively); a generated split never separates a component, and a supplied split
that shares one across files is rejected with the file and row of each side.

Classification compares a model's output, normalized the same way, with the vocabulary: an output
outside it is a failure and stays in the denominator. JSON extraction parses the whole response
(one optional `json` fence is removed), validates it against the schema, and compares fields:
strings after NFC and trimming, numbers numerically (booleans are not numbers), objects by key
set and value, arrays in order unless the schema marks them `x-unordered`. A field missing from
the output is wrong, a field absent from both is right, an extra top-level field makes the record
wrong. Unparseable, schema-invalid, failed and truncated outputs all count as wrong.

### Rows and ids

`{"id": "r-<12 hex>", "input": "...", "output": "label" or {...}, "group": "...", "source":
{"file", "row"}}`. The id is the first 12 hex digits of sha256(input, NUL, output); an identical
pair gets a `-<n>` suffix. The same example has the same id in any order.

### A completed run, `runs/<id>/`

| file | contents |
|---|---|
| `report.md`, `results.json` | the results, counts, durations, failures, artifact locations and next commands |
| `manifest.json` | identity, resolved config, split fingerprints, model and tokenizer revisions with the sha256 of every file, dependency versions, per-stage producers (engine, hardware, OS, numerics), the evaluation-protocol fingerprint, parent run |
| `predictions/<comparator>.jsonl` | one line per validation row: gold, raw output, parsed prediction, flags, finish reason |
| `disagreements.jsonl` | rows where a comparator and the trained student differ: `improvement` or `regression` |
| `inputs/` | the frozen train and validation rows, config, schema and identity. The final-test rows are not copied; their sha256 is in the manifest |
| `artifacts/` | the adapter and the training rows with their origin (`own` or `teacher`) |

The run is installed read-only, and only the accepted outputs of the winning owner reach it
(see [portability](portability.md#completion-is-a-fenced-transition)). `REPORT.md` at the project
root may change; a run's own report never does.

### Export and test records

`exports/<id>/record.json` names the base model and revision, the tokenizer files and chat-template
hash, the prompt template, the adapter and how it was merged (`--merge-dtype float32|bf16`, default bf16), each format and its quantization,
the source run and its evaluation references, the verification (runtime, settings, exact rows,
load failures, prediction differences, task metrics, the script run once) and the deployment
measurements with their boundaries and hardware. `tests/<id>/record.json` names the run, the
test-file hash, the metrics, and how many times the split had been scored before. In an export record, `quality_delta` in each verification run shows the source engine's score and the artifact's score on the same rows, the text-disagreement rate and, for JSON, the schema-valid rate. An export that
verified loaded and ran; it does not mean quality was preserved. A failure is a
record with `"status": "failed"`, never a missing directory. A later verification is a new record
that references the earlier one.

### Payloads and control state

Under `.spill/runs/<id>/`: `control.json` (one authority per run), `control.lock` (local disk), and
`payloads/<kind>-g<generation>-s<seq>-<attempt>/` holding the files and a `PAYLOAD.json` manifest
written last with each file's size and sha256. The control object holds `owner`, `generation`
(fencing), `lease_expires`, `revision`, `status` (`idle`, `running`, `handoff`, `transferred`,
`incoming`, `completed`, `bundled`), `checkpoint` (the accepted payload: dir, manifest hash, step,
cursor, sequence), `stages` (accepted stage payloads), `completed`, `handoff`, `locations` and a
bounded history. See [portability](portability.md).

### Schema migrations

### Metrics, protocol and evaluation records

`evaluation.metric` in streamweights.toml selects the primary metric; it must be one the task
supports (classification: `accuracy`, `macro_f1`; json: `whole_record_accuracy`,
`mean_field_accuracy`, `schema_valid_rate`, `parseable_rate`) or the build refuses with a message
naming the choices. Metric versions: classification 1, JSON 2. JSON version 2 changed one rule: an
output that parses but fails the schema earns no credit at all (no field-level and no whole-record
credit; trimming and number equality never rescue it) but still counts toward the parseable rate.
Version 1 compared fields before checking validity. Runs built with different metric versions are
not ranked together.

The evaluation protocol (rows, prompts, contract, decoding, max tokens, postprocessing, metric and its
version, requested precision) is fingerprinted. Its decoding block records `temperature`, `top_p`,
`stop`, `greedy` and `seed`; every one is put in each request body, and the MLX and PyTorch engines,
which decode greedily with no sampling and no stop sequences, refuse a request for a setting they
cannot apply, naming the engine and the setting, before anything runs. Each result row echoes the
settings as applied (`streamweights.decoding`).

What an evaluation actually ran under is recorded apart from the requested precision policy:
`conditions` holds the engine, device, actual numerics (weight, adapter and compute dtypes),
runtime and library versions and the decoding as applied. The run's original evaluation has the id
`<run id>/original`. `spill evaluate <run> --engine <engine>` writes `evaluations/<id>/record.json`
(immutable, references the run, uses the frozen validation rows only, shows the original scores next
to the new ones with both metric versions). `spill compare` labels each comparison common evaluator,
cross-runtime or incompatible, and takes `--use <run id>=<evaluation id>`.

A `parent` on a run records experiment lineage only: the new run trained from the base model, not from
the parent's weights.

A flat-layout folder (`evals.jsonl`, `train.jsonl`, `.build/`) opened by a project command gets a
`streamweights.toml` with `layout = "legacy-flat"` and the hash of each source file, announced in
one line. Nothing is moved or rewritten, a second open changes nothing, and `.build/` is not
reinterpreted: `spill build` and `spill resume` keep using it as before. There is one schema
version, 1; a future change will add a migration here before it adds a version.
