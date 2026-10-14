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
{"custom_id": "q1", "messages": [{"role": "user", "content": "..."}], "completion": "...", "completion_token_ids": [], "finish_reason": "stop", "teacher": {"model": "llama3.3:70b", "quant": "bf16", "weight_hash": "...", "tokenizer_sha256": "..."}, "logprobs": []}
```

With `--score` the record has `target` and `positions` instead of `completion` and `logprobs`.
`spill tune <student> <that file> --distill-loss kl|ce|mix [--kd-temp T] [--kd-alpha A]` trains on the
teacher's per-token top-k when the teacher's `tokenizer_sha256` is the student's; the default is `mix` then,
and `ce` otherwise. A teacher with another tokenizer contributes its text only, and `kl` or `mix` is refused
in one line saying so. [Teachers and training rows](#teachers-and-training-rows) has the loss.

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

A guided project starts from a CSV or JSONL of examples, an input and the output you want, and lives in one folder. The
folder is the unit: it holds the configuration, the data, the immutable history and the pointers
to the model files it was built on. Python never needs to see it twice.

```
tickets/
  streamweights.toml        the versioned project config
  schema.json               JSON Schema (json outputs only); system.txt if you gave --system
  rubric.md                 optional: the rubric a judge scores against (see below)
  score.py                  only with --score script:<file>: your scoring function, copied in
  data/source/<role>-<file>     your files, copied verbatim, hashed in the config
  data/{train,val,test}.jsonl   canonical rows with stable ids
  data/unlabeled.jsonl          only with --unlabeled: prompts a teacher answers (inputs only)
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
| `[task]` | `type` (`classification` for exact scoring, `json`, or `open` for judge scoring), the column mapping (`input`, `output`, optional `group`), optional `system` text |
| `[contract]` | classification: `labels` and where they came from (declared, or derived from the training rows only) and the label normalization; json: `schema_file`, its sha256, where it came from, and the comparison rules; open: the note that scores are against your reference |
| `[data]` | each source file with its sha256, the canonical files |
| `[split]` | `mode` (`generated`, `supplied` or `mixed`), `seed`, fractions, the duplicate and group normalization, the sha256 and size of each split |
| `[model]` | `student` and `student_why` (why it is the default, or `chosen with --student`), `teacher` and `judge` (empty unless you named one; `judge = "none"` opts out of the teacher judging), `engine`. Each is a curated tag or a Hugging Face repo in a supported family |
| `[training]` | epochs, learning rate, LoRA rank and alpha, seed, micro-batch, checkpoint interval; with a teacher: `distill_loss` (`auto`, `kl`, `ce` or `mix`), `kd_temp`, `kd_alpha`, `kd_topk` |
| `[evaluation]` | `scoring` (`exact`, `json`, `judge` or `script`), metric and metric version, protocol version, decoding, max tokens, precision, postprocessing; with a script, `script_file` and `script_sha256` |
| `[baseline]` | classification: the embedding model, pooling, classifier settings; open: the nearest-training-answer baseline (TF-IDF, no model) |

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

### How outputs are scored

`spill init` reads your outputs and states its choice, in the init line, in `spill plan` and in the
report. `--score exact|json|judge|script:<file>` overrides it.

| scoring | chosen when | what it computes |
|---|---|---|
| `exact` | the outputs are strings that form a small label set: at most 200 distinct values, each short, with no sentence punctuation, 80% of rows sharing their label with another row | exact match after normalization, macro-F1, and the embedding baseline |
| `json` | every output is a JSON object | parse, schema, field-level and whole-record accuracy (metric version 2) |
| `judge` | anything else | reference overlap against your outputs, always, labeled as overlap and never as accuracy: token F1 (bag of words), ROUGE-L (longest common word subsequence) and exact match, on words that are NFKC, casefolded runs of word characters. A judge named or accepted adds its own assessment |
| `script:<file>` | never automatically | your Python file's `score(row) -> float`, called once per evaluation row with `{"id", "input", "gold", "text", "pred"}` (`text` is the model's raw output); the mean is `script_score` |

A project made before scoring selection, with a recorded task type, maps to `exact` (classification)
or `json` and keeps its metric versions and its protocol fingerprint, so its runs still compare with
new runs on the same rows.

### Judged evaluation

A judge is one model's assessment of each output against a rubric. It is named in `[model] judge`
(or `--judge <model>`), accepted with `--judge recommended`, or is the teacher when you named a teacher
and no judge. Nothing is downloaded unless you name or accept it.

* **Pointwise.** Each output gets an integer from 1 to 5 against the rubric and against your reference.
  The judge replies `Rationale: ...` then `Score: n`; a reply with no score is counted as unparseable,
  not as a number.
* **Pairwise.** Two models' outputs on the same input are compared twice, with the two positions swapped.
  The two verdicts are combined: the same answer twice is a win or a loss, anything else (a flip, or a tie
  against a win) is a tie and counts as an order disagreement. The report gives the win, tie and loss rates,
  how often the two orders disagreed, and how often the judge picked the first position.
* **Calibration.** Where a reference is at most 5 words, the judge's score (4 or more means "judged right")
  is compared with whether the reference appears word for word in the output; the agreement is reported.
  It is omitted when no row qualifies.
* **Self-judging.** A judge never scores its own outputs. A comparator whose model is the judge (the teacher
  when the teacher judges, or the student when the student judges) is reported as reference only and left out
  of the judged rankings, and the report says so.
* **Records.** `runs/<id>/judgments.jsonl` has one line per judged output (`kind: pointwise`: comparator, id,
  score, rationale, the judge's reply, the output and the reference) and one per pair and row (`kind: pairwise`:
  both orders, each with the pick and the rationale, and the combined verdict). `spill evaluate` and
  `spill test` judge again on their rows and write the same fields into their records.

The rubric is `rubric.md` in the project if it exists, otherwise this default:

```
Score the answer from 1 to 5.
5: fully correct and complete for the task input, and it follows any instructions given.
4: correct, with a small omission or a small wording problem.
3: partly correct, or correct but missing something that matters.
2: mostly wrong or off-topic, with some relevant content.
1: wrong, empty, or unrelated.
When a reference answer is given, treat it as a good answer, not the only one: an answer that
says the same thing in other words deserves the same score. Judge the content, not the style.
```

The rubric text, the judge model's identity (revision and file hashes), the two prompt templates and the
judge's decoding are all in the evaluation-protocol fingerprint. Change any of them and `spill compare` calls
the runs incompatible and does not rank them, naming which part differs.

### Teachers and training rows

With labeled examples and no teacher, the student trains on your labels alone. A teacher runs only if you
name one or the project has unlabeled prompts (`--unlabeled <file>`: a CSV or JSONL with your input column,
or `prompt`). Unlabeled prompts that match any training, validation or test input are removed at `init`, so
no evaluation input can reach the teacher. Teachers answer the training partition and the unlabeled prompts
only. Each training row keeps its origin (`own`, `teacher` or `teacher_unlabeled`); your labels are repeated
twice for each teacher answer.

When the teacher and the student share a tokenizer (the same `tokenizer.json`), the teacher also keeps its
top-k log-probabilities at each token (`kd_topk`, default 20) and the student can learn from them:
`kl` is the KL from the teacher's top-k distribution, with the remaining probability in one "other" bucket,
to the student's, at temperature `kd_temp`; `ce` is cross-entropy on the teacher's tokens; `mix` is
`kd_alpha` of the first and the rest of the second. `auto` is `mix` when the log-probs are present and the
tokenizers match, otherwise `ce` on the teacher's text. A training line carries them as
`"teacher": {"token_ids": [...], "top_ids": [[...]], "top_lps": [[...]], "tokenizer_sha256": "..."}`, one entry
per teacher token; `spill tune --distill-loss kl|ce|mix` reads such files, and a `spill distill --logprobs K`
output is converted to them when the tokenizers match.

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
| `judgments.jsonl` | with a judge: every verdict and rationale |
| `artifacts/` | the adapter and the training rows with their origin (`own`, `teacher` or `teacher_unlabeled`) |

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
`mean_field_accuracy`, `schema_valid_rate`, `parseable_rate`; open: `token_f1`, `rouge_l`, `exact_match`,
`judge_mean`; every task: `script_score` with a scoring script) or the build refuses with a message
naming the choices. A judge makes `judge_mean` the primary metric of an open-ended project. Metric
versions: classification 1, JSON 2, open 1. JSON version 2 changed one rule: an
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
