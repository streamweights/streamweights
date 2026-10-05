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
