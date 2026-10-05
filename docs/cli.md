# Commands and flags

Every command ends by printing the one command most likely to come next (`next: ...`).
Long jobs (`run`, `distill`, `tune`, `build`) hold `caffeinate -i` while they run, warn in
the pre-run line when the Mac is on battery, and post a macOS notification when they finish
or stop. Any command, on launch, reports an interrupted job or build in one line with its
`spill resume` command.

## The loop

| command | what it does |
|---|---|
| `spill build <folder>` | the whole loop from a folder: distill, tune, eval, one table. See below |
| `spill example <name> [--quick]` | create `./<name>/` with ready files (`banking77`); `--quick` is the under-an-hour variant in `./<name>-quick/` |
| `spill run <model>[+<adapter>] <file>` | the full-size model over your file |
| `spill distill <teacher> <file>` | the teacher's completions, or with `--score` its log-probs over targets you supply |
| `spill tune <model> [<file>] --name <name>` | train a LoRA adapter against the full-precision base. With no file it uses the folder's newest `*.distill.jsonl` or `train.jsonl` |
| `spill eval <evals.jsonl> [<model>...]` | the same eval set through several models, one table. With no models it uses every model that has a run against the file's hash |
| `spill export <base>+<adapter>` | merge the adapter into the base: merged safetensors, optionally GGUF and Ollama |

## spill build

```
spill build <folder> [--student M] [--teacher M] [--compare M]... [--base M]
                     [--weight-own 2] [--epochs 2] [--notify URL] [--quiet] [--debug]
```

The files in the folder decide the path ([formats](formats.md#build-folders)):

| files | what build does |
|---|---|
| `evals.jsonl` + `train.jsonl` | eval the student base, tune on `train.jsonl`, eval the student with the adapter |
| `evals.jsonl` + `prompts.jsonl` | the teacher answers `prompts.jsonl`, tune on its answers, then eval the student base, the student with the adapter, and the teacher |
| all three | the teacher answers the prompts, tune on your answers (weighted 2:1 against the teacher's by default, `--weight-own`) plus the teacher's, then the same three evals |

- `--student` (default `qwen2.5:7b`), `--teacher` (default `llama3.3:70b`).
- `--base M` trains the adapter on `M` itself (a big model) instead of the student.
- `--compare M` adds `M`'s score to the table (repeatable).
- The adapter is named after the folder. Intermediates land in the folder or in
  `~/.streamweights/adapters/<name>`.
- The pre-run line states the path, the models, the per-stage time estimate and the total.
  Evals whose expected values are short labels get `max_tokens` 16, and the line says so.
- Resumable at any stage: `spill resume <folder>`, or run the same build again.
- It ends with the table (model, role, score, rows) and one line:
  `your model: <student>+<name> · spill export <student>+<name>`.

## spill run

`spill run <model>[+<adapter>] <input.jsonl|sample>`

| flag | meaning |
|---|---|
| `--quant 8bit\|4bit\|Q8_0\|Q4_K_M` | explicit opt-in to a quant (bf16 is the default) |
| `--out PATH` | where to copy the results |
| `--context N` | context length (default 4096) |
| `--parallel N` | override the computed batch (never required) |
| `--logprobs K` | per-token top-K log-probs, K up to 64 |
| `--full-logits` | also write float16 logits per row, for sets under 200 rows |
| `--no-prefix-reuse` | compute every row's whole prompt (for A/B checks) |
| `--notify URL` | POST a small JSON to this URL when done |
| `--quiet` | no live block |
| `--debug` | tracebacks |

**Shared-prefix reuse.** Rows in a job that begin with the same tokens (a system prompt and
the chat header) compute that prefix once and attend to it from every row, in prefill and in
decode, so a 900-token label list costs the same as a 20-token one. It engages when the
common prefix is at least 64 tokens across at least 4 rows, and the pre-run line states how
many tokens it saves. Output matches an unshared run up to float rounding; the manifest
records what was used.

## spill distill

`spill distill <teacher> <file>` with `--score` (teacher-forced: score the assistant targets
in the file, prefill only, no sampling), `--logprobs K` (default 32), `--quant`, `--out`,
`--context`, `--parallel`, `--notify`, `--quiet`, `--debug`.

## spill tune

`spill tune <model> [<file>] --name <name>` with `--rank 16`, `--alpha 32`, `--dropout 0`,
`--targets`, `--lr 1e-4`, `--schedule cosine|constant`, `--weight-decay`, `--steps`,
`--epochs`, `--batch`, `--grad-accum`, `--max-seq 2048`, `--seed`, `--path auto|resident|streamed`,
`--ckpt-every 50`, `--overwrite`, `--quiet`, `--debug`. Resumable (`spill resume <job>`),
clean on Ctrl-C.

## spill eval

`spill eval <evals.jsonl> [<model>...]` with `--metric exact_match|contains|regex|json_field|judge|script:<file.py>`,
`--judge MODEL`, `--rerun`, `--quant`, `--context`, `--parallel`, `--quiet`, `--debug`.
Each model without a finished run for this exact input (matched by hash) is run, every row is
scored, and one table is printed (model, quant, adapter, rows, metric mean, latency, tokens),
plus `diff.jsonl` with the rows where the models disagree.

## spill export

`spill export <base>+<adapter>` with `--out DIR`, `--gguf [bf16|q8_0|q4_k_m]` (bare `--gguf`
means `q8_0`), `--ollama`, `--name`.

The merged safetensors are written shard by shard (`W + scale * (A @ B)^T`, computed in float32
and stored in the base's dtype). GGUF conversion uses llama.cpp's own converter, downloaded
at the release tag of the llama.cpp binaries we ship and run in a private environment (never
vendored); `q4_k_m` is a bf16 conversion followed by `llama-quantize`. A Modelfile is written
beside the GGUF, and `--ollama` runs `ollama create` when ollama is installed and prints the
`ollama run` line.

## Everything else

| command | what it does |
|---|---|
| `spill check <file>` | validate a file of any shape; first error with its line number |
| `spill doctor` | chip, RAM, Metal working set, disk, Python, version, downloaded models, interrupted jobs, the largest model this machine can run overnight for eval, and the achieved TFLOP/s the estimates use |
| `spill models [--architectures]` | curated tags and architecture families, live |
| `spill runs` | every run with model, quant, adapter, input hash, rows, status |
| `spill adapters` | local LoRA adapters |
| `spill tail [job]` | follow a job's results; shows the live block while it runs |
| `spill resume [job\|folder]` | continue the latest or named job, or a build folder, from its checkpoint |
| `spill status` | all jobs with progress, tokens/s, ETA |

## The cost model

Every estimate (pre-run lines, the README, the diagrams) uses one model:

- decode on a streamed model is disk-bound: one weight pass per token step, about 34 s per
  pass for the 70B on a 48 GB M4 Pro;
- prefill is compute-bound: 2 x parameters x tokens;
- training is compute-bound: 6 x parameters x tokens (forward, recompute, backward);
- both divided by the achieved FLOP/s, which `spill tune` measures and records
  (`spill doctor` shows it). Until one is measured the estimate uses 5 TFLOP/s and says so.
