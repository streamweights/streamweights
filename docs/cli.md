# Commands

Generated from the real `--help` output by `scripts/make_cli_docs.py`; `tests/test_docs.py` fails if it is out of date. Commands are in the order you use them.

```
Usage: spill [OPTIONS] COMMAND [ARGS]...

  Build your own model on your Mac. Errors are one line; add --debug to any
  command for the traceback.

Options:
  --help  Show this message and exit.

Commands:
  build     Build your model from a folder: distill, tune, eval
  example   Create a ready-to-run example folder
  run       Run a JSONL of prompts through a model
  distill   Collect a big model's answers to learn from
  tune      Train a LoRA adapter on your data
  eval      Score models on your eval set, one table
  export    Merge an adapter into its base; GGUF and Ollama
  models    List the models spill knows and what they need
  adapters  List your trained adapters
  runs      List past runs
  status    Show jobs: progress, tokens/s, ETA
  tail      Follow the results of the latest job
  resume    Continue an interrupted job or build
  doctor    Check this machine and what it can run overnight
  check     Validate a JSONL file, with line-numbered errors

  Example: spill example banking77 --quick && spill build banking77-quick
```

## spill build

```
Usage: spill build [OPTIONS] {folder}

  Build your own model from a folder: distill, tune, eval, one table.

Arguments:
  folder  a folder with evals.jsonl and train.jsonl and/or prompts.jsonl
          [required]

Options:
  --student <str>       the small model to build on (default qwen2.5:7b)
  --teacher <str>       the big model that answers prompts.jsonl (default
                        llama3.3:70b)
  --compare <str>       add this model's score to the table (repeatable)
  --base <str>          train the adapter on this (big) model itself instead
                        of the student
  --weight-own <float>  with both train.jsonl and prompts.jsonl: your answers
                        count this many times to the teacher's 1  [default:
                        2.0]
  --epochs <float>      passes over the training data (default 2)
  --notify <str>        POST a small JSON to this URL when done or stopped
  --quiet               one progress line per stage
  --help                Show this message and exit.

  Example: spill build banking77-quick
```

## spill example

```
Usage: spill example [OPTIONS] [name]

  Create ./<name>/ with ready files (an exam, homework, and the prompt for
  untrained models).

Arguments:
  name  which example (banking77)  [default: banking77]

Options:
  --quick  the under-an-hour variant: 100 evals, 500 train rows, student
           qwen2.5:0.5b
  --force  write into a folder that already exists
  --help   Show this message and exit.

  Example: spill example banking77 --quick
```

## spill run

```
Usage: spill run [OPTIONS] {model} {input_jsonl}

  Run a JSONL of prompts (or `sample`) through a model, a curated tag or any
  HF repo id.

Arguments:
  model        curated tag (llama3.3:70b), HF repo id (org/name[@rev]),
               optionally +<adapter> (local dir or HF repo)  [required]
  input_jsonl  prompts JSONL (plain, OpenAI batch or chat rows), or `sample`
               [required]

Options:
  --quant <str>         8bit|4bit (mlx) or Q8_0|Q4_K_M (gguf); bf16 is the
                        default
  --out <path>          also copy results.jsonl here
  --context <int>       context window in tokens  [default: 4096]
  --quiet               one progress line, no live slot block
  --notify <str>        POST a small JSON to this URL when done
  --logprobs <int>      per-token top-K log-probs, K up to 64
  --state <str>         portable job state: a path, s3://, gs://, az://
                        (resumes from the latest checkpoint there)
  --weights <str>       stage the model from this location instead of Hugging
                        Face
  --engine <str>        mlx | torch-cpu | torch-cuda (default: chosen from the
                        hardware)
  --headless            JSON-lines events on stdout, exit 75 when preempted
                        (automatic when stdout is not a terminal)
  --config <path>       run the invocation stored in this job.json
  --emit-config <path>  write this invocation to job.json and exit
  --help                Show this message and exit.

  Example: spill run qwen2.5:0.5b sample
```

## spill distill

```
Usage: spill distill [OPTIONS] {teacher} {input_jsonl}

  Collect a big model's answers to learn from: completions plus per-token
  top-k log-probs, or its log-probs over targets you supply (--score).

Arguments:
  teacher      teacher model: tag or HF repo id (optionally +adapter)
               [required]
  input_jsonl  prompts (batch/chat JSONL), or with --score chat rows that end
               with an assistant target  [required]

Options:
  --score               teacher-forced: score the given assistant targets,
                        prefill only, no sampling
  --logprobs <int>      top-K per token, K up to 64  [default: 32]
  --quant <str>         8bit | 4bit; bf16 is the default
  --out <path>          distillation JSONL (default runs/<id>/distill.jsonl)
  --context <int>       context window in tokens  [default: 4096]
  --quiet               one progress line, no live slot block
  --notify <str>        POST a small JSON to this URL when done
  --state <str>         portable job state: a path, s3://, gs://, az://
                        (resumes from the latest checkpoint there)
  --weights <str>       stage the model from this location instead of Hugging
                        Face
  --engine <str>        mlx | torch-cpu | torch-cuda (default: chosen from the
                        hardware)
  --headless            JSON-lines events on stdout, exit 75 when preempted
                        (automatic when stdout is not a terminal)
  --config <path>       run the invocation stored in this job.json
  --emit-config <path>  write this invocation to job.json and exit
  --help                Show this message and exit.

  Example: spill distill qwen2.5:0.5b sample
```

## spill tune

```
Usage: spill tune [OPTIONS] {model} {train_jsonl}

  Train a LoRA adapter on your data against the full-precision base.

Arguments:
  model        base model: tag, Hugging Face repo id, or a local safetensors
               directory  [required]
  train_jsonl  training JSONL: {prompt, answer} rows, or OpenAI chat rows
               (messages)  [required]

Options:
  --name <str>          adapter name (see: spill adapters)  [required]
  --rank <int>          LoRA rank  [default: 16]
  --lr <float>          learning rate  [default: 0.0001]
  --steps <int>         optimizer steps (default: --epochs of the data)
  --epochs <float>      passes over the data  [default: 1.0]
  --batch <int>         micro-batch (default: 4 resident, sized from the
                        memory budget when streamed)
  --grad-accum <int>    micro-batches per optimizer step  [default: 1]
  --max-seq <int>       token cap per example; whole exchanges are dropped
                        from the left  [default: 2048]
  --path <str>          auto | resident | streamed  [default: auto]
  --overwrite           replace an existing adapter
  --quiet               a progress line every 10 steps
  --notify <str>        POST a small JSON to this URL when done
  --state <str>         portable job state: a path, s3://, gs://, az://
                        (resumes from the latest checkpoint there)
  --weights <str>       stage the model from this location instead of Hugging
                        Face
  --engine <str>        mlx | torch-cpu | torch-cuda (default: chosen from the
                        hardware)
  --headless            JSON-lines events on stdout, exit 75 when preempted
                        (automatic when stdout is not a terminal)
  --config <path>       run the invocation stored in this job.json
  --emit-config <path>  write this invocation to job.json and exit
  --help                Show this message and exit.

  Example: spill tune qwen2.5:0.5b banking77-quick/train.jsonl --name banking
```

## spill eval

```
Usage: spill eval [OPTIONS] {input_jsonl} [models]...

  Run the eval set against each model (reusing finished runs for this exact
  input), score it, and print one table plus the rows where the models
  disagree.

Arguments:
  input_jsonl  eval JSONL: prompts plus an `expected` field per row
               [required]
  models...    one or more models, each optionally +adapter; omitted: every
               model with a run against this file's hash

Options:
  --metric <str>        exact_match | contains | regex | json_field | judge |
                        script:<file.py>  [default: exact_match]
  --judge <str>         judge model (implies --metric judge)
  --rerun               ignore cached runs for this input hash
  --quant <str>         8bit | 4bit; bf16 is the default
  --context <int>       context window in tokens  [default: 4096]
  --quiet               one progress line, no live slot block
  --notify <str>        POST a small JSON to this URL when done
  --state <str>         portable job state: a path, s3://, gs://, az://
                        (resumes from the latest checkpoint there)
  --weights <str>       stage the model from this location instead of Hugging
                        Face
  --engine <str>        mlx | torch-cpu | torch-cuda (default: chosen from the
                        hardware)
  --headless            JSON-lines events on stdout, exit 75 when preempted
                        (automatic when stdout is not a terminal)
  --config <path>       run the invocation stored in this job.json
  --emit-config <path>  write this invocation to job.json and exit
  --help                Show this message and exit.

  Example: spill eval banking77-quick/evals.jsonl qwen2.5:0.5b
  qwen2.5:0.5b+banking
```

## spill export

```
Usage: spill export [OPTIONS] {model}

  Merge the adapter into the base: merged safetensors, optionally GGUF and
  Ollama.

Arguments:
  model  <base>+<adapter>, e.g. qwen2.5:7b+banking77  [required]

Options:
  --out <path>  directory for the merged model (default:
                <data>/exports/<base>+<adapter>)
  --gguf <str>  also write a GGUF: bf16 | q8_0 | q4_k_m (bare --gguf: q8_0)
  --ollama      run `ollama create` if ollama is installed (implies --gguf)
  --name <str>  Ollama model name
  --help        Show this message and exit.

  Example: spill export qwen2.5:0.5b+banking --gguf
```

## spill models

```
Usage: spill models [OPTIONS]

  Curated tags: size, family, placement on this machine, disk needed,
  downloaded.

Options:
  --architectures  print the architecture support table
  --help           Show this message and exit.

  Example: spill models
```

## spill adapters

```
Usage: spill adapters [OPTIONS]

  List the LoRA adapters you have trained or added (PEFT or mlx-lm layout).

Options:
  --help  Show this message and exit.

  Example: spill adapters
```

## spill runs

```
Usage: spill runs [OPTIONS]

  List runs with model, quant, adapter, input hash, rows and status.

Options:
  --limit <int>  most recent N runs  [default: 20]
  --help         Show this message and exit.

  Example: spill runs
```

## spill status

```
Usage: spill status [OPTIONS]

  List jobs with progress, tokens/s, ETA.

Options:
  --help  Show this message and exit.

  Example: spill status
```

## spill tail

```
Usage: spill tail [OPTIONS] [job]

  Follow results.jsonl of the latest (or named) job.

Arguments:
  job  job id (default: the latest job)

Options:
  --help  Show this message and exit.

  Example: spill tail
```

## spill resume

```
Usage: spill resume [OPTIONS] [job]

  Continue the latest or named job from its checkpoint; a folder continues its
  build.

Arguments:
  job  job id or build folder (default: the latest job)

Options:
  --headless       JSON-lines events on stdout, exit 75 when preempted
  --weights <str>  stage the model from this location instead of Hugging Face
  --help           Show this message and exit.

  Example: spill resume banking77-quick
```

## spill doctor

```
Usage: spill doctor [OPTIONS]

  One screen: chip, memory, disk, models, interrupted jobs, what runs
  overnight.

Options:
  --help  Show this message and exit.

  Example: spill doctor
```

## spill check

```
Usage: spill check [OPTIONS] {path}

  Validate a JSONL file; prints the first error with its line number.

Arguments:
  path  a batch, chat, eval or distillation-target JSONL  [required]

Options:
  --help  Show this message and exit.

  Example: spill check banking77-quick/evals.jsonl
```
