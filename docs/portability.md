---
description: How a spill job moves between machines: the portable checkpoint, what it contains, its size, numerics across hardware and preemption behavior.
---

# Portability: what moves, and what it costs

A `spill` job is a sequence of **quanta**. For training the quantum is one optimizer step. For
`run`, `distill` and `eval` it is one completed row. After every quantum the job could stop, and
any machine with a supported engine can execute the next one from a portable checkpoint. That is
the whole design: engines are thin (MLX on Apple silicon, PyTorch everywhere else, CPU and CUDA),
and streamweights owns only the streaming ring, the job layer and the CLI.

```
spill tune qwen2.5:7b train.jsonl --name mine --state s3://my-bucket/mine      # laptop
spill tune qwen2.5:7b train.jsonl --name mine --state s3://my-bucket/mine      # any GPU box, later
```

The second command finds the committed checkpoint at the URI and continues from it.

## What moves

`--state <uri>` takes a local path, `s3://`, `gs://`, `az://` (through fsspec; the cloud backends
come with `pip install "streamweights[cloud] @ git+https://github.com/streamweights/streamweights"`)
or `memory://` (tests). Everything below lives under that URI.

**A tune checkpoint**, `ckpt/step-<n>/`:

| file | contents |
|---|---|
| `params.safetensors` | the LoRA parameters, float32, named `layers.<k>.<module>.lora_a` (`[in, r]`) and `lora_b` (`[r, out]`) |
| `opt.safetensors` | AdamW state, float32: `m::<name>`, `v::<name>` |
| `state.json` | step, seed, data cursor, RNG per data stream, model id and weight fingerprint, adapter config, hyperparameters, the loss of every step so far, and the engine, hardware and numerics that produced each range of steps |
| `COMMIT` | written last: the size and sha256 of the three files above |

Writes are atomic. Locally each file is written to a temporary name and renamed. On an object
store a single put is atomic, and the multi-file checkpoint is made atomic by the `COMMIT` marker:
a reader only trusts a directory that has one, and checks the sizes and hashes in it. A writer
that dies halfway leaves an uncommitted directory that is ignored and removed by the next save.
The two newest committed checkpoints are kept.

**A row job**, `rows/seg-<n>.jsonl` plus `rows/seg-<n>.commit`: the result lines completed since
the last push, and a marker with their row ids, a hash, and the engine, hardware and numerics that
produced them. A machine that starts the same job restores every committed segment into its own
job directory (`results.jsonl` and the cursor of completed row ids) and runs only what is left.
`rows/identity.json` pins the model and the input, so a state location cannot be continued by a
different job.

**Any engine reads any engine's checkpoint.** MLX and PyTorch write and read the same files. The
data cursor is the whole data state: batches are a pure function of (seed, micro-batch index), so
there is no RNG stream to carry. A job refuses, in one line, to continue from a checkpoint made
with different weights (the fingerprint is a hash of the config, shard names, sizes and
safetensors headers), a different training file, a different learning rate, schedule, batch
shape or seed, or a different number of total steps (the cosine schedule is part of the
checkpoint's identity).

What does not move is the model: weights are a per-machine cache under `SPILL_HOME`.

## Numerics across hardware

The adapter and the optimizer are float32 on every engine. The base weights are bf16 on MLX and
CUDA. On a CPU they are bf16 where the CPU multiplies bf16 matrices fast and float32 where it does
not; the pre-run line says which (the check is a measured linear layer, not a CPU model list).
The checkpoint records the numerics of every range of steps, so a job that ran 50 steps in bf16
on a GPU and 50 in float32 on a CPU says so.

Measured on the 0.5B on one M4 Pro (`docs/reports/012-run-anywhere.md` has every number):

| comparison | result |
|---|---|
| torch-cpu streamed vs torch-cpu resident, float32, 20 prompts | greedy output identical on 20 of 20; maximum log-prob difference 0 |
| torch-cpu vs MLX, float32, 20 prompts | greedy identical on 20 of 20; maximum log-prob difference 0.000182 |
| torch-cpu streamed tune vs PEFT resident tune, float32, 50 steps | loss within 8.4e-7 relative per step; adapter cosine 0.999996 at the worst tensor |
| torch-cpu tune vs MLX tune, float32, 50 steps | loss within 6.1e-5 relative per step on text, 2.0e-3 on the toy task; both adapters score 1.0 on 60 held-out toy rows (the base scores 0) |
| bf16 | not an identity claim; statistics next to batch-shape noise in the report |

In bf16 two correct paths differ by rounding noise. Against a float32 reference, the per-tensor
cosine of one micro-batch's gradient has a mean of 0.974 and a minimum of 0.702 for torch bf16 and
0.988 and 0.742 for MLX bf16; the same MLX run changes its own gradient with the batch shape (one
micro-batch against two halves) to a mean cosine of 0.990 and a minimum of 0.876. Moving a job
between engines mid-run is the same kind of difference: continued across hardware, the loss curve
stays within the distance that two clean runs on different engines have from each other
(gate 8 in the report).

## A build moves too

`spill build <folder> --state <uri>` keeps the whole build in the same portable layout: the stage
reached, each stage's checkpoint, the intermediate files and the input fingerprint.

| under `--state` | contents |
|---|---|
| `state.json` | the plan, every stage's status and result, the engine, hardware, OS and numerics that produced each stage, the input fingerprint. One atomic object, written after everything it names |
| `stages/<stage>/` | the stage's own state: a tune checkpoint (`ckpt/`) or committed row segments (`rows/`) |
| `files/` | the teacher's answers, once the distill stage is done |
| `adapters/<name>/` | the tuned adapter in both layouts, once the tune stage is done |

Without `--state` it is the folder's own `.build/`. A build stopped on one engine or machine is
continued by running the same command, or `spill resume <folder> --state <uri>`, on another: stages
that finished are skipped, the stage that stopped continues at its row or step, and the models the
build started with are kept (the defaults are chosen from the machine only at the start). Edited inputs start the state fresh. The final table shows which engine, machine and OS made each stage; a stage that moved lists both.

`--stop-after tune:20` stops a build 20 steps into the tune stage (`distill:5`, `eval:base` and a
bare stage name work too); it exits 0 and the state is complete.

### The relay

`spill example relay && ./relay/relay.sh` starts a build, stops it partway through the tune stage and
finishes it somewhere else: in the `ghcr.io/streamweights/spill:cpu` container if Docker is
installed, otherwise on the other engine on the same machine, or with `--two-machines` it prints what to copy
and the command to run there.

`.github/workflows/relay.yml` does it on every push, between real machines: a build started on a Linux runner is finished on a macOS runner, and one started on macOS is finished on Linux, each stopped
with `--stop-after tune:20`; an uninterrupted Linux build is the reference. The last job checks
that:

- both relays completed;
- each relay's base and tuned scores are within the tolerance of the reference. The tolerance is
  measured, not chosen: the tiny build grades 20 rows, six runs of it on one Mac (two engines, two
  batch shapes, a build moved between engines in both directions) spread 0.10 in tuned score, and the tolerance is that spread plus one row, 0.15. It is stored with
  its justification in `docs/reports/014-noise-floor.json`. Agreement tighter than 0.15 is not claimed;
- no row and no step is missing or repeated (read off the committed segments and the checkpoint
  in the state);
- each stage's recorded machine and OS are the machine and OS of the job that ran it.

It writes a summary to the run page and commits nothing. Where the macOS runner has no Metal the macOS side runs on torch-cpu, and the table records which engine ran.

## What a quantum costs

- **A tune step** writes a checkpoint every `--ckpt-every` steps (default 50) and on stop. The
  rank 16 adapter on every linear of the 0.5B is 101 MB per checkpoint (35 MB of parameters and
  70 MB of optimizer state); the committed fixture, rank 4 on `q_proj` and `v_proj`, is 3.3 MB.
  The same rank 16 on the 70B has 207.1M parameters ([report 008](reports/008-phase3.md)), which
  is about 0.8 GB of float32 parameters and 1.7 GB of optimizer state: about 2.5 GB per
  checkpoint (computed from the parameter count, not a measured upload). Checkpoint size is
  about 100 MB on the 0.5B and about 2.5 GB on the 70B at rank 16 on every linear; it scales
  with rank and with the adapted modules, and not with the sequence length or the data.
- **A row** is pushed in segments of 25 rows or 60 seconds, whichever comes first, and once more
  when the job stops. A preempted job loses at most the rows in flight.

## Weight staging cost

Weights cache per machine under `SPILL_HOME/models`. The default source is Hugging Face. With
`--weights <uri>` a machine copies a model directory (the `*.safetensors` shards and the small
config and tokenizer files) from shared storage instead, once: a file whose size already matches
is not copied again. Staging is a plain copy, so the cost is size divided by the rate to the
store. Measured: the 0.5B (998 MB) staged from a local directory in 0.21 s (4.7 GB/s); a second
stage on the same machine copied nothing and took under a millisecond. From an object store the
time is the object store's rate to that machine, and the job prints the bytes and seconds
staging took. A model that is bigger than RAM is still streamed from the local copy, so staging is paid
once per machine and then every pass reads local disk.

## Preemption

On SIGTERM or SIGINT a headless job finishes the current quantum if it ends within 15 seconds,
otherwise abandons it, writes a checkpoint, emits `preempted` and exits with code 75 so that a
scheduler retries it (see [schedulers.md](schedulers.md)). Measured in the tests: SIGTERM to exit
in under a second on a tiny model; the deadline is 30 seconds in all cases.
