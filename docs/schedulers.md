# Running under a scheduler

Three complete examples live in [examples/schedulers/](../examples/schedulers/): SkyPilot (a
managed job on spot instances, any of H100, A100, L4, A10G or CPU only), Slurm (an sbatch script
with `--requeue` and a SIGTERM handler) and Kubernetes (a Job with a PVC, `restartPolicy:
OnFailure`, a CUDA image and a CPU variant). Each is a short README plus the files, validated
offline; `examples/schedulers/validation.md` records how, and what could not be checked.

They all do the same thing:

1. On a laptop, write the job as a file. `--emit-config` writes the exact, fully resolved
   invocation and exits without running it:

   ```
   spill tune qwen2.5:7b train.jsonl --name mine --state s3://bucket/state/mine \
         --weights s3://bucket/weights/qwen2.5-7b --emit-config tune.json
   ```

2. On the cluster, run it. `--config` loads the file (options given next to it win):

   ```
   spill tune --config tune.json --headless
   ```

3. When the scheduler stops the job, run the same command again. The job resumes from the
   committed checkpoint at `--state`, on whatever machine and engine the scheduler gives it.

Paths in a config are used as written, so a job meant for a cluster names its data and state by
URI or by a path that exists there (a bucket mount, a shared filesystem, a volume).
`run`, `distill`, `tune` and `eval` take `--config` and `--emit-config`.

## Headless mode

Enabled with `--headless`, or automatically when stdout is not a terminal (`SPILL_HEADLESS=0`
or `1` forces it off or on). Then stdout carries JSON lines and nothing else, and human text goes
to stderr. There are no progress bars, no notifications and no `caffeinate`.

Every event carries the same keys: `event`, `time`, `command`, `step`, `loss`, `tokens_per_s`,
`peak_mem_gb`, `eta_s` and `engine` (null where a key does not apply), plus fields of its own.

| event | when | extra fields |
|---|---|---|
| `start` | the job is sized and about to run | `job`, `model`, `placement`, `state`, `estimated_seconds`, `message` (the pre-run line) |
| `restore` | rows were restored from `--state` | `step` (rows restored), `uri` |
| `step` | a tune step finished | `steps`, `step_s` |
| `row` | a row finished (run, distill, eval) | `total`, `custom_id` |
| `checkpoint` | state was committed at `--state` | `uri`, `step` (or `rows`) |
| `preempted` | stopped by SIGTERM or SIGINT, checkpoint written | `signal`, `abandoned_quantum` |
| `done` | the command finished | `complete`, `stopped_early`, `seconds`, and the job's summary |
| `error` | the command failed | `message` (the one-line error, ending in the command to try) |

## Exit codes

| code | meaning |
|---|---|
| 0 | done (or stopped by `--stop-after`) |
| 1 | an error; the `error` event and stderr carry the one-line message |
| 75 | preempted: a checkpoint was written within 30 seconds and a retry resumes it |

A scheduler should treat 75 as "run me again", not as a failure. Kubernetes does this with
`restartPolicy: OnFailure`; the Slurm script requeues itself on 75; SkyPilot reruns the job after
it recovers a preempted spot instance.

## Engines under a scheduler

The engine is chosen on the machine that runs the job: Apple silicon uses MLX, a visible CUDA
device uses torch-cuda, anything else torch-cpu; `--engine` overrides. The checkpoint is the
same on all of them, so a job may start on one kind of machine and finish on another.
`spill doctor` shows the choice and a measured rate for each usable engine. The containers
(`ghcr.io/streamweights/spill:cpu` and `:cuda`) have `spill` as the entrypoint, are headless by
default and keep `SPILL_HOME` on the `/data` volume.
