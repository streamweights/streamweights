# Scheduler examples

`spill` jobs are built to be preempted. Every job is a sequence of quanta (a training step, or a
completed row) and any machine can continue from the checkpoint at `--state`. Under a scheduler
that means: run the job headless, point `--state` (and optionally `--weights`) at storage every
machine can reach, and let the scheduler rerun the same command when it stops the job. On SIGTERM
or SIGINT the job writes a checkpoint within 30 seconds and exits with code 75 so a scheduler
retries it; the retry resumes from the checkpoint.

| scheduler | directory | what it does |
|---|---|---|
| SkyPilot | [skypilot/](skypilot/) | a managed job on spot instances, any of H100, A100, L4, A10G or CPU only, with automatic recovery |
| Slurm | [slurm/](slurm/) | an sbatch script with `--requeue` and a SIGTERM handler, state on a shared filesystem |
| Kubernetes | [kubernetes/](kubernetes/) | a Job with a PVC for state and weights, `restartPolicy: OnFailure`, a CUDA image and a CPU variant |

All three run the same thing: a config written on a laptop,

```
spill tune qwen2.5:7b train.jsonl --name mine --state /data/state/mine \
      --weights /data/weights/qwen2.5-7b --emit-config tune.json
```

which writes `tune.json` (the exact, fully resolved invocation) and does not run it, and then on
the cluster

```
spill tune --config tune.json
```

The events a headless job writes to stdout, and the exit codes, are in
[../../docs/schedulers.md](../../docs/schedulers.md). `validation.md` records how each file here
was checked and what could not be checked.
