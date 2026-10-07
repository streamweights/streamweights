# Slurm

`tune.sbatch` runs `spill tune --config` headless with everything a Slurm job needs to be
interrupted and continued.

- **`--requeue`** lets Slurm put the job back in the queue after a node failure or preemption.
- **`--signal=B:TERM@120`** sends SIGTERM to the batch shell 120 seconds before the time limit.
  The `trap` forwards it to `spill`, which finishes or abandons the current step, writes a
  checkpoint inside 30 seconds and exits 75. The script then runs `scontrol requeue`, and the
  next run of the same script resumes from the checkpoint. Exit code 75 is the contract: any
  other non-zero code is a real failure and is passed through.
- **A shared filesystem for state.** `--state` in the config names a directory under `$SHARED`,
  which every node mounts. Weights can be staged from there too (`--weights`).

```
# on a laptop
spill tune qwen2.5:7b train.jsonl --name mine --state /shared/me/spill/state/mine \
      --weights /shared/me/spill/weights/qwen2.5-7b --emit-config tune.json
scp tune.json cluster:/shared/me/spill/jobs/

# on the cluster
sbatch examples/schedulers/slurm/tune.sbatch
```

The events the job writes (one JSON object per line: `start`, `step`, `checkpoint`,
`preempted`, `done`, `error`) land in `spill-tune-<jobid>.log`.
