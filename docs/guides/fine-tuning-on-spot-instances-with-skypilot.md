---
description: Run LLM fine-tuning on spot instances with SkyPilot. spill writes the job as a config file, SkyPilot relaunches it after each preemption, and it resumes from the checkpoint in your bucket.
---

# Run fine-tuning on spot instances with SkyPilot

Write the build as a config on your laptop with `--emit-config`, point `--state` at a bucket, and launch the shipped SkyPilot job with `sky jobs launch`. When a spot instance is reclaimed SkyPilot starts a new one and the job resumes from the last committed checkpoint, on whatever GPU or CPU it gets.

## The steps

1. Write the job as a file on your laptop. Nothing runs:

   ```
   spill build /bucket/data/mine --state /bucket/state/mine --emit-config build.json
   ```

2. Upload `build.json` to your bucket under `jobs/` (the data folder goes in the bucket too), then launch:

   ```
   sky jobs launch examples/schedulers/skypilot/job.yaml --env BUCKET=my-spill-bucket
   sky jobs queue
   ```

3. The task runs `spill build --config /bucket/jobs/build.json --headless` on spot instances, any of H100, A100, L4, A10G or CPU only. A recovered job runs the same command and continues at the stage and step where the build stopped, from `--state`. Headless jobs write JSON-lines events on stdout and exit 75 after a checkpoint on SIGTERM.

The checkpoint is engine-neutral, so a job can start on an L4 and finish on a CPU. At rank 16 a checkpoint is about 100 MB on the 0.5B and about 2.5 GB on the 70B ([portability](../portability.md)). The container images `ghcr.io/streamweights/spill:cpu` and `:cuda` run `spill` headless. NVIDIA is built and awaiting verification: report your hardware on [issue #1](https://github.com/streamweights/streamweights/issues/1). Slurm and Kubernetes examples are in [schedulers](../schedulers.md).

## Try it

```
sky jobs launch examples/schedulers/skypilot/job.yaml --env BUCKET=my-spill-bucket
```

Source and issues: [github.com/streamweights/streamweights](https://github.com/streamweights/streamweights).
