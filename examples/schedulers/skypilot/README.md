# SkyPilot

`job.yaml` is a managed-job task that runs `spill build`, headless, from an emitted config. `sky jobs launch` runs it on spot instances and relaunches it
on a new machine when the spot instance is reclaimed.

- **Any of H100, A100, L4, A10G, or CPU only.** The `any_of` list under `resources` lets
  SkyPilot take whichever is available and cheapest. The engine is chosen on the machine that
  runs the job (torch-cuda with a GPU, torch-cpu without), and the checkpoint is
  engine-neutral, so a job can start on an L4 and finish on a CPU.
- **Spot with automatic recovery.** `use_spot: true` and `job_recovery`. A recovered job runs
  the same `spill build --config ...` command; it resumes at the stage and step where the build stopped, from `--state`.
- **A bucket for state and weights.** `file_mounts` mounts a bucket at `/bucket`; the folder of
  data and the build's state live there (`--state /bucket/state/mine`); models download from
  Hugging Face to each machine.
- **Headless, from an emitted config.** Write the config on a laptop, upload it, and launch:

```
spill build /bucket/data/mine --state /bucket/state/mine --emit-config build.json
cp build.json /path/to/your/mount/jobs/       # or: gsutil cp / aws s3 cp build.json s3://my-spill-bucket/jobs/
sky jobs launch examples/schedulers/skypilot/job.yaml --env BUCKET=my-spill-bucket
sky jobs queue
```

Cloud credentials for `--state s3://...` or `gs://...` directly (without a mount) need the cloud
extras: `pip install "streamweights[cloud] @ git+https://github.com/streamweights/streamweights"`.
