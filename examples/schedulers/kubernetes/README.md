# Kubernetes

A Job that runs `spill tune` from a config, with a PersistentVolumeClaim for the state and the
weights cache.

| file | what |
|---|---|
| `pvc.yaml` | the volume: `/data/home` (SPILL_HOME, weights cache) and `/data/state` (`--state`) |
| `configmap.yaml` | the job config, as written by `--emit-config` on a laptop |
| `job-cuda.yaml` | the Job on an NVIDIA GPU, image `ghcr.io/streamweights/spill:cuda` |
| `job-cpu.yaml` | the same Job on CPU nodes, image `ghcr.io/streamweights/spill:cpu` |

`restartPolicy: OnFailure` is the retry: spill exits 75 after writing a checkpoint when it is
told to stop (a node drain, a preemption), Kubernetes restarts the container, and the same
command resumes from `/data/state`. `terminationGracePeriodSeconds: 45` leaves room for the 30
seconds a checkpoint may take.

```
spill tune qwen2.5:0.5b train.jsonl --name mine --state /data/state/mine --emit-config tune.json
kubectl create configmap spill-job --from-file=tune.json
kubectl apply -f examples/schedulers/kubernetes/pvc.yaml
kubectl apply -f examples/schedulers/kubernetes/job-cuda.yaml      # or job-cpu.yaml
kubectl logs -f job/spill-tune                                      # JSON-lines events
```

The training file referenced by the config (`/data/inputs/train.jsonl` in the example) must be on
the volume, or named by an `s3://`, `gs://` or `az://` URI.
