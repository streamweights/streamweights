# Validation log

Written by `python scripts/validate_schedulers.py`. Every file under this directory was checked offline; a row that says NOT RUN could not be checked on the machine that wrote this log.

| file | check | result |
|---|---|---|
| `examples/schedulers/kubernetes/configmap.yaml` | YAML syntax (PyYAML) | ok, 1 document(s) |
| `examples/schedulers/kubernetes/job-cpu.yaml` | YAML syntax (PyYAML) | ok, 1 document(s) |
| `examples/schedulers/kubernetes/job-cuda.yaml` | YAML syntax (PyYAML) | ok, 1 document(s) |
| `examples/schedulers/kubernetes/pvc.yaml` | YAML syntax (PyYAML) | ok, 1 document(s) |
| `examples/schedulers/skypilot/job.yaml` | YAML syntax (PyYAML) | ok, 1 document(s) |
| `examples/schedulers/skypilot/job.yaml` | SkyPilot 0.14.0 Task.from_yaml | ok: 5 resource option(s): [("{'A100': 1}", True), ("{'A10G': 1}", True), ("{'H100': 1}", True), ("{'L4': 1}", True), ('None', True)]; envs ['BUCKET', 'SPILL_HOME']; file mounts ['/bucket'] |
| `examples/schedulers/kubernetes/*.yaml` | kubectl apply --dry-run=client | NOT RUN: kubectl is not installed here |
| `examples/schedulers/kubernetes/configmap.yaml` | kubernetes-validate 1.30 (strict) | ok: ConfigMap |
| `examples/schedulers/kubernetes/job-cpu.yaml` | kubernetes-validate 1.30 (strict) | ok: Job |
| `examples/schedulers/kubernetes/job-cuda.yaml` | kubernetes-validate 1.30 (strict) | ok: Job |
| `examples/schedulers/kubernetes/pvc.yaml` | kubernetes-validate 1.30 (strict) | ok: PersistentVolumeClaim |
| `examples/schedulers/kubernetes/job-*.yaml` | restartPolicy: OnFailure | ok (both jobs) |
| `examples/schedulers/kubernetes/job-cuda.yaml` | uses PVC spill-data | ok |
| `examples/schedulers/slurm/tune.sbatch` | bash -n | ok |
| `examples/schedulers/slurm/tune.sbatch` | #SBATCH directives | ok: job-name, requeue, open-mode, signal, time, gres, cpus-per-task, mem, output |
| `examples/schedulers/slurm/tune.sbatch` | --requeue, SIGTERM trap, scontrol requeue | ok |
| `examples/schedulers/slurm/tune.sbatch` | sbatch --test-only | NOT RUN: sbatch is not installed here |
| `examples/schedulers/kubernetes/configmap.yaml` | embedded tune.json parses with the CLI | ok: spill tune qwen2.5:0.5b /data/inputs/train.jsonl --name mine --state /data/state/mine --steps 100 --lr 2e-05 --batch 4 --max-seq 256 |

Not validated here: kubectl apply --dry-run=client (examples/schedulers/kubernetes/*.yaml); sbatch --test-only (examples/schedulers/slurm/tune.sbatch)
