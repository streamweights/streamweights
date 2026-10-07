"""Validate every file under examples/schedulers/ offline and write examples/schedulers/validation.md.

  YAML syntax          every .yaml, with PyYAML
  SkyPilot             sky.Task.from_yaml in the interpreter named by $SKY_PYTHON (SkyPilot is not a
                       dependency of streamweights); skipped and logged when it is absent
  Kubernetes           kubectl apply --dry-run=client when kubectl exists; otherwise the
                       kubernetes-validate schemas for the Kubernetes version below
  Slurm                sbatch --test-only when sbatch exists; always bash -n and a check of every
                       #SBATCH directive against the known option list
  spill config         the job config inside the ConfigMap loads and parses with the real CLI
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "examples" / "schedulers"
K8S_VERSION = "1.30"
SBATCH_OPTS = {"job-name", "requeue", "open-mode", "signal", "time", "gres", "cpus-per-task",
               "mem", "output", "error", "nodes", "ntasks", "partition", "account", "qos",
               "constraint", "exclusive", "mail-type", "mail-user", "array", "chdir",
               "gpus", "gpus-per-node", "ntasks-per-node", "mem-per-cpu", "begin", "dependency",
               "no-requeue", "export", "nodelist", "exclude", "reservation", "comment"}

log: list[tuple[str, str, str]] = []      # (file, check, result)


def note(file, check, result):
    log.append((str(Path(file).relative_to(REPO)) if Path(file).is_absolute() else str(file),
                check, result))
    print(f"{log[-1][0]:60s} {check:34s} {result}")


def yaml_files():
    return sorted(ROOT.rglob("*.yaml"))


def check_yaml():
    for f in yaml_files():
        try:
            docs = list(yaml.safe_load_all(f.read_text()))
            note(f, "YAML syntax (PyYAML)", f"ok, {len(docs)} document(s)")
        except yaml.YAMLError as e:
            note(f, "YAML syntax (PyYAML)", f"FAILED: {e}")
            raise SystemExit(1)


def check_skypilot():
    f = ROOT / "skypilot" / "job.yaml"
    py = os.environ.get("SKY_PYTHON")
    if not py:
        note(f, "SkyPilot Task.from_yaml", "NOT RUN: set SKY_PYTHON to an interpreter with skypilot")
        return
    code = ("import sys, sky;"
            "t = sky.Task.from_yaml(sys.argv[1]);"
            "r = list(t.resources);"
            "print(len(r), 'resource option(s):', sorted({(str(x.accelerators), x.use_spot) "
            "for x in r}, key=str));"
            "print('envs', sorted(t.envs));"
            "print('file mounts', sorted(t.storage_mounts))")
    p = subprocess.run([py, "-c", code, str(f)], capture_output=True, text=True,
                       env={**os.environ, "BUCKET": "my-spill-bucket"})
    if p.returncode:
        note(f, "SkyPilot Task.from_yaml", "FAILED: " + (p.stderr.strip().splitlines() or ["?"])[-1])
        raise SystemExit(1)
    ver = subprocess.run([py, "-c", "import sky; print(sky.__version__)"], capture_output=True,
                         text=True).stdout.strip()
    note(f, f"SkyPilot {ver} Task.from_yaml", "ok: " + p.stdout.strip().replace("\n", "; "))


def check_k8s():
    files = [f for f in yaml_files() if f.parent.name == "kubernetes"]
    if shutil.which("kubectl"):
        for f in files:
            p = subprocess.run(["kubectl", "apply", "--dry-run=client", "-f", str(f)],
                               capture_output=True, text=True)
            note(f, "kubectl apply --dry-run=client",
                 "ok" if p.returncode == 0 else "FAILED: " + p.stderr.strip()[:120])
    else:
        note("examples/schedulers/kubernetes/*.yaml", "kubectl apply --dry-run=client",
             "NOT RUN: kubectl is not installed here")
    try:
        import kubernetes_validate as kv
    except ImportError:
        note("examples/schedulers/kubernetes/*.yaml", "kubernetes-validate schemas",
             "NOT RUN: pip install kubernetes-validate")
        return
    for f in files:
        for doc in yaml.safe_load_all(f.read_text()):
            try:
                kv.validate(doc, K8S_VERSION, strict=True)
                note(f, f"kubernetes-validate {K8S_VERSION} (strict)", f"ok: {doc['kind']}")
            except Exception as e:
                note(f, f"kubernetes-validate {K8S_VERSION} (strict)", f"FAILED: {e}")
                raise SystemExit(1)
    # the invariants the directive asks for
    job = yaml.safe_load((ROOT / "kubernetes/job-cuda.yaml").read_text())
    assert job["spec"]["template"]["spec"]["restartPolicy"] == "OnFailure"
    cpu = yaml.safe_load((ROOT / "kubernetes/job-cpu.yaml").read_text())
    assert cpu["spec"]["template"]["spec"]["restartPolicy"] == "OnFailure"
    note("examples/schedulers/kubernetes/job-*.yaml", "restartPolicy: OnFailure", "ok (both jobs)")
    note("examples/schedulers/kubernetes/job-cuda.yaml", "uses PVC spill-data", "ok"
         if job["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"]
         == "spill-data" else "FAILED")


def check_slurm():
    f = ROOT / "slurm" / "tune.sbatch"
    p = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
    note(f, "bash -n", "ok" if p.returncode == 0 else "FAILED: " + p.stderr[:120])
    if p.returncode:
        raise SystemExit(1)
    opts = re.findall(r"^#SBATCH\s+--([a-z-]+)", f.read_text(), re.M)
    bad = [o for o in opts if o not in SBATCH_OPTS]
    note(f, "#SBATCH directives", f"ok: {', '.join(opts)}" if not bad else f"FAILED: unknown {bad}")
    text = f.read_text()
    assert "--requeue" in text and "trap on_term TERM" in text and "scontrol requeue" in text
    note(f, "--requeue, SIGTERM trap, scontrol requeue", "ok")
    if shutil.which("sbatch"):
        p = subprocess.run(["sbatch", "--test-only", str(f)], capture_output=True, text=True)
        note(f, "sbatch --test-only", "ok" if p.returncode == 0 else "FAILED: " + p.stderr[:120])
    else:
        note(f, "sbatch --test-only", "NOT RUN: sbatch is not installed here")


def check_config():
    sys.path.insert(0, str(REPO))
    from typer.main import get_command

    from streamweights import jobconfig
    from streamweights.cli import app
    cm = yaml.safe_load((ROOT / "kubernetes/configmap.yaml").read_text())
    cfg = json.loads(cm["data"]["tune.json"])
    tmp = REPO / "state" / "files" / "validate-tune.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(cfg))
    loaded = jobconfig.load(tmp)
    cmd = get_command(app).commands[loaded["command"]]
    argv = jobconfig.to_argv(loaded, cmd, [])
    again = jobconfig.build(loaded["command"], cmd, argv)
    assert again["options"]["state"] == "/data/state/mine" and again["options"]["steps"] == 100
    note("examples/schedulers/kubernetes/configmap.yaml", "embedded tune.json parses with the CLI",
         "ok: spill tune " + " ".join(argv))


if __name__ == "__main__":
    check_yaml()
    check_skypilot()
    check_k8s()
    check_slurm()
    check_config()
    out = ["# Validation log", "",
           "Written by `python scripts/validate_schedulers.py`. Every file under this directory was "
           "checked offline; a row that says NOT RUN could not be checked on the machine that "
           "wrote this log.", "", "| file | check | result |", "|---|---|---|"]
    out += [f"| `{a}` | {b} | {c} |" for a, b, c in log]
    not_run = [r for r in log if r[2].startswith("NOT RUN")]
    out += ["", "Not validated here: " + ("; ".join(f"{b} ({a})" for a, b, _ in not_run)
                                          if not_run else "nothing")]
    (ROOT / "validation.md").write_text("\n".join(out) + "\n")
    print("wrote", ROOT / "validation.md")
