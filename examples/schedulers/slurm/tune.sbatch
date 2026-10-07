#!/bin/bash
# spill tune under Slurm: requeueable, with a SIGTERM handler, state on a shared filesystem.
#
#   sbatch examples/schedulers/slurm/tune.sbatch
#
# Slurm sends SIGTERM to the batch shell 120 s before the time limit (--signal=B:TERM@120) and
# on preemption. The handler forwards it to spill, which writes a checkpoint within 30 seconds
# and exits 75; the script then requeues itself and the next run resumes from the checkpoint.

#SBATCH --job-name=spill-tune
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.log

set -u

SHARED=${SHARED:-/shared/$USER/spill}       # a filesystem every node mounts
CONFIG=${CONFIG:-$SHARED/jobs/tune.json}    # written on a laptop with --emit-config
export SPILL_HOME=${SPILL_HOME:-$SHARED/home}
export SPILL_HEADLESS=1

# once per cluster: python -m venv $SHARED/venv && $SHARED/venv/bin/pip install \
#   "streamweights[cloud] @ git+https://github.com/streamweights/streamweights"
source "$SHARED/venv/bin/activate"

spill_pid=""
on_term() {
  echo "$(date -Is) slurm sent SIGTERM; asking spill to checkpoint"
  if [ -n "$spill_pid" ]; then
    kill -TERM "$spill_pid" 2>/dev/null
    wait "$spill_pid"
    rc=$?
    if [ "$rc" -eq 75 ]; then
      echo "$(date -Is) spill checkpointed and exited 75; requeueing job $SLURM_JOB_ID"
      scontrol requeue "$SLURM_JOB_ID"
      exit 0
    fi
    exit "$rc"
  fi
  exit 143
}
trap on_term TERM

# the config names --state under $SHARED, so every node sees the same checkpoint
spill tune --config "$CONFIG" --headless &
spill_pid=$!
wait "$spill_pid"
rc=$?
spill_pid=""

if [ "$rc" -eq 75 ]; then
  echo "$(date -Is) spill was preempted (exit 75); requeueing job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$rc"
