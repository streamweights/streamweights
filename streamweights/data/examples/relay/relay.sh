#!/bin/sh
# Start a build here, stop it partway through the tune stage, finish it somewhere else.
#
#   ./relay/relay.sh                  finish in a Linux container if Docker is installed,
#                                     otherwise on the other engine on this machine
#   ./relay/relay.sh --no-docker      skip Docker
#   ./relay/relay.sh --two-machines   stop, then print what to copy and the one command to run
#   ./relay/relay.sh --stop=30        stop after 30 tune steps (default 20 of 50)
#
# Run it from the folder that holds relay/. Everything spill prints goes to relay.log.
set -eu

STOP=20
DOCKER=1
TWO=0
for a in "$@"; do
  case "$a" in
    --two-machines) TWO=1 ;;
    --no-docker) DOCKER=0 ;;
    --stop=*) STOP="${a#--stop=}" ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "relay: unknown option $a"; exit 2 ;;
  esac
done

IMAGE="${SPILL_IMAGE:-ghcr.io/streamweights/spill:cpu}"
cd "$(dirname "$0")/.."
LOG=relay.log
SPILL_HEADLESS=0
export SPILL_HEADLESS
rm -rf relay-state relay-ref relay-ref-state relay/.build
: > "$LOG"

say() { printf '%s\n' "$*"; }
run() {
  if ! "$@" >>"$LOG" 2>&1; then
    say "relay: a step failed. The end of $LOG:"
    tail -n 5 "$LOG"
    exit 1
  fi
}

ENGINES=$(spill doctor --engines)
FIRST=$(printf '%s\n' "$ENGINES" | sed -n 1p)

say "1. start the build here on $FIRST, $(uname -s)"
run spill build relay --state ./relay-state --stop-after "tune:$STOP"
say "   stopped after $STOP tune steps; the state is in ./relay-state"

if [ "$TWO" = 1 ]; then
  say "2. to finish it on another machine, copy these two folders there:"
  say "     relay/  relay-state/"
  say "   or point both machines at shared storage and pass it as --state:"
  say "     spill build relay --state s3://your-bucket/relay"
  say "   then run, on the other machine, from the folder that holds them:"
  say "     spill resume relay --state relay-state"
  say "   and see the table at any time with:"
  say "     spill build relay --state relay-state --table"
  exit 0
fi

if [ "$DOCKER" = 1 ] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  CACHE="${SPILL_RELAY_CACHE:-$HOME/.cache/spill-relay}"
  mkdir -p "$CACHE"
  say "2. finish it in a Linux container: $IMAGE"
  run docker run --rm --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e SPILL_HOME=/spill-home -e HF_HOME=/spill-home/huggingface -e SPILL_HEADLESS=0 \
    -v "$PWD/relay:/work/relay" -v "$PWD/relay-state:/work/relay-state" \
    -v "$CACHE:/spill-home" -w /work "$IMAGE" resume relay --state /work/relay-state
else
  OTHER=$(printf '%s\n' "$ENGINES" | sed -n 2p)
  if [ -n "$OTHER" ]; then
    say "2. finish it on $OTHER (a different engine on the same machine, not a second machine)"
  else
    OTHER="$FIRST"
    say "2. only one engine here ($FIRST), so this finishes on the same engine"
    say "   (use --two-machines for a real hop)"
  fi
  run spill resume relay --state ./relay-state --engine "$OTHER"
fi
say "   finished"

say "3. the same build, start to finish, for a reference"
cp -R relay relay-ref
rm -rf relay-ref/.build
run spill build relay-ref --state ./relay-ref-state
say "   done"

say ""
say "the final table: who made each stage, and the score next to the reference"
spill build relay --state ./relay-state --table --reference ./relay-ref-state
