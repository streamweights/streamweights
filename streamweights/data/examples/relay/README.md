# relay

One build, started on one machine and finished on another. `relay.sh` starts
`spill build relay --state ./relay-state`, stops it partway through the tune stage, finishes
it somewhere else, and prints a table of which engine, machine and OS made each stage next to
the score of an uninterrupted build.

```
$ spill example relay && ./relay/relay.sh
```

How it finishes, chosen in this order:

1. Docker is installed: it runs `ghcr.io/streamweights/spill:cpu` with `relay/` and
   `relay-state/` mounted and runs `spill resume` inside the Linux container.
2. No Docker: `spill resume --engine` with the other engine on this machine (and one line that
   says it is a different engine on the same machine, not a second machine).
3. `./relay/relay.sh --two-machines`: it prints what to copy, or which shared path or bucket to
   point `--state` at, and the one command to run on the other machine.

The data is the tiny banking77 build (10 intents, 20 exam rows, 100 training rows). Data:
BANKING77 (Casanueva et al., 2020), PolyAI, CC BY 4.0; attribution in the README of
`spill example banking77`.

## A run

TRANSCRIPT
