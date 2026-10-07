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

The Docker mode, as run by the relay workflow on a Linux runner (`.github/workflows/relay.yml`, the Docker relay job). The machine column lists the host and then the container that finished the tune:

```
1. start the build here on torch-cpu, Linux
   stopped after 20 tune steps; the state is in ./relay-state
2. finish it in a Linux container: ghcr.io/streamweights/spill:cpu
   finished
3. the same build, start to finish, for a reference
   done

the final table: who made each stage, and the score next to the reference
stage         engine     machine                        os            score  reference
1 eval:base   torch-cpu  runnervmmprz5                  Linux x86_64  0.400  0.400
2 tune        torch-cpu  runnervmmprz5 -> 964408b0e5ea  Linux x86_64  -      -
3 eval:tuned  torch-cpu  964408b0e5ea                   Linux x86_64  1.000  1.000
```

A relay passes when the score is within the measured noise of the reference: on this tiny build (20 exam rows) six runs spread 0.10, so the tolerance is 0.15, and no tighter agreement is claimed.
