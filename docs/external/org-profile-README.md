# streamweights

**Build a small model for your task, on hardware you control.**

Bring labeled examples. Fine-tune locally, compare against simple baselines, and export to GGUF or safetensors. Pause and resume supported training runs across Mac and Linux.

```
pip install git+https://github.com/streamweights/streamweights
spill example banking77 --tiny
spill init banking77-tiny/banking77.csv --input text --output label --project tickets
spill build tickets
```

- Repo: [streamweights/streamweights](https://github.com/streamweights/streamweights)
- Docs and guides: [streamweights.github.io/streamweights](https://streamweights.github.io/streamweights/)
- Optional: a teacher bigger than RAM, streamed from disk ([measured on a 48 GB MacBook Pro](https://streamweights.github.io/streamweights/reports/002-phase1/))
