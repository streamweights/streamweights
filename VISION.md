# Vision

## The product

streamweights (command: `spill`) is a developer tool for building your own model for your own task, learning from large models, on hardware you have.

You bring examples of your task: an input and the output you want. The tool learns from a large model where that helps, trains a small model you own, measures it honestly against simpler approaches on rows it never saw, and exports it.

## Pillars

1. **One kind of project.** Examples of input and desired output. Labeling, extraction and open-ended answers are the same workflow. Scoring is chosen from the outputs and stated, never assumed.
2. **Large models are first-class.** A teacher or a judge may be bigger than the machine's memory. Weights stream from disk, so a 70B model is a usable teacher and judge on a laptop, slowly. Both are optional, and neither is ever downloaded unless the user names it or accepts a stated recommendation.
3. **Honest evaluation.** The eval set is the central artifact. Splits, leakage rules, the reserved final test and protocol fingerprints stay. A judged score is one model's assessment and is labeled that way, never presented as accuracy.
4. **Portable pieces.** A job is a sequence of small, checkpointed quanta (an optimizer step, a completed row). Any machine with a supported engine can continue the next one from a portable checkpoint. The project folder is the unit; execution is separable from it.
5. **Data stays yours.** Local by default; remote storage only when explicitly configured.

## Direction

Deadline-driven execution: say when you need the result, and the work runs on whatever capacity meets that deadline. It starts with the developer's own machines and accounts. Planning against a deadline comes first, using calibrated rates and labeling each estimate as measured or assumed. Remote execution and scheduling come only after that, and never silently drop quality to meet a deadline.
