---
description: Directive 020 measured. Any task as one kind of project, a 1.5B teacher and judge on MLX and torch-cpu, judged evaluation of open-ended output with its calibration and its failures, logit distillation, deadline planning.
---

# 020 Restore the vision

Directive 020 puts back what the front door had narrowed: any input-to-output task as one kind of project, a large teacher that is optional and named by you, and honest evaluation of open-ended output. Everything below ran on the Apple M4 Pro with 48 GB (MLX on the Metal GPU, and torch-cpu on the same machine's CPU), with `qwen2.5:0.5b` as the student and `qwen2.5:1.5b` as teacher and judge. No 7B or 70B model ran. The captures are in `docs/reports/data/020-*.json`; `scripts/capture_020.py` produces them and `scripts/check_020_capture.py` checks each against its gate, and the same script runs in CI on Linux (torch-cpu).

## Gates

| gate | result | evidence |
|---|---|---|
| Any task, end to end, MLX (Metal GPU) | pass: init, plan, build (1.5B teacher, 0.5B student, 1.5B judge, both judged modes), report, export, inference | `020-judged-mlx.json` |
| Any task, end to end, torch-cpu | pass: the same sequence | `020-judged-torch-cpu.json` |
| Labeled project, no teacher | pass: four stages, no teacher stage, only the student and the embedding baseline allowed through `SPILL_ALLOWED_MODELS` | `020-labeled-mlx.json`, `tests/test_teacher_optional.py` |
| Open-ended labeled project, no judge | pass: reference-overlap metrics, the plan shows the judge recommendation, only the student allowed through | `020-nojudge-mlx.json`, `tests/test_open_build.py` |
| Scoring selection | pass: exact, json and judge chosen on three fixtures, the `script:` override, a legacy project keeps its method and compares | `tests/test_scoring.py` |
| Judge integrity | pass: judge identity, rubric, prompt templates and decoding change the protocol and the run; both orders run and their disagreement is reported; the self-judged row is reference only; calibration appears when applicable | `tests/test_judge.py`, `tests/test_open_build.py` |
| Logit distillation | pass: loss and gradient against a float64 reference (and its finite differences), MLX against PyTorch, PEFT against streamed, MLX resident against MLX streamed; the ce and mix students both reported | `tests/test_kd.py`, `tests/test_kd_mlx.py`, `020-kd-mlx.json` |
| Model choices | pass: tags and Hugging Face repos in all three roles, recorded and shown in plan; an unsupported architecture refused in one line naming the family | `tests/test_model_choices.py` |
| Deadlines | pass: feasible combinations with labeled estimates, nothing downloaded, an honest refusal when nothing fits, a build that never starts or lowers anything | `020-deadline-mlx.json`, `tests/test_deadline.py` |
| Nothing regresses | the suites and workflows named in the pull request | the pull request |

## The open-ended example

`spill example dolly-qa --tiny` is 84 rows: a question about a short Wikipedia passage and the answer a person wrote (databricks-dolly-15k, closed_qa category, CC BY-SA 3.0; the license was read from the dataset card on 2026-10-13 and is recorded in the example's `NOTICE.txt`). `spill init` chose `judge` scoring from the outputs (84 distinct values in 84 rows) and split them 58 train, 13 validation and 13 final test. Teacher and judge were both `qwen2.5:1.5b` (the judge defaulted to the teacher, and the teacher's own row is therefore reference only); the student trained on 174 rows, 58 of them its own labels twice over and 58 the teacher's answers with the teacher's top-20 log-probabilities (`mix`).

### Reference overlap and judge, MLX (Metal)

| comparator | token F1 | ROUGE-L | exact match | judge score (1 to 5) |
|---|---|---|---|---|
| nearest training answer (TF-IDF, no model) | 0.108 | 0.098 | 0.000 | 2.462 |
| student, prompted, untrained | 0.441 | 0.419 | 0.077 | 4.846 |
| student, trained | 0.590 | 0.573 | 0.154 | 4.538 |
| teacher | 0.459 | 0.426 | 0.077 | reference only (it is the judge) |

13 validation rows. Token F1, ROUGE-L and exact match are overlap with the reference, not accuracy; the judge score is one model's assessment, not accuracy.

Pairwise, both orders, 13 rows: the trained student against the untrained one won 0, tied 11 and lost 2, and the two orders disagreed on 8 rows (62%); against the nearest-answer baseline it won 8, tied 4 and lost 0 of the 12 rows the judge scored, with 4 disagreements (33%). Calibration: the judge agreed with the word-for-word check on 80% of the 15 rows whose reference has at most 5 words (3 judge-only positives, no match-only ones).

### The same project on torch-cpu

| comparator | token F1 | ROUGE-L | exact match | judge score (1 to 5) |
|---|---|---|---|---|
| nearest training answer (TF-IDF, no model) | 0.108 | 0.098 | 0.000 | 2.462 |
| student, prompted, untrained | 0.451 | 0.430 | 0.077 | 4.846 |
| student, trained | 0.531 | 0.506 | 0.154 | 4.308 |
| teacher | 0.457 | 0.424 | 0.077 | reference only |

The two engines train different adapters (the final training loss was 0.2399 on MLX and 0.2421 on torch-cpu; the training numerics differ), so the trained rows differ; the untrained and baseline judge scores are identical. Pairwise on torch-cpu: trained against untrained won 0, tied 10, lost 3 of 13 with 7 disagreements (54%); against the baseline won 8, tied 4, lost 0 of 12 with 4 (33%); calibration again 80% of 15.

### What these numbers do and do not say

* The student the judge prefers is the untrained one (4.846 against 4.538 on MLX and 4.308 on torch-cpu), while the reference overlap says the trained student is closer to the reference (0.590 against 0.441 on MLX, 0.531 against 0.451 on torch-cpu). The two measures disagree, the sets are 13 rows, and the tool states no significance. The honest reading is that a student trained on 58 examples for 88 steps is not shown here to be better by a judge, and is closer to the reference wording.
* A 1.5B judge is a weak instrument. In an earlier template, the same judge gave a 5 to a retrieved answer about the wrong city's population by grading the reference instead of the answer; calibration on the same 15 rows was 60%. The prompt templates were then rewritten once so the reference is labeled as a guide and the answer to grade is stated, which lowered the nearest-answer baseline from 3.923 to 2.462 and raised calibration to 80%. That rewrite was made after looking at this project's validation rows, so these numbers are not a held-out measurement of the final templates; they are what the shipped templates produce on this data.
* The judge disagreed with itself across the two orders on a third to nearly two thirds of the rows it compared, which is why a disagreement counts as a tie and is reported next to every pairwise result. First-position picks were 62% and 64% on MLX.
* The exported model loaded in Transformers and ran: 5 of 8 predictions differ from the training engine's (2 of 8 on torch-cpu). Token F1 on those 8 rows went from 0.596 to 0.487 on MLX and from 0.497 to 0.393 on torch-cpu, the cost of merging a bf16 adapter and running float32. "Verified" means it loaded and ran, not that quality was kept.

### Time

| stage | MLX (Metal) | torch-cpu |
|---|---|---|
| teacher answers 58 prompts with top-20 log-probs | 14 s | 56 s |
| train, 88 steps | 35 s | 2 min |
| student untrained, trained, teacher on 13 rows | 4 s, 2 s, 4 s | 9 s, 5 s, 21 s |
| judge, 91 requests | 30 s | 2 min |
| `spill build` as a whole | 90.7 s | 305.8 s |
| `spill export` | 14.1 s | 11.6 s |

The plan estimated 2 min for the MLX build and 4 min for torch-cpu, each stage labeled an assumption except training (a measurement from an earlier run). The judge stage was estimated at 12 s and took 30 s on MLX, and at 51 s against 2 min on torch-cpu: the cost model under-counts the judge's long prompts.

## The README capture

`docs/reports/data/020-readme-capture.json` is the README's "Try it" run again on the Apple GPU with this code (`scripts/capture_readme.py`, `SPILL_MIN_FREE_GB=2`): the tiny banking77 build took 13 s (13.4 s) on MLX. Its plan printed `total: 12 s plus unknown (baseline: no measurement on this machine yet)`, because the embedding baseline had no timing on this machine yet; its build printed `estimate 11 s (measurement)` for tuning, which took 8 s, `stage 1 done in 2 s`, `stage 3 done in 1 s` and `stage 4 done in 1 s`. The scores are the same as in [report 018](018-readme-pass.md): baseline 1.000, prompted untrained 0.333, trained 0.889 on 18 validation rows.

## Without a teacher, without a judge

`020-labeled-mlx.json`: `spill build` on the tiny banking77 project took 13.7 s with four stages (the embedding baseline, tuning, the untrained and the trained student), with `SPILL_ALLOWED_MODELS` listing only `qwen2.5:0.5b` and the embedding model, so any other model would have been refused before it was fetched or loaded. `020-nojudge-mlx.json`: the same for the open-ended project, 19.5 s, with token F1 against the reference (the trained student 0.591, the untrained one 0.441, the nearest-answer baseline 0.105) and a plan that prints `judge: none named. Judged scores need a judge; recommended: qwen2.5:7b (...)` with its 15.23 GB download and an estimated 57 s of judging, labeled an assumption, and downloads nothing.

## Logit distillation: ce and mix

The same teacher and student and the same 58 training prompts, trained twice (`020-kd-mlx.json`): `--distill-loss ce` (the teacher's tokens as text) and `mix` (the teacher's top-20 distribution at temperature 1, half weight). Token F1 against the reference on the 13 validation rows: ce 0.562, mix 0.590; the final training loss was 0.2067 and 0.2399 (the two are different losses and are not comparable). `spill compare` ranked them as a common-evaluator comparison and listed `training.distill_loss` as the only difference. Thirteen rows move a score by several points; this is an observed difference, not a finding about mix.

The loss itself is checked, not only the runs: `tests/test_kd.py` compares the PyTorch loss and its gradient with a float64 numpy reference to 2e-6, and that reference's gradient with central finite differences to 1e-7; `tests/test_kd_mlx.py` compares MLX with the reference and with PyTorch to the same tolerance, the MLX streamed trainer with the PyTorch streamed one on one batch, and MLX's resident tuner (mlx-lm) with its streamed trainer over a six-step job (loss within 0.1% per step, adapter cosines above 0.999), for `ce`, `mix` and `kl`.

## Deadline planning

`spill plan dolly --deadline 30m` on this machine weighed 48 combinations of student (0.5B, 1.5B, 7B), teacher, judge and engine and found 30 estimated to fit; it recommended the 7B student with the 1.5B teacher and judge on MLX (8 min, assumed, with the 15.2 GB download of `qwen2.5:7b` shown and not fetched), and said the project as configured (2 min) fits. With `--deadline 5s` it said nothing fits, named the closest (the 0.5B student alone, 26 s, 21 s over) and added that nothing was lowered. `spill build dolly --deadline 5s` did not start. Every estimate here is the cost model with its stated rate, labeled assumed, except training on the 0.5B; download time is not counted.

## Decisions

* **Task types.** The internal task type stays `classification`, `json` and a new `open`; the scoring method (`exact`, `json`, `judge`, `script`) is recorded beside it. Legacy configs have no scoring key and map from the task type, and nothing in the protocol fingerprint changes for them (checked: a legacy-shaped and a new project on the same rows compare as a common evaluator). `--task` remains as a hidden alias of `--score exact|json`.
* **Automatic choice.** `exact` needs at most 200 distinct outputs, each short, with no sentence punctuation, and 80% of rows sharing their label with another row (the earlier rule, with the cap). Everything else that is not all JSON objects is `judge`. A project of 300 unique short answers is therefore judge-scored.
* **Open-ended baseline.** The embedding baseline needs labels. The open-ended baseline is the training answer whose input is closest by TF-IDF; it uses no model and downloads nothing.
* **Judge default.** A named teacher judges unless a judge is named or `--judge none`; the teacher's own row is then reference only. The recommendation is the largest curated model other than the student whose bf16 weights fit in 60% of memory and never below 1.5B; on this machine that is `qwen2.5:7b`, whose recommendation shows its size and is never fetched.
* **Judge parsing.** A reply with no parseable score is unscored and counted, never a number. Pairwise: the same verdict twice is the verdict, anything else is a tie and an order disagreement.
* **Judge templates.** Rewritten once after the first run (above); the template version stays 1 because nothing was released.
* **KD.** The teacher's "other" bucket and the student's are both floored at 1e-6; the KL is multiplied by T squared; `auto` is `mix` when the two models share a `tokenizer.json`. Teacher rows keep the teacher's own tokens (they are not re-tokenized); your labels keep text tokenization. A teacher answer that the contract changes (a label in another case) trains as text without log-probs.
* **Unlabeled prompts.** Added as `spill init --unlabeled`, with the leakage rule: a prompt matching any training, validation or test input is removed at init. They require a teacher.
* **Default student.** Stays `qwen2.5:0.5b` on every machine, with the reason printed per machine; a bigger default would change every first result and the README's capture. `--student` is the choice.
* **`build --deadline`.** Builds only the configured combination and only if it fits; if it does not, it does not start, lists what fits and prints the command for the recommended combination. It does not pick one for you.
* **Scoring override at build.** `--score` is an `init` option only: changing how outputs are scored changes the contract and the split-time checks, so it is a new project (`spill init ... --force`).
* **Pinned model.** `qwen2.5:1.5b` was added to `models.yaml` with the revision `989aa7980e4c`, its file hashes and size from the Hub (no weights were needed to pin it); its tokenizer files hash identically to `qwen2.5:0.5b`.
* **README test cap.** `tests/test_docs.py` capped README prose at half the pre-018 length plus ten words (681); directive 020 adds a section and an example, so the cap is now 1,100 words.
