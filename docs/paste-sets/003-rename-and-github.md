# Paste set 003 — rename and publish to GitHub

Received 2026-10-01. Saved verbatim.

---

TERMINAL DIRECTIVE: RENAME AND PUBLISH TO GITHUB. Run to the terminal state in item 10. Record progress in state memory. The only user interaction permitted is item 4's one-time code; otherwise never end a turn with a question. If something blocks you, make the most reasonable choice, log it under Decisions, and continue.

Names, final: GitHub org streamweights, repo streamweights/streamweights, PyPI distribution and Python package streamweights, CLI command spill. The working name "Spillway" is retired everywhere.

Save this paste set verbatim as docs/paste-sets/003-rename-and-github.md. Save Appendix A below as docs/plan.md. Commit.
Rename. Python package directory spillway/ becomes streamweights/; all imports updated. pyproject.toml: name streamweights, console script spill = "streamweights.cli:app". Every CLI usage string, pre-run line, next-command hint, log prefix, and error message that said spillway or Spillway now says spill; file and job directory names that contained the old name are renamed. CLAUDE.md, both earlier paste sets' saved copies (leave their content as history but add a one-line note at the top of each: "Names changed in 003: Spillway → streamweights, CLI → spill"), and both reports: update prose references so a reader is never confused. Port 11435 stays. Run the full test suite and the 0.5b end-to-end; both green before continuing. Commit.
Repo hygiene before anything leaves the machine. .gitignore excludes models/, jobs/, state/*.json except state/hardware.json and state/calibration.json, bin/, and anything over 10 MB. Scan the full git history for any blob over 10 MB; if any exist, rewrite history to remove them and say exactly which. Confirm no tokens, keys, or home-directory paths with a personal username are in the history or in state/hardware.json. Add LICENSE (Apache-2.0, copyright "streamweights contributors"). Write README.md containing only: the one-sentence purpose; the golden path from docs/plan.md Section 2 with the pre-run line updated to this machine's real numbers from the Phase 0 report; one paragraph "Status: Phase 0 complete, Phase 1 (streaming runner) in progress" with the Phase 0 headline numbers (mmap at 11 to 13% of NVMe rate, 175 s per pass on 70B Q8_0, OOM from batch 32); and a one-line pointer to docs/plan.md. No badges, no marketing, no feature list. Commit.
GitHub auth. gh auth status. If the authenticated account is amrishkapoor-tt or any account that is not a member of the streamweights org, run gh auth login --hostname github.com --web --scopes repo,read:org,workflow and print the one-time device code and URL clearly; wait for completion. This is the only point where you may pause for the user.
Verify the org. gh api user/orgs must list streamweights. If it does not, print the authenticated username and the org list with a one-line instruction, then stop.
Create and push. gh repo create streamweights/streamweights --public --source . --remote origin --push --description "Run the unmodified full-size model against your eval set, on your machine, for free, overnight." Default branch main. Verify with gh repo view streamweights/streamweights.
Repo settings via gh api: wiki off, projects off, issues on, squash merge only, delete branch on merge. Topics: llm, inference, evaluation, local-first, mlx, llama-cpp, batch-inference.
Tracked measurements. Confirm docs/reports/, docs/plan.md, docs/paste-sets/, examples/evals-2000.jsonl, state/hardware.json, and state/measurements-*.json are tracked and pushed.
Local pointer. docs/REPO.md with: repo URL, org, default branch, package name, CLI name, and the statement that the PyPI name streamweights is reserved by intent but nothing is published yet. Check whether the PyPI names streamweights and spill are currently unregistered (query pypi.org JSON endpoints; do not register anything) and record the answer.
Terminal state. Print: repo URL, authenticated account, default branch, pushed repo size, blobs removed from history (if any), PyPI availability for both names, test suite result after the rename, and a Decisions list. Stop.

Appendix A: docs/plan.md

Copy everything between the markers into docs/plan.md, exactly.

===BEGIN PLAN===

streamweights: technical plan (v2)

Repo and package: streamweights. Command: spill. Local by default; spills to the cloud only when the latency budget demands it.

1. The wedge

Developers cannot test against models bigger than their machine without renting GPUs, and the moment they rent, the whole dev loop changes. Ollama answers "what fits." Ollama Cloud answers "chat with a bigger one on our servers." Neither answers:

Run the unmodified full-size model against my eval set, on my machine, for free, overnight, with no data leaving the building.

That is streamweights. Slow per prompt, exact, free, local. Batching makes it practical: a model streamed from disk pays one full weight read per forward pass whether the batch holds 1 prompt or 64, so throughput scales with batch size until the GPU is busy. It is a correctness tier, not a chat tier, and it is the thing an interactive-chat product is not shaped to offer.

Everything else in this plan (interactive routing, cloud burst, adapters) is a completeness feature that makes the wedge easier to live with. None of it is the reason to use streamweights.

2. Pillar: developer experience

Non-negotiable. The wedge is only worth building if a developer gets a result from it in the first ten minutes without reading docs. Every design choice is tested against the following, in order:

One install, zero config. pip install streamweights, then spill. No YAML before the first result. Hardware is probed, never declared.
No new concepts on the way in. Input is the OpenAI batch JSONL developers already have. Endpoints are the Ollama and OpenAI API shapes they already call. Existing clients work by changing a base URL.
First result in minutes, full result overnight. A batch job returns the first completed rows while the rest run. Progress, tokens per second, and an ETA from the first minute; tail results as they land.
Never silently slow, never silently expensive. Before a job starts, spill states which model, which tier, which quant, estimated wall time, estimated cost (zero for local), and why. One line, then it runs.
Interruptible and resumable by default. Close the laptop, come back, spill resume. No lost work.
Models by name, quants by policy. llama3.3:70b means the full bf16 by default on the batch tier. A quant is an explicit opt-in and is always stated in output metadata.
The CLI tells you the next thing. Every command ends with the one command most likely to come next. No dashboards in v1.

Golden path:

pip install streamweights
spill run llama3.3:70b evals.jsonl
# spill: 70B bf16 (141 GB) does not fit in 48 GB RAM; streaming from NVMe at ~5 GB/s.
# 2,000 prompts, batch 128, est. 14 h. Cost: $0. Results -> evals.out.jsonl (tail with: spill tail)

That interaction is the product. If a change makes it longer, the change is wrong.

3. Tiers
Tier    Where    Engine    Speed    Cost    Role
T2 local-streamed    Developer machine, NVMe    MLX streaming runner (macOS); llama.cpp or custom runner elsewhere    Seconds to a minute per forward pass, batched    Free    The wedge. Eval sets, regression suites, dataset generation; exact weights
T0 local-resident    Developer machine    mlx-lm (macOS), llama.cpp elsewhere    Interactive    Free    Models that fit; interactive dev
T1 burst    Developer's own cloud account    vLLM    Interactive    Paid    Escape hatch: latency-bound work on big models, models outside any catalog, privacy cases forbidding a third-party host, adapter training
Training    T1 only    LoRA (Unsloth / PEFT)    n/a    Paid    Adapters; streaming does not help training
4. Components
CLI (spill run | tail | resume | status | models): the primary interface.
Gateway: localhost, port 11435. Ollama shapes and OpenAI shapes including /v1/batches and /v1/files.
Batch job engine: OpenAI-batch-compatible; JSONL in, streamed results JSONL, progress, ETA, per-row checkpoint, resumable. Batch size computed from the memory budget (KV-bound).
Registry: tag to artifacts. bf16 safetensors (default on macOS), GGUF quants, MLX quants, adapters. Resident-bytes estimate per quant and context length.
Router: tier decision from request class, fit check against probed hardware, policy. Every decision logged with its reason; the reason is what the CLI prints.
Engines: upstream and unmodified where possible; streamweights owns the streaming runner only.
Adapters (last): train on T1, evaluate on T2 against the full-precision base, serve on T0 or T1.
5. Phases and gates

Phase 0 (complete): the wedge on mmap. Result: pipeline works end to end; mmap reaches 11 to 13% of NVMe sequential rate and OOMs from batch 32 up. Streaming runner justified by about 8 to 12x.

Phase 1: streaming runner (MLX on macOS). Layer-ordered reads at drive rate into a ring of host buffers; compute layer k while k+1..k+N-1 are in flight; weights never resident, so the memory budget goes to KV cache. Gate: 70%+ of NVMe sequential rate sustained at the auto batch size, at least 3x Phase 0 aggregate tokens/s, identical greedy output to the resident engine, no DX regression.

Phase 2: interactive completeness. T0 for interactive requests that fit; T1 as the escape hatch with cold-start state and cost guard. Gate: golden path unchanged; interactive requests to fitting models behave like Ollama.

Phase 3: adapters. Train on T1, evaluate on T2 against the full base, serve on T0/T1. Gate: JSONL to served adapter with no command outside the spill CLI.

6. Risks
Risk    Mitigation
Developers see "14 hours" and leave    Early rows stream immediately; the estimate is honest; Phase 2 makes the same command useful daily for small models
Ollama adds an offline batch mode    Their product pressure is interactive chat. Ship the wedge and the resume/tail/ETA polish first; adapters evaluated against full-precision bases deepen the gap
bf16 on disk is huge    State it at download time with a disk check; offer 8-bit as the explicit alternative; never substitute silently
Scope creep into a serving framework    Engines stay upstream where possible; streamweights owns CLI, gateway, jobs, registry, router, and the streaming runner
7. Stack

Python 3.12, uv, FastAPI, Typer. MLX and mlx-lm on macOS. llama.cpp binaries downloaded per platform, not vendored, for non-Apple hardware. Modal SDK for T1 (Phase 2).
===END PLAN===
