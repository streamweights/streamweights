TERMINAL DIRECTIVE: NEW TAGLINE, README FOR THE CURRENT STATE, REPO CLEANUP, AND SEO.

Rules, absolute:

Never print a status summary or progress report before the terminal state. Printing one ends your turn and stops all work. Until the end, output only tool calls.
Never end the turn to wait. Any wait is a foreground shell loop under the tool timeout. No background tasks, no scheduled wake-ups, no /loop, no questions to the user.
If something blocks you, make the most reasonable choice, log it under Decisions, and continue.
No Claude attribution on any commit. No em-dashes in any file you write.
No model runs of any size. No PyPI publishing, no tags.
Never post publicly on the user's behalf. No issues, pull requests, or comments on other repositories; drafts only.

1. Start.

Save this paste set verbatim as docs/paste-sets/013-tagline-readme-seo.md and commit.
Add standing rules to CLAUDE.md:
"Every directive ends by updating the README and the docs site to reflect the current state, with measured numbers only."
The never-stop rules above.
Commit after every numbered item.

2. Tagline and description everywhere.

Tagline, two lines:
Headline: "A 70B model doesn't fit on your laptop. Build your own model from it anyway."
Subline: "Distill, fine-tune and evaluate on whatever hardware you have. Start a job anywhere, finish it anywhere."
Proof line, directly under the tagline in the README and on the docs home page: "Measured: Llama 3.3 70B, 141 GB unquantized, run on a 48 GB MacBook Pro." Link it to the report that measured it.
Repo description, under 350 characters, concrete terms for search: "Build your own LLM from a 70B model bigger than your RAM: distill, LoRA fine-tune and evaluate locally on Apple silicon (MLX) or Linux (PyTorch). Jobs checkpoint portably and resume on any machine."
Use the headline and subline in:
the README;
the pyproject.toml description (headline only);
the spill --help header (headline only);
the docs site home page;
the org profile.
Update pyproject.toml keywords and classifiers for macOS, Linux, Apple silicon, MLX, PyTorch, LLM, fine-tuning, LoRA, distillation, and evaluation.

3. README, rewritten for the current state. Under 170 lines, no badges, no superlatives, no unmeasured numbers. The first paragraph after the proof line uses the words people search for: fine-tune, LLM, Mac, Linux, LoRA, distill, 70B, bigger than RAM, local. In this order:

Title, tagline (headline and subline), proof line.
Quick start. Install from the GitHub URL. spill example banking77 --quick && spill build banking77-quick, with its real score table and wall time.
Platforms, directly under Quick start. A small table:
rows: Apple silicon (MLX), Linux CPU (PyTorch), NVIDIA (PyTorch);
columns: run/distill/tune/eval/export, build, verified on.
NVIDIA reads "built, awaiting verification", followed by the one-line verify_cuda command and an invitation to report results on issue #1.
State that build currently runs on Apple silicon.
How it works in one picture. flow.svg, with the three lines for exam, homework, and your model.
Copy or surpass. Two short paragraphs.
Which path, and how long. Measured rates only (70B eval pass time, 70B training tokens per night, 7B at the measured TFLOP/s), each stated as a rate, never a total that wasn't run.
Start anywhere, finish anywhere. One short paragraph:
every job is a sequence of steps or rows saved in a portable format;
stop on one machine, resume on another, and the result stays within the noise between two clean runs (the 012 gate numbers, one line);
links to headless mode, the container images, and the SkyPilot, Slurm and Kubernetes examples.
Ship it. spill export to GGUF and Ollama.
What build does. The individual commands, one line each.
Requirements, plus spill doctor.
Under the hood. One paragraph on streaming, one on portability.
Status and roadmap. Done so far; next the verification batch, then PyPI, VLMs, MoE. Links to docs/plan.md and the reports.
Feedback, License.

Every image has descriptive alt text. Every claim links to the report that measured it.

4. Repo cleanup.

Remove or update anything stale: references to Spillway, outdated phase claims, superseded docs, dead links, unused files, empty directories.
Make these match the code and each other: docs/formats.md, docs/cli.md (regenerated from real --help output), docs/models.md, docs/linux.md, docs/portability.md (including checkpoint sizes: about 100 MB for a rank-16 adapter on the 0.5B and about 2.5 GB on the 70B), docs/schedulers.md, docs/plan.md.
Add a link checker for internal links and anchors to the docs test.
The test suite stays green and CI stays green on macOS and Linux.

5. Docs site for search.

Build a GitHub Pages site with MkDocs Material from docs/, deployed by a GitHub Actions workflow on pushes to main. Enable Pages through gh api. Set the site URL as the repo homepage.
Add a sitemap, per-page meta descriptions, canonical URLs, and Open Graph tags.
The home page is the README content.
Add answer-shaped guide pages, each titled as a real search and containing only real commands and measured numbers from the reports:
"Fine-tune an LLM on a Mac"
"Run a 70B model on a 48 GB Mac"
"Distill a large model into a small one locally"
"LoRA fine-tuning without a big GPU"
"Resume a fine-tuning job on a different machine"
"Run fine-tuning on spot instances with SkyPilot"
Each page is short, opens with a two-sentence answer, ends with the command to try, and links to the repo.

6. GitHub surface.

Set the description from item 2 and the homepage to the docs site.
Set 20 topics: llm, fine-tuning, lora, distillation, knowledge-distillation, mlx, apple-silicon, pytorch, local-llm, llm-training, llm-evaluation, llama, qwen, macos, linux, gguf, ollama, skypilot, huggingface, on-device-ai.
Create the org profile: a streamweights/.github repo with profile/README.md containing the headline, subline, proof line, and install line.
Add issue templates:
bug report, asking for spill doctor output;
hardware report, asking for verify_cuda or spill doctor output.
Add a short CONTRIBUTING.md.
Create two or three genuinely small, real tasks from docs/plan.md as issues on this repo, labeled good first issue.

7. Social preview.

Generate docs/img/social-preview.png at 1280×640: minimal, readable at thumbnail size, containing:
the name;
the headline;
the proof line;
the flow diagram.
GitHub has no API to upload it, so record its path and the upload location (Settings, General, Social preview) for the terminal state.

8. Outreach drafts, not posted. In docs/outreach/:

awesome-lists.md: the three to five most relevant awesome lists for MLX, local LLMs and LLM fine-tuning, each with its repo URL, the section the entry belongs in, and a one-line entry in that list's own format.
huggingface-library.md: a draft of the pull request to Hugging Face's library registry (huggingface.js), with the snippet that would show spill on compatible model pages, and the exact file and format it would change.

9. Terminal state.

Commit and push.
Confirm:
CI is green;
the docs site is live, with its URL;
the description, homepage and topics are set;
the org profile renders.
Only now print:
the repo URL and the docs site URL;
the social preview path and where to upload it;
the outreach draft paths;
the issues labeled good first issue;
anything removed in cleanup;
Decisions.
Stop.
