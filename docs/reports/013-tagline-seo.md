---
description: Directive 013 report: the new tagline, the README rewrite, repo cleanup, the docs site and the GitHub surface, with the decisions made along the way.
---

# Tagline, README, cleanup and docs site (directive 013)

No model was run. Nothing here is a measurement; every number in the README and the guides still comes from reports 001 to 012.

## What changed

- Tagline (headline and subline) and the proof line, linked to [report 002](002-phase1.md), in the README, the docs home page, the org profile, the `pyproject.toml` description (headline only) and `spill --help` (headline only). `docs/cli.md` was regenerated from the real help output.
- README rewritten under 170 lines, in the order the directive set.
- Docs site: MkDocs Material from `docs/`, built and deployed by `.github/workflows/docs.yml` on every push to main. Sitemap, per-page descriptions, canonical URLs and Open Graph tags. Six guides.
- Tests: an anchor-aware link checker over every markdown page, the number check extended to the guides, a check that the home page renders with working links, and a check that the guides are answer-shaped.
- GitHub: description, homepage and 20 topics; issue templates for bug and hardware reports; CONTRIBUTING.md; three good-first issues; the `streamweights/.github` profile repo.
- Social preview drawn at `docs/img/social-preview.png` by `scripts/make_social_preview.py`; GitHub has no API to upload it.

## Decisions

- The home page is `docs/index.md`, generated from the README at build time by `scripts/make_site_home.py` and not committed, so the two cannot drift. Links outside `docs/` are rewritten to GitHub.
- The Hugging Face draft targets `local-apps.ts`, not `model-libraries.ts`: spill publishes no models with its own `library_name`.
- Checkpoint sizes: about 100 MB for the 0.5B is measured (101 MB); about 2.5 GB for the 70B is computed from the 207.1M adapter parameters in [report 008](008-phase3.md) (float32 parameters plus two optimizer moments), and the docs say so.
- The README's "Linux CPU" row says verified on the 0.5B gates and the Linux CI suite, and "NVIDIA" says awaiting verification; build is stated as Apple silicon only.
- Awesome-list entries were drafted for four lists; one candidate was dropped because its list is generated, not edited.
