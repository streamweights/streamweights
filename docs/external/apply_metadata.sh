#!/usr/bin/env bash
# Applies the parts of the new positioning that live outside this repository. Run after the pull
# request merges (it is not part of the build):
#   - the GitHub About description of streamweights/streamweights
#   - the org profile README (streamweights/.github, profile/README.md), from docs/external/org-profile-README.md
#   - the org description
# Not automatable: the social preview image (Settings, General, Social preview): upload docs/img/social-preview.png.
set -euo pipefail
cd "$(dirname "$0")/../.."
HEAD="Build a small model for your task, on hardware you control."
SUP="Bring labeled examples. Fine-tune locally, compare against simple baselines, and export to GGUF or safetensors. Pause and resume supported training runs across Mac and Linux."

gh api -X PATCH repos/streamweights/streamweights -f description="$HEAD $SUP" -f homepage="https://streamweights.github.io/streamweights/"

SHA=$(gh api repos/streamweights/.github/contents/profile/README.md -q .sha)
gh api -X PUT repos/streamweights/.github/contents/profile/README.md \
  -f message="Profile: new headline and supporting copy" -f sha="$SHA" \
  -f content="$(base64 < docs/external/org-profile-README.md | tr -d '\n')"

gh api -X PATCH orgs/streamweights -f description="$HEAD"
