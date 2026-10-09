## Merge hygiene

The pull request for this directive contains an empty verification commit for `8c4b7fa` (the commit that fixed the CPU image build by leaving `boto3` unpinned in the `cloud` extra).

Resolution, from the CI job `boto3 and s3fs resolution` on the pull request: a clean Python 3.12 venv with `pip install ".[cloud]"`, the CPU image and the CUDA image all resolved the same versions:

| package | version |
|---|---|
| boto3 | 1.43.106 |
| botocore | 1.43.106 |
| aiobotocore | 3.9.2 |
| s3fs | 2026.9.0 |

Branch protection on `main`, set through `gh api` (the account permits it): pull requests are required (0 required approvals, since there is one maintainer), and these status checks must pass before merging: `ubuntu-latest / Python 3.10`, `ubuntu-latest / Python 3.12`, `macos-14 / Python 3.12`, `S3 ownership and handoff against a pinned MinIO (Linux)`, `Offline bundle gate (container with --network none)` (workflow CI); `Check the relays against the reference` (Relay); `cpu image (build, smoke test, push)` and `cuda image (build, push)` (Containers); `build` (Docs site). Administrators are not forced to follow the rules (`enforce_admins` is off), so the owner can still override in an emergency. To make Containers and Docs site checks exist on pull requests, both workflows now also run on `pull_request` (they push images and deploy only from `main`).
