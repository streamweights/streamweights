# llama.cpp build in use

Not vendored; prebuilt release downloaded into `bin/` (gitignored). Re-fetch with:

```
curl -sL -o bin/llama.tar.gz https://github.com/ggml-org/llama.cpp/releases/download/b11311/llama-b11311-bin-macos-arm64.tar.gz
tar -xzf bin/llama.tar.gz -C bin && rm bin/llama.tar.gz
ln -sf llama-b11311/llama-server bin/llama-server
```

## Version

- Release tag **b11311** (`version: 0.5.0-dev, build 11311, commit f7b384c1e`), published 2026-10-01 (UTC), macOS arm64, AppleClang 21.
- Nightly pointer release at fetch time was v0.5.0 → b11146; b11311 was the newest tagged binary release.

## Capability findings (this build)

| Capability | Supported | How |
|---|---|---|
| Memory-mapped loading | yes | `-lm, --load-mode MODE` with modes `auto` (default; mmap unless device unsupported), `mmap`, `mmap+mlock`. The old `--no-mmap` flag is replaced by this unified flag. |
| Partial GPU offload | yes | `-ngl, --gpu-layers, --n-gpu-layers N` |
| Parallel slots + continuous batching | yes | `-np, --parallel N` (default -1 = auto), `-cb, --cont-batching` (on by default; `--no-cont-batching` to disable) |
| bf16 GGUF | yes | Metal backend dylib shipped (`libggml-metal`); bf16 GGUF loads verified empirically in item 10 (Metal bf16 path, M4 native bf16). |
