"""Verify the torch-cuda engine on a CUDA machine. The script is part of the package, so the same
thing runs anywhere spill is installed:

    docker run --gpus all ghcr.io/streamweights/spill:cuda python -m streamweights.verify_cuda
    python scripts/verify_cuda.py            # from a checkout

It runs the identity gates (torch-cuda against torch-cpu and resident references), the resume gate
from the committed MLX checkpoint, a streamed-inference throughput measurement and a streamed-tune
step-time measurement on qwen2.5:0.5b, prints pass or fail per gate and writes a JSON file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamweights.verify_cuda import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
