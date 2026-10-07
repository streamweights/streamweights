# spill on NVIDIA GPUs: the official PyTorch CUDA runtime image plus streamweights.
#   docker run --gpus all --rm -v spill-data:/data ghcr.io/streamweights/spill:cuda tune ...
#   docker run --gpus all ghcr.io/streamweights/spill:cuda python -m streamweights.verify_cuda
# Headless by default (JSON-lines events on stdout, exit 75 when preempted); weights and job
# state live on the /data volume (SPILL_HOME), so a restarted container resumes.
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    SPILL_HOME=/data \
    SPILL_HEADLESS=1 \
    HF_HOME=/data/huggingface

WORKDIR /opt/streamweights
COPY pyproject.toml README.md LICENSE ./
COPY streamweights ./streamweights
RUN pip install ".[cloud]"

VOLUME /data
WORKDIR /data
ENTRYPOINT ["spill"]
CMD ["--help"]
