# spill on CPU: python slim plus the CPU build of PyTorch.
#   docker run --rm -v spill-data:/data ghcr.io/streamweights/spill:cpu eval sample qwen2.5:0.5b
# Headless by default (JSON-lines events on stdout, exit 75 when preempted); weights and job
# state live on the /data volume (SPILL_HOME), so a restarted container resumes.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    SPILL_HOME=/data \
    SPILL_HEADLESS=1 \
    HF_HOME=/data/huggingface

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/streamweights
COPY pyproject.toml README.md LICENSE ./
COPY streamweights ./streamweights
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install ".[cloud]"

VOLUME /data
WORKDIR /data
ENTRYPOINT ["spill"]
CMD ["--help"]
