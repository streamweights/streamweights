# Runtime dependencies only, for the offline bundle gate: everything streamweights needs is
# installed here, then streamweights itself is removed, so the gate can install the wheel from a
# local file with no index and no network (docker run --network none).
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY streamweights ./streamweights
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install ".[cloud]" build \
 && pip uninstall -y streamweights \
 && rm -rf /src
