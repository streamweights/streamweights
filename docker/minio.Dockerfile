# MinIO built from the source of one pinned release tag. The upstream images for this tag are no
# longer published, and the release binaries are no longer served, so the integration job builds
# the server it tests against from the tag, and logs `minio --version` (which carries the
# release and the commit). Used only when `docker pull` of the upstream image fails.
ARG GO_VERSION=1.25
FROM golang:${GO_VERSION} AS build
ARG RELEASE=RELEASE.2025-10-15T17-29-55Z
RUN git clone --depth 1 --branch ${RELEASE} https://github.com/minio/minio /src
WORKDIR /src
RUN go build -trimpath -ldflags "$(go run buildscripts/gen-ldflags.go)" -o /out/minio .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/minio /usr/bin/minio
EXPOSE 9000
ENTRYPOINT ["/usr/bin/minio"]
