# Reproducible eval environment for a VibeSys mstar task (GPU serving).
#
# Deliberately MINIMAL: a CUDA base + uv + build tools. mstar and its deps are
# NOT baked here — the candidate workspace is bind-mounted at run time and the
# server launches via `uv run` (run.sh), which resolves mstar + its deps from
# the candidate's own pyproject/uv.lock into a self-contained uv environment
# (own Python stdlib, so no "No module named encodings" under sandboxing; the
# candidate's torch/CUDA pins are honored). VibeSys injects the coding-agent CLI
# and bind-mounts /model. The CUDA base MUST match the host driver.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv (mstar's expected toolchain). Provides `uv run`, `uv python`.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# Pre-install a uv-managed Python 3.12 so first `uv run` doesn't fetch it.
RUN uv python install 3.12

# uv copies (not hardlinks) into the bind-mounted workspace venv, and a shared
# cache dir survives across candidate containers if the host mounts one.
ENV UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv

WORKDIR /workspace
