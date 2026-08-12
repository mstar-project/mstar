# Reproducible eval image for a VibeSys mstar task.
#
# Bakes mstar + the model's pip extra so every candidate round shares one
# dependency closure. VibeSys mounts the candidate workspace over /workspace and
# handles GPU passthrough (--docker). The CUDA base MUST match the host driver.
#
# Built by mstar-vibesys with --build-arg MSTAR_EXTRA=<extra>.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG MSTAR_EXTRA=all
ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv (mstar's expected toolchain) provides `uv run`, which the eval commands use.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /opt/mstar
# Copy only dependency metadata first for layer caching; the candidate's live
# source is bind-mounted at run time, so we install mstar in editable-ish mode
# from a shallow copy here just to resolve the environment.
COPY pyproject.toml README.md ./
RUN uv venv --python ${PYTHON_VERSION} --seed /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" VIRTUAL_ENV=/opt/venv
RUN uv pip install --torch-backend=auto -e ".[${MSTAR_EXTRA}]" || \
    echo "WARN: extra '${MSTAR_EXTRA}' failed; fill in the model's deps in pyproject before a real run"

# flash-attn is deliberately not in any extra (no PyPI wheels). If the model
# needs it, add the matching prebuilt wheel install here, e.g.:
# RUN uv pip install https://.../flash_attn-<ver>+cu124torch2.x-cp312-linux_x86_64.whl

# httpx + numpy are needed by the framework-owned checker/benchmark.
RUN uv pip install "httpx>=0.27" "numpy>=1.26"

WORKDIR /workspace
