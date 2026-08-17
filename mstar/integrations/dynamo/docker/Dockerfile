# M* serving image with the Dynamo backend layered on top.
#
# Build from the repository root:
#
#   docker build -f mstar/integrations/dynamo/docker/Dockerfile -t mstar-dynamo .
#
# The intermediate `serve` stage is a plain M* serving image with no Dynamo
# dependency; build it alone with `--target serve -t mstar-serve`. The final
# stage adds the mstar[dynamo] extra, so the ai-dynamo version pin lives in
# pyproject.toml only.
#
# For a CUDA 13 host, override the three build args together:
#   --build-arg CUDA_IMAGE=docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130
#   --build-arg FLASH_ATTN_WHEEL=...cu13torch2.9...  (see docs/installation.rst)

# A -devel base is required at runtime, not just at build time: the vendored
# MoE align kernel and FlashInfer both JIT-compile CUDA kernels on first use
# and need nvcc. Fully qualified so rootless podman/buildah builds don't trip
# on short-name resolution.
ARG CUDA_IMAGE=docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04

FROM ${CUDA_IMAGE} AS serve

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG TORCHAUDIO_VERSION=2.9.1
# torchcodec releases pair with torch minors (0.9.x <-> torch 2.9); a newer
# torchcodec against this torch fails at runtime with undefined symbols.
ARG TORCHCODEC_VERSION=0.9.1
# flash-attn ships no PyPI wheels; install the GitHub release wheel whose
# cu1x / torchX.Y / cpXY tags match the torch above (docs/installation.rst).
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# uv provisions its own CPython so the image doesn't depend on the base's
# python. Torch is pinned from the explicit CUDA index because the build
# machine has no GPU for --torch-backend=auto to probe.
RUN uv venv --python 3.12 --seed /opt/mstar/venv \
    && uv pip install --python /opt/mstar/venv/bin/python \
        --index-url ${TORCH_INDEX} \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
    && uv pip install --python /opt/mstar/venv/bin/python ${FLASH_ATTN_WHEEL}

WORKDIR /opt/mstar/src
COPY . .

# The torch/torchcodec pins ride along so the resolver can't upgrade them
# (breaking the flash-attn wheel match or the torch pairing) to satisfy a
# newer extra; the installed +cu12x/+cu13x builds already satisfy them.
RUN uv pip install --python /opt/mstar/venv/bin/python -e ".[all]" \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        torchcodec==${TORCHCODEC_VERSION}

# The venv must lead PATH: FlashInfer's JIT invokes ninja (installed in the
# venv) via PATH lookup at request time.
ENV PATH=/opt/mstar/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

CMD ["mstar-serve", "--help"]


FROM serve AS dynamo

RUN uv pip install --python /opt/mstar/venv/bin/python -e ".[dynamo]"

ENTRYPOINT ["/opt/mstar/venv/bin/python", "-m", "mstar.integrations.dynamo"]
