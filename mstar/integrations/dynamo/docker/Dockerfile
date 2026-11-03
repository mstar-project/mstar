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
# The default is CUDA 13, which needs an r580+ driver: flash-attn publishes
# its torch 2.10 wheel only for CUDA 13. For an r570 host, override these
# together and supply a flash-attn wheel built from source for cu12 and
# torch 2.10 (docs/installation.rst):
#   --build-arg CUDA_IMAGE=docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
#   --build-arg FLASH_ATTN_WHEEL=<your cu12 torch2.10 wheel>
#   --build-arg MOONCAKE_PACKAGE=mooncake-transfer-engine

# A -devel base is required at runtime, not just at build time: the vendored
# MoE align kernel and FlashInfer both JIT-compile CUDA kernels on first use
# and need nvcc. Fully qualified so rootless podman/buildah builds don't trip
# on short-name resolution.
ARG CUDA_IMAGE=docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04

FROM ${CUDA_IMAGE} AS serve

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
# Keep torch below 2.13 when overriding: newer Inductor buffer allocation
# breaks CUDA-graph capture.
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0
# torchcodec releases pair with torch minors (0.10.x <-> torch 2.10); a newer
# torchcodec against this torch fails at runtime with undefined symbols.
ARG TORCHCODEC_VERSION=0.10.0
# flash-attn ships no PyPI wheels; install the GitHub release wheel whose
# cu1x / torchX.Y / cpXY tags match the torch above (docs/installation.rst).
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# Mooncake publishes one wheel per CUDA major: -cuda13 here, plain
# mooncake-transfer-engine (the extras' default) for CUDA 12.
ARG MOONCAKE_PACKAGE=mooncake-transfer-engine-cuda13

# libibverbs1, ibverbs-providers (libmlx5) and libcurl4: Mooncake's engine
# links them, and the CUDA base image ships none of them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        ibverbs-providers \
        libcurl4 \
        libibverbs1 \
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

# Kokoro's English G2P loads this spaCy tagger and would otherwise download it
# on first use, which an offline deployment cannot. Its wheel declares no
# spaCy requirement, so the install caps spaCy at the minor it was built for.
ARG EN_CORE_WEB_SM_VERSION=3.8.0

# The torch/torchcodec pins ride along so the resolver can't upgrade them
# (breaking the flash-attn wheel match or the torch pairing) to satisfy a
# newer extra; the installed +cu12x/+cu13x builds already satisfy them.
# setuptools arrives with torch from the torch index, which serves 78.1.0
# (CVE-2025-47273, fixed in 78.1.1); the floor pulls a fixed one from PyPI.
RUN uv pip install --python /opt/mstar/venv/bin/python -e ".[all]" \
        "spacy~=${EN_CORE_WEB_SM_VERSION}" \
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-${EN_CORE_WEB_SM_VERSION}/en_core_web_sm-${EN_CORE_WEB_SM_VERSION}-py3-none-any.whl" \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        torchcodec==${TORCHCODEC_VERSION} \
        "setuptools>=78.1.1"

# Qwen3-TTS's speech tokenizer is not in [all]: --no-deps keeps its
# transformers==4.57.3 pin out of the environment (docs/installation.rst).
RUN uv pip install --python /opt/mstar/venv/bin/python --no-deps "qwen-tts==0.1.1"

# Mooncake's default wheel is built for CUDA 12 (it links libcudart.so.12), so
# a CUDA 13 image swaps it for the -cuda13 build of the version the extras
# resolved (#480). --no-deps keeps the resolver away from the pins above. The
# extras still name the CUDA 12 package; `uv pip check` doesn't flag that
# because it skips extra-only requirements. Both wheels ship the same
# `mooncake/` files, so re-running the .[all] install puts the CUDA 12 wheel
# back over them and its uninstall then deletes them: --reinstall rewrites
# them even when the -cuda13 dist-info is still registered.
RUN if [ "${MOONCAKE_PACKAGE}" != mooncake-transfer-engine ]; then \
        v=$(/opt/mstar/venv/bin/python -c "import importlib.metadata as m; print(m.version('mooncake-transfer-engine'))") \
        && uv pip uninstall --python /opt/mstar/venv/bin/python mooncake-transfer-engine \
        && uv pip install --python /opt/mstar/venv/bin/python --reinstall --no-deps "${MOONCAKE_PACKAGE}==${v}"; \
    fi

# The venv must lead PATH: FlashInfer's JIT invokes ninja (installed in the
# venv) via PATH lookup at request time.
ENV PATH=/opt/mstar/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

CMD ["mstar-serve", "--help"]


FROM serve AS dynamo

RUN uv pip install --python /opt/mstar/venv/bin/python -e ".[dynamo]"

ENTRYPOINT ["/opt/mstar/venv/bin/python", "-m", "mstar.integrations.dynamo"]
