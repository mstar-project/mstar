#!/bin/bash
#
# Launch a single-GPU mminf server hosting Pi0.5 (lerobot/pi05_base).
#
# Usage:
#   bash test/pi05/launch_server_pi05.sh                     # uses $USER and GPU 0
#   bash test/pi05/launch_server_pi05.sh keisuke 2,3         # custom user + GPU(s)
#
# Required local files:
#   * configs/pi05.yaml  (vit_encoder + LLM colocated on rank 0)
#
# Notes:
#   * Pi0.5 weights live at lerobot/pi05_base on HuggingFace as a single
#     ~14 GB safetensors blob. The first launch will download them via
#     huggingface_hub into $CACHE_DIR; subsequent launches reuse the cache.
#   * mminf's Pi05Model loader (mminf/model/pi05/weight_loader.py) handles
#     the lerobot -> mminf state-dict remap automatically inside
#     get_submodule(), so no manual conversion step is needed.
#   * The server listens on TCP $PORT and uses ZMQ over IPC under
#     /tmp/mminf_$WHO/ for the conductor / worker / API hand-off.

set -euo pipefail

if [ -f ".env" ]; then
    source ".env"
else
    echo "Error: No .env file found. Run:  cp .sample.env .env  and configure it. Make sure the .env file is in your current working directory."
    exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

if [[ -v PI05_CACHE_DIR ]]; then
    echo "Cache dir set to: $PI05_CACHE_DIR"
else
    echo "Error: environment variable \"PI05_CACHE_DIR\" not found. Please set it in .env!"
    exit 1
fi

mkdir -p "${PI05_CACHE_DIR}"

echo "[pi05] launching server"
echo "  user:    ${WHO}"
echo "  devices: ${DEVICES}"
echo "  port:    ${PORT}"
echo "  cache:   ${PI05_CACHE_DIR}"

CUDA_VISIBLE_DEVICES="${DEVICES}" python mminf/api_server/entrypoint.py \
    --config configs/pi05.yaml \
    --port "${PORT}" \
    --cache-dir "${PI05_CACHE_DIR}" \
    --socket-path-prefix "/tmp/mminf_${WHO}/" \
    --upload-dir "/tmp/mminf_uploads_${WHO}/" \
    --port $PORT \
    --tensor-comm-protocol $TENSOR_PROTOCOL \
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0}
