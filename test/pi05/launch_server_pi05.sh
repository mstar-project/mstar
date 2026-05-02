#!/bin/bash
#
# Launch a single-GPU mminf server hosting Pi0.5 (lerobot/pi05_base).
#
# Usage:
#   bash test/pi05/launch_server_pi05.sh                     # uses $USER and GPU 0
#   bash test/pi05/launch_server_pi05.sh keisuke 2,3         # custom user + GPU(s)
#
# Required local files:
#   * configs/pi05.yaml         default — base pi0.5 (action_horizon=50)
#   * configs/pi05_droid.yaml   DROID benchmark variant (action_horizon=15)
#
# Override which yaml is used via the PI05_CONFIG env var:
#   PI05_CONFIG=configs/pi05_droid.yaml bash test/pi05/launch_server_pi05.sh
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

# Pick the yaml: default to base pi05.yaml; override with PI05_CONFIG env var
# to swap in a variant (e.g. configs/pi05_droid.yaml for the DROID benchmark).
PI05_CONFIG_PATH="${PI05_CONFIG:-configs/pi05.yaml}"

echo "[pi05] launching server"
echo "  user:    ${WHO}"
echo "  devices: ${DEVICES}"
echo "  port:    ${PORT}"
echo "  cache:   ${PI05_CACHE_DIR}"
echo "  config:  ${PI05_CONFIG_PATH}"

CUDA_VISIBLE_DEVICES="${DEVICES}" python mminf/api_server/entrypoint.py \
    --config "${PI05_CONFIG_PATH}" \
    --port "${PORT}" \
    --cache-dir "${PI05_CACHE_DIR}" \
    --socket-path-prefix "/tmp/mminf_${WHO}/" \
    --upload-dir "/tmp/mminf_uploads_${WHO}/" \
    --port $PORT \
    --tensor-comm-protocol $TENSOR_PROTOCOL \
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0}
