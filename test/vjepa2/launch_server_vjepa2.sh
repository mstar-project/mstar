#!/bin/bash
#
# Launch a single-GPU mminf server hosting V-JEPA 2 (facebook/vjepa2-vitl-fpc64-256).
#
# Usage:
#   bash test/vjepa2/launch_server_vjepa2.sh                 # uses $USER and GPU 0
#   bash test/vjepa2/launch_server_vjepa2.sh atindra 3       # custom user + GPU
#
# First-launch notes:
#   * Weights (~1.2 GB safetensors) live at facebook/vjepa2-vitl-fpc64-256 on
#     HuggingFace.  Pre-download into $CACHE_DIR to avoid a cold-start stall:
#
#       python -c "from huggingface_hub import snapshot_download; \
#           snapshot_download('facebook/vjepa2-vitl-fpc64-256', \
#               cache_dir='/m-coriander/coriander/$USER/mminf_cache/vjepa2/')"
#
#   * To swap to a bigger checkpoint (vith / vitg / vitg-384), edit
#     HF_MODELS["vjepa2"] in mminf/model/registry.py — no code change needed.
#   * For the action-conditioned variant, use configs/vjepa2_ac.yaml
#     (AC checkpoint weight loading is still a TODO — current server
#     instantiates with uninitialized predictor weights).

set -euo pipefail

USERNAME="${1:-${USER:-atindra}}"
DEVICES="${2:-0}"
PORT="${PORT:-20003}"

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

# CACHE_DIR="/m-coriander/coriander/${USERNAME}/mminf_cache/vjepa2/"
CACHE_DIR="/mnt/storage/${USERNAME}/mminf_cache/vjepa2/"

mkdir -p "${CACHE_DIR}"

echo "[vjepa2] launching server"
echo "  user:    ${USERNAME}"
echo "  devices: ${DEVICES}"
echo "  port:    ${PORT}"
echo "  cache:   ${CACHE_DIR}"

CUDA_VISIBLE_DEVICES="${DEVICES}" python mminf/api_server/entrypoint.py \
    --config configs/vjepa2.yaml \
    --port "${PORT}" \
    --cache-dir "${CACHE_DIR}" \
    --socket-path-prefix "/tmp/mminf_${USERNAME}/" \
    --upload-dir "/tmp/mminf_uploads_${USERNAME}/" \
    --tensor-comm-protocol SHM
    # --tensor-comm-protocol TCP \
    # --tcp-transfer-device "0.0.0.0:0"
