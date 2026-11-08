#!/bin/bash
#
# Launch a single-GPU mstar server hosting V-JEPA 2 (facebook/vjepa2-vitl-fpc64-256).
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
#               cache_dir='/m-coriander/coriander/$USER/mstar_cache/vjepa2/')"
#
#   * To swap to a bigger checkpoint (vith / vitg / vitg-384), edit
#     HF_MODELS["vjepa2"] in mstar/model/registry.py — no code change needed.
#   * For the action-conditioned variant, use configs/vjepa2_ac.yaml
#     (AC checkpoint weight loading is still a TODO — current server
#     instantiates with uninitialized predictor weights).

set -euo pipefail

if [ -f "./.env" ]; then
    source ".env"
else
    echo "Error: No .env file found. Run:  \"cp .sample.env .env\" and configure it. Make sure the .env file is in your current working directory."
    exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

if [[ -v VJEPA_CACHE_DIR ]]; then
    echo "Cache dir set to: $VJEPA_CACHE_DIR"
else
    echo "Error: environment variable \"VJEPA_CACHE_DIR\" not found. Please set it in .env!"
    exit 1
fi

mkdir -p "${VJEPA_CACHE_DIR}"

echo "[vjepa2] launching server"
echo "  user:    ${WHO}"
echo "  devices: ${DEVICES}"
echo "  port:    ${PORT}"
echo "  cache:   ${VJEPA_CACHE_DIR}"

CUDA_VISIBLE_DEVICES="${DEVICES}" python mstar/api_server/entrypoint.py \
    --config configs/vjepa2.yaml \
    --port "${PORT}" \
    --cache-dir "${VJEPA_CACHE_DIR}" \
    --socket-path-prefix "/tmp/mstar_${WHO}_${PORT}/" \
    --upload-dir "/tmp/mstar_uploads_${WHO}/" \
    --tensor-comm-protocol $TENSOR_PROTOCOL \
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0}
