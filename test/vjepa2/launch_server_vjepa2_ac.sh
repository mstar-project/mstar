#!/bin/bash
#
# Launch a single-GPU mstar server hosting V-JEPA 2-AC
# (facebook/vjepa2-ac-vitg — action-conditioned ViT-g).
#
# Usage:
#   bash test/vjepa2/launch_server_vjepa2_ac.sh                 # uses $USER and GPU 0
#   bash test/vjepa2/launch_server_vjepa2_ac.sh atindra 3       # custom user + GPU
#
# First-launch notes:
#   * The HF V-JEPA 2 collection does NOT host an AC checkpoint (as of
#     2026-04, only base + SSv2 / Diving-48 classification variants are on
#     HF).  Weights come straight from the public S3 mirror at
#     ``https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`` (~11.7 GB,
#     no auth needed).  The server's weight loader wraps
#     ``torch.hub.download_url_to_file`` so download happens automatically on
#     first model.get_submodule() call — but that's a ~12 GB cold-start
#     stall if the file isn't cached.  Pre-download:
#
#       python -c "import torch.hub; torch.hub.download_url_to_file( \
#           'https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt', \
#           '/m-coriander/coriander/$USER/mstar_cache/vjepa2/vjepa2-ac-vitg.pt', \
#           progress=True)"
#
#   * Clients MUST send per-timestep 7-DOF actions + states via ``model_kwargs``
#     when POSTing to ``/generate`` — the AC predictor's graph node lists them as
#     required inputs.  Shapes: both ``[T_action, 7]`` where
#     ``T_action = frames_per_clip / tubelet_size = 64 / 2 = 32`` for this checkpoint.
#     See test/vjepa2/video_request_ac.sh for a canonical request.

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

echo "[vjepa2-ac] launching server"
echo "  user:    ${WHO}"
echo "  devices: ${DEVICES}"
echo "  port:    ${PORT}"
echo "  cache:   ${VJEPA_CACHE_DIR}"

CUDA_VISIBLE_DEVICES="${DEVICES}" python mstar/api_server/entrypoint.py \
    --config configs/vjepa2_ac.yaml \
    --port "${PORT}" \
    --cache-dir "${VJEPA_CACHE_DIR}" \
    --socket-path-prefix "/tmp/mstar_${WHO}_${PORT}/" \
    --upload-dir "/tmp/mstar_uploads_${WHO}/" \
    --tensor-comm-protocol $TENSOR_PROTOCOL \
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0}
