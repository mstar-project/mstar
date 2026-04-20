#!/bin/bash
#
# Launch a single-GPU mminf server hosting V-JEPA 2-AC
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
#           '/m-coriander/coriander/$USER/mminf_cache/vjepa2/vjepa2-ac-vitg.pt', \
#           progress=True)"
#
#   * Clients MUST send per-timestep 7-DOF actions + states via ``model_kwargs``
#     when POSTing to ``/generate`` — the AC predictor's graph node lists them as
#     required inputs.  Shapes: both ``[T_action, 7]`` where
#     ``T_action = frames_per_clip / tubelet_size = 64 / 2 = 32`` for this checkpoint.
#     See test/vjepa2/video_request_ac.sh for a canonical request.

set -euo pipefail

USERNAME="${1:-${USER:-atindra}}"
DEVICES="${2:-0}"
PORT="${PORT:-20003}"

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

CACHE_DIR="/m-coriander/coriander/${USERNAME}/mminf_cache/vjepa2/"
mkdir -p "${CACHE_DIR}"

echo "[vjepa2-ac] launching server"
echo "  user:    ${USERNAME}"
echo "  devices: ${DEVICES}"
echo "  port:    ${PORT}"
echo "  cache:   ${CACHE_DIR}"

CUDA_VISIBLE_DEVICES="${DEVICES}" python mminf/api_server/entrypoint.py \
    --config configs/vjepa2_ac.yaml \
    --port "${PORT}" \
    --cache-dir "${CACHE_DIR}" \
    --socket-path-prefix "/tmp/mminf_${USERNAME}/" \
    --upload-dir "/tmp/mminf_uploads_${USERNAME}/" \
    --tensor-comm-protocol SHM
