#!/bin/bash
# Launch the Zonos2 TTS server.
#
# Colocated (default): the LLM (prefill+decode) and the DAC vocoder share one
# GPU. Two-GPU: pass CONFIG=configs/zonos2.yaml and DEVICES=0,1.
#
# Point ZONOS2_MODEL_PATH at a checkpoint in the reference layout — a directory
# with params.json + model.pth (or an HF repo id). Without it the server still
# starts, but with uninitialized (noise) weights.
#
#   ZONOS2_MODEL_PATH=/path/to/zonos2_ckpt bash test/zonos2/launch_server_zonos2.sh
#
# Requires: pip install descript-audio-codec   (for the DAC vocoder)

set -euo pipefail

DEVICES="${DEVICES:-0}"
PORT="${PORT:-20002}"
CONFIG="${CONFIG:-configs/zonos2_colocated.yaml}"
TENSOR_PROTOCOL="${TENSOR_PROTOCOL:-SHM}"
WHO="${WHO:-$USER}"

# Defaults to the public HF checkpoint (mirrors mstar.model.registry). Override
# by exporting ZONOS2_MODEL_PATH=/path/to/checkpoint (dir with params.json + model.pth).
ZONOS2_MODEL_PATH="${ZONOS2_MODEL_PATH:-Zyphra/ZONOS2}"
echo "ZONOS2_MODEL_PATH=$ZONOS2_MODEL_PATH"

export ZONOS2_MODEL_PATH  # read by mstar.model.registry for model_path_hf

SOCKET_DIR="/tmp/mstar_${WHO}/"

# Killed/crashed server runs leave orphaned multiprocessing launchers behind
# (reparented to init) plus their .ipc socket files. New requests then connect
# to a dead socket and hang until timeout. Consider implementing this in other scripts

cleanup_stale_server() {
    local orphans
    orphans=$(ps -o pid=,ppid=,cmd= -u "$WHO" 2>/dev/null \
        | awk '$2==1 && /multiprocessing\.(resource_tracker|spawn)/ && /(mstar|ZONOS2)\/\.venv/ {print $1}')
    if [ -n "$orphans" ]; then
        echo "Reaping stale mstar orphans:" $orphans
        kill -9 $orphans 2>/dev/null || true
    fi
    if compgen -G "${SOCKET_DIR}*.ipc" > /dev/null; then
        echo "Purging stale sockets in $SOCKET_DIR"
        rm -f "${SOCKET_DIR}"*.ipc
    fi
}
cleanup_stale_server

CUDA_VISIBLE_DEVICES="$DEVICES" python mstar/api_server/entrypoint.py \
    --config "$CONFIG" \
    --port "$PORT" \
    --cache-dir "${CACHE_DIR:-$HOME/.cache/huggingface}" \
    --tensor-comm-protocol "$TENSOR_PROTOCOL" \
    --socket-path-prefix "/tmp/mstar_${WHO}/" \
    --upload-dir "/tmp/mstar_uploads_${WHO}/"
