#!/bin/bash

# Launch the Orpheus TTS server on two GPUs.
# GPU 0 runs the LLM (prefill + decode) and GPU 1 runs the SNAC audio decoder.

# DEVICES="${1:-0,1}"
DEVICES=5,6,7

# export LD_LIBRARY_PATH=/m-coriander/coriander/keisuke/miniconda3/envs/mmstar/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Clean stale IPC sockets to avoid ZMQ race conditions
# rm -rf /tmp/mminf
username="${1:-${USER:-naomi}}"

CACHE_DIR=/m-coriander/coriander/$username/mminf_cache/qwen3omni/

CUDA_VISIBLE_DEVICES=$DEVICES python mminf/api_server/entrypoint.py \
    --config configs/qwen3omni.yaml --port 20001 \
    --cache-dir $CACHE_DIR \
    --socket-path-prefix /tmp/mminf_${username}/ \
    --upload-dir /tmp/mminf_uploads_${username}/ \
    --tensor-comm-protocol TCP --tcp-transfer-device "0.0.0.0:0"
