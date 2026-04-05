#!/bin/bash

# Launch the Orpheus TTS server on two GPUs.
# GPU 0 runs the LLM (prefill + decode) and GPU 1 runs the SNAC audio decoder.

DEVICES="${1:-0,1}"

export LD_LIBRARY_PATH=/m-coriander/coriander/keisuke/miniconda3/envs/mmstar/lib:$LD_LIBRARY_PATH

# Clean stale IPC sockets to avoid ZMQ race conditions
rm -rf /tmp/mminf

CUDA_VISIBLE_DEVICES=$DEVICES python mminf/api_server/entrypoint.py \
    --config configs/orpheus.yaml --port 20001 \
    --log-level DEBUG --tensor-comm-protocol TCP --tcp-transfer-device "0.0.0.0:0"
