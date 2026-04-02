#!/bin/bash

# Launch the Orpheus TTS server on two GPUs.
# GPU 0 runs the LLM decode walk and GPU 1 runs the SNAC audio decode walk.

DEVICES="${1:-0,1}"

CUDA_VISIBLE_DEVICES=$DEVICES python mminf/api_server/entrypoint.py \
    --config configs/orpheus.yaml --port 20001 \
    --log-level DEBUG
