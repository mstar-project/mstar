#!/bin/bash

# Launch the Orpheus TTS server on a single GPU.
# Llama 3.2 3B (~6GB) + SNAC (~100MB) fit comfortably on one GPU.

DEVICE="${1:-0}"

CUDA_VISIBLE_DEVICES=$DEVICE python mminf/api_server/entrypoint.py \
    --config configs/orpheus.yaml --port 20001 \
    --log-level DEBUG
