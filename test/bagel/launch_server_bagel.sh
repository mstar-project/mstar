#!/bin/bash

CACHE_DIR=/mnt/storage/naomi/mminf/bagel/
DEVICES=1,2

CUDA_VISIBLE_DEVICES=$DEVICES python mminf/api_server/entrypoint.py \
    --config configs/bagel.yaml \
    --cache-dir $CACHE_DIR \
    --socket-path-prefix /tmp/mminf_2/
    # --log-level DEBUG