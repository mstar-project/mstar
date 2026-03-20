#!/bin/bash

WHO=...

CACHE_DIR=/mnt/storage/$WHO/mminf/bagel/
DEVICES=2,3

CUDA_VISIBLE_DEVICES=$DEVICES nsys profile --trace=cuda,nvtx --output=bagel_profile --force-overwrite=true \
    python -m mminf.api_server.entrypoint --config configs/bagel.yaml --enable-nvtx \
    --cache-dir $CACHE_DIR \
        --socket-path-prefix /tmp/mminf_$WHO/ \
        --upload-dir /tmp/mminf_uploads_$WHO/