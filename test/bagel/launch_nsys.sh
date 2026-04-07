#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "Error: No .env file found. Run:  cp test/bagel/.sample.env test/bagel/.env  and configure it."
    exit 1
fi

CUDA_VISIBLE_DEVICES=$DEVICES nsys profile --trace=cuda,nvtx --output=bagel_profile --force-overwrite=true \
    python -m mminf.api_server.entrypoint --config configs/bagel.yaml --enable-nvtx \
    --cache-dir $CACHE_DIR \
    --socket-path-prefix /tmp/mminf_$WHO/ \
    --upload-dir /tmp/mminf_uploads_$WHO/
