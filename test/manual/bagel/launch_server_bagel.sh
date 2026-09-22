#!/bin/bash

if [ -f "./.env" ]; then
    source ".env"
else
    echo "Error: No .env file found. Run:  \"cp .sample.env .env\" and configure it. Make sure the .env file is in your current working directory."
    exit 1
fi

# coriander may need:
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [[ -v BAGEL_CACHE_DIR ]]; then
    echo "Cache dir set to: $BAGEL_CACHE_DIR"
else
    echo "Error: environment variable \"BAGEL_CACHE_DIR\" not found. Please set it in .env!"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$DEVICES python mstar/api_server/entrypoint.py \
    --config configs/bagel_cfg_parallel.yaml \
    --cache-dir $BAGEL_CACHE_DIR \
    --socket-path-prefix /tmp/mstar_$WHO/ \
    --upload-dir /tmp/mstar_uploads_$WHO/ \
    --port $PORT \
    --tensor-comm-protocol $TENSOR_PROTOCOL \
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0}
    # --log-level DEBUG
