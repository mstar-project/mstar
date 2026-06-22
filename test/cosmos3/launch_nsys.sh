#!/bin/bash

if [ -f "./.env" ]; then
    source ".env"
else
    echo "Error: No .env file found. Run:  \"cp .sample.env .env\" and configure it. Make sure the .env file is in your current working directory."
    exit 1
fi

export TMPDIR=/mnt/storage/naomi/tmp
CUDA_VISIBLE_DEVICES=$DEVICES COSMOS3_DENSE_FA3=1 nsys profile --trace=cuda,nvtx --output=cosmos_profile --force-overwrite=true \
    python mstar/api_server/entrypoint.py \
    --config configs/cosmos3_nano.yaml --enable-nvtx \
    --socket-path-prefix /tmp/mstar_$WHO/ \
    --upload-dir /tmp/mstar_uploads_$WHO/ \
    --port $PORT \
    --tensor-comm-protocol $TENSOR_PROTOCOL \
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0}
    # --log-level DEBUG
