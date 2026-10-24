#!/bin/bash

if [ -f ".env" ]; then
    source ".env"
else
    echo "Error: No .env file found. Run:  cp .sample.env .env  and configure it. Make sure the .env file is in your current working directory."
    exit 1
fi

if [[ -v PI05_CACHE_DIR ]]; then
    echo "Cache dir set to: $PI05_CACHE_DIR"
else
    echo "Error: environment variable \"PI05_CACHE_DIR\" not found. Please set it in .env!"
    exit 1
fi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH 

echo "[pi05] launching server"
echo "  user:    ${WHO}"
echo "  devices: ${DEVICES}"
echo "  port:    ${PORT}"
echo "  cache:   ${PI05_CACHE_DIR}"

CUDA_VISIBLE_DEVICES=$DEVICES nsys profile --trace=cuda,nvtx --output=pi05_profile --force-overwrite=true \
    python -m mstar.api_server.entrypoint --config configs/pi05_droid.yaml --enable-nvtx \
    --cache-dir $PI05_CACHE_DIR \
    --socket-path-prefix /tmp/mstar_${WHO}_${PORT}/ \
    --upload-dir /tmp/mstar_uploads_$WHO/ \
    --port $PORT \
    --tensor-comm-protocol $TENSOR_PROTOCOL \
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0}
