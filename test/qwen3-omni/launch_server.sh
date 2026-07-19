#!/bin/bash

if [ -f "./.env" ]; then
    source ".env"
else
    echo "Error: No .env file found. Run:  \"cp .sample.env .env\" and configure it. Make sure the .env file is in your current working directory."
    exit 1
fi

# export LD_LIBRARY_PATH=/m-coriander/coriander/keisuke/miniconda3/envs/mmstar/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


if [[ -v QWEN3OMNI_CACHE_DIR ]]; then
    echo "Cache dir set to: $QWEN3OMNI_CACHE_DIR"
else
    echo "Error: environment variable \"QWEN3OMNI_CACHE_DIR\" not found. Please set it in .env!"
    exit 1
fi

export MSTAR_SHM_ARENA=0
# export MSTAR_SHM_ARENA_STATS_INTERVAL_S=10
CUDA_VISIBLE_DEVICES=$DEVICES python mstar/api_server/entrypoint.py \
    --config configs/qwen3omni_2gpu.yaml --log-stats \
    --cache-dir $QWEN3OMNI_CACHE_DIR \
    --socket-path-prefix /tmp/mstar_${WHO}/ \
    --upload-dir /tmp/mstar_uploads_${WHO}/ \
    --port $PORT \
    --tensor-comm-protocol $TENSOR_PROTOCOL \
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0}
