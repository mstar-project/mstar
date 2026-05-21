#!/bin/bash

if [ -f "./.env" ]; then
    source ".env"
else
    echo "Error: No .env file found. Run:  \"cp .sample.env .env\" and configure it. Make sure the .env file is in your current working directory."
    exit 1
fi

# Launch the Orpheus TTS server on two GPUs.
# GPU 0 runs the LLM (prefill + decode) and GPU 1 runs the SNAC audio decoder.

# export LD_LIBRARY_PATH=/m-coriander/coriander/keisuke/miniconda3/envs/mmstar/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [[ -v ORPHEUS_CACHE_DIR ]]; then
    echo "Cache dir set to: $ORPHEUS_CACHE_DIR"
else
    echo "Error: environment variable \"ORPHEUS_CACHE_DIR\" not found. Please set it in .env!"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$DEVICES nsys profile --trace=cuda,nvtx --output=orpheus_profile --force-overwrite=true \
    python mminf/api_server/entrypoint.py --enable-nvtx \
    --config configs/orpheus_colocated.yaml --port $PORT \
    --cache-dir $ORPHEUS_CACHE_DIR \
    --socket-path-prefix /tmp/mminf_${WHO}/ \
    --upload-dir /tmp/mminf_uploads_${WHO}/ \
    --tensor-comm-protocol $TENSOR_PROTOCOL \
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0}
