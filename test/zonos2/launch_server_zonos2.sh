#!/bin/bash

if [ -f "./.env" ]; then
    source ".env"
else
    echo "Error: No .env file found. Run:  \"cp .sample.env .env\" and configure it. Make sure the .env file is in your current working directory."
    exit 1
fi

# Launch the Zonos2 TTS server.
# Colocated: the LLM (prefill + decode) and the DAC vocoder share GPU 0.
# For two GPUs (LLM on rank 0, DAC on rank 1) use configs/zonos2.yaml below.
#
# Requires: pip install descript-audio-codec   (for the DAC vocoder)

# coriander may need:
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [[ -v ZONOS2_CACHE_DIR ]]; then
    echo "Cache dir set to: $ZONOS2_CACHE_DIR"
else
    echo "Error: environment variable \"ZONOS2_CACHE_DIR\" not found. Please set it in .env!"
    exit 1
fi

# The voice-clone speaker encoder lives on the hub as remote code, so
# transformers writes its module files to HF_MODULES_CACHE at load time. The
# default sits under a shared HF_HOME that is often not writable, which fails
# the load rather than falling back. Keep it beside the weights we already own.
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-${ZONOS2_CACHE_DIR%/}/hf_modules}

CUDA_VISIBLE_DEVICES=$DEVICES python mstar/api_server/entrypoint.py \
    --config configs/zonos2_colocated.yaml \
    --cache-dir $ZONOS2_CACHE_DIR \
    --socket-path-prefix /tmp/mstar_$WHO/ \
    --upload-dir /tmp/mstar_uploads_$WHO/ \
    --port $PORT \
    --tensor-comm-protocol $TENSOR_PROTOCOL \
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0}
