#!/bin/bash
# Launch the PersonaPlex-7B server (offline, preloaded-input replay; one GPU).
#
#   ./test/personaplex/launch_server.sh
#
# Voice presets are fetched from the gated repo's voices.tgz on first use
# unless model_kwargs.voice_dir in the config points at a local directory.

if [ -f "./.env" ]; then
    source "./.env"
elif [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
else
    echo "Error: no .env found. Run: cp test/personaplex/.sample.env test/personaplex/.env and edit it."
    exit 1
fi

CONFIG=configs/personaplex.yaml
echo "Serving $CONFIG on GPU=$DEVICES port=$PORT"

CACHE_ARG=""
if [[ -n "$PERSONAPLEX_CACHE_DIR" ]]; then
    CACHE_ARG="--cache-dir $PERSONAPLEX_CACHE_DIR"
fi

CUDA_VISIBLE_DEVICES=$DEVICES python mstar/api_server/entrypoint.py \
    --config $CONFIG --host ${HOST:-127.0.0.1} --port $PORT \
    $CACHE_ARG \
    --socket-path-prefix /tmp/mstar_${WHO}_${PORT}/ \
    --upload-dir /tmp/mstar_uploads_${WHO}/ \
    --tensor-comm-protocol ${TENSOR_PROTOCOL:-SHM} \
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0}
