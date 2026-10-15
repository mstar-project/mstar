#!/bin/bash
# Launch the Nemotron-Duplex (VoiceChat-11B) full-duplex server.
#
#   ./launch_server.sh              # single-GPU layout (configs/nemotron_duplex.yaml)
#   ./launch_server.sh disagg       # 3-GPU disaggregated layout (nemotron_duplex_disagg.yaml)
#
# Both layouts serve the same text + audio; disagg puts each stream-consuming
# loop (Encoder+LLM / Talker / Codec) on its own GPU/worker thread.

if [ -f "./.env" ]; then
    source "./.env"
elif [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
else
    echo "Error: no .env found. Run: cp test/nemotron_duplex/.sample.env test/nemotron_duplex/.env and edit it."
    exit 1
fi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CONFIG=configs/nemotron_duplex.yaml
if [ "$1" == "disagg" ]; then
    CONFIG=configs/nemotron_duplex_disagg.yaml
fi
echo "Serving $CONFIG on GPUs=$DEVICES port=$PORT"

CACHE_ARG=""
if [[ -n "$NEMOTRON_DUPLEX_CACHE_DIR" ]]; then
    CACHE_ARG="--cache-dir $NEMOTRON_DUPLEX_CACHE_DIR"
fi

CUDA_VISIBLE_DEVICES=$DEVICES python mstar/api_server/entrypoint.py \
    --config $CONFIG --host ${HOST:-127.0.0.1} --port $PORT \
    $CACHE_ARG \
    --socket-path-prefix /tmp/mstar_${WHO}/ \
    --upload-dir /tmp/mstar_uploads_${WHO}/ \
    --tensor-comm-protocol ${TENSOR_PROTOCOL:-SHM} \
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0}
