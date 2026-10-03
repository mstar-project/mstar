#!/bin/bash

if [ -f "./.env" ]; then
    source ".env"
else
    echo "Error: No .env file found. Run:  \"cp .sample.env .env\" and configure it. Make sure the .env file is in your current working directory."
    exit 1
fi

# Launch the Zonos2 TTS server.
# Colocated: the LLM (prefill + decode), the DAC vocoder, and the voice-clone
# speaker encoder share GPU 0. For two GPUs (LLM + encoder on rank 0, DAC on
# rank 1) use configs/zonos2.yaml below.
#
# Requires: pip install descript-audio-codec    (the DAC vocoder)
#           pip install transformers torchcodec (the Qwen speech encoder that
#                                                voice cloning needs)

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
    --tcp-transfer-device ${TCP_DEVICE:-0.0.0.0.0} &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT INT TERM

# Warm up: a cold first prefill compiles for up to ~2 min. Two prompt lengths
# per walk reach the dynamic-shape graph. ZONOS2_WARMUP=0 skips this.
if [[ "${ZONOS2_WARMUP:-1}" == 1 ]]; then
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
        kill -0 $SERVER_PID 2>/dev/null || exit 1
        sleep 5
    done
    WARM_DIR=$(mktemp -d)
    URL="http://127.0.0.1:$PORT/generate"
    ok=1
    python test/zonos2/tts_request.py --url "$URL" --output "$WARM_DIR/a.wav" \
        --text "Warming up." || ok=0
    python test/zonos2/tts_request.py --url "$URL" --output "$WARM_DIR/b.wav" \
        --text "The quick brown fox jumps over the lazy dog, twice." || ok=0
    if [[ $ok == 1 ]]; then
        python test/zonos2/tts_request.py --url "$URL" --ref-audio "$WARM_DIR/b.wav" \
            --output "$WARM_DIR/c.wav" --text "Warming up." || ok=0
        python test/zonos2/tts_request.py --url "$URL" --ref-audio "$WARM_DIR/b.wav" \
            --output "$WARM_DIR/d.wav" \
            --text "The quick brown fox jumps over the lazy dog, twice." || ok=0
    fi
    rm -rf "$WARM_DIR"
    if [[ $ok == 1 ]]; then
        echo "Zonos2 warm-up done; serving on port $PORT"
    else
        echo "Zonos2 warm-up FAILED; see the server log above" >&2
    fi
fi
wait $SERVER_PID
