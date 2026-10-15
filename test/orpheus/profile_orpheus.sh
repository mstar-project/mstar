#!/bin/bash
#
# Profile Orpheus model execution with nsys + NVTX markers.
# Launches the server under nsys, waits for full readiness (including model
# loading and CUDA graph warmup), sends a warmup request, then fires
# concurrent requests for profiling.
#
# Usage:
#   bash test/orpheus/profile_orpheus.sh [NUM_CONCURRENT_REQUESTS] [GPUS]
#
# Example:
#   bash test/orpheus/profile_orpheus.sh 4 0

set -euo pipefail

NUM_REQUESTS="${1:-4}"
DEVICES="${2:-0}"
PORT="${PORT:-20001}"
export PORT
USERNAME="${USER:-keisuke}"
CACHE_DIR="/m-coriander/coriander/${USERNAME}/mstar_cache/orpheus/"
SOCKET_PREFIX="/tmp/mstar_${USERNAME}/"
UPLOAD_DIR="/tmp/mstar_uploads_${USERNAME}/"
OUTPUT_DIR="nsys_profiles"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_NAME="${OUTPUT_DIR}/orpheus_bs${NUM_REQUESTS}_${TIMESTAMP}"
SERVER_LOG="${OUTPUT_DIR}/server_${TIMESTAMP}.log"

PYTHON="/m-coriander/coriander/keisuke/miniconda3/envs/mstar/bin/python"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export LD_LIBRARY_PATH="/m-coriander/coriander/keisuke/miniconda3/envs/mstar/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "$REPO_DIR"

mkdir -p "$OUTPUT_DIR"

echo "=== Orpheus NVTX Profiling ==="
echo "GPUs:               $DEVICES"
echo "Concurrent requests: $NUM_REQUESTS"
echo "Output:             ${PROFILE_NAME}.nsys-rep"
echo ""

# Launch server under nsys
echo "[1/4] Starting server under nsys..."
CUDA_VISIBLE_DEVICES=$DEVICES nsys profile \
    --trace=cuda,nvtx,osrt \
    --output="$PROFILE_NAME" \
    --force-overwrite=true \
    $PYTHON mstar/api_server/entrypoint.py \
        --config configs/orpheus_colocated.yaml \
        --port "$PORT" \
        --cache-dir "$CACHE_DIR" \
        --socket-path-prefix "$SOCKET_PREFIX" \
        --upload-dir "$UPLOAD_DIR" \
        --tensor-comm-protocol TCP \
        --tcp-transfer-device "0.0.0.0:0" \
        --enable-nvtx \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for HTTP endpoint
echo "[2/4] Waiting for HTTP endpoint..."
for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "  HTTP endpoint up after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died. Log:"
        tail -30 "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: Server did not become ready in time."
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

# Wait for model loading + warmup (CUDA graphs, torch.compile)
echo "  Waiting for model loading and warmup..."
for i in $(seq 1 300); do
    if grep -q "Worker worker_0: engine runs" "$SERVER_LOG" 2>/dev/null; then
        echo "  Model fully loaded and warmed up after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died during warmup. Log:"
        tail -30 "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done

# Send a warmup request to trigger any remaining JIT compilation
echo "[3/4] Sending warmup request..."
$PYTHON test/orpheus/tts_request.py \
    --text "Hello world." \
    --voice tara \
    --output "${OUTPUT_DIR}/warmup.wav" 2>&1 | sed 's/^/  /' || echo "  (warmup request failed, continuing anyway)"

echo "[4/4] Sending $NUM_REQUESTS concurrent requests..."

TEXTS=(
    "The quick brown fox jumps over the lazy dog near the river bank."
    "In a hole in the ground there lived a hobbit, not a nasty dirty wet hole."
    "To be or not to be, that is the question, whether tis nobler in the mind."
    "Space, the final frontier. These are the voyages of the starship Enterprise."
    "It was a bright cold day in April, and the clocks were striking thirteen."
    "All happy families are alike, each unhappy family is unhappy in its own way."
    "Call me Ishmael. Some years ago, never mind how long precisely, having money."
    "It is a truth universally acknowledged that a man in possession of a fortune."
)
VOICES=(tara zoe zac jess leo mia julia leah)

# Fire concurrent requests
PIDS=()
for i in $(seq 0 $((NUM_REQUESTS - 1))); do
    idx=$((i % ${#TEXTS[@]}))
    text="${TEXTS[$idx]}"
    voice="${VOICES[$idx]}"
    outfile="${OUTPUT_DIR}/orpheus_output_${i}.wav"

    $PYTHON test/orpheus/tts_request.py \
        --text "$text" \
        --voice "$voice" \
        --output "$outfile" &
    PIDS+=($!)
    echo "  Request $i (voice=$voice): PID $!"
done

echo "Waiting for all requests to complete..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "  Request PID $pid failed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Results ==="
echo "Completed: $((NUM_REQUESTS - FAILED))/$NUM_REQUESTS requests"

# Stop server (nsys will finalize the profile)
echo "Stopping server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo ""
echo "Profile saved to: ${PROFILE_NAME}.nsys-rep"
echo "Open with: nsys-ui ${PROFILE_NAME}.nsys-rep"
echo ""
echo "NVTX markers to look for:"
echo "  engine.ar.LLM.prefill.bs<N>        - LLM prefill (batch size N)"
echo "  engine.ar.LLM.decode.bs<N>         - LLM autoregressive decode (batch size N)"
echo "  engine.audio_codec.snac_decoder.snac_chunk.bs<N> - SNAC audio decode (batch size N)"
echo ""
echo "Server log: $SERVER_LOG"
