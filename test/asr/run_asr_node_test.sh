#!/usr/bin/env bash
# Self-contained ASR validation on a fresh GPU node (H100, 5090, ...).
#
# Sets up the environment, launches the mstar server for each ASR model,
# runs the LibriSpeech test-clean WER eval against it, prints the
# results, and tears the server down. Designed to be the single command
# run after cloning the repo on a new machine:
#
#   ./test/asr/run_asr_node_test.sh [--num-requests 50] [--model whisper_large|higgs_audio]
#
# Environment notes baked in (learned on the RTX 5090 bring-up):
#   * flashinfer JIT-compiles kernels on first use and shells out to
#     `ninja` — the venv bin dir must be on PATH for the server process.
#   * torchcodec dlopens libav*; if the system has no FFmpeg, a
#     user-local shared build is installed under ~/.local/ffmpeg.
#   * The ZMQ --socket-path-prefix must be UNIQUE PER LAUNCH: a reused
#     prefix delivers stale messages from a previous run and crashes the
#     worker main loop (requests then hang silently).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NUM_REQUESTS=50
MODELS=(whisper_large higgs_audio)
PORT_BASE=18810

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
    --model)        MODELS=("$2");     shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---------------- environment setup ----------------

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found; install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "== Creating venv"
  uv venv
fi
echo "== Installing mstar[asr] + ninja"
uv pip install -q -e ".[asr]" ninja

export PATH="$REPO_ROOT/.venv/bin:$PATH"

# torchcodec needs FFmpeg shared libraries (versions 4-8).
if ! .venv/bin/python -c "import torchcodec.decoders" 2>/dev/null; then
  FFMPEG_DIR="$HOME/.local/ffmpeg"
  if [[ ! -f "$FFMPEG_DIR/lib/libavcodec.so" ]]; then
    echo "== Installing user-local FFmpeg 7.1 shared libs to $FFMPEG_DIR"
    mkdir -p "$FFMPEG_DIR"
    curl -sL -o /tmp/ffshared.tar.xz \
      https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-shared-7.1.tar.xz
    tar xf /tmp/ffshared.tar.xz --strip-components=1 -C "$FFMPEG_DIR"
    rm -f /tmp/ffshared.tar.xz
  fi
  export LD_LIBRARY_PATH="$FFMPEG_DIR/lib:${LD_LIBRARY_PATH:-}"
  .venv/bin/python -c "import torchcodec.decoders" \
    || { echo "ERROR: torchcodec still cannot load FFmpeg" >&2; exit 1; }
fi

.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('== GPU:', torch.cuda.get_device_name(0))"

# ---------------- per-model server + eval ----------------

CACHE_DIR="${ASR_EVAL_CACHE:-/tmp/mstar-eval-cache}"
RUN_TAG="$(date +%s)"
declare -A CONFIGS=([whisper_large]=configs/whisper_large.yaml [higgs_audio]=configs/higgs_audio.yaml)

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    pkill -9 -P "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

FAIL=0
for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  port=$((PORT_BASE + i))
  log="/tmp/asr_node_test_${model}_${RUN_TAG}.log"

  echo ""
  echo "======================================================="
  echo "  $model  (port $port, log $log)"
  echo "======================================================="

  # Fresh socket prefix per launch — see header note.
  .venv/bin/python mstar/api_server/entrypoint.py \
    --config "${CONFIGS[$model]}" \
    --socket-path-prefix "/tmp/mstar_${model}_${RUN_TAG}/" \
    --upload-dir "/tmp/mstar_uploads_${model}_${RUN_TAG}/" \
    --port "$port" \
    --tensor-comm-protocol SHM > "$log" 2>&1 &
  SERVER_PID=$!

  echo "-- waiting for health (first launch downloads weights + JIT-compiles kernels)"
  ok=0
  for _ in $(seq 1 200); do
    if curl -s -m 2 "http://localhost:${port}/health" >/dev/null 2>&1; then ok=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 3
  done
  if [[ "$ok" != 1 ]]; then
    echo "ERROR: $model server failed to become healthy; last log lines:" >&2
    tail -20 "$log" >&2
    FAIL=1
    cleanup; SERVER_PID=""
    continue
  fi

  if ! .venv/bin/python -m benchmark.asr_eval \
      --url "http://localhost:${port}" \
      --model "$model" \
      --num-requests "$NUM_REQUESTS" \
      --max-concurrency 4 \
      --local-cache "$CACHE_DIR" \
      --output-json "/tmp/asr_node_test_${model}_${RUN_TAG}.json"; then
    echo "ERROR: eval failed for $model" >&2
    FAIL=1
  fi

  cleanup; SERVER_PID=""
done

echo ""
if [[ "$FAIL" != 0 ]]; then
  echo "ASR node test FAILED — see logs above."
  exit 1
fi
echo "ASR node test complete. Result JSONs: /tmp/asr_node_test_*_${RUN_TAG}.json"
