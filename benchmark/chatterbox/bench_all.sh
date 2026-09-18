#!/usr/bin/env bash
# Chatterbox benchmark driver: M* vs Chatterbox-TTS-Server vs chatterbox-vllm
# on one GPU, following docs/benchmark protocol (closed loop, concurrency
# 1/8/32, 200 shared sentences, WER guard). Every step is a plain command so a
# single line can be re-run by hand.
#
#   bench_all.sh env                       # record GPU / versions / git SHA
#   bench_all.sh mstar   <config.yaml> <c> # serve M* with the config, run the closed loop at concurrency c
#   bench_all.sh tts     <original|turbo> <c>  # same against Chatterbox-TTS-Server
#   bench_all.sh vllm    <batch>           # chatterbox-vllm offline at batch size = concurrency
#   bench_all.sh wer     <wav dir>         # transcribe + score a finished run
#
# Environment: SENTENCES (prompt file), RESULTS (output root), VOICE (preset
# file name resolved by every system, default Abigail.wav), VOICES_DIR (the
# preset directory, default the TTS server's), WS (workspace with the baseline
# venvs), NUM (requests, default 200). For M* variants: EXTRA_MODEL_KWARGS
# (extra indented "  key: value" model_kwargs lines) and RUN_TAG (run name suffix).
set -euo pipefail

WS=${WS:-$(cd "$(dirname "$0")/../../.." && pwd)}
MSTAR=${MSTAR:-$WS/mstar}
SENTENCES=${SENTENCES:-/scratch/m000137-pm06/atj10/mstar-ws/commons/bench/data/tts/sentences_200.txt}
RESULTS=${RESULTS:-$WS/results/$(date +%F)}
VOICE=${VOICE:-Abigail.wav}
VOICES_DIR=${VOICES_DIR:-$WS/refs/Chatterbox-TTS-Server/voices}
NUM=${NUM:-200}
WARMUP=${WARMUP:-3}
PORT=${PORT:-8000}
export CHATTERBOX_BENCH_VOICE=$VOICE
mkdir -p "$RESULTS"

# mstar-serve spawns a conductor and workers; killing only the front process
# leaves them holding the GPU (and the next server on the same IPC prefix talks
# to the stale workers). Start every server as its own process group with a
# private socket prefix and take the whole group down.
start_group() {  # start_group <log> <cmd...> -> sets GROUP_PID
  local log=$1; shift
  setsid "$@" > "$log" 2>&1 &
  GROUP_PID=$!
}

stop_group() {
  [ -n "${GROUP_PID:-}" ] || return 0
  kill -TERM -- "-$GROUP_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$GROUP_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$GROUP_PID" 2>/dev/null || true
  GROUP_PID=
  # sweep workers this venv spawned that escaped the group
  pkill -9 -u "$USER" -f "$WS/[.]venv/bin/python -c from multiprocessing" 2>/dev/null || true
  sleep 3
}

wait_http() {  # wait_http <url> <seconds>
  local url=$1 deadline=$((SECONDS + $2))
  until curl -sf "$url" > /dev/null; do
    (( SECONDS < deadline )) || { echo "timeout waiting for $url" >&2; return 1; }
    sleep 2
  done
}

runner() {  # runner <base url> <concurrency> <out dir>
  local url=$1 c=$2 out=$3
  mkdir -p "$out/wavs"
  { git -C "$MSTAR" rev-parse HEAD; git -C "$MSTAR" status --short | head -20; } > "$out/sha.txt" 2>/dev/null || true
  ( cd "$MSTAR" && python -m benchmark.runner --url "$url" --model chatterbox \
      --inference-system ours_openai --request-type text_to_speech \
      --dataset text --request-txt-file "$SENTENCES" \
      --profiling-type closed_loop --max-concurrency "$c" \
      --num-requests "$NUM" --num-warmup "$WARMUP" \
      --output-dir "$out/wavs" --local-cache "$out/cache" ) 2>&1 | tee "$out/runner.log"
  wer "$out/wavs" "$out/wer.json"
}

wer() {  # wer <wav dir> [out json]
  local wavs=$1 out=${2:-$1/../wer.json}
  ( cd "$MSTAR" && HF_HUB_OFFLINE=1 python benchmark/chatterbox/wer_eval.py \
      --wavs "$wavs" --sentences "$SENTENCES" --out "$out" ) 2>&1 | tee "$(dirname "$out")/wer.log"
}

record_env() {
  local out=$RESULTS/env.txt
  { date -Is; hostname; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit --format=csv
    ( cd "$MSTAR" && git rev-parse HEAD && python -c "import torch, flashinfer; print('torch', torch.__version__, 'flashinfer', flashinfer.__version__)" )
    "$WS/baselines/tts-server/.venv/bin/python" -c "import torch, chatterbox, transformers; print('tts-server torch', torch.__version__, 'transformers', transformers.__version__)"
    "$WS/baselines/chatterbox-vllm/.venv/bin/python" -c "import torch, vllm; print('chatterbox-vllm torch', torch.__version__, 'vllm', vllm.__version__)"
  } 2>&1 | tee "$out"
}

case ${1:-} in
  env) record_env ;;
  mstar)
    cfg=$2 c=$3 out=$RESULTS/mstar_$(basename "${cfg%.yaml}")${RUN_TAG:+_$RUN_TAG}_c$c
    mkdir -p "$out"
    # same preset voices as the TTS server, appended as model_kwargs to a copy of the config;
    # EXTRA_MODEL_KWARGS (indented "  key: value" lines) adds deployment knobs, RUN_TAG names the run
    { cat "$MSTAR/$cfg"; printf 'model_kwargs:\n  voices_dir: %s\n%s' "$VOICES_DIR" "${EXTRA_MODEL_KWARGS:-}"; } > "$out/config.yaml"
    cd "$MSTAR"
    # shellcheck disable=SC2086  # SERVE_EXTRA_ARGS is a list of extra mstar-serve flags (e.g. --log-level DEBUG)
    start_group "$out/server.log" mstar-serve --config "$out/config.yaml" --port "$PORT" \
        --tensor-comm-protocol SHM --socket-path-prefix "/tmp/mstar_${USER}_bench_$$/" \
        --log-stats --log-stats-file "$out/stats.log" ${SERVE_EXTRA_ARGS:-}
    trap 'stop_group' EXIT
    wait_http "http://127.0.0.1:$PORT/health" 900
    # one long-timeout request first: JIT kernels, CUDA-graph and torch.compile
    # warm-up must not eat into the runner's own warmup (300 s client timeout)
    curl -s --max-time 1800 "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' \
        -d "{\"model\":\"chatterbox\",\"input\":\"Warming up the server before the benchmark run.\",\"voice\":\"$VOICE\"}" \
        -o "$out/warmup.wav"
    runner "http://127.0.0.1:$PORT" "$c" "$out"
    ;;
  tts)
    variant=$2 c=$3 out=$RESULTS/tts_server_${variant}_c$c
    run_dir=$WS/baselines/tts-server/run_$variant
    port=$(awk '/^  port:/ {print $2}' "$run_dir/config.yaml")
    mkdir -p "$out"
    cd "$run_dir"
    HF_HUB_OFFLINE=1 start_group "$out/server.log" "$WS/baselines/tts-server/.venv/bin/python" \
        "$WS/refs/Chatterbox-TTS-Server/server.py"
    trap 'stop_group' EXIT
    wait_http "http://127.0.0.1:$port/docs" 900
    runner "http://127.0.0.1:$port" "$c" "$out"
    ;;
  vllm)
    b=$2 out=$RESULTS/chatterbox_vllm_c$b
    mkdir -p "$out/t3-model"
    # run from the results dir: the port symlinks its T3 weights into ./t3-model, next to
    # the vLLM model config files it ships in its repo
    cp -rn "$WS/refs/chatterbox-vllm/t3-model/." "$out/t3-model/" 2>/dev/null || true
    # the port registers its custom tokenizer at import time (no vLLM plugin entry point), which a
    # spawned engine-core process never sees: run the engine core in-process
    ( cd "$out" && HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 "$WS/baselines/chatterbox-vllm/.venv/bin/python" \
        "$MSTAR/benchmark/chatterbox/bench_chatterbox_vllm.py" --sentences "$SENTENCES" --num "$NUM" \
        --warmup "$WARMUP" --batch "$b" --out "$out" ) 2>&1 | tee "$out.log"
    wer "$out" "$out/wer.json"
    ;;
  wer) wer "$2" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
