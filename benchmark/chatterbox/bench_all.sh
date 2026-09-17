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
# venvs), NUM (requests, default 200).
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
    cfg=$2 c=$3 out=$RESULTS/mstar_$(basename "${cfg%.yaml}")_c$c
    mkdir -p "$out"
    # same preset voices as the TTS server, appended as model_kwargs to a copy of the config
    { cat "$MSTAR/$cfg"; printf 'model_kwargs:\n  voices_dir: %s\n' "$VOICES_DIR"; } > "$out/config.yaml"
    ( cd "$MSTAR" && mstar-serve --config "$out/config.yaml" --port "$PORT" > "$out/server.log" 2>&1 ) &
    server=$!
    trap 'kill $server 2>/dev/null || true' EXIT
    wait_http "http://127.0.0.1:$PORT/health" 900
    runner "http://127.0.0.1:$PORT" "$c" "$out"
    ;;
  tts)
    variant=$2 c=$3 out=$RESULTS/tts_server_${variant}_c$c
    run_dir=$WS/baselines/tts-server/run_$variant
    port=$(awk '/^  port:/ {print $2}' "$run_dir/config.yaml")
    mkdir -p "$out"
    ( cd "$run_dir" && HF_HUB_OFFLINE=1 "$WS/baselines/tts-server/.venv/bin/python" \
        "$WS/refs/Chatterbox-TTS-Server/server.py" > "$out/server.log" 2>&1 ) &
    server=$!
    trap 'kill $server 2>/dev/null || true' EXIT
    wait_http "http://127.0.0.1:$port/docs" 900
    runner "http://127.0.0.1:$port" "$c" "$out"
    ;;
  vllm)
    b=$2 out=$RESULTS/chatterbox_vllm_c$b
    ( cd "$MSTAR" && HF_HUB_OFFLINE=1 "$WS/baselines/chatterbox-vllm/.venv/bin/python" \
        benchmark/chatterbox/bench_chatterbox_vllm.py --sentences "$SENTENCES" --num "$NUM" \
        --warmup "$WARMUP" --batch "$b" --out "$out" ) 2>&1 | tee "$out.log"
    wer "$out" "$out/wer.json"
    ;;
  wer) wer "$2" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
