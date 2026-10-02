#!/usr/bin/env bash
# Protocol sweep for a TTS server on the OpenAI speech route (BENCHMARK_PROTOCOL.md, TTS row):
# concurrency 1, 8 and 32, N repeats each, on a fixed sentence file, with the environment recorded.
#
#   benchmark/run_tts_benchmark.sh <url> <label> <out_root> <sentences.txt> [repeats=3] [model=kokoro]
#
# Writes <out_root>/<label>/c<C>_r<R>/results.json per run plus <out_root>/<label>/env.txt.
set -euo pipefail
URL=${1:?url}; LABEL=${2:?label}; OUT=${3:?out_root}; SENTENCES=${4:?sentences.txt}
REPEATS=${5:-3}; MODEL=${6:-kokoro}
DEST="$OUT/$LABEL"; mkdir -p "$DEST"
{
  date -Is; echo "url=$URL model=$MODEL sentences=$SENTENCES"
  echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm --format=csv
  command -v nvidia-smi >/dev/null && nvidia-smi -q -d CLOCK | sed -n '1,40p'
  python -c "import torch; print('torch', torch.__version__)"
} > "$DEST/env.txt" 2>&1
for C in 1 8 32; do
  for R in $(seq 1 "$REPEATS"); do
    python -m benchmark.runner --url "$URL" --model "$MODEL" --inference-system ours_openai \
      --request-type text_to_speech --dataset text --request-txt-file "$SENTENCES" \
      --num-requests 200 --num-warmup 5 --profiling-type closed_loop --max-concurrency "$C" \
      --output-dir "$DEST/c${C}_r${R}" | tee "$DEST/c${C}_r${R}.log"
  done
done
