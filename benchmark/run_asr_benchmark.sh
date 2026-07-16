#!/usr/bin/env bash
# ASR benchmark script: throughput and latency for Whisper-large-v3 and Higgs Audio.
#
# Usage:
#   ./benchmark/run_asr_benchmark.sh --url http://localhost:8000 [options]
#
# The script runs each ASR model in sequence.  Start the appropriate mstar
# server before invoking.  Results are written to $OUTPUT_DIR/<model>/.
#
# Required:
#   --url  <base_url>       mstar server base URL
#
# Optional:
#   --model  <name>         Run only this model (whisper_large|higgs_audio)
#   --num-requests  <n>     Total requests per model (default: 100)
#   --batch-size  <b>       Batch size for offline profiling (default: 4)
#   --max-concurrency <c>   Concurrency cap for closed-loop profiling (default: 4)
#   --profiling-type <t>    offline|closed_loop|online (default: offline)
#   --output-dir  <dir>     Directory for results JSON (default: ./asr_benchmark_results)
#   --local-cache  <dir>    Local cache for audio files (default: ./mstar-benchmark-cache)
#   --num-warmup  <n>       Warmup requests (default: 5)

set -euo pipefail

URL=""
MODEL=""
NUM_REQUESTS=100
BATCH_SIZE=4
MAX_CONCURRENCY=4
PROFILING_TYPE="offline"
OUTPUT_DIR="./asr_benchmark_results"
LOCAL_CACHE="./mstar-benchmark-cache"
NUM_WARMUP=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)             URL="$2";             shift 2 ;;
    --model)           MODEL="$2";           shift 2 ;;
    --num-requests)    NUM_REQUESTS="$2";    shift 2 ;;
    --batch-size)      BATCH_SIZE="$2";      shift 2 ;;
    --max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
    --profiling-type)  PROFILING_TYPE="$2";  shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
    --local-cache)     LOCAL_CACHE="$2";     shift 2 ;;
    --num-warmup)      NUM_WARMUP="$2";      shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$URL" ]]; then
  echo "ERROR: --url is required" >&2
  exit 1
fi

ASR_MODELS=(whisper_large higgs_audio)

if [[ -n "$MODEL" ]]; then
  ASR_MODELS=("$MODEL")
fi

mkdir -p "$OUTPUT_DIR"

run_model() {
  local model="$1"
  local model_out="${OUTPUT_DIR}/${model}"
  mkdir -p "$model_out"

  echo ""
  echo "========================================================"
  echo "  Benchmarking: ${model}"
  echo "  Profiling:    ${PROFILING_TYPE}"
  echo "  Requests:     ${NUM_REQUESTS}"
  echo "========================================================"

  local extra_args=()
  if [[ "$PROFILING_TYPE" == "offline" ]]; then
    extra_args+=(--batch-size "$BATCH_SIZE")
  elif [[ "$PROFILING_TYPE" == "closed_loop" ]]; then
    extra_args+=(--max-concurrency "$MAX_CONCURRENCY")
  fi

  python -m benchmark.runner \
    --url "$URL" \
    --model "$model" \
    --request-type audio_to_text \
    --dataset libri \
    --num-requests "$NUM_REQUESTS" \
    --num-warmup "$NUM_WARMUP" \
    --profiling-type "$PROFILING_TYPE" \
    --local-cache "$LOCAL_CACHE" \
    --output-dir "$model_out" \
    "${extra_args[@]+"${extra_args[@]}"}"

  echo "  Results saved to: ${model_out}/results.json"
}

for m in "${ASR_MODELS[@]}"; do
  run_model "$m"
done

echo ""
echo "All ASR benchmarks complete.  Results in: ${OUTPUT_DIR}"

# Print a quick comparison table if all three models ran
if [[ -z "$MODEL" ]] && command -v python &>/dev/null; then
  export _ASR_OUTPUT_DIR="$OUTPUT_DIR"
  python - <<'PYEOF'
import json, os, sys

output_dir = os.environ.get("_ASR_OUTPUT_DIR", "./asr_benchmark_results")
models = ["whisper_large", "higgs_audio"]

rows = []
for m in models:
    path = os.path.join(output_dir, m, "results.json")
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        data = json.load(f)
    rows.append((m, data))

if not rows:
    sys.exit(0)

print("\n--- ASR Throughput Comparison ---")
print(f"{'Model':<20} {'Requests':>10} {'Wall(s)':>10} {'Req/s':>10} {'P50(ms)':>10} {'P99(ms)':>10}")
print("-" * 72)
for model, d in rows:
    n = d.get("completed", d.get("num_requests", 0))
    wall = d.get("wall_time_s", 0)
    rps = n / wall if wall > 0 else 0
    p50 = d.get("jct_median_ms", 0)
    p99 = d.get("jct_p99_ms", 0)
    print(f"{model:<20} {n:>10} {wall:>10.2f} {rps:>10.2f} {p50:>10.1f} {p99:>10.1f}")
PYEOF
fi
