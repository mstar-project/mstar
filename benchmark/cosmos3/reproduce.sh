#!/bin/bash
# Reproduce the Cosmos3-Nano serving benchmarks (M* vs vLLM-Omni): t2i / t2v / i2v
# latency and t2i throughput under concurrency. Both engines expose the OpenAI
# /v1/images/generations + /v1/videos APIs, so the client scripts in this dir hit
# both identically (same prompt / tiers / steps / guidance / seed).
#
# Measured on 1x H100 80GB, CUDA 13. Serve one engine per GPU; run them on
# separate GPUs so the bench clients can hit both back-to-back.
#
# Set for your machine before serving:
#   SNAP     = Cosmos3-Nano HF snapshot dir   (hf download nvidia/Cosmos3-Nano)
#   MSTAR    = this repo checkout
#   HF_TOKEN = your Hugging Face token         (Cosmos3-Nano is gated)
set -eu

# --------------------------------------------------------------------------
# Serve M* (this repo). torch.compile + CUDA graphs are on by default.
# COSMOS3_GEN_CAPTURE_RES bakes a denoise graph per benchmarked resolution;
# COSMOS3_GEN_CAPTURE_BS additionally captures batched (concurrent) denoise
# steps, which the throughput sweep needs to scale past one request.
#   usage: serve_mstar <gpu> <port>
# --------------------------------------------------------------------------
serve_mstar() {
  : "${SNAP:?set SNAP to the Cosmos3-Nano snapshot dir}"
  : "${MSTAR:?set MSTAR to the repo checkout}"
  local sock upload
  sock=$(mktemp -d); upload=$(mktemp -d)
  CUDA_VISIBLE_DEVICES="$1" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    COSMOS3_GEN_CAPTURE_RES=192x320,480x832,720x1280 \
    COSMOS3_GEN_CAPTURE_BS=1,4,8 \
    COSMOS3_NANO_DIR="$SNAP" PYTHONPATH="$MSTAR" \
    python "$MSTAR/mstar/api_server/entrypoint.py" \
    --config "$MSTAR/configs/cosmos3_nano.yaml" \
    --socket-path-prefix "$sock/" --upload-dir "$upload/" \
    --port "$2" --mooncake-port "$(($2 + 1000))" --tensor-comm-protocol SHM
}

# --------------------------------------------------------------------------
# Serve vLLM-Omni (baseline). Prebuilt cu13 wheel; same OpenAI API.
#   usage: serve_vllm <gpu> <port>
# --------------------------------------------------------------------------
serve_vllm() {
  CUDA_VISIBLE_DEVICES="$1" \
    vllm serve nvidia/Cosmos3-Nano --omni --no-guardrails \
    --host 0.0.0.0 --port "$2" --init-timeout 1800
}

# --------------------------------------------------------------------------
# Benchmarks. Serve each engine first (e.g. `serve_mstar 0 18300` and
# `serve_vllm 1 8200` in separate shells), then run the clients below.
# Defaults: 256p/480p/720p tiers, 50 steps (t2i), gs 6, seed 0.
# --------------------------------------------------------------------------
here=$(dirname "$0")
run_benches() {  # args: <mstar_port> <vllm_port>
  local mp="$1" vp="$2"
  # t2i latency (median of N, per tier)
  python "$here/bench_t2i_oai.py"   --port "$mp" --model cosmos3_nano        --tag mstar
  python "$here/bench_t2i_oai.py"   --port "$vp" --model nvidia/Cosmos3-Nano --tag vllm
  # t2v latency (189 frames, 35 steps)
  python "$here/video_bench.py"     --engine ours --port "$mp"
  python "$here/video_bench.py"     --engine vllm --port "$vp"
  # i2v latency (same, plus a conditioning frame)
  python "$here/video_bench.py"     --engine ours --port "$mp" --image cond.jpg
  python "$here/video_bench.py"     --engine vllm --port "$vp" --image cond.jpg
  # t2i throughput under concurrency (bs 1/4/8)
  python "$here/bench_throughput.py" --port "$mp" --model cosmos3_nano        --tag mstar
  python "$here/bench_throughput.py" --port "$vp" --model nvidia/Cosmos3-Nano --tag vllm
}

case "${1:-}" in
  serve-mstar) shift; serve_mstar "$@";;
  serve-vllm)  shift; serve_vllm "$@";;
  bench)       shift; run_benches "$@";;
  *) echo "usage: $0 {serve-mstar <gpu> <port> | serve-vllm <gpu> <port> | bench <mstar_port> <vllm_port>}";;
esac
