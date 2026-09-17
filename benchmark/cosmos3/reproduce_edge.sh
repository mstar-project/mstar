#!/bin/bash
# Reproduce the Cosmos3-Edge serving benchmarks: M* vs vLLM-Omni (generator:
# i2v/t2v at 480p, action policy) and M* vs vLLM (reasoner: TTFT / decode tok/s).
# Both engines expose OpenAI-compatible routes, so the clients in this dir hit
# them identically (same prompt / size / frames / steps / guidance / seed).
#
# Protocol: same H100, same node, back-to-back, warmup excluded, >= 3 repeats.
# Set for your machine before serving:
#   SNAP   = Cosmos3-Edge HF snapshot dir (hf download nvidia/Cosmos3-Edge)
#   MSTAR  = this repo checkout
#   VLLM_OMNI_PY / VLLM_PY = python of the pinned baseline envs (vllm-omni 0.28 / vllm 0.29)
set -eu

# --------------------------------------------------------------------------
# Serve M* (this repo): configs/cosmos3_edge.yaml serves the generator walks and
# the reasoner on one GPU. Denoise CUDA graphs are captured for the 480p tier.
#   usage: serve_mstar <gpu> <port>
# --------------------------------------------------------------------------
serve_mstar() {
  : "${MSTAR:?set MSTAR to the repo checkout}"
  local sock upload
  sock=$(mktemp -d); upload=$(mktemp -d)
  CUDA_VISIBLE_DEVICES="$1" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    COSMOS3_GEN_CAPTURE_RES=192x320,480x832,640x640 \
    PYTHONPATH="$MSTAR" \
    python "$MSTAR/mstar/api_server/entrypoint.py" \
    --config "$MSTAR/configs/cosmos3_edge.yaml" \
    --socket-path-prefix "$sock/" --upload-dir "$upload/" \
    --port "$2" --mooncake-port "$(($2 + 1000))" --tensor-comm-protocol SHM
}

# vLLM-Omni generator baseline (recipes/cosmos3/Cosmos3-Edge.md flags).
#   usage: serve_vllm_omni <gpu> <port>
serve_vllm_omni() {
  : "${VLLM_OMNI_PY:?set VLLM_OMNI_PY to the vllm-omni env python}"
  CUDA_VISIBLE_DEVICES="$1" "$VLLM_OMNI_PY" -m vllm.entrypoints.cli.main serve nvidia/Cosmos3-Edge --omni \
    --no-guardrails --host 0.0.0.0 --port "$2" --init-timeout 1800
}

# vLLM reasoner baseline (model-card flags).
#   usage: serve_vllm_reasoner <gpu> <port>
serve_vllm_reasoner() {
  : "${VLLM_PY:?set VLLM_PY to the vllm env python}"
  CUDA_VISIBLE_DEVICES="$1" "$VLLM_PY" -m vllm.entrypoints.cli.main serve nvidia/Cosmos3-Edge \
    --host 0.0.0.0 --port "$2" --max-model-len 131072 --allowed-local-media-path / \
    --mm-processor-kwargs '{"do_resize": true, "min_pixels": 4096, "max_pixels": 16777216}' \
    --media-io-kwargs '{"video": {"num_frames": 256}}'
}

here=$(dirname "$0")

# Streaming rollout (M* only; no baseline streams frames): TTFF, window cadence
# and frames/s for kv / chained windows vs the same clip generated whole.
#   usage: bench_stream <mstar_port> [cond_image.jpg]
bench_stream() {
  local mp="$1" img="${2:-}" extra=()
  [ -n "$img" ] && extra=(--image "$img")
  for mode in kv chained none; do
    python "$here/bench_stream_video.py" --port "$mp" --mode "$mode" --size 832x480 \
      --frames 241 --window-frames 29 --steps 20 --gs 6.0 --rounds 2 "${extra[@]}"
  done
}
# Generator: i2v 480p x 121 frames x 20 steps (the model-card recipe), t2v same, t2i 640x640.
#   usage: bench_generator <mstar_port> <vllm_omni_port> <cond_image.jpg>
bench_generator() {
  local mp="$1" vp="$2" img="$3"
  python "$here/video_bench.py" --engine ours --port "$mp" --model cosmos3_edge \
    --tiers 832x480 --frames 121 --steps 20 --gs 6.0 --flow-shift 12.0 --rounds 3 --image "$img"
  python "$here/video_bench.py" --engine vllm --port "$vp" --model nvidia/Cosmos3-Edge \
    --tiers 832x480 --frames 121 --steps 20 --gs 6.0 --flow-shift 12.0 --rounds 3 --image "$img"
  python "$here/video_bench.py" --engine ours --port "$mp" --model cosmos3_edge \
    --tiers 832x480 --frames 121 --steps 20 --gs 6.0 --flow-shift 12.0 --rounds 3
  python "$here/video_bench.py" --engine vllm --port "$vp" --model nvidia/Cosmos3-Edge \
    --tiers 832x480 --frames 121 --steps 20 --gs 6.0 --flow-shift 12.0 --rounds 3
  python "$here/bench_t2i_oai.py" --port "$mp" --model cosmos3_edge        --sizes 640x640 --tag mstar
  python "$here/bench_t2i_oai.py" --port "$vp" --model nvidia/Cosmos3-Edge --sizes 640x640 --tag vllm
}
# Reasoner: TTFT + decode tok/s, image prompt, concurrency 1/8/32.
#   usage: bench_reasoner <mstar_port> <vllm_port> <image.png>
bench_reasoner() {
  local mp="$1" vp="$2" img="$3"
  python "$here/bench_chat_oai.py" --port "$mp" --model cosmos3_edge        --image "$img" --tag mstar
  python "$here/bench_chat_oai.py" --port "$vp" --model nvidia/Cosmos3-Edge --image "$img" --tag vllm
}

case "${1:-}" in
  serve-mstar)         shift; serve_mstar "$@";;
  serve-vllm-omni)     shift; serve_vllm_omni "$@";;
  serve-vllm-reasoner) shift; serve_vllm_reasoner "$@";;
  bench-generator)     shift; bench_generator "$@";;
  bench-reasoner)      shift; bench_reasoner "$@";;
  bench-stream)        shift; bench_stream "$@";;
  *) echo "usage: $0 {serve-mstar <gpu> <port> | serve-vllm-omni <gpu> <port> | serve-vllm-reasoner <gpu> <port> | bench-generator <mp> <vp> <img> | bench-reasoner <mp> <vp> <img> | bench-stream <mp> [img]}";;
esac
