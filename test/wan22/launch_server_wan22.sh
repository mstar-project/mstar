#!/bin/bash
#
# Launch a single-GPU M* server hosting Wan2.2-TI2V-5B, instrumented for the video
# benchmark: per-request profiling is appended to $STATS_FILE, which
# benchmark/wan22/video_bench_wan22.py parses for the phase breakdown and e2e. It
# adds no GPU sync and is always on.
#
# Usage:  bash test/wan22/launch_server_wan22.sh
#
# Env (all optional):
#   DEVICES     CUDA_VISIBLE_DEVICES     (default 0)
#   PORT        HTTP port                (default 8100)
#   WAN22_CONFIG  server yaml            (default configs/wan22.yaml)
#   SCRATCH     scratch root             (default $TMPDIR, else /tmp — the sockets,
#                                         uploads and logs all live under it, so point
#                                         it at a disk with room if / is tight)
#   STATS_FILE  profiling log path       (default $SCRATCH/wan22_stats.txt)
#   TENSOR_PROTOCOL  SHM|TCP|RDMA        (default SHM — single-GPU colocated)
#
# Note: the Wan2.2-TI2V-5B checkpoint (Wan-AI/Wan2.2-TI2V-5B-Diffusers) is fetched via
# huggingface_hub into $HF_HUB_CACHE on first launch (~32 GB).
set -euo pipefail

DEVICES="${DEVICES:-0}"
PORT="${PORT:-8100}"
WAN22_CONFIG="${WAN22_CONFIG:-configs/wan22.yaml}"
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}}"
STATS_FILE="${STATS_FILE:-$SCRATCH/wan22_stats.txt}"
TENSOR_PROTOCOL="${TENSOR_PROTOCOL:-SHM}"
WHO="${USER:-mstar}"

# The socket/upload namespace MUST include the port. This launcher is documented
# as the way to run one server per GPU, and two servers that share a socket-path
# prefix share one SHM/ZMQ namespace: they then silently cross-deliver each other's
# messages ("Message for unknown request <id>" in the wrong server's log) and both
# clients hang while the GPUs sit idle with the model resident. Keying on $USER
# alone is not enough — the same user launching a second server is exactly the case
# that breaks. Port is unique per server by construction, so it is the right key.
INSTANCE="${WHO}_${PORT}"

mkdir -p "$SCRATCH" "$SCRATCH/mstar_sock_$INSTANCE" "$SCRATCH/mstar_uploads_$INSTANCE"
# Fresh profiling log per launch so the benchmark's offset slicing starts clean.
: > "$STATS_FILE"

echo "[wan22] launching server"
echo "  devices:     $DEVICES     port: $PORT     protocol: $TENSOR_PROTOCOL"
echo "  config:      $WAN22_CONFIG"
echo "  stats file:  $STATS_FILE"

CUDA_VISIBLE_DEVICES="$DEVICES" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python mstar/api_server/entrypoint.py \
    --config "$WAN22_CONFIG" \
    --port "$PORT" \
    --mooncake-port "$((PORT + 1000))" \
    --socket-path-prefix "$SCRATCH/mstar_sock_$INSTANCE/" \
    --upload-dir "$SCRATCH/mstar_uploads_$INSTANCE/" \
    --tensor-comm-protocol "$TENSOR_PROTOCOL" \
    --tcp-transfer-device 0.0.0.0.0 \
    --log-stats-file "$STATS_FILE"
