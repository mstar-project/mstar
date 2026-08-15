#!/bin/bash
#
# Launch a single-GPU M* server hosting LingBot-Video dense-1.3B text-to-video.
# Pair it with test/lingbot/t2v_request.sh to generate a clip.
#
# Usage:  bash test/lingbot/launch_server_lingbot.sh
#
# Env (all optional):
#   DEVICES     CUDA_VISIBLE_DEVICES     (default 0)
#   PORT        HTTP port                (default 8200)
#   LINGBOT_CONFIG  server yaml          (default configs/lingbot.yaml)
#   CACHE_DIR   HuggingFace cache dir    (default $HF_HOME — point it at a disk with
#                                         room; the dense checkpoint is tens of GB)
#   SCRATCH     scratch root             (default $TMPDIR, else /tmp — sockets, uploads
#                                         and logs live under it; use a roomy disk)
#   STATS_FILE  per-request profiling log (unset = off)
#   TENSOR_PROTOCOL  SHM|TCP|RDMA        (default SHM — single-GPU colocated)
#
# Note: LingBot has no `mstar serve lingbot` default-config entry yet, so this
# launcher drives entrypoint.py with --config directly (as the wan22 launcher does).
# The checkpoint (robbyant/lingbot-video-dense-1.3b) is fetched via huggingface_hub
# into the cache on first launch. The first request is slow (lazy torch.compile).
set -euo pipefail

# Anchor to THIS checkout's repo root (test/lingbot/ -> repo root). Prepending it
# to PYTHONPATH makes ``import mstar`` resolve to this worktree even when the venv's
# editable install points at a different checkout — the normal case when several
# git worktrees share one env. Without this, a server launched from a feature
# worktree silently loads the base checkout's mstar (its model registry lacks this
# model -> "Unknown model name").
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

DEVICES="${DEVICES:-0}"
PORT="${PORT:-8200}"
LINGBOT_CONFIG="${LINGBOT_CONFIG:-$REPO_ROOT/configs/lingbot.yaml}"
CACHE_DIR="${CACHE_DIR:-${HF_HOME:-}}"
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}}"
STATS_FILE="${STATS_FILE:-}"
TENSOR_PROTOCOL="${TENSOR_PROTOCOL:-SHM}"
WHO="${USER:-mstar}"

# Serving must not block on the network. The text-encoder processor / config loads
# lazily on the first request via HuggingFace ``from_pretrained``, which pings the
# Hub — that hangs on offline/restricted networks and is throttled when
# unauthenticated (the request then appears to "never finish"). When the checkpoint
# is already cached (CACHE_DIR given) force HF offline so those loads use the local
# cache only. First-ever download: run once with ``HF_HUB_OFFLINE=0`` (and no
# CACHE_DIR, or an empty one) to populate the cache, or set ``HF_TOKEN``.
if [ -n "$CACHE_DIR" ]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

# The socket/upload namespace MUST include the port. Two servers that share a
# socket-path prefix share one SHM/ZMQ namespace and silently cross-deliver each
# other's messages ("Message for unknown request <id>") while both clients hang.
# Port is unique per server by construction, so it is the right key.
INSTANCE="${WHO}_${PORT}"

mkdir -p "$SCRATCH" "$SCRATCH/mstar_sock_$INSTANCE" "$SCRATCH/mstar_uploads_$INSTANCE"

echo "[lingbot] launching server"
echo "  devices:   $DEVICES     port: $PORT     protocol: $TENSOR_PROTOCOL"
echo "  config:    $LINGBOT_CONFIG"
echo "  cache dir: ${CACHE_DIR:-<huggingface default>}"

ARGS=(
  --config "$LINGBOT_CONFIG"
  --port "$PORT"
  --mooncake-port "$((PORT + 1000))"
  --socket-path-prefix "$SCRATCH/mstar_sock_$INSTANCE/"
  --upload-dir "$SCRATCH/mstar_uploads_$INSTANCE/"
  --tensor-comm-protocol "$TENSOR_PROTOCOL"
)
[ -n "$CACHE_DIR" ] && ARGS+=(--cache-dir "$CACHE_DIR")
[ -n "${LOG_LEVEL:-}" ] && ARGS+=(--log-level "$LOG_LEVEL")
if [ -n "$STATS_FILE" ]; then
  : > "$STATS_FILE"                       # fresh log per launch
  ARGS+=(--log-stats-file "$STATS_FILE")
  echo "  stats:     $STATS_FILE"
fi

# Match the reference pipeline's attention backend (its inference scripts export
# DIFFUSERS_ATTN_BACKEND=_native_flash); the diffusers default "native" (SDPA)
# drifts the denoise trajectory slightly vs the upstream baseline. Fall back to the
# default if this diffusers build doesn't know the name (an invalid value raises at
# import — the server would never start and requests would hang).
ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-_native_flash}"
if ! python -c "from diffusers.models.attention_dispatch import AttentionBackendName as B; B('$ATTN_BACKEND')" >/dev/null 2>&1; then
  echo "[lingbot] attention backend '$ATTN_BACKEND' not available in this diffusers build; using the default" >&2
  ATTN_BACKEND=""
fi
[ -n "$ATTN_BACKEND" ] && export DIFFUSERS_ATTN_BACKEND="$ATTN_BACKEND"

CUDA_VISIBLE_DEVICES="$DEVICES" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python "$REPO_ROOT/mstar/api_server/entrypoint.py" "${ARGS[@]}"
