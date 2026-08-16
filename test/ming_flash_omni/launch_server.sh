#!/bin/bash
#
# Launch an M* server hosting Ming-flash-omni-2.0 (inclusionAI), the Ling-2.0
# sparse-MoE omni model. Pair it with the request scripts in this directory:
#
#   test/ming_flash_omni/t2t_request.py   text  -> text   (thinker only)
#   test/ming_flash_omni/a2t_request.py   audio -> text   (audio encoder + thinker)
#   test/ming_flash_omni/t2s_request.py   text  -> speech (thinker + talker)
#
# Usage:  bash test/ming_flash_omni/launch_server.sh
#
# Env (all optional):
#   DEVICES     CUDA_VISIBLE_DEVICES     (default 0,1,2,3,4,5,6,7 — the thinker is TP=8)
#   PORT        HTTP port                (default 8000)
#   MING_CONFIG server yaml              (default configs/ming_flash_omni.yaml;
#                                         use configs/ming_flash_omni_thinker_only.yaml
#                                         for a text-out deploy with no talker)
#   CACHE_DIR   HuggingFace cache dir    (default $HF_HOME — point it at a disk with
#                                         room; the checkpoint is ~238 GB / 42 shards)
#   SCRATCH     scratch root             (default $TMPDIR, else /tmp — sockets, uploads
#                                         and logs live under it; use a roomy disk)
#   STATS_FILE  per-request profiling log (unset = off)
#   TENSOR_PROTOCOL  SHM|TCP|RDMA        (default SHM — single-node colocated ranks.
#                                         The RDMA default needs InfiniBand.)
#
# Sizing: the thinker holds 100B total params (6B active) and is sharded TP=8 by
# configs/ming_flash_omni.yaml — ~40 GB/rank on 80 GB H100s. TP=4 has been observed
# to OOM at ~78.5/80 GB per rank during checkpoint streaming, so 8 GPUs is the
# supported layout. The stateless encoders + the talker are ~1.5 GB each and
# colocate on rank 0.
#
# Note: `mstar serve <name>` (the quickstart path) has a hardcoded model
# allow-list that does not include Ming, so this launcher drives the API server
# with --config directly.
set -euo pipefail

# Anchor to THIS checkout's repo root (test/ming_flash_omni/ -> repo root).
# Prepending it to PYTHONPATH makes ``import mstar`` resolve to this worktree even
# when the venv's editable install points at a different checkout — the normal case
# when several git worktrees share one env. Without this, a server launched from a
# feature worktree silently loads the base checkout's mstar, whose model registry
# lacks this model ("Unknown model name: 'ming_flash_omni'").
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-8000}"
MING_CONFIG="${MING_CONFIG:-$REPO_ROOT/configs/ming_flash_omni.yaml}"
CACHE_DIR="${CACHE_DIR:-${HF_HOME:-}}"
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}}"
STATS_FILE="${STATS_FILE:-}"
TENSOR_PROTOCOL="${TENSOR_PROTOCOL:-SHM}"
WHO="${USER:-mstar}"

# Ming's tokenizer/processor (BailingMM2Processor) ships as remote code in the
# checkpoint repo; transformers loads it with trust_remote_code. MING_CODE_DIR
# overrides the location when a local clone of the model repo is preferred over
# the cached snapshot.
[ -n "${MING_CODE_DIR:-}" ] && export MING_CODE_DIR

# Serving must not block on the network: the processor/config loads lazily via
# HuggingFace ``from_pretrained``, which pings the Hub — that hangs on offline or
# restricted networks. With a populated cache, force HF offline so those loads
# read the local snapshot only. First-ever download: run once with
# HF_HUB_OFFLINE=0 (and HF_TOKEN set if the repo is gated).
if [ -n "$CACHE_DIR" ]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

# 42-shard streaming load at TP=8 fragments the allocator; expandable segments
# keeps the peak under the 80 GB ceiling.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The socket/upload namespace MUST include the port. Two servers that share a
# socket-path prefix share one SHM/ZMQ namespace and silently cross-deliver each
# other's messages ("Message for unknown request <id>") while both clients hang.
INSTANCE="${WHO}_${PORT}"
mkdir -p "$SCRATCH" "$SCRATCH/mstar_sock_$INSTANCE" "$SCRATCH/mstar_uploads_$INSTANCE"

echo "[ming] launching server"
echo "  devices:   $DEVICES     port: $PORT     protocol: $TENSOR_PROTOCOL"
echo "  config:    $MING_CONFIG"
echo "  cache dir: ${CACHE_DIR:-<huggingface default>}"

ARGS=(
  --config "$MING_CONFIG"
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

CUDA_VISIBLE_DEVICES="$DEVICES" \
  python "$REPO_ROOT/mstar/api_server/entrypoint.py" "${ARGS[@]}"
