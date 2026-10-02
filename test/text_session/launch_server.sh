#!/bin/bash
# BAGEL's LLM, text only, with persistent sessions on its KV. One GPU.
#
#   CUDA_VISIBLE_DEVICES=0 bash test/text_session/launch_server.sh
#
# Then drive a session with:  python test/text_session/session_request.py
set -euo pipefail

PORT=${PORT:-8000}
WHO=${WHO:-$(whoami)}
DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# A prefix of its own: --port alone does not isolate the ZMQ IPC sockets, so two
# servers on one host steal each other's handshake.
PREFIX=${SOCKET_PREFIX:-/tmp/mstar_${WHO}_session/}

# $PYTHON so a shell without the venv activated does not fall back to a system
# interpreter that has none of the dependencies.
PYTHON=${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}
PYTHON=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}

CUDA_VISIBLE_DEVICES=$DEVICES "$PYTHON" -m mstar.api_server.entrypoint \
    --config configs/test_text_session.yaml \
    --port "$PORT" \
    --socket-path-prefix "$PREFIX" \
    --upload-dir "/tmp/mstar_uploads_${WHO}/" \
    --tensor-comm-protocol SHM \
    "$@"
