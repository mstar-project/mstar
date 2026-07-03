#!/bin/bash
# Two-"host" loopback rig: runs mstar-serve (head) and one mstar-node agent on
# this machine, serves two pinned-seed requests across the host split, then
# kills the remote worker and checks the failure surfaces as a fast HTTP 500.
#
# Usage:
#   test/multinode/sim_two_hosts.sh [head_gpu] [agent_gpu] [port]
#
# Requirements: BAGEL weights in the local HF cache, and a tensor transport
# that works on this machine (the rig uses Mooncake TCP; on hosts without an
# active RDMA fabric the transfer engine needs the explicit device below).
set -u
HEAD_GPU="${1:-0}"; AGENT_GPU="${2:-$HEAD_GPU}"; PORT="${3:-8172}"
TCP_DEVICE="${MSTAR_TEST_TCP_DEVICE:-0.0.0.0.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CFG="$HERE/bagel_pd_two_hosts_sim.yaml"
WORKDIR="$(mktemp -d /tmp/mstar_sim2h_XXXX)"
HEAD_LOG=$WORKDIR/head.log
AGENT_LOG=$WORKDIR/agent.log

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then echo "port $PORT busy"; exit 3; fi

setsid env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" CUDA_VISIBLE_DEVICES="$HEAD_GPU" \
  mstar-serve --config "$CFG" --host 127.0.0.1 --port "$PORT" \
  --socket-path-prefix "$WORKDIR/head_sockets" --upload-dir "$WORKDIR/uploads" \
  --tensor-comm-protocol TCP --tcp-transfer-device "$TCP_DEVICE" \
  > "$HEAD_LOG" 2>&1 &
HEAD=$!

setsid env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" CUDA_VISIBLE_DEVICES="$AGENT_GPU" \
  mstar-node --config "$CFG" --node-rank 1 > "$AGENT_LOG" 2>&1 &
AGENT=$!

cleanup() {
  kill -INT -- -$HEAD 2>/dev/null; kill -INT -- -$AGENT 2>/dev/null
  sleep 6
  kill -9 -- -$HEAD 2>/dev/null; kill -9 -- -$AGENT 2>/dev/null
  echo "logs kept in $WORKDIR"
}
trap cleanup EXIT

for i in $(seq 1 80); do
  curl -s --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q healthy && break
  grep -qE "Traceback" "$HEAD_LOG" && { echo "head failed:"; tail -6 "$HEAD_LOG"; exit 1; }
  grep -qE "Traceback" "$AGENT_LOG" && { echo "agent failed:"; tail -6 "$AGENT_LOG"; exit 1; }
  kill -0 $HEAD 2>/dev/null || { echo "head exited:"; tail -6 "$HEAD_LOG"; exit 1; }
  [ "$i" = 80 ] && { echo "startup timeout"; tail -4 "$HEAD_LOG" "$AGENT_LOG"; exit 1; }
  sleep 10
done
echo "server ready"
grep -m1 -o "Node agent 1 joined.*" "$HEAD_LOG"

python - "$PORT" <<'EOF' || exit 1
import base64, hashlib, sys, requests
port = sys.argv[1]
for rid, text in [
    ("sim2h-1", "Describe a sunset over the ocean in two sentences."),
    ("sim2h-2", "What is the capital of France? Answer in one word."),
]:
    r = requests.post(f"http://127.0.0.1:{port}/generate", timeout=180, data={
        "text": text, "output_modalities": "text",
        "streaming": "false", "request_id": rid,
    })
    r.raise_for_status()
    payload = b"".join(base64.b64decode(c["data"]) for c in r.json()["outputs"].get("text", []))
    assert payload, f"{rid}: empty output"
    print(f"{rid}: bytes={len(payload)} sha256={hashlib.sha256(payload).hexdigest()[:16]} "
          f"{payload[:60].decode('utf-8', 'replace')!r}")
EOF

WPID=$(pgrep -g $AGENT -f spawn_main | head -1)
echo "killing remote worker pid=$WPID"
kill -9 "$WPID"
sleep 3
code=$(curl -s -o "$WORKDIR/kill_probe.json" -w "%{http_code}" --max-time 60 -X POST \
  "http://127.0.0.1:$PORT/generate" \
  -F text='hello again' -F output_modalities=text -F streaming=false \
  -F request_id=sim2h-post-kill)
echo "post-kill request: http=$code $(head -c 160 "$WORKDIR/kill_probe.json")"
[ "$code" = "500" ] || { echo "expected HTTP 500 after worker death"; exit 1; }

echo "sim_two_hosts: OK"
