#!/usr/bin/env bash
# TP async scheduling A/B on coriander: qwen3omni thinker TP2 (GPUs 0,1,2),
# arm A = MSTAR_TP_ASYNC_SCHED=0 (serial baseline), arm B = =1.
#
# Per arm: serve from THIS worktree (PYTHONPATH pins the import — the shared
# venv's editable install points at the primary checkout, which is another
# branch), wait for /health, (1) probe 4 fixed prompts greedily and save the
# text (bit-identity check across arms), (2) run the team benchmark client
# (text_to_text, B=1, N requests), teardown, verify GPUs are free. Never
# starts on a busy box; hands the GPUs back on every exit path.
#
# Usage (on the box):  bash env/tp-async-ab.sh [A|B|AB]     (default AB)
# Env knobs: NREQ (bench requests, default 20), GPUS (default 0,1,2),
#            CFG (default configs/qwen3omni_thinker_tp2.yaml), TAG (label).
set -uo pipefail

P=/m-coriander/coriander/kirill
WT=$P/mstar-tp
D=$P/tp-async
mkdir -p "$D" "$P/logs" "$P/tmp/mstar-sockets" "$P/tmp/mstar-uploads"
ARMS=${1:-AB}
NREQ=${NREQ:-20}
GPUS=${GPUS:-0,1,2}
CFG=${CFG:-configs/qwen3omni_thinker_tp2.yaml}
TAG=${TAG:-$(date -u +%Y%m%dT%H%M%S)}
RUNLOG=$D/ab-$TAG.log
say() { echo "[tp-ab] $(date -u +%H:%M:%S) $*" | tee -a "$RUNLOG"; }

export HOME=$P HF_HOME=/m-coriander/coriander/hf HF_HUB_OFFLINE=1
export TMPDIR=$P/tmp XDG_CACHE_HOME=$P/.cache
export TORCHINDUCTOR_CACHE_DIR=$P/.cache/torchinductor TRITON_CACHE_DIR=$P/.cache/triton
export LD_PRELOAD=/usr/lib64/libcudnn.so.9
export MSTAR_DIST_TIMEOUT_S=7200 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export PATH="$P/mstar/.venv/bin:$PATH"
export PYTHONPATH="$WT"
PY=$P/mstar/.venv/bin/python
PORT=${PORT:-8017}

cd "$WT" || exit 2
say "worktree $(git rev-parse --short HEAD) $(git log --oneline -1 | cut -c10-80)"
say "import check: $($PY -c 'import mstar,sys; print(mstar.__file__)')"
case "$($PY -c 'import mstar; print(mstar.__file__)')" in
  "$WT"/*) ;;
  *) say "ABORT: mstar imports from the wrong tree"; exit 2;;
esac

kirill_gpu_pids() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d " ")" = "kirill" ] && echo "$p"
  done
}
busy_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v g="$GPUS" 'BEGIN{n=split(g,a,",");for(i=1;i<=n;i++)w[a[i]]=1} ($1 in w)&&$2>1000{c++} END{print c+0}'
}
teardown() {
  local pgid=${1:-}
  [ -n "$pgid" ] && { kill -TERM -"$pgid" 2>/dev/null; sleep 12; kill -9 -"$pgid" 2>/dev/null; }
  pkill -9 -u kirill -f "[m]star-serve.*--port $PORT" 2>/dev/null
  for i in $(seq 1 12); do
    pids=$(kirill_gpu_pids)
    [ -z "$pids" ] && { say "teardown clean (${i}0s)"; return 0; }
    echo "$pids" | xargs -r kill -9 2>/dev/null; sleep 10
  done
  say "teardown DIRTY - check GPUs by hand"; return 1
}

probe() {  # $1 = out file
  $PY - "$1" "$PORT" <<'PYEOF'
import json, sys, urllib.request
out, port = sys.argv[1], sys.argv[2]
prompts = [
    "What is the capital of France? Talk about the city.",
    "Explain the difference between machine learning and deep learning.",
    "Write 5 haikus about autumn leaves.",
    "List the planets of the solar system in order and one fact about each.",
]
res = []
for p in prompts:
    body = json.dumps({
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": p}],
        "temperature": 0.0, "max_tokens": 128, "stream": False,
        "modalities": ["text"],
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.load(r)
    txt = j["choices"][0]["message"]["content"]
    usage = j.get("usage", {})
    res.append({"prompt": p, "text": txt, "usage": usage})
    print(f"[probe] {usage.get('completion_tokens','?')} tok: {txt[:80]!r}")
json.dump(res, open(out, "w"), indent=1)
PYEOF
}

run_arm() {  # $1 = A|B
  local arm=$1 flag
  [ "$arm" = "B" ] && flag=1 || flag=0
  local b; b=$(busy_gpus)
  if [ "$b" -gt 0 ]; then say "ABORT arm $arm: $b of GPUs {$GPUS} busy"; nvidia-smi >> "$RUNLOG"; return 3; fi
  local ts; ts=$(date -u +%Y%m%dT%H%M%S)
  local log=$P/logs/mstar-tpasync-$arm-$ts.log
  { date -u; nvidia-smi; who; free -g | head -2; git -C "$WT" log --oneline -1; echo "MSTAR_TP_ASYNC_SCHED=$flag CFG=$CFG GPUS=$GPUS"; } > "$D/$arm-$TAG.provenance"
  say "arm $arm: MSTAR_TP_ASYNC_SCHED=$flag serve $CFG on GPUs $GPUS -> $log"
  CUDA_VISIBLE_DEVICES=$GPUS MSTAR_TP_ASYNC_SCHED=$flag MSTAR_PHASE_TIMING=${MSTAR_PHASE_TIMING:-200} \
    setsid $P/mstar/.venv/bin/mstar-serve \
      --config "$CFG" --tensor-comm-protocol SHM \
      --socket-path-prefix "$P/tmp/mstar-sockets/" --upload-dir "$P/tmp/mstar-uploads/" \
      --timeout 7200 --host 0.0.0.0 --port "$PORT" > "$log" 2>&1 &
  local pgid=$!
  local ready=""
  for i in $(seq 1 240); do
    sleep 10
    if curl -sf "localhost:$PORT/health" > /dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$pgid" 2>/dev/null; then say "arm $arm FAILED: server exited during load (see $log)"; teardown "$pgid"; return 1; fi
  done
  if [ -z "$ready" ]; then say "arm $arm FAILED: readiness timeout"; teardown "$pgid"; return 1; fi
  say "arm $arm READY after ~$((i*10))s; probing"
  if ! probe "$D/probe-$arm-$TAG.json" >> "$RUNLOG" 2>&1; then say "arm $arm probe FAILED"; fi
  say "arm $arm bench: text_to_text B=1 N=$NREQ"
  if $PY -m benchmark.runner --model qwen3omni --request-type text_to_text \
      --dataset text --request-txt-file benchmark/assets/simple_text_queries.txt \
      --profiling-type offline --batch-size 1 --num-requests "$NREQ" --num-warmup 3 \
      --local-cache "$P/tmp/bench-cache" --url "http://localhost:$PORT" \
      --inference-system ours > "$D/bench-$arm-$TAG.txt" 2>&1; then
    say "arm $arm bench done"
  else
    say "arm $arm bench FAILED (see $D/bench-$arm-$TAG.txt)"
  fi
  # serve-side evidence: eager/tripwires + phase timing + follow-spec activity
  {
    echo "== $arm: grep summary =="
    echo "phase-timing lines: $(grep -c 'phase-timing' "$log")"
    grep 'phase-timing' "$log" | tail -2
    echo "Follow-speculating: $(grep -c 'Follow-speculating' "$log")   dropped-head: $(grep -c 'dropped speculative head' "$log")   Tracebacks: $(grep -c Traceback "$log")"
  } | tee -a "$RUNLOG"
  teardown "$pgid"
}

case "$ARMS" in
  A)  run_arm A ;;
  B)  run_arm B ;;
  AB) run_arm A; run_arm B ;;
  BA) run_arm B; run_arm A ;;
  *) echo "usage: $0 [A|B|AB|BA]"; exit 2;;
esac

say "== summary =="
for arm in A B; do
  f=$D/bench-$arm-$TAG.txt
  [ -f "$f" ] && { echo "--- $arm ---"; grep -iE "throughput|tok/s|tokens|latency|TTFT|ITL|p50|mean" "$f" | head -20; } | tee -a "$RUNLOG"
done
if [ -f "$D/probe-A-$TAG.json" ] && [ -f "$D/probe-B-$TAG.json" ]; then
  if $PY - "$D/probe-A-$TAG.json" "$D/probe-B-$TAG.json" <<'PYEOF'
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
same = all(x["text"] == y["text"] for x, y in zip(a, b))
for i, (x, y) in enumerate(zip(a, b)):
    print(f"[probe-diff] prompt {i}: {'IDENTICAL' if x['text']==y['text'] else 'DIFFERENT'} "
          f"({x['usage'].get('completion_tokens')} vs {y['usage'].get('completion_tokens')} tok)")
sys.exit(0 if same else 1)
PYEOF
  then say "PROBE: arms A and B produced IDENTICAL text on all prompts"; else say "PROBE: arms DIFFER (see probe json)"; fi
fi
say "done: $RUNLOG"
