#!/usr/bin/env bash
# Start, stop and inspect inference servers (M*, vLLM, vllm-omni, SGLang, ...) for
# benchmarking, without the traps that keep recurring:
#   - a second server on a busy port "starts", and its health check is answered by the old one;
#   - a PID file holding a wrapper shell, so stop leaves the real server (and its GPU memory) up;
#   - `pkill -f <pattern>` matching the caller's own shell;
#   - waiting forever on a server that already crashed;
#   - multiprocessing workers orphaned after a hard kill.
#
# usage:
#   server.sh start <name> <port> [--health PATH] [--timeout SEC] [--log FILE] -- <command ...>
#   server.sh stop <name>
#   server.sh status
#   server.sh mps-start <gpu>     # per-user MPS daemon for one physical GPU
#   server.sh mps-stop <gpu>
#
# The command runs in its own session; its PID is recorded from inside it, so it is the
# command itself, not a shell wrapper. A command that is itself a wrapper (`nsys launch`)
# may start the server in yet another session, so readiness checks that the port's listener
# is the command or one of its descendants, and `stop` signals both, escalates to the
# group, then reaps every descendant and anything left in the session.
#
# Environment for the command is inherited: e.g.
#   CUDA_HOME=... PYTHONPATH=$WT server.sh start main 8100 -- mstar serve qwen3 --gpus 0
# State: ${SERVER_STATE_DIR:-/tmp/servers_$USER}/<name>.{pid,port,log,cmd}
set -uo pipefail

STATE=${SERVER_STATE_DIR:-/tmp/servers_$USER}
mkdir -p "$STATE"

die() { echo "server.sh: $*" >&2; exit 1; }

port_holder() {  # prints "pid" of the process listening on $1, if any
  ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}

session_pids() {  # every process whose session id is $1
  ps -e -o pid=,sid= | awk -v s="$1" '$2 == s {print $1}'
}

descendants() {  # every process below $1 (wrappers such as `nsys launch` put the server in a new session)
  ps -e -o pid=,ppid= | awk -v root="$1" '
    { parent[$1] = $2 }
    END { for (p in parent) { q = p; while (q in parent && q != root && q > 1) q = parent[q]; if (q == root && p != root) print p } }'
}

is_ours() {  # is $1 the started process $2 or below it
  [ "$1" = "$2" ] || descendants "$2" | grep -qx "$1"
}

gpu_memory() {
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null
}

cmd_start() {
  local name=$1 port=$2; shift 2
  local health=/health timeout=900 log="$STATE/$name.log"
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    case $1 in
      --health) health=$2; shift 2;;
      --timeout) timeout=$2; shift 2;;
      --log) log=$2; shift 2;;
      *) die "unknown option $1";;
    esac
  done
  [ "${1:-}" = "--" ] || die "missing -- before the command"; shift
  [ $# -gt 0 ] || die "no command"
  if [ -f "$STATE/$name.pid" ] && kill -0 "$(cat "$STATE/$name.pid")" 2>/dev/null; then
    die "'$name' is already running (pid $(cat "$STATE/$name.pid")); stop it first"
  fi
  local holder; holder=$(port_holder "$port")
  if [ -n "$holder" ] || ss -ltnH "sport = :$port" | grep -q .; then
    die "port $port is busy (pid ${holder:-unknown}: $(ps -o args= -p "${holder:-0}" 2>/dev/null | cut -c1-100)). Stop that server first; a health check would be answered by it."
  fi
  rm -f "$STATE/$name.pid"
  printf '%q ' "$@" > "$STATE/$name.cmd"
  echo "$port" > "$STATE/$name.port"
  # setsid: own session and process group. The inner shell records its own PID and then
  # execs the command, so the recorded PID is the server itself.
  setsid bash -c 'echo $$ > "$0"; exec "$@"' "$STATE/$name.pid" "$@" > "$log" 2>&1 < /dev/null &
  for _ in $(seq 50); do [ -s "$STATE/$name.pid" ] && break; sleep 0.1; done
  local pid; pid=$(cat "$STATE/$name.pid" 2>/dev/null) || die "server did not start"
  echo "started '$name' pid $pid, log $log"
  local url="http://127.0.0.1:$port$health" t0=$SECONDS
  while true; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "server.sh: '$name' exited during startup; last lines of $log:" >&2
      tail -30 "$log" >&2
      exit 1
    fi
    if grep -q "address already in use" "$log" 2>/dev/null; then
      echo "server.sh: '$name' could not bind port $port:" >&2
      grep -m3 "address already in use" "$log" >&2
      cmd_stop "$name" >/dev/null
      exit 1
    fi
    if curl -sf -o /dev/null "$url"; then
      holder=$(port_holder "$port")
      if [ -n "$holder" ] && ! is_ours "$holder" "$pid"; then
        echo "server.sh: port $port is answered by pid $holder, which is not '$name' (pid $pid) or below it; stopping '$name'" >&2
        cmd_stop "$name" >/dev/null
        exit 1
      fi
      [ -n "$holder" ] && echo "$holder" > "$STATE/$name.listener"
      echo "'$name' ready on $port after $((SECONDS - t0))s (listener pid ${holder:-?})"
      return 0
    fi
    if [ $((SECONDS - t0)) -ge "$timeout" ]; then
      echo "server.sh: '$name' not healthy after ${timeout}s; stopping it" >&2
      tail -20 "$log" >&2
      cmd_stop "$name" >/dev/null
      exit 1
    fi
    sleep 3
  done
}

cmd_stop() {
  local name=$1
  [ -f "$STATE/$name.pid" ] || die "no server named '$name' (known: $(ls "$STATE" | sed -n 's/\.pid$//p' | tr '\n' ' '))"
  local pid port listener roots tree r
  pid=$(cat "$STATE/$name.pid"); port=$(cat "$STATE/$name.port" 2>/dev/null)
  listener=$(cat "$STATE/$name.listener" 2>/dev/null)
  # The started process and the actual server. They differ under a wrapper such as
  # `nsys`, which may not forward SIGINT and may already have exited, leaving the server
  # re-parented to init: each is signalled, waited on and reaped in its own right.
  roots=$(for r in $pid $listener; do kill -0 "$r" 2>/dev/null && echo "$r"; done | sort -u)
  tree=$(for r in $roots; do descendants "$r"; done)  # before signalling, while parent links exist
  alive() { for r in $roots; do kill -0 "$r" 2>/dev/null && return 0; done; return 1; }
  if [ -n "$roots" ]; then
    kill -INT $roots 2>/dev/null
    for _ in $(seq 60); do alive || break; sleep 1; done
    if alive; then
      echo "'$name' ignored SIGINT for 60s; SIGTERM to its process groups"
      for r in $roots; do kill -TERM -- "-$(ps -o pgid= -p "$r" | tr -d ' ')" 2>/dev/null; done
      for _ in $(seq 20); do alive || break; sleep 1; done
    fi
  fi
  # whatever is left of the trees or the sessions: workers, compile pools, a wrapper's children
  local left
  left=$(for p in $roots $tree $(for r in $pid $listener; do session_pids "$r"; done); do
    kill -0 "$p" 2>/dev/null && echo "$p"; done | sort -u)
  if [ -n "$left" ]; then
    echo "reaping leftover processes of '$name': $(echo $left)"
    kill -KILL $left 2>/dev/null
    sleep 2
  fi
  if [ -n "$port" ]; then
    for _ in $(seq 20); do ss -ltnH "sport = :$port" | grep -q . || break; sleep 1; done
    if ss -ltnH "sport = :$port" | grep -q .; then
      die "port $port is still held by pid $(port_holder "$port") after stopping '$name'"
    fi
  fi
  rm -f "$STATE/$name.pid" "$STATE/$name.listener"
  echo "stopped '$name'"
  gpu_memory
}

cmd_status() {
  local f name pid port
  for f in "$STATE"/*.pid; do
    [ -e "$f" ] || { echo "no servers recorded in $STATE"; break; }
    name=$(basename "$f" .pid); pid=$(cat "$f"); port=$(cat "$STATE/$name.port" 2>/dev/null)
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name: pid $pid port $port up ($(session_pids "$pid" | wc -l) processes) :: $(cat "$STATE/$name.cmd")"
    else
      echo "$name: pid $pid port $port DEAD (stale record) :: $(cat "$STATE/$name.cmd")"
    fi
  done
  gpu_memory
}

# MPS lets several processes on one GPU (M*'s per-node workers, vllm-omni's stages) run
# kernels concurrently instead of time-slicing. A client must use the daemon's pipe
# directory, and sees the daemon's GPU as device 0 whatever its physical index.
mps_dir() { echo "/tmp/mps_${USER}_gpu$1"; }

cmd_mps_start() {
  local gpu=$1 d; d=$(mps_dir "$gpu")
  mkdir -p "$d/pipe" "$d/log"
  if CUDA_MPS_PIPE_DIRECTORY="$d/pipe" bash -c 'echo get_server_list | nvidia-cuda-mps-control' >/dev/null 2>&1; then
    echo "MPS daemon for GPU $gpu already running"
  else
    CUDA_VISIBLE_DEVICES=$gpu CUDA_MPS_PIPE_DIRECTORY="$d/pipe" CUDA_MPS_LOG_DIRECTORY="$d/log" \
      nvidia-cuda-mps-control -d || die "could not start the MPS daemon"
    echo "MPS daemon started for physical GPU $gpu"
  fi
  echo "Clients need (and see this GPU as device 0):"
  echo "  export CUDA_MPS_PIPE_DIRECTORY=$d/pipe CUDA_MPS_LOG_DIRECTORY=$d/log CUDA_VISIBLE_DEVICES=0"
}

cmd_mps_stop() {
  local gpu=$1 d; d=$(mps_dir "$gpu")
  CUDA_MPS_PIPE_DIRECTORY="$d/pipe" bash -c 'echo quit | nvidia-cuda-mps-control' && echo "MPS daemon for GPU $gpu stopped"
}

case ${1:-} in
  start) shift; [ $# -ge 2 ] || die "usage: start <name> <port> [opts] -- <command>"; cmd_start "$@";;
  stop) [ $# -eq 2 ] || die "usage: stop <name>"; cmd_stop "$2";;
  status) cmd_status;;
  mps-start) [ $# -eq 2 ] || die "usage: mps-start <gpu>"; cmd_mps_start "$2";;
  mps-stop) [ $# -eq 2 ] || die "usage: mps-stop <gpu>"; cmd_mps_stop "$2";;
  *) sed -n '2,24p' "$0"; exit 1;;
esac
