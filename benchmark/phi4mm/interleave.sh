#!/bin/bash
# Two servers resident at once, measured in turn (A B B A ...), so a drift in the
# host's speed lands on both. Before each run, a busy loop pinned to one core reads the
# clock a busy core gets, since a shared host can withhold turbo from sustained work.
#
# usage: benchmark/phi4mm/interleave.sh <engine A> <url A> <engine B> <url B> <workload> \
#            <out.jsonl> [rounds=2] [concurrency=32] [requests=8*concurrency]
#   engines: mstar | vllm. Media offsets advance per run (OFFSET=<first>), so an engine
#   that caches encoder outputs never sees an image or clip twice. CLOCK_CPU picks the
#   probe's core (default 100).
set -eu
EA=$1 UA=$2 EB=$3 UB=$4 W=$5 OUT=$6 R=${7:-2} C=${8:-32}
N=${9:-$((C * 8))} CPU=${CLOCK_CPU:-100} OFF=${OFFSET:-0}
HERE=$(cd "$(dirname "$0")" && pwd)

clock_mhz() {
  taskset -c "$CPU" python3 -c 'import time; t = time.time()
while time.time() - t < 1.5: pass' &
  local pid=$!
  sleep 1
  echo $(($(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_cur_freq) / 1000))
  wait $pid
}

run() {
  local eng=$1 url=$2 mhz
  mhz=$(clock_mhz)
  python "$HERE/bench.py" --url "$url" --engine "$eng" --workload "$W" --concurrency "$C" \
    --requests "$N" --warmup "$C" --output-len 128 --offset "$OFF" 2>&1 | tail -1 |
    python3 -c "import json, sys
d = json.loads(sys.stdin.read()); d['busy_core_mhz'] = $mhz; d['arm'] = '$eng@$url'
open('$OUT', 'a').write(json.dumps(d) + '\n')
print(d['arm'], d['workload'], 'tok/s', d['output_tok_s'], 'itl p50', d['itl_ms'].get('p50'),
      'ttft p50', d['ttft_ms'].get('p50'), 'failed', d['failed'], 'tokens', d['mean_output_tokens'],
      'SHORT' if d['short_output'] else '', 'busy core', $mhz, 'MHz')"
  OFF=$((OFF + N + C))
}

for r in $(seq "$R"); do
  if [ $((r % 2)) = 1 ]; then run "$EA" "$UA"; run "$EB" "$UB"; else run "$EB" "$UB"; run "$EA" "$UA"; fi
done
python3 - "$OUT" "$W" <<'PY'
import json, statistics, sys
from collections import defaultdict
by = defaultdict(list)
for line in open(sys.argv[1]):
    d = json.loads(line)
    if d["workload"] == sys.argv[2]:
        by[d["arm"]].append(d["output_tok_s"])
for arm, xs in by.items():
    print(arm, sys.argv[2], "runs", len(xs), "median", statistics.median(xs), "min", min(xs), "max", max(xs))
PY
