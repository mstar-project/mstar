#!/usr/bin/env bash
# Fill one of the review bot's cache volumes from main: the compile caches each model's first server
# start writes, and with run.json's weights true, the model weights too. review-cache.yml launches it with `gpu.py launch --run
# --cache-write ID` on an Eras box, where that volume is /cache, read-write. Only main's code runs
# here, never a PR's; live tests mount the volumes read-only.
#
# With weights, they download straight onto the volume (HF_HOME=/cache/hf), so a deadline keeps every
# file that finished, and the next refresh downloads what is missing; without, they go to the box's
# own disk and are lost with it. The compile caches are saved to the volume every 5 models and at
# the end. No new model starts once the weights' disk has under 100 GB free.
#
# Inputs in /run/live-pr: run.json (repo, main_sha, models: "all" or names, skip: names,
# weights: "true" or "false").
set -uo pipefail

W=/workspace/live-pr IN=/run/live-pr OUT=/workspace/live-pr/out
mkdir -p "$OUT" "$W/home"
export HOME=$W/home
exec > >(tee -a "$OUT/refresh.log") 2>&1
field() { python3 -c "import json; print(json.load(open('$IN/run.json'))['$1'])"; }
finish() {  # the Pod is the only copy of the log: stay up until the runner has it (15 min at most)
  echo "$1"; touch "$OUT/done"
  for _ in $(seq 90); do [ -e "$OUT/collected" ] && break; sleep 10; done
  exit 0
}
mountpoint -q /cache || finish "cache volume not mounted"
# huggingface_hub never resumes a partial download (each attempt writes its own
# <etag>.<random>.incomplete), so those a deadline cut short would only take up room.
[ ! -d /cache/hf/hub ] || find /cache/hf/hub -name '*.incomplete' -delete
# The compile caches go to bootstrap's CACHE_TAR, live-pr-compile-cache beside the HF home: on the
# volume either way.
if [ "$(field weights)" = true ]; then
  export HF_HOME=/cache/hf
else
  export HF_HOME=$W/hf
  mkdir -p /cache/live-pr-compile-cache && ln -sfn /cache/live-pr-compile-cache "$W/live-pr-compile-cache"
fi

MAIN_SHA=$(field main_sha)
git clone -q "https://github.com/$(field repo).git" "$W/main" && git -C "$W/main" checkout -q "$MAIN_SHA" \
  || finish "couldn't check out main"
bash "$W/main/.claude/skills/live-pr-test/scripts/bootstrap.sh" 0 --ref "$MAIN_SHA" --no-main \
  --workdir "$W/live-pr" --hf-home "$HF_HOME" || finish "bootstrap failed"
# shellcheck disable=SC1091
source "$W/live-pr/env.sh" && arm pr
export HF_HOME=$HF_PERSIST  # servers download where the weights are kept, even where bootstrap would serve a copy

models=$(field models)
[ "$models" = all ] && models=$(python -c "from mstar.cli.main import DEFAULT_CONFIGS as d; print(' '.join(d))")
n=0 left=""
for m in $models; do
  case " $(field skip) " in *" $m "*) continue ;; esac
  if [ "$(df --output=avail -BG "$HF_HOME" | tail -n 1 | tr -dc 0-9)" -lt 100 ]; then left="$left $m"; continue; fi
  echo "== $m"
  timeout 2400 mstar serve "$m" --port 18000 > "$OUT/serve-$m.log" 2>&1 &
  pid=$!
  for _ in $(seq 240); do
    curl -sf localhost:18000/health > /dev/null && { echo "ready: $m"; break; }
    kill -0 "$pid" 2> /dev/null || { echo "exited: $m (see serve-$m.log)"; break; }
    sleep 10
  done
  kill "$pid" 2> /dev/null; wait "$pid" 2> /dev/null
  n=$((n + 1))
  [ $((n % 5)) -ne 0 ] || save_caches
done
[ -z "$left" ] || echo "left out, under 100 GB free for weights:$left"
save_caches || finish "couldn't save the compile caches"
df -h /cache
finish "refresh done"
