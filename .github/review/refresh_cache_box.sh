#!/usr/bin/env bash
# Fill the review bot's cache volumes from main: model weights, and the compile caches each model's
# first server start writes. review-cache.yml launches it with `gpu.py launch --run --cache rw` on
# an Eras box, where the volumes are /cache/1, /cache/2, ... read-write. Only main's code runs here,
# never a PR's; live tests mount the volumes read-only.
#
# Inputs in /run/live-pr: run.json (repo, main_sha, models: "all" or names, skip: names).
set -uo pipefail

W=/workspace/live-pr IN=/run/live-pr OUT=/workspace/live-pr/out
mkdir -p "$OUT" "$W/home" "$W/live-pr-compile-cache"
export HOME=$W/home HF_HOME=$W/hf
exec > >(tee -a "$OUT/refresh.log") 2>&1
field() { python3 -c "import json; print(json.load(open('$IN/run.json'))['$1'])"; }
finish() {  # the Pod is the only copy of the log: stay up until the runner has it (15 min at most)
  echo "$1"; touch "$OUT/done"
  for _ in $(seq 90); do [ -e "$OUT/collected" ] && break; sleep 10; done
  exit 0
}

# What the volumes hold already, linked in, so only what's missing downloads.
mkdir -p "$HF_HOME/hub"
for m in /cache/*/hub/models--*; do [ -e "$HF_HOME/hub/${m##*/}" ] || cp -rs "$m" "$HF_HOME/hub/"; done
had=$(ls "$HF_HOME/hub")
cp /cache/1/live-pr-compile-cache/*.tar "$W/live-pr-compile-cache/" 2> /dev/null

MAIN_SHA=$(field main_sha)
git clone -q "https://github.com/$(field repo).git" "$W/main" && git -C "$W/main" checkout -q "$MAIN_SHA" \
  || finish "couldn't check out main"
bash "$W/main/.claude/skills/live-pr-test/scripts/bootstrap.sh" 0 --ref "$MAIN_SHA" --no-main \
  --workdir "$W/live-pr" --hf-home "$HF_HOME" || finish "bootstrap failed"
# shellcheck disable=SC1091
source "$W/live-pr/env.sh" && arm pr

models=$(field models)
[ "$models" = all ] && models=$(python -c "from mstar.cli.main import DEFAULT_CONFIGS as d; print(' '.join(d))")
for m in $models; do
  case " $(field skip) " in *" $m "*) continue ;; esac
  echo "== $m"
  timeout 2400 mstar serve "$m" --port 18000 > "$OUT/serve-$m.log" 2>&1 &
  pid=$!
  for _ in $(seq 240); do
    curl -sf localhost:18000/health > /dev/null && { echo "ready: $m"; break; }
    kill -0 "$pid" 2> /dev/null || { echo "exited: $m (see serve-$m.log)"; break; }
    sleep 10
  done
  kill "$pid" 2> /dev/null; wait "$pid" 2> /dev/null
done

save_caches && mkdir -p /cache/1/live-pr-compile-cache && cp "$W"/live-pr-compile-cache/*.tar /cache/1/live-pr-compile-cache/
# Each newly downloaded model goes to the volume with the most room.
for d in "$HF_HOME"/hub/models--*; do
  case $'\n'"$had"$'\n' in *$'\n'"${d##*/}"$'\n'*) continue ;; esac
  best=$(df --output=avail,target /cache/* | tail -n +2 | sort -n | tail -n 1 | awk '{print $2}')
  mkdir -p "$best/hub" && cp -a "$d" "$best/hub/" && echo "cached ${d##*/} on $best"
done
df -h /cache/*
finish "refresh done"
