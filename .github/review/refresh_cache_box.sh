#!/usr/bin/env bash
# Fill one of the review bot's cache volumes from main: the compile caches each model's first server
# start writes, and with run.json's weights true, the model weights too. review-cache.yml launches it
# with `gpu.py launch --run --cache-write ID` on an Eras box, where that volume is /cache, read-write.
# Only main's code runs here, never a PR's; live tests mount the volumes read-only.
#
# With weights, they download straight onto the volume (HF_HOME=/cache/hf), so a deadline keeps every
# file that finished, and the next refresh downloads what is missing, and no new model starts once
# the volume has under 100 GB free; without, they go to the box's own disk and each model's are
# deleted once its server exits. The compile caches are saved to the volume every 5 models and at
# the end. A refresh starts from an empty compile directory, so the tarball holds only what this
# main builds, and writes the main commit and torch version beside it (<tarball>.info).
#
# Inputs in /run/live-pr: box_common.sh, and run.json (repo, main_sha, models: "all" or names, skip: names,
# weights: "true" or "false").
set -uo pipefail

W=/workspace/live-pr IN=/run/live-pr OUT=/workspace/live-pr/out
# shellcheck disable=SC1091
source "$IN/box_common.sh"
mkdir -p "$OUT" "$W/home"
export HOME=$W/home
exec > >(tee -a "$OUT/refresh.log") 2>&1
finish() { echo "$1"; touch "$OUT/done"; await_collection; exit 0; }
mountpoint -q /cache || finish "cache volume not mounted"
# huggingface_hub never resumes a partial download (each attempt writes its own
# <etag>.<random>.incomplete), so those a deadline cut short would only take up room.
[ ! -d /cache/hf/hub ] || find /cache/hf/hub -name '*.incomplete' -delete
# The compile caches go to bootstrap's CACHE_TAR, live-pr-compile-cache beside the HF home: on the
# volume either way.
weights=$(field weights)
if [ "$weights" = true ]; then
  export HF_HOME=/cache/hf
else
  export HF_HOME=$W/hf
  mkdir -p /cache/live-pr-compile-cache && ln -sfn /cache/live-pr-compile-cache "$W/live-pr-compile-cache"
fi

MAIN_SHA=$(field main_sha)
clone_main "$MAIN_SHA"
bash "$W/main/.claude/skills/live-pr-test/scripts/bootstrap.sh" 0 --ref "$MAIN_SHA" --no-main --no-restore \
  --workdir "$W/live-pr" --hf-home "$HF_HOME" || finish "bootstrap failed"
# shellcheck disable=SC1091
source "$W/live-pr/env.sh" && arm pr
export HF_HOME=$HF_PERSIST  # servers download where the weights are kept, even where bootstrap would serve a copy
save_and_stamp() {
  save_caches && printf 'main %s\ntorch %s\n' "$MAIN_SHA" "$(python -c 'import torch; print(torch.__version__)')" \
    > "$CACHE_TAR.info"
}

models=$(field models)
[ "$models" = all ] && models=$(python -c "from mstar.cli.main import DEFAULT_CONFIGS as d; print(' '.join(d))")
n=0 left=""
for m in $models; do
  case " $(field skip) " in *" $m "*) continue ;; esac
  if [ "$weights" = true ] && [ "$(df --output=avail -BG "$HF_HOME" | tail -n 1 | tr -dc 0-9)" -lt 100 ]; then
    left="$left $m"; continue
  fi
  echo "== $m"
  timeout 2400 mstar serve "$m" --port 18000 > "$OUT/serve-$m.log" 2>&1 &
  pid=$!
  for _ in $(seq 240); do
    curl -sf localhost:18000/health > /dev/null && { echo "ready: $m"; break; }
    kill -0 "$pid" 2> /dev/null || { echo "exited: $m (see serve-$m.log)"; break; }
    sleep 10
  done
  kill "$pid" 2> /dev/null; wait "$pid" 2> /dev/null
  [ "$weights" = true ] || rm -rf "$HF_HOME"/hub/models--*  # the box's disk holds one model's at a time
  n=$((n + 1))
  [ $((n % 5)) -ne 0 ] || save_and_stamp
done
[ -z "$left" ] || echo "left out, under 100 GB free on the volume:$left"
save_and_stamp || finish "couldn't save the compile caches"
df -h /cache
finish "refresh done"
