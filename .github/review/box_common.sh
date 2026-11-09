#!/usr/bin/env bash
# Shared by the GPU box scripts, sourced after they set W, IN and OUT. gpu_review.py passes this file
# with `--input` beside run.json. Each script keeps its own `finish`, which is where they differ.

# field NAME: one field of run.json.
field() { python3 -c "import json; print(json.load(open('$IN/run.json'))['$1'])"; }

# The Pod is the only copy of the results: stay up until the runner has them (15 min at most).
await_collection() {
  for _ in $(seq 90); do [ -e "$OUT/collected" ] && break; sleep 10; done
}

# clone_main SHA: main at that commit in $W/main, or `finish` with why.
clone_main() {
  git clone -q "https://github.com/$(field repo).git" "$W/main" && git -C "$W/main" checkout -q "$1" \
    || finish "couldn't check out main"
}
