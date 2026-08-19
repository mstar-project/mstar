#!/usr/bin/env bash
# GLM-5.2 MTP perf A/B — every arm on ONE commit, in ONE box window.
#
#   arm A  defaults                       (pre-final-norm pairing, no sync capture)
#   arm B  MSTAR_GLM52_MTP_PAIR_POSTNORM=1 (vLLM's trunk-pairing convention)
#   arm C  MSTAR_GLM52_MTP_CAPTURE_SYNC=1  (padded sync-pass piecewise graph)
#
# Each arm differs from arm A by exactly ONE env var, so each delta is a single
# lever. Arm A runs first and does double duty: it is also the check that the
# text_inputs fix (5719fa2b) restored the baseline — expect ~49.65 tok/s and
# 3264 tokens (the buggy 08-10 runs read 44.35/3192 and 33.02/3264).
#
# WHY THIS REPLACES glm-m3/m3-mtp-pairnorm-ab.sh
# That script runs only the POSTNORM arm and compares it against a STORED
# arm-A number (bench-mstar-b1-mtp-k2-prenorm.txt: 49.65, measured 2026-08-09
# on eb0f3ea2). Eight commits have landed since, including the text_inputs
# fix, which is exactly the kind of change that moves tok/s. A cross-commit
# comparison confounds the lever with everything else in that range — the one
# mistake this lane has paid for repeatedly. Both arms, one commit, one window.
#
# Bundling the arms is deliberate: box ACQUISITION is the scarce resource, not
# box time. The 08-10 attempt waited 40 min and never got a window at all.
# Arms are banked independently, so losing the box after A and B still answers
# the top-ranked question.
#
# WHAT IT REFUSES TO REPORT — the failure mode here is a plausible number, not
# an error, so every arm is gated on:
#   1. tree      — HEAD and a clean tree, re-checked per arm (the shared
#                  checkout can move under a 12 h wait)
#   2. config    — the generated yaml really declares k=2
#   3. freshness — the bench artifact must postdate the arm's start, so a stale
#                  file cannot be read as a result (this happened on 08-10)
#   4. EAGER     — zero "running EAGER" lines; one uncaptured step forks the
#                  token stream, so this is correctness, not perf
#   5. tokens    — 3264 at k=2. Greedy verify emits the target's own greedy
#                  stream (mtp_greedy_verify: drafts[:n_acc] + bonus, all equal
#                  to target argmax), so this total is INVARIANT across arms
#                  regardless of acceptance. Any other total means divergence.
#   6. throughput— a tok/s line actually parsed (an empty one would otherwise
#                  awk-coerce to 0 and print a fabricated delta)
#   7. arm tag   — the acceptance line names its own pairing arm; it must match
#
# AND IT ALWAYS HANDS THE BOX BACK. m3-sweep2.sh's SERVER-UP-BENCH-FAILED path
# deliberately leaves the server UP for a client retry. Without a teardown of
# our own that strands 8 H200s on a shared box and makes every later waiter
# read "holder: kirill" until someone notices by hand.
#
# Usage:  bash env/m3-trunk-pairing-ab.sh              # wait, then A B C
#         ARMS="A B" bash env/m3-trunk-pairing-ab.sh   # pairing only
#         SKIP_WAIT=1 ARMS=C bash env/...              # box already ours
set -uo pipefail

# Overridable only so the driver's own logic can be exercised against fakes
# before it is trusted with a 95-minute box window. On the box, leave unset.
P=${P:-/m-coriander/coriander/kirill}
D=$P/glm-m3
K=${K:-2}   # the shipped combined config is k=3: `K=3 ARMS="S L" bash env/m3-trunk-pairing-ab.sh`
TS=$(date -u +%Y%m%dT%H%M%S)
RUN=$D/perf-ab-$TS
BENCH=$D/bench-mstar-b1-mtp-k$K.txt
PROV=$D/bench-mtp-b1-k$K.provenance

ARMS=${ARMS:-"A B C"}
WAIT_MAX_H=${WAIT_MAX_H:-12}
SKIP_WAIT=${SKIP_WAIT:-0}
DRAIN_MAX_S=${DRAIN_MAX_S:-180}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}

# 10, not 30. The 30-minute sustained-clear rule assumes contention means
# someone's LONG job, where landing in their gap is the hazard. Measured over
# 2h14m on 2026-08-10/11, this box's steady state is the opposite: a stream of
# 1-4 minute jobs from kanzhu and garv901, most under the 1000 MiB threshold.
# The longest quiet stretch in that whole window was 28 minutes and the median
# was 2, so a 30-minute bar is unsatisfiable — the run waited all night and
# measured nothing. The bar is also guarding something cheap: mstar-glm-serve
# aborts on a busy GPU within ~10 s, and m3-sweep2.sh breaks its health poll
# as soon as that process dies, so a raced launch costs seconds rather than
# the 50-minute readiness timeout. Retrying a cheap race beats never starting.
FREE_FOR_MIN=${FREE_FOR_MIN:-10}

WANT_TOKENS=3264                                   # k=2, 20 prompts, invariant
VLLM_REF="104.71 text tok/s (p1=0.866 p2=0.614)"   # same checkpoint, k=2

mkdir -p "$RUN"
say() { echo "[perf-ab] $(date -u +%m-%dT%H:%M:%S) $*" | tee -a "$RUN/run.log"; }
exec >> "$RUN/console.log" 2>&1

# ---------------------------------------------------------------- box state
# FAIL CLOSED. A bare `awk '$1>1000'` over empty input prints 0, so an
# nvidia-smi hiccup would read as "box completely clear" and launch onto a
# contended box. Demand all 8 lines or report the box as busy.
busy_gpus() {
  local out n
  out=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  n=$(printf '%s\n' "$out" | grep -c '^[0-9]\+$')
  if [ "$n" -ne 8 ]; then echo 8; return; fi
  printf '%s\n' "$out" | awk '$1>1000{c++} END{print c+0}'
}
others_on_gpus() {
  local p u
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
    u=$(ps -o user= -p "$p" 2>/dev/null | tr -d " ")
    [ -n "$u" ] && [ "$u" != "kirill" ] && { echo "$u pid=$p"; return 0; }
  done
  return 1
}
our_gpu_pids() {
  local p
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
    [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d " ")" = "kirill" ] && echo "$p"
  done
}

# Hand the box back. Only ever kills OUR pids — another user's job is never
# touched, even when we are sharing the box with one.
release_box() {
  local why=$1 i pids
  pids=$(our_gpu_pids)
  if [ -z "$pids" ] && ! pgrep -u kirill -f "[m]star-serve" > /dev/null; then
    return 0
  fi
  say "release_box ($why): tearing down our serve — pids [$(echo "$pids" | tr '\n' ' ')]"
  pkill -9 -u kirill -f "[m]star-serve" 2>/dev/null
  pkill -u kirill -f "[m]3-hangcatch" 2>/dev/null
  for i in $(seq 1 12); do
    pids=$(our_gpu_pids)
    [ -z "$pids" ] && { say "release_box: clean after ${i}0s"; return 0; }
    echo "$pids" | xargs -r kill -9 2>/dev/null
    sleep 10
  done
  say "release_box: DIRTY after 120s — our pids still on the GPUs, check by hand"
  return 1
}

# Only fires if we actually launched something. Covers ssh drop, SIGTERM, and
# any early exit added later.
LAUNCHED=0
on_exit() { [ "$LAUNCHED" = 1 ] && release_box "exit trap"; return 0; }
trap on_exit EXIT
trap 'on_exit; exit 130' INT TERM

# Wait out OUR OWN residue only (the sweep's teardown returns after 120 s of
# retries whether or not it succeeded). Another user landing is not residue —
# that aborts immediately rather than trampling them.
drain_ours() {
  local waited=0
  while [ "$(busy_gpus)" -ne 0 ]; do
    others_on_gpus > /dev/null && return 0     # not ours; caller's gate decides
    [ "$waited" -ge "$DRAIN_MAX_S" ] && { say "drain: our residue still present after ${waited}s"; return 1; }
    [ "$waited" = 0 ] && say "drain: waiting out our own GPU residue"
    sleep 15; waited=$(( waited + 15 ))
  done
  return 0
}

# ---------------------------------------------------------------- preflight
cd "$P/mstar" || { say "no checkout at $P/mstar"; exit 1; }
HEAD_SHA=$(git rev-parse HEAD)
say "commit under test: $(git log --oneline -1)"
if [ -n "$(git status --porcelain)" ]; then
  say "REFUSING: working tree is dirty — a number from an unnamed tree is not evidence"
  git status --porcelain | head -10 | while read -r l; do say "  $l"; done
  echo "DIRTY-TREE $(date -u)" > "$RUN/ab.done"; exit 1
fi
echo "$HEAD_SHA" > "$RUN/commit.txt"
cp -p "$P/mstar/env/coriander-venv-freeze.txt" "$RUN/" 2>/dev/null
"$P/mstar/.venv/bin/pip" freeze 2>/dev/null > "$RUN/venv-at-run.txt"
say "arms: $ARMS"

# ------------------------------------------------------------------- waiter
# Any occupancy resets the clock. Called before every arm, not just the first:
# an arm that loses the box to a racing job goes back to waiting instead of
# abandoning the measurement.
wait_for_box() {
  [ "$SKIP_WAIT" = "1" ] && return 0
  local announced=0 clear_since="" last_report=0 now held deadline
  deadline=$(( $(date +%s) + WAIT_MAX_H * 3600 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    now=$(date +%s)
    if [ "$(busy_gpus)" -eq 0 ] && ! others_on_gpus > /dev/null; then
      [ -z "$clear_since" ] && { clear_since=$now; say "box went clear — need ${FREE_FOR_MIN}m of quiet"; }
      held=$(( (now - clear_since) / 60 ))
      [ "$held" -ge "$FREE_FOR_MIN" ] && { say "box clear ${held}m — taking it"; return 0; }
      if [ $(( now - last_report )) -ge 600 ]; then
        say "box clear ${held}/${FREE_FOR_MIN}m"; last_report=$now
      fi
    else
      if [ -n "$clear_since" ]; then
        say "clock RESET after $(( (now - clear_since) / 60 ))m — $(busy_gpus)/8 busy, holder: $(others_on_gpus || echo kirill)"
        clear_since=""; last_report=0
      elif [ "$announced" = 0 ]; then
        say "waiting — $(busy_gpus)/8 busy, holder: $(others_on_gpus || echo none)"; announced=1
      fi
    fi
    sleep 60
  done
  say "gave up after ${WAIT_MAX_H}h without a ${FREE_FOR_MIN}m clear window"
  return 1
}

# ---------------------------------------------------------------- one arm
run_arm() {
  local arm=$1 label want_tag out=$RUN/arm$1
  mkdir -p "$out"
  case "$arm" in
    A) label="defaults (pre-final-norm pairing)";        want_tag="pre-final-norm (default)" ;;
    B) label="PAIR_POSTNORM=1 (vLLM convention)";        want_tag="POST-final-norm (vLLM convention)" ;;
    C) label="CAPTURE_SYNC=1 (padded sync graph)";       want_tag="pre-final-norm (default)" ;;
    # 2026-08-19 arms. S = the shipped combined config (post-norm + captured
    # sync, k=3) with the captured MTP prefill and prefill drafts pinned OFF —
    # i.e. T0's config on THIS tree, so S − T0(66.81) isolates what the
    # sync-free planning (not flag-gated) is worth. L = the new defaults:
    # captured MTP prefill ON plus the prefill-drafts edge ON.
    S) label="shipped combined (post-norm+sync capture), prefill eager, no prefill drafts"
       want_tag="POST-final-norm (vLLM convention)" ;;
    L) label="new defaults: + captured MTP prefill + prefill drafts"
       want_tag="POST-final-norm (vLLM convention)" ;;
    # F = L plus the fused shared+routed MoE all-reduce (default-off flag). NOTE:
    # this one may legitimately move the token count (bf16 sum order) — a
    # TRIPWIRE on tokens here means "read armF/bench.txt and diff the streams",
    # not "the code is broken".
    F) label="L + MSTAR_GLM52_MOE_FUSED_ALLREDUCE=1 (one all-reduce per MoE layer)"
       want_tag="POST-final-norm (vLLM convention)" ;;
    *) say "unknown arm '$arm'"; echo "UNKNOWN-ARM" > "$out/verdict.txt"; return 1 ;;
  esac

  # ---- gate 1: the tree has not moved under us. The shared checkout is
  # documented as often ahead, and the waiter can sit for 12 h.
  local now_sha; now_sha=$(git -C "$P/mstar" rev-parse HEAD)
  if [ "$now_sha" != "$HEAD_SHA" ]; then
    say "arm $arm ABORT: checkout moved $HEAD_SHA -> $now_sha since preflight"
    echo "TREE-MOVED $now_sha" > "$out/verdict.txt"; return 1
  fi
  if [ -n "$(git -C "$P/mstar" status --porcelain)" ]; then
    say "arm $arm ABORT: tree went dirty since preflight"
    echo "TREE-DIRTY" > "$out/verdict.txt"; return 1
  fi

  drain_ours
  if [ "$(busy_gpus)" -ne 0 ] || others_on_gpus > /dev/null; then
    say "arm $arm ABORT before launch: $(busy_gpus)/8 busy, holder: $(others_on_gpus || echo kirill)"
    echo "RACED" > "$out/verdict.txt"; return 1
  fi

  # Displace the previous artifacts rather than trusting mtime alone: if this
  # arm dies before writing, there is no file left to misread.
  [ -f "$BENCH" ] && mv "$BENCH" "$RUN/displaced-bench-before-arm$arm.txt"
  [ -f "$PROV" ] && mv "$PROV" "$RUN/displaced-prov-before-arm$arm.txt"
  rm -f "$D/m3-sweep.done"

  local start; start=$(date +%s)
  LAUNCHED=1
  say "arm $arm ($label) launching, k=$K${2:+ (attempt $2)}"
  # Sample who else lands on the GPUs for the duration of the arm. With
  # FREE_FOR_MIN at 10 a short job CAN arrive mid-run; a number measured while
  # someone else shared the SMs is not necessarily wrong, but it is not
  # comparable to a clean one, so it gets attributed instead of silently
  # trusted. Not a tripwire: withholding it would lose real data.
  ( while :; do others_on_gpus >> "$out/contention.txt" 2>/dev/null; sleep 30; done ) &
  local sampler=$!
  (
    # Pin the baseline OFF explicitly (not `unset`): as of 2026-08-11 the code
    # default for both flags is ON, so arm A must force them off to measure the
    # pre-norm / eager-sync baseline rather than the shipped combined path.
    export MSTAR_GLM52_MTP_PAIR_POSTNORM=0 MSTAR_GLM52_MTP_CAPTURE_SYNC=0 \
           MSTAR_GLM52_MTP_PREFILL_DRAFTS=0 MSTAR_GLM52_MTP_CAPTURE_PREFILL=0
    case "$arm" in
      B) export MSTAR_GLM52_MTP_PAIR_POSTNORM=1 ;;
      C) export MSTAR_GLM52_MTP_CAPTURE_SYNC=1 ;;
      S) export MSTAR_GLM52_MTP_PAIR_POSTNORM=1 MSTAR_GLM52_MTP_CAPTURE_SYNC=1 \
                MSTAR_PHASE_TIMING=200 MSTAR_GLM52_MTP_STEP_TIMING=200 ;;
      L) export MSTAR_GLM52_MTP_PAIR_POSTNORM=1 MSTAR_GLM52_MTP_CAPTURE_SYNC=1 \
                MSTAR_GLM52_MTP_CAPTURE_PREFILL=1 MSTAR_GLM52_MTP_PREFILL_DRAFTS=1 \
                MSTAR_PHASE_TIMING=200 MSTAR_GLM52_MTP_STEP_TIMING=200 ;;
      F) export MSTAR_GLM52_MTP_PAIR_POSTNORM=1 MSTAR_GLM52_MTP_CAPTURE_SYNC=1 \
                MSTAR_GLM52_MTP_CAPTURE_PREFILL=1 MSTAR_GLM52_MTP_PREFILL_DRAFTS=1 \
                MSTAR_GLM52_MOE_FUSED_ALLREDUCE=1 MSTAR_PHASE_TIMING=200 ;;
    esac
    env | grep -E '^MSTAR_GLM52' | sort > "$out/env.txt"
    HOLD_MIN=0 bash "$D/m3-sweep2.sh" "$K"
  )
  kill "$sampler" 2>/dev/null
  local sweep; sweep=$(cat "$D/m3-sweep.done" 2>/dev/null || echo MISSING)
  say "arm $arm sweep marker: $sweep"
  local shared="none"
  if [ -s "$out/contention.txt" ]; then
    shared=$(sort -u "$out/contention.txt" | cut -d' ' -f1 | sort -u | tr '\n' ' ')
    touch "$out/contended"
    say "arm $arm CONTENDED: shared the box with $shared during this arm"
  fi

  # The sweep leaves the server UP on bench failure, by design. Take the box
  # back before the next arm reads our own server as contention.
  release_box "after arm $arm"

  # ---- gate 2: the generated config really declares this k
  local cfg=$D/configs/glm52_tp8_mtp_fast_k$K.yaml
  if ! grep -qE "^[[:space:]]*mtp_num_draft_tokens:[[:space:]]*$K([[:space:]]|\$)" "$cfg" 2>/dev/null; then
    say "arm $arm TRIPWIRE: $cfg does not declare mtp_num_draft_tokens: $K"
    echo "BAD-CONFIG sweep=$sweep" > "$out/verdict.txt"; return 1
  fi
  cp -p "$cfg" "$out/config.yaml"

  # ---- gate 3: freshness
  if [ ! -f "$BENCH" ]; then
    say "arm $arm FAILED: no bench artifact (sweep=$sweep)"
    echo "NO-ARTIFACT sweep=$sweep" > "$out/verdict.txt"; return 1
  fi
  if [ "$(stat -c %Y "$BENCH")" -lt "$start" ]; then
    say "arm $arm FAILED: bench artifact predates the arm — STALE, refusing to read it"
    echo "STALE-ARTIFACT sweep=$sweep" > "$out/verdict.txt"; return 1
  fi
  cp -p "$BENCH" "$out/bench.txt"
  [ -f "$PROV" ] && cp -p "$PROV" "$out/provenance.txt"

  # ---- this arm's serve log (exclude the watchdog/supervision siblings)
  local log
  log=$(find "$P/logs" -maxdepth 1 -name 'mstar-glm52-mtp-*.log' \
        ! -name '*-watchdog-*' ! -name '*-sup-*' -newermt "@$start" \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  if [ -z "$log" ]; then
    say "arm $arm FAILED: no serve log newer than the arm start"
    echo "NO-SERVE-LOG sweep=$sweep" > "$out/verdict.txt"; return 1
  fi
  say "arm $arm serve log: $log"
  echo "$log" > "$out/serve-log-path.txt"
  grep -hE "by position|emitted/step" "$log" | tail -4 > "$out/acceptance.txt"
  grep -h "running EAGER" "$log" > "$out/eager.txt"

  local eager toks tps tag
  eager=$(grep -c "running EAGER" "$log")
  toks=$(grep -oE 'Text tokens: [0-9]+ total' "$out/bench.txt" | grep -oE '[0-9]+' | head -1)
  tps=$(grep -oE '[0-9.]+ text tok/s' "$out/bench.txt" | tail -1 | grep -oE '^[0-9.]+')
  tag=$(grep -h "by position" "$log" | tail -1 \
        | sed -n 's/.*\[trunk pairing: \([^]]*\)\].*/\1/p')

  {
    echo "arm       : $arm ($label)"
    echo "tok/s     : ${tps:-MISSING}"
    echo "tokens    : ${toks:-MISSING} (want $WANT_TOKENS)"
    echo "EAGER hits: $eager (want 0)"
    echo "pairing   : ${tag:-MISSING} (want $want_tag)"
    echo "sweep     : $sweep"
    echo "shared box: $shared"
  } > "$out/summary.txt"

  local bad=0
  # A non-numeric count means the grep itself failed; treat that as a tripwire
  # rather than letting `[ "" -ne 0 ]` error out and fail open.
  if ! [[ "$eager" =~ ^[0-9]+$ ]]; then
    say "arm $arm TRIPWIRE: could not count EAGER lines in $log"; bad=1; eager=-1
  elif [ "$eager" -ne 0 ]; then
    say "arm $arm TRIPWIRE: $eager EAGER warnings — this is not the captured stream"; bad=1
  fi
  if [ "${toks:-x}" != "$WANT_TOKENS" ]; then
    say "arm $arm TRIPWIRE: ${toks:-MISSING} tokens != $WANT_TOKENS — stream diverged"; bad=1
  fi
  if [ -z "$tps" ]; then
    say "arm $arm TRIPWIRE: no throughput line in the bench output"; bad=1
  fi
  if [ -z "$tag" ]; then
    say "arm $arm TRIPWIRE: no acceptance line — cannot attribute this number to an arm"; bad=1
  elif [ "$tag" != "$want_tag" ]; then
    say "arm $arm TRIPWIRE: log says pairing '$tag', expected '$want_tag' — MISLABELLED"; bad=1
  fi
  if [ "$bad" = 1 ]; then
    echo "TRIPWIRE sweep=$sweep" > "$out/verdict.txt"
    say "arm $arm: number WITHHELD, tripwires above"; return 1
  fi

  echo "OK" > "$out/verdict.txt"
  echo "$tps" > "$out/tokps.txt"
  say "arm $arm CLEAN: $tps tok/s, $toks tokens, 0 eager, [$tag]"
  return 0
}

# ------------------------------------------------------------------- run
# An arm that never got the box, or lost it, is RETRIED after re-waiting. An
# arm that actually ran and failed a correctness tripwire is NOT retried —
# that is a real result about the code, and re-running it just burns the box.
# Without this the 08-10/11 attempt waited all night and measured nothing:
# one racing job at the wrong moment ended the whole run.
had_box=0
for arm in $ARMS; do
  attempt=1
  while :; do
    if [ "$had_box" != 1 ]; then
      if ! wait_for_box; then
        mkdir -p "$RUN/arm$arm"; echo "NO-WINDOW" > "$RUN/arm$arm/verdict.txt"
        say "arm $arm: no ${FREE_FOR_MIN}m window inside ${WAIT_MAX_H}h"; break
      fi
    fi
    had_box=0
    if run_arm "$arm" "$attempt"; then had_box=1; break; fi
    case "$(cat "$RUN/arm$arm/verdict.txt" 2>/dev/null)" in
      RACED*|NO-ARTIFACT*|NO-SERVE-LOG*|STALE-ARTIFACT*)
        if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
          say "arm $arm: lost the box on all $MAX_ATTEMPTS attempts — moving on"; break
        fi
        say "arm $arm: attempt $attempt lost the box — re-waiting"
        mv "$RUN/arm$arm" "$RUN/arm$arm-lost$attempt"
        attempt=$(( attempt + 1 )) ;;
      *) break ;;   # TREE-*, BAD-CONFIG, TRIPWIRE — real, do not retry
    esac
  done
done

# ---------------------------------------------------------------- report
say "==== GLM-5.2 MTP PERF A/B — k=$K, B=1, TP8, commit $(cut -c1-8 "$RUN/commit.txt") ===="
for arm in $ARMS; do
  if [ -f "$RUN/arm$arm/summary.txt" ]; then
    while read -r l; do say "  $l"; done < "$RUN/arm$arm/summary.txt"
    while read -r l; do say "    $l"; done < "$RUN/arm$arm/acceptance.txt"
  else
    say "  arm $arm: $(cat "$RUN/arm$arm/verdict.txt" 2>/dev/null || echo 'did not run')"
  fi
done
say "  reference vLLM k=$K: $VLLM_REF"

# Deltas only against a CLEAN arm A, and only from numbers that actually parsed.
if [ -f "$RUN/armA/tokps.txt" ]; then
  A_T=$(cat "$RUN/armA/tokps.txt")
  for arm in $ARMS; do
    [ "$arm" = "A" ] && continue
    if [ -f "$RUN/arm$arm/tokps.txt" ]; then
      note=""
      { [ -f "$RUN/armA/contended" ] || [ -f "$RUN/arm$arm/contended" ]; } \
        && note="  [CONTENDED — one or both arms shared the box; treat as indicative]"
      say "  DELTA arm$arm - armA: $(awk -v a="$A_T" -v b="$(cat "$RUN/arm$arm/tokps.txt")" \
            'BEGIN{printf "%+.2f tok/s", b-a}')$note"
    else
      say "  DELTA arm$arm - armA: unavailable ($(cat "$RUN/arm$arm/verdict.txt" 2>/dev/null || echo 'did not run'))"
    fi
  done
else
  say "  no clean arm A — every delta is unavailable, and no absolute number here is the baseline"
fi

# Leave the canonical k=2 path meaning what every other script assumes it
# means: the DEFAULT config's number, or NOTHING. After the last arm it holds
# that arm's output — for B or C a non-default number under the canonical
# name, which is exactly the stale artifact that made a waiter misreport on
# 08-10. Clear it unconditionally, then restore only a clean arm A.
[ -f "$BENCH" ] && mv "$BENCH" "$RUN/displaced-bench-final.txt"
[ -f "$PROV" ] && mv "$PROV" "$RUN/displaced-prov-final.txt"
# Gate on arm A's VERDICT, not on its file existing: armA/bench.txt is copied
# in before the EAGER/token/tag tripwires run, so a diverged or eager arm A
# has a bench.txt too. Promoting that to the canonical name would republish a
# number the gates just rejected.
if [ "$(cat "$RUN/armA/verdict.txt" 2>/dev/null)" = OK ] && [ -f "$RUN/armA/bench.txt" ] \
   && [ ! -f "$RUN/armA/contended" ]; then
  cp -p "$RUN/armA/bench.txt" "$BENCH"
  [ -f "$RUN/armA/provenance.txt" ] && cp -p "$RUN/armA/provenance.txt" "$PROV"
  say "canonical $BENCH restored to arm A (defaults)"
else
  say "canonical $BENCH left ABSENT — arm A was not clean-and-uncontended; nothing here is the k=$K number"
fi

release_box "final"
if [ "$(busy_gpus)" -ne 0 ]; then
  say "WARNING: $(busy_gpus)/8 GPUs still busy — holder: $(others_on_gpus || echo kirill)"
fi

ok=$(grep -lx OK "$RUN"/arm*/verdict.txt 2>/dev/null | wc -l | tr -d ' ')
n=$(echo "$ARMS" | wc -w | tr -d ' ')
echo "DONE arms=$ARMS clean=$ok/$n $(date -u)" > "$RUN/ab.done"
say "complete: $ok/$n arms clean — artifacts: $RUN"
