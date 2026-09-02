#!/usr/bin/env bash
# Self-test for env/m3-trunk-pairing-ab.sh. No GPU, no box, ~5 seconds.
#
# The driver decides whether a 95-minute box window produced a trustworthy
# number. Its own logic therefore has to be exercised somewhere cheaper than
# the box. This stands up a fake $P tree, a fake nvidia-smi reporting 8 idle
# GPUs, and a fake m3-sweep2.sh that can be told to succeed, to fail the bench
# while leaving the server up, to emit a diverged token total, to emit an
# EAGER warning, or to yield the box — then checks the driver reports and
# withholds the right things in each case.
#
# RUN IT ON THE BOX, not a mac: the driver uses GNU find -printf/-newermt and
# GNU stat -c, and BSD versions silently do something else.
#   scp env/m3-*.sh coriander:/m-coriander/coriander/kirill/tmp/ab-selftest/
#   TMPBASE=/m-coriander/coriander/kirill/tmp \
#   DRIVER=$PWD/m3-trunk-pairing-ab.sh bash m3-ab-selftest.sh
#
# Two real bugs it caught before the first box run, both of the shape this
# lane keeps paying for — a rejected number published under a trusted name:
#   - the canonical bench-mstar-b1-mtp-k2.txt kept the LAST arm output, so a
#     non-default arm C number sat at the path every other script reads
#   - the arm-A restore was gated on the file existing rather than on the
#     verdict, so a diverged (3192-token) arm A was promoted to canonical
SB=$(mktemp -d -p "${TMPBASE:-/tmp}")
DRIVER=${DRIVER:-$(dirname "$0")/m3-trunk-pairing-ab.sh}

mkfake() {           # scenario -> builds a fresh $SB/P tree + fake bins
  rm -rf "$SB/P" "$SB/bin"
  mkdir -p "$SB/P/mstar/env" "$SB/P/mstar/configs" "$SB/P/glm-m3/configs" "$SB/P/logs" "$SB/bin"

  # fake repo
  git -C "$SB/P/mstar" init -q 2>/dev/null
  git -C "$SB/P/mstar" config user.email t@t; git -C "$SB/P/mstar" config user.name t
  printf 'mtp_num_draft_tokens: 2\n' > "$SB/P/mstar/configs/glm52_tp8_mtp_fast.yaml"
  printf 'frozen\n' > "$SB/P/mstar/env/coriander-venv-freeze.txt"
  mkdir -p "$SB/P/mstar/.venv/bin"
  printf '#!/bin/sh\necho "pkg==1.0"\n' > "$SB/P/mstar/.venv/bin/pip"; chmod +x "$SB/P/mstar/.venv/bin/pip"
  git -C "$SB/P/mstar" add -A >/dev/null; git -C "$SB/P/mstar" commit -qm init

  # fake nvidia-smi: 8 idle GPUs, no compute apps. With FAKE_BUSY=1 it reports
  # pid 1 (root, i.e. not kirill) holding a GPU, which is how the driver sees
  # another user's job.
  cat > "$SB/bin/nvidia-smi" <<'EOF'
#!/bin/sh
case "$*" in
  *compute-apps*) [ "${FAKE_BUSY:-0}" = 1 ] && echo 1; exit 0 ;;   # pid only, as the real query returns
  *memory.used*)  if [ "${FAKE_BUSY:-0}" = 1 ]; then echo 5000; else echo 4; fi
                  for i in 2 3 4 5 6 7 8; do echo 4; done ;;
esac
EOF
  printf '#!/bin/sh\nexit 1\n' > "$SB/bin/pkill"
  printf '#!/bin/sh\nexit 1\n' > "$SB/bin/pgrep"
  chmod +x "$SB/bin"/*
}

# fake sweep: $1 = k. Behaviour driven by $SWEEP_MODE
mksweep() {
  cat > "$SB/P/glm-m3/m3-sweep2.sh" <<EOF
#!/usr/bin/env bash
D=$SB/P/glm-m3; P=$SB/P; K=\$1
sed "s/mtp_num_draft_tokens: .*/mtp_num_draft_tokens: \$K/" \\
  "\$P/mstar/configs/glm52_tp8_mtp_fast.yaml" > "\$D/configs/glm52_tp8_mtp_fast_k\$K.yaml"
TS=\$(date -u +%Y%m%dT%H%M%S)\$RANDOM
LOG=\$P/logs/mstar-glm52-mtp-\$TS.log
touch "\$P/logs/mstar-glm52-mtp-watchdog-\$TS.log" "\$P/logs/mstar-glm52-mtp-sup-\$TS.log"
TAG="pre-final-norm (default)"
[ "\${MSTAR_GLM52_MTP_PAIR_POSTNORM:-0}" = 1 ] && TAG="POST-final-norm (vLLM convention)"
echo "MTP acceptance: 2.03 emitted/step (ceiling 3, plain decode would be 1.00) — draft acceptance rate 0.51 over 1600 request-steps." > "\$LOG"
echo "MTP acceptance by position: n_acc histogram [100, 200, 300], conditional accept per position 0.77 0.34 [trunk pairing: \$TAG]" >> "\$LOG"
case "\$SWEEP_MODE" in
  ok)
      TPS=49.65; [ "\${MSTAR_GLM52_MTP_PAIR_POSTNORM:-0}" = 1 ] && TPS=58.40
      [ "\${MSTAR_GLM52_MTP_CAPTURE_SYNC:-0}" = 1 ] && TPS=63.10
      printf -- '--- Benchmark Results (wall time: 65.74s) ---\nText tokens: 3264 total (163.2 avg/req)\nThroughput: 0.30 req/s (successful only)\nThroughput: %s text tok/s\n' "\$TPS" > "\$D/bench-mstar-b1-mtp-k\$K.txt"
      echo prov > "\$D/bench-mtp-b1-k\$K.provenance"
      echo "DONE \$(date -u)" > "\$D/m3-sweep.done" ;;
  benchfail)
      echo "Traceback: client blew up" > "\$D/bench-mstar-b1-mtp-k\$K.txt"
      echo "SERVER-UP-BENCH-FAILED k=\$K \$(date -u)" > "\$D/m3-sweep.done"; exit 1 ;;
  diverged)
      printf -- 'Text tokens: 3192 total\nThroughput: 44.35 text tok/s\n' > "\$D/bench-mstar-b1-mtp-k\$K.txt"
      echo "DONE" > "\$D/m3-sweep.done" ;;
  eager)
      echo "MTP decode sync pass running EAGER (bs=1): no mtp_sync piecewise bucket" >> "\$LOG"
      printf -- 'Text tokens: 3264 total\nThroughput: 33.02 text tok/s\n' > "\$D/bench-mstar-b1-mtp-k\$K.txt"
      echo "DONE" > "\$D/m3-sweep.done" ;;
  yield)
      echo "PARTIAL \$(date -u)" > "\$D/m3-sweep.done"; exit 1 ;;
esac
EOF
  chmod +x "$SB/P/glm-m3/m3-sweep2.sh"
}

run() {  # $1 scenario name, $2 SWEEP_MODE, $3 ARMS, $4 FAKE_BUSY
  mkfake; mksweep
  echo "════════ $1 (mode=$2 arms=$3 busy=${4:-0})"
  PATH="$SB/bin:$PATH" P="$SB/P" SKIP_WAIT=1 ARMS="$3" SWEEP_MODE="$2" \
    FAKE_BUSY="${4:-0}" MAX_ATTEMPTS=3 DRAIN_MAX_S=1 bash "$DRIVER"
  local rd; rd=$(ls -td "$SB"/P/glm-m3/perf-ab-* | head -1)
  sed 's/^/   /' "$rd/run.log"
  echo "   --- marker: $(cat "$rd/ab.done" 2>/dev/null || echo MISSING)"
  echo "   --- canonical bench now: $(grep -oE '[0-9.]+ text tok/s' "$SB/P/glm-m3/bench-mstar-b1-mtp-k2.txt" 2>/dev/null || echo NONE)"
}

run "all three arms clean"        ok        "A B C"
run "bench fails (server leak)"   benchfail "A B"
run "stream diverged (3192)"      diverged  "A"
run "eager warning present"       eager     "A"
run "sweep yields, no artifact"   yield     "A B"
run "box never free (retry cap)"  ok        "A"      1
echo "sandbox: $SB"
