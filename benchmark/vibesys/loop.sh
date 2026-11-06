#!/bin/bash
# Works through a brief in Claude Code, headless, one call after another from inside a worktree,
# with the tool permissions in <state-dir>/settings.json. Each call resumes the conversation the
# brief started, until the brief changes other than by gaining text at its end. The loop stops at
# the epoch time in <state-dir>/deadline (72 hours after the first call unless written there), or
# after the call during which <state-dir>/STOP appears. Starting with a changed brief clears STOP.
set -uo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <brief> <worktree> <state-dir>" >&2; exit 2; }
brief=$(realpath "$1") wt=$(realpath "$2") state=$(realpath "$3")
sent=$state/brief.sent
[ -s "$state/settings.json" ] || { echo "no $state/settings.json" >&2; exit 2; }
say() { printf '[loop %s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
[ -s "$state/deadline" ] || echo $(( $(date +%s) + 72 * 3600 )) >"$state/deadline"
[ -s "$sent" ] && ! cmp -s "$brief" "$sent" && rm -f "$state/STOP"
cd "$wt" || exit 1

n=0 fails=0 pause=10
while [ ! -e "$state/STOP" ] && [ "$(date +%s)" -lt "$(cat "$state/deadline")" ]; do
  n=$((n + 1))
  # The brief is copied before the call, so an edit made during the call is sent by the next one.
  cp "$brief" "$state/brief.next"
  if [ -s "$sent" ] && cmp -s "$state/brief.next" "$sent"; then
    flags=(--resume "$(cat "$state/session")")
    prompt="Continue your brief ($brief). Read your results and status files to see where you are, then take the next step. Nobody will answer questions, so keep going."
  elif [ -s "$sent" ] && head -c "$(stat -c%s "$sent")" "$state/brief.next" | cmp -s - "$sent"; then
    flags=(--resume "$(cat "$state/session")")
    prompt="Your brief ($brief) gained this text at its end. It overrides the rest of the brief where they conflict, and a round you were in may have been cut off. Read your results and status files, then take the next step. Nobody will answer questions, so keep going.

$(tail -c +$(( $(stat -c%s "$sent") + 1 )) "$state/brief.next")"
  else
    [ -s "$sent" ] && mv "$sent" "$state/brief.prev-$(date -u +%m%dT%H%M).md"
    flags=()
    prompt=$(cat "$state/brief.next")
  fi

  t0=$(date +%s)
  claude -p "${flags[@]}" --settings "$state/settings.json" --output-format json "$prompt" \
    >"$state/last.json" 2>"$state/last.err"
  code=$?
  read -r session is_error limited summary < <(python3 - "$state/last.json" "$state/last.err" <<'PY'
import json, re, sys
limit = re.compile(r"usage limit|rate limit|limit reached|overloaded", re.I)
err = open(sys.argv[2]).read()
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print("-", 1, int(bool(limit.search(err))), "unparsed output")
    sys.exit()
text = " ".join((d.get("result") or "").split())
# Only an error's text is checked, since a reply that mentions a limit is not one.
limited = bool(limit.search(err)) or (bool(d.get("is_error")) and bool(limit.search(text)))
print(d.get("session_id") or "-", int(bool(d.get("is_error"))), int(limited), f"turns={d.get('num_turns')} {text[:300]}")
PY
)
  say "call $n exit=$code $(( $(date +%s) - t0 ))s error=$is_error $summary"

  # A call that started a session holds the brief, even when it then failed.
  if [ "$session" != "-" ]; then
    echo "$session" >"$state/session"
    mv "$state/brief.next" "$sent"
  fi
  if [ "$code" = 0 ] && [ "$is_error" = 0 ]; then
    fails=0 pause=10
  elif [ "$limited" = 1 ]; then
    pause=1800
  else
    fails=$((fails + 1)) pause=$(( pause < 900 ? pause * 2 + 60 : 1800 ))
    if [ "$fails" -ge 3 ] && [ -s "$sent" ]; then
      mv "$sent" "$state/brief.prev-$(date -u +%m%dT%H%M).md"
      fails=0
      say "3 failed calls in a row, the next call starts a fresh conversation from the brief"
    fi
  fi
  [ -e "$state/STOP" ] && break
  sleep "$pause"
done
say "stopped after call $n ($( [ -e "$state/STOP" ] && echo STOP file || echo deadline))"
