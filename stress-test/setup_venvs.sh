#!/usr/bin/env bash
# Build (or check) the two virtualenvs the resource_pools_2 reproduction needs.
#
#   bash stress-test/setup_venvs.sh --check    verify paths + torch version, install nothing
#   bash stress-test/setup_venvs.sh            create/refresh both venvs
#
# Env (all optional, defaults shown):
#   STRESS_WORK=$PWD/stress-run       root for venvs, checkouts and run output
#   STRESS_VENVS=$STRESS_WORK/venvs   where the venvs live
#   CO_BASE=$STRESS_WORK/co-base      checkout at the merge base 6666c623
#   CO_HEAD=$STRESS_WORK/co-head      checkout at 485beeb7 (or 5eeb434f)
#   FLASH_ATTN_WHEEL=<path>           optional; installed --no-deps so torch cannot move
set -u
STRESS_WORK="${STRESS_WORK:-$PWD/stress-run}"
STRESS_VENVS="${STRESS_VENVS:-$STRESS_WORK/venvs}"
CO_BASE="${CO_BASE:-$STRESS_WORK/co-base}"
CO_HEAD="${CO_HEAD:-$STRESS_WORK/co-head}"
TORCH_VERSION="${TORCH_VERSION:-2.12.1}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

rc=0
note() { printf '%s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; rc=1; }
ok()   { printf 'ok    %s\n' "$*"; }

note "=== stress-test/setup_venvs.sh$([ "$CHECK" = 1 ] && echo ' (--check)') ==="
note "  STRESS_VENVS=$STRESS_VENVS"
note "  CO_BASE=$CO_BASE"
note "  CO_HEAD=$CO_HEAD"
note "  torch==${TORCH_VERSION} backend=${TORCH_BACKEND}"
note ""

command -v uv >/dev/null 2>&1 || fail "uv not on PATH (https://docs.astral.sh/uv/)"

# The head checkout may sit at either the current head or the commit the
# original numbers were taken at; both are valid arms.
for pair in "mstar-base:$CO_BASE:6666c623" "mstar-head:$CO_HEAD:485beeb7|5eeb434f"; do
  NAME="${pair%%:*}"; rest="${pair#*:}"; CO="${rest%%:*}"; WANT_SHA="${rest##*:}"
  V="$STRESS_VENVS/$NAME"

  if [ -d "$CO/.git" ] || [ -f "$CO/.git" ]; then
    HAVE=$(git -C "$CO" rev-parse --short=8 HEAD 2>/dev/null)
    case "|$WANT_SHA|" in
      *"|${HAVE:-?}|"*) ok "$NAME checkout $CO at $HAVE" ;;
      *) fail "$NAME checkout $CO is at ${HAVE:-?}, expected $WANT_SHA" ;;
    esac
  else
    fail "$NAME checkout $CO does not exist (git worktree add \"$CO\" ${WANT_SHA%%|*} --detach)"
  fi

  if [ "$CHECK" = "1" ]; then
    if [ -x "$V/bin/python" ]; then
      HAVE_T=$("$V/bin/python" -c 'import torch;print(torch.__version__)' 2>/dev/null)
      case "$HAVE_T" in
        "${TORCH_VERSION}+${TORCH_BACKEND}") ok "$NAME venv $V torch $HAVE_T" ;;
        "") fail "$NAME venv $V exists but torch does not import" ;;
        *)  fail "$NAME venv $V has torch $HAVE_T, expected ${TORCH_VERSION}+${TORCH_BACKEND}" ;;
      esac
      HAVE_M=$(cd / && "$V/bin/python" -c 'import mstar,os;print(os.path.realpath(os.path.dirname(mstar.__file__)))' 2>/dev/null)
      case "$HAVE_M" in
        "$(cd "$CO" 2>/dev/null && pwd -P)/mstar") ok "$NAME mstar resolves to $HAVE_M" ;;
        "") fail "$NAME cannot import mstar" ;;
        *)  fail "$NAME mstar resolves to $HAVE_M, expected $CO/mstar" ;;
      esac
    else
      fail "$NAME venv $V missing (run without --check to build it)"
    fi
    continue
  fi

  note "--- building $NAME in $V"
  uv venv --clear --python 3.12 "$V" || { fail "$NAME venv create"; continue; }
  uv pip install --python "$V/bin/python" --torch-backend="$TORCH_BACKEND" \
      "torch==${TORCH_VERSION}" torchvision || { fail "$NAME torch"; continue; }
  uv pip install --python "$V/bin/python" --torch-backend="$TORCH_BACKEND" -e "$CO[all]" \
      || { fail "$NAME mstar[all]"; continue; }
  if [ -n "${FLASH_ATTN_WHEEL:-}" ] && [ -f "${FLASH_ATTN_WHEEL}" ]; then
    # --no-deps: a resolving install of this wheel will drag torch forward.
    uv pip install --python "$V/bin/python" --no-deps "$FLASH_ATTN_WHEEL" || fail "$NAME flash-attn"
  fi
  HAVE_T=$("$V/bin/python" -c 'import torch;print(torch.__version__)' 2>/dev/null)
  [ "$HAVE_T" = "${TORCH_VERSION}+${TORCH_BACKEND}" ] \
    && ok "$NAME built, torch $HAVE_T" \
    || fail "$NAME built but torch is $HAVE_T (expected ${TORCH_VERSION}+${TORCH_BACKEND})"
done

note ""
if [ "$rc" = "0" ]; then note "ALL CHECKS PASSED"; else note "one or more checks FAILED"; fi
exit $rc
