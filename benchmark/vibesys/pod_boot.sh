#!/bin/bash
# Prepares a fresh GPU pod for a loop and starts it, with everything kept under /workspace so
# a restart resumes where it left off. Each step skips what is already done, and progress goes
# to stdout, which RunPod serves as the pod log. Runs as root, configured by the environment:
#
#   POD_NAME      the agent works in /workspace/wt-$POD_NAME on branch perf/$POD_NAME, and names
#                 any other branch it makes perf/$POD_NAME-*
#   MSTAR_REF     the mstar-project/mstar ref checked out as run-base, the loop's first baseline
#   HARNESS_REF   the mstar-project/mstar ref this harness is copied from on the first boot (main,
#                 or pull/<n>/head)
#   BRIEF_GZB64   the brief, gzipped and base64-encoded (`gzip -9nc brief.md | base64 -w0`)
#   GIT_NAME, GIT_EMAIL   the identity of the agent's commits
#   LOOP_DEADLINE optional, the epoch time the loop stops at
#   FORK, GH_TOKEN        optional: every 15 minutes the agent's branches, and run-base as
#                         perf/$POD_NAME-base, are force-pushed to the GitHub repository FORK
#                         (owner/name) with this token
#   HF_TOKEN      optional, for gated weights
#
# Claude Code needs one login per pod: ssh in, run `claude auth login`, and restart the pod.
set -uo pipefail
W=/workspace
log() { echo "[boot $(date -u +%FT%TZ)] $*"; }
: "${POD_NAME:?}" "${MSTAR_REF:?}" "${HARNESS_REF:?}" "${GIT_NAME:?}" "${GIT_EMAIL:?}"
# The token goes to a file that only the push job reads. It leaves this script's environment
# before tmux starts, since tmux passes its first client's environment to every session.
if [ -n "${GH_TOKEN:-}" ] && [ -n "${FORK:-}" ]; then
  (umask 077; printf '%s' "$GH_TOKEN" >/root/.fork_token)
fi
unset GH_TOKEN
export HOME=/root CARGO_HOME=$W/.cargo RUSTUP_HOME=$W/.rustup UV_PYTHON_INSTALL_DIR=$W/uv-python
export PATH=/root/.local/bin:$W/.cargo/bin:$PATH
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv-cache UV_TORCH_BACKEND=auto CLAUDE_CONFIG_DIR=$W/home/.claude
mkdir -p "$HF_HOME" "$CLAUDE_CONFIG_DIR" "$W/results" "$W/briefs" "$W/loop"
WT=$W/wt-$POD_NAME
log "start ref=$MSTAR_REF gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | paste -sd';')"

# RunPod passes only RUNPOD_* and PATH to SSH shells, so the variables a login or the loop
# needs are written out here.
{
  printf 'export CLAUDE_CONFIG_DIR=%q POD_NAME=%q UV_PYTHON_INSTALL_DIR=%q\n' "$CLAUDE_CONFIG_DIR" "$POD_NAME" "$UV_PYTHON_INSTALL_DIR"
  printf 'export HF_HOME=%q HF_TOKEN=%q UV_CACHE_DIR=%q UV_TORCH_BACKEND=auto\n' "$HF_HOME" "${HF_TOKEN:-}" "$UV_CACHE_DIR"
  printf 'export CARGO_HOME=%q RUSTUP_HOME=%q PATH=/root/.local/bin:%q/.cargo/bin:$PATH\n' "$CARGO_HOME" "$RUSTUP_HOME" "$W"
  printf '[ -f %q/venv/bin/activate ] && . %q/venv/bin/activate\n' "$W" "$W"
} >/root/.pod_env
chmod 600 /root/.pod_env
grep -q pod_env /root/.bashrc 2>/dev/null || echo '. /root/.pod_env' >>/root/.bashrc

if ! command -v cmake >/dev/null || ! command -v tmux >/dev/null; then
  apt-get update -qq >/dev/null && apt-get install -y -qq cmake pkg-config tmux >/dev/null 2>&1
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash >/dev/null 2>&1
log "tools uv=$(uv --version 2>&1) cargo=$(cargo --version 2>&1) claude=$(claude --version 2>&1 | head -1)"

if [ ! -d "$W/mstar/.git" ]; then
  git clone -q https://github.com/mstar-project/mstar "$W/mstar" || { log "FAIL clone"; exit 1; }
fi
cd "$W/mstar"
if ! git rev-parse -q --verify run-base >/dev/null; then
  git fetch -q origin "$MSTAR_REF" && git checkout -q -b run-base FETCH_HEAD || { log "FAIL checkout $MSTAR_REF"; exit 1; }
fi
log "repo $(git rev-parse --short run-base) $(git log -1 --format=%s run-base)"
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"
# run-base may predate this harness, so the loop and the results CLI run from their own copy,
# taken once so the row schema can't change under a live results file.
if [ ! -d "$W/vibesys/benchmark/vibesys" ]; then
  git fetch -q origin "$HARNESS_REF" && mkdir -p "$W/vibesys" &&
    git archive FETCH_HEAD benchmark/__init__.py benchmark/vibesys | tar -x -C "$W/vibesys" ||
    { log "FAIL harness from $HARNESS_REF"; exit 1; }
fi
if [ ! -d "$WT" ]; then
  git worktree prune
  git worktree add -q -b "perf/$POD_NAME" "$WT" run-base 2>/dev/null || git worktree add -q "$WT" "perf/$POD_NAME" ||
    { log "FAIL worktree $WT"; exit 1; }
fi

[ -x "$W/venv/bin/python" ] || uv venv -q --python 3.12 --seed "$W/venv"
. "$W/venv/bin/activate"
# The editable install decides which checkout `mstar` imports from, so it is the agent's worktree.
# Checked from / so that the current directory can't stand in for the install.
where=$(cd / && python -c 'import mstar, pathlib; print(pathlib.Path(mstar.__file__).parents[1])' 2>/dev/null)
if [ "$where" != "$WT" ]; then
  log "installing M* from $WT (log: $W/install.log)"
  (cd "$WT" && uv pip install -e ".[all]") >"$W/install.log" 2>&1 ||
    { log "FAIL install, last lines:"; tail -25 "$W/install.log"; exit 1; }
fi
check=$(cd / && python -c 'import torch, mstar; print("torch", torch.__version__, "gpus", torch.cuda.device_count(), "ok", torch.cuda.is_available())' 2>&1 | tail -1)
log "python $check"
case "$check" in *"ok True"*) ;; *) log "FAIL python check"; exit 1 ;; esac

# `vibesys` is `python -m benchmark.vibesys` from this pod's copy, whatever the current checkout holds.
cat >/usr/local/bin/vibesys <<'EOF'
#!/bin/bash
exec python -c 'import sys; sys.path.insert(0, "/workspace/vibesys"); from benchmark.vibesys.cli import main; sys.exit(main(sys.argv[1:]))' "$@"
EOF
# Rows also go to the pod log, so they can be collected through the RunPod API without SSH.
cat >/usr/local/bin/mstar-report <<'EOF'
#!/bin/bash
out=$(vibesys report --results /workspace/results.jsonl "$@") || exit
case "$out" in
  "{"*) printf '[result %s] %s\n' "$POD_NAME" "$out" >/proc/1/fd/1 ;;
  *) echo "$out" ;;
esac
EOF
chmod +x /usr/local/bin/vibesys /usr/local/bin/mstar-report
# A restart empties the pod log, so the counted rows are printed into it again.
if [ -s "$W/results.jsonl" ]; then
  grep '"status": *"counted"' "$W/results.jsonl" | sed "s/^/[result $POD_NAME] /"
fi

# Paths in Claude Code permission rules are absolute when they start with //.
cat >"$W/loop/settings.json" <<JSON
{
  "permissions": {
    "allow": ["Bash", "Read", "Glob", "Grep", "Edit(/$W/**)", "Write(/$W/**)", "Task", "TodoWrite"],
    "deny": ["Edit(/$W/loop/**)", "Write(/$W/loop/**)", "Edit(/$W/vibesys/**)", "Write(/$W/vibesys/**)",
             "Edit(/$W/results.jsonl)", "Write(/$W/results.jsonl)",
             "Bash(git push:*)", "Bash(git remote:*)", "Bash(gh:*)", "WebFetch", "WebSearch"]
  }
}
JSON

if [ -s /root/.fork_token ]; then
  cat >/usr/local/bin/pod-push <<EOF
#!/bin/bash
cd $W/mstar || exit 1
while true; do
  refs=(\$(git for-each-ref --format='%(refname)' refs/heads/perf/$POD_NAME 'refs/heads/perf/$POD_NAME-*'))
  git -c credential.helper= \\
    -c 'credential.helper=!f() { echo username=x-access-token; printf "password=%s\\n" "\$(cat /root/.fork_token)"; }; f' \\
    push -q --force https://github.com/$FORK.git run-base:refs/heads/perf/$POD_NAME-base "\${refs[@]}" 2>&1 |
    tail -2 | sed 's/^/[push] /'
  sleep 900
done
EOF
  chmod +x /usr/local/bin/pod-push
  tmux has-session -t push 2>/dev/null || tmux new-session -d -s push "pod-push >/proc/1/fd/1 2>&1" ||
    log "FAIL push job"
fi

if [ -n "${BRIEF_GZB64:-}" ]; then
  echo "$BRIEF_GZB64" | base64 -d | gunzip >"$W/briefs/brief.new" && mv "$W/briefs/brief.new" "$W/briefs/brief.md" ||
    { log "FAIL brief decode"; exit 1; }
fi
[[ "${LOOP_DEADLINE:-}" =~ ^[0-9]+$ ]] && echo "$LOOP_DEADLINE" >"$W/loop/deadline"
if [ ! -s "$W/briefs/brief.md" ]; then
  log "READY, no brief"
elif [ ! -s "$CLAUDE_CONFIG_DIR/.credentials.json" ]; then
  log "READY, waiting for a Claude login"
elif tmux has-session -t loop 2>/dev/null; then
  log "READY, loop already running"
elif tmux new-session -d -s loop \
  ". /root/.pod_env; bash $W/vibesys/benchmark/vibesys/loop.sh $W/briefs/brief.md $WT $W/loop >/proc/1/fd/1 2>&1"; then
  log "READY, loop started ($(wc -c <"$W/briefs/brief.md") byte brief)"
else
  log "FAIL loop start"
fi
