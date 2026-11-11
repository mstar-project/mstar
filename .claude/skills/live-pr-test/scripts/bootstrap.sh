#!/usr/bin/env bash
# Prepare a GPU box to live-test one PR: a clone + venv + Rust extension per arm.
#
# usage: bootstrap.sh <pr-number> [--workdir DIR] [--hf-home DIR] [--base BRANCH] [--ref SHA] [--no-main] [--untrusted] [--no-restore]
#
#   --ref SHA pins the PR arm to that commit rather than whatever the PR's head is now.
#   --no-restore starts from an empty compile directory (a cache refresh keeps only what it builds).
#
#   $WORKDIR/pr     PR head (refs/pull/<n>/head), its own venv and mstar_rust build
#   $WORKDIR/main   the PR's base (origin/main, or --base for a stacked PR), likewise (skipped with --no-main)
#   $WORKDIR/results/{media,bench,logs}   everything the run produces; copied back at the end
#   $WORKDIR/env.sh   source this in every shell before using an arm
#
# --untrusted: the PR needed the trust question (SKILL.md step 1). Its compile caches hold code
# that later runs load (Inductor's FX cache runs Python, Triton and FlashInfer load .so files),
# so they go in $WORKDIR/compile rather than the HOME cache other runs share, and save_caches
# refuses to put them in the shared tarball.
#
# Each arm gets its own venv because the editable install maps `mstar` to one checkout's
# path, and a PR that touches rust/ needs its own mstar_rust: one shared venv would make
# every A/B a checkout + rebuild between arms (see the benchmarking skill). Both venvs are
# resolved back to back so their dependency versions match; env.txt records any diff.
#
# Runs on the box, not on your laptop. Idempotent: rerunning updates both arms.
set -euo pipefail

USAGE="usage: bootstrap.sh <pr-number> [--workdir DIR] [--hf-home DIR] [--base BRANCH] [--ref SHA] [--no-main] [--untrusted] [--no-restore]"
# before anything is created or installed: a mistaken run (--help, a typo) changes nothing
case ${1:-} in
  -h|--help) echo "$USAGE"; exit 0;;
  ''|*[!0-9]*) echo "bootstrap.sh: the first argument is the PR number, got '${1:-}'" >&2; echo "$USAGE" >&2; exit 2;;
esac
PR=$1
shift
WORKDIR=$HOME/live-pr
HF=${HF_HOME:-}
WITH_MAIN=1
BASE=main
REF=
UNTRUSTED=0
RESTORE=1
while [ $# -gt 0 ]; do
  case $1 in
    --workdir) WORKDIR=$2; shift 2;;
    --hf-home) HF=$2; shift 2;;
    --base) BASE=$2; shift 2;;
    --ref) REF=$2; shift 2;;
    --no-main) WITH_MAIN=0; shift;;
    --untrusted) UNTRUSTED=1; shift;;
    --no-restore) RESTORE=0; shift;;
    -h|--help) echo "$USAGE"; exit 0;;
    *) echo "bootstrap.sh: unknown option $1" >&2; echo "$USAGE" >&2; exit 2;;
  esac
done
[ -n "$HF" ] || { echo "bootstrap.sh: pass --hf-home (the persistent HF cache)" >&2; exit 1; }
REPO=https://github.com/mstar-project/mstar.git
mkdir -p "$WORKDIR"/results/{media,bench,logs}

# Caches stay on local disk: RunPod's /workspace is S3-backed (geesefs) and refuses chmod,
# which breaks uv's unpacking and git, and reads through it are slow: a TP4 load of a 27B
# checkpoint mmap'd from it paged in at ~4 MB/s per worker, and even a sequential copy ran
# at ~100 MB/s against ~1 GB/s from the Hub. So servers read models from local disk
# (stage_model), and compile caches travel as one tarball (save_caches).
# On a non-FUSE filesystem (cluster NFS, Lambda) servers read the HF cache in place.
mkdir -p "$HF"
HF_SERVE=$HF
case $(stat -f -c %T "$HF") in
  fuse*) HF_SERVE=$HOME/hf;;
esac
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .) || true
[[ $SM =~ ^[0-9]+$ ]] || SM=unknown  # a failed nvidia-smi prints its error on stdout
CACHE_TAR=$(dirname "$HF")/live-pr-compile-cache/sm$SM.tar
[ ! -e "$WORKDIR/compile/UNTRUSTED" ] || UNTRUSTED=1  # a rerun of an untrusted work directory stays untrusted
COMPILE=$HOME/.cache/compile
[ "$UNTRUSTED" = 0 ] || COMPILE=$WORKDIR/compile
mkdir -p "$COMPILE"
[ "$UNTRUSTED" = 0 ] || touch "$COMPILE/UNTRUSTED"  # in the cache itself, so no later env.sh can clear it
cat > "$WORKDIR/env.sh" <<EOF
export HF_PERSIST=$HF HF_HOME=$HF_SERVE
export UV_CACHE_DIR=\$HOME/.cache/uv PIP_CACHE_DIR=\$HOME/.cache/pip
export UV_LINK_MODE=hardlink
export PATH=\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH
export MSTAR_SHM_ARENA=1
# Inductor/Triton autotune results and FlashInfer JIT builds, restored from and saved to
# CACHE_TAR so a new box doesn't recompile every kernel. Keyed by compute capability.
export COMPILE_CACHE=$COMPILE
export TORCHINDUCTOR_CACHE_DIR=$COMPILE/inductor TRITON_CACHE_DIR=$COMPILE/triton
export FLASHINFER_WORKSPACE_BASE=$COMPILE
export CACHE_TAR="$CACHE_TAR"
export LIVE_PR=$PR LIVE_WORKDIR=$WORKDIR LIVE_RESULTS=$WORKDIR/results
# arm <pr|main>: activate that arm's venv and cd into its checkout
arm() { deactivate 2>/dev/null || true; cd "$WORKDIR/\$1" && source .venv/bin/activate; }
# stage_model <repo>: put a model in the local HF cache servers read. The Hub is the fast
# source; the persistent cache is the fallback (a gated repo, no token, Hub down).
# Run it before a model's first server start, never while a server holds its files.
stage_model() {
  [ "\$HF_PERSIST" = "\$HF_HOME" ] && return 0
  local d=models--\${1//\//--}
  [ -d "\$HF_HOME/hub/\$d/snapshots" ] && return 0
  "$WORKDIR/pr/.venv/bin/hf" download "\$1" >/dev/null && return 0
  [ -d "\$HF_PERSIST/hub/\$d/snapshots" ] || { echo "stage_model: \$1 not on the Hub or in \$HF_PERSIST" >&2; return 1; }
  mkdir -p "\$HF_HOME/hub/\$d" && cp -rL "\$HF_PERSIST/hub/\$d/refs" "\$HF_PERSIST/hub/\$d/snapshots" "\$HF_HOME/hub/\$d/"
}
# save_caches: write the compile caches back to persistent storage (before terminating)
save_caches() {
  [ ! -e "\$COMPILE_CACHE/UNTRUSTED" ] || { echo "save_caches: these caches ran a PR that needed the trust check (--untrusted)" >&2; return 1; }
  mkdir -p "\$(dirname "\$CACHE_TAR")" && tar -C "\$(dirname "\$COMPILE_CACHE")" -cf "\$HOME/compile-cache.tar" compile \
    && cp "\$HOME/compile-cache.tar" "\$CACHE_TAR.tmp" && mv "\$CACHE_TAR.tmp" "\$CACHE_TAR" && ls -lh "\$CACHE_TAR"
}
EOF
# shellcheck disable=SC1091
source "$WORKDIR/env.sh"
mkdir -p "$HF" "$HF_HOME"
if [ "$RESTORE" = 1 ] && [ -f "$CACHE_TAR" ]; then
  cp "$CACHE_TAR" "$HOME/compile-cache.tar" && tar -C "$(dirname "$COMPILE")" -xf "$HOME/compile-cache.tar" \
    && echo "compile caches restored from $CACHE_TAR ($(du -sh "$COMPILE" | cut -f1))"
fi

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

checkout() {  # checkout <dir> <ref> <refspec>
  local dir=$1 ref=$2 spec=$3
  [ -d "$dir/.git" ] || git clone -q "$REPO" "$dir"
  git -C "$dir" fetch -q origin "$spec"
  git -C "$dir" checkout -q --detach "$ref"
}

build_arm() {  # build_arm <dir>
  local dir=$1
  (
    cd "$dir"
    [ -d .venv ] || uv venv -q --python 3.12 --seed
    source .venv/bin/activate
    uv pip install -q --torch-backend=auto -e ".[all]" maturin
    maturin develop -q --release -m rust/Cargo.toml
    # mstar-server, the HTTP frontend mstar serve launches
    [ ! -d rust/server ] || (cd rust/server && cargo build --release -q)
    find mstar -name __pycache__ -type d -prune -exec rm -rf {} +
    python -c "import mstar, mstar_rust, mstar.conductor.conductor, mstar.worker.worker; print('import ok:', mstar.__file__)"
  )
}

if [ -n "$REF" ]; then checkout "$WORKDIR/pr" "$REF" "$REF"; else checkout "$WORKDIR/pr" FETCH_HEAD "pull/$PR/head"; fi
build_arm "$WORKDIR/pr"
if [ "$WITH_MAIN" = 1 ]; then
  checkout "$WORKDIR/main" FETCH_HEAD "$BASE"
  build_arm "$WORKDIR/main"
fi

{
  echo "pr $PR head: $(git -C "$WORKDIR/pr" rev-parse HEAD)"
  [ "$WITH_MAIN" = 1 ] && echo "base $BASE: $(git -C "$WORKDIR/main" rev-parse HEAD)"
  git -C "$WORKDIR/pr" fetch -q origin "$BASE"
  echo "merge-base with $BASE: $(git -C "$WORKDIR/pr" merge-base HEAD FETCH_HEAD 2>/dev/null || echo unknown)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  echo "cpus: $(nproc)  mem: $(free -g | awk '/Mem:/{print $2}')G"
  "$WORKDIR/pr/.venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  [ "$RESTORE" = 0 ] || [ ! -f "$CACHE_TAR.info" ] || echo "compile cache: $(paste -sd' ' "$CACHE_TAR.info")"
  if [ "$WITH_MAIN" = 1 ]; then
    echo "--- dependency diff (base vs pr venv) ---"
    diff <(cd "$WORKDIR/main" && uv pip freeze -q --python .venv/bin/python | grep -v '^-e') \
         <(cd "$WORKDIR/pr" && uv pip freeze -q --python .venv/bin/python | grep -v '^-e') || true
  fi
} | tee "$WORKDIR/results/env.txt"
df -h "$WORKDIR" | tail -1
echo BOOTSTRAP_OK
