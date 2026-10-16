#!/usr/bin/env bash
# Prepare a GPU box to live-test one PR: a clone + venv + Rust extension per arm.
#
# usage: bootstrap.sh <pr-number> [--workdir DIR] [--hf-home DIR] [--base BRANCH] [--no-main]
#
#   $WORKDIR/pr     PR head (refs/pull/<n>/head), its own venv and mstar_rust build
#   $WORKDIR/main   the PR's base (origin/main, or --base for a stacked PR), likewise (skipped with --no-main)
#   $WORKDIR/results/{media,bench,logs}   everything the run produces; copied back at the end
#   $WORKDIR/env.sh   source this in every shell before using an arm
#
# Each arm gets its own venv because the editable install maps `mstar` to one checkout's
# path, and a PR that touches rust/ needs its own mstar_rust: one shared venv would make
# every A/B a checkout + rebuild between arms (see the benchmarking skill). Both venvs are
# resolved back to back so their dependency versions match; env.txt records any diff.
#
# Runs on the box, not on your laptop. Idempotent: rerunning updates both arms.
set -euo pipefail

PR=${1:?usage: bootstrap.sh <pr-number> [--workdir DIR] [--hf-home DIR] [--base BRANCH] [--no-main]}
shift
WORKDIR=$HOME/live-pr
HF=${HF_HOME:-}
WITH_MAIN=1
BASE=main
while [ $# -gt 0 ]; do
  case $1 in
    --workdir) WORKDIR=$2; shift 2;;
    --hf-home) HF=$2; shift 2;;
    --base) BASE=$2; shift 2;;
    --no-main) WITH_MAIN=0; shift;;
    *) echo "bootstrap.sh: unknown option $1" >&2; exit 1;;
  esac
done
[ -n "$HF" ] || { echo "bootstrap.sh: pass --hf-home (the persistent HF cache)" >&2; exit 1; }
REPO=https://github.com/mstar-project/mstar.git
mkdir -p "$WORKDIR"/results/{media,bench,logs}

# Caches stay on local disk: RunPod's /workspace is S3-backed (geesefs) and refuses chmod,
# which breaks uv's unpacking and git. Only the HF cache lives on persistent storage.
cat > "$WORKDIR/env.sh" <<EOF
export HF_HOME=$HF
export UV_CACHE_DIR=\$HOME/.cache/uv PIP_CACHE_DIR=\$HOME/.cache/pip
export UV_LINK_MODE=hardlink
export PATH=\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH
export MSTAR_SHM_ARENA=1
export LIVE_PR=$PR LIVE_WORKDIR=$WORKDIR LIVE_RESULTS=$WORKDIR/results
# arm <pr|main>: activate that arm's venv and cd into its checkout
arm() { deactivate 2>/dev/null || true; cd "$WORKDIR/\$1" && source .venv/bin/activate; }
EOF
# shellcheck disable=SC1091
source "$WORKDIR/env.sh"
mkdir -p "$HF"

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
    find mstar -name __pycache__ -type d -prune -exec rm -rf {} +
    python -c "import mstar, mstar_rust, mstar.conductor.conductor, mstar.worker.worker; print('import ok:', mstar.__file__)"
  )
}

checkout "$WORKDIR/pr" FETCH_HEAD "pull/$PR/head"
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
  if [ "$WITH_MAIN" = 1 ]; then
    echo "--- dependency diff (base vs pr venv) ---"
    diff <(cd "$WORKDIR/main" && uv pip freeze -q --python .venv/bin/python | grep -v '^-e') \
         <(cd "$WORKDIR/pr" && uv pip freeze -q --python .venv/bin/python | grep -v '^-e') || true
  fi
} | tee "$WORKDIR/results/env.txt"
df -h "$WORKDIR" | tail -1
echo BOOTSTRAP_OK
