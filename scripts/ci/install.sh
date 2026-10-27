#!/usr/bin/env bash
# Install one Mac check's dependencies, following .github/workflows/ci.yml.
# The caller supplies Python 3.12 and, for Rust, Cargo and libzmq development files.
set -euo pipefail

if [[ $# != 1 || ! $1 =~ ^(lint|cpu-smoke|dynamo|rust)$ ]]; then
    echo "Usage: $0 {lint|cpu-smoke|dynamo|rust}" >&2
    exit 2
fi
job=$1
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
python -c 'import sys; sys.version_info[:2] == (3, 12) or sys.exit("CI requires Python 3.12")'
python -m pip install --upgrade pip

case "$job" in
    lint)
        python -m pip install ruff
        ;;
    cpu-smoke|dynamo)
        python -m pip install -r scripts/ci/cpu-torch.txt \
            --index-url https://download.pytorch.org/whl/cpu
        if [[ $job == cpu-smoke ]]; then
            # flex_attention imports torch._inductor, which needs Triton even
            # on CPU; 3.6.0 is the version torch 2.10.0 pins.
            python -m pip install triton==3.6.0
            python -m pip install -e '.[dev,waypoint]'
            # BAGEL's CPU graph/KV tests need these without the CUDA extras.
            python -m pip install einops regex transformers
            python -m pip check
        else
            python -m pip install -e '.[dynamo]' pytest
        fi
        ;;
    rust)
        python -m pip install maturin
        # pyyaml: runtime tests import mstar.model.base.
        python -m pip install pytest pyzmq msgpack numpy pyyaml
        # The tensor-transport stack imports torch and Triton. Preserve this
        # job's existing unpinned versions and minimal mstar installation.
        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        python -m pip install triton
        python -m pip install --no-deps -e .
        ;;
esac
