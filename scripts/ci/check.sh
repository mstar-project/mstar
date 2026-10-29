#!/usr/bin/env bash
# Run the optional Mac checks inside a prepared Linux ARM container.
set -euo pipefail

if [[ $# != 1 || ! $1 =~ ^(lint|cpu-smoke|dynamo|rust)$ ]]; then
    echo "Usage: $0 {lint|cpu-smoke|dynamo|rust}" >&2
    exit 2
fi
job=$1
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
python -c 'import sys; sys.version_info[:2] == (3, 12) or sys.exit("CI requires Python 3.12")'
report_dir=${CI_REPORT_DIR:-$repo_root}
if [[ $job != lint ]]; then
    mkdir -p "$report_dir"
fi

case "$job" in
    lint)
        python -m ruff check --output-format=github .
        python -m unittest discover -s test/ci -v
        ;;
    cpu-smoke)
        # Portable model tests still compile C++ kernels.
        command -v "${CXX:-c++}" >/dev/null || {
            echo "CPU smoke requires a C++ compiler (for example, g++)." >&2
            exit 1
        }
        python scripts/ci/select_cpu_tests.py > "$report_dir/cpu-smoke-tests.txt"
        # PyTorch 2.9.1's oneDNN BF16 Linear crashes in the tested Apple VM.
        # Keep BF16 tests, using PyTorch's CPU fallback in this process only.
        HF_HUB_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
            python -c '
import runpy
import torch
torch.backends.mkldnn.enabled = False
print("Mac CPU smoke: oneDNN disabled for ARM BF16 compatibility", flush=True)
runpy.run_module("pytest", run_name="__main__")
' -q -ra --strict-markers --durations=20 \
            --junitxml="$report_dir/cpu-smoke.xml" "@$report_dir/cpu-smoke-tests.txt"
        ;;
    dynamo)
        export HF_HUB_OFFLINE=1
        # These tests otherwise skip when the native Dynamo bindings are absent.
        python -c 'import dynamo.llm; import dynamo.runtime'
        python -m pytest test/modular/test_dynamo_bridges.py \
            test/modular/test_dynamo_registration_smoke.py -v \
            --junitxml="$report_dir/dynamo.xml"
        ;;
    rust)
        # Native tests link libpython (unlike an extension module). Resolve its
        # location from the active interpreter, including outside setup-python.
        python_lib=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
        export LD_LIBRARY_PATH="$python_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        (cd rust && cargo test --release --no-default-features --features python)
        (cd rust/server && cargo build --release)

        # A fresh wheel directory prevents installing an earlier cached build.
        wheel_dir=$(mktemp -d)
        trap 'rm -rf -- "$wheel_dir"' EXIT
        (cd rust && maturin build --release --out "$wheel_dir")
        python -m pip install --force-reinstall --no-deps "$wheel_dir"/*.whl

        # Require the artifacts just built; do not accept skip-only success or
        # select an unrelated mstar-server from the user's PATH.
        export MSTAR_SERVER_BIN="$repo_root/rust/server/target/release/mstar-server"
        test -x "$MSTAR_SERVER_BIN"
        python -c 'import mstar_rust; from mstar.api_server.rust_frontend import find_server_binary; print(find_server_binary())'
        python -m pytest test/rust/ -v --junitxml="$report_dir/rust.xml"
        ;;
esac
