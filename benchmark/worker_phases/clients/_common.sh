#!/bin/bash
# Shared client settings. Sourced, not run.
#
# Closed-loop keeps a fixed number of requests in flight, so the worker sits
# at a steady batch size for most of the run -- which is what the server
# wrapper segments on. Override anything with an env var.
set -uo pipefail
URL=${URL:-http://127.0.0.1:8000}
CONC=${CONC:-16}
WARMUP=${WARMUP:-2}
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
run_bench() { echo "+ python -m benchmark.runner --url $URL $*"; python -m benchmark.runner --url "$URL" "$@"; }
