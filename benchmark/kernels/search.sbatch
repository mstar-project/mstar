#!/bin/bash
#SBATCH -J kernel-search
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o kernel-search-%j.log
# One kernel search on one GPU, start to finish: probe the registry, build the
# candidates, check them against the reference, time them, write the results
# store and the table, exit. Pass the site's partition (and account or QOS) on
# the command line, from the repository root:
#
#   sbatch -p <gpu partition> [--qos=...] benchmark/kernels/search.sbatch rms_norm \
#       --store /persistent/kernel-results --budget 8
#
# KERNEL_SEARCH_SETUP, if set, is sourced first (a venv or module loads).
# Searches are rankings on the device they ran on; a winner ships only after a
# model-level A/B.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
[ -n "${KERNEL_SEARCH_SETUP:-}" ] && source "$KERNEL_SEARCH_SETUP"
SPEC=${1:?usage: search.sbatch <spec> [benchmark.kernels.search arguments]}
shift
OUT=${KERNEL_SEARCH_OUT:-kernel-search-${SLURM_JOB_ID:-local}}
mkdir -p "$OUT"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} start=$(date -Is) head=$(git rev-parse --short HEAD 2>/dev/null)"
python -m mstar.kernels --probe | tee "$OUT/registry.txt"
python -m benchmark.kernels.search --spec "$SPEC" --job "${SLURM_JOB_ID:-local}" \
    --table "$OUT/$SPEC.md" --json "$OUT/$SPEC.json" "$@"
echo "end=$(date -Is)"
