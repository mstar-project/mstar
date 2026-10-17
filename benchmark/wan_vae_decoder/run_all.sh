#!/usr/bin/env bash
# The headline numbers on one GPU: latents and the fp32 reference, stock
# diffusers, and the change ladder through M*'s streaming decoder node
# (Cosmos3VAEDecoderARSubmodule), one variant per process with fresh compile
# caches. Refuses to run on a GPU with other compute processes.
#
#   GPU=0 VIDEO=clip.mp4 bash run_all.sh              # Wan 2.1 VAE
#   GPU=0 VIDEO=clip.mp4 WAN22=1 bash run_all.sh      # also the Wan 2.2 VAE
#
# Env: GPU (index, default 0), MSTAR (checkout whose model code the M* variants
# import, default this repo), RUN (run name), SUITES, WAN22, VIDEO (source clip;
# random latents without it), VARIANT_TIMEOUT (s, default 2400). Results (one
# JSON per variant, median ms per chunk, peak memory, PSNR) go to runs/$RUN/.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MSTAR="${MSTAR:-$(cd "$HERE/../.." && pwd)}"
GPU="${GPU:-0}"
RUN="${RUN:-$(date +%Y%m%d-%H%M%S)}"
OUT="$HERE/runs/$RUN"
VIDEO="${VIDEO:-}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
export PATH="$MSTAR/.venv/bin:$PATH" MSTAR HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU"
cd "$HERE"

others="$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"
if [ -n "$others" ]; then
  echo "[run_all] GPU $GPU has other compute processes; not starting:"
  echo "$others"
  exit 2
fi
nvidia-smi -i "$GPU" --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv | tee "$OUT/gpu.csv"

run() {  # run <results-subdir> <bench args...>: one variant, fresh Inductor/Triton caches
  local sub="$1"; shift
  local cache="$OUT/cache/$sub-$*"
  export TORCHINDUCTOR_CACHE_DIR="${cache// /_}/inductor" TRITON_CACHE_DIR="${cache// /_}/triton"
  echo "[run_all] $(date +%T) bench $*"
  timeout "${VARIANT_TIMEOUT:-2400}" python bench_wan_decoder.py --out "$OUT/$sub" "$@" \
    || echo "[run_all] FAILED (exit $?): $*"
}

suite() {  # suite <wan21|wan22>
  local vae="$1" dir="$OUT/$1"
  local lat="$dir/latents.pt" ref="$dir/ref_fp32_outputs.pt"
  mkdir -p "$dir"
  echo "[run_all] $(date +%T) prepare latents ($vae)"
  if [ -f "$VIDEO" ]; then
    python prepare_latents.py --vae "$vae" --video "$VIDEO" --chunks 85 --out "$lat"
  else
    python prepare_latents.py --vae "$vae" --random --chunks 85 --out "$lat"
  fi
  local c=(--vae "$vae" --latents "$lat")
  run "$vae" --variant ref_fp32 "${c[@]}" --repeats 1 --save-ref "$ref"
  c+=(--ref "$ref")
  run "$vae" --variant diffusers_bf16 "${c[@]}"
  for step in 0 1 2; do run "$vae" --variant "mstar_step$step" "${c[@]}"; done
  run "$vae" --variant mstar_warm "${c[@]}"        # this change's defaults (one latent per decoder call)
  run "$vae" --variant mstar_warm_lpc0 "${c[@]}"   # opt-in: a window's new latents in one decoder call
  python - "$dir" <<'EOF'
import json, pathlib, sys
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    if p.name.endswith(".latencies.json"):
        continue
    r = json.loads(p.read_text())
    print(f"{r['variant']:<18} {r['median_ms']:>8.2f} ms/chunk  peak {r.get('peak_allocated_gib', '-')} GiB"
          f"  PSNR u8 {r.get('psnr_u8_db_vs_fp32', '-')}")
EOF
}

SUITES="${SUITES:-wan21}"
[ "${WAN22:-0}" = 1 ] && SUITES="$SUITES wan22"
for s in $SUITES; do suite "$s"; done
echo "[run_all] $(date +%T) done -> $OUT"
