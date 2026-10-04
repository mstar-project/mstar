#!/bin/bash
# Validate the speculative loop-counter fixes for concurrent klein image generation.
#
# Run it inside an existing allocation (an `srun --gres=gpu:2 ... bash` is fine);
# it serves on ONE GPU and does not call sbatch.
#
#     bash validate_loop_iter_fix.sh                 # uses $PWD as the repo
#     REPO=/path/to/mstar bash validate_loop_iter_fix.sh
#
# Two checks, each comparing a request served CONCURRENTLY against the same
# request served ALONE. They must match (>= 40 dB; identical is `inf`):
#
#   1. image edit        benchmark/flux2_klein/edit_probe.py, concurrency 4
#   2. text-to-image     4 concurrent generations vs the same 4 serially
#
# Before the fix both FAIL at ~10-18 dB, because a request folded into a
# speculative batch re-runs denoise step 0 (sigma ~ 1.0) over already-denoised
# latents and the image comes back as patch-scale noise.
#
# Prints a PASS/FAIL line per check and exits non-zero if either fails.
set -u

REPO=${REPO:-$PWD}
OUT=${OUT:-$REPO/.bench_outs/loop_iter_fix}
PORT=${PORT:-$((20000 + RANDOM % 10000))}
STEPS=${STEPS:-4}
cd "$REPO" || { echo "no such repo: $REPO"; exit 2; }
mkdir -p "$OUT"
LOG=$OUT/serve.log

# Activate the venv if there is one and we are not already inside it.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
command -v mstar >/dev/null || { echo "mstar not on PATH -- activate the env first"; exit 2; }

# The in-flight-flag fix is PYTHON-ONLY so far: the Rust runtime has the same two
# holes (runtime.rs:736/768 force the flag down instead of restoring it) and has not
# been ported. Default to the Python runtime so the fix is actually exercised.
export MSTAR_RUST_GRAPH=${MSTAR_RUST_GRAPH:-0}

echo "repo=$REPO  port=$PORT  steps=$STEPS"
echo "git:  $(git rev-parse --short HEAD 2>/dev/null)  $(git status --porcelain 2>/dev/null | grep -cv '^??') modified file(s)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -4

echo "=== starting server (default config: async_scheduling ON) $(date -u +%H:%M:%S) ==="
mstar serve flux2_klein --port "$PORT" > "$LOG" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }
trap cleanup EXIT

for _ in $(seq 1 420); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$SRV" 2>/dev/null || { echo "server died:"; tail -40 "$LOG"; exit 1; }
    sleep 10
done
curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 || {
    echo "server never became healthy:"; tail -40 "$LOG"; exit 1; }
echo "  ready; runtime: $(grep -o 'graph runtime: [a-z]*' "$LOG" | head -1)"

RC=0

echo
echo "=== 1/2  image edit, concurrency 4  $(date -u +%H:%M:%S) ==="
if [ -f benchmark/flux2_klein/edit_probe.py ]; then
    python benchmark/flux2_klein/edit_probe.py --url "http://localhost:$PORT" \
        --out-dir "$OUT/edit" --concurrency 4 --rounds 1 --steps "$STEPS" \
        --json "$OUT/edit.json" 2>&1 | grep -E "in a batch|two-reference|PASS|FAIL"
    # edit_probe prints its verdict but always exits 0, so read the JSON instead
    python - "$OUT/edit.json" <<'PY' || RC=1
import json, sys
w = json.load(open(sys.argv[1])).get("worst_psnr_db")
ok = w is None or w >= 40          # null means "identical" (inf)
print(f"EDIT: {'PASS' if ok else 'FAIL'} (worst {'inf' if w is None else f'{w:.2f}'} dB, need >= 40)")
sys.exit(0 if ok else 1)
PY
else
    echo "  edit_probe.py not on this branch -- skipping"
fi

echo
echo "=== 2/2  text-to-image, 4 concurrent vs 4 serial  $(date -u +%H:%M:%S) ==="
python - "$PORT" "$STEPS" "$OUT" <<'PY' || RC=1
import sys, io, math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from mstar import MStarClient

port, steps, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
(out / "t2i").mkdir(parents=True, exist_ok=True)
c = MStarClient(f"http://localhost:{port}")
PROMPTS = [
    "A cat holding a sign that says hello world, studio lighting, detailed fur",
    "An astronaut riding a horse on Mars, cinematic wide shot",
    "A bowl of ramen on a wooden table, soft window light, steam",
    "A watercolor painting of a lighthouse in a storm",
]

def gen(i):
    return c.generate_image(prompt=PROMPTS[i], size="1024x1024", seed=i,
                            num_inference_steps=steps)

def psnr(a, b):
    x = np.asarray(Image.open(io.BytesIO(a)).convert("RGB"), np.float64)
    y = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), np.float64)
    if x.shape != y.shape:
        return float("nan")
    mse = float(np.mean((x - y) ** 2))
    return math.inf if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)

solo = [gen(i) for i in range(4)]
with ThreadPoolExecutor(max_workers=4) as ex:
    batched = list(ex.map(gen, range(4)))

worst = math.inf
for i, (b, s) in enumerate(zip(batched, solo)):
    v = psnr(b, s)
    worst = min(worst, v)
    (out / "t2i" / f"concurrent_{i}.png").write_bytes(b)
    (out / "t2i" / f"solo_{i}.png").write_bytes(s)
    print(f"  t2i {i} concurrent vs solo: {'inf' if math.isinf(v) else f'{v:.2f}'} dB")
ok = worst >= 40
print(f"T2I: {'PASS' if ok else 'FAIL'} (worst {'inf' if math.isinf(worst) else f'{worst:.2f}'} dB, need >= 40)")
sys.exit(0 if ok else 1)
PY

echo
echo "=== per-request denoise step sequences (from prepare_inputs) ==="
# A clean request is k=0,1,2,...  A repeat (0,0,2,3 / 0,1,2,0) is the loop-counter bug.
# check_stop reads a different info object and can look clean while this does not.
echo "  (a bad pass reading k=0: key_present=False means an unseeded fwd_info,"
echo "   key_present=True means a seeded one that never advanced)"
grep -E "prepare request .* k=0/.* key_present=False" "$LOG" 2>/dev/null | tail -5
grep -o "prepare request [0-9a-f-]* k=[0-9]*" "$LOG" 2>/dev/null \
  | awk '{split($NF,a,"="); print $3, a[2]}' \
  | awk '{seq[$1]=seq[$1]" "$2} END {for (r in seq) print substr(r,1,8)":"seq[r]}' \
  | sort | awk '{n=split($0,f," "); ok=1; for(i=2;i<=n;i++) if(f[i]+0 != i-2) ok=0;
                 print $0 (ok?"":"   <<< REPEATED/MISSING STEP")}'
echo
echo "=== images in $OUT ; server log $LOG ==="
[ "$RC" = 0 ] && echo "RESULT: PASS -- concurrent requests match their solo images" \
              || echo "RESULT: FAIL -- the loop-counter bug is still reachable"
exit $RC
