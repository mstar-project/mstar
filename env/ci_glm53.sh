#!/usr/bin/env bash
# CI gate for GLM-5.3-Flash (glm5_next) on the v1 resource-pool engine. One
# entry point, laptop and box: on a GPU-less machine the GPU section reports
# SKIP and the rest must still be green — the port is CPU-verified first.
#
# Steps (each lands in the summary table; any FAIL exits non-zero):
#   ruff      mstar/model/glm5_next, the two engine resources the port added
#             (attn/mla*, slot_state/) and the lane's test files below
#   modular   the glm5_next modular tests PLUS the engine tests the port
#             leans on (test_mla_attention.py, test_slot_state_resource.py):
#             each file in ITS OWN pytest process first (no shared-process
#             stub pollution), then all together (catches cross-file leaks).
#             test/modular/conftest.py stubs triton; flashinfer is stubbed
#             per test where a kernel path needs it.
#   registry  CPU import contract, run UNDER PYTEST
#             (test/modular/test_glm5next_registry_contract.py): main's
#             sampler does an unguarded `import triton`, so the lane's
#             bare-python tier cannot pass on a laptop venv. The test keeps
#             both tiers: HARD (fresh interpreter: kda/mhc/config/kda_state/
#             weight_loader import with NO flashinfer/triton in sys.modules)
#             and TOLERANT (full model module + the lazy MODEL_REGISTRY
#             (module, class) tuple resolves; a missing non-mstar module
#             skips, a missing mstar.* module fails).
#   loader    real-checkpoint header smoke (env/smoke_glm53_loader.py) —
#             opt-in: needs the GLM-5.3-Flash snapshot (GLM53_CKPT or the
#             HF cache); absent = SKIP.
#   gpu       test/integration/test_glm5next_*.py one-process-each, opt-in:
#             runs only where nvidia-smi sees a device, or GLM53_GPU=1
#             forces the attempt (the files skip themselves off CUDA).
#
# usage (from anywhere; the script cd's to its own repo root):
#   bash env/ci_glm53.sh                     # .venv next to the repo root
#   VENV=/path/to/venv bash env/ci_glm53.sh  # any venv with torch + pytest + ruff
#   PY=/path/to/python bash env/ci_glm53.sh  # or name the interpreter directly
#                                            # (ruff is then taken from PATH)
# VENV defaults to .venv; the laptop scratch venv is not portable, pass it
# explicitly. The box venvs (/data/kvasilev-cd4cf3/... on Laude,
# $P/mstar/.venv on coriander) go through VENV= as well.
set -euo pipefail
shopt -s nullglob

cd "$(cd "$(dirname "$0")/.." && pwd)"
# Resolve `mstar` to THIS worktree for every step, incl. scripts run without
# -m (the loader smoke): a box venv may have another checkout's mstar in
# site-packages, and PYTHONPATH is searched first.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# The CPU tiers must not see a GPU even on a box: the modular tests build on
# device cpu and the conftest stubs assume no triton. GLM53_GPU=1 (or a real
# nvidia-smi) opts the integration tier in below, in its own processes.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
VENV=${VENV:-.venv}
if [ -n "${PY:-}" ]; then
  RUFF=${RUFF:-ruff}
else
  PY=$VENV/bin/python
  RUFF=${RUFF:-$VENV/bin/ruff}
fi
[ -x "$PY" ] || { echo "FATAL: no python at $PY (VENV=$VENV; pass VENV=... or PY=...)" >&2; exit 2; }
command -v "$RUFF" >/dev/null 2>&1 || { echo "FATAL: no ruff at $RUFF (pip install ruff into the venv)" >&2; exit 2; }
PYTEST=("$PY" -m pytest -q -p no:cacheprovider)

NAMES=(); CODES=()   # parallel arrays: macOS bash 3.2 has no declare -A
record() { NAMES+=("$1"); CODES+=("$2"); }
step() {             # step <name> <cmd...> — run, record, never abort the run
  local name=$1 rc=0; shift
  echo "──── $name"
  "$@" || rc=$?
  case $rc in        # 75 = EX_TEMPFAIL, the deliberate skip code below
    0)  record "$name" PASS ;;
    75) record "$name" SKIP ;;
    *)  record "$name" "FAIL($rc)" ;;
  esac
}
cpu_env() { CUDA_VISIBLE_DEVICES="" "$@"; }

# The lane's modular files + the two engine-resource suites the port added.
# An engine suite that vanished is a FAIL in the table, not a quiet shrink.
MOD=(test/modular/test_glm5next_*.py)
ENGINE=()
for f in test/modular/test_mla_attention.py test/modular/test_slot_state_resource.py; do
  if [ -f "$f" ]; then ENGINE+=("$f"); else record "modular:$(basename "$f")" "FAIL(missing)"; fi
done
GPU=(test/integration/test_glm5next_*.py)

# (1) ruff — the package, the engine pieces it added, the env scripts, and
# exactly the test files this gate runs.
step "ruff" "$RUFF" check mstar/model/glm5_next \
  mstar/engine/resources/attn/mla.py mstar/engine/resources/attn/mla_wrapper.py \
  mstar/engine/resources/slot_state env/smoke_glm53_loader.py env/gpu0_keeper.py \
  ${MOD[@]+"${MOD[@]}"} ${ENGINE[@]+"${ENGINE[@]}"} ${GPU[@]+"${GPU[@]}"}

# (2) modular tests: isolated first, then one shared process. An empty glob
# is a FAIL — a suite that vanished must never read as green.
if [ ${#MOD[@]} -eq 0 ]; then
  record "modular" "FAIL(no test/modular/test_glm5next_*.py)"
else
  for f in "${MOD[@]}" ${ENGINE[@]+"${ENGINE[@]}"}; do
    step "modular:$(basename "$f") [solo]" cpu_env "${PYTEST[@]}" "$f"
  done
  step "modular:all-together" cpu_env "${PYTEST[@]}" "${MOD[@]}" ${ENGINE[@]+"${ENGINE[@]}"}
fi

# (3) registry / CPU-import contract — a pytest file (see header). It is
# already in MOD via the glob, so this re-run only pins the name in the
# summary; a vanished file is a FAIL, not a silent pass.
REG=test/modular/test_glm5next_registry_contract.py
if [ -f "$REG" ]; then
  step "registry-cpu-import" cpu_env "${PYTEST[@]}" "$REG"
else
  record "registry-cpu-import" "FAIL(missing $REG)"
fi

# (3b) real-checkpoint loader smoke — headers only (no tensor data, no GPU),
# but needs the ~306 GB GLM-5.3-Flash snapshot present, so it is box-gated:
# set GLM53_CKPT, or it auto-finds the HF cache snapshot; absent = SKIP. This
# is the only gate that sees real shapes/dtypes (the index cross-check cannot),
# incl. the fp32-preservation coverage that keeps mHC base/scale off bf16.
CKPT=${GLM53_CKPT:-}
if [ -z "$CKPT" ]; then
  for d in "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--zai-org--GLM-5.3-Flash/snapshots/*/ \
           /data/*/hf/hub/models--zai-org--GLM-5.3-Flash/snapshots/*/; do
    [ -f "$d/model.safetensors.index.json" ] && { CKPT=$d; break; }
  done
fi
if [ -n "$CKPT" ] && [ -f "$CKPT/model.safetensors.index.json" ]; then
  step "loader-smoke [real ckpt]" cpu_env "$PY" env/smoke_glm53_loader.py "$CKPT"
else
  record "loader-smoke [real ckpt]" "SKIP(no GLM-5.3-Flash snapshot; set GLM53_CKPT)"
fi

# (4) GPU integration — opt-in, one pytest process per file. The files carry
# their own `skipif(not torch.cuda.is_available())`, so a forced run on a
# CPU box reports pytest's "skipped" (exit 0 = PASS here, with 0 tests run);
# the default gate on nvidia-smi keeps a laptop from even spawning them.
if [ "${GLM53_GPU:-0}" = "1" ] || { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }; then
  if [ ${#GPU[@]} -eq 0 ]; then
    record "gpu:integration" "SKIP(no test/integration/test_glm5next_*.py)"
  else
    for f in "${GPU[@]}"; do
      step "gpu:$(basename "$f") [solo]" "${PYTEST[@]}" "$f"
    done
  fi
else
  record "gpu:integration" "SKIP(no nvidia-smi; GLM53_GPU=1 forces; ${#GPU[@]} file(s) waiting)"
fi

# ── summary ──────────────────────────────────────────────────────────────
echo
echo "──── ci_glm53 summary"
fail=0
for i in "${!NAMES[@]}"; do
  printf '  %-28s %s\n' "${CODES[$i]}" "${NAMES[$i]}"
  case ${CODES[$i]} in FAIL*) fail=1 ;; esac
done
if [ "$fail" -eq 1 ]; then echo "CI FAIL (${SECONDS}s)"; exit 1; fi
echo "CI PASS (${SECONDS}s)"
