#!/usr/bin/env bash
# Build the transformers-main "oracle" venv on the Laude node — the reference
# glm5_next implementation for a port-vs-reference parity gate.
# (v1-engine port of the lane script: unchanged — the oracle is engine-agnostic.)
#
# Why a separate venv: the model serve venv pins transformers 4.57.3 (no
# glm5_next); the reference lives on transformers main. CPU torch is enough —
# parity runs a tiny random-weight config, not the 320 B checkpoint.
#
# Box gotchas this encodes (2026-08-31): `python3.12 -m venv` is broken here
# (Debian's python3.12-venv / ensurepip missing, no sudo) — use the bootstrap
# `uv` at $D/.local/bin/uv, which manages its own python + pip.
#
#   bash env/build_oracle_laude.sh   # on the Laude node
set -euo pipefail
D=${D:-/data/kvasilev-cd4cf3}
UV=$D/.local/bin/uv
VENV=$D/envs/glm5next-oracle
export UV_CACHE_DIR=$D/.cache/uv

rm -rf "$VENV"
"$UV" venv "$VENV" --python 3.12
P=$VENV/bin/python
"$UV" pip install --python "$P" torch --index-url https://download.pytorch.org/whl/cpu
"$UV" pip install --python "$P" \
    "transformers @ git+https://github.com/huggingface/transformers" \
    accelerate safetensors einops

echo "=== viability: transformers-main instantiates glm5_next ==="
"$P" - <<'PY'
import transformers
print("transformers", transformers.__version__)
from transformers.models.glm5_next import configuration_glm5_next as C
C.Glm5NextTextConfig(num_hidden_layers=4, hidden_size=64, num_attention_heads=4,
    num_key_value_heads=4, intermediate_size=128, moe_intermediate_size=64,
    n_routed_experts=4, vocab_size=256)
from transformers.models.glm5_next import modeling_glm5_next as M
print("classes:", [n for n in dir(M) if n.startswith("Glm5Next")
                   and ("Model" in n or "ForCausal" in n)])
PY
echo "oracle venv ready: $VENV"
