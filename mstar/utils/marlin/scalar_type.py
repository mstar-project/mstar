"""vLLM ``ScalarType`` ids mirrored in Python for the Marlin ops."""
from __future__ import annotations

# vllm::kU4B8.id(): (mantissa=4 << 8) | (bias=8 << 17) | (nan_repr=1 << 50).
UINT4B8_ID = (4 << 8) | (8 << 17) | (1 << 50)  # == 1125899907892224

assert UINT4B8_ID == 1125899907892224, "uint4b8 id drifted from the vendored header"
