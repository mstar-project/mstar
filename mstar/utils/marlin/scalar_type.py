"""vLLM ``ScalarType`` ids mirrored in Python for the Marlin ops.

The vendored ``core/scalar_type.hpp`` packs a scalar type into a single int64 id
(field order exponent|mantissa|signed|bias|finite|nan_repr). The Marlin GEMM op
takes ``b_type_id`` and reconstructs the type via ``ScalarType::from_id``, so the
Python side must pass the exact same id. Only symmetric INT4 (``uint4b8``) is
built. The id is computed directly from the header's bit layout and re-validated
at runtime by the C++ op's ``TORCH_CHECK(b_type == kU4B8)``.

  uint4b8 = ScalarType::uint(size_bits=4, bias=8)
          = (mantissa=4 << 8) | (bias=8 << 17) | (nan_repr=NAN_IEEE_754=1 << 50)
"""
from __future__ import annotations

# vllm::kU4B8.id() — symmetric GPTQ-style INT4 (offset-binary, subtract 8).
UINT4B8_ID = (4 << 8) | (8 << 17) | (1 << 50)  # == 1125899907892224

assert UINT4B8_ID == 1125899907892224, "uint4b8 id drifted from the vendored header"
