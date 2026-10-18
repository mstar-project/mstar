"""RoPE on AMD GPUs: FlashInfer's ROCm fork, reading a precomputed cos/sin cache.

The fork's cos/sin-cache kernel was the fastest measured on gfx942 at decode
sizes, and it is the only form that covers Llama 3.1 scaling quickly: the fork's
on-the-fly Llama 3.1 kernel is several times slower. ``PositionManager`` folds
the frequency scaling and the rotary width into the fp32 cache. Imported only on
ROCm.
"""

import torch

from mstar.utils.flashinfer_rocm import import_flashinfer_rocm

_rope = import_flashinfer_rocm("RoPE").rope


def rope_apply_qk_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    rotary_dim: int | None,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float | None = None,
    high_freq_factor: float | None = None,
    old_context_len: float | None = None,
) -> None:
    """Rotate the first ``cos_sin_cache.shape[-1]`` dims of each head in place."""
    if cos_sin_cache is None:
        raise RuntimeError(
            "RoPE on AMD GPUs reads the cos/sin cache PositionManager builds; "
            "none was passed"
        )
    _rope.apply_rope_with_cos_sin_cache_inplace(
        pos_ids, q, k, q.shape[-1], cos_sin_cache, is_neox=not interleave,
    )
