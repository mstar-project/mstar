"""RoPE on AMD GPUs: FlashInfer's ROCm fork.

With the fp32 cos/sin cache ``PositionManager`` builds from its config (frequency
scale and Llama 3.1 blend applied), the fork's cos/sin-cache kernel: the fastest
measured on gfx942 at decode sizes, where the fork's on-the-fly Llama 3.1 kernel
is several times slower. Without one (a layer passing arguments the cache was not
built from), the on-the-fly kernels NVIDIA GPUs run. Imported only on ROCm.
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
    """Rotate q and k in place; the cache, when given, fixes the angles."""
    if cos_sin_cache is not None:
        _rope.apply_rope_with_cos_sin_cache_inplace(
            pos_ids, q, k, q.shape[-1], cos_sin_cache, is_neox=not interleave,
        )
        return
    rope_kwargs = dict(
        rotary_dim=rotary_dim, interleave=interleave,
        rope_scale=rope_scale, rope_theta=rope_theta,
    )
    llama31 = None not in (low_freq_factor, high_freq_factor, old_context_len)
    # the Llama 3.1 blend is the identity at scale 1, and its kernel is slower
    if llama31 and rope_scale != 1.0:
        _rope.apply_llama31_rope_pos_ids_inplace(
            q, k, pos_ids, **rope_kwargs,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            old_context_len=old_context_len,
        )
    else:
        _rope.apply_rope_pos_ids_inplace(q, k, pos_ids, **rope_kwargs)
