"""The rope kernel, behind a custom op.

FlashInfer reaches its kernel through a TVM-FFI call dynamo can't trace and
can't run on fake tensors. Called directly it breaks the graph once per layer,
and each break makes the layer body a frame dynamo recompiles per
``layer_idx``; behind an op with a registered fake the layer loop stays one
graph.
"""

import torch


@torch.library.custom_op("mstar::rope_apply_qk_inplace", mutates_args={"q", "k"})
def rope_apply_qk_inplace(
    q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    rotary_dim: int | None, interleave: bool,
    rope_scale: float, rope_theta: float,
    low_freq_factor: float | None = None,
    high_freq_factor: float | None = None,
    old_context_len: float | None = None,
) -> None:
    """Rotate q and k in place at ``pos_ids``."""
    import flashinfer

    rope_kwargs = dict(
        rotary_dim=rotary_dim, interleave=interleave,
        rope_scale=rope_scale, rope_theta=rope_theta,
    )
    llama31 = (
        low_freq_factor is not None
        and high_freq_factor is not None
        and old_context_len is not None
    )
    if not llama31:
        flashinfer.rope.apply_rope_pos_ids_inplace(q, k, pos_ids, **rope_kwargs)
    else:
        flashinfer.rope.apply_llama31_rope_pos_ids_inplace(
            q, k, pos_ids, **rope_kwargs,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            old_context_len=old_context_len,
        )


@torch.library.register_kernel(rope_apply_qk_inplace, "xpu")
def _rope_apply_qk_inplace_xpu(
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
    """Rotate q and k in place with the vllm XPU fused kernel."""
    if any(
        value is not None
        for value in (low_freq_factor, high_freq_factor, old_context_len)
    ):
        raise NotImplementedError(
            "Llama 3.1 RoPE scaling is not implemented for XPU"
        )
    if cos_sin_cache is None:
        raise RuntimeError("XPU RoPE requires a precomputed cos/sin cache")

    import vllm_xpu_kernels._C  # noqa: F401

    torch.ops._C.rotary_embedding(
        pos_ids,
        q,
        k,
        q.shape[-1],
        cos_sin_cache,
        not interleave,
    )


@rope_apply_qk_inplace.register_fake
def _rope_apply_qk_inplace_fake(
    q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    rotary_dim: int | None, interleave: bool,
    rope_scale: float, rope_theta: float,
    low_freq_factor: float | None = None,
    high_freq_factor: float | None = None,
    old_context_len: float | None = None,
) -> None:
    return None
