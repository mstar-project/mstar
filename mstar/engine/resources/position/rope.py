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
    """Reject devices without a registered RoPE kernel."""
    raise NotImplementedError(
        f"RoPE is only implemented for CUDA, XPU and CPU, not {q.device.type}"
    )


@torch.library.register_kernel(rope_apply_qk_inplace, "cuda")
def _rope_apply_qk_inplace_cuda(
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
    """Rotate q and k in place with FlashInfer's CUDA kernel."""
    import flashinfer

    if cos_sin_cache is not None:
        # a precomputed (rounded) table: PositionConfig.rope_table_dtype
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
            pos_ids,
            q.view(q.shape[0], -1),
            k.view(k.shape[0], -1),
            q.shape[-1],
            cos_sin_cache,
            is_neox=not interleave,
        )
        return
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


@torch.library.register_kernel(rope_apply_qk_inplace, "cpu")
def _rope_apply_qk_inplace_cpu(
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
    """Plain RoPE in torch, for the CPU tests of models that rotate (a drafter's layers):
    FlashInfer's formula (``rope_scale`` divides the positions), fp32 angles."""
    llama31 = any(value is not None for value in (low_freq_factor, high_freq_factor, old_context_len))
    if llama31 and rope_scale != 1.0:
        # at a scale of 1 the Llama 3.1 rule leaves every frequency as it is
        raise NotImplementedError("Llama 3.1 RoPE scaling is not implemented on CPU")
    dim = rotary_dim or q.shape[-1]
    if cos_sin_cache is not None:
        cos, sin = cos_sin_cache[pos_ids.long()].float().chunk(2, dim=-1)
    else:
        inv = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        angles = torch.outer(pos_ids.float() / rope_scale, inv)
        cos, sin = angles.cos(), angles.sin()
    cos, sin = cos[:, None, :], sin[:, None, :]
    for x in (q, k):
        rot = x[..., :dim].float()
        if interleave:
            a, b = rot[..., 0::2], rot[..., 1::2]
            out = torch.stack((a * cos - b * sin, b * cos + a * sin), dim=-1).flatten(-2)
        else:
            a, b = rot[..., : dim // 2], rot[..., dim // 2:]
            out = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
        x[..., :dim].copy_(out)


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
