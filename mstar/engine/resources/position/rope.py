"""The rope kernel, behind a custom op.

FlashInfer reaches its kernel through a TVM-FFI call dynamo can't trace and
can't run on fake tensors. Called directly it breaks the graph once per layer,
and each break makes the layer body a frame dynamo recompiles per
``layer_idx``; behind an op with a registered fake the layer loop stays one
graph.

FlashInfer takes fp16 and bf16 only; an fp32 q/k runs the portable path, which
follows its frequency and rotation math.
"""

import math

import torch

_FUSED_DTYPES = (torch.float16, torch.bfloat16)


def _rope_portable_inplace(
    q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor,
    rotary_dim: int | None, interleave: bool, rope_scale: float, rope_theta: float,
    low_freq_factor: float | None = None,
    high_freq_factor: float | None = None,
    old_context_len: float | None = None,
) -> None:
    """FlashInfer's ``apply_(llama31_)rope_pos_ids_inplace`` in torch, in the
    input's dtype; dims past ``rotary_dim`` are left as they are."""
    rotary_dim = rotary_dim or q.shape[-1]
    half = rotary_dim // 2
    idx = torch.arange(rotary_dim, device=q.device)
    pair = idx // 2 if interleave else idx % half
    freq = (1.0 / rope_theta) ** (2 * pair.to(torch.float32) / rotary_dim)
    if low_freq_factor is not None and high_freq_factor is not None and old_context_len is not None:
        smooth_a = old_context_len / (2 * math.pi * high_freq_factor - 2 * math.pi * low_freq_factor)
        smooth_b = -1.0 / (high_freq_factor / low_freq_factor - 1.0)
    else:
        smooth_a = smooth_b = 0.0
    smooth = (freq * smooth_a + smooth_b).clamp(0.0, 1.0)
    freq = (1 - smooth) * (freq / rope_scale) + smooth * freq
    angle = pos_ids.to(torch.float32)[:, None] * freq[None, :]
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
    if interleave:
        partner = idx ^ 1
        sign = torch.where(idx % 2 == 0, -1.0, 1.0)
    else:
        partner = torch.where(idx < half, idx + half, idx - half)
        sign = torch.where(idx < half, -1.0, 1.0)
    for x in (q, k):
        rot = x[..., :rotary_dim].to(torch.float32)
        out = rot * cos + sign * rot[..., partner] * sin
        x[..., :rotary_dim] = out.to(x.dtype)


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
        f"RoPE is only implemented for CUDA and XPU, not {q.device.type}"
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
    if q.dtype not in _FUSED_DTYPES:
        _rope_portable_inplace(
            q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta,
            low_freq_factor, high_freq_factor, old_context_len,
        )
        return
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
