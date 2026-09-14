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


@rope_apply_qk_inplace.register_fake
def _rope_apply_qk_inplace_fake(
    q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor,
    rotary_dim: int | None, interleave: bool,
    rope_scale: float, rope_theta: float,
    low_freq_factor: float | None = None,
    high_freq_factor: float | None = None,
    old_context_len: float | None = None,
) -> None:
    return None
