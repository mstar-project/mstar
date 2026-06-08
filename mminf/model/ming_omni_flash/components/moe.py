"""Ling-2.0 MoE block (``MultiRouter`` flavour).

Ling-2.0 doesn't use a single sparse-MoE block — it ships **three**
:class:`LingMoeRouter` instances per layer (text ``gate``, ``image_gate``,
``audio_gate``). Per-token routing decisions are then mixed: for tokens
flagged by ``image_mask`` we use the image gate's choices; for
``audio_mask`` we use the audio gate; otherwise the text gate. Same
fused expert pool dispatches all of them.

This is the per-layer FFN for layers ``layer_idx >= first_k_dense_replace``
(layer 0 uses a plain :class:`mminf.model.components.mlp.GatedMLP` instead;
that branch lives in :class:`LingDecoderLayer`).

Reference: vllm-omni's ``BailingMoeV2SparseMoeBlock`` at
``/tmp/vllm-omni/vllm_omni/model_executor/models/ming_flash_omni/modeling_bailing_moe_v2.py:304-433``.

Step-3b scope: TP=1, no KV cache, no weight loader. The fused expert
parameters use the same packed layout
(``experts.gate_up_proj`` / ``experts.down_proj``) as mminf's
:class:`SparseMoeBlock`, so the eventual weight loader (step 3c) can
reuse the existing fused-checkpoint primitives.
"""

from __future__ import annotations

import torch
from torch import nn

from mminf.model.components.mlp import GatedMLP
from mminf.model.components.moe import _dispatch
from mminf.model.ming_omni_flash.components.router import LingMoeRouter


def _normalize_modality_mask(
    mask: torch.Tensor | None, num_tokens: int, name: str,
) -> torch.Tensor | None:
    """Reshape a modality mask to ``(num_tokens, 1)`` bool, or pass through None.

    Accepts ``(num_tokens,)``, ``(num_tokens, 1)``, or ``(B, T)`` /
    ``(B, T, 1)`` shapes — the last two get flattened. Anything else
    raises.
    """
    if mask is None:
        return None
    if mask.dim() == 1:
        if mask.shape[0] != num_tokens:
            raise ValueError(
                f"{name} length {mask.shape[0]} != num_tokens={num_tokens}"
            )
        return mask.reshape(num_tokens, 1).bool()
    if mask.dim() == 2:
        # Either (B, T) or (num_tokens, 1). Disambiguate by total count.
        if mask.numel() != num_tokens:
            raise ValueError(
                f"{name} shape {tuple(mask.shape)} has {mask.numel()} elements; "
                f"expected num_tokens={num_tokens}"
            )
        return mask.reshape(num_tokens, 1).bool()
    if mask.dim() == 3:
        if mask.shape[-1] != 1 or mask.numel() != num_tokens:
            raise ValueError(
                f"{name} shape {tuple(mask.shape)} not compatible with "
                f"num_tokens={num_tokens}"
            )
        return mask.reshape(num_tokens, 1).bool()
    raise ValueError(
        f"{name} must be 1D, 2D, or 3D; got shape {tuple(mask.shape)}"
    )


class LingMoeBlock(nn.Module):
    """Ling-2.0 MoE FFN with text/image/audio gate selection per token.

    Args:
        hidden_size: model hidden dim.
        num_experts: total routed experts.
        num_experts_per_tok: top-k experts per token.
        moe_intermediate_size: per-expert intermediate dim.
        num_shared_experts: number of shared experts. Released ckpt uses
            1 — that becomes a single GatedMLP of size
            ``moe_intermediate_size * num_shared_experts``.
        n_group: expert groups (must divide num_experts).
        topk_group: top groups used per token.
        routed_scaling_factor: post-renormalisation scaling on routed
            weights (baked into the gate's output, not applied again here).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        num_shared_experts: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        router_kwargs = dict(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.gate = LingMoeRouter(**router_kwargs)
        self.image_gate = LingMoeRouter(**router_kwargs)
        self.audio_gate = LingMoeRouter(**router_kwargs)

        # Fused expert weights — match mminf's SparseMoeBlock layout so
        # the step-3c weight loader can map per-expert
        # gate_proj / up_proj / down_proj keys into them.
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, moe_intermediate_size)
        )

        # Shared expert: a GatedMLP with intermediate size scaled by
        # num_shared_experts (so num_shared_experts=1 makes it the same
        # width as one routed expert; num_shared_experts=N would make
        # it N× wider — but the released ckpt only ships num_shared=1).
        if num_shared_experts <= 0:
            raise ValueError(
                "LingMoeBlock requires num_shared_experts >= 1; released "
                "Ming-flash-omni-2.0 has 1. For num_shared_experts=0 use "
                "mminf.model.components.moe.SparseMoeBlock directly."
            )
        self.shared_expert = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size * num_shared_experts,
            activation="silu",
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route + dispatch + add shared expert output.

        Args:
            hidden_states: ``(..., hidden_size)``. Flattened to ``(N, H)``
                for routing/dispatch; reshaped back at the end.
            image_mask: bool, True for tokens that should route via
                ``image_gate``. Any shape that flattens to ``(N, 1)``.
            audio_mask: same shape rules, routes via ``audio_gate``.

        Returns:
            Tensor of the same shape as ``hidden_states``.
        """
        input_shape = hidden_states.shape
        flat = hidden_states.view(-1, hidden_states.shape[-1]).contiguous()
        num_tokens = flat.shape[0]

        # Text-gate baseline routing (always computed).
        _, topk_weight, topk_idx = self.gate(flat)

        image_mask = _normalize_modality_mask(image_mask, num_tokens, "image_mask")
        audio_mask = _normalize_modality_mask(audio_mask, num_tokens, "audio_mask")

        if image_mask is not None:
            _, img_w, img_idx = self.image_gate(flat)
            topk_idx = torch.where(image_mask, img_idx, topk_idx)
            topk_weight = torch.where(image_mask, img_w, topk_weight)
        if audio_mask is not None:
            _, aud_w, aud_idx = self.audio_gate(flat)
            topk_idx = torch.where(audio_mask, aud_idx, topk_idx)
            topk_weight = torch.where(audio_mask, aud_w, topk_weight)

        routed = _dispatch(
            flat,
            self.experts.gate_up_proj,
            self.experts.down_proj,
            self.num_experts,
            topk_idx,
            topk_weight,
        )
        shared = self.shared_expert(flat)
        # Upstream sums routed + shared without an additional gate
        # (BailingMoeV2SparseMoeBlock.forward:429). The scaling lives
        # inside topk_weight via the router's routed_scaling_factor.
        return (routed + shared).view(input_shape)
