"""Ling-2.0 MoE block (TP-aware ``MultiRouter`` flavour).

Same 3-router text/image/audio gate selection as step 3b, now with
per-rank expert sharding when ``comm_group.world_size > 1``:

  * Fused expert tensors hold ``(E, 2*shard_inter, hidden)`` and
    ``(E, hidden, shard_inter)`` per rank, where
    ``shard_inter = moe_intermediate_size // tp_size``.
  * Mminf's ``_gate_up_weight_loader`` / ``_down_proj_weight_loader``
    handle per-rank slicing during checkpoint load — these get
    attached to the params via the ``_attach_weight_loaders`` dance
    that survives ``.to_empty`` / ``.to(...)``.
  * Shared expert is a ``ParallelGatedMLP`` so its ``down_proj``
    all-reduces internally.
  * Forward TP path mirrors :class:`ParallelSparseMoeBlock._dispatch_tp`:
    `fused_experts(..., reduce_results=False)` → ``all_reduce`` →
    ``moe_sum_reduce_triton``.

Routers (``LingMoeRouter``) stay replicated across ranks — gates must
make identical decisions so every rank dispatches tokens to the same
experts.

Reference: vllm-omni's ``BailingMoeV2SparseMoeBlock`` (lines 304-433)
+ mstar's :class:`ParallelSparseMoeBlock`
(`mstar/model/components/moe.py:318-414`).
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from mstar.distributed.communication import TPCommGroup
from mstar.distributed.utils import divide
from mstar.model.components.distributed.mlp import ParallelGatedMLP
from mstar.model.components.mlp import GatedMLP
from mstar.model.components.moe import (
    _dispatch,
    _down_proj_weight_loader,
    _gate_up_weight_loader,
    dispatch_experts_fused,
)
from mstar.model.ming_omni_flash.components.router import LingMoeRouter


def _normalize_modality_mask(
    mask: torch.Tensor | None, num_tokens: int, name: str,
) -> torch.Tensor | None:
    """Reshape a modality mask to ``(num_tokens, 1)`` bool, or pass through None."""
    if mask is None:
        return None
    if mask.dim() == 1:
        if mask.shape[0] != num_tokens:
            raise ValueError(
                f"{name} length {mask.shape[0]} != num_tokens={num_tokens}"
            )
        return mask.reshape(num_tokens, 1).bool()
    if mask.dim() == 2:
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

    Constructor takes the FULL ``moe_intermediate_size``; the per-rank
    ``shard_inter`` is computed from ``comm_group.world_size``.

    Args:
        hidden_size: model hidden dim.
        num_experts: total routed experts.
        num_experts_per_tok: top-k experts per token.
        moe_intermediate_size: per-expert intermediate dim (FULL —
            sharding handled internally).
        num_shared_experts: number of shared experts (1 on the released
            ckpt). The shared expert is a ``ParallelGatedMLP`` of width
            ``moe_intermediate_size * num_shared_experts``.
        n_group, topk_group, routed_scaling_factor: passed to the
            :class:`LingMoeRouter`s.
        comm_group: TP comm group; defaults to single-rank trivial.
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
        comm_group: TPCommGroup | None = None,
    ) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = TPCommGroup.trivial()
        self.comm_group = comm_group
        tp_size = comm_group.world_size
        tp_rank = comm_group.rank

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
        # Routers — replicated. All ranks must agree on which experts a
        # given token routes to, so gate weights are loaded identically
        # per rank (default weight_loader, no shard_id).
        self.gate = LingMoeRouter(**router_kwargs)
        self.image_gate = LingMoeRouter(**router_kwargs)
        self.audio_gate = LingMoeRouter(**router_kwargs)

        # Fused expert tensors with per-rank intermediate shard.
        shard_inter = divide(moe_intermediate_size, tp_size)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * shard_inter, hidden_size)
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, shard_inter)
        )

        # Shared expert: ParallelGatedMLP. Its down_proj all-reduces, so
        # the shared output already lives on the full hidden state at
        # every rank.
        if num_shared_experts <= 0:
            raise ValueError(
                "LingMoeBlock requires num_shared_experts >= 1; released "
                "Ming-flash-omni-2.0 has 1."
            )
        self.shared_expert = ParallelGatedMLP(
            comm_group=comm_group,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size * num_shared_experts,
            bias=False,
        )

        self._attach_weight_loaders(tp_rank, tp_size, moe_intermediate_size)

    # ------------------------------------------------------------------
    # Weight loader plumbing — mirrors ParallelSparseMoeBlock
    # ------------------------------------------------------------------

    def _attach_weight_loaders(
        self, tp_rank: int, tp_size: int, full_inter: int,
    ) -> None:
        """Attach mstar's per-rank fused-expert weight loaders.

        The loaders accept shard ids ``"gate:N"``, ``"up:N"``, ``"down:N"``
        and slice along the intermediate dim per rank, then write into
        the right expert slot. ``load_hf_weights`` dispatches based on
        the ``StackedParamRule.shard_id`` we configure in the loader.
        """
        self.experts.gate_up_proj.weight_loader = partial(
            _gate_up_weight_loader, tp_rank, tp_size, full_inter,
        )
        self.experts.down_proj.weight_loader = partial(
            _down_proj_weight_loader, tp_rank, tp_size, full_inter,
        )

    def _apply(self, fn, recurse=True):
        """Re-attach loaders after any ``to_empty`` / ``.to(...)`` since
        those operations re-allocate Parameters and drop attached
        attributes on the old objects."""
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders(
            self.comm_group.rank,
            self.comm_group.world_size,
            self.moe_intermediate_size,
        )
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route + dispatch + add shared expert output.

        TP=1 path uses the direct ``_dispatch`` helper (mstar's
        triton-fused or naive loop depending on availability). TP>1
        path uses the unreduced fused_experts call + manual all-reduce
        + sum-reduce — mirrors :class:`ParallelSparseMoeBlock._dispatch_tp`.
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

        if self.comm_group.world_size == 1:
            routed = _dispatch(
                flat,
                self.experts.gate_up_proj,
                self.experts.down_proj,
                self.num_experts,
                topk_idx,
                topk_weight,
            )
        else:
            routed = self._dispatch_tp(flat, topk_weight, topk_idx)

        shared = self.shared_expert(flat)
        # Upstream sums routed + shared without an additional gate
        # (BailingMoeV2SparseMoeBlock.forward:429). The
        # routed_scaling_factor is baked into topk_weight via the router.
        return (routed + shared).view(input_shape)

    def _dispatch_tp(
        self,
        flat: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """TP>1 expert dispatch.

        Identical to :func:`ParallelSparseMoeBlock._dispatch_tp` — runs
        fused_experts WITHOUT the final per-token reduce, all-reduces
        the per-rank partial results across TP ranks, then sum-reduces
        across top-k. Result is the full-precision routed output at
        every rank.

        Falls back to the naive per-expert loop in
        :func:`dispatch_experts_fused` when ``sgl_kernel`` isn't loadable
        (e.g. ABI-mismatched against the installed torch). The naive path
        already returns ``(tokens, hidden)`` summed across top-k, so we
        all-reduce that directly — math is equivalent because sum-over-TP
        and sum-over-top-k commute.
        """
        from mstar.utils.fused_moe.align import has_sgl_kernel

        if has_sgl_kernel():
            from mstar.utils.fused_moe import fused_experts, moe_sum_reduce_triton

            cache3 = fused_experts(
                flat,
                self.experts.gate_up_proj,
                self.experts.down_proj,
                routing_weights,
                selected_experts,
                reduce_results=False,
            )
            self.comm_group.all_reduce(cache3)
            output = torch.empty_like(flat)
            moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
            return output

        partial = dispatch_experts_fused(
            flat,
            self.experts.gate_up_proj,
            self.experts.down_proj,
            self.experts.gate_up_proj.shape[0],
            selected_experts,
            routing_weights,
        )
        self.comm_group.all_reduce(partial)
        return partial


__all__ = ["LingMoeBlock", "GatedMLP"]  # GatedMLP re-export for back-compat
