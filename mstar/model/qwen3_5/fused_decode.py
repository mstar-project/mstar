"""Qwen3.5's decode layer stack as one persistent kernel per 4-layer block, for captured decode
buckets of 1-2 rows (``LLMSubmodule.get_fused_decode_configs``).

Each block (3 gated-delta-net layers then a full-attention layer, each with its MLP) is one launch
of ``mstar.kernels.megakernel.block``: it reads the recurrent pool's state and conv blocks through
the step's slots, writes the new K/V into the pages at the row's last position off the attention
plan's block table and lengths, and leaves the next block's normed input and residual. The tables
are the planner's (host plan or resident gather), so nothing here plans or indexes a resource.

The kernel walks a block's layers by stride, so the block's GDN projections and MLPs are stacked
by layer and the modules' parameters re-pointed into the stacks (no second copy); only the down
projections are stored again, transposed, so a CTA's rows of it are contiguous.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.kernels.megakernel import block_refusal
from mstar.kernels.megakernel.block import (
    BlockGeometry,
    BlockScratch,
    BlockWeights,
    LaunchConfig,
    max_chunks,
    run_block,
)
from mstar.model.qwen3_5.config import FULL_ATTENTION, LINEAR_ATTENTION

BLOCK_LAYERS = (LINEAR_ATTENTION, LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION)


def stack_refusal(lm, device: torch.device, page_size: int, state_dtype: torch.dtype, adapters: bool) -> str | None:
    """Why the block kernel can't run this stack, or None."""
    config = lm.config
    types = tuple(config.layer_types)
    if len(types) % 4 or any(types[i:i + 4] != BLOCK_LAYERS for i in range(0, len(types), 4)):
        return "the layers don't group into 3 gated-delta-net layers then 1 full-attention layer"
    gdn, attn = lm.layers[0].self_attn, lm.layers[3].self_attn
    return block_refusal(
        device,
        hidden_size=config.hidden_size,
        gdn_head_dims=(gdn.head_k_dim, gdn.head_v_dim),
        attn_head_dim=attn.head_dim,
        kv_heads=attn.num_kv_heads,
        page_size=page_size,
        state_dtype=state_dtype,
        conv_taps=gdn.conv1d.weight.shape[-1],
        gated_attention=attn.output_gate,
        adapters=adapters,
    )


def _stack_into(params: list[nn.Parameter]) -> torch.Tensor:
    """One ``[n, ...]`` tensor of ``params``, which become views of it."""
    stacked = torch.stack([p.data for p in params]).contiguous()
    for i, p in enumerate(params):
        p.data = stacked[i]
    return stacked


class FusedDecodeStack:
    """The stacked weights and launch settings of every block of one language model."""

    def __init__(self, lm, device: torch.device):
        config = lm.config
        layers = lm.layers
        self.geom = BlockGeometry.of(config.hidden_size, config.intermediate_size, layers[0].self_attn,
                                     layers[3].self_attn)
        self.eps = config.rms_norm_eps
        self.launch = LaunchConfig(grid=torch.cuda.get_device_properties(device).multi_processor_count)
        self.blocks: list[BlockWeights] = []
        copy = lambda ts: torch.stack([t.detach() for t in ts]).contiguous()  # noqa: E731
        for first in range(0, len(layers), 4):
            block = layers[first:first + 4]
            gdn = [layer.self_attn for layer in block[:3]]
            attn = block[3].self_attn
            mlps = [layer.mlp for layer in block]
            nxt = [layers[i + 1].input_layernorm if i + 1 < len(layers) else lm.norm
                   for i in range(first, first + 4)]
            self.blocks.append(BlockWeights(
                win=_stack_into([m.in_proj_fused.weight for m in gdn]),
                convw=copy([m.conv1d.weight.squeeze(1) for m in gdn]),
                alog=copy([m.A_log for m in gdn]),
                dtb=copy([m.dt_bias for m in gdn]),
                gnw=copy([m.norm.weight for m in gdn]),
                gwo=_stack_into([m.out_proj.weight for m in gdn]),
                wqkv=attn.qkv_proj.weight.detach(), qnw=attn.q_norm.weight.detach(),
                knw=attn.k_norm.weight.detach(), awo=attn.o_proj.weight.detach(),
                wgu=_stack_into([m.gate_up_proj.weight for m in mlps]),
                wdt=copy([m.down_proj.weight.t() for m in mlps]),
                wpost=copy([layer.post_attention_layernorm.weight for layer in block]),
                wnext=copy([n.weight for n in nxt]),
            ))

    def run(self, h: torch.Tensor, residual: torch.Tensor, conv: torch.Tensor, state: torch.Tensor,
            slots: torch.Tensor, kv_layers: list[torch.Tensor], block_table: torch.Tensor,
            lengths: torch.Tensor, max_seq_len: int, cos: torch.Tensor, sin: torch.Tensor,
            scratch) -> torch.Tensor:
        """The stack over the step's rows from layer 0's normed input ``h`` and ``residual``; the
        final norm's output. ``scratch`` is the capture's (``FusedScratch``)."""
        rows = h.shape[0]
        chunks = max_chunks(max_seq_len, self.launch.ch)
        work = BlockScratch.allocate(self.geom, rows, chunks, scratch.zeros)
        hs = (h, torch.empty_like(h))
        rs = (residual, torch.empty_like(residual))
        for k, weights in enumerate(self.blocks):
            run_block(self.geom, weights, work, self.launch, hs[k % 2], rs[k % 2], rs[(k + 1) % 2], hs[(k + 1) % 2],
                      conv, state, 3 * k, slots, cos, sin, kv_layers[k], block_table, lengths, self.eps, chunks)
        return hs[len(self.blocks) % 2]
