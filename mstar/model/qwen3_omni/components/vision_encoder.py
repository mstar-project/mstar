"""Native mstar reimplementation of Qwen3OmniMoeVisionEncoder (SigLIP2-style ViT + spatial merge + DeepStack).

Submodule names mirror HF so ``load_weights_from_hf_shards(..., prefix="thinker.visual")`` works unchanged.
Uses ``flash_attn_varlen_func`` directly instead of the HF FA2 wrapper (~3.3 s/image on H100); multiple
images batch via ``cu_seqlens``.
Output: ``forward(...) -> (vision_embeds[merged_tokens, out_hidden], deepstack[list])``.
"""
from __future__ import annotations

import logging

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)

from mstar.model.qwen3_omni.components import audio_encoder as AE
from mstar.model.qwen3_omni.components.audio_encoder import varlen_attention

logger = logging.getLogger(__name__)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q, orig_k = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q), k_embed.to(orig_k)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class VisionPatchEmbed(nn.Module):
    """Patch embedding; weight stored as Conv3d to match HF checkpoint layout.
    F.linear is used instead: cuDNN bf16 Conv3d is ~3.2 s/image on H100 for kernel==stride shapes;
    the equivalent matmul is ~40 us.
    """

    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_size
        ks = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=ks, stride=ks, bias=True)

    def forward(self, x):
        w = self.proj.weight.reshape(self.embed_dim, -1)
        x = x.reshape(-1, w.shape[1]).to(w.dtype)
        return torch.nn.functional.linear(x, w, self.proj.bias)


class VisionPatchMerger(nn.Module):
    """Mirrors HF Qwen3OmniMoeVisionPatchMerger (incl. use_postshuffle_norm)."""

    def __init__(self, config, use_postshuffle_norm=False):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.mlp = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.out_hidden_size),
        ])

    def forward(self, hidden):
        hidden = self.ln_q(hidden.view(-1, self.hidden_size) if self.use_postshuffle_norm else hidden).view(
            -1, self.hidden_size)
        for layer in self.mlp:
            hidden = layer(hidden)
        return hidden


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, cu_seqlens, max_seqlen, position_embeddings):
        s = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(s, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        o = varlen_attention(q, k, v, cu_seqlens, max_seqlen, self.scaling)
        return self.proj(o.reshape(s, -1))


class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, hidden_states, cu_seqlens, max_seqlen, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, max_seqlen, position_embeddings)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class NativeQwen3OmniVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.depth)])
        self.merger = VisionPatchMerger(config, use_postshuffle_norm=False)
        self.merger_list = nn.ModuleList([
            VisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in range(len(config.deepstack_visual_indexes))
        ])

    def _block_loop_tail(self, hidden_states, cu_seqlens, max_seqlen, position_embeddings):
        """Block loop + DeepStack mergers + final merger; constants for a fixed grid_thw, replayable as a CUDA graph."""
        deepstack_features = []
        for i, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens, max_seqlen, position_embeddings)
            if i in self.deepstack_visual_indexes:
                k = self.deepstack_visual_indexes.index(i)
                deepstack_features.append(self.merger_list[k](hidden_states))
        merged = self.merger(hidden_states)
        return merged, deepstack_features

    def get_piecewise_cuda_graph_config(self, device, autocast_dtype):
        """Build the ``PiecewisePackedConfig`` migrating the block-loop CUDA graph
        onto ``PiecewiseCudaGraphRunner``.

        The captured region is ``_block_loop_tail`` (block loop + DeepStack
        mergers + final merger) over a packed ``[total_tokens, hidden_size]``
        hidden state. ``position_embeddings`` (cos/sin) and ``cu_seqlens`` are
        STATIC inputs copied per replay — that is what lets one bucket serve many
        grid layouts — while the FlashInfer ragged wrapper is re-planned per
        replay with the real per-segment lengths.

        Returns None (eager path) when FlashInfer isn't the active varlen backend.
        Bucket lists are env-overridable
        (``MSTAR_ENCODER_CG_BS_VISION`` segment counts,
        ``MSTAR_ENCODER_CG_TOKENS_VISION`` token counts); each bucket owns a
        persistent 128 MiB workspace, so keep the product modest. Token buckets
        MUST be divisible by ``spatial_merge_size**2`` (the merger reshapes
        ``[total_tokens, H] -> [total_tokens/merge_sq, H*merge_sq]``); a bucket
        that isn't simply fails capture and falls back to eager."""
        import mstar.model.qwen3_omni.components.audio_encoder as AE
        from mstar.engine.cuda_graph_config import PiecewisePackedConfig
        if not (AE._FLASHINFER_AVAILABLE and AE._VARLEN_BACKEND == "flashinfer"):
            return None

        hidden_size = self.config.hidden_size
        num_heads = self.config.num_heads
        head_dim = hidden_size // num_heads
        scale = head_dim ** -0.5

        def make_static_inputs(shape):
            # x in autocast dtype (same-dtype replay copy). cos/sin are per-layout
            # and float32 (RoPE precision), so they MUST be static inputs copied
            # per replay, not constants baked into the graph. cu_seqlens likewise
            # static (the graph can no longer bake in one layout).
            return {
                "x": torch.zeros(shape.total_tokens, hidden_size,
                                 dtype=autocast_dtype, device=device),
                "cu_seqlens": torch.zeros(shape.bs + 1, dtype=torch.int32,
                                          device=device),
                "pos_cos": torch.zeros(shape.total_tokens, head_dim,
                                       dtype=torch.float32, device=device),
                "pos_sin": torch.zeros(shape.total_tokens, head_dim,
                                       dtype=torch.float32, device=device),
            }

        def make_attn_state(shape):
            return AE.make_fi_graph_state(device, shape.bs)

        def plan_attn_fn(state, shape, seq_lens):
            AE.plan_fi_graph_state(state, seq_lens, num_heads, head_dim, scale,
                                   autocast_dtype)

        def capture_fn(static_inputs, static_cm=None, attn_state=None, **kw):
            # max_seqlen unused (FI-external path ignores it; capture never takes
            # the flash-attn branch). Route attention through the runner wrapper.
            pos = (static_inputs["pos_cos"], static_inputs["pos_sin"])
            AE.set_fi_override(attn_state)
            try:
                merged, deepstack = self._block_loop_tail(
                    static_inputs["x"], static_inputs["cu_seqlens"], 0, pos)
            finally:
                AE.set_fi_override(None)
            out = {"merged": merged}
            for i, d in enumerate(deepstack):
                out[f"deepstack_{i}"] = d
            return out

        # Buckets are measured (MSTAR_ENCODER_CG_PROBE=1), covering i2t (one image
        # per request: segments=1, tokens {576,704,768,896,1024}) AND i2s (up to
        # four images: segments=4, ~4096 tokens) — sizing from i2t alone left
        # multi-image requests with no bucket. Low rungs match the single-image
        # layouts exactly; upper rungs cover multi-image.
        # ONE bs value above the observed max of 4: bs is crossed with
        # total_tokens, so extra values multiply graphs and 128 MiB workspaces,
        # while padding bs up is free (trailing segments are zero-length).
        # Token buckets MUST stay divisible by spatial_merge_size**2 (=4).
        return PiecewisePackedConfig(
            capture_fn=capture_fn,
            make_static_inputs=make_static_inputs,
            make_attn_state=make_attn_state,
            plan_attn_fn=plan_attn_fn,
            uses_kv_cache=False,
            total_tokens=AE._encoder_int_list(
                "MSTAR_ENCODER_CG_TOKENS_VISION",
                "576,704,768,896,1024,1280,1536,2048,3072,4096"),
            capture_batch_sizes=AE._encoder_int_list("MSTAR_ENCODER_CG_BS_VISION",
                                                     "8"),
        )

    @torch.no_grad()
    def forward(self, pixel_values, grid_thw):
        dtype = self.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(dtype)

        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw, num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=self.config.spatial_merge_size)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)

        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = (self.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        seq_len = hidden_states.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())

        # Piecewise path (default): replay the block loop on a bucketed graph,
        # re-planning ragged attention with the real per-segment lengths and
        # copying cos/sin + hidden state into the runner-owned static buffers. The
        # runner is threaded on by the submodule; absent it (or no fitting bucket)
        # we fall through to eager.
        runner = getattr(self, "_piecewise_runner", None)
        n_seg = cu_seqlens.shape[0] - 1
        total_tokens = hidden_states.shape[0]
        if AE._encoder_probe():
            # Harvest the layout, capture nothing, run eager.
            AE.note_encoder_layout("vision", n_seg, total_tokens, fitted=False)
            AE.note_encoder_path("vision.probe")
            return self._block_loop_tail(
                hidden_states, cu_seqlens, max_seqlen, position_embeddings)
        if runner is not None:
            fitted = runner.can_run(n_seg, total_tokens)
            AE.note_encoder_layout("vision", n_seg, total_tokens, fitted)
            if fitted:
                seg_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
                cos, sin = position_embeddings
                out = runner.run(
                    static_inputs={
                        "x": hidden_states,
                        "cu_seqlens": cu_seqlens.to(torch.int32),
                        "pos_cos": cos,
                        "pos_sin": sin,
                    },
                    seq_lens=seg_lens,
                    real_bs=n_seg,
                )
                # The mergers reduce the token axis by spatial_merge_size**2, so
                # the merged outputs have fewer rows than PiecewiseOutput's
                # token-count real_len; re-slice to the real merged length (the
                # leading rows — zero-padding lands in trailing merged tokens).
                merged_len = total_tokens // (self.spatial_merge_size ** 2)
                n_deepstack = len(self.deepstack_visual_indexes)
                merged = out.get_view("merged")[:merged_len].clone()
                deepstack = [
                    out.get_view(f"deepstack_{i}")[:merged_len].clone()
                    for i in range(n_deepstack)
                ]
                AE.note_encoder_path("vision.piecewise")
                return merged, deepstack

        AE.note_encoder_path("vision.eager")
        return self._block_loop_tail(hidden_states, cu_seqlens, max_seqlen, position_embeddings)
