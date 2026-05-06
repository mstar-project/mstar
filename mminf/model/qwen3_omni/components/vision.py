"""mminf-native Qwen3-Omni vision encoder.

Hand-rolled replacement for ``transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoeVisionEncoder``.

Why we have our own:

* HF's reference is correct but full of defensive code that fights
  ``torch.compile`` (``attention_interface`` registry lookup,
  ``try/except RuntimeError`` for AMD/RDNA3, dynamic dispatch through
  the ``ALL_ATTENTION_FUNCTIONS`` table, etc.). Even with FA2 enabled
  it leaves perf on the table because ``torch.compile(fullgraph=False)``
  graph-breaks all over those branches and recompiles per shape.
* vllm-omni demonstrated that a lean from-scratch implementation runs
  cleanly in eager mode (no compile decorator at all) and is faster.
  This file is the mminf equivalent: a minimal eager-friendly path
  that calls ``flash_attn_varlen_func`` directly with no wrapper layers.

Compared to vllm-omni's version we drop everything tensor-parallel
(QKVParallelLinear, ColumnParallelLinear, RowParallelLinear, etc.)
because mminf runs the encoder single-GPU per partition, and we drop
the multi-backend dispatch (FlashInfer-CuDNN / Triton / SDPA fallbacks)
because we hard-require ``flash-attn`` for this branch and the FA2
path is what we want to optimize for.

Parameter names match HF (``patch_embed.proj.*``, ``blocks.{i}.attn.qkv.*``,
``merger.mlp.0.*``, etc.) so ``load_weights_from_hf_shards`` works
without a ``WeightConverter``.

Returns a ``(merged_hidden_states, deepstack_features_list)`` tuple to
match the existing ``VisionEncoderSubmodule`` consumer at
``mminf/model/qwen3_omni/submodules.py:191-208`` (which already accepts
either an HF ``BaseModelOutputWithDeepstackFeatures`` or a tuple).
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


class Qwen3OmniVisionRotaryEmbedding(nn.Module):
    """Pre-computes the RoPE inv_freq table for vision attention.

    Matches HF's ``Qwen3OmniMoeVisionRotaryEmbedding`` exactly.
    Constructed with ``dim = head_dim // 2`` because Qwen3-Omni's vision
    RoPE rotates only half of head_dim (the other half stays as
    identity-rotated; see ``apply_vision_rope`` below). ``inv_freq`` is
    therefore length ``head_dim // 4``.

    Output of ``forward(seqlen)`` is a freq table of shape
    ``(seqlen, head_dim // 4)``; the encoder later indexes this table
    with the per-token ``(h_idx, w_idx)`` pairs and concatenates to
    produce a per-token rotary embedding of shape ``(seq_len, head_dim // 2)``.
    """

    inv_freq: torch.Tensor  # tells the linter about the buffer

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard RoPE rotate-half: (x_lo, x_hi) -> (-x_hi, x_lo)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_vision_rope(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-token 2D RoPE (cos/sin precomputed by the encoder).

    ``q``, ``k``: shape ``(seq_len, num_heads, head_dim)``.
    ``cos``, ``sin``: shape ``(seq_len, head_dim)`` — the encoder builds
    these by concatenating the per-token freq table with itself along
    the last dim and taking cos/sin (matches HF's
    ``modeling_qwen3_omni_moe.py:1220-1221``).
    """
    cos_b = cos.unsqueeze(1)  # broadcast over num_heads
    sin_b = sin.unsqueeze(1)
    q_out = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_out = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Patch embed / MLP / Patch merger / Block / Attention
# ---------------------------------------------------------------------------


class Qwen3OmniVisionPatchEmbed(nn.Module):
    """Patch embedding. Stored as Conv3d for HF state_dict compatibility,
    but executed as a Linear since kernel == stride == input spatial dims
    makes the Conv3d mathematically identical to a single GEMM. cuDNN's
    Conv3d algo selection for these dims was 4-5 orders of magnitude
    slower than cuBLAS GEMM (1.1s vs 20us for 256 tokens on H100 bf16).
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        self.in_features = (
            self.in_channels * self.temporal_patch_size
            * self.patch_size * self.patch_size
        )

        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim,
            kernel_size=kernel_size, stride=kernel_size, bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        weight = self.proj.weight.view(self.embed_dim, self.in_features)
        return F.linear(hidden_states.to(dtype=target_dtype), weight, self.proj.bias)


class Qwen3OmniVisionMLP(nn.Module):
    """Two-layer MLP. HF uses GELU (``hidden_act`` in vision config).

    Parameter names match HF (``linear_fc1``, ``linear_fc2``) so weight
    loading is direct.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True,
        )
        self.linear_fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True,
        )
        # Resolve activation from config; default to gelu for Qwen3-Omni vision.
        act_name = getattr(config, "hidden_act", "gelu")
        self.act_fn = _resolve_activation(act_name)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


def _resolve_activation(name: str):
    name = (name or "gelu").lower()
    if name == "gelu":
        return F.gelu
    if name == "gelu_pytorch_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "silu":
        return F.silu
    if name == "relu":
        return F.relu
    raise ValueError(f"Unsupported vision MLP activation {name!r}")


class Qwen3OmniVisionPatchMerger(nn.Module):
    """LayerNorm + MLP that collapses ``spatial_merge_size**2`` adjacent
    patches into a single output token.

    Two flavors, distinguished by ``use_postshuffle_norm``:

    * ``use_postshuffle_norm=False`` (the final ``merger``): LayerNorm
      operates on the per-patch hidden_size. Then concat
      ``spatial_merge_size**2`` patches along feature dim and project
      with a 2-layer MLP into ``out_hidden_size``.
    * ``use_postshuffle_norm=True`` (each ``merger_list[i]`` for
      deepstack): LayerNorm operates on the post-shuffle width
      (``hidden_size * spatial_merge_size**2``). Same MLP after.

    Parameter names: ``ln_q.weight/bias``, ``mlp.0.weight/bias`` (Linear),
    ``mlp.1`` (GELU; no params), ``mlp.2.weight/bias`` (Linear). Matches
    HF.
    """

    def __init__(self, config, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        merge_unit = config.spatial_merge_size * config.spatial_merge_size
        self.hidden_size = config.hidden_size * merge_unit
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.ln_q = nn.LayerNorm(norm_dim, eps=1e-6)
        # nn.ModuleList with [Linear, GELU, Linear] so positional indices
        # 0, 1, 2 in the state dict line up with HF.
        self.mlp = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.out_hidden_size),
        ])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden = self.ln_q(hidden.view(-1, self.hidden_size))
        else:
            hidden = self.ln_q(hidden).view(-1, self.hidden_size)
        for layer in self.mlp:
            hidden = layer(hidden)
        return hidden


class Qwen3OmniVisionAttention(nn.Module):
    """Vision attention with FA2 varlen. No GQA (num_heads == num_kv_heads).

    Calls ``flash_attn_varlen_func`` directly (skipping vLLM's
    ``MMEncoderAttention`` wrapper and HF's ``attention_interface``
    registry lookup). RoPE applied to Q/K via ``apply_vision_rope`` —
    plain elementwise mul + rotate-half, no kernel call.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        assert self.dim % self.num_heads == 0, (
            f"hidden_size {self.dim} must be divisible by num_heads {self.num_heads}"
        )
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Fused QKV. Matches HF parameter name ``attn.qkv``.
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_len, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)  # (seq, 3 * dim)
        # (seq, 3, num_heads, head_dim) → unbind into per-projection tensors.
        qkv = qkv.view(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # each (seq, num_heads, head_dim)

        q, k = apply_vision_rope(q, k, cos, sin)

        # FA2 varlen attention. Inputs must be contiguous in (seq, head, head_dim).
        # We dtype-promote scale to float for FA2 (it accepts a Python float).
        from flash_attn import flash_attn_varlen_func

        attn_out = flash_attn_varlen_func(
            q.contiguous(), k.contiguous(), v.contiguous(),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False,
        )  # (seq, num_heads, head_dim)

        attn_out = attn_out.reshape(seq_len, self.dim)
        return self.proj(attn_out)


class Qwen3OmniVisionBlock(nn.Module):
    """Pre-norm transformer block: LN → attn → residual → LN → MLP → residual."""

    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3OmniVisionAttention(config)
        self.mlp = Qwen3OmniVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, cos=cos, sin=sin,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level encoder
# ---------------------------------------------------------------------------


class Qwen3OmniVisionEncoder(nn.Module):
    """mminf-native Qwen3-Omni vision encoder.

    Drop-in replacement for HF's ``Qwen3OmniMoeVisionEncoder`` — same
    parameter names so ``load_weights_from_hf_shards`` works directly.

    Forward signature matches the existing
    ``VisionEncoderSubmodule.forward`` contract: takes ``pixel_values``
    of shape ``(seq_len, in_channels * temporal_patch_size * patch_size**2)``
    and ``grid_thw`` of shape ``(num_images_or_videos, 3)``, returns a
    tuple ``(merged_hidden_states, deepstack_features_list)``.
    """

    def __init__(self, config) -> None:
        super().__init__()
        # ``config`` here is ``Qwen3OmniModelConfig.vision`` (i.e. the
        # vision sub-config). It carries hidden_size, num_heads,
        # spatial_merge_size, depth, deepstack_visual_indexes,
        # patch_size, temporal_patch_size, in_channels,
        # num_position_embeddings, out_hidden_size, intermediate_size,
        # hidden_act.
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_size = config.patch_size

        self.patch_embed = Qwen3OmniVisionPatchEmbed(config)

        # Absolute pos embed table (looked up via fast_pos_embed_interpolate).
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)

        # RoPE table — head_dim // 2 frequencies (vision RoPE rotates half).
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3OmniVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            Qwen3OmniVisionBlock(config) for _ in range(config.depth)
        ])
        self.merger = Qwen3OmniVisionPatchMerger(config, use_postshuffle_norm=False)

        # One PatchMerger per deepstack tap. HF stores these under
        # ``merger_list``; the encoder forward iterates
        # ``deepstack_visual_indexes`` and applies ``merger_list[i]`` to
        # the hidden state at layer ``deepstack_visual_indexes[i]``.
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.merger_list = nn.ModuleList([
            Qwen3OmniVisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in self.deepstack_visual_indexes
        ])

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    # ------------------------------------------------------------------
    # Position-id construction
    # ------------------------------------------------------------------

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Build the per-token freq vector for 2D vision RoPE.

        Output shape: ``(total_tokens, head_dim // 2)``. The encoder
        forward concatenates this with itself to make the final
        ``(total_tokens, head_dim)`` and takes cos/sin.

        Mirrors HF (``modeling_qwen3_omni_moe.py:1092-1130``) — patches
        in the same spatial-merge unit get adjacent token positions so
        that the merger's downstream concat groups them correctly.
        """
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, head_dim // 4)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset:offset + num_tokens] = coords
            offset += num_tokens

        # Index the freq table per (h, w), flatten to head_dim // 2.
        embeddings = freq_table[pos_ids]  # (total_tokens, 2, head_dim // 4)
        embeddings = embeddings.flatten(1)  # (total_tokens, head_dim // 2)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bilinear-interpolate the absolute pos_embed table to the
        actual (h, w) grid each image/video occupies.

        Mirrors HF's reference (``modeling_qwen3_omni_moe.py:1132-1193``)
        with the same spatial-merge permutation pattern at the end.
        """
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list: list[list[int]] = [[] for _ in range(4)]
        weight_list: list[list[float]] = [[] for _ in range(4)]

        for _t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=device,
        )
        # (4, total_h_w, hidden) — the four bilinear-corner contributions.
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # Split per (h*w), then permute so the spatial-merge unit groups
        # are contiguous (mirrors HF), and replicate across temporal frames.
        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws, strict=True)]
        )

        merge_size = self.spatial_merge_size
        permuted: list[torch.Tensor] = []
        for pe, t, h, w in zip(
            patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=True,
        ):
            pe = pe.repeat(t, 1)
            pe = (
                pe.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            permuted.append(pe)
        return torch.cat(permuted)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the full vision encoder.

        Args:
          hidden_states: ``(seq_len, in_channels * temporal_patch_size * patch_size**2)``
            packed pixel tensor produced by HF's video/image processor.
          grid_thw: ``(num_images_or_videos, 3)`` int tensor with
            ``(temporal_patches, height_patches, width_patches)`` per item.

        Returns:
          ``(merged_hidden_states, deepstack_features_list)`` where
          ``merged_hidden_states`` is the final post-merger output
          (matches HF's ``pooler_output``) and ``deepstack_features_list``
          is one tensor per ``deepstack_visual_indexes`` entry (matches
          HF's ``deepstack_features``).
        """
        target_dtype = self.dtype
        hidden_states = self.patch_embed(hidden_states)

        # Absolute (interpolated) pos embed.
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # 2D RoPE table → cos/sin of shape (seq_len, head_dim).
        rotary_pos_emb = self.rot_pos_emb(grid_thw)  # (seq_len, head_dim // 2)
        seq_len = hidden_states.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().to(target_dtype)
        sin = emb.sin().to(target_dtype)

        # cu_seqlens for FA2 varlen: each (t, h, w) image contributes
        # ``t`` segments of length ``h * w``. Build directly on-device
        # to avoid GPU↔CPU sync.
        seq_lens_per_segment = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        )
        cu_seqlens = seq_lens_per_segment.cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(torch.int32)
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        # Run blocks; tap deepstack hidden states at the specified layers.
        deepstack_lookup = {
            layer_idx: i for i, layer_idx in enumerate(self.deepstack_visual_indexes)
        }
        deepstack_features: list[torch.Tensor | None] = [None] * len(self.merger_list)

        for layer_idx, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cos=cos, sin=sin,
            )
            if layer_idx in deepstack_lookup:
                ds_idx = deepstack_lookup[layer_idx]
                deepstack_features[ds_idx] = self.merger_list[ds_idx](hidden_states)

        merged_hidden_states = self.merger(hidden_states)

        # Drop any None placeholders (shouldn't happen if config is sane).
        deepstack_out = [f for f in deepstack_features if f is not None]
        return merged_hidden_states, deepstack_out
