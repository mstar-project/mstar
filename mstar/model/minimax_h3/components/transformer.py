"""The MiniMax-H3 transformer: a 50-block single-stream DiT over one packed sequence per
request, each row modulated by AdaLN rows picked from a ``(timestep, modality)`` table.
Ported from diffusers' ``transformer_minimax_h3.py`` with its op order and dtypes.

Where it differs from the reference without changing the numbers:

* The token refiner reads no timestep, so ``H3TextConditioner`` runs it once per request
  on the text node.
* The reference indexes AdaLN rows among a step's distinct timesteps (``torch.unique``).
  Here a step carries ``(video, audio, keyframe)`` and a row picks its own by class, so
  the table's shape is the same every step.
* Under Ulysses SP each rank holds all the weights and ``1/P`` of the rows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.state import gather, scatter, split_range
from mstar.model.components.diffusion.attention import attend
from mstar.model.components.distributed.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from mstar.model.minimax_h3.components.qk_norm_rope import qk_norm_rope
from mstar.model.minimax_h3.config import DIT_ROWS, NUM_MODALITIES, PROMPT_ROWS, H3TransformerConfig


class RopeTable(NamedTuple):
    """``cos`` / ``sin`` ``[S, head_dim]`` of the packed rows, in the activation dtype, over
    the whole head: the pass-through channels at cos 1, sin 0, and rotate-half's sign
    folded into ``sin``."""

    cos: torch.Tensor
    sin: torch.Tensor
    rotary_dim: int

    def rows(self, start: int, count: int) -> "RopeTable":
        return RopeTable(self.cos.narrow(0, start, count), self.sin.narrow(0, start, count), self.rotary_dim)


def rope_inv_freq(cfg: H3TransformerConfig) -> torch.Tensor:
    """``MiniMaxH3RotaryPosEmbed.inv_freq``: one frequency table shared by the three axes,
    computed on the CPU in fp32 as the reference builds it at init."""
    n = cfg.rope_inv_freq_len
    return 1.0 / (cfg.rope_theta ** (torch.arange(0, 2 * n, 2, dtype=torch.float32) / (2 * n)))


def rope_table(position_ids: torch.Tensor, inv_freq: torch.Tensor, dtype: torch.dtype, head_dim: int) -> RopeTable:
    """``MiniMaxH3RotaryPosEmbed.forward``: ``[S, 3]`` positions (rounded to fp32) times one
    frequency table per axis, concatenated ``(t, h, w)`` and doubled, then cos/sin in fp32
    on the positions' device, cast to ``dtype`` where the reference casts them."""
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.to(position_ids.device).view(1, 1, -1)
    freqs = torch.cat(freqs.unbind(dim=1), dim=-1)
    freqs = torch.cat((freqs, freqs), dim=-1)
    cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
    rot, half = cos.shape[-1], cos.shape[-1] // 2
    rest = (cos.shape[0], head_dim - rot)
    return RopeTable(
        torch.cat((cos, cos.new_ones(rest)), dim=-1),
        torch.cat((-sin[:, :half], sin[:, half:], sin.new_zeros(rest)), dim=-1),
        rot,
    )


def apply_rope(x: torch.Tensor, rope: RopeTable) -> torch.Tensor:
    """``_apply_rotary_emb`` on ``[B, S, H, D]``: rotate-half over the leading
    ``rotary_dim`` channels in the activation dtype, the rest passed through. Written
    as one product over the whole head (``x * cos + partner * sin``, each channel's
    partner the one rotate-half moves into it), which rounds every channel as the
    reference does and lets a compiled block fuse it with the q/k norm before it."""
    rot = rope.rotary_dim
    half = rot // 2
    partner = torch.cat((x[..., half:rot], x[..., :half], x[..., rot:]), dim=-1)
    return x * rope.cos[None, :, None, :] + partner * rope.sin[None, :, None, :]


def timestep_sinusoid(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """``Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)`` in fp32, on unscaled
    ``t`` in [0, 1]: diffusers' ``get_timestep_embedding`` op for op."""
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return torch.cat([emb[:, half:], emb[:, :half]], dim=-1)


class H3TimeEmbedder(nn.Module):
    """``TimestepEmbedding``: linear -> SiLU -> linear, fp32."""

    def __init__(self, cfg: H3TransformerConfig):
        super().__init__()
        self.proj_in = nn.Linear(cfg.timestep_input_dim, cfg.time_embed_hidden_size)
        self.proj_out = nn.Linear(cfg.time_embed_hidden_size, cfg.time_embed_dim)
        self.dim = cfg.timestep_input_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t = timestep_sinusoid(timesteps, self.dim).to(self.proj_in.weight.dtype)
        return self.proj_out(F.silu(self.proj_in(t)))


class H3Attention(nn.Module):
    """Self-attention: one fused q/k/v GEMM (heads sharded over ``comm_group``), per-head
    q/k RMSNorm, partial 3-axis rope, attention through ``array``'s read, output GEMM."""

    def __init__(self, cfg: H3TransformerConfig, comm_group: CommGroup, array: str, layer: int):
        super().__init__()
        heads, head_dim = cfg.num_attention_heads, cfg.attention_head_dim
        self.array = array
        self.layer = torch.tensor(layer, device="cpu")
        self.state = None
        self.head_dim = head_dim
        self.local_heads = heads // comm_group.world_size
        self.qkv_proj = QKVParallelLinear(comm_group, cfg.hidden_size, head_dim, heads, heads, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=cfg.qk_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=cfg.qk_norm_eps)
        self.out_proj = RowParallelLinear(comm_group, heads * head_dim, cfg.hidden_size, bias=False)

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.state = handles.get(self.array)

    def forward(self, x: torch.Tensor, rope: RopeTable | None = None) -> torch.Tensor:
        local = self.local_heads * self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([local, local, local], dim=-1)
        v = v.unflatten(-1, (self.local_heads, self.head_dim))
        if rope is not None and qkv.shape[0] == 1 and torch.compiler.is_compiling():
            # one kernel for both norms and the rope (qk_norm_rope.py); eager keeps the torch ops
            weights = torch.stack([self.q_norm.weight, self.k_norm.weight])
            qk = qk_norm_rope(qkv[0], weights, rope.cos, rope.sin, self.q_norm.eps, self.local_heads, rope.rotary_dim)
            q, k = qk[None].split(self.local_heads, dim=2)
        else:
            q = self.q_norm(q.unflatten(-1, (self.local_heads, self.head_dim)))
            k = self.k_norm(k.unflatten(-1, (self.local_heads, self.head_dim)))
            if rope is not None:
                q, k = apply_rope(q, rope), apply_rope(k, rope)
        out = attend(self.state, self.layer, q, k, v)
        return self.out_proj(out.flatten(2, 3).type_as(q))


class H3FeedForward(nn.Module):
    """SwiGLU: ``fc1`` is ``[gate; value]`` (the checkpoint's order), ``silu(gate) * value``."""

    def __init__(self, cfg: H3TransformerConfig, comm_group: CommGroup):
        super().__init__()
        inner = cfg.ffn_hidden_size
        self.local = inner // comm_group.world_size
        self.fc1 = MergedColumnParallelLinear(comm_group, cfg.hidden_size, [inner, inner], bias=False)
        self.fc2 = RowParallelLinear(comm_group, inner, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc1(x).split([self.local, self.local], dim=-1)
        return self.fc2(value * F.silu(gate))


class H3Block(nn.Module):
    """``MiniMaxH3TransformerBlock``: AdaLN-modulated pre-norm attention and feed-forward.

    ``adaln_proj`` maps the step's time embeddings ``[B * n, time_embed_dim]`` (the ``n``
    timesteps of each row of the batch) to ``[B, n * 3, 6, hidden]``: per (timestep,
    modality), the ``shift, scale, gate`` of the attention and of the feed-forward."""

    def __init__(self, cfg: H3TransformerConfig, comm_group: CommGroup, layer: int):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.norm1 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.attn = H3Attention(cfg, comm_group, DIT_ROWS, layer)
        self.norm2 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.mlp = H3FeedForward(cfg, comm_group)
        self.adaln_proj = nn.Module()
        self.adaln_proj.linear = nn.Linear(cfg.time_embed_dim, cfg.adaln_out_features)

    def modulation(self, temb: torch.Tensor, batch: int) -> torch.Tensor:
        """``[B, n * 3, 6, hidden]``: the activation at ``temb``'s precision (fp32), then
        the cast to the bf16 projection, as the reference does."""
        linear = self.adaln_proj.linear
        mods = linear(F.silu(temb).to(linear.weight.dtype))
        return mods.view(batch, -1, 6, self.hidden_size)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, row_index: torch.Tensor, rope: RopeTable) -> torch.Tensor:
        """``x`` ``[B, S, hidden]``; ``row_index`` ``[S]`` selects each row's table entry
        (``timestep * 3 + tag``)."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(temb, x.shape[0]).unbind(2)

        def rows(term: torch.Tensor) -> torch.Tensor:
            # [B, n * 3, hidden] -> [B, S, hidden] where it is used, so a compiled block carries
            # the small table across its break at the attention, not a gathered row per term
            return term.index_select(1, row_index)

        h = self.norm1(x) * (1.0 + rows(scale_msa)) + rows(shift_msa)
        x = x + rows(gate_msa) * self.attn(h, rope)
        h = self.norm2(x) * (1.0 + rows(scale_mlp)) + rows(shift_mlp)
        return x + rows(gate_mlp) * self.mlp(h)


class H3RefinerBlock(nn.Module):
    """``MiniMaxH3TokenRefinerBlock``: a plain pre-norm block, no AdaLN, no rope."""

    def __init__(self, cfg: H3TransformerConfig, layer: int):
        super().__init__()
        trivial = CommGroup.trivial()
        self.norm1 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.attn = H3Attention(cfg, trivial, PROMPT_ROWS, layer)
        self.norm2 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.mlp = H3FeedForward(cfg, trivial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class H3TextConditioner(nn.Module):
    """The prompt's path into the DiT: ``condition_proj`` and the token refiner."""

    def __init__(self, cfg: H3TransformerConfig):
        super().__init__()
        self.condition_proj = nn.Linear(cfg.text_dim, cfg.hidden_size)
        self.token_refiner = nn.Module()
        self.token_refiner.blocks = nn.ModuleList(H3RefinerBlock(cfg, i) for i in range(cfg.token_refiner_num_layers))
        self.token_refiner.final_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.final_norm_eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[B, L, text_dim]`` -> ``[B, L, hidden]`` in the projection's dtype."""
        x = self.condition_proj(hidden.to(self.condition_proj.weight.dtype))
        for block in self.token_refiner.blocks:
            x = block(x)
        return self.token_refiner.final_norm(x)


class StepHeads(nn.Module):
    """One linear head per sampling step, stacked: ``weight [n, out, in]``, ``bias [n, out]``.
    ``forward`` runs the head ``index`` (a ``[1]`` device tensor) names as a plain linear, so
    a captured step reads its head from the step's scalar rather than a Python int."""

    def __init__(self, num_heads: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_heads, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(num_heads, out_features))

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.index_select(0, index)[0], self.bias.index_select(0, index)[0])


class H3FinalLayer(nn.Module):
    """``norm_out`` + the fp32 heads (video, and audio when the model has it). The
    modulation is per timestep."""

    def __init__(self, cfg: H3TransformerConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.final_norm_eps)
        self.adaln_proj = nn.Module()
        self.adaln_proj.linear = nn.Linear(cfg.time_embed_dim, cfg.final_adaln_out_features)
        if cfg.num_output_heads == 1:
            self.video_out = nn.Linear(cfg.hidden_size, cfg.patch_dim)
        else:
            self.video_out = StepHeads(cfg.num_output_heads, cfg.hidden_size, cfg.patch_dim)
        self.audio_out = nn.Linear(cfg.hidden_size, cfg.audio_latents_dim) if cfg.audio_enabled else None

    def forward(self, x: torch.Tensor, temb: torch.Tensor, row_class: torch.Tensor) -> torch.Tensor:
        """``[B, S, hidden]`` -> the normed, modulated rows in the heads' dtype."""
        linear = self.adaln_proj.linear
        mods = linear(F.silu(temb).to(linear.weight.dtype)).view(x.shape[0], -1, 2, self.hidden_size)
        shift, scale = mods.index_select(1, row_class).unbind(dim=2)
        h = self.norm(x) * (1.0 + scale) + shift
        return h.to(self.video_out.weight.dtype)


class H3Inputs(NamedTuple):
    """One step's per-request inputs, batch-leading."""

    video: torch.Tensor      # [B, Nv, patch_dim] fp32 rows: keyframe rows, then generated rows
    audio: torch.Tensor      # [B, Na, audio_dim] fp32
    text: torch.Tensor       # [B, L, hidden] refined prompt rows
    timesteps: torch.Tensor  # [B, 3] fp32: (video, audio, keyframe) t in [0, 1], by T_* class


class H3StepLayout(NamedTuple):
    """One bucket's per-row tensors on the device, built once per layout. ``S`` counts the
    ``num_pad_rows`` padding rows at the tail: an attention segment of their own, zeros
    in, and read by no output."""

    rope: RopeTable           # [S, head_dim]
    row_index: torch.Tensor   # [S] AdaLN table row: timestep class * 3 + modality tag
    row_class: torch.Tensor   # [S] timestep class
    no_class: torch.Tensor    # [S] zeros: every row on the one timestep of a single-timestep step
    tags: torch.Tensor        # [S] modality tag
    text_len: int
    num_condition_rows: int   # keyframe rows, leading the video rows
    num_audio_rows: int
    num_pad_rows: int = 0
    # Set where the prompt's length is not the step's to read (one captured graph per row
    # bucket): the prompt arrives padded with zero rows to the bucket's capacity,
    # ``row_source`` [S] picks each packed row from ``[prompt rows | video rows]`` (a padding
    # row from a zero prompt row) and ``video_rows`` [Nv] are the video rows' packed rows.
    # Video only; ``text_len`` and ``num_pad_rows`` then go unread.
    row_source: torch.Tensor | None = None
    video_rows: torch.Tensor | None = None

    @property
    def num_rows(self) -> int:
        """``S``: the packed rows, padding included."""
        return self.row_index.shape[0]


class H3DiT(nn.Module):
    """The block stack and its ends. ``forward`` returns the velocity of every video row
    (keyframe rows included, which the caller masks) and every audio row, fp32."""

    def __init__(self, cfg: H3TransformerConfig, comm_group: CommGroup | None = None):
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.cfg = cfg
        self.video_patch_proj = nn.Linear(cfg.patch_dim, cfg.hidden_size)
        self.audio_patch_proj = nn.Linear(cfg.audio_latents_dim, cfg.hidden_size) if cfg.audio_enabled else None
        self.time_embedder = H3TimeEmbedder(cfg)
        self.blocks = nn.ModuleList(H3Block(cfg, comm_group, i) for i in range(cfg.num_layers))
        self.final_layer = H3FinalLayer(cfg)

    # the modules the reference keeps in fp32 (``_keep_in_fp32_modules``)
    FP32_MODULES = ("video_patch_proj", "audio_patch_proj", "time_embedder", "final_layer.video_out",
                    "final_layer.audio_out")

    def keep_fp32(self) -> None:
        for name in self.FP32_MODULES:
            parent, _, leaf = name.rpartition(".")
            module = getattr(self.get_submodule(parent), leaf)
            if module is not None:  # a model without audio has no audio modules
                module.float()

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].norm1.weight.dtype

    def time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """``[B, n]`` -> ``[B * n, time_embed_dim]`` fp32."""
        return self.time_embedder(timesteps.reshape(-1))

    def packed_rows(self, inputs: H3Inputs, layout: H3StepLayout, start: int, count: int) -> torch.Tensor:
        """Rows ``[start, start + count)`` of ``[text | keyframe rows | audio | generated video |
        padding]`` in the prompt's dtype, each modality projected over its rows there alone."""
        dtype, cond = inputs.text.dtype, layout.num_condition_rows
        sizes = (layout.text_len, cond, layout.num_audio_rows, inputs.video.shape[1] - cond, layout.num_pad_rows)
        spans, first = [], 0
        for size in sizes:
            # the range's part of this piece, counted from the piece's first row
            spans.append((min(max(start - first, 0), size), min(max(start + count - first, 0), size)))
            first += size
        (t0, t1), (c0, c1), (a0, a1), (g0, g1), (p0, p1) = spans

        def project(linear: nn.Module, rows: torch.Tensor) -> torch.Tensor:
            return linear(rows.to(linear.weight.dtype)).to(dtype)

        # keyframe and generated rows are one tensor, and a range holding both holds the end of
        # one and the start of the other: one projection over the video rows it holds
        v0, v1 = (c0 if c1 > c0 else cond + g0), (cond + g1 if g1 > g0 else c1)
        video = project(self.video_patch_proj, inputs.video[:, v0:v1]) if v1 > v0 else None
        pieces = [inputs.text[:, t0:t1]] if t1 > t0 else []
        if c1 > c0:
            pieces.append(video[:, :c1 - c0])
        if a1 > a0:
            pieces.append(project(self.audio_patch_proj, inputs.audio[:, a0:a1]))
        if g1 > g0:
            pieces.append(video[:, c1 - c0:])
        if p1 > p0:
            pieces.append(inputs.text.new_zeros(inputs.text.shape[0], p1 - p0, inputs.text.shape[2]))
        return torch.cat(pieces, dim=1)

    def sourced_rows(self, inputs: H3Inputs, layout: H3StepLayout, start: int, count: int) -> torch.Tensor:
        """:meth:`packed_rows` by ``layout.row_source``: every video row projected (a row's projection
        is the same whichever rows run with it), then the range's rows picked."""
        if layout.num_audio_rows:
            raise ValueError("a layout of row sources holds no audio rows")
        linear = self.video_patch_proj
        video = linear(inputs.video.to(linear.weight.dtype)).to(inputs.text.dtype)
        return torch.cat([inputs.text, video], dim=1).index_select(1, layout.row_source.narrow(0, start, count))

    def forward(
        self, inputs: H3Inputs, layout: H3StepLayout, row_class: torch.Tensor | None = None, shard=None,
        head: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``inputs.timesteps`` is ``[B, n]``: one timestep per class ``layout.row_class``
        indexes (H3: video, audio, keyframe), or ``[B, 1]`` when all are one value: the
        reference then runs its AdaLN GEMMs on one row, which rounds differently from three.
        ``row_class`` (tests) indexes timesteps laid out as the reference's distinct values.
        `attend` attends within the step's segments (`attention_segments`), so no request row
        sees a padding row. ``shard``: the node's sequence shard's handle, which on a step it
        splits gives this rank a contiguous range of the packed rows: only those are built and
        run, and the attention's resource exchanges them. ``head`` (``[1]``) picks the step's
        video head when the model has one per step. The audio velocity is None for a model
        without audio."""
        cond, pad = layout.num_condition_rows, layout.num_pad_rows
        split = None if shard is None else split_range(shard)
        start, count = (0, layout.num_rows) if split is None else (split.start, split.count)
        if layout.row_source is None:
            x = self.packed_rows(inputs, layout, start, count)
        else:
            x = self.sourced_rows(inputs, layout, start, count)
        rope, row_index = layout.rope, layout.row_index
        if row_class is not None:
            row_class = torch.cat([row_class, row_class.new_zeros(pad)])
            row_index = row_class * NUM_MODALITIES + layout.tags
        elif inputs.timesteps.shape[1] == 1:
            row_class, row_index = layout.no_class, layout.tags
        else:
            row_class = layout.row_class
        if split is not None:
            rope = rope.rows(start, count)
            row_index, row_class = scatter(shard, row_index, row_class, dim=0)
        temb = self.time_embedding(inputs.timesteps)
        for block in self.blocks:
            x = block(x, temb, row_index, rope)
        h = self.final_layer(x, temb, row_class)
        final = self.final_layer
        video_out = final.video_out(h) if head is None else final.video_out(h, head)
        # the heads over this rank's rows, then the rows gathered: [B, S, patch_dim (+ audio_dim)]
        out = video_out if final.audio_out is None else torch.cat([video_out, final.audio_out(h)], dim=-1)
        if split is not None:
            out = gather(shard, out, dim=1)
        if layout.video_rows is not None:
            return out.index_select(1, layout.video_rows), None
        text, audio_rows, end = layout.text_len, layout.num_audio_rows, layout.num_rows - pad
        patch_dim = self.cfg.patch_dim
        if final.audio_out is None:
            return out[:, text:end], None
        video_v = torch.cat(
            [out[:, text:text + cond, :patch_dim], out[:, text + cond + audio_rows:end, :patch_dim]], dim=1,
        )
        audio_v = out[:, text + cond:text + cond + audio_rows, patch_dim:]
        return video_v, audio_v
