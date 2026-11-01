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
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.distributed.sequence_parallel import (
    gather_sequence,
    sp_seq_split,
    ulysses_attention,
)
from mstar.model.minimax_h3.config import NUM_MODALITIES, H3TransformerConfig

# (q, k, v) -> out over [B, S, H, D]
Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


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
    q/k RMSNorm, partial 3-axis rope, attention, output GEMM."""

    def __init__(self, cfg: H3TransformerConfig, comm_group: CommGroup):
        super().__init__()
        heads, head_dim = cfg.num_attention_heads, cfg.attention_head_dim
        self.head_dim = head_dim
        self.local_heads = heads // comm_group.world_size
        self.qkv_proj = QKVParallelLinear(comm_group, cfg.hidden_size, head_dim, heads, heads, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=cfg.qk_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=cfg.qk_norm_eps)
        self.out_proj = RowParallelLinear(comm_group, heads * head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, attend: Attend, rope: RopeTable | None = None, sp=None) -> torch.Tensor:
        local = self.local_heads * self.head_dim
        q, k, v = self.qkv_proj(x).split([local, local, local], dim=-1)
        q = self.q_norm(q.unflatten(-1, (self.local_heads, self.head_dim)))
        k = self.k_norm(k.unflatten(-1, (self.local_heads, self.head_dim)))
        v = v.unflatten(-1, (self.local_heads, self.head_dim))
        if rope is not None:
            q, k = apply_rope(q, rope), apply_rope(k, rope)
        out = attend(q, k, v) if sp is None else sp.attend(q, k, v, attend)
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

    def __init__(self, cfg: H3TransformerConfig, comm_group: CommGroup):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.norm1 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.attn = H3Attention(cfg, comm_group)
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

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor, row_index: torch.Tensor, rope: RopeTable, attend: Attend, sp=None,
    ) -> torch.Tensor:
        """``x`` ``[B, S, hidden]``; ``row_index`` ``[S]`` selects each row's table entry
        (``timestep * 3 + tag``)."""
        mods = self.modulation(temb, x.shape[0]).index_select(1, row_index)  # [B, S, 6, hidden]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.unbind(dim=2)
        h = self.norm1(x) * (1.0 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(h, attend, rope, sp)
        h = self.norm2(x) * (1.0 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.mlp(h)


class H3RefinerBlock(nn.Module):
    """``MiniMaxH3TokenRefinerBlock``: a plain pre-norm block, no AdaLN, no rope."""

    def __init__(self, cfg: H3TransformerConfig):
        super().__init__()
        trivial = CommGroup.trivial()
        self.norm1 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.attn = H3Attention(cfg, trivial)
        self.norm2 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.mlp = H3FeedForward(cfg, trivial)

    def forward(self, x: torch.Tensor, attend: Attend) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attend)
        return x + self.mlp(self.norm2(x))


class H3TextConditioner(nn.Module):
    """The prompt's path into the DiT: ``condition_proj`` and the token refiner."""

    def __init__(self, cfg: H3TransformerConfig):
        super().__init__()
        self.condition_proj = nn.Linear(cfg.text_dim, cfg.hidden_size)
        self.token_refiner = nn.Module()
        self.token_refiner.blocks = nn.ModuleList(H3RefinerBlock(cfg) for _ in range(cfg.token_refiner_num_layers))
        self.token_refiner.final_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.final_norm_eps)

    def forward(self, hidden: torch.Tensor, attend: Attend) -> torch.Tensor:
        """``[B, L, text_dim]`` -> ``[B, L, hidden]`` in the projection's dtype."""
        x = self.condition_proj(hidden.to(self.condition_proj.weight.dtype))
        for block in self.token_refiner.blocks:
            x = block(x, attend)
        return self.token_refiner.final_norm(x)


class H3FinalLayer(nn.Module):
    """``norm_out`` + the two fp32 heads. The modulation is per timestep."""

    def __init__(self, cfg: H3TransformerConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.final_norm_eps)
        self.adaln_proj = nn.Module()
        self.adaln_proj.linear = nn.Linear(cfg.time_embed_dim, cfg.final_adaln_out_features)
        self.video_out = nn.Linear(cfg.hidden_size, cfg.patch_dim)
        self.audio_out = nn.Linear(cfg.hidden_size, cfg.audio_latents_dim)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, row_class: torch.Tensor) -> torch.Tensor:
        """``[B, S, hidden]`` -> the normed, modulated rows in the heads' dtype."""
        linear = self.adaln_proj.linear
        mods = linear(F.silu(temb).to(linear.weight.dtype)).view(x.shape[0], -1, 2, self.hidden_size)
        shift, scale = mods.index_select(1, row_class).unbind(dim=2)
        h = self.norm(x) * (1.0 + scale) + shift
        return h.to(self.video_out.weight.dtype)


class SeqParallel(NamedTuple):
    """Ulysses over the packed rows: ``sizes`` is each rank's share of them."""

    group: CommGroup
    sizes: list[int]

    @property
    def start(self) -> int:
        return sum(self.sizes[: self.group.rank])

    @property
    def count(self) -> int:
        return self.sizes[self.group.rank]

    def attend(self, q, k, v, run: Attend) -> torch.Tensor:
        return ulysses_attention(self.group, q, k, v, run, self.sizes, seq_dim=1, head_dim=2)


class H3Inputs(NamedTuple):
    """One step's per-request inputs, batch-leading."""

    video: torch.Tensor      # [B, Nv, patch_dim] fp32 rows: keyframe rows, then generated rows
    audio: torch.Tensor      # [B, Na, audio_dim] fp32
    text: torch.Tensor       # [B, L, hidden] refined prompt rows
    timesteps: torch.Tensor  # [B, 3] fp32: (video, audio, keyframe) t in [0, 1], by T_* class


class H3StepLayout(NamedTuple):
    """One bucket's per-row tensors on the device, built once per layout."""

    rope: RopeTable           # [S, head_dim]
    row_index: torch.Tensor   # [S] AdaLN table row: timestep class * 3 + modality tag
    row_class: torch.Tensor   # [S] timestep class
    no_class: torch.Tensor    # [S] zeros: every row on the one timestep of a single-timestep step
    tags: torch.Tensor        # [S] modality tag
    text_len: int
    num_condition_rows: int   # keyframe rows, leading the video rows
    num_audio_rows: int


class H3DiT(nn.Module):
    """The block stack and its ends. ``forward`` returns the velocity of every video row
    (keyframe rows included, which the caller masks) and every audio row, fp32."""

    def __init__(self, cfg: H3TransformerConfig, comm_group: CommGroup | None = None,
                 sp_group: CommGroup | None = None):
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.cfg = cfg
        self.sp_group = sp_group or CommGroup.trivial()
        self.video_patch_proj = nn.Linear(cfg.patch_dim, cfg.hidden_size)
        self.audio_patch_proj = nn.Linear(cfg.audio_latents_dim, cfg.hidden_size)
        self.time_embedder = H3TimeEmbedder(cfg)
        self.blocks = nn.ModuleList(H3Block(cfg, comm_group) for _ in range(cfg.num_layers))
        self.final_layer = H3FinalLayer(cfg)

    # the modules the reference keeps in fp32 (``_keep_in_fp32_modules``)
    FP32_MODULES = ("video_patch_proj", "audio_patch_proj", "time_embedder", "final_layer.video_out",
                    "final_layer.audio_out")

    def keep_fp32(self) -> None:
        for name in self.FP32_MODULES:
            self.get_submodule(name).float()

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].norm1.weight.dtype

    def time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """``[B, n]`` -> ``[B * n, time_embed_dim]`` fp32."""
        return self.time_embedder(timesteps.reshape(-1))

    def forward(
        self, inputs: H3Inputs, layout: H3StepLayout, attend: Attend, row_class: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``inputs.timesteps`` is ``[B, 3]`` (video, audio, keyframe), or ``[B, 1]`` when all
        are one value: the reference then runs its AdaLN GEMMs on one row, which rounds
        differently from three. ``row_class`` (tests) indexes timesteps laid out as the
        reference's distinct values."""
        # [text | keyframe rows | audio | generated video], in the prompt's dtype
        dtype = inputs.text.dtype
        cond = layout.num_condition_rows
        video = self.video_patch_proj(inputs.video.to(self.video_patch_proj.weight.dtype)).to(dtype)
        audio = self.audio_patch_proj(inputs.audio.to(self.audio_patch_proj.weight.dtype)).to(dtype)
        x = torch.cat([inputs.text, video[:, :cond], audio, video[:, cond:]], dim=1)
        rope, row_index = layout.rope, layout.row_index
        if row_class is not None:
            row_index = row_class * NUM_MODALITIES + layout.tags
        elif inputs.timesteps.shape[1] == 1:
            row_class, row_index = layout.no_class, layout.tags
        else:
            row_class = layout.row_class
        sp = None
        if self.sp_group.world_size > 1:
            sp = SeqParallel(self.sp_group, sp_seq_split(x.shape[1], self.sp_group.world_size))
            x = x.narrow(1, sp.start, sp.count)
            rope = rope.rows(sp.start, sp.count)
            row_index = row_index.narrow(0, sp.start, sp.count)
            row_class = row_class.narrow(0, sp.start, sp.count)
        temb = self.time_embedding(inputs.timesteps)
        for block in self.blocks:
            x = block(x, temb, row_index, rope, attend, sp)
        h = self.final_layer(x, temb, row_class)
        # both heads over this rank's rows, then the rows gathered: [B, S, patch_dim + audio_dim]
        out = torch.cat([self.final_layer.video_out(h), self.final_layer.audio_out(h)], dim=-1)
        if sp is not None:
            out = gather_sequence(sp.group, out, sp.sizes, dim=1)
        text, audio_rows = layout.text_len, layout.num_audio_rows
        patch_dim = self.cfg.patch_dim
        video_v = torch.cat(
            [out[:, text:text + cond, :patch_dim], out[:, text + cond + audio_rows:, :patch_dim]], dim=1,
        )
        audio_v = out[:, text + cond:text + cond + audio_rows, patch_dim:]
        return video_v, audio_v
