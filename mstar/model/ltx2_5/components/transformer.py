"""The LTX-2.5 joint audio-video DiT (``LTX2VideoTransformer3DModel``), native.

Per block: video and audio self-attention, each stream's text cross-attention,
audio->video and video->audio cross-attention, then each stream's feed-forward;
every sub-layer is AdaLN-modulated by the timestep. Ported from diffusers'
``transformer_ltx2.py`` with its op order kept (see ``LTX2DiT.forward``).

Inputs are already patchified tokens (patch size 1): video ``[B, Lv, 128]`` and
audio ``[B, La, 128]``. Rotary tables are per request shape, so the caller builds
them once (``build_rope``) and passes them in. The six attentions run through
``Attend`` callables the caller supplies (``LTX2Attends``), which is where the
engine's ragged attention resource plugs in.

Only the configuration LTX-2.5 ships is implemented: cross-attention AdaLN on both
streams with a timestep-dependent prompt modulation (``use_prompt_adaln_single``),
gated attention and split RoPE; ``LTX2TransformerConfig.validate`` rejects others.
Spatio-temporal guidance and modality isolation (the SFT recipe's extra passes) are
not ported yet.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.ltx2_5.components.layers import (
    Attend,
    GatedAttention,
    GeluFeedForward,
    RotaryTable,
    rms_norm_no_weight,
    split_rope_table,
)
from mstar.model.ltx2_5.config import LTX2TransformerConfig


def timestep_sinusoid(timesteps: torch.Tensor) -> torch.Tensor:
    """``Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)``, fp32. The
    reference's own function, so the frequencies round exactly as the checkpoint saw."""
    from diffusers.models.embeddings import get_timestep_embedding

    return get_timestep_embedding(timesteps, 256, flip_sin_to_cos=True, downscale_freq_shift=0)


class TimestepEmbedder(nn.Module):
    """``TimestepEmbedding``: linear -> SiLU -> linear over the 256-channel sinusoid."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.linear_2(nn.functional.silu(self.linear_1(t)))


class AdaLNSingle(nn.Module):
    """``LTX2AdaLayerNormSingle``: the timestep embedding and ``num_mod`` modulation rows.

    Returns ``(mods [B, 1, num_mod * dim], embedded_timestep [B, 1, dim])``. The
    sinusoid is fp32 and cast to the weights' dtype before the MLP, as the reference.
    """

    def __init__(self, dim: int, num_mod: int):
        super().__init__()
        self.num_mod = num_mod
        self.emb = nn.Module()
        self.emb.timestep_embedder = TimestepEmbedder(dim)
        self.linear = nn.Linear(dim, num_mod * dim)

    def forward(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = timestep.shape[0]
        sinusoid = timestep_sinusoid(timestep.flatten()).to(self.linear.weight.dtype)
        embedded = self.emb.timestep_embedder(sinusoid)
        mods = self.linear(nn.functional.silu(embedded))
        return mods.view(batch, -1, mods.shape[-1]), embedded.view(batch, -1, embedded.shape[-1])


def modulation(table: torch.Tensor, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``get_mod_params``: the per-block table plus the global rows, one ``[B, T, dim]``
    tensor per row."""
    batch, tokens = temb.shape[0], temb.shape[1]
    values = table[None, None].to(temb.device) + temb.reshape(batch, tokens, table.shape[0], -1)
    return values.unbind(dim=2)


class LTX2Rope(NamedTuple):
    """One request shape's rotary tables, built once and reused every step."""

    video: RotaryTable           # video self-attention, [B, Lv, 32, 64]
    audio: RotaryTable           # audio self-attention, [B, La, 32, 32]
    cross_video: RotaryTable     # video side of a2v / v2a, time axis only
    cross_audio: RotaryTable     # audio side of a2v / v2a


class LTX2Attends(NamedTuple):
    """The attention each of a block's six attentions runs through."""

    video_self: Attend
    audio_self: Attend
    video_text: Attend
    audio_text: Attend
    audio_to_video: Attend      # q: video, kv: audio
    video_to_audio: Attend      # q: audio, kv: video


class TemporalEmbeddings(NamedTuple):
    """The step's global modulation rows, shared by every block."""

    video: torch.Tensor
    audio: torch.Tensor
    video_prompt: torch.Tensor
    audio_prompt: torch.Tensor
    video_cross_scale_shift: torch.Tensor
    audio_cross_scale_shift: torch.Tensor
    video_cross_gate: torch.Tensor
    audio_cross_gate: torch.Tensor


class LTX2Block(nn.Module):
    def __init__(self, cfg: LTX2TransformerConfig, comm_group: CommGroup):
        super().__init__()
        dim, adim = cfg.inner_dim, cfg.audio_inner_dim
        self.eps = cfg.norm_eps
        common = dict(comm_group=comm_group, bias=cfg.attention_bias, out_bias=cfg.attention_out_bias, eps=cfg.norm_eps)
        self.attn1 = GatedAttention(
            dim, cfg.num_attention_heads, cfg.attention_head_dim, gated=cfg.gated_attn, **common,
        )
        self.audio_attn1 = GatedAttention(
            adim, cfg.audio_num_attention_heads, cfg.audio_attention_head_dim, gated=cfg.audio_gated_attn, **common,
        )
        self.attn2 = GatedAttention(
            dim, cfg.num_attention_heads, cfg.attention_head_dim, kv_dim=cfg.cross_attention_dim,
            gated=cfg.gated_attn, **common,
        )
        self.audio_attn2 = GatedAttention(
            adim, cfg.audio_num_attention_heads, cfg.audio_attention_head_dim, kv_dim=cfg.audio_cross_attention_dim,
            gated=cfg.audio_gated_attn, **common,
        )
        # both cross-modal attentions use the audio stream's head layout
        self.audio_to_video_attn = GatedAttention(
            dim, cfg.audio_num_attention_heads, cfg.audio_attention_head_dim, kv_dim=adim,
            gated=cfg.gated_attn, **common,
        )
        self.video_to_audio_attn = GatedAttention(
            adim, cfg.audio_num_attention_heads, cfg.audio_attention_head_dim, kv_dim=dim,
            gated=cfg.audio_gated_attn, **common,
        )
        self.ff = GeluFeedForward(dim, comm_group, bias=cfg.ff_bias)
        self.audio_ff = GeluFeedForward(adim, comm_group, bias=cfg.audio_ff_bias)
        # rows: shift/scale/gate for self-attn, for the FF, then for the text query
        self.scale_shift_table = nn.Parameter(torch.empty(9, dim))
        self.audio_scale_shift_table = nn.Parameter(torch.empty(9, adim))
        # rows: shift, scale of the text keys/values
        self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, dim))
        self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty(2, adim))
        # rows: a2v scale, a2v shift, v2a scale, v2a shift, gate
        self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.empty(5, dim))
        self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.empty(5, adim))

    def forward(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        text: torch.Tensor,
        audio_text: torch.Tensor,
        temb: TemporalEmbeddings,
        rope: LTX2Rope,
        attends: LTX2Attends,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = self.eps
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_tq, scale_tq, gate_tq = (
            modulation(self.scale_shift_table, temb.video)
        )
        (a_shift_msa, a_scale_msa, a_gate_msa, a_shift_mlp, a_scale_mlp, a_gate_mlp,
         a_shift_tq, a_scale_tq, a_gate_tq) = modulation(self.audio_scale_shift_table, temb.audio)

        # 1. self-attention
        h = rms_norm_no_weight(x, eps) * (1 + scale_msa) + shift_msa
        x = x + self.attn1(h, attends.video_self, q_rope=rope.video) * gate_msa
        h = rms_norm_no_weight(a, eps) * (1 + a_scale_msa) + a_shift_msa
        a = a + self.audio_attn1(h, attends.audio_self, q_rope=rope.audio) * a_gate_msa

        # 2. text cross-attention; the text keys/values are modulated too
        shift_tkv, scale_tkv = modulation(self.prompt_scale_shift_table, temb.video_prompt)
        a_shift_tkv, a_scale_tkv = modulation(self.audio_prompt_scale_shift_table, temb.audio_prompt)
        h = rms_norm_no_weight(x, eps) * (1 + scale_tq) + shift_tq
        context = text * (1 + scale_tkv) + shift_tkv
        x = x + self.attn2(h, attends.video_text, context=context) * gate_tq
        h = rms_norm_no_weight(a, eps) * (1 + a_scale_tq) + a_shift_tq
        context = audio_text * (1 + a_scale_tkv) + a_shift_tkv
        a = a + self.audio_attn2(h, attends.audio_text, context=context) * a_gate_tq

        # 3. audio <-> video; both directions read the streams as they are here
        norm_x, norm_a = rms_norm_no_weight(x, eps), rms_norm_no_weight(a, eps)
        v_table = self.video_a2v_cross_attn_scale_shift_table
        a_table = self.audio_a2v_cross_attn_scale_shift_table
        v_a2v_scale, v_a2v_shift, v_v2a_scale, v_v2a_shift = modulation(v_table[:4], temb.video_cross_scale_shift)
        a2v_gate = modulation(v_table[4:], temb.video_cross_gate)[0]
        a_a2v_scale, a_a2v_shift, a_v2a_scale, a_v2a_shift = modulation(a_table[:4], temb.audio_cross_scale_shift)
        v2a_gate = modulation(a_table[4:], temb.audio_cross_gate)[0]

        q_in = norm_x * (1 + v_a2v_scale) + v_a2v_shift
        kv_in = norm_a * (1 + a_a2v_scale) + a_a2v_shift
        x = x + a2v_gate * self.audio_to_video_attn(
            q_in, attends.audio_to_video, context=kv_in, q_rope=rope.cross_video, k_rope=rope.cross_audio,
        )
        q_in = norm_a * (1 + a_v2a_scale) + a_v2a_shift
        kv_in = norm_x * (1 + v_v2a_scale) + v_v2a_shift
        a = a + v2a_gate * self.video_to_audio_attn(
            q_in, attends.video_to_audio, context=kv_in, q_rope=rope.cross_audio, k_rope=rope.cross_video,
        )

        # 4. feed-forward
        x = x + self.ff(rms_norm_no_weight(x, eps) * (1 + scale_mlp) + shift_mlp) * gate_mlp
        a = a + self.audio_ff(rms_norm_no_weight(a, eps) * (1 + a_scale_mlp) + a_shift_mlp) * a_gate_mlp
        return x, a


class LTX2DiT(nn.Module):
    def __init__(self, cfg: LTX2TransformerConfig, comm_group: CommGroup | None = None):
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.cfg = cfg
        dim, adim = cfg.inner_dim, cfg.audio_inner_dim
        self.proj_in = nn.Linear(cfg.in_channels, dim)
        self.audio_proj_in = nn.Linear(cfg.audio_in_channels, adim)
        # Marks single-pixel-frame keyframe tokens in the keyframe pipelines; text-to-video
        # never applies it, but the checkpoint carries it.
        self.keyframes_abs_pos_embedding = nn.Parameter(torch.empty(1, dim))
        self.time_embed = AdaLNSingle(dim, 9)
        self.audio_time_embed = AdaLNSingle(adim, 9)
        self.prompt_adaln = AdaLNSingle(dim, 2)
        self.audio_prompt_adaln = AdaLNSingle(adim, 2)
        self.av_cross_attn_video_scale_shift = AdaLNSingle(dim, 4)
        self.av_cross_attn_audio_scale_shift = AdaLNSingle(adim, 4)
        self.av_cross_attn_video_a2v_gate = AdaLNSingle(dim, 1)
        self.av_cross_attn_audio_v2a_gate = AdaLNSingle(adim, 1)
        self.transformer_blocks = nn.ModuleList([LTX2Block(cfg, comm_group) for _ in range(cfg.num_layers)])
        self.scale_shift_table = nn.Parameter(torch.empty(2, dim))
        self.audio_scale_shift_table = nn.Parameter(torch.empty(2, adim))
        self.proj_out = nn.Linear(dim, cfg.out_channels)
        self.audio_proj_out = nn.Linear(adim, cfg.audio_out_channels)

    @property
    def dtype(self) -> torch.dtype:
        return self.proj_in.weight.dtype

    def temporal_embeddings(self, timestep: torch.Tensor) -> tuple[TemporalEmbeddings, torch.Tensor, torch.Tensor]:
        """The global modulation rows for one step, plus the output layers' embeddings.

        Text-to-video conditions every token of a request on its one timestep, and
        the reference feeds that same ``t`` as ``sigma`` (prompt modulation) and, with
        ``use_cross_timestep``, as the other modality's timestep for the cross-modal
        rows, so a single ``[B]`` timestep drives all eight embeddings.
        """
        scale = self.cfg.cross_attn_timestep_scale_multiplier / self.cfg.timestep_scale_multiplier
        video, video_embedded = self.time_embed(timestep)
        audio, audio_embedded = self.audio_time_embed(timestep)
        temb = TemporalEmbeddings(
            video=video,
            audio=audio,
            video_prompt=self.prompt_adaln(timestep)[0],
            audio_prompt=self.audio_prompt_adaln(timestep)[0],
            video_cross_scale_shift=self.av_cross_attn_video_scale_shift(timestep)[0],
            audio_cross_scale_shift=self.av_cross_attn_audio_scale_shift(timestep)[0],
            video_cross_gate=self.av_cross_attn_video_a2v_gate(timestep * scale)[0],
            audio_cross_gate=self.av_cross_attn_audio_v2a_gate(timestep * scale)[0],
        )
        return temb, video_embedded, audio_embedded

    def forward(
        self,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        text: torch.Tensor,
        audio_text: torch.Tensor,
        timestep: torch.Tensor,
        rope: LTX2Rope,
        attends: LTX2Attends,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One velocity prediction. ``latents`` / ``audio_latents`` in the model dtype;
        ``timestep`` is ``sigma * 1000``, ``[B]`` fp32. Returns ``(video, audio)``
        velocities in the model dtype."""
        x = self.proj_in(latents)
        a = self.audio_proj_in(audio_latents)
        temb, video_embedded, audio_embedded = self.temporal_embeddings(timestep)
        for block in self.transformer_blocks:
            x, a = block(x, a, text, audio_text, temb, rope, attends)

        shift, scale = (self.scale_shift_table[None, None] + video_embedded[:, :, None]).unbind(dim=2)
        x = nn.functional.layer_norm(x, (x.shape[-1],), eps=1e-6) * (1 + scale) + shift
        a_shift, a_scale = (self.audio_scale_shift_table[None, None] + audio_embedded[:, :, None]).unbind(dim=2)
        a = nn.functional.layer_norm(a, (a.shape[-1],), eps=1e-6) * (1 + a_scale) + a_shift
        return self.proj_out(x), self.audio_proj_out(a)


def video_positions(
    cfg: LTX2TransformerConfig, frames: int, height: int, width: int, fps: float, device,
) -> torch.Tensor:
    """``prepare_video_coords`` reduced to patch midpoints: ``[1, F*H*W, 3]`` fp32 of
    (seconds, pixel row, pixel column), the first latent frame causally shifted."""
    grid = torch.meshgrid(
        torch.arange(frames, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    starts = torch.stack(grid, dim=0).flatten(1)                # [3, N] latent coords
    bounds = torch.stack([starts, starts + 1], dim=-1)          # patch size 1 everywhere
    scale = torch.tensor(cfg.vae_scale_factors, device=device, dtype=torch.float32).view(3, 1, 1)
    pixels = bounds * scale
    temporal = cfg.vae_scale_factors[0]
    pixels[0] = (pixels[0] + cfg.causal_offset - temporal).clamp(min=0) / fps
    return ((pixels[..., 0] + pixels[..., 1]) / 2.0).T.unsqueeze(0)


def audio_positions(cfg: LTX2TransformerConfig, audio_frames: int, device) -> torch.Tensor:
    """``prepare_audio_coords`` reduced to midpoints: ``[1, La, 1]`` seconds."""
    grid = torch.arange(audio_frames, dtype=torch.float32, device=device)
    factor = cfg.audio_scale_factor

    def seconds(mel: torch.Tensor) -> torch.Tensor:
        mel = (mel + cfg.causal_offset - factor).clip(min=0)
        return mel * cfg.audio_hop_length / cfg.audio_sampling_rate

    start, end = seconds(grid * factor), seconds((grid + 1) * factor)
    return ((start + end) / 2.0).view(1, -1, 1)


def build_rope(
    cfg: LTX2TransformerConfig, frames: int, height: int, width: int, fps: float, audio_frames: int, device,
) -> LTX2Rope:
    """Every rotary table of one request shape (latent frames x rows x columns)."""
    video = video_positions(cfg, frames, height, width, fps, device)
    audio = audio_positions(cfg, audio_frames, device)
    max_t = float(max(cfg.pos_embed_max_pos, cfg.audio_pos_embed_max_pos))
    return LTX2Rope(
        video=split_rope_table(
            video, cfg.inner_dim, cfg.num_attention_heads, cfg.rope_theta,
            (float(cfg.pos_embed_max_pos), float(cfg.base_height), float(cfg.base_width)),
        ),
        audio=split_rope_table(
            audio, cfg.audio_inner_dim, cfg.audio_num_attention_heads, cfg.rope_theta,
            (float(cfg.audio_pos_embed_max_pos),),
        ),
        # The cross-modal tables use the audio head layout and time alone. Their
        # head split follows the video stream's head count in the reference
        # (``num_attention_heads``); LTX-2.5 has 32 heads in both streams.
        cross_video=split_rope_table(
            video[..., :1], cfg.audio_cross_attention_dim, cfg.num_attention_heads, cfg.rope_theta, (max_t,),
        ),
        cross_audio=split_rope_table(
            audio, cfg.audio_cross_attention_dim, cfg.audio_num_attention_heads, cfg.rope_theta, (max_t,),
        ),
    )
