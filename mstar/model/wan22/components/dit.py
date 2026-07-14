"""Native Wan2.2-TI2V-5B video DiT (dense 5B transformer).

Exact port of diffusers 0.39.0 ``WanTransformer3DModel``, restricted to the
TI2V-5B checkpoint's configuration: no image-conditioning branch, and the
per-token timestep path (the dit submodule always feeds a ``[batch, seq]``
timestep grid).

Numerics contract (bf16 weights with fp32 islands):

  * fp32 islands, matching the reference's ``_keep_in_fp32_modules``: the
    time embedder's two Linears, every ``scale_shift_table``, and each block's
    ``norm2`` affine. ``cast_serving_dtypes()`` pins them; the rest is bf16.
  * The fp32 -> bf16 boundaries are the reference's ``type_as`` calls, and they
    must stay where they are: the timestep embedding and modulation run fp32
    and cast back per block, RoPE is applied fp32, the self-attention and FFN
    residual adds upcast to fp32, and the cross-attention residual add stays
    bf16.
  * The RoPE cos/sin tables are DERIVED state, not checkpoint state. Non-
    persistent buffers hold garbage after ``to_empty``, so they are built
    lazily per device (on CPU first, for device-independent values) rather
    than registered as buffers.

Two shared components are deliberately not reused. ``components.norm.RMSNorm``
dispatches to a FlashInfer kernel that rejects sm_120; the reference op is a
plain ``torch.nn.RMSNorm`` across heads. ``components.attention.Attention`` is
built around a KV-cache handle and a token-flat layout, and a stateless
bidirectional video DiT with 3D RoPE would have to override all of it. The
two-linear MLPs ARE the shared ``components.mlp.MLP``.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.mlp import MLP
from mstar.model.wan22.config import Wan22Config

# 3D RoPE base frequency (reference WanRotaryPosEmbed default `theta`).
ROPE_THETA = 10000.0


class WanFP32LayerNorm(nn.LayerNorm):
    """LayerNorm computed in fp32 regardless of input dtype (diffusers
    ``FP32LayerNorm``): upcast input, normalize with fp32 weights, cast back
    to the input dtype."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(inputs.dtype)


def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """diffusers ``get_timestep_embedding`` with the Wan condition embedder's
    fixed arguments (``flip_sin_to_cos=True``, ``downscale_freq_shift=0``,
    ``scale=1``, ``max_period=10000``; even ``embedding_dim``). fp32 in,
    fp32 out ``[N, embedding_dim]``."""
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
    emb = torch.exp(exponent / half_dim)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)


class Wan22RoPE3D:
    """3D-factorized rotary tables for the post-patchify (t, h, w) grid.

    The head dim splits into three axis bands — ``h = w = 2 * (head_dim //
    6)``, ``t`` takes the remainder (44/42/42 for head_dim 128) — each with
    its own 1D table over ``max_seq_len`` positions, computed in float64 and
    stored fp32 concatenated to ``[max_seq_len, head_dim]`` (reference
    ``WanRotaryPosEmbed.__init__``). Tables are built on CPU on first use
    (bit-identical regardless of the eventual device) and cached per device.

    Deliberately not an ``nn.Module``: the tables are derived state that must
    never ride through ``to_empty``/``state_dict``, and keeping them out of
    the module tree means no buffer for the loader or ``.to()`` to corrupt.
    """

    def __init__(self, attention_head_dim: int, max_seq_len: int):
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        self.axis_dims = (attention_head_dim - h_dim - w_dim, h_dim, w_dim)
        self.max_seq_len = max_seq_len
        self._tables: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def _build_cpu_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        cos_parts, sin_parts = [], []
        for dim in self.axis_dims:
            freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
            freqs = torch.outer(torch.arange(self.max_seq_len), freqs)
            cos_parts.append(freqs.cos().repeat_interleave(2, dim=1, output_size=dim).float())
            sin_parts.append(freqs.sin().repeat_interleave(2, dim=1, output_size=dim).float())
        return torch.cat(cos_parts, dim=1), torch.cat(sin_parts, dim=1)

    def tables(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device not in self._tables:
            if not self._tables:
                self._tables[torch.device("cpu")] = self._build_cpu_tables()
            cpu_cos, cpu_sin = self._tables[torch.device("cpu")]
            self._tables[device] = (cpu_cos.to(device), cpu_sin.to(device))
        return self._tables[device]

    def __call__(self, grid: tuple[int, int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-grid frequencies, each ``[1, t*h*w, 1, head_dim]`` fp32
        (reference ``WanRotaryPosEmbed.forward``)."""
        ppf, pph, ppw = grid
        freqs_cos, freqs_sin = self.tables(device)
        split_cos = freqs_cos.split(list(self.axis_dims), dim=1)
        split_sin = freqs_sin.split(list(self.axis_dims), dim=1)

        def expand_axes(split: tuple[torch.Tensor, ...]) -> torch.Tensor:
            f = split[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
            h = split[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
            w = split[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
            return torch.cat([f, h, w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return expand_axes(split_cos), expand_axes(split_sin)


def _apply_rotary_emb(
    hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved-pair rotary application (reference ``WanAttnProcessor``
    inner ``apply_rotary_emb``): fp32 math via type promotion against the
    fp32 tables, cast back to the input dtype at the output-buffer
    assignment."""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class Wan22DiTAttention(nn.Module):
    """Multi-head attention with across-heads RMSNorm on q/k (the
    checkpoint's ``rms_norm_across_heads``: one 3072-wide norm before the
    head split, not per-head) and SDPA. Self-attention applies 3D RoPE;
    cross-attention reads k/v from the 512-token text stream and skips RoPE.
    All four projections carry bias (reference ``WanAttention``)."""

    def __init__(self, dim: int, num_heads: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.k_norm = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        kv_source = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = self.q_norm(self.q_proj(hidden_states))
        key = self.k_norm(self.k_proj(kv_source))
        value = self.v_proj(kv_source)

        query = query.unflatten(2, (self.num_heads, -1))
        key = key.unflatten(2, (self.num_heads, -1))
        value = value.unflatten(2, (self.num_heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        # [B, S, H, D] -> [B, H, S, D] SDPA -> back (the reference's
        # dispatch_attention_fn native path; bidirectional, no mask).
        out = F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
        ).permute(0, 2, 1, 3)
        return self.o_proj(out.flatten(2, 3))


class Wan22TimeTextEmbedding(nn.Module):
    """Condition embedder: per-token timestep embedding + text projection
    (reference ``WanTimeTextImageEmbedding``, image branch absent on
    TI2V-5B).

    ``time_embedder`` is an fp32 island: the sinusoidal embedding and both
    its Linears run fp32; ``temb`` crosses to bf16 at ``type_as(text)``, so
    the downstream ``time_proj`` modulation projection is a bf16 matmul.
    """

    def __init__(self, dim: int, freq_dim: int, text_dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.time_embedder = MLP(freq_dim, dim, dim, activation="silu", bias=True)
        self.time_proj = nn.Linear(dim, dim * 6)
        self.text_embedder = MLP(text_dim, dim, dim, activation="gelu_tanh", bias=True)

    def forward(
        self, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``timestep`` ``[B, seq]`` (per-token, ``expand_timesteps``);
        returns ``temb [B, seq, dim]``, ``timestep_proj [B, seq, 6, dim]``,
        projected text ``[B, 512, dim]`` — all in the text embeds' dtype."""
        batch, seq_len = timestep.shape
        sincos = _sinusoidal_timestep_embedding(timestep.flatten(), self.freq_dim)
        temb = self.time_embedder(sincos.unflatten(0, (batch, seq_len)))
        temb = temb.type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(F.silu(temb)).unflatten(2, (6, -1))
        return temb, timestep_proj, self.text_embedder(encoder_hidden_states)


class Wan22DiTBlock(nn.Module):
    """One transformer block: adaLN-modulated self-attention, cross-attention
    to the text stream, adaLN-modulated FFN (reference ``WanTransformerBlock``,
    per-token ``temb.ndim == 4`` branch).

    The modulation runs fp32 (the scale-shift table is an fp32 island), the
    self-attention and FFN residual adds upcast to fp32, and the cross-attention
    residual stays bf16. All three follow the reference exactly.
    """

    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float):
        super().__init__()
        self.norm1 = WanFP32LayerNorm(dim, eps, elementwise_affine=False)
        self.self_attn = Wan22DiTAttention(dim, num_heads, eps)
        self.cross_attn = Wan22DiTAttention(dim, num_heads, eps)
        self.norm2 = WanFP32LayerNorm(dim, eps, elementwise_affine=True)
        self.ffn = MLP(dim, ffn_dim, dim, activation="gelu_tanh", bias=True)
        self.norm3 = WanFP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.empty(1, 6, dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # [1, 1, 6, dim] + [B, seq, 6, dim] -> six [B, seq, dim] fp32 tensors.
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            (self.scale_shift_table.unsqueeze(0) + timestep_proj.float()).chunk(6, dim=2)
        )
        shift_msa, scale_msa, gate_msa = shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2)
        c_shift_msa, c_scale_msa, c_gate_msa = c_shift_msa.squeeze(2), c_scale_msa.squeeze(2), c_gate_msa.squeeze(2)

        norm_hidden = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.self_attn(norm_hidden, rotary_emb=rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        norm_hidden = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.cross_attn(norm_hidden, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        norm_hidden = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden)
        return (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)


class Wan22DiT(nn.Module):
    """The dense 5B TI2V video DiT: Conv3d patchify -> 30 blocks -> adaLN
    output head + unpatchify. Built on the meta device and materialized by
    ``weight_loader.build_wan22_dit`` (meta -> ``cast_serving_dtypes`` ->
    ``to_empty(device)`` -> checkpoint load)."""

    def __init__(self, config: Wan22Config):
        super().__init__()
        if config.qk_norm != "rms_norm_across_heads" or not config.cross_attn_norm:
            # These two config facts are baked into Wan22DiTAttention /
            # Wan22DiTBlock (across-heads q/k RMSNorm, affine norm2 before
            # cross-attention); a checkpoint that drifts cannot be honored
            # (same hard-fail stance as _refresh_checkpoint_defaults).
            raise ValueError(
                f"Wan22DiT implements qk_norm='rms_norm_across_heads' with cross_attn_norm=True; "
                f"got qk_norm={config.qk_norm!r}, cross_attn_norm={config.cross_attn_norm!r}."
            )
        self.config = config
        dim = config.hidden_size
        self.rope = Wan22RoPE3D(config.attention_head_dim, config.rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            config.in_channels, dim, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.condition_embedder = Wan22TimeTextEmbedding(dim, config.freq_dim, config.text_dim)
        self.blocks = nn.ModuleList(
            Wan22DiTBlock(dim, config.ffn_dim, config.num_attention_heads, config.eps)
            for _ in range(config.num_layers)
        )
        self.norm_out = WanFP32LayerNorm(dim, config.eps, elementwise_affine=False)
        self.proj_out = nn.Linear(dim, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(torch.empty(1, 2, dim))

    @property
    def dtype(self) -> torch.dtype:
        """Bulk compute dtype (the non-island weights); callers cast inputs
        to this, mirroring diffusers ``ModelMixin.dtype``."""
        return self.patch_embedding.weight.dtype

    def cast_serving_dtypes(self) -> "Wan22DiT":
        """bf16 everywhere except the fp32 islands (the checkpoint's
        ``_keep_in_fp32_modules``: time_embedder, scale_shift_tables, block
        norm2 affines). Called on the meta module BEFORE ``to_empty`` so
        storage is allocated directly in the serving dtypes."""
        self.to(torch.bfloat16)
        self.condition_embedder.time_embedder.to(torch.float32)
        self.scale_shift_table.data = self.scale_shift_table.data.to(torch.float32)
        for block in self.blocks:
            block.norm2.to(torch.float32)
            block.scale_shift_table.data = block.scale_shift_table.data.to(torch.float32)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """``hidden_states`` ``[B, C, F, H, W]`` (bf16 latents), ``timestep``
        ``[B, post-patch seq]`` per-token grid (``expand_timesteps`` — the
        only timestep form mstar's dit submodule produces), text embeds
        ``[B, 512, text_dim]`` bf16. Returns ``[B, C_out, F, H, W]`` bf16."""
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        grid = (num_frames // p_t, height // p_h, width // p_w)
        if timestep.ndim != 2:
            raise ValueError(
                f"Wan22DiT requires a per-token [batch, seq] timestep grid; got ndim={timestep.ndim}."
            )

        rotary_emb = self.rope(grid, hidden_states.device)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states
        )

        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # Output head: 2-chunk adaLN in fp32 ([1, 1, 2, dim] island table +
        # bf16 temb promotes), then project and unpatchify.
        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2), scale.squeeze(2)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, *grid, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
