"""Cosmos3 dual-pathway Mixture-of-Transformers DiT.

Each decoder layer carries two parameter sets that run side by side:

  * UND (understanding / text-conditioning) pathway — ``to_{q,k,v,out}``,
    ``norm_{q,k}``, ``mlp``, ``input_layernorm``, ``post_attention_layernorm``.
    Causal self-attention over the text prefix; never attends to GEN tokens.
  * GEN (generation / denoiser) pathway — ``add_{q,k,v}_proj``, ``to_add_out``,
    ``norm_added_{q,k}``, ``mlp_moe_gen``, ``input_layernorm_moe_gen``,
    ``post_attention_layernorm_moe_gen``. Full (non-causal) attention where
    GEN queries attend to ``cat([k_und, k_gen])`` / ``cat([v_und, v_gen])``.

The module mirrors the published diffusers checkpoint layout one-to-one, so the
flat ``layers.N.*`` safetensors keys load with no key remapping beyond dropping
the unused text ``lm_head``.

UND and GEN run together in one fused pass every denoising step. Projections are
plain ``nn.Linear`` here; tensor-parallel variants are a later concern.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps
from torch import nn


class RMSNorm(nn.Module):
    """Weight-only RMS normalization (no bias).

    Replicates the diffusers ``RMSNorm`` dtype ordering exactly: variance in
    fp32, normalize, then round the normalized activations to the (bf16) weight
    dtype *before* the weight multiply. Matching this rounding point matters for
    tight bf16 parity across 36 layers' worth of norms.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
            return hidden_states * self.weight
        return (hidden_states * self.weight).to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Cosmos3RotaryEmbedding(nn.Module):
    """3D interleaved mRoPE (``Cosmos3VLTextRotaryEmbedding``).

    ``inv_freq`` is recomputed on the fly from ``rope_theta``/``head_dim`` rather
    than registered as a buffer: the model is materialized via ``meta`` +
    ``to_empty``, which leaves registered buffers uninitialized. Recompute is
    cheap (``head_dim/2`` values, once per forward).
    """

    def __init__(self, head_dim: int, rope_theta: float, rope_axes_dim: tuple[int, int, int]):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rope_axes_dim = tuple(rope_axes_dim)

    def apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """Reorganize chunked ``[TTT…HHH…WWW]`` frequencies into interleaved
        ``[THTHWHTHW…TT]`` (preserves frequency continuity across the 3 grids)."""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.rope_axes_dim[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)  # [3,B,N]
        inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(device)
        position_ids_expanded = position_ids[:, :, None, :].float()  # [3,B,1,N]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)  # [3,B,N,head_dim//2]
        freqs = self.apply_interleaved_mrope(freqs)  # [B,N,head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B,N,head_dim]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


class TimestepEmbedder(nn.Module):
    """Two-layer MLP over sinusoidal timestep features (``linear_1``/``linear_2``).

    Matches diffusers ``TimestepEmbedding`` (act = SiLU, no cond/post-act). Kept
    in fp32 at build time, like diffusers' ``_keep_in_fp32_modules``.
    """

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class Cosmos3MLP(nn.Module):
    """SwiGLU feed-forward (``gate_proj``/``up_proj``/``down_proj``, no bias)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Cosmos3PackedMoTAttention(nn.Module):
    """Dual-pathway packed attention: separate unfused projections + QK-norm for
    the understanding (causal) and generation (full) token streams.

    Mirrors diffusers ``Cosmos3AttnProcessor``: QK-norm is applied per-head
    *before* RoPE; the UND stream self-attends causally, the GEN stream attends
    non-causally to ``cat([und, gen])``. GQA (32 Q / 8 KV heads) is handled by
    ``F.scaled_dot_product_attention(enable_gqa=True)``.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        q_dim = num_attention_heads * head_dim
        kv_dim = num_key_value_heads * head_dim

        # Understanding pathway.
        self.to_q = nn.Linear(hidden_size, q_dim, bias=attention_bias)
        self.to_k = nn.Linear(hidden_size, kv_dim, bias=attention_bias)
        self.to_v = nn.Linear(hidden_size, kv_dim, bias=attention_bias)
        self.to_out = nn.Linear(q_dim, hidden_size, bias=attention_bias)
        self.norm_q = RMSNorm(head_dim, eps=rms_norm_eps)
        self.norm_k = RMSNorm(head_dim, eps=rms_norm_eps)

        # Generation pathway.
        self.add_q_proj = nn.Linear(hidden_size, q_dim, bias=attention_bias)
        self.add_k_proj = nn.Linear(hidden_size, kv_dim, bias=attention_bias)
        self.add_v_proj = nn.Linear(hidden_size, kv_dim, bias=attention_bias)
        self.to_add_out = nn.Linear(q_dim, hidden_size, bias=attention_bias)
        self.norm_added_q = RMSNorm(head_dim, eps=rms_norm_eps)
        self.norm_added_k = RMSNorm(head_dim, eps=rms_norm_eps)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [N, H, D]; cos/sin: [N, D] -> [N, 1, D] for broadcast over heads.
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return x * cos + _rotate_half(x) * sin

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
        # q: [Nq, Hq, D]; k/v: [Nk, Hkv, D] -> [Nq, Hq*D]. SDPA wants [B, H, S, D].
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=True)
        return out.transpose(1, 2).squeeze(0).flatten(-2, -1)

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim

        q_und = self.to_q(und_seq).view(-1, H, D)
        k_und = self.to_k(und_seq).view(-1, Hkv, D)
        v_und = self.to_v(und_seq).view(-1, Hkv, D)
        q_gen = self.add_q_proj(gen_seq).view(-1, H, D)
        k_gen = self.add_k_proj(gen_seq).view(-1, Hkv, D)
        v_gen = self.add_v_proj(gen_seq).view(-1, Hkv, D)

        q_und = self.norm_q(q_und)
        k_und = self.norm_k(k_und)
        q_gen = self.norm_added_q(q_gen)
        k_gen = self.norm_added_k(k_gen)

        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        q_und = self._apply_rope(q_und, cos_und, sin_und)
        k_und = self._apply_rope(k_und, cos_und, sin_und)
        q_gen = self._apply_rope(q_gen, cos_gen, sin_gen)
        k_gen = self._apply_rope(k_gen, cos_gen, sin_gen)

        # UND: causal self-attention over text.
        causal_out = self._attend(q_und, k_und, v_und, is_causal=True)
        # GEN: full attention over [und | gen].
        all_k = torch.cat([k_und, k_gen], dim=0)
        all_v = torch.cat([v_und, v_gen], dim=0)
        full_out = self._attend(q_gen, all_k, all_v, is_causal=False)

        return self.to_out(causal_out), self.to_add_out(full_out)

    # ------------------------------------------------------------------
    # Cached-attention variants: the two pathways run in separate passes and
    # share their K/V through a paged cache handle instead of in-pass concat.
    # The understanding pass writes its K/V (causal); the generation pass reads
    # that frozen K/V plus its own (non-causal) — causality is fixed by the
    # handle's attention plan, not here.
    # ------------------------------------------------------------------

    def forward_und(self, und_seq: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache_handle) -> torch.Tensor:
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        q = self.norm_q(self.to_q(und_seq).view(-1, H, D))
        k = self.norm_k(self.to_k(und_seq).view(-1, Hkv, D))
        v = self.to_v(und_seq).view(-1, Hkv, D)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        out = cache_handle.run_attention(q=q, k=k, v=v).reshape(-1, H * D)
        return self.to_out(out)

    def forward_gen(self, gen_seq: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache_handle) -> torch.Tensor:
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        q = self.norm_added_q(self.add_q_proj(gen_seq).view(-1, H, D))
        k = self.norm_added_k(self.add_k_proj(gen_seq).view(-1, Hkv, D))
        v = self.add_v_proj(gen_seq).view(-1, Hkv, D)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        out = cache_handle.run_attention(q=q, k=k, v=v).reshape(-1, H * D)
        return self.to_add_out(out)


class Cosmos3MoTDecoderLayer(nn.Module):
    """One dual-pathway decoder layer (UND + GEN parameter sets)."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        attention_bias: bool,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.self_attn = Cosmos3PackedMoTAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = Cosmos3MLP(hidden_size, intermediate_size)
        self.mlp_moe_gen = Cosmos3MLP(hidden_size, intermediate_size)

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_layernorm_moe_gen = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm_moe_gen = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        und_norm = self.input_layernorm(und_seq)
        gen_norm = self.input_layernorm_moe_gen(gen_seq)

        und_attn_out, gen_attn_out = self.self_attn(und_norm, gen_norm, rotary_emb)
        residual_und = und_seq + und_attn_out
        residual_gen = gen_seq + gen_attn_out

        mlp_out_und = self.mlp(self.post_attention_layernorm(residual_und))
        mlp_out_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))

        return residual_und + mlp_out_und, residual_gen + mlp_out_gen

    def forward_und(self, und_seq: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache_handle) -> torch.Tensor:
        und_norm = self.input_layernorm(und_seq)
        attn_out = self.self_attn.forward_und(und_norm, cos, sin, cache_handle)
        residual = und_seq + attn_out
        return residual + self.mlp(self.post_attention_layernorm(residual))

    def forward_gen(self, gen_seq: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache_handle) -> torch.Tensor:
        gen_norm = self.input_layernorm_moe_gen(gen_seq)
        attn_out = self.self_attn.forward_gen(gen_norm, cos, sin, cache_handle)
        residual = gen_seq + attn_out
        return residual + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual))


class DomainAwareLinear(nn.Module):
    """Per-embodiment affine map: one *full* (weight, bias) pair per action
    embodiment domain, both looked up from embedding tables keyed by a domain id.

    ``fc`` holds each domain's flattened weight (shape ``[num_domains,
    out*in]``, viewed as ``[in, out]`` so the map is ``x @ W`` — note the
    weight is stored transposed relative to ``nn.Linear``); ``bias`` holds each
    domain's ``[out]`` bias. Matches the checkpoint's
    ``action_proj_{in,out}.{fc,bias}.weight`` shapes one-to-one."""

    def __init__(self, in_features: int, out_features: int, num_domains: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_domains = num_domains
        self.fc = nn.Embedding(num_domains, out_features * in_features)
        self.bias = nn.Embedding(num_domains, out_features)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        domain_id = domain_id.to(device=x.device, dtype=torch.long).reshape(-1)
        weight = self.fc(domain_id).view(domain_id.shape[0], self.in_features, self.out_features)
        bias = self.bias(domain_id).view(domain_id.shape[0], self.out_features)
        if x.ndim == 2:  # [B, in] -> [B, out]
            return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        return torch.bmm(x, weight) + bias.unsqueeze(1)  # [B, T, in] -> [B, T, out]


class Cosmos3OmniTransformer(nn.Module):
    """The full Cosmos3 generator backbone.

    ``state_dict()`` keys reproduce the published ``transformer/`` checkpoint
    exactly, except the text ``lm_head`` is intentionally absent: generation
    predicts flow velocity through ``proj_out`` and never decodes text logits.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        h = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, h)
        self.layers = nn.ModuleList(
            Cosmos3MoTDecoderLayer(
                hidden_size=h,
                head_dim=config.head_dim,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                intermediate_size=config.intermediate_size,
                attention_bias=config.attention_bias,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(h, eps=config.rms_norm_eps)
        self.norm_moe_gen = RMSNorm(h, eps=config.rms_norm_eps)
        self.rotary_emb = Cosmos3RotaryEmbedding(
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_axes_dim=config.rope_axes_dim,
        )

        # Vision latent in/out projections + timestep embedder.
        self.proj_in = nn.Linear(config.patch_latent_dim, h, bias=True)
        self.proj_out = nn.Linear(h, config.patch_latent_dim, bias=True)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedder(in_channels=256, time_embed_dim=h)

        # Sound (AVAE-latent) heads.
        if config.sound_gen:
            if config.sound_dim is None:
                raise ValueError("sound_dim must be set when sound_gen is True")
            self.audio_proj_in = nn.Linear(config.sound_dim, h, bias=True)
            self.audio_proj_out = nn.Linear(h, config.sound_dim, bias=True)
            self.audio_modality_embed = nn.Parameter(torch.zeros(h))

        # Action heads (per-embodiment domain-aware projections).
        if config.action_gen:
            self.action_proj_in = DomainAwareLinear(
                config.max_action_dim, h, config.num_embodiment_domains
            )
            self.action_proj_out = DomainAwareLinear(
                h, config.max_action_dim, config.num_embodiment_domains
            )
            self.action_modality_embed = nn.Parameter(torch.zeros(h))

    # ------------------------------------------------------------------
    # Pure-tensor packing/unpacking helpers (ported from diffusers).
    # ------------------------------------------------------------------

    def _apply_timestep_embeds_to_noisy_tokens(
        self,
        packed_tokens: torch.Tensor,
        packed_timestep_embeds: torch.Tensor,
        noisy_frame_indexes: list[torch.Tensor],
        token_shapes: list[tuple[int, ...]],
    ) -> torch.Tensor:
        start_noisy_index = 0
        flattened_noisy_frame_indexes: list[torch.Tensor] = []
        for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes, strict=True):
            spatial_numel_i = math.prod(token_shape_i[1:])
            spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)
            frame_offsets = (noisy_indexes_i * spatial_numel_i).unsqueeze(-1) + spatial_indexes_i + start_noisy_index
            flattened_noisy_frame_indexes.append(frame_offsets.flatten())
            start_noisy_index += token_shape_i[0] * spatial_numel_i
        flattened = torch.cat(flattened_noisy_frame_indexes, dim=0).unsqueeze(-1).expand(-1, packed_tokens.shape[1])
        return packed_tokens.scatter_add(dim=0, index=flattened, src=packed_timestep_embeds)

    def _patchify_and_pack_latents(
        self, tokens_vision: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        p = self.config.latent_patch_size
        latent_channel = self.config.latent_channel
        packed_latent: list[torch.Tensor] = []
        original_latent_shapes: list[tuple[int, int, int]] = []
        for latent in tokens_vision:
            latent = latent.squeeze(0)  # [C, T, H, W]
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros(
                    (latent_channel, t_actual, h_padded, w_padded), device=latent.device, dtype=latent.dtype
                )
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded
            h_patches = h_padded // p
            w_patches = w_padded // p
            latent = latent.reshape(latent_channel, t_actual, h_patches, p, w_patches, p)
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * latent_channel)
            packed_latent.append(latent)
        return torch.cat(packed_latent, dim=0), original_latent_shapes

    def _unpatchify_and_unpack_latents(
        self,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: list[tuple[int, int, int]],
        noisy_frame_indexes_vision: list[torch.Tensor],
        original_latent_shapes: list[tuple[int, int, int]],
    ) -> list[torch.Tensor]:
        p = self.config.latent_patch_size
        latent_channel = self.config.latent_channel
        unpatchified_latents: list[torch.Tensor] = []
        start_idx = 0
        for token_shape, noisy_frame_indexes, original_shape in zip(
            token_shapes_vision, noisy_frame_indexes_vision, original_latent_shapes, strict=True
        ):
            t_c = token_shape[0]
            _, h_orig, w_orig = original_shape
            h_padded = ((h_orig + p - 1) // p) * p
            w_padded = ((w_orig + p - 1) // p) * p
            h_patches = h_padded // p
            w_patches = w_padded // p
            t_n = len(noisy_frame_indexes)
            output_tensor = torch.zeros(
                (latent_channel, t_c, h_orig, w_orig), device=packed_mse_preds.device, dtype=packed_mse_preds.dtype
            )
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                latent_patches = packed_mse_preds[start_idx:end_idx]
                latent_patches = latent_patches.reshape(t_n, h_patches, w_patches, p, p, latent_channel)
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)
                latent = latent.reshape(latent_channel, t_n, h_patches * p, w_patches * p)
                latent = latent[:, :, :h_orig, :w_orig]
                output_tensor[:, noisy_frame_indexes] = latent
                start_idx = end_idx
            unpatchified_latents.append(output_tensor.unsqueeze(0))
        return unpatchified_latents

    def _pack_sound_latents(
        self, tokens_sound: list[torch.Tensor], token_shapes_sound: list[tuple[int, int, int]]
    ) -> torch.Tensor:
        return torch.cat(
            [sound[:, : shape[0]].permute(1, 0) for sound, shape in zip(tokens_sound, token_shapes_sound, strict=True)],
            dim=0,
        )

    def _unpack_sound_latents(
        self,
        packed_preds: torch.Tensor,
        token_shapes_sound: list[tuple[int, int, int]],
        noisy_frame_indexes_sound: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        sound_dim = self.config.sound_dim
        unpacked: list[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_idxs in zip(token_shapes_sound, noisy_frame_indexes_sound, strict=True):
            T = shape[0]
            output = torch.zeros((sound_dim, T), device=packed_preds.device, dtype=packed_preds.dtype)
            t_n = len(noisy_idxs)
            if t_n > 0:
                output[:, noisy_idxs] = packed_preds[start_idx : start_idx + t_n].T
                start_idx += t_n
            unpacked.append(output)
        return unpacked

    # ------------------------------------------------------------------
    # forward: full per-step pass — encode text/vision, run layers, decode velocity.
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        position_ids: torch.Tensor,
        und_len: int,
        sequence_length: int,
        vision_tokens: list[torch.Tensor],
        vision_token_shapes: list[tuple[int, int, int]],
        vision_sequence_indexes: torch.Tensor,
        vision_mse_loss_indexes: torch.Tensor,
        vision_timesteps: torch.Tensor,
        vision_noisy_frame_indexes: list[torch.Tensor],
        sound_tokens: list[torch.Tensor] | None = None,
        sound_token_shapes: list[tuple[int, int, int]] | None = None,
        sound_sequence_indexes: torch.Tensor | None = None,
        sound_mse_loss_indexes: torch.Tensor | None = None,
        sound_timesteps: torch.Tensor | None = None,
        sound_noisy_frame_indexes: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        has_sound = sound_tokens is not None and sound_sequence_indexes is not None

        # Embed text into the joint hidden_states buffer at its sequence positions.
        packed_text_embedding = self.embed_tokens(input_ids)
        target_dtype = packed_text_embedding.dtype
        hidden_states = packed_text_embedding.new_zeros(size=(sequence_length, self.config.hidden_size))
        hidden_states[text_indexes] = packed_text_embedding

        # Patchify + project vision latents, then scatter-add timestep embeds to noisy frames.
        packed_tokens_vision, original_latent_shapes = self._patchify_and_pack_latents(vision_tokens)
        packed_tokens_vision = self.proj_in(packed_tokens_vision)
        timesteps_vision = vision_timesteps * self.config.timestep_scale
        packed_timestep_embeds_vision = self.time_embedder(self.time_proj(timesteps_vision)).to(target_dtype)
        packed_tokens_vision = self._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed_tokens_vision,
            packed_timestep_embeds=packed_timestep_embeds_vision,
            noisy_frame_indexes=vision_noisy_frame_indexes,
            token_shapes=vision_token_shapes,
        )
        hidden_states[vision_sequence_indexes] = packed_tokens_vision

        # Pack + project sound latents (all sound frames noisy).
        if has_sound:
            packed_tokens_sound = self._pack_sound_latents(sound_tokens, sound_token_shapes).to(target_dtype)
            packed_tokens_sound = self.audio_proj_in(packed_tokens_sound) + self.audio_modality_embed
            timesteps_sound = sound_timesteps * self.config.timestep_scale
            packed_timestep_embeds_sound = self.time_embedder(self.time_proj(timesteps_sound)).to(target_dtype)
            packed_tokens_sound = self._apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_sound,
                packed_timestep_embeds=packed_timestep_embeds_sound,
                noisy_frame_indexes=sound_noisy_frame_indexes,
                token_shapes=sound_token_shapes,
            )
            hidden_states[sound_sequence_indexes] = packed_tokens_sound

        # mRoPE once for the joint sequence, then slice into und/gen halves.
        cos, sin = self.rotary_emb(
            position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)

        und_seq = hidden_states[:und_len]
        gen_seq = hidden_states[und_len:]
        rotary_emb = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])
        for decoder_layer in self.layers:
            und_seq, gen_seq = decoder_layer(und_seq, gen_seq, rotary_emb)
        und_out = self.norm(und_seq)
        gen_out = self.norm_moe_gen(gen_seq)
        last_hidden_state = torch.cat([und_out, gen_out], dim=0)

        # Decode vision velocity from the joint hidden state.
        preds_vision_packed = self.proj_out(last_hidden_state[vision_mse_loss_indexes])
        preds_vision = self._unpatchify_and_unpack_latents(
            preds_vision_packed,
            token_shapes_vision=vision_token_shapes,
            noisy_frame_indexes_vision=vision_noisy_frame_indexes,
            original_latent_shapes=original_latent_shapes,
        )

        preds_sound: list[torch.Tensor] | None = None
        if has_sound:
            preds_sound_packed = self.audio_proj_out(last_hidden_state[sound_mse_loss_indexes])
            preds_sound = self._unpack_sound_latents(preds_sound_packed, sound_token_shapes, sound_noisy_frame_indexes)

        return preds_vision, preds_sound

    # ------------------------------------------------------------------
    # Cache-once engine path: the understanding tower runs once and writes its
    # K/V; the generation tower then runs per denoising step, re-reading that
    # frozen K/V. Because the text tokens never receive a timestep embedding,
    # their K/V is step-independent, so caching it once is exact. ``cache_handle``
    # is a paged attention handle (set_layer_idx / run_attention / advance_seq_lens);
    # the attention plan (causal vs not, which label) is configured by the caller.
    # ------------------------------------------------------------------

    def _rotary(self, position_ids: torch.Tensor, device, dtype):
        """cos/sin of shape [N, head_dim] for a [3, N] block of 3D mRoPE ids."""
        cos, sin = self.rotary_emb(position_ids.unsqueeze(1), device=device, dtype=dtype)
        return cos.squeeze(0), sin.squeeze(0)

    def prefill_und(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, cache_handle
    ) -> None:
        """Run the understanding tower over the text prefix, writing per-layer K/V
        to the cache under the active label and committing the prefix length.
        ``position_ids`` are the text segment's 3D mRoPE ids ([3, und_len])."""
        und_seq = self.embed_tokens(input_ids)
        cos, sin = self._rotary(position_ids, und_seq.device, und_seq.dtype)
        for i, layer in enumerate(self.layers):
            cache_handle.set_layer_idx(i)
            und_seq = layer.forward_und(und_seq, cos, sin, cache_handle)
        cache_handle.advance_seq_lens()

    def denoise_step(
        self,
        latents: torch.Tensor,
        vision_timesteps: torch.Tensor,
        position_ids: torch.Tensor,
        vision_token_shapes: list[tuple[int, int, int]],
        vision_noisy_frame_indexes: list[torch.Tensor],
        vision_mse_loss_indexes: torch.Tensor,
        cache_handle,
    ) -> torch.Tensor:
        """One generation-tower evaluation against the frozen understanding K/V.

        Patchifies ``latents`` ([1, C, T, H, W]), scatter-adds the timestep
        embedding to the noisy tokens, runs the generation layers (each reading
        the active label's cached understanding K/V plus its own freshly written
        K/V), and decodes the flow velocity. ``position_ids`` are the vision
        segment's 3D mRoPE ids ([3, num_vision]); ``vision_mse_loss_indexes`` are
        gen-relative (into the vision token block). Returns the velocity latent
        ([1, C, T, H, W])."""
        packed, original_latent_shapes = self._patchify_and_pack_latents([latents])
        packed = self.proj_in(packed)
        target_dtype = packed.dtype
        timesteps = vision_timesteps * self.config.timestep_scale
        ts_embeds = self.time_embedder(self.time_proj(timesteps)).to(target_dtype)
        gen_seq = self._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed,
            packed_timestep_embeds=ts_embeds,
            noisy_frame_indexes=vision_noisy_frame_indexes,
            token_shapes=vision_token_shapes,
        )
        cos, sin = self._rotary(position_ids, gen_seq.device, gen_seq.dtype)
        for i, layer in enumerate(self.layers):
            cache_handle.set_layer_idx(i)
            gen_seq = layer.forward_gen(gen_seq, cos, sin, cache_handle)
        gen_out = self.norm_moe_gen(gen_seq)
        preds_packed = self.proj_out(gen_out[vision_mse_loss_indexes])
        preds = self._unpatchify_and_unpack_latents(
            preds_packed,
            token_shapes_vision=vision_token_shapes,
            noisy_frame_indexes_vision=vision_noisy_frame_indexes,
            original_latent_shapes=original_latent_shapes,
        )
        return preds[0]
