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

UND and GEN run together in one fused pass every denoising step. The attention
and MLP projections are tensor-parallel: with a trivial (world-size-1) comm
group they behave exactly like plain ``nn.Linear``; with a real group the
q/k/v and gate/up projections are column-sharded along the head / intermediate
dim and the out / down projections row-shard their input and all-reduce.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.sequence_parallel import (
    gather_sequence,
    scatter_sequence,
    sp_head_gather,
    sp_head_slice,
    sp_seq_split,
    ulysses_attention,
)
from mstar.model.components.distributed.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)


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
        # Outer product position ⊗ inv_freq via broadcast multiply. The original
        # form built stride-0 broadcast views and ran a batched matmul whose
        # output the CUDA-graph memory pool can mis-capture at some sequence
        # lengths (the rotary table comes out wrong on replay, scrambling the
        # image). A plain broadcast multiply produces a fresh contiguous tensor
        # and is capture-faithful — bit-identical eagerly.
        freqs = position_ids[:, :, :, None].float() * inv_freq.view(1, 1, 1, -1)  # [3,B,N,head_dim//2]
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
    """SwiGLU feed-forward (``gate_proj``/``up_proj``/``down_proj``, no bias).

    Tensor-parallel: ``gate_proj``/``up_proj`` are column-sharded along the
    intermediate dim and ``down_proj`` row-shards its input and all-reduces.
    A trivial comm group (world size 1) makes these plain linears.

    Deliberately not ``ParallelGatedMLP``: that class fuses gate/up into one
    ``MergedColumnParallelLinear`` and needs stacked-parameter loader rules,
    while the Cosmos3 backbone keeps every projection unfused (attention
    included) so ``state_dict()`` keys match the published checkpoint
    one-to-one and the loader stays a plain name-matching stream.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        comm_group: CommGroup | None = None,
    ):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.gate_proj = ColumnParallelLinear(comm_group, hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(comm_group, hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(comm_group, intermediate_size, hidden_size, bias=False)
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
        comm_group: CommGroup | None = None,
        sp_group: CommGroup | None = None,
    ):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        if sp_group is None:
            sp_group = CommGroup.trivial()
        self.sp_group = sp_group
        tp_size = comm_group.world_size
        sp_size = sp_group.world_size
        # Ulysses sequence parallelism redistributes heads across the SP group
        # via an all-to-all around attention, so the attention runs at effective
        # head-degree tp*sp — both head counts must divide that product.
        eff_size = tp_size * sp_size
        if num_attention_heads % eff_size or num_key_value_heads % eff_size:
            raise ValueError(
                f"tp_size*sp_size ({tp_size}*{sp_size}={eff_size}) must divide "
                f"both num_attention_heads ({num_attention_heads}) and "
                f"num_key_value_heads ({num_key_value_heads})"
            )
        self.head_dim = head_dim
        # TP-local (post-projection) head counts: the column-parallel q/k/v
        # projections shard heads by tp_size, and the SP all-to-all further
        # splits these to tp*sp heads around the attention kernel.
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_key_value_heads = num_key_value_heads // tp_size

        q_dim = num_attention_heads * head_dim
        kv_dim = num_key_value_heads * head_dim

        # Understanding pathway.
        self.to_q = ColumnParallelLinear(comm_group, hidden_size, q_dim, bias=attention_bias)
        self.to_k = ColumnParallelLinear(comm_group, hidden_size, kv_dim, bias=attention_bias)
        self.to_v = ColumnParallelLinear(comm_group, hidden_size, kv_dim, bias=attention_bias)
        self.to_out = RowParallelLinear(comm_group, q_dim, hidden_size, bias=attention_bias)
        self.norm_q = RMSNorm(head_dim, eps=rms_norm_eps)
        self.norm_k = RMSNorm(head_dim, eps=rms_norm_eps)

        # Generation pathway.
        self.add_q_proj = ColumnParallelLinear(comm_group, hidden_size, q_dim, bias=attention_bias)
        self.add_k_proj = ColumnParallelLinear(comm_group, hidden_size, kv_dim, bias=attention_bias)
        self.add_v_proj = ColumnParallelLinear(comm_group, hidden_size, kv_dim, bias=attention_bias)
        self.to_add_out = RowParallelLinear(comm_group, q_dim, hidden_size, bias=attention_bias)
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

    # Reference fused-pass attention (both pathways in one call, in-pass K/V
    # concat): exercised only by the fused test pipeline + parity tests.
    # Production serving runs the cached variants below (forward_und /
    # forward_gen) against the paged cache.
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
        if self.sp_group.world_size > 1:
            # The UND prefix is replicated across the SP group (small text). Keep
            # this rank's head-group so the cached prefix K/V lands on the same
            # head partition the GEN all-to-all produces, then gather heads back
            # for the row-parallel output projection.
            q = sp_head_slice(self.sp_group, q)
            k = sp_head_slice(self.sp_group, k)
            v = sp_head_slice(self.sp_group, v)
            out = cache_handle.run_attention(q=q, k=k, v=v)
            out = sp_head_gather(self.sp_group, out).reshape(-1, H * D)
        else:
            out = cache_handle.run_attention(q=q, k=k, v=v).reshape(-1, H * D)
        return self.to_out(out)

    def forward_gen(
        self, gen_seq: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        cache_handle, seq_sizes: list[int] | None = None,
        prefer_all_gather: bool = False,
    ) -> torch.Tensor:
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        q = self.norm_added_q(self.add_q_proj(gen_seq).view(-1, H, D))
        k = self.norm_added_k(self.add_k_proj(gen_seq).view(-1, Hkv, D))
        v = self.add_v_proj(gen_seq).view(-1, Hkv, D)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        # Ulysses all-to-all wraps run_attention: gen_seq is sequence-sharded
        # across the SP group, so attention runs over the full sequence at
        # tp*sp head-degree, then the result is re-sharded back. Trivial SP
        # group -> passthrough (byte-identical to the non-SP path). The captured
        # denoise forward sets prefer_all_gather (the all-to-all does not replay
        # from a CUDA graph; all-gather does).
        out = ulysses_attention(
            self.sp_group, q, k, v, cache_handle.run_attention, seq_sizes,
            prefer_all_gather=prefer_all_gather,
        ).reshape(-1, H * D)
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
        comm_group: CommGroup | None = None,
        sp_group: CommGroup | None = None,
    ):
        super().__init__()
        self.self_attn = Cosmos3PackedMoTAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
            comm_group=comm_group,
            sp_group=sp_group,
        )
        self.mlp = Cosmos3MLP(hidden_size, intermediate_size, comm_group=comm_group)
        self.mlp_moe_gen = Cosmos3MLP(hidden_size, intermediate_size, comm_group=comm_group)

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

    def forward_gen(
        self, gen_seq: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        cache_handle, seq_sizes: list[int] | None = None,
        prefer_all_gather: bool = False,
    ) -> torch.Tensor:
        gen_norm = self.input_layernorm_moe_gen(gen_seq)
        attn_out = self.self_attn.forward_gen(
            gen_norm, cos, sin, cache_handle, seq_sizes, prefer_all_gather
        )
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

    def __init__(self, config, comm_group: CommGroup | None = None, sp_group: CommGroup | None = None):
        super().__init__()
        self.config = config
        h = config.hidden_size
        if sp_group is None:
            sp_group = CommGroup.trivial()
        self.sp_group = sp_group
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group

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
                comm_group=comm_group,
                sp_group=sp_group,
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
        self.action_dim = config.max_action_dim
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

    def _embed_action(
        self,
        action_latents: torch.Tensor,
        action_domain_id: torch.Tensor,
        action_timesteps: torch.Tensor,
        action_token_shapes: list[tuple[int, int, int]],
        action_noisy_frame_indexes: list[torch.Tensor],
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Project action tokens ([1, T, D]) into the hidden space: domain-aware
        in-projection + the action modality embedding, then scatter-add the
        timestep embedding to the noisy (predicted) action tokens only. Returns
        [T, hidden]."""
        packed = self.action_proj_in(action_latents, action_domain_id)[0]  # [T, hidden]
        packed = packed + self.action_modality_embed.to(packed.dtype)
        ts = action_timesteps * self.config.timestep_scale
        ts_embeds = self.time_embedder(self.time_proj(ts)).to(target_dtype)
        return self._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed,
            packed_timestep_embeds=ts_embeds,
            noisy_frame_indexes=action_noisy_frame_indexes,
            token_shapes=action_token_shapes,
        )

    def _embed_sound(
        self,
        sound_latents: torch.Tensor,
        sound_timesteps: torch.Tensor,
        sound_token_shapes: list[tuple[int, int, int]],
        sound_noisy_frame_indexes: list[torch.Tensor],
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Project sound latents ([1, C, T] or [C, T]) into the hidden space:
        audio in-projection + the audio modality embedding, then scatter-add the
        timestep embedding to the noisy sound frames (all of them — sound
        generation has no clean conditioning tokens). Returns [T, hidden]."""
        latents = sound_latents[0] if sound_latents.ndim == 3 else sound_latents
        packed = self._pack_sound_latents([latents], sound_token_shapes).to(target_dtype)
        packed = self.audio_proj_in(packed) + self.audio_modality_embed
        ts = sound_timesteps * self.config.timestep_scale
        ts_embeds = self.time_embedder(self.time_proj(ts)).to(target_dtype)
        return self._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed,
            packed_timestep_embeds=ts_embeds,
            noisy_frame_indexes=sound_noisy_frame_indexes,
            token_shapes=sound_token_shapes,
        )

    def _decode_sound(
        self,
        gen_hidden: torch.Tensor,
        sound_token_shapes: list[tuple[int, int, int]],
        sound_noisy_frame_indexes: list[torch.Tensor],
    ) -> torch.Tensor:
        """Audio out-projection of the noisy sound hidden states back to latent
        space, scattered into a full [1, C, T] tensor (clean frames left zero,
        mirroring ``_unpack_sound_latents``)."""
        preds = self.audio_proj_out(gen_hidden)  # [n_noisy, C]
        t_s = sound_token_shapes[0][0]
        out = preds.new_zeros((self.config.sound_dim, t_s))
        noisy = sound_noisy_frame_indexes[0]
        if noisy.numel() > 0:
            out[:, noisy] = preds.T
        return out.unsqueeze(0)  # [1, C, T]

    def _decode_action(
        self,
        gen_hidden: torch.Tensor,
        action_domain_id: torch.Tensor,
        action_token_shapes: list[tuple[int, int, int]],
        action_noisy_frame_indexes: list[torch.Tensor],
    ) -> torch.Tensor:
        """Domain-aware out-projection of the noisy action hidden states back to
        action space, scattered into a full [1, T, D] tensor (clean tokens left
        zero, matching the velocity mask the scheduler applies)."""
        preds = self.action_proj_out(gen_hidden.unsqueeze(0), action_domain_id)[0]  # [n_noisy, D]
        t_a = action_token_shapes[0][0]
        out = preds.new_zeros((t_a, self.action_dim))
        noisy = action_noisy_frame_indexes[0]
        if noisy.numel() > 0:
            out[noisy] = preds
        return out.unsqueeze(0)  # [1, T, D]

    # ------------------------------------------------------------------
    # forward: full per-step pass — encode text/vision, run layers, decode
    # velocity. Reference path for the fused test pipeline + parity tests;
    # production serving uses prefill_und + denoise_step* below.
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
        action_tokens: torch.Tensor | None = None,
        action_token_shapes: list[tuple[int, int, int]] | None = None,
        action_sequence_indexes: torch.Tensor | None = None,
        action_mse_loss_indexes: torch.Tensor | None = None,
        action_timesteps: torch.Tensor | None = None,
        action_noisy_frame_indexes: list[torch.Tensor] | None = None,
        action_domain_id: torch.Tensor | None = None,
    ) -> tuple:
        # Returns ``(vision, sound)`` for video/sound generation (diffusers-
        # compatible) or ``(vision, action, sound)`` when action tokens are given.
        has_sound = sound_tokens is not None and sound_sequence_indexes is not None
        has_action = action_tokens is not None and action_sequence_indexes is not None

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

        # Project + place action tokens (after the vision block in the gen
        # sequence): domain-aware in-projection + modality embed, timestep embed
        # added only to noisy (predicted) action tokens.
        if has_action:
            packed_tokens_action = self._embed_action(
                action_tokens, action_domain_id, action_timesteps,
                action_token_shapes, action_noisy_frame_indexes, target_dtype,
            )
            hidden_states[action_sequence_indexes] = packed_tokens_action

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

        preds_action: torch.Tensor | None = None
        if has_action:
            preds_action = self._decode_action(
                last_hidden_state[action_mse_loss_indexes],
                action_domain_id, action_token_shapes, action_noisy_frame_indexes,
            )

        preds_sound: list[torch.Tensor] | None = None
        if has_sound:
            preds_sound_packed = self.audio_proj_out(last_hidden_state[sound_mse_loss_indexes])
            preds_sound = self._unpack_sound_latents(preds_sound_packed, sound_token_shapes, sound_noisy_frame_indexes)

        # Video/sound generation keeps the diffusers ``(vision, sound)`` return so
        # this module is a drop-in for the diffusers transformer; action
        # generation additionally returns the predicted action band.
        if has_action:
            return preds_vision, preds_action, preds_sound
        return preds_vision, preds_sound

    # ------------------------------------------------------------------
    # Cache-once engine path: the understanding tower runs once and writes its
    # K/V; the generation tower then runs per denoising step, re-reading that
    # frozen K/V. Because the text tokens never receive a timestep embedding,
    # their K/V is step-independent, so caching it once is exact. ``cache_handle``
    # is a paged attention handle (set_layer_idx / run_attention / advance_seq_lens);
    # the attention plan (causal vs not, which label) is configured by the caller.
    # These entry points pack text + vision, plus an optional action or sound
    # band appended to the generation block (matching the reference ``forward``'s
    # ``[vision | action | sound]`` order).
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

    def _sp_run_gen_layers(self, gen_seq, cos, sin, cache_handle, prefer_all_gather=False):
        """Run the generation layer stack, sequence-parallel-sharded across the
        SP group when active. ``gen_seq``/``cos``/``sin`` are the FULL sequence
        (identical on every SP rank); returns the FULL post-layer sequence.

        This is the only sequence-parallel bracket in the denoise path: the
        prologue (patchify, proj_in, timestep scatter-add, action concat, rotary)
        and epilogue (final norm, proj_out, unpatchify) all use absolute token
        indices and must run on the full sequence, so SP shards only the layer
        stack — where essentially all the compute is. The Ulysses all-to-all
        inside each layer is a redistribute-and-reconstruct, so a contiguous
        scatter + rank-order gather preserves the exact token order (including a
        packed ``[cond | uncond]`` CFG batch)."""
        sp = self.sp_group
        if sp.world_size == 1:
            for i, layer in enumerate(self.layers):
                cache_handle.set_layer_idx(i)
                gen_seq = layer.forward_gen(gen_seq, cos, sin, cache_handle)
            return gen_seq
        seq_sizes = sp_seq_split(gen_seq.shape[0], sp.world_size)
        gen_seq = scatter_sequence(sp, gen_seq, seq_sizes)
        cos = scatter_sequence(sp, cos, seq_sizes)
        sin = scatter_sequence(sp, sin, seq_sizes)
        for i, layer in enumerate(self.layers):
            cache_handle.set_layer_idx(i)
            gen_seq = layer.forward_gen(
                gen_seq, cos, sin, cache_handle, seq_sizes, prefer_all_gather
            )
        return gather_sequence(sp, gen_seq, seq_sizes)

    def denoise_step(
        self,
        latents: torch.Tensor,
        vision_timesteps: torch.Tensor,
        position_ids: torch.Tensor,
        vision_token_shapes: list[tuple[int, int, int]],
        vision_noisy_frame_indexes: list[torch.Tensor],
        vision_mse_loss_indexes: torch.Tensor,
        cache_handle,
        action_latents: torch.Tensor | None = None,
        action_token_shapes: list[tuple[int, int, int]] | None = None,
        action_noisy_frame_indexes: list[torch.Tensor] | None = None,
        action_mse_gen_indexes: torch.Tensor | None = None,
        action_timesteps: torch.Tensor | None = None,
        action_domain_id: torch.Tensor | None = None,
        sound_latents: torch.Tensor | None = None,
        sound_token_shapes: list[tuple[int, int, int]] | None = None,
        sound_noisy_frame_indexes: list[torch.Tensor] | None = None,
        sound_mse_gen_indexes: torch.Tensor | None = None,
        sound_timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """One generation-tower evaluation against the frozen understanding K/V.

        Patchifies ``latents`` ([1, C, T, H, W]), scatter-adds the timestep
        embedding to the noisy tokens, runs the generation layers (each reading
        the active label's cached understanding K/V plus its own freshly written
        K/V), and decodes the flow velocity. ``position_ids`` are the generation
        segment's 3D mRoPE ids ([3, num_gen]) — the vision band, then the action
        or sound band when present. ``vision_mse_loss_indexes`` /
        ``action_mse_gen_indexes`` / ``sound_mse_gen_indexes`` index into the
        generation token block. With an extra band the generation sequence is
        ``[vision tokens | action or sound tokens]`` and the call returns
        ``(video_velocity, action_velocity)`` / ``(video_velocity,
        sound_velocity)``."""
        has_action = action_latents is not None
        has_sound = sound_latents is not None
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
        if has_action:
            action_seq = self._embed_action(
                action_latents, action_domain_id, action_timesteps,
                action_token_shapes, action_noisy_frame_indexes, target_dtype,
            )
            gen_seq = torch.cat([gen_seq, action_seq], dim=0)
        if has_sound:
            sound_seq = self._embed_sound(
                sound_latents, sound_timesteps, sound_token_shapes,
                sound_noisy_frame_indexes, target_dtype,
            )
            gen_seq = torch.cat([gen_seq, sound_seq], dim=0)

        cos, sin = self._rotary(position_ids, gen_seq.device, gen_seq.dtype)
        gen_seq = self._sp_run_gen_layers(gen_seq, cos, sin, cache_handle)
        gen_out = self.norm_moe_gen(gen_seq)
        preds_packed = self.proj_out(gen_out[vision_mse_loss_indexes])
        preds = self._unpatchify_and_unpack_latents(
            preds_packed,
            token_shapes_vision=vision_token_shapes,
            noisy_frame_indexes_vision=vision_noisy_frame_indexes,
            original_latent_shapes=original_latent_shapes,
        )
        outputs: list[torch.Tensor] = [preds[0]]
        if has_action:
            outputs.append(self._decode_action(
                gen_out[action_mse_gen_indexes], action_domain_id,
                action_token_shapes, action_noisy_frame_indexes,
            ))
        if has_sound:
            outputs.append(self._decode_sound(
                gen_out[sound_mse_gen_indexes], sound_token_shapes, sound_noisy_frame_indexes,
            ))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def denoise_step_batched_cfg(
        self,
        latents: torch.Tensor,
        vision_timesteps: torch.Tensor,
        position_ids_cond: torch.Tensor,
        position_ids_uncond: torch.Tensor,
        vision_token_shapes: list[tuple[int, int, int]],
        vision_noisy_frame_indexes: list[torch.Tensor],
        vision_mse_loss_indexes: torch.Tensor,
        cache_handle,
        action_latents: torch.Tensor | None = None,
        action_token_shapes: list[tuple[int, int, int]] | None = None,
        action_noisy_frame_indexes: list[torch.Tensor] | None = None,
        action_mse_gen_indexes: torch.Tensor | None = None,
        action_timesteps: torch.Tensor | None = None,
        action_domain_id: torch.Tensor | None = None,
        sound_latents: torch.Tensor | None = None,
        sound_token_shapes: list[tuple[int, int, int]] | None = None,
        sound_noisy_frame_indexes: list[torch.Tensor] | None = None,
        sound_mse_gen_indexes: torch.Tensor | None = None,
        sound_timesteps: torch.Tensor | None = None,
        prefer_all_gather: bool = False,
    ):
        """Conditional and unconditional generation in one batched pass.

        The two classifier-free-guidance branches share identical generation
        tokens — same latents, same timestep, so the patchified input and its
        timestep embedding are built once and repeated. They differ only in (a)
        the text-conditioning K/V they attend to (held under two cache labels)
        and (b) their rotary positions: the media band starts just after each
        branch's text, and the two prompts have different lengths. So pack
        ``[cond tokens | uncond tokens]`` into one sequence carrying per-branch
        positions, and let the handle's batched plan route each branch to its
        own label's pages. Returns the conditional and unconditional results in
        the same form as ``denoise_step`` (a velocity, or a (video, action) /
        (video, sound) pair when the extra band is present)."""
        has_action = action_latents is not None
        has_sound = sound_latents is not None
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
        if has_action:
            action_seq = self._embed_action(
                action_latents, action_domain_id, action_timesteps,
                action_token_shapes, action_noisy_frame_indexes, target_dtype,
            )
            gen_seq = torch.cat([gen_seq, action_seq], dim=0)
        if has_sound:
            sound_seq = self._embed_sound(
                sound_latents, sound_timesteps, sound_token_shapes,
                sound_noisy_frame_indexes, target_dtype,
            )
            gen_seq = torch.cat([gen_seq, sound_seq], dim=0)

        n = gen_seq.shape[0]
        gen_seq = torch.cat([gen_seq, gen_seq], dim=0)
        cos_c, sin_c = self._rotary(position_ids_cond, gen_seq.device, gen_seq.dtype)
        cos_u, sin_u = self._rotary(position_ids_uncond, gen_seq.device, gen_seq.dtype)
        cos = torch.cat([cos_c, cos_u], dim=0)
        sin = torch.cat([sin_c, sin_u], dim=0)

        gen_seq = self._sp_run_gen_layers(
            gen_seq, cos, sin, cache_handle, prefer_all_gather=prefer_all_gather
        )
        gen_out = self.norm_moe_gen(gen_seq)

        def _decode(out):
            preds_packed = self.proj_out(out[vision_mse_loss_indexes])
            preds = self._unpatchify_and_unpack_latents(
                preds_packed,
                token_shapes_vision=vision_token_shapes,
                noisy_frame_indexes_vision=vision_noisy_frame_indexes,
                original_latent_shapes=original_latent_shapes,
            )
            outputs: list[torch.Tensor] = [preds[0]]
            if has_action:
                outputs.append(self._decode_action(
                    out[action_mse_gen_indexes], action_domain_id,
                    action_token_shapes, action_noisy_frame_indexes,
                ))
            if has_sound:
                outputs.append(self._decode_sound(
                    out[sound_mse_gen_indexes], sound_token_shapes, sound_noisy_frame_indexes,
                ))
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        return _decode(gen_out[:n]), _decode(gen_out[n:])

    def denoise_step_batched(self, requests: list[dict], cache_handle):
        """Denoise one step for several requests at once (image / video).

        Each request carries its own latents, timestep, rotary positions (which
        differ per request, and per guidance branch) and token layout. Every
        request contributes a conditional and an unconditional sequence, packed
        as ``[cond r0 | cond r1 | ... | uncond r0 | uncond r1 | ...]`` to match
        the order the handle's batched plan lays out its entries. The layers run
        once over the whole pack; the cache routes each piece to its own request
        and guidance label. Returns one ``(cond, uncond)`` pair per request, in
        request order — each branch a velocity, or a ``(video, sound)`` pair for
        a request with a sound band.

        Each ``requests`` entry is a dict with: ``latents``, ``vision_timesteps``,
        ``position_ids_cond``, ``position_ids_uncond``, ``vision_token_shapes``,
        ``vision_noisy_frame_indexes``, ``vision_mse_loss_indexes``; sound-band
        requests additionally carry ``sound_latents``, ``sound_timesteps``,
        ``sound_token_shapes``, ``sound_noisy_frame_indexes``,
        ``sound_mse_gen_indexes``."""
        gen_seqs, shapes, cos_cond, sin_cond, cos_uncond, sin_uncond = [], [], [], [], [], []
        for req in requests:
            packed, original_latent_shapes = self._patchify_and_pack_latents([req["latents"]])
            packed = self.proj_in(packed)
            ts_embeds = self.time_embedder(
                self.time_proj(req["vision_timesteps"] * self.config.timestep_scale)
            ).to(packed.dtype)
            gen_seq = self._apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed,
                packed_timestep_embeds=ts_embeds,
                noisy_frame_indexes=req["vision_noisy_frame_indexes"],
                token_shapes=req["vision_token_shapes"],
            )
            if req.get("sound_latents") is not None:
                sound_seq = self._embed_sound(
                    req["sound_latents"], req["sound_timesteps"], req["sound_token_shapes"],
                    req["sound_noisy_frame_indexes"], packed.dtype,
                )
                gen_seq = torch.cat([gen_seq, sound_seq], dim=0)
            gen_seqs.append(gen_seq)
            shapes.append(original_latent_shapes)
            cc, sc = self._rotary(req["position_ids_cond"], gen_seq.device, gen_seq.dtype)
            cu, su = self._rotary(req["position_ids_uncond"], gen_seq.device, gen_seq.dtype)
            cos_cond.append(cc)
            sin_cond.append(sc)
            cos_uncond.append(cu)
            sin_uncond.append(su)

        # Conditional block first (all requests), then unconditional block.
        all_gen = torch.cat(gen_seqs + gen_seqs, dim=0)
        cos = torch.cat(cos_cond + cos_uncond, dim=0)
        sin = torch.cat(sin_cond + sin_uncond, dim=0)
        all_gen = self._sp_run_gen_layers(all_gen, cos, sin, cache_handle)
        gen_out = self.norm_moe_gen(all_gen)

        sizes = [g.shape[0] for g in gen_seqs]
        total = sum(sizes)
        cond_out, uncond_out = gen_out[:total], gen_out[total:]

        def _decode(out, req, original_latent_shapes):
            preds_packed = self.proj_out(out[req["vision_mse_loss_indexes"]])
            preds = self._unpatchify_and_unpack_latents(
                preds_packed,
                token_shapes_vision=req["vision_token_shapes"],
                noisy_frame_indexes_vision=req["vision_noisy_frame_indexes"],
                original_latent_shapes=original_latent_shapes,
            )
            if req.get("sound_latents") is None:
                return preds[0]
            sound_pred = self._decode_sound(
                out[req["sound_mse_gen_indexes"]], req["sound_token_shapes"],
                req["sound_noisy_frame_indexes"],
            )
            return preds[0], sound_pred

        results, off = [], 0
        for i, req in enumerate(requests):
            n = sizes[i]
            cond_v = _decode(cond_out[off:off + n], req, shapes[i])
            uncond_v = _decode(uncond_out[off:off + n], req, shapes[i])
            off += n
            results.append((cond_v, uncond_v))
        return results

    def denoise_step_action_batched(self, requests: list[dict], cache_handle, with_cfg: bool):
        """Joint ``[video | action]`` denoise for several action requests at once.

        The action analogue of ``denoise_step_batched``. Each request carries its
        own video latents, action latents, per-band timesteps, rotary positions
        (per guidance branch), token layout and embodiment domain id; its
        generation block is ``[vision tokens | action tokens]``. With classifier-
        free guidance every request contributes a conditional and an
        unconditional copy, packed ``[cond r0 | ... | cond rN | uncond r0 | ... |
        uncond rN]`` to match the handle's batched plan; without guidance (the
        guidance-scale-1 forward/inverse-dynamics and base policy case) each
        request contributes a single sequence ``[r0 | r1 | ... | rN]``. The layers
        run once over the whole pack; the cache routes each piece to its own
        request and guidance label. The per-request action projection is
        domain-aware, so requests from different embodiments can share the batch.

        Returns one entry per request, in request order: a tuple of branch
        results, each a ``(video_velocity, action_velocity)`` pair — one branch
        without guidance, ``(conditional, unconditional)`` with.

        Each ``requests`` entry is a dict with: ``latents``, ``action_latents``,
        ``vision_timesteps``, ``action_timesteps``, ``position_ids_cond``
        (plus ``position_ids_uncond`` when ``with_cfg``), ``vision_token_shapes``,
        ``vision_noisy_frame_indexes``, ``vision_mse_loss_indexes``,
        ``action_token_shapes``, ``action_noisy_frame_indexes``,
        ``action_mse_gen_indexes``, ``action_domain_id``."""
        gen_seqs, shapes, cos_cond, sin_cond, cos_uncond, sin_uncond = [], [], [], [], [], []
        for req in requests:
            packed, original_latent_shapes = self._patchify_and_pack_latents([req["latents"]])
            packed = self.proj_in(packed)
            target_dtype = packed.dtype
            ts_embeds = self.time_embedder(
                self.time_proj(req["vision_timesteps"] * self.config.timestep_scale)
            ).to(target_dtype)
            gen_seq = self._apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed,
                packed_timestep_embeds=ts_embeds,
                noisy_frame_indexes=req["vision_noisy_frame_indexes"],
                token_shapes=req["vision_token_shapes"],
            )
            action_seq = self._embed_action(
                req["action_latents"], req["action_domain_id"], req["action_timesteps"],
                req["action_token_shapes"], req["action_noisy_frame_indexes"], target_dtype,
            )
            gen_seq = torch.cat([gen_seq, action_seq], dim=0)
            gen_seqs.append(gen_seq)
            shapes.append(original_latent_shapes)
            cc, sc = self._rotary(req["position_ids_cond"], gen_seq.device, gen_seq.dtype)
            cos_cond.append(cc)
            sin_cond.append(sc)
            if with_cfg:
                cu, su = self._rotary(req["position_ids_uncond"], gen_seq.device, gen_seq.dtype)
                cos_uncond.append(cu)
                sin_uncond.append(su)

        if with_cfg:
            all_gen = torch.cat(gen_seqs + gen_seqs, dim=0)
            cos = torch.cat(cos_cond + cos_uncond, dim=0)
            sin = torch.cat(sin_cond + sin_uncond, dim=0)
        else:
            all_gen = torch.cat(gen_seqs, dim=0)
            cos = torch.cat(cos_cond, dim=0)
            sin = torch.cat(sin_cond, dim=0)

        all_gen = self._sp_run_gen_layers(all_gen, cos, sin, cache_handle)
        gen_out = self.norm_moe_gen(all_gen)

        sizes = [g.shape[0] for g in gen_seqs]
        total = sum(sizes)
        offsets, acc = [], 0
        for n in sizes:
            offsets.append(acc)
            acc += n

        def _decode(out, req, original_latent_shapes):
            preds_packed = self.proj_out(out[req["vision_mse_loss_indexes"]])
            preds = self._unpatchify_and_unpack_latents(
                preds_packed,
                token_shapes_vision=req["vision_token_shapes"],
                noisy_frame_indexes_vision=req["vision_noisy_frame_indexes"],
                original_latent_shapes=original_latent_shapes,
            )
            action_pred = self._decode_action(
                out[req["action_mse_gen_indexes"]], req["action_domain_id"],
                req["action_token_shapes"], req["action_noisy_frame_indexes"],
            )
            return preds[0], action_pred

        cond_block = gen_out[:total]
        uncond_block = gen_out[total:] if with_cfg else None
        results = []
        for i, req in enumerate(requests):
            o, n = offsets[i], sizes[i]
            cond_res = _decode(cond_block[o:o + n], req, shapes[i])
            if with_cfg:
                uncond_res = _decode(uncond_block[o:o + n], req, shapes[i])
                results.append((cond_res, uncond_res))
            else:
                results.append((cond_res,))
        return results
