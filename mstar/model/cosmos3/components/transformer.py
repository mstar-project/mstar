"""Cosmos3 dual-pathway Mixture-of-Transformers DiT (parameter structure).

Each decoder layer carries two parameter sets that run side by side:

  * UND (understanding / text-conditioning) pathway — ``to_{q,k,v,out}``,
    ``norm_{q,k}``, ``mlp``, ``input_layernorm``, ``post_attention_layernorm``.
    Causal self-attention over the text prefix; never attends to GEN tokens.
  * GEN (generation / denoiser) pathway — ``add_{q,k,v}_proj``, ``to_add_out``,
    ``norm_added_{q,k}``, ``mlp_moe_gen``, ``input_layernorm_moe_gen``,
    ``post_attention_layernorm_moe_gen``. Full (non-causal) attention where
    GEN queries attend to ``cat([k_und, k_gen])`` / ``cat([v_und, v_gen])``.

The module mirrors the published diffusers checkpoint layout one-to-one, so
the flat ``layers.N.*`` safetensors keys load with no key remapping beyond
dropping the unused text ``lm_head``. Projections are plain ``nn.Linear`` here;
tensor-parallel variants are a later concern. The forward pass (patchify,
timestep scatter, mRoPE, joint attention, unpatchify) is wired separately.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Weight-only RMS normalization (no bias), matching the checkpoint's
    ``*.weight`` parameter and the model's ``rms_norm_eps``."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class TimestepEmbedder(nn.Module):
    """Two-layer MLP over sinusoidal timestep features (``linear_1``/``linear_2``)."""

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
    the understanding (causal) and generation (full) token streams."""

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

    def forward(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError("joint attention forward not yet wired")


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

    def forward(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError("decoder layer forward not yet wired")


class DomainAwareLinear(nn.Module):
    """Per-embodiment affine map: one shared weight (``fc``) plus a per-domain
    additive bias looked up from an embedding table (``bias``). Used by the
    action projection heads, keyed by an embodiment-domain id."""

    def __init__(self, in_features: int, out_features: int, num_domains: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Embedding(num_domains, out_features)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + self.bias(domain_id)


class Cosmos3OmniTransformer(nn.Module):
    """The full Cosmos3 generator backbone (parameter structure).

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

        # Vision latent in/out projections + timestep embedder.
        self.proj_in = nn.Linear(config.patch_latent_dim, h, bias=True)
        self.proj_out = nn.Linear(h, config.patch_latent_dim, bias=True)
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

    def forward(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError("Cosmos3 transformer forward not yet wired")
