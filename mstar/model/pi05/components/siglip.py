"""SigLIP vision encoder for Pi0.5 (native mminf port).

Ports the inference path of HuggingFace's ``SiglipVisionModel`` (So400m/14)
into mminf so we own the code and can fuse projections. Differences from the
transformers implementation:

  * **Fused QKV** — the three ``q/k/v_proj`` GEMMs are merged into one
    ``QKVParallelLinear`` (loaded from the separate checkpoint keys via the
    ``q/k/v`` stacked-param rules; see ``SIGLIP_STACKED_PARAMS``).
  * **SDPA attention** — full bidirectional ``scaled_dot_product_attention``.
    We do NOT use flash-attn or the Triton ``sliding_window_attn`` here: the
    encoder runs in **fp32** (Pi05VitEncoderSubmodule forces it, since bf16
    rounding over 27 layers perturbs the actions) and flash-attn is fp16/bf16
    only, while the Triton kernel is causal-only and rejects head_dim=72.
  * **Inference-only** — all weight-init, gradient-checkpointing, the text
    tower, pooling head, and variable-resolution position interpolation are
    dropped. Images are a fixed 224x224 → 256 patches.

Only ``last_hidden_state`` is consumed downstream (``vision_use_head=False``
in the original), so the pooling head is omitted entirely.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import TPCommGroup
from mstar.model.components.distributed.linear import QKVParallelLinear
from mstar.model.loader import StackedParamRule
from mstar.model.pi05.config import Pi05Config

# SigLIP architectural constants not carried on Pi05Config. These match
# HF ``SiglipVisionConfig`` defaults for the So400m checkpoint.
_LAYER_NORM_EPS = 1e-6

# Route the checkpoint's separate q/k/v projection keys into the fused
# ``qkv_proj`` parameter. Consumed by ``load_hf_weights`` when loading the
# encoder (the SigLIP MLP is ungated, so there are no gate/up rules).
SIGLIP_STACKED_PARAMS: list[StackedParamRule] = [
    StackedParamRule(".qkv_proj", ".q_proj", "q"),
    StackedParamRule(".qkv_proj", ".k_proj", "k"),
    StackedParamRule(".qkv_proj", ".v_proj", "v"),
]


class _SiglipVisionEmbeddings(nn.Module):
    """Conv patch embedding + learned position embedding.

    Fixed-resolution only: 224x224 input → a 16x16 grid of 14px patches →
    256 tokens. Position ids are computed inline (no buffer) so the module
    has no non-persistent state to re-materialize after ``to_empty``.
    """

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.embed_dim = config.vit_hidden_size
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=config.vit_patch_size,
            stride=config.vit_patch_size,
            padding="valid",
        )
        self.num_positions = (config.vit_image_size // config.vit_patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (N, 3, H, W) -> patches (N, embed_dim, gh, gw).
        patch_embeds = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)  # (N, num_patches, embed_dim)
        positions = torch.arange(self.num_positions, device=embeddings.device)
        return embeddings + self.position_embedding(positions)


class _SiglipAttention(nn.Module):
    """Bidirectional multi-head self-attention with a fused QKV projection.

    Full MHA (no GQA): num_kv_heads == num_heads. Attention is computed
    per-image over its own 256 patches (the batch dim isolates images), so
    no attention mask is needed.
    """

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.embed_dim = config.vit_hidden_size
        self.num_heads = config.vit_num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"vit_hidden_size {self.embed_dim} not divisible by "
                f"vit_num_heads {self.num_heads}"
            )
        self.scale = self.head_dim**-0.5

        # Trivial (single-rank) comm group: reuses the TP-aware fused-QKV
        # loader without any actual sharding. bias=True — SigLIP projects
        # q/k/v with bias.
        self.qkv_proj = QKVParallelLinear(
            comm_group=TPCommGroup.trivial(),
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,
            bias=True,
        )
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)  # (N, seq, 3*embed_dim)
        q, k, v = qkv.split([self.embed_dim, self.embed_dim, self.embed_dim], dim=-1)

        # (N, seq, embed) -> (N, heads, seq, head_dim) for SDPA.
        def to_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(n, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            to_heads(q), to_heads(k), to_heads(v), scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(n, seq_len, self.embed_dim)
        return self.out_proj(out)


class _SiglipMLP(nn.Module):
    """Ungated 2-layer MLP with gelu-tanh activation."""

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.fc1 = nn.Linear(config.vit_hidden_size, config.vit_intermediate_size)
        self.activation_fn = nn.GELU(approximate="tanh")  # gelu_pytorch_tanh
        self.fc2 = nn.Linear(config.vit_intermediate_size, config.vit_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class _SiglipEncoderLayer(nn.Module):
    """Pre-norm transformer block: ln1→attn→res, ln2→mlp→res."""

    def __init__(self, config: Pi05Config):
        super().__init__()
        embed_dim = config.vit_hidden_size
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=_LAYER_NORM_EPS)
        self.self_attn = _SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=_LAYER_NORM_EPS)
        self.mlp = _SiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class _SiglipEncoder(nn.Module):
    """Stack of encoder layers. Named to match the ``encoder.layers.N``
    checkpoint key layout so weights load without per-layer remapping."""

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [_SiglipEncoderLayer(config) for _ in range(config.vit_num_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _SiglipVisionTransformer(nn.Module):
    """Embeddings → encoder stack → final layer norm."""

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.embeddings = _SiglipVisionEmbeddings(config)
        self.encoder = _SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.vit_hidden_size, eps=_LAYER_NORM_EPS)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        return self.post_layernorm(hidden_states)


class Pi05SiglipEncoder(nn.Module):
    """SigLIP image encoder + linear connector to the LLM hidden size."""

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config
        self.vision_model = _SiglipVisionTransformer(config)
        self.connector = nn.Linear(config.vit_hidden_size, config.hidden_size, bias=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into LLM-space tokens.

        Args:
            pixel_values: Tensor of shape ``(N, 3, H, W)`` where ``N`` is the
                total number of images across cameras and requests.

        Returns:
            Tensor of shape ``(N, tokens_per_image, hidden_size)``.
        """
        features = self.vision_model(pixel_values)  # (N, num_patches, vit_hidden)
        return self.connector(features)
