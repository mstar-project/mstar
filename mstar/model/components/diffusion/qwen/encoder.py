"""Native Qwen-family prompt encoder that returns tapped hidden states.

FLUX.2 [klein] and Z-Image condition their DiTs on intermediate hidden states of
a Qwen3 language model (klein: layers 9/18/27 of Qwen3-4B/8B concatenated), not on
its logits; MiniMax-H3 on a Qwen3-VL language model's layer 50; Qwen-Image on a
Qwen2.5-VL language model's normed last hidden state. This module is that encoder
built from M*'s transformer components: ``DecoderLayer`` + ``GatedMLP`` + an HF-order
RMSNorm with a GQA attention (per-head q/k RMSNorm for Qwen3, biased q/k/v for Qwen2),
run only through the deepest tapped layer. The checkpoint facts, the VL ones included
(the image token, the MRoPE split and layout), are the config's
(``QwenEncoderConfig.from_dict`` / ``from_qwen3_vl`` / ``from_qwen2_5_vl``). A prompt with
images comes in with the vision tower's output (``vl_prompt.VisionTokens``), which the
encoder splices in, positions with 3-D MRoPE and, for Qwen3-VL, adds deepstack features to.

Prompts arrive right-padded to one length (the pipelines pad to
``max_sequence_length`` and feed every position, pad tokens included, to the
DiT), so the padded positions' hidden states must be exactly what the reference
LM computes for them: a query at position ``i`` attends causally to keys ``<= i``
that are not padding. That mask is built once per batch and handed to SDPA, the
reference's own kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components import DecoderLayer, GatedMLP
from mstar.model.components.diffusion.qwen.vl_prompt import VisionTokens, vl_positions
from mstar.model.components.linear import FusedColumnLinear
from mstar.model.components.mrope import compute_3d_cos_sin
from mstar.model.components.norm import QwenRMSNorm


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def qwen_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    """HF's default RoPE ``inv_freq`` (``_compute_default_rope_parameters``), computed on the
    CPU as the reference does at module init: CPU and GPU ``pow`` differ in the last ulp for
    some entries, and that ulp survives into the bf16 sin table and the attention output."""
    steps = torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float)
    return 1.0 / (theta ** (steps / head_dim))


def qwen_rotary_tables(
    positions: torch.Tensor, head_dim: int, theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF ``Qwen3RotaryEmbedding.forward``: fp32 ``cos, sin [S, head_dim]`` laid out ``[freqs | freqs]``,
    the angles from the same K=1 matmul on the positions' device."""
    inv_freq = qwen_inv_freq(head_dim, theta).to(positions.device)
    freqs = (inv_freq[None, :, None] @ positions[None, None, :].float()).transpose(1, 2)[0]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def padded_causal_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """``[B, 1, S, S]`` boolean SDPA mask: causal, and keys that are padding are never
    attended (the reference ``create_causal_mask`` semantics for a right-padded batch)."""
    seq = attention_mask.shape[1]
    causal = torch.ones(seq, seq, dtype=torch.bool, device=attention_mask.device).tril()
    return causal[None, None] & attention_mask.bool()[:, None, None, :]


@dataclass(frozen=True)
class QwenEncoderConfig:
    """``text_encoder/config.json`` of the Qwen LM the prompt runs through, plus the
    pipeline's tapping recipe (``text_encoder_out_layers``, ``max_sequence_length``)."""

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    pad_token_id: int = 151643
    # Pipeline recipe (Flux2KleinPipeline.encode_prompt defaults). A tap is HF's
    # hidden_states[k]: layer k's output, and at k == num_hidden_layers the normed last one.
    hidden_state_layers: tuple[int, ...] = (9, 18, 27)
    max_sequence_length: int = 512
    # One q/k/v GEMM, or three as transformers runs them. cuBLAS may pick another kernel
    # for the fused shape, and then k and v differ from the reference's in the last bit
    # (Qwen3-VL-32B's 1024-wide k/v do; Qwen3-4B/8B's don't).
    fused_qkv: bool = True
    # The attention's checkpoint facts: Qwen3 normalizes each head's q and k, Qwen2 biases q/k/v.
    qk_norm: bool = True
    qkv_bias: bool = False
    # A VL checkpoint's: the image pad token and the MRoPE split, interleaved (Qwen3-VL) or
    # sectioned (Qwen2/2.5-VL). None on a text-only LM, which takes no images.
    image_token_id: int | None = None
    mrope_section: tuple[int, ...] | None = None
    mrope_interleaved: bool = True

    def __post_init__(self):
        if not self.hidden_state_layers or min(self.hidden_state_layers) < 1:
            raise ValueError(
                f"hidden_state_layers must name decoder layers (1-indexed); got {self.hidden_state_layers}"
            )
        if max(self.hidden_state_layers) > self.num_hidden_layers:
            raise ValueError(
                f"hidden_state_layers {self.hidden_state_layers} must lie within the {self.num_hidden_layers} layers"
            )

    @property
    def num_layers_needed(self) -> int:
        """Decoder layers that must run to produce the deepest tap
        (``hidden_states[k]`` is the output of layer ``k``, 1-indexed)."""
        return max(self.hidden_state_layers)

    @property
    def taps_final_norm(self) -> bool:
        """Whether a tap is HF's ``hidden_states[num_hidden_layers]``, the normed last hidden state."""
        return self.num_hidden_layers in self.hidden_state_layers

    @property
    def output_dim(self) -> int:
        return self.hidden_size * len(self.hidden_state_layers)

    @classmethod
    def from_dict(cls, cfg: dict, **overrides) -> "QwenEncoderConfig":
        if cfg.get("model_type") not in (None, "qwen3"):
            raise NotImplementedError(f"text encoder model_type {cfg.get('model_type')!r} is not Qwen3")
        if cfg.get("rope_scaling"):
            raise NotImplementedError("Qwen3 text encoder with rope_scaling is not supported")
        return cls(**{**cls._lm_fields(cfg), **overrides})

    @staticmethod
    def _lm_fields(cfg: dict) -> dict:
        return dict(
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta", 1_000_000.0)),
            pad_token_id=int(cfg.get("pad_token_id") or 151643),
        )

    @classmethod
    def from_qwen3_vl(cls, cfg: dict, **overrides) -> "QwenEncoderConfig":
        """The language model of a Qwen3-VL checkpoint (its ``config.json``, which nests
        it under ``text_config``).

        Qwen3-VL rotates with interleaved MRoPE. A text token sits at the same position
        on all three axes, so interleaving selects among three identical angle rows and
        its rotation equals plain RoPE bit for bit: the encoder's own tables serve a
        text-only prompt, and a prompt with images takes its 3-D tables."""
        text = cfg.get("text_config", cfg)
        if text.get("model_type") not in (None, "qwen3_vl_text"):
            raise NotImplementedError(f"text model_type {text.get('model_type')!r} is not Qwen3-VL's")
        rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default" or not rope.get("mrope_interleaved", True):
            raise NotImplementedError(f"Qwen3-VL rope {rope!r}: only default interleaved MRoPE is supported")
        theta = text.get("rope_theta", rope.get("rope_theta", 5_000_000.0))
        return cls(**{
            **cls._lm_fields({**text, "rope_theta": theta}),
            "image_token_id": int(cfg.get("image_token_id", 151655)),
            "mrope_section": tuple(int(n) for n in rope.get("mrope_section", (24, 20, 20))),
            **overrides,
        })

    @classmethod
    def from_qwen2_5_vl(cls, cfg: dict, **overrides) -> "QwenEncoderConfig":
        """The language model of a Qwen2.5-VL checkpoint: a Qwen2 LM (biased q/k/v, no
        q/k norm) rotating with sectioned MRoPE."""
        text = {**cfg, **cfg.get("text_config", {})}
        if cfg.get("model_type") not in (None, "qwen2_5_vl"):
            raise NotImplementedError(f"model_type {cfg.get('model_type')!r} is not Qwen2.5-VL's")
        rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
        kind = rope.get("rope_type", rope.get("type", "default"))
        if kind not in ("default", "mrope") or "mrope_section" not in rope:
            raise NotImplementedError(f"Qwen2.5-VL rope {rope!r}: only default sectioned MRoPE is supported")
        theta = text.get("rope_theta", rope.get("rope_theta", 1_000_000.0))
        return cls(**{
            **cls._lm_fields({**text, "rope_theta": theta}),
            "qk_norm": False,
            "qkv_bias": True,
            "image_token_id": int(cfg["image_token_id"]),
            "mrope_section": tuple(int(n) for n in rope["mrope_section"]),
            "mrope_interleaved": False,
            **overrides,
        })


class QwenEncoderAttention(nn.Module):
    """GQA self-attention: fused (or HF's three) q/k/v GEMMs, with per-head q/k RMSNorm
    (Qwen3) or biased q/k/v (Qwen2), masked SDPA."""

    def __init__(self, config: QwenEncoderConfig):
        super().__init__()
        c = config
        self.num_heads, self.num_kv_heads, self.head_dim = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        kv = self.num_kv_heads * c.head_dim
        sizes = {"q": self.num_heads * c.head_dim, "k": kv, "v": kv}
        if c.fused_qkv:
            self.qkv_proj = FusedColumnLinear(c.hidden_size, sizes, bias=c.qkv_bias)
        else:
            self.qkv_proj = None
            self.q_proj, self.k_proj, self.v_proj = (
                nn.Linear(c.hidden_size, size, bias=c.qkv_bias) for size in sizes.values()
            )
        self.o_proj = nn.Linear(self.num_heads * c.head_dim, c.hidden_size, bias=False)
        self.q_norm = QwenRMSNorm(c.head_dim, eps=c.rms_norm_eps) if c.qk_norm else None
        self.k_norm = QwenRMSNorm(c.head_dim, eps=c.rms_norm_eps) if c.qk_norm else None
        # Set per forward by the encoder (shared across layers).
        self.rotary: tuple[torch.Tensor, torch.Tensor] | None = None
        self.attn_mask: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        if self.qkv_proj is None:
            q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        else:
            q, k, v = self.qkv_proj(hidden_states).split(
                [self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim],
                dim=-1,
            )
        q = q.view(batch, seq, self.num_heads, self.head_dim)
        k = k.view(batch, seq, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = v.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary
        cos, sin = cos.to(q.dtype)[None, None], sin.to(q.dtype)[None, None]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        groups = self.num_heads // self.num_kv_heads
        if self.attn_mask is None:
            # HF's mask-free path (no padding in the batch): ``is_causal`` and SDPA's native GQA
            # (``use_gqa_in_sdpa``), which select a different kernel than the masked path below;
            # both must be mirrored for the last bit to agree.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=groups > 1)
        else:
            if groups > 1:
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim))


class QwenHiddenStateEncoder(nn.Module):
    """Embeddings + the first ``max(taps)`` decoder layers of a Qwen LM (and its final norm
    when a tap is the last layer); returns the hidden states of each tapped layer,
    concatenated along the feature dim (``hidden_states[k]`` in HF indexing)."""

    def __init__(self, config: QwenEncoderConfig):
        super().__init__()
        c = self.config = config
        self.tap_layers = tuple(c.hidden_state_layers)
        self._rotary_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(
                self_attn=QwenEncoderAttention(c),
                mlp=GatedMLP(c.hidden_size, c.intermediate_size, activation="silu", bias=False),
                input_layernorm=QwenRMSNorm(c.hidden_size, eps=c.rms_norm_eps),
                post_attention_layernorm=QwenRMSNorm(c.hidden_size, eps=c.rms_norm_eps),
            )
            for _ in range(c.num_layers_needed)
        )
        self.norm = QwenRMSNorm(c.hidden_size, eps=c.rms_norm_eps) if c.taps_final_norm else None

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        padded: bool | None = None,
        *,
        images: VisionTokens | None = None,
    ) -> torch.Tensor:
        """``input_ids, attention_mask [B, S]`` (right padded) -> ``[B, S, len(taps) * hidden]``.

        Follows HF's two attention paths exactly: a batch without padding attends mask-free
        (``is_causal`` + native GQA), a padded one through a boolean causal-and-key-padding
        mask; the two pick different SDPA kernels, so the choice is part of parity.

        ``padded`` is whether any row has padding, when the caller knows it on the host;
        None reads it off ``attention_mask``, a host sync (and not graph-capturable).

        ``images`` is the vision tower's output for a one-prompt batch whose ``<|image_pad|>``
        runs hold the images: their tokens replace the pads' embeddings, the prompt rotates
        with its 3-D MRoPE positions, and each deepstack feature is added at the image rows
        after its decoder layer (``Qwen3VLTextModel._deepstack_process``).
        """
        seq = input_ids.shape[1]
        if seq > self.config.max_sequence_length:
            raise ValueError(f"{seq} prompt tokens, past the encoder's {self.config.max_sequence_length}")
        deepstack = None
        if images is None:
            hidden = self.embed_tokens(input_ids)
            rotary = self._text_rotary(input_ids.device, seq)
        else:
            hidden, rotary, deepstack = self._vl_inputs(input_ids, images)
        # No padding anywhere in the batch: HF's create_causal_mask returns None and attention
        # runs its is_causal path; mirror that rather than passing an all-True mask.
        if padded is None:
            padded = not bool(attention_mask.all())
        mask = padded_causal_mask(attention_mask) if padded else None
        taps = []
        for layer_idx, layer in enumerate(self.layers, start=1):
            layer.self_attn.rotary = rotary
            layer.self_attn.attn_mask = mask
            hidden = layer(hidden)
            if layer_idx in self.tap_layers:
                # HF records a layer's output before the deepstack add, and the last one normed
                taps.append(self.norm(hidden) if layer_idx == self.config.num_hidden_layers else hidden)
            if deepstack is not None and layer_idx <= len(deepstack[1]):
                rows, features = deepstack
                hidden = hidden.clone()
                hidden[:, rows] = hidden[:, rows] + features[layer_idx - 1].to(hidden.dtype)
        # stack(dim=1).permute(0, 2, 1, 3).reshape(...) == concatenation along the feature dim
        return torch.cat(taps, dim=-1)

    def _text_rotary(self, device: torch.device, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        # one table per device at the longest length, sliced: building it copies inv_freq from
        # the host, which a captured forward cannot do, and a table a capture read must outlive it
        tables = self._rotary_cache.get(device)
        if tables is None:
            c = self.config
            tables = self._rotary_cache[device] = qwen_rotary_tables(
                torch.arange(c.max_sequence_length, device=device), c.head_dim, c.rope_theta,
            )
        return tables[0][:seq], tables[1][:seq]

    def _vl_inputs(self, input_ids: torch.Tensor, images: VisionTokens):
        """The embeddings, rotary tables and deepstack rows of a one-prompt batch with images."""
        c = self.config
        if c.image_token_id is None:
            raise ValueError("this text encoder's checkpoint takes no images")
        if input_ids.shape[0] != 1:
            raise ValueError(f"a prompt with images runs alone, got a batch of {input_ids.shape[0]}")
        ids = input_ids[0]
        embeds = self.embed_tokens(ids)
        rows = (ids == c.image_token_id).nonzero().squeeze(-1)
        if rows.numel() != images.embeds.shape[0]:
            raise ValueError(f"{rows.numel()} image pads for {images.embeds.shape[0]} image tokens")
        embeds = embeds.index_copy(0, rows, images.embeds.to(embeds.dtype))
        positions = vl_positions(ids, c.image_token_id, images.grids, images.merge)
        # the frequencies computed on the CPU, as the reference does at init (not a buffer: the
        # encoder is built on the meta device and materialized uninitialized)
        inv_freq = qwen_inv_freq(c.head_dim, c.rope_theta).to(ids.device)
        rotary = compute_3d_cos_sin(positions, inv_freq, list(c.mrope_section), interleaved=c.mrope_interleaved)
        return embeds[None], rotary, ((rows, list(images.deepstack)) if images.deepstack else None)
