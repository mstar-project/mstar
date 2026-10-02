"""AuT: the Qwen3-family audio encoder (Qwen3-Omni, Qwen3-ASR), natively.

A Whisper-style log-mel input (128 bins, 100 frames/s) is cut into 1 s
chunks of ``2 * n_window`` frames, each chunk runs through three stride-2
``Conv2d`` layers (100 frames -> 13 tokens, so ~13 tokens/s) and a linear
``conv_out``, gets the same 13 sinusoidal positions, and the valid tokens of
every chunk are packed back into one ``[total_tokens, d_model]`` sequence.
The transformer stack then attends *within windows* of ``n_window_infer``
frames (8 s -> 104 tokens): block-diagonal, bidirectional, no cache. A final
LayerNorm and a two-layer GELU projector map the tokens into the LLM's width.

Windows are the attention's only structure, so the stack is packed varlen
attention: every window is a segment, and any number of requests can be
packed into one forward. On the engine it runs through the cacheless
``RaggedAttentionSpec`` resource, whose plan the runner builds outside the
graph — which is what lets the block loop replay as a CUDA graph. Without a
bound resource (CPU tests, standalone use) it falls back to SDPA per window.

Parameter paths mirror HF's ``audio_tower.*`` (``Qwen3OmniMoeAudioEncoder``,
which Qwen3-ASR reuses unchanged), with ``q/k/v_proj`` fused into
``qkv_proj`` by the loader's Whisper stacked rules.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.audio_features import sinusoid_positions
from mstar.model.components.linear import FusedColumnLinear


@dataclass
class AuTEncoderConfig:
    d_model: int = 1024
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    num_mel_bins: int = 128
    # half a conv chunk, in mel frames: a chunk is 2 * n_window frames (1 s)
    n_window: int = 50
    # attention window, in mel frames (8 s)
    n_window_infer: int = 800
    # conv chunks per convolution launch (an OOM guard on long audio)
    conv_chunksize: int = 500
    downsample_hidden_size: int = 480
    # the LLM's hidden size
    output_dim: int = 2048
    max_source_positions: int = 1500
    activation_function: str = "gelu"

    @classmethod
    def from_hf(cls, hf: dict) -> "AuTEncoderConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in hf.items() if k in names})

    @property
    def head_dim(self) -> int:
        return self.d_model // self.encoder_attention_heads

    @property
    def chunk_frames(self) -> int:
        return 2 * self.n_window

    @staticmethod
    def tokens_for_frames(num_frames: int) -> int:
        """Encoder tokens for ``num_frames`` mel frames: 13 per full 100-frame
        chunk, plus three stride-2 convolutions' worth of the remainder."""
        full, rem = divmod(num_frames, 100)
        rem_tokens = (((rem - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1 if rem else 0
        return full * 13 + rem_tokens

    @property
    def tokens_per_chunk(self) -> int:
        return self.tokens_for_frames(self.chunk_frames)

    @property
    def window_tokens(self) -> int:
        return self.tokens_per_chunk * (self.n_window_infer // self.chunk_frames)

    def window_lengths(self, num_tokens: int) -> list[int]:
        """Attention segments for one request's tokens: full windows, then the
        remainder."""
        full, rem = divmod(num_tokens, self.window_tokens)
        return [self.window_tokens] * full + ([rem] if rem else [])

    @property
    def conv_out_features(self) -> int:
        mel = self.num_mel_bins
        for _ in range(3):
            mel = (mel + 1) // 2
        return self.downsample_hidden_size * mel


def _activation(name: str):
    if name == "gelu":
        return F.gelu
    if name in ("silu", "swish"):
        return F.silu
    if name == "relu":
        return F.relu
    raise ValueError(f"unsupported activation {name!r}")


class AuTAttention(nn.Module):
    """Bidirectional attention within the packed windows of one forward.

    ``attn_key`` names the node's ``RaggedAttentionSpec`` resource; bound at
    load, it attends through the runner's plan. Unbound, SDPA runs per window
    from ``cu_seqlens``.
    """

    def __init__(self, d_model: int, num_heads: int, attn_key: str | None):
        super().__init__()
        if d_model % num_heads:
            raise ValueError(f"d_model {d_model} is not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self._attn_key = attn_key
        self.ragged = None
        self.qkv_proj = FusedColumnLinear(d_model, {"q": d_model, "k": d_model, "v": d_model}, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def bind_resources(self, resources: dict) -> None:
        self.ragged = None if self._attn_key is None else resources.get(self._attn_key)

    @staticmethod
    @torch.compiler.disable
    def _sdpa_per_window(q, k, v, cu_seqlens) -> torch.Tensor:
        out = torch.empty_like(q)
        bounds = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            if b <= a:
                continue
            o = F.scaled_dot_product_attention(
                q[a:b].transpose(0, 1), k[a:b].transpose(0, 1), v[a:b].transpose(0, 1),
            )
            out[a:b] = o.transpose(0, 1)
        return out

    def forward(self, hidden_states: torch.Tensor, cu_seqlens) -> torch.Tensor:
        n, d_model = hidden_states.shape
        q, k, v = self.qkv_proj(hidden_states).split(d_model, dim=-1)
        q = q.view(n, self.num_heads, self.head_dim)
        k = k.view(n, self.num_heads, self.head_dim)
        v = v.view(n, self.num_heads, self.head_dim)
        if self.ragged is not None:
            out = self.ragged.run(q, k, v)
        else:
            out = self._sdpa_per_window(q, k, v, cu_seqlens)
        return self.out_proj(out.reshape(n, d_model))


class AuTEncoderLayer(nn.Module):
    def __init__(self, config: AuTEncoderConfig, attn_key: str | None):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn = AuTAttention(config.d_model, config.encoder_attention_heads, attn_key)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.act = _activation(config.activation_function)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.self_attn_layer_norm(hidden_states), cu_seqlens)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.fc2(self.act(self.fc1(self.final_layer_norm(hidden_states))))
        return residual + hidden_states


@dataclass
class AuTLayout:
    """Where one packed forward's tokens come from and how they attend."""
    # tokens per request, in batch order
    tokens_per_request: list[int]
    # attention windows over the packed sequence: [0, w1, w1 + w2, ...]
    cu_seqlens: list[int]

    @property
    def total_tokens(self) -> int:
        return self.cu_seqlens[-1]

    @property
    def window_lengths(self) -> list[int]:
        return [b - a for a, b in zip(self.cu_seqlens[:-1], self.cu_seqlens[1:], strict=True)]


class AuTEncoder(nn.Module):
    """Packed audio encoder; parameter paths mirror HF's ``audio_tower.*``."""

    def __init__(self, config: AuTEncoderConfig, attn_key: str | None = None):
        super().__init__()
        self.config = config
        hidden = config.downsample_hidden_size
        self.conv2d1 = nn.Conv2d(1, hidden, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(hidden, hidden, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(hidden, hidden, 3, 2, padding=1)
        self.conv_out = nn.Linear(config.conv_out_features, config.d_model, bias=False)
        self.register_buffer(
            "positional_embedding",
            sinusoid_positions(config.max_source_positions, config.d_model),
            persistent=False,
        )
        self.layers = nn.ModuleList(
            [AuTEncoderLayer(config, attn_key) for _ in range(config.encoder_layers)]
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = _activation(config.activation_function)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    # -- loading -----------------------------------------------------------

    def reset_buffers(self) -> None:
        """Recompute the (non-persistent) sinusoid table. Needed after a
        meta-device construction + ``to_empty``, which leaves buffers
        uninitialized; ``load_weights`` calls it."""
        cfg = self.config
        table = sinusoid_positions(cfg.max_source_positions, cfg.d_model)
        self.positional_embedding.copy_(table.to(self.positional_embedding.dtype))

    def load_weights(self, weights) -> set[str]:
        """Stream ``audio_tower.*``-relative ``(name, tensor)`` pairs in; raise
        if any parameter is left unloaded."""
        from mstar.model.loader import WHISPER_STACKED_PARAMS, load_hf_weights

        loaded = load_hf_weights(self, weights, stacked_params=WHISPER_STACKED_PARAMS)
        missing = sorted(set(dict(self.named_parameters())) - loaded)
        if missing:
            raise RuntimeError(
                f"AuT encoder checkpoint left {len(missing)} parameter(s) unloaded, e.g. {missing[:5]}"
            )
        self.reset_buffers()
        return loaded

    # -- layout ------------------------------------------------------------

    def layout(self, feature_lens: list[int]) -> AuTLayout:
        """Token counts and attention windows for a batch of mel lengths.
        Host-side arithmetic only, so a submodule can declare its step from it."""
        tokens = [self.config.tokens_for_frames(int(n)) for n in feature_lens]
        cu = [0]
        for n in tokens:
            for w in self.config.window_lengths(n):
                cu.append(cu[-1] + w)
        return AuTLayout(tokens_per_request=tokens, cu_seqlens=cu)

    # -- forward -----------------------------------------------------------

    def frontend(self, features: torch.Tensor, feature_lens: list[int]) -> torch.Tensor:
        """``(B, num_mel_bins, T_max)`` zero-padded mel + valid lengths ->
        packed ``(total_tokens, d_model)`` chunk embeddings with positions.

        Every request is cut into ``chunk_frames`` pieces; the last piece is
        zero-padded to a full chunk (as HF's ``pad_sequence`` does whenever a
        batch holds one full chunk), convolved, and trimmed back to its valid
        tokens.
        """
        cfg = self.config
        chunk = cfg.chunk_frames
        chunks: list[torch.Tensor] = []
        valid: list[int] = []
        for b, n_frames in enumerate(feature_lens):
            n_frames = int(n_frames)
            n_chunks = -(-n_frames // chunk)
            padded = F.pad(features[b, :, :n_frames], (0, n_chunks * chunk - n_frames))
            chunks.append(padded.view(cfg.num_mel_bins, n_chunks, chunk).permute(1, 0, 2))
            widths = [chunk] * (n_chunks - 1) + [n_frames - chunk * (n_chunks - 1)]
            valid.extend(cfg.tokens_for_frames(w) for w in widths)
        x = torch.cat(chunks, dim=0).unsqueeze(1)  # (N, 1, mel, chunk)

        embeds = []
        for piece in x.split(cfg.conv_chunksize, dim=0):
            piece = self.act(self.conv2d1(piece))
            piece = self.act(self.conv2d2(piece))
            piece = self.act(self.conv2d3(piece))
            embeds.append(piece)
        x = torch.cat(embeds, dim=0)  # (N, C, mel', t)
        n, c, f, t = x.shape
        x = self.conv_out(x.permute(0, 3, 1, 2).reshape(n, t, c * f))
        x = x + self.positional_embedding[:t].to(x.dtype)
        keep = torch.arange(t, device=x.device)[None, :] < torch.tensor(valid, device=x.device)[:, None]
        return x[keep]

    def encode(self, hidden_states: torch.Tensor, cu_seqlens) -> torch.Tensor:
        """The transformer stack and projector over a packed sequence:
        ``(total_tokens, d_model)`` -> ``(total_tokens, output_dim)``. This is
        the region a submodule captures."""
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens)
        hidden_states = self.ln_post(hidden_states)
        return self.proj2(self.act(self.proj1(hidden_states)))

    def forward(self, features: torch.Tensor, feature_lens: list[int]) -> tuple[torch.Tensor, AuTLayout]:
        layout = self.layout(feature_lens)
        hidden = self.frontend(features, feature_lens)
        cu = torch.tensor(layout.cu_seqlens, dtype=torch.int32, device=hidden.device)
        return self.encode(hidden, cu), layout
