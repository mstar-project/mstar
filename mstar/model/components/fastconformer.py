"""FastConformer speech encoder (NVIDIA Canary, Parakeet, Nemotron ASR,
Nemotron VoiceChat).

Three stride-2 convolutions cut the 10 ms mel frames 8x to one frame per 80 ms,
then each Conformer block runs half an FFN, relative-position self-attention, a
depthwise convolution and another half FFN. The position term is a dense float
bias on the attention scores, so attention over a clip runs through SDPA rather
than a paged kernel. Cache-aware streaming checkpoints differ only in causal
convolutions, a LayerNorm in the conv module and chunked attention; with no
right context they also run as a stream, attending in a bounded KV
(``FastConformerEncoder.blocks``).
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.fastconformer_kernels import (
    add_layer_norm,
    bias_relu_mask_,
    glu_depthwise_conv,
    use_triton,
)

if TYPE_CHECKING:
    from mstar.model.components.audio_features import NemoLogMel


@dataclass
class FastConformerConfig:
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 8
    intermediate_size: int = 4096
    conv_kernel_size: int = 9
    num_mel_bins: int = 128
    subsampling_factor: int = 8
    subsampling_channels: int = 256
    subsampling_kernel_size: int = 3
    subsampling_stride: int = 2
    attention_bias: bool = True
    convolution_bias: bool = True
    scale_input: bool = False
    # cache-aware streaming checkpoints
    causal: bool = False
    left_context: int = -1  # frames each chunk sees behind it; -1 is unlimited
    lookahead: int = 0  # frames past a chunk's first frame, so a chunk is lookahead + 1 frames

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_subsampling_layers(self) -> int:
        return int(math.log2(self.subsampling_factor))

    @classmethod
    def from_hf(cls, enc: dict, lookahead: int | None = None) -> FastConformerConfig:
        """From an HF ``encoder_config`` (``parakeet_encoder`` or
        ``nemotron_asr_streaming_encoder``)."""
        causal = enc.get("model_type") == "nemotron_asr_streaming_encoder"
        return cls(
            hidden_size=enc["hidden_size"],
            num_layers=enc["num_hidden_layers"],
            num_heads=enc["num_attention_heads"],
            intermediate_size=enc["intermediate_size"],
            conv_kernel_size=enc["conv_kernel_size"],
            num_mel_bins=enc["num_mel_bins"],
            subsampling_factor=enc["subsampling_factor"],
            subsampling_channels=enc["subsampling_conv_channels"],
            subsampling_kernel_size=enc["subsampling_conv_kernel_size"],
            subsampling_stride=enc["subsampling_conv_stride"],
            attention_bias=enc.get("attention_bias", True),
            convolution_bias=enc.get("convolution_bias", True),
            scale_input=enc.get("scale_input", False),
            causal=causal,
            left_context=enc["sliding_window"] - 1 if causal else -1,
            lookahead=(
                enc.get("default_num_lookahead_tokens", 0) if lookahead is None else lookahead
            ) if causal else 0,
        )

    @classmethod
    def from_nemo(cls, enc: dict) -> FastConformerConfig:
        """From a NeMo ``ConformerEncoder`` config (``perception.encoder``, as
        Canary-Qwen and Nemotron VoiceChat ship it), NeMo's defaults for the keys
        it leaves out. NeMo sets causality per part; this encoder is causal or
        not as a whole, so a checkpoint whose parts disagree is refused."""
        if enc.get("self_attention_model", "rel_pos") != "rel_pos" or enc.get("subsampling") != "dw_striding":
            raise ValueError("only rel_pos / dw_striding FastConformer encoders are supported")
        k = enc["conv_kernel_size"]
        causal = bool(enc.get("causal_downsampling", False))
        context = list(enc.get("att_context_size") or [-1, -1])
        if len(context) != 2 or not all(isinstance(c, int) for c in context):
            raise ValueError(f"att_context_size must be one [left, right] pair; got {context}")
        left, right = context
        conv_context = enc.get("conv_context_size")
        if conv_context is None or conv_context == "causal":
            conv_context = [k - 1, 0] if conv_context else [(k - 1) // 2] * 2
        wrong = {}
        if list(conv_context) != ([k - 1, 0] if causal else [(k - 1) // 2] * 2):
            wrong["conv_context_size"] = enc.get("conv_context_size")
        if enc.get("conv_norm_type", "batch_norm") != ("layer_norm" if causal else "batch_norm"):
            wrong["conv_norm_type"] = enc.get("conv_norm_type", "batch_norm")
        style = enc.get("att_context_style", "regular")
        # with no right context, regular and chunked_limited attention mask alike
        if causal and not (style == "chunked_limited" or (style == "regular" and right == 0)):
            wrong["att_context_style"] = style
        if (causal and right < 0) or (not causal and context != [-1, -1]):
            # a causal encoder's chunk is right + 1 frames
            wrong["att_context_size"] = context
        if not enc.get("untie_biases", True):
            wrong["untie_biases"] = False
        if wrong:
            raise ValueError(f"unsupported FastConformer encoder settings {wrong} with causal_downsampling={causal}")
        use_bias = enc.get("use_bias", True)
        return cls(
            hidden_size=enc["d_model"],
            num_layers=enc["n_layers"],
            num_heads=enc["n_heads"],
            intermediate_size=enc["d_model"] * enc["ff_expansion_factor"],
            conv_kernel_size=k,
            num_mel_bins=enc["feat_in"],
            subsampling_factor=enc["subsampling_factor"],
            subsampling_channels=enc["subsampling_conv_channels"],
            attention_bias=use_bias,
            convolution_bias=use_bias,
            # NeMo scales the input unless told not to
            scale_input=enc.get("xscaling", True),
            causal=causal,
            left_context=left if causal else -1,
            lookahead=right if causal else 0,
        )

    def output_lengths(self, lengths):
        """Encoder frames for ``lengths`` mel frames (an int or a tensor)."""
        for _ in range(self.num_subsampling_layers):
            lengths = _conv_out_len(lengths, self.subsampling_kernel_size, self.subsampling_stride, self.causal)
        return lengths


def _conv_out_len(lengths, kernel: int, stride: int, causal: bool):
    """Frames out of one strided conv, for an int or a tensor of lengths."""
    # symmetric padding (k-1)//2 per side offline; (k-1, s-1) causal
    pad = (kernel - 1) + (stride - 1) if causal else (kernel - 1) // 2 * 2
    return (lengths + pad - kernel) // stride + 1


def _time_mask(lengths: torch.Tensor | None, max_len: int) -> torch.Tensor | None:
    if lengths is None:
        return None
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class _Subsampling(nn.Module):
    """Depthwise-separable striding over (time, mel): a full conv into
    ``subsampling_channels``, then depthwise + pointwise pairs, then a linear
    from the flattened channels x remaining mel bins.

    ``layers`` keeps the checkpoint's ModuleList layout, ReLUs included, so
    parameter names match. Padded frames are zeroed after every conv a ReLU
    follows, so a batch gives each request what it would get alone; a
    depthwise conv feeds a 1x1 conv, which keeps padding in its own frames,
    so that conv's mask covers both.
    """

    def __init__(self, config: FastConformerConfig):
        super().__init__()
        self.config = config
        k, s, ch = config.subsampling_kernel_size, config.subsampling_stride, config.subsampling_channels
        pad = 0 if config.causal else (k - 1) // 2
        layers: list[nn.Module] = [nn.Conv2d(1, ch, k, stride=s, padding=pad), nn.ReLU()]
        for _ in range(config.num_subsampling_layers - 1):
            layers += [
                nn.Conv2d(ch, ch, k, stride=s, padding=pad, groups=ch),
                nn.Conv2d(ch, ch, kernel_size=1),
                nn.ReLU(),
            ]
        self.layers = nn.ModuleList(layers)
        freq = config.num_mel_bins
        for _ in range(config.num_subsampling_layers):
            freq = _conv_out_len(freq, k, s, config.causal)
        self.linear = nn.Linear(ch * freq, config.hidden_size)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """``(B, T, mel)`` -> ``(B, T', hidden)``."""
        k, s = self.config.subsampling_kernel_size, self.config.subsampling_stride
        x = features.unsqueeze(1)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.ReLU):
                continue  # applied with the conv before it
            if self.config.causal and layer.stride[0] > 1:
                # causal: k-1 frames of history plus s-1 on the right, on both axes
                x = F.pad(x, (k - 1, s - 1, k - 1, s - 1))
            if lengths is not None and layer.stride[0] > 1:
                lengths = _conv_out_len(lengths, k, s, self.config.causal)
            if not isinstance(self.layers[i + 1], nn.ReLU):
                x = layer(x)
            elif use_triton(x):
                x = bias_relu_mask_(
                    F.conv2d(x, layer.weight, None, layer.stride, layer.padding, layer.dilation, layer.groups),
                    layer.bias, lengths,
                )
            else:
                x = layer(x)
                if lengths is not None:
                    x = x * _time_mask(lengths, x.shape[2])[:, None, :, None]
                x = torch.relu(x)
        b, c, t, f = x.shape
        return self.linear(x.transpose(1, 2).reshape(b, t, c * f))


def _streaming_subsampling_name(name: str) -> str:
    """HF's streaming checkpoints name the subsampling convs ``conv_in`` and
    ``layers.N.depthwise_conv`` / ``pointwise_conv``; map those onto the flat
    layout, which NeMo's names already follow."""
    if name.startswith("conv_in."):
        return "layers.0." + name.removeprefix("conv_in.")
    match = re.fullmatch(r"layers\.(\d+)\.(depthwise_conv|pointwise_conv)\.(.+)", name)
    if match is None:
        return name
    idx, kind, rest = match.groups()
    return f"layers.{2 + 3 * int(idx) + (kind == 'pointwise_conv')}.{rest}"


def rel_positional_encoding(seq_len: int, hidden_size: int, device, dtype: torch.dtype) -> torch.Tensor:
    """Transformer-XL sinusoids for distances ``T-1 .. -(T-1)``, sin and cos
    interleaved per frequency. Built in fp32 each call: a buffer would be
    rounded by the engine's dtype cast."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hidden_size, 2, device=device, dtype=torch.float32) / hidden_size))
    positions = torch.arange(seq_len - 1, -seq_len, -1, device=device, dtype=torch.float32)
    freqs = positions[:, None] * inv_freq
    return torch.stack([freqs.sin(), freqs.cos()], dim=-1).flatten(1).to(dtype)


class RelPositionAttention(nn.Module):
    """``softmax(((q + u) k^T + skew((q + v) R^T)) / sqrt(d))`` with ``R``
    the projected position sinusoids; the second term enters SDPA as a bias.
    A stream step attends its new frames in the bounded KV ``kv_key`` names
    instead, which keeps each stream's window of keys and values, with the
    distances it can see projected once, at bind (``band``)."""

    def __init__(self, config: FastConformerConfig, layer_idx: int = 0, kv_key: str | None = None):
        super().__init__()
        h, d = config.num_heads, config.head_dim
        self.num_heads, self.head_dim = h, d
        self.layer_idx, self._kv_key = layer_idx, kv_key
        self.left_context = config.left_context
        self.kv = None
        self.band = None
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * h * d, bias=config.attention_bias)
        self.o_proj = nn.Linear(h * d, config.hidden_size, bias=config.attention_bias)
        self.relative_k_proj = nn.Linear(config.hidden_size, h * d, bias=False)
        self.bias_u = nn.Parameter(torch.zeros(h, d))
        self.bias_v = nn.Parameter(torch.zeros(h, d))

    def bind_resources(self, resources: dict) -> None:
        if self._kv_key is not None:
            self.kv = resources[self._kv_key]
            # distances 0..left_context, all a stream's frame sees (row d is
            # distance d), projected in the weights' dtype, which is what autocast
            # computes it in: fixed weights, so once, here, before any capture
            w = self.relative_k_proj.weight
            n = self.left_context + 1
            with torch.no_grad():
                band = rel_positional_encoding(n, w.shape[1], w.device, w.dtype)[:n].flip(0)
                self.band = self.relative_k_proj(band).view(n, self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor | None, mask: torch.Tensor | None, stream: bool = False,
    ) -> torch.Tensor:
        """``x`` is ``(B, T, hidden)``, ``pos_emb`` ``(P, hidden)`` with
        ``P >= 2T-1`` (rows past ``2T-1`` are zero padding) and ``mask``
        ``(B, 1, T, T)`` with True where a query may attend; or ``stream``, a
        stream step, which takes neither."""
        b, t, _ = x.shape
        h, d = self.num_heads, self.head_dim
        q, k, v = self.qkv_proj(x).view(b, t, 3, h, d).unbind(2)  # (B, T, H, D) each
        if stream:
            return self._attend_stream(q, k, v)
        r = self.relative_k_proj(pos_emb).view(-1, h, d).permute(1, 2, 0)  # (H, D, P)
        # one GEMM per head over all B*T queries: no copy of r per row
        bd = torch.bmm((q + self.bias_v).reshape(b * t, h, d).transpose(0, 1), r)  # (H, B*T, P)
        # column T-1-i+j holds distance i-j for query i, key j
        p = bd.shape[-1]
        bd = bd.as_strided((b, h, t, t), (t * p, b * t * p, p - 1, 1), storage_offset=bd.storage_offset() + t - 1)
        bias = torch.mul(bd, self.head_dim ** -0.5, out=torch.empty(b, h, t, t, dtype=bd.dtype, device=bd.device))
        # (H, B*T, P) is the largest tensor in the encoder: let it go
        del bd
        if mask is not None:
            bias.masked_fill_(~mask, torch.finfo(bias.dtype).min)
        q, k, v = (q + self.bias_u).transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, h * d))

    def _attend_stream(self, q, k, v) -> torch.Tensor:
        """Each new frame attends the ``left_context`` frames before it and
        itself, the older ones from the KV, which then keeps this step's. The
        position term depends on the distance alone, so it goes in as the band
        of distances, unscaled: the kernel scales the sum."""
        b, t, h, d = q.shape
        r = self.band  # (H, left + 1, D)
        # in k's dtype: under autocast the bias add promotes q, and the kernel takes one dtype
        q_u = (q + self.bias_u).to(k.dtype).transpose(1, 2)
        bias = torch.matmul((q + self.bias_v).transpose(1, 2), r.transpose(-2, -1))  # (B, H, T, left + 1)
        out = self.kv.attend(self.layer_idx, q_u, k.transpose(1, 2), v.transpose(1, 2), rel_bias=bias,
                             ieee=k.dtype == torch.float32)
        return self.o_proj(out.reshape(b, t, h * d))


class ConvModule(nn.Module):
    """Pointwise to 2x channels, GLU, depthwise conv over time, norm, SiLU,
    pointwise. The offline checkpoints' BatchNorm is folded into the depthwise
    conv at load (it is a per-channel affine at inference); the streaming ones
    use a LayerNorm, which stays. A stream step's causal conv continues each
    stream's last ``k - 1`` inputs, kept in the pool ``conv_key`` names."""

    def __init__(self, config: FastConformerConfig, layer_idx: int = 0, conv_key: str | None = None):
        super().__init__()
        c, k = config.hidden_size, config.conv_kernel_size
        self.causal = config.causal
        self.kernel_size = k
        self.layer_idx, self._conv_key = layer_idx, conv_key
        self.state = None
        self.pointwise_conv1 = nn.Linear(c, 2 * c, bias=config.convolution_bias)
        # bias always on: the folded BatchNorm shift lands here
        self.depthwise_conv = nn.Conv1d(c, c, k, groups=c, bias=True, padding=0 if self.causal else (k - 1) // 2)
        self.norm = nn.LayerNorm(c) if self.causal else None
        self.pointwise_conv2 = nn.Linear(c, c, bias=config.convolution_bias)

    def bind_resources(self, resources: dict) -> None:
        if self._conv_key is not None:
            self.state = resources[self._conv_key]

    def forward(
        self, x: torch.Tensor, valid: torch.Tensor | None, tails: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``tails`` marks a stream step: per row, where its newest ``k - 1``
        conv inputs sit in the cached ones followed by the step's (``blocks``)."""
        x = self.pointwise_conv1(x)
        if tails is not None:
            return self._stream(x, tails)
        if use_triton(x):
            conv = self.depthwise_conv
            left_pad = self.kernel_size - 1 if self.causal else conv.padding[0]
            x = glu_depthwise_conv(x, valid, conv.weight, conv.bias, left_pad, silu=self.norm is None)
            return self.pointwise_conv2(x if self.norm is None else F.silu(self.norm(x)))
        x = F.glu(x, dim=-1)
        if valid is not None:
            # padding must not reach real frames through the kernel's window
            x = x.masked_fill(~valid[:, :, None], 0.0)
        x = x.transpose(1, 2)
        if self.causal:
            x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.depthwise_conv(x).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return self.pointwise_conv2(F.silu(x))

    def _stream(self, x: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        # eager: the Triton kernel takes no cached inputs
        cached = self.state.gather("conv", self.layer_idx)  # (B, C, k - 1)
        x = torch.cat([cached.to(x.dtype), F.glu(x, dim=-1).transpose(1, 2)], dim=2)
        self.state.scatter_("conv", x.gather(2, tails).to(cached.dtype), self.layer_idx)
        x = self.depthwise_conv(x).transpose(1, 2)
        return self.pointwise_conv2(F.silu(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, config: FastConformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.silu(self.linear1(x)))


class FastConformerBlock(nn.Module):
    def __init__(
        self, config: FastConformerConfig, layer_idx: int = 0, kv_key: str | None = None, conv_key: str | None = None,
    ):
        super().__init__()
        c = config.hidden_size
        self.norm_feed_forward1 = nn.LayerNorm(c)
        self.feed_forward1 = FeedForward(config)
        self.norm_self_att = nn.LayerNorm(c)
        self.self_attn = RelPositionAttention(config, layer_idx, kv_key)
        self.norm_conv = nn.LayerNorm(c)
        self.conv = ConvModule(config, layer_idx, conv_key)
        self.norm_feed_forward2 = nn.LayerNorm(c)
        self.feed_forward2 = FeedForward(config)
        self.norm_out = nn.LayerNorm(c)

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor | None, mask: torch.Tensor | None, valid: torch.Tensor | None,
        tails: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Offline: ``pos_emb``, ``mask`` and ``valid``. A stream step: ``tails``
        (see ``ConvModule.forward``), which streams the attention as well."""
        stream = tails is not None
        if use_triton(x):
            # each residual add runs in the kernel of the norm that reads it
            x, h = add_layer_norm(x, self.feed_forward1(self.norm_feed_forward1(x)), 0.5, self.norm_self_att)
            x, h = add_layer_norm(x, self.self_attn(h, pos_emb, mask, stream), 1.0, self.norm_conv)
            x, h = add_layer_norm(x, self.conv(h, valid, tails), 1.0, self.norm_feed_forward2)
            return add_layer_norm(x, self.feed_forward2(h), 0.5, self.norm_out, keep_sum=False)[1]
        # alpha folds the half step into the add (0.5 * y is exact either way)
        x = torch.add(x, self.feed_forward1(self.norm_feed_forward1(x)), alpha=0.5)
        x = x + self.self_attn(self.norm_self_att(x), pos_emb, mask, stream)
        x = x + self.conv(self.norm_conv(x), valid, tails)
        x = torch.add(x, self.feed_forward2(self.norm_feed_forward2(x)), alpha=0.5)
        return self.norm_out(x)


def chunked_attention_mask(seq_len: int, left_context: int, lookahead: int, device) -> torch.Tensor:
    """``(T, T)``: a query sees its own chunk and ``left_context // chunk``
    chunks behind it, chunk = ``lookahead + 1`` frames."""
    chunk = lookahead + 1
    idx = torch.arange(seq_len, device=device) // chunk
    diff = idx[:, None] - idx[None, :]
    left_chunks = left_context // chunk if left_context >= 0 else seq_len
    return (diff >= 0) & (diff <= left_chunks)


_NEMO_NAMES = (
    ("pre_encode.conv.", "subsampling.layers."),
    ("pre_encode.out.", "subsampling.linear."),
    (".self_attn.linear_q.", ".self_attn.q_proj."),
    (".self_attn.linear_k.", ".self_attn.k_proj."),
    (".self_attn.linear_v.", ".self_attn.v_proj."),
    (".self_attn.linear_out.", ".self_attn.o_proj."),
    (".self_attn.linear_pos.", ".self_attn.relative_k_proj."),
    (".self_attn.pos_bias_u", ".self_attn.bias_u"),
    (".self_attn.pos_bias_v", ".self_attn.bias_v"),
    (".conv.batch_norm.", ".conv.norm."),
)


def nemo_to_hf_name(name: str) -> str:
    """A NeMo ``ConformerEncoder`` parameter name in HF's naming, for
    checkpoints that ship the encoder as NeMo saved it (Canary-Qwen)."""
    for nemo, hf in _NEMO_NAMES:
        name = name.replace(nemo, hf)
    return name


class FastConformerEncoder(nn.Module):
    """``(B, T, mel)`` features and per-request frame counts in,
    ``(B, T', hidden)`` and per-request encoder lengths out: ``subsample``
    then ``blocks``, the split NeMo's ``bypass_pre_encode`` makes.

    A causal encoder also runs as a stream, a few frames at a time: ``blocks``
    with ``new`` attends in the bounded KV ``kv_key`` names (each frame over the
    ``left_context`` before it) and continues each conv from the recurrent-state
    pool ``conv_key`` names, which hold a stream's state between steps."""

    def __init__(self, config: FastConformerConfig, kv_key: str | None = None, conv_key: str | None = None):
        super().__init__()
        self.config = config
        self.kv_key, self.conv_key = kv_key, conv_key
        self.subsampling = _Subsampling(config)
        self.layers = nn.ModuleList(
            FastConformerBlock(config, i, kv_key, conv_key) for i in range(config.num_layers)
        )

    @classmethod
    def load(
        cls,
        config: FastConformerConfig,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        kv_key: str | None = None,
        conv_key: str | None = None,
    ) -> FastConformerEncoder:
        """Built on ``device`` in ``dtype`` from HF-named ``weights``
        (``load_hf_weights``), every parameter loaded, in eval mode."""
        from mstar.model.utils import check_loaded

        with torch.device("meta"):
            encoder = cls(config, kv_key, conv_key)
        # cast first, so storage is allocated in the target dtype
        if dtype is not None:
            encoder = encoder.to(dtype)
        encoder = encoder.to_empty(device=device)
        check_loaded(encoder, encoder.load_hf_weights(weights), "FastConformer encoder")
        return encoder.eval()

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``lengths`` (mel frames per row) masks the padding; leave it out
        when no row is padded. No host reads, so the forward captures."""
        x = self.subsample(features, lengths)
        if lengths is None:
            return self.blocks(x), torch.full((x.shape[0],), x.shape[1], device=x.device)
        out_lengths = self.config.output_lengths(lengths)
        return self.blocks(x, out_lengths), out_lengths

    def subsample(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, T, mel)`` -> ``(B, T', hidden)`` block inputs; ``lengths`` as ``forward``'s."""
        x = self.subsampling(features, lengths)
        if self.config.scale_input:
            x = x * math.sqrt(self.config.hidden_size)
        return x

    def blocks(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None, new: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The Conformer blocks over ``x (B, T', hidden)``: offline over the
        clip, ``lengths`` (encoder frames per row) masking the padding; or, given
        ``new`` (a device ``(B,)`` count of each stream's new frames, ``<= T'``),
        one stream step under the plans its declaration made for the bound KV
        and conv pool. A stream with no new frame keeps its conv state."""
        if new is not None:
            return self._stream_blocks(x, new)
        t = x.shape[1]
        valid = _time_mask(lengths, t)
        mask = None
        if self.config.causal:
            mask = chunked_attention_mask(t, self.config.left_context, self.config.lookahead, x.device)[None, None]
        if valid is not None:
            keys = valid[:, None, None, :]
            mask = keys if mask is None else mask & keys
        pos_emb = rel_positional_encoding(t, self.config.hidden_size, x.device, x.dtype)
        # zero rows to a multiple of 8 distances keep the position GEMM's output
        # rows 16-byte aligned, which the fast GEMM kernels need
        pos_emb = F.pad(pos_emb, (0, 0, 0, -pos_emb.shape[0] % 8))
        for layer in self.layers:
            x = layer(x, pos_emb, mask, valid)
        return x

    def _stream_blocks(self, x: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], self.config
        keep = c.conv_kernel_size - 1
        # each stream keeps its newest k - 1 conv inputs: new .. new + k - 1 of cached ++ this step's
        tails = (new.view(b, 1, 1) + torch.arange(keep, device=x.device).view(1, 1, keep)).expand(
            b, c.hidden_size, keep)
        for layer in self.layers:
            x = layer(x, None, None, None, tails=tails)
        return x

    # ------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------

    def load_hf_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load HF-named weights (``subsampling.*``, ``layers.N.*``; strip any
        outer prefix first): Q/K/V fuse into ``qkv_proj``, 1x1 convs become
        linears, an offline checkpoint's BatchNorm folds into the depthwise
        conv in fp32. Returns the names loaded."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        qkv: dict[int, dict[str, torch.Tensor]] = {}
        conv: dict[int, dict[str, torch.Tensor]] = {}

        def put(name: str, tensor: torch.Tensor) -> None:
            if name not in params:
                raise KeyError(f"unexpected FastConformer weight {name!r}")
            target = params[name]
            if tensor.shape != target.shape:
                raise ValueError(f"{name}: checkpoint {tuple(tensor.shape)} vs model {tuple(target.shape)}")
            with torch.no_grad():
                target.copy_(tensor)
            loaded.add(name)

        for name, tensor in weights:
            if name.endswith("num_batches_tracked"):
                continue
            if name.startswith("subsampling.") and self.config.causal:
                name = "subsampling." + _streaming_subsampling_name(name.removeprefix("subsampling."))
            parts = name.split(".")
            if parts[0] == "layers" and parts[2] == "self_attn" and parts[3] in ("q_proj", "k_proj", "v_proj"):
                qkv.setdefault(int(parts[1]), {})[f"{parts[3][0]}.{parts[4]}"] = tensor
                continue
            if parts[0] == "layers" and parts[2] == "conv" and parts[3] in ("depthwise_conv", "norm"):
                conv.setdefault(int(parts[1]), {})[f"{parts[3]}.{parts[4]}"] = tensor
                continue
            if parts[0] == "layers" and parts[2] == "conv" and parts[3].startswith("pointwise_conv"):
                tensor = tensor.squeeze(-1)
            put(name, tensor)

        for idx, shards in qkv.items():
            prefix = f"layers.{idx}.self_attn.qkv_proj"
            put(f"{prefix}.weight", torch.cat([shards[f"{s}.weight"] for s in "qkv"]))
            if "q.bias" in shards:
                put(f"{prefix}.bias", torch.cat([shards[f"{s}.bias"] for s in "qkv"]))

        for idx, found in conv.items():
            prefix = f"layers.{idx}.conv"
            weight = found["depthwise_conv.weight"].float()
            bias = found.get("depthwise_conv.bias")
            bias = torch.zeros(weight.shape[0], device=weight.device) if bias is None else bias.float()
            if self.config.causal:
                put(f"{prefix}.norm.weight", found["norm.weight"])
                put(f"{prefix}.norm.bias", found["norm.bias"])
            else:
                mean, var, gamma, beta = (
                    found[f"norm.{k}"].float() for k in ("running_mean", "running_var", "weight", "bias")
                )
                scale = gamma / torch.sqrt(var + 1e-5)
                weight = weight * scale[:, None, None]
                bias = (bias - mean) * scale + beta
            put(f"{prefix}.depthwise_conv.weight", weight)
            put(f"{prefix}.depthwise_conv.bias", bias)
        return loaded


def load_nemo_perception(
    config: FastConformerConfig,
    out_dim: int,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    kv_key: str | None = None,
    conv_key: str | None = None,
    **log_mel_settings: Any,
) -> tuple[FastConformerEncoder, nn.Linear, NemoLogMel]:
    """A NeMo ``AudioPerceptionModule`` (Canary-Qwen's ``perception.``,
    Nemotron VoiceChat's ``stt_model.perception.``) from its weights, named
    without that prefix: the encoder, built in ``dtype`` and bound to
    ``kv_key`` / ``conv_key`` when it streams; ``proj``, its projection into
    ``out_dim``; and the featurizer with ``log_mel_settings`` and the
    checkpoint's own (rounded) filters and window."""
    from mstar.model.components.audio_features import NemoLogMel

    with torch.device("meta"):
        proj = nn.Linear(config.hidden_size, out_dim)
    proj = (proj if dtype is None else proj.to(dtype)).to_empty(device=device)
    featurizer: dict[str, torch.Tensor] = {}
    head: dict[str, torch.Tensor] = {}

    def encoder_weights() -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if name.startswith("encoder."):
                yield nemo_to_hf_name(name.removeprefix("encoder.")), tensor
            elif name.startswith("preprocessor.featurizer."):
                featurizer[name.removeprefix("preprocessor.featurizer.")] = tensor
            elif name.startswith("proj."):
                head[name.removeprefix("proj.")] = tensor
            else:
                # e.g. a modality adapter's: skipping it would serve without it
                raise ValueError(f"unexpected NeMo perception weight {name!r}")

    encoder = FastConformerEncoder.load(
        config, encoder_weights(), device=device, dtype=dtype, kv_key=kv_key, conv_key=conv_key,
    )
    with torch.no_grad():
        proj.weight.copy_(head["weight"])
        proj.bias.copy_(head["bias"])
    log_mel = NemoLogMel(
        num_mel_bins=config.num_mel_bins, device=device,
        mel_filters=featurizer["fb"], window=featurizer["window"], **log_mel_settings,
    )
    return encoder, proj.eval(), log_mel
