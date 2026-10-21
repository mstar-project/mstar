# Copyright (c) Fish Audio
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M*: adapted from the fish-speech 0.1.0 PyPI
# release (https://pypi.org/project/fish-speech/0.1.0/, wheel SHA-256
# 1ea6123c7b9a441bc48f296439ae0e987621b780f9314ff3eb5ad072ebe9e7ac,
# ``fish_speech/models/dac/modded_dac.py`` and ``rvq.py``) through the
# inference-only copy vendored by vLLM-Omni (Apache-2.0,
# ``vllm_omni/model_executor/models/fish_speech/dac_modules`` at f69b1f2b1).
#
# Original file was released under the Apache License 2.0, with the full
# license text available at http://www.apache.org/licenses/LICENSE-2.0.
#
# This modified file is released under the same license.
"""Fish Audio "modded DAC" codec: 10 codebooks at 21.5 Hz <-> 44.1 kHz audio.

Decode path: per-codebook embeddings -> the quantizer's 8-layer causal
transformer (128-frame attention window, RoPE) -> 2 causal upsampling stages
-> the causal conv decoder (x512). Every stage is causal and length
preserving, so W frames decode to exactly ``W * 2048`` samples.

Changes from the reference, none of which change the math:

* inference only: no training losses, dropout or ``audiotools`` bases;
* weight norm is folded into plain conv weights at load time (the same
  ``torch._weight_norm`` the parametrization evaluates per call);
* the transformers take explicit frame positions, so a window that starts
  ``offset`` frames into a stream rotates its frames exactly as one decode of
  the whole stream does (the RoPE table is bfloat16, so attention is not
  shift invariant);
* ``FishDACDecoder.decode_window`` decodes the conv stack over a slice of the
  window only (its new frames and their causal receptive field).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.fish_s2_pro.components.dac_layers import ResidualVectorQuantize, Snake1d
from mstar.model.fish_s2_pro.config import FishDACConfig

# ---------------------------------------------------------------------------
# Causal convolutions
# ---------------------------------------------------------------------------


def _extra_padding_for_conv1d(length: int, kernel_size: int, stride: int, padding_total: int) -> int:
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


class CausalConvNet(nn.Module):
    """Left-padded ``Conv1d``; right-padded only to complete the last stride."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, stride: int = 1, groups: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - self.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extra = _extra_padding_for_conv1d(x.shape[-1], self.kernel_size, self.stride, self.padding)
        return self.conv(F.pad(x, (self.padding, extra)))


class CausalTransConvNet(nn.Module):
    """``ConvTranspose1d`` trimmed on the right to ``length * stride`` samples."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        trim = self.kernel_size - self.stride
        return x[..., : x.shape[-1] - trim]


class ConvNeXtBlock(nn.Module):
    """Causal depthwise conv -> LayerNorm -> MLP -> layer scale, plus residual."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, kernel_size: int = 7):
        super().__init__()
        self.dwconv = CausalConvNet(dim, dim, kernel_size=kernel_size, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwconv(x).permute(0, 2, 1)
        y = self.pwconv2(self.act(self.pwconv1(self.norm(y))))
        return x + (self.gamma * y).permute(0, 2, 1)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(dim),
            CausalConvNet(dim, dim, kernel_size=7, dilation=dilation),
            Snake1d(dim),
            CausalConvNet(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        trim = x.shape[-1] - y.shape[-1]
        if trim > 0:
            x = x[..., :-trim]
        return x + y


# ---------------------------------------------------------------------------
# Window-limited causal transformer (quantizer pre/post module, encoder)
# ---------------------------------------------------------------------------


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000) -> torch.Tensor:
    """The reference's bfloat16 ``[seq_len, n_elem // 2, 2]`` cos/sin table."""
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs_cis = torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))
    return torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1).to(torch.bfloat16)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE. ``x``: ``[B, T, H, D]``; ``freqs_cis``: ``[B, T, D/2, 2]``."""
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.unsqueeze(2)
    out = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    return out.flatten(3).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int, head_dim: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.wqkv = nn.Linear(dim, 3 * n_head * head_dim, bias=False)
        self.wo = nn.Linear(n_head * head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        size = self.n_head * self.head_dim
        q, k, v = self.wqkv(x).split([size, size, size], dim=-1)
        q = apply_rotary_emb(q.view(bsz, seqlen, self.n_head, self.head_dim), freqs_cis)
        k = apply_rotary_emb(k.view(bsz, seqlen, self.n_head, self.head_dim), freqs_cis)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
        )
        return self.wo(y.transpose(1, 2).reshape(bsz, seqlen, size))


class FeedForward(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_head: int, head_dim: int, intermediate_size: int, norm_eps: float):
        super().__init__()
        self.attention = Attention(dim, n_head, head_dim)
        self.feed_forward = FeedForward(dim, intermediate_size)
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.attention_layer_scale = LayerScale(dim)
        self.ffn_layer_scale = LayerScale(dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x + self.attention_layer_scale(self.attention(self.attention_norm(x), freqs_cis, mask))
        return h + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(h)))


class WindowLimitedTransformer(nn.Module):
    """Causal transformer whose frames attend to the last ``window`` frames.

    Channels first in and out (``[B, D, T]``). ``positions`` (``[B, T]``) are
    the frames' indices in their stream; ``None`` means ``0 .. T-1``.
    """

    def __init__(
        self,
        dim: int,
        n_layer: int,
        n_head: int,
        head_dim: int,
        intermediate_size: int,
        window: int,
        block_size: int,
        rope_base: float,
        norm_eps: float,
    ):
        super().__init__()
        self.window = window
        self.layers = nn.ModuleList(
            TransformerBlock(dim, n_head, head_dim, intermediate_size, norm_eps) for _ in range(n_layer)
        )
        self.norm = RMSNorm(dim, norm_eps)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(block_size, head_dim, rope_base),
            persistent=False,
        )

    @property
    def receptive_field(self) -> int:
        """Earlier frames an output frame depends on."""
        return len(self.layers) * (self.window - 1)

    def window_mask(self, length: int, device: torch.device) -> torch.Tensor:
        rows = torch.arange(length, device=device).view(-1, 1)
        cols = torch.arange(length, device=device).view(1, -1)
        return ((cols <= rows) & (cols > rows - self.window))[None, None]

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x.transpose(1, 2)
        bsz, length, _ = x.shape
        if positions is None:
            positions = torch.arange(length, device=x.device).expand(bsz, length)
        freqs_cis = self.freqs_cis[positions]
        mask = self.window_mask(length, x.device)
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        return self.norm(x).transpose(1, 2)


# ---------------------------------------------------------------------------
# Encoder / decoder conv stacks
# ---------------------------------------------------------------------------


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int, transformer: nn.Module | None):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            CausalConvNet(dim // 2, dim, kernel_size=2 * stride, stride=stride),
            transformer if transformer is not None else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, cfg: FishDACConfig):
        super().__init__()
        d_model = cfg.encoder_dim
        blocks: list[nn.Module] = [CausalConvNet(1, d_model, kernel_size=7)]
        for stride, n_layer in zip(cfg.encoder_rates, cfg.encoder_transformer_layers, strict=True):
            d_model *= 2
            transformer = None
            if n_layer:
                transformer = WindowLimitedTransformer(
                    d_model,
                    n_layer,
                    d_model // 64,
                    64,
                    d_model * 3,
                    cfg.encoder_window,
                    cfg.encoder_block_size,
                    cfg.rope_base,
                    cfg.norm_eps,
                )
            blocks.append(EncoderBlock(d_model, stride, transformer))
        blocks += [Snake1d(d_model), CausalConvNet(d_model, cfg.latent_dim, kernel_size=3)]
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            CausalTransConvNet(input_dim, output_dim, kernel_size=2 * stride, stride=stride),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, cfg: FishDACConfig):
        super().__init__()
        channels = cfg.decoder_dim
        layers: list[nn.Module] = [CausalConvNet(cfg.latent_dim, channels, kernel_size=7)]
        for i, stride in enumerate(cfg.decoder_rates):
            layers.append(DecoderBlock(channels // 2**i, channels // 2 ** (i + 1), stride))
        output_dim = channels // 2 ** len(cfg.decoder_rates)
        layers += [Snake1d(output_dim), CausalConvNet(output_dim, 1, kernel_size=7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Quantizer (1 semantic + 9 residual codebooks at 1/4 of the latent rate)
# ---------------------------------------------------------------------------


def _quantizer_transformer(cfg: FishDACConfig) -> WindowLimitedTransformer:
    return WindowLimitedTransformer(
        cfg.latent_dim,
        cfg.post_layers,
        cfg.post_heads,
        cfg.post_head_dim,
        cfg.post_intermediate_size,
        cfg.post_window,
        cfg.post_block_size,
        cfg.rope_base,
        cfg.norm_eps,
    )


class DownsampleResidualVectorQuantize(nn.Module):
    """The reference quantizer, built for one direction (``decode`` or not)."""

    def __init__(self, cfg: FishDACConfig, *, decode: bool):
        super().__init__()
        dim = cfg.latent_dim
        self.semantic_quantizer = ResidualVectorQuantize(dim, 1, cfg.semantic_codebook_size, cfg.codebook_dim)
        self.quantizer = ResidualVectorQuantize(dim, cfg.n_residual_codebooks, cfg.codebook_size, cfg.codebook_dim)
        if decode:
            self.post_module = _quantizer_transformer(cfg)
            self.upsample = nn.Sequential(
                *[
                    nn.Sequential(CausalTransConvNet(dim, dim, kernel_size=f, stride=f), ConvNeXtBlock(dim))
                    for f in reversed(cfg.downsample_factor)
                ]
            )
        else:
            self.downsample = nn.Sequential(
                *[
                    nn.Sequential(CausalConvNet(dim, dim, kernel_size=f, stride=f), ConvNeXtBlock(dim))
                    for f in cfg.downsample_factor
                ]
            )
            self.pre_module = _quantizer_transformer(cfg)

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        """``[B, 10, T]`` codes -> ``[B, D, T]`` (out-of-range codes clamp, as the reference)."""
        semantic = codes[:, :1].clamp(max=self.semantic_quantizer.codebook_size - 1)
        residual = codes[:, 1:].clamp(max=self.quantizer.codebook_size - 1)
        return self.semantic_quantizer.from_codes(semantic) + self.quantizer.from_codes(residual)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """``[B, D, 4T]`` latents -> ``[B, 10, T]`` codes."""
        z = self.pre_module(self.downsample(z))
        semantic_z, semantic_codes = self.semantic_quantizer.quantize(z)
        _, residual_codes = self.quantizer.quantize(z - semantic_z)
        return torch.cat([semantic_codes, residual_codes], dim=1)


# ---------------------------------------------------------------------------
# Top-level decoder / encoder
# ---------------------------------------------------------------------------


def _earliest_input(module: nn.Module, out_index: int) -> int:
    """First input position output ``out_index`` of ``module`` depends on."""
    if isinstance(module, (Snake1d, nn.Tanh, nn.Identity)):
        return out_index
    if isinstance(module, CausalConvNet):
        return out_index * module.stride - module.padding
    if isinstance(module, CausalTransConvNet):
        # input i writes outputs i*stride .. i*stride + kernel - 1
        return -((module.kernel_size - 1 - out_index) // module.stride)
    if isinstance(module, ConvNeXtBlock):
        return _earliest_input(module.dwconv, out_index)
    if isinstance(module, (ResidualUnit, DecoderBlock, nn.Sequential)):
        blocks = module.block if hasattr(module, "block") else module
        for block in reversed(blocks):
            out_index = _earliest_input(block, out_index)
        return out_index
    raise TypeError(f"no causal receptive-field rule for {type(module).__name__}")


class FishDACDecoder(nn.Module):
    """Codes ``[B, 10, T]`` -> audio ``[B, T * 2048]`` (``quantizer.*`` + ``decoder.*`` of ``codec.pth``)."""

    def __init__(self, cfg: FishDACConfig):
        super().__init__()
        self.config = cfg
        self.quantizer = DownsampleResidualVectorQuantize(cfg, decode=True)
        self.decoder = Decoder(cfg)
        self.hop_length = cfg.hop_length

    def conv_stack(self) -> list[nn.Module]:
        """Modules after the post transformer, in execution order."""
        return [*self.quantizer.upsample, *self.decoder.model]

    def conv_receptive_field(self) -> int:
        """Frames of post-transformer context the conv stack needs for exact audio."""
        frame = 10_000
        index = frame * self.hop_length
        for module in reversed(self.conv_stack()):
            index = _earliest_input(module, index)
        return frame - index

    def latents(self, codes: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        return self.quantizer.post_module(self.quantizer.embed(codes), positions)

    def run_conv_stack(self, latents: torch.Tensor) -> torch.Tensor:
        for module in self.conv_stack():
            latents = module(latents)
        return latents.squeeze(1)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """One decode of whole streams (frame positions from 0)."""
        return self.run_conv_stack(self.latents(codes))

    def decode_window(
        self,
        codes: torch.Tensor,
        start_offset: torch.Tensor,
        conv_start: torch.Tensor,
        conv_frames: int,
    ) -> torch.Tensor:
        """Audio of frames ``conv_start .. conv_start + conv_frames`` of each window.

        ``codes`` ``[B, 10, W]`` are windows whose first frame is frame
        ``start_offset[b]`` of its stream; the transformer runs over the whole
        window and the conv stack over ``conv_frames`` frames from
        ``conv_start[b]`` (reading clamped padding past the window, which
        causal convolutions keep out of the frames before it).
        """
        width = codes.shape[-1]
        positions = start_offset.view(-1, 1) + torch.arange(width, device=codes.device)
        latents = self.latents(codes, positions.clamp(max=self.config.max_frames - 1))
        index = (conv_start.view(-1, 1) + torch.arange(conv_frames, device=codes.device)).clamp(max=width - 1)
        latents = latents.gather(2, index.unsqueeze(1).expand(-1, latents.shape[1], -1))
        return self.run_conv_stack(latents)


class FishDACEncoder(nn.Module):
    """Audio ``[B, 1, N]`` at 44.1 kHz -> codes ``[B, 10, ceil(N / 2048)]``."""

    def __init__(self, cfg: FishDACConfig):
        super().__init__()
        self.config = cfg
        self.encoder = Encoder(cfg)
        self.quantizer = DownsampleResidualVectorQuantize(cfg, decode=False)
        self.hop_length = cfg.hop_length

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        length = audio.shape[-1]
        audio = F.pad(audio, (0, -(-length // self.hop_length) * self.hop_length - length))
        return self.quantizer.encode(self.encoder(audio))


# ---------------------------------------------------------------------------
# codec.pth loading
# ---------------------------------------------------------------------------

# Buffers the reference keeps in codec.pth that this port derives instead
# (the boolean causal masks; the window mask is built per call).
_DERIVED_KEYS = (".causal_mask",)


def _fold_weight_norm(state: dict[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield checkpoint tensors with every weight-normed weight folded."""
    pairs = (
        ("parametrizations.weight.original0", "parametrizations.weight.original1"),
        ("weight_g", "weight_v"),
    )

    def has_leaf(name: str, leaf: str) -> bool:
        return name == leaf or name.endswith("." + leaf)

    for name, tensor in state.items():
        for g_leaf, v_leaf in pairs:
            if has_leaf(name, v_leaf):
                break
            if has_leaf(name, g_leaf):
                base = name[: -len(g_leaf)]
                yield base + "weight", torch._weight_norm(state[base + v_leaf], tensor, 0)
                break
        else:
            yield name, tensor


# An absolute bound: four bf16 steps for |x| in [0.5, 1), and more steps for
# smaller values (the bf16 step halves with each halving of |x|). A 1% change
# of the rope base already moves some entries by more than 1.
_ROPE_TABLE_ATOL = 2.0**-6


def load_codec_state(module: nn.Module, state: dict[str, torch.Tensor], prefixes: Iterable[str]) -> None:
    """Load the ``prefixes`` subtrees of ``codec.pth`` into ``module``; fail on any gap.

    RoPE tables (``freqs_cis``) are loaded from the checkpoint, so every host
    runs the same bf16 values: ``torch.polar`` plus a bf16 round differs by one
    step on some CPUs and libm builds. The module's own table must still agree
    within :data:`_ROPE_TABLE_ATOL`; a larger gap means the architecture
    constants are wrong.
    """
    prefixes = tuple(prefixes)
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    loaded: set[str] = set()
    for name, tensor in _fold_weight_norm(state):
        if not name.startswith(prefixes) or name.endswith(_DERIVED_KEYS):
            continue
        if name.endswith(".freqs_cis"):
            own = buffers.get(name)
            if own is None:
                raise RuntimeError(f"Fish codec RoPE table {name} has no place in this port")
            if own.shape != tensor.shape:
                raise RuntimeError(
                    f"Fish codec RoPE table {name}: checkpoint {tuple(tensor.shape)} vs {tuple(own.shape)}"
                )
            gap = (own.detach().cpu().float() - tensor.float()).abs().max().item()
            if gap > _ROPE_TABLE_ATOL:
                raise RuntimeError(
                    f"Fish codec RoPE table {name} differs from the checkpoint's by up to {gap:.3g}; "
                    f"the tolerance is {_ROPE_TABLE_ATOL:g}"
                )
            with torch.no_grad():
                own.copy_(tensor.to(dtype=own.dtype))
            continue
        target = params.get(name)
        if target is None:
            raise RuntimeError(f"Fish codec checkpoint tensor {name} has no place in this port")
        if target.shape != tensor.shape:
            raise RuntimeError(f"Fish codec {name}: checkpoint {tuple(tensor.shape)} vs {tuple(target.shape)}")
        with torch.no_grad():
            target.copy_(tensor.to(dtype=target.dtype))
        loaded.add(name)
    missing = sorted(set(params) - loaded)
    if missing:
        raise RuntimeError(f"Fish codec checkpoint did not initialize {len(missing)} tensors: {missing[:8]}")


def read_codec_checkpoint(path: str) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return state.get("generator", state)
