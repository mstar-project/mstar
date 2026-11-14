"""Decode helpers for the diffusers Wan VAE (``AutoencoderKLWan``), shared by the
Wan VAE decoder nodes.

``AutoencoderKLWan.decode`` runs its causal decoder one latent frame per call
against a feature cache that lives on the VAE instance and is cleared at the
start and end of every decode. ``decode_causal`` runs the same calls against a
cache the caller owns, so a stream can keep it warm across calls: each causal
conv keeps the last frames of its input, and the next call continues from them
exactly as one decode of the whole stream would. ``to_channels_last`` switches
the decode path to channels-last weights and activations, the layout cuDNN's
conv3d kernels run in, so no conv pays a layout conversion on the way in or out.
``rewrite_decoder`` (opt-in, near-lossless) changes how three pieces of the decode
are computed: the temporal upsamplers leave zero frames for the clip start instead
of diffusers' "Rep" marker, so every call can run compiled; nearest-x2 + 3x3 conv
runs as one stride-2 transposed conv; the mid-block attention runs as two GEMMs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanAttentionBlock,
    WanCausalConv3d,
    WanResample,
    WanUpsample,
    unpatchify,
)
from torch import nn


class ChannelsLastCausalConv3d(WanCausalConv3d):
    """``WanCausalConv3d`` whose input reaches the kernel channels-last.

    Only the temporal context is concatenated (the cached frames, then zero
    frames for whatever the cache does not cover); the H/W zero padding
    happens inside the conv instead of in a padded copy. Same math as the
    parent's ``F.pad`` + conv with padding 0.
    """

    def forward(self, x, cache_x=None):
        pad_w, _, pad_h, _, pad_t, _ = self._padding
        if pad_t > 0:
            if cache_x is not None:
                x = torch.cat([cache_x.to(x.device), x], dim=2)
                pad_t -= cache_x.shape[2]
            if pad_t > 0:
                zeros = x.new_zeros(*x.shape[:2], pad_t, *x.shape[3:])
                x = torch.cat([zeros.contiguous(memory_format=torch.channels_last_3d), x], dim=2)
        x = x.contiguous(memory_format=torch.channels_last_3d)
        return F.conv3d(x, self.weight, self.bias, self.stride, (0, pad_h, pad_w), self.dilation, self.groups)


class ChannelsLastConv2d(nn.Conv2d):
    """The decoder's per-frame 2D convs (spatial upsample, attention
    projections) with a channels-last input."""

    def forward(self, x):
        return super().forward(x.contiguous(memory_format=torch.channels_last))


class ChannelsLastUpsample(WanUpsample):
    """``WanUpsample`` on a channels-last input. The temporal upsample's frame
    interleave hands it a contiguous tensor; converting before the 2x upsample
    moves a quarter of the bytes the conv after it would."""

    def forward(self, x):
        return super().forward(x.contiguous(memory_format=torch.channels_last))


# frames a temporal upsampler's causal conv keeps between calls (diffusers' CACHE_T)
_CACHE_T = 2

# Per axis, the taps of nearest-x2 then a 3x3 conv (pad 1) as a stride-2 transposed conv with a 4-tap kernel
# (padding 1): output row 2i takes input rows i-1, i with taps (W0, W1 + W2), row 2i + 1 rows i, i + 1 with
# (W0 + W1, W2), which is the transposed conv's K = [W2, W1 + W2, W0 + W1, W0]. Row k of this matrix is K[k].
_PHASE_TAPS = ((0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0))


def phase_upsample_weight(conv: nn.Conv2d) -> torch.Tensor:
    """``conv``'s 3x3 weight ``[co, ci, 3, 3]`` as the ``[ci, co, 4, 4]`` weight of the stride-2 transposed conv
    that equals nearest-x2 then ``conv``: the taps summed in fp64, rounded once to ``conv``'s dtype."""
    taps = torch.tensor(_PHASE_TAPS, dtype=torch.float64, device=conv.weight.device)
    weight = torch.einsum("hr,oirs,ks->iohk", taps, conv.weight.detach().double(), taps)
    return weight.to(conv.weight.dtype).contiguous(memory_format=torch.channels_last)


class RewrittenResample(WanResample):
    """``WanResample`` for ``rewrite_decoder``, same math up to rounding.

    At the clip start a temporal upsampler stores two zero frames where diffusers stores
    "Rep". "Rep" makes the next call pad its causal conv with zeros, and these frames are
    that padding (bit for bit in eager), but the next call is now a steady-state call that
    reuses the steady-state compiled graph. Nearest-x2 then the 3x3 conv runs as one
    stride-2 transposed conv (``phase_weight``): 4 taps an output pixel instead of 9, and
    no 2x-upsampled tensor."""

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # noqa: B006 (diffusers' signature)
        if self.mode not in ("upsample2d", "upsample3d"):
            return super().forward(x, feat_cache, feat_idx)
        if self.mode == "upsample3d" and feat_cache is not None:
            x = self._upsample_time(x, feat_cache, feat_idx)
        b, c, t, h, w = x.size()
        x = self._upsample_space(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        return x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

    def _upsample_time(self, x: torch.Tensor, feat_cache: list, feat_idx: list) -> torch.Tensor:
        b, c, t, h, w = x.size()
        idx = feat_idx[0]
        feat_idx[0] += 1
        if feat_cache[idx] is None:
            feat_cache[idx] = x.new_zeros(b, c, _CACHE_T, h, w).contiguous(memory_format=torch.channels_last_3d)
            return x
        cache_x = x[:, :, -_CACHE_T:].clone()
        if cache_x.shape[2] < _CACHE_T:
            cache_x = torch.cat([feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2)
        x = self.time_conv(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        x = x.reshape(b, 2, c, t, h, w)
        return torch.stack((x[:, 0], x[:, 1]), 3).reshape(b, c, t * 2, h, w)

    def _upsample_space(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous(memory_format=torch.channels_last)
        return F.conv_transpose2d(x, self.phase_weight, self.resample[1].bias, stride=2, padding=1)


class RewrittenAttentionBlock(WanAttentionBlock):
    """``WanAttentionBlock`` for ``rewrite_decoder``: the single-head attention (head dim 384
    on the Wan2.1 VAE, 1024 on Wan2.2's, past every flash kernel) as fp32 scores from one
    GEMM, a softmax and a second GEMM, instead of SDPA's memory-efficient kernel."""

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, c * 3, h * w).permute(0, 2, 1).chunk(3, dim=-1)
        if q.dtype not in (torch.float16, torch.bfloat16):
            scores = torch.bmm(q, k.transpose(1, 2))
        elif q.is_cuda:
            scores = torch.bmm(q, k.transpose(1, 2), out_dtype=torch.float32)
        else:
            scores = torch.bmm(q.float(), k.float().transpose(1, 2))
        p = torch.softmax(scores * c**-0.5, dim=-1).to(v.dtype)
        x = torch.bmm(p, v).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4) + identity


def rewrite_decoder(vae: nn.Module) -> None:
    """Switch ``vae``'s decoder to ``RewrittenResample`` / ``RewrittenAttentionBlock``, in
    place; idempotent. ``to_default_layout`` switches it back."""
    for module in vae.decoder.modules():
        if type(module) is WanResample and module.mode in ("upsample2d", "upsample3d"):
            module.register_buffer("phase_weight", phase_upsample_weight(module.resample[1]), persistent=False)
            module.__class__ = RewrittenResample
        elif type(module) is WanAttentionBlock:
            module.__class__ = RewrittenAttentionBlock


def is_rewritten(vae: nn.Module) -> bool:
    return any(isinstance(m, (RewrittenResample, RewrittenAttentionBlock)) for m in vae.decoder.modules())


def to_channels_last(vae: nn.Module) -> None:
    """Channels-last weights and activations for ``vae``'s decode path
    (``post_quant_conv`` and ``decoder``), in place; idempotent. The encoder
    is left alone (``encoder_to_channels_last``)."""
    _to_channels_last(getattr(vae, "post_quant_conv", None), getattr(vae, "decoder", None))


def encoder_to_channels_last(vae: nn.Module) -> None:
    """``to_channels_last`` for the encode path (``encoder`` and
    ``quant_conv``), in place; idempotent. Not ``is_channels_last``: that
    describes the decode path."""
    _to_channels_last(getattr(vae, "encoder", None), getattr(vae, "quant_conv", None))


def _to_channels_last(*roots: nn.Module | None) -> None:
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if type(module) is WanCausalConv3d:
                module.__class__ = ChannelsLastCausalConv3d
            elif type(module) is nn.Conv2d:
                module.__class__ = ChannelsLastConv2d
            elif type(module) is WanUpsample:
                module.__class__ = ChannelsLastUpsample
            if isinstance(module, nn.Conv3d):
                module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last_3d)
            elif isinstance(module, nn.Conv2d):
                module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last)


_DEFAULT_LAYOUT_CLASS = {
    ChannelsLastCausalConv3d: WanCausalConv3d,
    ChannelsLastConv2d: nn.Conv2d,
    ChannelsLastUpsample: WanUpsample,
    RewrittenResample: WanResample,
    RewrittenAttentionBlock: WanAttentionBlock,
}


def to_default_layout(vae: nn.Module) -> None:
    """Undo ``to_channels_last`` (and ``rewrite_decoder``), in place; idempotent. The
    weights become contiguous again (the layout they load in) and the modules their
    diffusers classes, so ``vae.decode`` is stock ``vae.decode`` bit for bit."""
    if not is_channels_last(vae) and not is_rewritten(vae):
        return
    for root in (getattr(vae, "post_quant_conv", None), getattr(vae, "decoder", None)):
        if root is None:
            continue
        for module in root.modules():
            module.__class__ = _DEFAULT_LAYOUT_CLASS.get(type(module), type(module))
            if isinstance(module, (nn.Conv3d, nn.Conv2d)):
                # Not .contiguous(): with size-1 dims (a 1x1x1 kernel) both
                # layouts count as contiguous and it would keep these strides.
                module.weight.data = module.weight.data.clone(memory_format=torch.contiguous_format)


def is_channels_last(vae: nn.Module) -> bool:
    return isinstance(getattr(vae, "post_quant_conv", None), ChannelsLastCausalConv3d)


class FeatCache(list):
    """A decoder feature cache that a compiled call writes as compact copies.

    Diffusers caches ``x[:, :, -CACHE_T:].clone()``. On channels-last
    activations at batch 1 that slice is dense, so the clone has the slice's
    strides and Inductor removes it as a no-op: a compiled call returns the
    slot as a view of the conv input it was sliced from (the cached frames
    plus all of the call's new frames), and the view keeps that buffer alive
    for as long as the stream holds the cache. Under compile a slot is
    written as a copy with ``_compact_strides`` instead, at the write, so the
    buffer can be freed after its conv; ``compact_feat_cache`` catches any
    slot that still comes back as a view. Eager writes get the same strides
    (a re-view of their own storage), so the next compiled call's stride
    guards see one layout whichever kind of call wrote the cache.
    """

    def __setitem__(self, index, value):
        if torch.is_tensor(value):
            value = _compact(value)
        super().__setitem__(index, value)


def _compact_strides(t: torch.Tensor) -> list[int]:
    """Gap-free strides in ``t``'s dimension order; size-1 dims outermost."""
    strides, step = [0] * t.dim(), 1
    for d in sorted((d for d in range(t.dim()) if t.shape[d] != 1), key=t.stride):
        strides[d] = step
        step *= t.shape[d]
    return [step if n == 1 else s for s, n in zip(strides, t.shape, strict=True)]


def _compact(t: torch.Tensor) -> torch.Tensor:
    strides = _compact_strides(t)
    if torch.compiler.is_compiling():
        return torch.empty_strided(t.shape, strides, dtype=t.dtype, device=t.device).copy_(t)
    if list(t.stride()) == strides and t.untyped_storage().nbytes() == t.numel() * t.element_size():
        return t
    if t.storage_offset() == 0 and t.untyped_storage().nbytes() == t.numel() * t.element_size():
        return t.as_strided(t.shape, strides)  # only size-1 dims' strides differ
    return torch.empty_strided(t.shape, strides, dtype=t.dtype, device=t.device).copy_(t)


def compact_feat_cache(feat_cache: list) -> None:
    """Give every slot compact storage with ``_compact_strides``, in place,
    so the cache holds only ``feat_cache_bytes``."""
    for i, t in enumerate(feat_cache):
        if torch.is_tensor(t):
            list.__setitem__(feat_cache, i, _compact(t))


def new_feat_cache(vae: nn.Module) -> FeatCache:
    """An empty decoder feature cache: one slot per causal conv, as
    ``AutoencoderKLWan.clear_cache`` builds it."""
    counts = getattr(vae, "_cached_conv_counts", None)
    if counts is not None:
        return FeatCache([None] * counts["decoder"])
    return FeatCache([None] * sum(isinstance(m, WanCausalConv3d) for m in vae.decoder.modules()))


def feat_cache_bytes(feat_cache: list) -> int:
    return sum(t.numel() * t.element_size() for t in feat_cache if torch.is_tensor(t))


def compile_decode(fn, deterministic: bool = False):
    """``torch.compile`` a Wan VAE decode callable; every compiled Wan VAE
    decode goes through here.

    Comprehensive padding is off. With it, Inductor on torch 2.13 pads the
    strides it picks for ``up_blocks[1]``'s shortcut ``x.clone()`` as if
    channels-first, while the upsample conv that the clone is fused into
    writes it channels-last: a Wan 2.2 (``is_residual``) decode then reads
    wrong values past the buffer's end (wrong frames, NaNs or an illegal
    memory access). On torch 2.12.1 the ``DecoderStep`` paths graph-break
    before that graph forms, so the option leaves them byte-identical. The
    stateless whole-clip decode does form it there: the option moves its
    output by up to 2 uint8 levels, at the same speed.

    ``deterministic`` adds the engine's deterministic compile option
    (``mstar.platforms.compile.deterministic_inductor_options``); see
    ``DecoderStep``.
    """
    import torch._inductor.config as inductor_config

    options = {}
    if hasattr(inductor_config, "comprehensive_padding"):
        options["comprehensive_padding"] = False
    if deterministic:
        from mstar.platforms.compile import deterministic_inductor_options

        options.update(deterministic_inductor_options())
    return torch.compile(fn, fullgraph=False, dynamic=False, options=options or None)


def _cache_signature(feat_cache: list) -> tuple:
    return tuple(t if t is None or isinstance(t, str) else tuple(t.shape) for t in feat_cache)


class DecoderStep:
    """One ``WanDecoder3d`` call against a caller-owned cache, optionally
    compiled.

    Only steady-state calls run compiled. A stream's first two calls (the
    clip-start latent, then the call that still finds diffusers' ``"Rep"``
    marker in the temporal upsamples' slots) run eagerly: compiled, the
    second one returns wrong frames and a wrong cache at 832x480 (seen on
    torch 2.12.1, diffusers 0.39, cuDNN 9.20), while every steady-state call
    matches eager.

    A compiled bf16 graph's first execution in a process returns corrupted
    frames on some torch/cuDNN stacks (every later call of the same graph is
    correct; the whole-clip decoder's ``_warmed_decode_shapes`` works around
    the same thing), so a compiled step runs each new call signature once on
    a copy of the cache and discards the result before serving it. The signature is
    the input's shape and strides plus the cache's slot shapes, so the warm-up
    sees the real layout.

    ``deterministic`` compiles in Inductor's deterministic mode (the engine's
    option, logged where torch has none): each reduction kernel's config is
    then picked by a fixed rule. Otherwise it is benchmarked on first use with
    a cold cache, and processes that pick differently decode the same latents
    a few uint8 levels apart.
    """

    def __init__(self, vae: nn.Module, compile: bool, deterministic: bool = False, compile_clip_start: bool = False):
        self.compiled = compile
        # The clip-start call compiled too, as its own graph: for a decoder whose temporal
        # upsamplers leave no "Rep" marker (``rewrite_decoder``), so the second call is a
        # steady-state one and every call runs compiled.
        self.compile_clip_start = compile_clip_start
        self._decoder = vae.decoder
        self._fn = compile_decode(self._decoder.forward, deterministic=deterministic) if compile else None
        self._warmed: set[tuple] = set()

    def __call__(self, x: torch.Tensor, feat_cache: list, first_chunk: bool = False) -> torch.Tensor:
        if (not self.compiled or (first_chunk and not self.compile_clip_start)
                or any(isinstance(t, str) for t in feat_cache)):
            return self._decoder(x, feat_cache=feat_cache, feat_idx=[0], first_chunk=first_chunk)
        # A slice of a clip keeps the clip's batch stride, which the compiled
        # graph guards on: without this, every new clip length recompiles.
        x = _compact(x)
        if x.dtype == torch.bfloat16:
            key = (tuple(x.shape), x.stride(), _cache_signature(feat_cache), first_chunk)
            if key not in self._warmed:
                scratch = type(feat_cache)(t.clone() if torch.is_tensor(t) else t for t in feat_cache)
                self._fn(x, feat_cache=scratch, feat_idx=[0], first_chunk=first_chunk)
                self._warmed.add(key)
        out = self._fn(x, feat_cache=feat_cache, feat_idx=[0], first_chunk=first_chunk)
        compact_feat_cache(feat_cache)
        return out


def post_quant(vae: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """``vae.post_quant_conv(z)``, in the decode path's layout: the decoder's input."""
    x = vae.post_quant_conv(z)
    if is_channels_last(vae):
        x = x.contiguous(memory_format=torch.channels_last_3d)
    return x


def decode_causal(
    vae: nn.Module,
    z: torch.Tensor,
    feat_cache: list,
    step: DecoderStep,
    *,
    clip_start: bool,
    latents_per_call: int = 1,
    post_quantized: bool = False,
) -> torch.Tensor:
    """Decode the latent frames ``z`` (``[B, C, T, h, w]``, already
    de-normalized) as the continuation of the stream whose decoder cache is
    ``feat_cache``, updated in place. Returns ``[B, 3, F, H, W]`` in [-1, 1].

    ``clip_start``: ``z``'s first frame is the stream's first latent, which
    decodes to one frame with ``feat_cache`` empty; every later latent decodes
    to ``scale_factor_temporal`` frames. With an empty cache and
    ``clip_start`` this is ``vae.decode(z)``, call for call.

    ``latents_per_call`` latents go through the decoder per call (0: all the
    rest at once; the clip-start latent always decodes alone). Same math as
    one per call; fewer, larger kernels, at a peak activation memory that
    grows with the frames per call.

    ``post_quantized``: ``z`` is already ``post_quant(vae, z)``, e.g. a slice of a
    whole clip's, so a clip decoded in pieces runs that conv once over the clip as
    ``vae.decode`` does.
    """
    x = z if post_quantized else post_quant(vae, z)
    outs = []
    start = 0
    if clip_start:
        outs.append(step(x[:, :, :1], feat_cache, first_chunk=True))
        start = 1
    per_call = latents_per_call if latents_per_call > 0 else x.shape[2]
    for i in range(start, x.shape[2], per_call):
        outs.append(step(x[:, :, i : i + per_call], feat_cache))
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
    if vae.config.patch_size is not None:
        out = unpatchify(out, patch_size=vae.config.patch_size)
    return torch.clamp(out, min=-1.0, max=1.0)


class FastDecoder:
    """A decoder node's shared Wan VAE decoder: ``vae`` (already in its decode
    dtype) switched to channels-last with ``channels_last``, in place and before
    the step compiles, so the trace sees the weights the decode runs with; and
    its ``DecoderStep``.

    ``decode`` is a whole clip from an empty cache; ``decode_stream`` continues
    a caller-owned one. Both first put the VAE back in this decode's layout,
    since a node may switch it to the loaded layout for a stock decode
    (``use_stock_layout``). Every switch runs outside inference mode: a weight
    made in it is an inference tensor, whose dispatch keys the compiled graph's
    parameter guards reject (a recompile)."""

    def __init__(self, vae: nn.Module, *, channels_last: bool = True, compile: bool = True, deterministic: bool = True,
                 rewrite: bool = False):
        self.vae = vae
        self.channels_last = channels_last
        # ``rewrite_decoder`` (opt-in, near-lossless), with every call compiled
        self.rewrite = rewrite
        self._use_layout(True)
        self.step = DecoderStep(vae, compile=compile, deterministic=deterministic, compile_clip_start=rewrite)

    def _use_layout(self, decode_layout: bool) -> None:
        """``decode_layout``: this decode's (channels-last if set, rewritten if set), else stock."""
        want = (self.channels_last, self.rewrite) if decode_layout else (False, False)
        if want != (is_channels_last(self.vae), is_rewritten(self.vae)):
            with torch.inference_mode(False):
                to_default_layout(self.vae)
                if want[0]:
                    to_channels_last(self.vae)
                if want[1]:
                    rewrite_decoder(self.vae)

    def use_stock_layout(self) -> None:
        """Switch the VAE to the layout it loaded in, for a stock ``vae.decode``."""
        self._use_layout(False)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``decode_causal`` of a whole clip: ``vae.decode(z)``, call for call."""
        return self.decode_stream(z, new_feat_cache(self.vae), clip_start=True)

    def decode_stream(
        self, z: torch.Tensor, feat_cache: list, *, clip_start: bool, latents_per_call: int = 1,
        post_quantized: bool = False,
    ) -> torch.Tensor:
        """``decode_causal`` continuing the stream whose cache is ``feat_cache``."""
        self._use_layout(True)
        with torch.autocast(device_type=z.device.type, enabled=False):
            return decode_causal(
                self.vae, z, feat_cache, self.step, clip_start=clip_start, latents_per_call=latents_per_call,
                post_quantized=post_quantized,
            )

    def post_quant(self, z: torch.Tensor) -> torch.Tensor:
        """``post_quant`` in this decode's layout, for ``decode_stream(..., post_quantized=True)``."""
        self._use_layout(True)
        with torch.autocast(device_type=z.device.type, enabled=False):
            return post_quant(self.vae, z)
