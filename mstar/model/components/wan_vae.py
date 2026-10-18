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
"""

from __future__ import annotations

import contextlib
import os

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanUpsample, unpatchify
from torch import nn

from mstar.platforms import current_platform


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


def to_channels_last(vae: nn.Module) -> None:
    """Channels-last weights and activations for ``vae``'s decode path
    (``post_quant_conv`` and ``decoder``), in place; idempotent. The encoder
    is left alone."""
    for root in (getattr(vae, "post_quant_conv", None), getattr(vae, "decoder", None)):
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
}


def to_default_layout(vae: nn.Module) -> None:
    """Undo ``to_channels_last``, in place; idempotent. The weights become
    contiguous again (the layout they load in) and the modules their diffusers
    classes, so ``vae.decode`` is stock ``vae.decode`` bit for bit."""
    if not is_channels_last(vae):
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


def _has_deterministic_mode() -> bool:
    import torch._inductor.config as inductor_config

    return hasattr(inductor_config, "deterministic")


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

    ``deterministic`` turns on Inductor's deterministic mode where this
    torch has it (2.9.1 does not); see ``DecoderStep``.
    """
    import torch._inductor.config as inductor_config

    options = {}
    if hasattr(inductor_config, "comprehensive_padding"):
        options["comprehensive_padding"] = False
    if deterministic and _has_deterministic_mode():
        options["deterministic"] = True
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

    ``deterministic`` compiles in Inductor's deterministic mode where this
    torch has it (2.9.1 does not): each reduction kernel's config is then
    picked by a fixed rule. Otherwise it is benchmarked on first use with a
    cold cache, and processes that pick differently decode the same latents
    a few uint8 levels apart.
    """

    def __init__(self, vae: nn.Module, compile: bool, deterministic: bool = False):
        self.compiled = compile
        self._decoder = vae.decoder
        self._fn = compile_decode(self._decoder.forward, deterministic=deterministic) if compile else None
        self._warmed: set[tuple] = set()

    def __call__(self, x: torch.Tensor, feat_cache: list, first_chunk: bool = False) -> torch.Tensor:
        if not self.compiled or first_chunk or any(isinstance(t, str) for t in feat_cache):
            return self._decoder(x, feat_cache=feat_cache, feat_idx=[0], first_chunk=first_chunk)
        # A slice of a clip keeps the clip's batch stride, which the compiled
        # graph guards on: without this, every new clip length recompiles.
        x = _compact(x)
        if x.dtype == torch.bfloat16:
            key = (tuple(x.shape), x.stride(), _cache_signature(feat_cache))
            if key not in self._warmed:
                scratch = type(feat_cache)(t.clone() if torch.is_tensor(t) else t for t in feat_cache)
                self._fn(x, feat_cache=scratch, feat_idx=[0])
                self._warmed.add(key)
        out = self._fn(x, feat_cache=feat_cache, feat_idx=[0])
        compact_feat_cache(feat_cache)
        return out


def decode_causal(
    vae: nn.Module,
    z: torch.Tensor,
    feat_cache: list,
    step: DecoderStep,
    *,
    clip_start: bool,
    latents_per_call: int = 1,
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
    """
    x = vae.post_quant_conv(z)
    if is_channels_last(vae):
        x = x.contiguous(memory_format=torch.channels_last_3d)
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


@contextlib.contextmanager
def cudnn_benchmark(enabled: bool):
    """Within the block, let cuDNN time its conv algorithms per new shape
    (once, then cached) instead of taking its heuristic pick; restores the
    process-wide flag on exit. Not safe against convs on other threads: the
    decoder nodes run every forward on the worker's one GPU thread."""
    if not enabled:
        yield
        return
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previous


def decode_dtype(override_env: str) -> torch.dtype:
    """Dtype for the Wan VAE decode.

    If ``override_env`` is set it wins (truthy means fp32). On CUDA bf16
    conv3d is only fast from cuDNN 9.16, older versions are 2-10x slower than
    fp32. On ROCm bf16 is the fast one (3.5x vs fp32 on an MI300X), and
    cudnn.version() there is MIOpen's, so it can't be compared to 9.16.
    """
    override = os.environ.get(override_env)
    if override is not None:
        return torch.float32 if override.lower() not in ("0", "false", "no", "off") else torch.bfloat16
    if current_platform.is_rocm():
        return torch.bfloat16
    version = torch.backends.cudnn.version() or 0
    return torch.bfloat16 if version >= 91600 else torch.float32


def conv_library_version() -> str:
    """Conv library name and version, for logging."""
    library = "MIOpen" if current_platform.is_rocm() else "cuDNN"
    return f"{library} {torch.backends.cudnn.version()}"
