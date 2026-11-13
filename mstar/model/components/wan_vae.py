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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    DupUp3D,
    WanAttentionBlock,
    WanCausalConv3d,
    WanDecoder3d,
    WanMidBlock,
    WanResample,
    WanResidualBlock,
    WanResidualUpBlock,
    WanRMS_norm,
    WanUpBlock,
    WanUpsample,
    unpatchify,
)
from torch import nn

if TYPE_CHECKING:
    from mstar.distributed.communication import CommGroup


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
}


def to_default_layout(vae: nn.Module) -> None:
    """Undo ``to_channels_last``, in place; idempotent. The weights become
    contiguous again (the layout they load in) and the modules their diffusers
    classes, so ``vae.decode`` is stock ``vae.decode`` bit for bit."""
    if not is_channels_last(vae):
        return
    if is_height_sharded(vae):
        # its layers expect a row shard; a stock decode would hand them whole frames
        raise ValueError("A height-sharded Wan VAE decodes only through its HeightShardedDecoder")
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

    def __init__(self, vae: nn.Module, compile: bool, deterministic: bool = False):
        self.compiled = compile
        self._decoder = vae.decoder
        self._fn = compile_decode(self._decoder.forward, deterministic=deterministic) if compile else None
        self._warmed: set[tuple] = set()
        # Joins every warm-up key. A height-sharded decode returns the clip's row
        # split, which all its ranks share: the warm-up runs collectives, so
        # whether a call is new has to be decided alike on every rank, and a
        # rank's own shapes alone can repeat on one rank but not the other.
        self.key_extra: Callable[[], tuple] = tuple

    def __call__(self, x: torch.Tensor, feat_cache: list, first_chunk: bool = False) -> torch.Tensor:
        if not self.compiled or first_chunk or any(isinstance(t, str) for t in feat_cache):
            return self._decoder(x, feat_cache=feat_cache, feat_idx=[0], first_chunk=first_chunk)
        # A slice of a clip keeps the clip's batch stride, which the compiled
        # graph guards on: without this, every new clip length recompiles.
        x = _compact(x)
        if x.dtype == torch.bfloat16:
            key = (tuple(x.shape), x.stride(), _cache_signature(feat_cache), self.key_extra())
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

    def __init__(self, vae: nn.Module, *, channels_last: bool = True, compile: bool = True, deterministic: bool = True):
        self.vae = vae
        self.channels_last = channels_last
        if channels_last:
            self._use_layout(True)
        self.step = DecoderStep(vae, compile=compile, deterministic=deterministic)

    def _use_layout(self, channels_last: bool) -> None:
        if channels_last != is_channels_last(self.vae):
            with torch.inference_mode(False):
                (to_channels_last if channels_last else to_default_layout)(self.vae)

    def use_stock_layout(self) -> None:
        """Switch the VAE to the layout it loaded in, for a stock ``vae.decode``."""
        self._use_layout(False)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``decode_causal`` of a whole clip: ``vae.decode(z)``, call for call."""
        return self.decode_stream(z, new_feat_cache(self.vae), clip_start=True)

    def gather(self, frames: torch.Tensor) -> torch.Tensor:
        """The whole frames of the last decode: here, its output as it is."""
        return frames

    def decode_stream(
        self, z: torch.Tensor, feat_cache: list, *, clip_start: bool, latents_per_call: int = 1,
    ) -> torch.Tensor:
        """``decode_causal`` continuing the stream whose cache is ``feat_cache``."""
        self._use_layout(self.channels_last)
        with torch.autocast(device_type=z.device.type, enabled=False):
            return decode_causal(
                self.vae, z, feat_cache, self.step, clip_start=clip_start, latents_per_call=latents_per_call,
            )


# ---------------------------------------------------------------------------
# Height-sharded decode
#
# Each rank of a decoder node's TP group decodes its own band of rows of every
# frame, and the frames are gathered at the end. Only three kinds of decoder
# layer see across rows: a conv with a 3-row kernel needs one row of each
# neighbour's band (a halo, zeros at the frame's edges, as the conv's own zero
# padding), and the mid-block attention needs the whole frame (gathered, then
# each rank keeps its rows). Everything else (RMS norms, the 2x nearest
# upsample, ``DupUp3D``, unpatchify) works per pixel, and a band stays aligned
# through every 2x upsample. So the output is the whole-clip decode's, up to
# the rounding of conv algorithms that cuDNN picks per shape.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HeightShard:
    """The row split a sharded decoder's layers share: the group, and the
    latent rows each rank holds for the clip being decoded (``None`` for a
    clip too short to split, which every rank then decodes whole)."""
    group: CommGroup
    rows: tuple[int, ...] | None = None

    def level_rows(self, local_rows: int) -> list[int]:
        """Every rank's rows at the resolution of an activation that has
        ``local_rows`` rows here: the latent split, scaled."""
        scale, rest = divmod(local_rows, self.rows[self.group.rank])
        assert rest == 0, (local_rows, self.rows)
        return [r * scale for r in self.rows]


def split_rows(height: int, world: int) -> tuple[int, ...]:
    """Bands as even as rows allow, the last ``height % world`` one row taller:
    rank 0 also leads the instance and holds more beside the decode, so it gets
    the smaller band. Every rank must agree; a band that runs one rank out of
    memory mid-decode leaves the others in a collective."""
    base, extra = divmod(height, world)
    return tuple(base + (r >= world - extra) for r in range(world))


def gather_rows(shard: HeightShard, x: torch.Tensor, dim: int) -> torch.Tensor:
    """Every rank's band of ``x`` along ``dim``, in rank order. Uneven bands are
    padded to the tallest for the all-gather and trimmed after."""
    sizes = shard.level_rows(x.shape[dim])
    tallest = max(sizes)
    if x.shape[dim] < tallest:
        pad = x.new_zeros(*x.shape[:dim], tallest - x.shape[dim], *x.shape[dim + 1:])
        x = torch.cat([x, pad], dim=dim)
    bands = shard.group.all_gather(x.contiguous(), dim=dim).split(tallest, dim=dim)
    return torch.cat([band.narrow(dim, 0, n) for band, n in zip(bands, sizes, strict=True)], dim=dim)


def _neighbour_rows(shard: HeightShard, top: torch.Tensor, bottom: torch.Tensor, dim: int):
    """The row above this band (the previous rank's last) and the row below it
    (the next rank's first): one all-gather of every band's two edge rows."""
    group = shard.group
    edges = group.all_gather(torch.cat([top, bottom], dim=dim).contiguous(), dim=dim).split(1, dim=dim)
    above = edges[2 * group.rank - 1] if group.rank > 0 else torch.zeros_like(top)
    below = edges[2 * group.rank + 2] if group.rank < group.world_size - 1 else torch.zeros_like(bottom)
    return above, below


def _with_halo(
    shard: HeightShard, x: torch.Tensor, row_dim: int, layout: torch.memory_format,
    cached: torch.Tensor | None = None, zero_frames: int = 0,
) -> torch.Tensor:
    """A conv's input in its one ``layout`` copy, as the unsharded conv makes it:
    ``zero_frames`` zero frames, then ``cached``, then ``x`` (frames on dim 2),
    with a neighbour row above and below along ``row_dim``."""
    lead = zero_frames + (cached.shape[2] if cached is not None else 0)
    shape = list(x.shape)
    rows = shape[row_dim]
    shape[row_dim] = rows + 2
    if lead:
        shape[2] += lead
    out = torch.empty(shape, dtype=x.dtype, device=x.device, memory_format=layout)
    body = out.narrow(row_dim, 1, rows)
    if zero_frames:
        body.narrow(2, 0, zero_frames).zero_()
    if cached is not None:
        body.narrow(2, zero_frames, cached.shape[2]).copy_(cached)
    (body.narrow(2, lead, x.shape[2]) if lead else body).copy_(x)
    above, below = _neighbour_rows(shard, out.narrow(row_dim, 1, 1), out.narrow(row_dim, rows, 1), row_dim)
    out.narrow(row_dim, 0, 1).copy_(above)
    out.narrow(row_dim, rows + 1, 1).copy_(below)
    return out


def _check_rows(conv: nn.Module, kernel: int, padding: int, stride: int, dilation: int) -> None:
    """A band is exact under a zero-padded row kernel that pads one row each side (or none)."""
    if (stride != 1 or dilation != 1 or padding not in (0, 1) or kernel != 2 * padding + 1
            or conv.padding_mode != "zeros"):
        raise ValueError(
            f"Can't split {type(conv).__name__} by rows: kernel {kernel}, padding {padding} ({conv.padding_mode}), "
            f"stride {stride}, dilation {dilation}"
        )


class HeightShardedCausalConv3d(ChannelsLastCausalConv3d):
    """A causal conv on a row band: the cached frames, the band and its halo
    built in one channels-last copy, then the conv with no row padding."""
    shard: HeightShard

    @classmethod
    def adopt(cls, module: ChannelsLastCausalConv3d) -> bool:
        _check_rows(module, module.kernel_size[1], module._padding[2], module.stride[1], module.dilation[1])
        return True

    def forward(self, x, cache_x=None):
        pad_w, _, pad_h, _, pad_t, _ = self._padding
        if pad_h == 0 or self.shard.rows is None:
            return super().forward(x, cache_x)
        cached = None
        if pad_t > 0 and cache_x is not None:
            cached = cache_x.to(x.device)
            pad_t -= cached.shape[2]
        x = _with_halo(self.shard, x, 3, torch.channels_last_3d, cached=cached, zero_frames=max(pad_t, 0))
        return F.conv3d(x, self.weight, self.bias, self.stride, (0, 0, pad_w), self.dilation, self.groups)


class HeightShardedConv2d(ChannelsLastConv2d):
    """The per-frame 3x3 conv after a spatial upsample, on a row band."""
    shard: HeightShard

    @classmethod
    def adopt(cls, module: ChannelsLastConv2d) -> bool:
        if module.kernel_size == (1, 1):
            return False  # the attention's projections, per pixel
        _check_rows(module, module.kernel_size[0], module.padding[0], module.stride[0], module.dilation[0])
        return module.padding[0] == 1

    def forward(self, x):
        if self.shard.rows is None:
            return super().forward(x)
        x = _with_halo(self.shard, x, 2, torch.channels_last)
        return F.conv2d(x, self.weight, self.bias, self.stride, (0, self.padding[1]), self.dilation, self.groups)


class HeightShardedAttentionBlock(WanAttentionBlock):
    """The mid-block attention over the whole frame, each rank keeping its rows."""
    shard: HeightShard

    @classmethod
    def adopt(cls, module: WanAttentionBlock) -> bool:
        return True

    def forward(self, x):
        if self.shard.rows is None:
            return super().forward(x)
        sizes = self.shard.level_rows(x.shape[3])
        rank = self.shard.group.rank
        full = super().forward(gather_rows(self.shard, x, dim=3))
        return full.narrow(3, sum(sizes[:rank]), sizes[rank])


# The decoder's layers that see across rows, by exact class.
_SHARDED_CLASS = {
    ChannelsLastCausalConv3d: HeightShardedCausalConv3d,
    ChannelsLastConv2d: HeightShardedConv2d,
    WanAttentionBlock: HeightShardedAttentionBlock,
}
# Every other layer of the two Wan decoders, by exact class: each works per pixel
# (RMS norms over channels, activations, the 2x nearest upsample, DupUp3D's
# channel-to-pixel shuffle) or only holds layers that are checked themselves. A
# band passes through them as the whole frame would. A class outside both sets,
# a subclass included (it may override ``forward``), refuses the conversion.
_PER_PIXEL = frozenset({
    WanDecoder3d, WanMidBlock, WanResidualBlock, WanUpBlock, WanResidualUpBlock, WanResample,
    WanRMS_norm, DupUp3D, ChannelsLastUpsample, nn.SiLU, nn.Dropout, nn.Identity, nn.ModuleList, nn.Sequential,
})


def is_height_sharded(vae: nn.Module) -> bool:
    return any(isinstance(m, HeightShardedCausalConv3d) for m in vae.decoder.modules())


def to_height_sharded(vae: nn.Module, group: CommGroup) -> HeightShard:
    """Switch a channels-last decoder's row-crossing layers to their sharded
    classes, in place, all sharing one ``HeightShard``. Refuses the conversion
    for a conv it can't split exactly (only one-row-padded 3-row kernels can be)
    and for any layer it doesn't know to work per pixel (``_PER_PIXEL``)."""
    if not is_channels_last(vae):
        raise ValueError("Shard a Wan VAE decoder after switching it to channels-last")
    shard = HeightShard(group)
    for module in vae.decoder.modules():
        kind = type(module)
        sharded = _SHARDED_CLASS.get(kind)
        if sharded is not None:
            if sharded.adopt(module):
                module.__class__ = sharded
                module.shard = shard
            continue  # adopt refuses what it can't split; False is a per-pixel kernel (1x1)
        if kind not in _PER_PIXEL:
            raise ValueError(f"Can't split a Wan VAE decoder by rows: {kind.__name__} isn't known to work per pixel")
        if kind is ChannelsLastUpsample and module.mode not in ("nearest", "nearest-exact"):
            raise ValueError(f"Can't split a Wan VAE decoder by rows: a {module.mode} upsample blends rows")
    return shard


class HeightShardedDecoder(FastDecoder):
    """A ``FastDecoder`` whose ranks each decode a band of rows, over ``group``.

    Every rank of the group calls ``decode`` with the same latents, in the same
    order (a lockstep node's steps), and gets back its band;
    ``gather`` assembles the frames (after any per-pixel conversion, e.g. to
    uint8, which is cheaper to gather). Collective: every rank makes the same
    calls. A clip with fewer latent rows than ranks decodes whole on each.
    """

    def __init__(self, vae: nn.Module, group: CommGroup, *, compile: bool = True, deterministic: bool = True):
        super().__init__(vae, channels_last=True, compile=compile, deterministic=deterministic)
        self.shard = to_height_sharded(vae, group)
        self.step.key_extra = lambda: (self.shard.rows,)

    def use_stock_layout(self) -> None:
        raise ValueError("A height-sharded decoder has no stock layout")

    def decode_stream(
        self, z: torch.Tensor, feat_cache: list, *, clip_start: bool, latents_per_call: int = 1,
    ) -> torch.Tensor:
        """This rank's band of the decode of ``z`` (every rank's whole latents)."""
        group = self.shard.group
        rows = split_rows(z.shape[3], group.world_size)
        if min(rows) == 0:
            self.shard.rows = None
        else:
            self.shard.rows = rows
            z = z.narrow(3, sum(rows[:group.rank]), rows[group.rank])
        return super().decode_stream(z, feat_cache, clip_start=clip_start, latents_per_call=latents_per_call)

    def gather(self, band: torch.Tensor) -> torch.Tensor:
        """The whole frames from every rank's band of the last decode (rows on dim 3)."""
        if self.shard.rows is None:
            return band
        return gather_rows(self.shard, band, dim=3)
