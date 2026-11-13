"""The Praxis-1 encoder tower: the 14-block trunk that produces the taps.

This is the piece the action expert reads and the decoder is conditioned on.
It packs five kinds of token into one sequence, runs them through a stack
whose feed-forward changes kind partway down, and hands back both a
projected conditioning tensor and eight intermediate block outputs.

The sequence is ``[registers | anchor | text | noised | clean]`` and the
order is not cosmetic. The globals (registers, anchor, text) are a
contiguous prefix, which is what lets the visibility rule in
:mod:`mstar.model.praxis1.components.chunk_mask` be "key 0 is always
visible" and what lets the output projection reach the video tokens with a
single ``narrow``. Put the clean copies before the noised chunks and every
span in :class:`TapFeatures` points at the wrong tokens while every shape
stays valid.

Five conventions here are inversions of the obvious reading, and each one
runs and produces plausible numbers when taken the other way:

* **the dense blocks are a prefix, not an interleave.** The stack is built
  from ``num_dense = len(dense_blocks)`` and blocks ``0 .. num_dense - 1``
  get the dense SwiGLU; everything after is mixture-of-experts. The
  shipped encoder config lists ``dense_blocks: [0, 1]``, which happens to
  be a prefix, so a port that reads the list as a *set of indices* and an
  interleaving port both load the same checkpoint without complaint. They
  then run two different models, because a dense block and an MoE block
  hold different tensors under the same ``blocks.N.ffn`` prefix,

* **the per-block scale is ``1 / sqrt(i + 1)``, not ``1``.** It is a
  *persistent* buffer, so the checkpoint's bf16-rounded value is the value
  in use rather than the float32 one computed at construction. It
  multiplies the output of *both* norms in the block, not the residual, so
  dropping it leaves a working transformer that is simply scaled wrong
  from block 1 onward,

* **the taps are the block output after the second residual add**, not the
  normed input of the next block and not the attention output. Eight of
  the fourteen blocks are tapped, at ``(0, 2, 4, 6, 7, 9, 11, 13)``: note
  the stride breaks between 6 and 7, so an arithmetic "every other block"
  guess gives seven of eight taps right,

* **``embeddings`` in :class:`TapFeatures` is the input of block 0**, which
  is not a tap. Block 0 *is* tapped, so both tensors exist and both are
  ``(1, S, dim)``; handing the expert the tap where it wants the embeddings
  gives it a tensor one whole block too late,

* **the clean tokens are modulated at timestep 0, not at the chunk's own
  timestep.** The clean copies are the tokens the model is allowed to have
  seen already, so they carry the timestep of fully denoised video. Feeding
  them ``timesteps`` instead is a single argument away and silently tells
  the model its context is as noisy as its target.

The timestep enters AdaLN style exactly once, at the tower input: a
``Linear(time_dim, 2 * dim)`` on the Fourier features gives
``(shift, scale)`` per chunk, and the video tokens of chunk ``k`` become
``x * (1 + scale_k) + shift_k``. The registers, the anchor and the text are
**not** modulated at all. The encoder's stack carries no per-block
modulator (the shipped encoder config has no ``modulation_dim``; the
decoder's does), so an AdaLN inside the blocks here would be a different
architecture that still trains.

Two oddities kept deliberately:

* ``positive_zero_residual_add``: a literal ``+ 0.0`` on the stack output.
  It cannot change a finite value, and that is the point. It exists to
  preserve the *sign* of zeros: ``-0.0 + 0.0`` is ``+0.0``, so the original
  implementation's output has positive zeros wherever the stack produced
  negative ones, and a consumer that divides by, copysigns from or compares
  the sign of a zero would see the other sign without it. It is read from
  the checkpoint config rather than assumed. Measured, it is **inert on
  everything this tower returns**: nothing reads the stack output directly,
  it goes through a linear projection, and a dot product cannot see the sign
  of a zero addend. Kept anyway, because the tensor it guards is what a
  later slice consumes and because removing a documented checkpoint field on
  the grounds that one slice cannot observe it is how a port drifts,

* the ``split``/``cat`` and ``cat([x])`` round trips in the projections.
  These are value identities. They are there so that the tensor handed to
  each GEMM has the memory layout the original implementation gave it, and
  so that the backward of the split is a ``cat`` that hands the linear
  layer a contiguous gradient. A GEMM's accumulation order follows its
  operand layout, so removing them is a bit-level change on an accelerator
  even though it is free on CPU.

Attention itself and the visibility mask are the caller's, as in
:mod:`mstar.model.praxis1.components.attention`: the served path wants the
engine's kernel over a block mask, which M* cannot express yet, so the
tower threads an ``attend`` callable and a mask through to every block.
With neither, it falls back to the dense boolean mask of
:func:`dense_chunk_causal_mask` and plain SDPA, which is the parity driver.

Not here, on purpose: the inference context cache (``ctx``, ``KVSplice``),
which reuses the globals' and clean tokens' activations across Euler steps,
and the per-device position cache. Both are performance work that has to be
checked against the cached path's own splicing rules, and neither changes a
value on the uncached path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn

from mstar.model.praxis1.components.attention import Attention, sdpa_attend
from mstar.model.praxis1.components.chunk_mask import encoder_codes, visible
from mstar.model.praxis1.components.moe import MoEFeedForward, SwiGLUFeedForward
from mstar.model.praxis1.components.norm_time import (
    Float32RMSNorm,
    TimestepEmbedding,
    ZeroCenteredRMSNorm,
)
from mstar.model.praxis1.components.rope import FRAME_STRIDE, RotaryEmbedding, video_positions

# The module keys of the input and output projections. They are the
# checkpoint's own names, so the tower loads the reference's state dict
# without a remap; a rename here is a weight-loader change, not a cleanup.
TEXT = "TextEncodingTensor"
VIDEO = "VideoTensor"

# The eight tapped blocks of the shipped 14-block encoder. The gap between
# 6 and 7 is why this is a table and not a stride.
DEFAULT_TAP_BLOCKS = (0, 2, 4, 6, 7, 9, 11, 13)


@dataclass
class TapFeatures:
    """What one encoder call hands the action expert.

    ``taps`` holds the tapped block outputs ``(1, S, dim)`` in the order of
    ``tap_blocks``, not in sorted order: the expert reads them positionally,
    one per pair of its own layers, so reordering them rewires the expert.

    ``embeddings`` is the **input of block 0**, before any block has run. It
    is not one of the taps. Spans are ``(start, length)`` token ranges into
    the encoder sequence: ``global_spans`` carries ``register``, ``anchor``
    and ``text``, ``noised_span`` the noised chunks and ``clean_span`` the
    clean copies, which is None for a window of a single chunk.
    """

    taps: list[torch.Tensor]
    embeddings: torch.Tensor
    frames_per_chunk: int
    tokens_per_frame: int
    rope_ids: torch.Tensor
    global_spans: dict[str, tuple[int, int]]
    noised_span: tuple[int, int]
    clean_span: tuple[int, int] | None
    n_chunks: int


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x * scale + shift`` computed in float32 and cast back to ``x``.

    The widening is the reference's ``dtype="mixed"``: a bfloat16 token is
    multiplied and shifted in float32 and only the result is rounded. Doing
    it in bfloat16 rounds twice and is a different number. ``scale`` arrives
    as ``1 + scale``, already formed by the caller in the projection's dtype,
    which is also what the reference does.

    The reference has a fused Triton kernel for this on CUDA that may
    contract the multiply and the add into one FMA, so its rounding is not
    the rounding of these two ops. That kernel is the contract on an
    accelerator; this is the contract everywhere else.
    """
    out = x.to(torch.float32) * scale.to(torch.float32) + shift.to(torch.float32)
    return out.to(x.dtype)


def chunk_rows(rows: torch.Tensor, chunk_tokens: Sequence[int]) -> torch.Tensor:
    """Expand row ``k`` of ``rows`` ``(1, K, C)`` over ``chunk_tokens[k]`` tokens.

    One modulation row per chunk, broadcast to that chunk's tokens. The
    chunks are given their token counts explicitly rather than derived,
    because the noised slot has ``K`` chunks and the clean slot has ``K - 1``
    of the same size, and a single "tokens per chunk" would hide which of
    the two a caller meant.
    """
    return torch.cat(
        [rows[:, k : k + 1].expand(-1, n, -1) for k, n in enumerate(chunk_tokens)],
        dim=1,
    )


def chunk_causal_positions(
    num_registers: int,
    anchor_hw: tuple[int, int],
    num_text: int,
    noised_thw: tuple[int, int, int],
    num_clean_frames: int,
    shifts: tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """Rotary positions ``(S,)`` int64 of the chunk-causal layout.

    The three slots are kept apart by whole frame blocks rather than by
    running the index on: the registers sit at position 0, the anchor takes
    frame block 0, the text frame block 1, and the noised video starts at
    frame block 2. Packing them contiguously instead would put text tokens a
    few positions from anchor pixels, which at this rotary base is
    indistinguishable from them being the same pixel.

    **Clean frame j carries the positions of noised frame j**, so a clean
    token and its noised counterpart are at the same rotary position and the
    mask is the only thing that tells them apart. ``shifts`` adds the
    ``(anchor, text, video)`` offsets training randomized over; inference
    passes zeros. The clean frames move with the noised ones, since they
    share their positions.
    """
    anchor_shift, text_shift, video_shift = shifts
    regs = torch.zeros(num_registers, dtype=torch.int64)
    anchor = video_positions(1, *anchor_hw, start=anchor_shift)
    text = torch.arange(1, num_text + 1) + FRAME_STRIDE + text_shift
    noised = video_positions(*noised_thw, start=2 * FRAME_STRIDE + video_shift)
    tokens_per_frame = noised_thw[1] * noised_thw[2]
    clean = noised[: num_clean_frames * tokens_per_frame]
    return torch.cat([regs, anchor, text, noised, clean])


def dense_chunk_causal_mask(
    num_registers: int,
    num_anchor: int,
    num_text: int,
    tokens_per_chunk: int,
    n_chunks: int,
    n_clean_chunks: int,
) -> torch.Tensor:
    """The dense ``(S, S)`` visibility mask of one encoder window.

    The parity driver, not the served path: it materializes ``S * S`` bools,
    which at the real window size is tens of gigabytes. The served path
    wants the kernel's block mask, which is the resource M* does not have
    yet.
    """
    codes = encoder_codes(num_registers, num_anchor, num_text, tokens_per_chunk, n_chunks, n_clean_chunks)
    return visible(codes)


def may_defer(index: int, last: int, tap_blocks: Sequence[int]) -> bool:
    """Whether block ``index`` may hand its feed-forward output to the next one.

    A fused add-and-norm kernel can fold a block's second residual add into
    the next block's input norm, which saves a pass over the activations. The
    cost is that the block's output is then never materialized: what exists
    is the pre-add residual plus a separate feed-forward result.

    That is fine in the middle of the stack and wrong in exactly two places,
    which is what this predicate says: **never at the last block**, whose
    output is the tower's, and **never at a tapped block**, whose output the
    action expert reads. Defer at a tap and the tap is the block's residual
    *without its feed-forward contribution*: a tensor of the right shape and
    the right magnitude, missing a third of the block.
    """
    return index != last and index not in tap_blocks


class TrunkBlock(nn.Module):
    """One trunk block: ``x + attn(norm(x) * s)`` then ``x + ffn(norm(x) * s)``.

    ``s`` is ``1 / sqrt(layer_index)`` with a **one-based** ``layer_index``,
    so block 0 scales by 1 and the scaling only starts to bite further down.
    It is a persistent buffer: the checkpoint carries it, rounded to
    bfloat16, and loading overwrites the float32 value computed here. Making
    it non-persistent leaves the exact float32 value in place, which is a
    different number from block 1 on.

    The same ``s`` scales both norms. It is applied to the *normed* tensor
    rather than to the residual branch, so it changes what attention and the
    feed-forward see, not how much of their output is kept.

    The feed-forward is whatever the stack handed in, dense or mixture of
    experts. The block does not know which, and that is deliberate: it is
    the stack's prefix rule that decides, in one place.
    """

    s: torch.Tensor

    def __init__(
        self,
        ffn: nn.Module,
        dim: int = 3072,
        head_dim: int = 128,
        num_kv_heads: int = 8,
        layer_index: int = 1,
    ) -> None:
        super().__init__()
        self.register_buffer("s", torch.tensor(1 / (layer_index**0.5)))
        self.ln_in = Float32RMSNorm(dim)
        self.mha = Attention(dim, head_dim, num_kv_heads)
        self.ln_out = Float32RMSNorm(dim)
        self.ffn = ffn

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        attend: Callable = sdpa_attend,
        pending: torch.Tensor | None = None,
        defer: bool = False,
        modulation: tuple[nn.Module, torch.Tensor, str | None] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``pending`` is the previous block's deferred feed-forward output.

        With ``defer`` the block returns ``(residual, ffn_out)`` instead of
        their sum, leaving the add to the next block. Both are value
        identities in eager mode, which is what makes the deferred path
        testable here: the point of :func:`may_defer` is not the arithmetic
        but which intermediate tensors still exist afterwards.
        """
        if pending is not None:
            x = x + pending
        if modulation is not None:
            modulator, x_mod, dtype = modulation
            shift, scale = modulator(x_mod).chunk(2, dim=-1)
            if dtype in ("mixed", "float32"):
                x = modulate(x, shift, 1 + scale)
            elif dtype == "bfloat16":
                x = (x.to(torch.bfloat16) * (1 + scale).to(torch.bfloat16) + shift.to(torch.bfloat16)).to(x.dtype)
            elif dtype is None:
                x = x * (1 + scale) + shift
            else:
                raise ValueError(f"unsupported modulation dtype: {dtype}")
        normed_in = self.ln_in(x) * self.s
        x = x + self.mha(normed_in, rope, mask, attend)
        normed_out = self.ln_out(x) * self.s
        f = self.ffn(normed_out)
        if defer:
            return x, f
        return x + f


class TrunkStack(nn.Module):
    """The stack of blocks under a tower, with its shared rotary table.

    Blocks ``0 .. num_dense - 1`` carry the dense SwiGLU and the rest carry
    the mixture of experts. See the module docstring for why a prefix and a
    set of indices are not the same thing even when the config lists a
    prefix.

    The rotary table is built once per call and shared by every block, so a
    position error is uniform across the stack rather than drifting, which is
    why a wrong layout here tends to look like a quality problem instead of a
    crash. The decoder tower shares this class; nothing in it is specific to
    the encoder except the defaults.

    ``moe_dispatch_chunk_size`` is the reference ``Transformer``'s argument
    of the same name and reaches every MoE block. It defaults to ``None``
    only because the reference does; every published checkpoint sets it, and
    a stack built without it runs the other MoE combine (see
    :mod:`~mstar.model.praxis1.components.moe`).
    """

    def __init__(
        self,
        depth: int = 14,
        num_dense: int = 2,
        dim: int = 3072,
        head_dim: int = 128,
        num_kv_heads: int = 8,
        dense_hidden: int = 11520,
        moe_hidden: int = 1280,
        num_experts: int = 128,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        rope_effective_dim: int = 64,
        rope_base: int = 26_000_000,
        rope_table_size: int = 1_000_000,
        moe_dispatch_chunk_size: int | None = None,
        mod_dim: int | None = None,
        modulation_dtype: str | None = "mixed",
    ) -> None:
        super().__init__()
        if not 0 <= num_dense <= depth:
            raise ValueError(f"num_dense {num_dense} is outside a depth of {depth}")
        self.rope = RotaryEmbedding(head_dim, rope_effective_dim, rope_base, rope_table_size)
        blocks = []
        for i in range(depth):
            ffn: nn.Module
            if i < num_dense:
                ffn = SwiGLUFeedForward(dim, dense_hidden)
            else:
                ffn = MoEFeedForward(
                    dim,
                    moe_hidden,
                    num_experts,
                    top_k,
                    routed_scaling_factor,
                    dispatch_chunk_size=moe_dispatch_chunk_size,
                )
            # layer_index is one-based: block 0 scales by 1.
            blocks.append(TrunkBlock(ffn, dim, head_dim, num_kv_heads, layer_index=i + 1))
        self.blocks = nn.ModuleList(blocks)
        self.modulators = (
            nn.ModuleList([nn.Linear(mod_dim, 2 * dim) for _ in range(depth)]) if mod_dim is not None else None
        )
        self.modulation_dtype = modulation_dtype

    def rope_table(self, rope_idx: torch.Tensor, seq_len: int) -> torch.Tensor:
        """The rotary table for ``rope_idx`` ``(B, N)``, cut to ``seq_len`` rows."""
        return self.rope(rope_idx)[:, :seq_len]

    def forward(
        self,
        x: torch.Tensor,
        rope_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
        tap_blocks: Sequence[int] = (),
        attend: Callable = sdpa_attend,
        fuse_residual: bool = False,
        x_mod: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Map ``x`` ``(B, N, C)`` to ``(B, N, C)`` and return the taps.

        The taps come out in the order of ``tap_blocks``, and each one is the
        block's output **after the second residual add**. With
        ``fuse_residual`` the blocks defer that add wherever
        :func:`may_defer` allows, which leaves every tap and the output
        unchanged and is why the flag is safe to turn on; it exists so the
        rule is exercised rather than merely documented.
        """
        rope = self.rope_table(rope_idx, x.size(-2))
        taps: dict[int, torch.Tensor] = {}
        pending: torch.Tensor | None = None
        last = len(self.blocks) - 1
        for i, block in enumerate(self.blocks):
            defer = fuse_residual and may_defer(i, last, tap_blocks)
            if self.modulators is not None and x_mod is None:
                raise ValueError("a modulated stack needs x_mod")
            modulation = (self.modulators[i], x_mod, self.modulation_dtype) if self.modulators is not None else None
            result = block(x, rope, mask, attend, pending, defer, modulation)
            x, pending = result if defer else (result, None)
            if i in tap_blocks:
                taps[i] = x
        return x, [taps[i] for i in tap_blocks]


class TextProjection(nn.Module):
    """Text encoder states to trunk tokens: ``norm(linear(gelu(linear(x))))``.

    The hidden width is ``4 * dim``, so this is four times the size of a
    naive adapter, and the tail norm is the ``1 + weight`` RMS norm computed
    in the *input's* dtype rather than the trunk's float32 one. Those are two
    different classes in :mod:`norm_time` and picking the trunk's here is a
    silent precision change.

    The ``cat([x])`` is a copy that fixes the layout handed to the first
    GEMM. See the module docstring.
    """

    def __init__(self, c_in: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(c_in, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            ZeroCenteredRMSNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([x], dim=1))


class VideoInputProjection(nn.Module):
    """Latent channels to tokens: ``(1, C, T, h, w)`` to ``(1, T, h, w, dim)``.

    Channels move to the end, the grid flattens to rows, one ``Linear``
    runs, and the grid is restored. The video projection is
    **shared between the noised and the clean tokens and the anchor**: one
    module, called three times, so a per-slot projection would load the same
    weights three ways and train differently.
    """

    def __init__(self, c_in: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(c_in, dim)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v = v.movedim(1, -1)
        shape = v.shape
        out = self.proj(torch.cat([v.flatten(1, -2)], dim=1))
        return out.view(*shape[:-1], -1)


class EncoderOutputProjection(nn.Module):
    """Trunk tokens to the decoder's conditioning: one ``Linear(dim, dim)``.

    No norm and no activation. The decoder's modulation is what consumes
    this, and it norms its own input.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([x], dim=1))


class EncoderTower(nn.Module):
    """The encoder tower: pack the window, run the stack, return cond and taps.

    ``positive_zero_residual_add`` is read from the checkpoint config rather
    than assumed. See the module docstring for what the ``+ 0.0`` is for.

    The projections live in ``ModuleDict``s keyed by the checkpoint's own
    tensor names, so the reference's state dict loads here unchanged.
    """

    def __init__(
        self,
        core: TrunkStack,
        dim: int = 3072,
        text_dim: int = 2048,
        video_dim: int = 192,
        time_dim: int = 128,
        num_registers: int = 4,
        tap_blocks: tuple[int, ...] = DEFAULT_TAP_BLOCKS,
        positive_zero_residual_add: bool = True,
    ) -> None:
        super().__init__()
        self.registers = nn.Parameter(torch.zeros(1, num_registers, dim))
        self.t_embedding = TimestepEmbedding(time_dim)
        # Only the video tokens are modulated, so there is one projection and
        # it is keyed by the video tensor's name.
        self.t_proj = nn.ModuleDict({VIDEO: nn.Linear(time_dim, 2 * dim)})
        self.input_projections = nn.ModuleDict(
            {TEXT: TextProjection(text_dim, dim), VIDEO: VideoInputProjection(video_dim, dim)}
        )
        self.output_projections = nn.ModuleDict({VIDEO: EncoderOutputProjection(dim)})
        self.core = core
        self.tap_blocks = tuple(tap_blocks)
        self.positive_zero_residual_add = positive_zero_residual_add

    def chunk_modulate(self, flat: torch.Tensor, t_emb: torch.Tensor, chunk_tokens: Sequence[int]) -> torch.Tensor:
        """AdaLN-style modulation of each chunk's tokens by its timestep row.

        ``1 + scale`` is formed in the projection's own dtype, before the
        float32 widening inside :func:`modulate`. Folding the 1 into the
        float32 product instead changes a bfloat16 result.
        """
        shift, scale = self.t_proj[VIDEO](t_emb).chunk(2, dim=-1)
        return modulate(flat, chunk_rows(shift, chunk_tokens), chunk_rows(1 + scale, chunk_tokens))

    def forward_chunk_causal(
        self,
        text: torch.Tensor,
        anchor: torch.Tensor,
        clean: torch.Tensor | None,
        noised: torch.Tensor,
        timesteps: torch.Tensor,
        rope_shifts: tuple[int, int, int] = (0, 0, 0),
        mask: torch.Tensor | None = None,
        attend: Callable = sdpa_attend,
        fuse_residual: bool = False,
    ) -> tuple[torch.Tensor, TapFeatures]:
        """Encode a chunk-causal window of K chunks.

        ``noised`` is ``(1, video_dim, K * F, h, w)``: K noised chunks of F
        frames. ``clean`` is ``(1, video_dim, (K - 1) * F, h, w)``, the clean
        copy of chunks ``0 .. K - 2``, or None for K = 1. ``timesteps`` is
        ``(1, K)``, one per noised chunk; the clean copy is modulated at 0.
        ``anchor`` is a single video frame ``(1, video_dim, 1, h, w)``.

        Returns the conditioning of the **noised** chunks only, shaped
        ``(1, K * F, h, w, dim)``, and the taps. The output projection runs
        over the noised and the clean tokens together and the clean half is
        then dropped: one GEMM of the original shape rather than two, which
        is the shape the original implementation's bits come from.
        """
        kk = int(timesteps.shape[1])
        frames, height, width = (int(d) for d in noised.shape[2:])
        if frames % kk:
            raise ValueError(f"{frames} noised frames do not split into {kk} chunks")
        chunk = frames // kk
        n_clean_frames = 0 if clean is None else int(clean.shape[2])
        if n_clean_frames != (kk - 1) * chunk:
            raise ValueError(f"{kk} chunks need {(kk - 1) * chunk} clean frames, got {n_clean_frames}")
        tpf = height * width

        # One projection call per input, as the original implementation has
        # it: a single call over a concatenation would be one larger GEMM
        # with a different accumulation order.
        anchor_tok = self.input_projections[VIDEO](anchor).flatten(1, -2)
        text_tok = self.input_projections[TEXT](text)
        noised_tok = self.input_projections[VIDEO](noised)
        noised_shape = noised_tok.shape
        noised_tok = self.chunk_modulate(noised_tok.flatten(1, -2), self.t_embedding(timesteps), [chunk * tpf] * kk)
        parts = [self.registers, anchor_tok, text_tok, noised_tok]
        if clean is not None:
            clean_tok = self.input_projections[VIDEO](clean).flatten(1, -2)
            # Timestep 0, not the chunk's own: the clean copy is what the
            # model may already have seen, so it is fully denoised.
            t_clean = self.t_embedding(torch.zeros_like(timesteps))
            parts.append(self.chunk_modulate(clean_tok, t_clean, [chunk * tpf] * (kk - 1)))
        x = torch.cat(parts, dim=1)

        n_reg = self.registers.size(1)
        n_anchor = anchor_tok.size(1)
        n_text = text_tok.size(1)
        n_global = n_reg + n_anchor + n_text
        n_noised = frames * tpf
        n_clean = n_clean_frames * tpf

        # Built on the CPU and moved every call. The reference caches the
        # device copy per window layout, which is a performance change with
        # no effect on the values.
        positions = chunk_causal_positions(
            n_reg,
            (int(anchor.shape[3]), int(anchor.shape[4])),
            n_text,
            (frames, height, width),
            n_clean_frames,
            rope_shifts,
        )
        positions = positions.unsqueeze(0).to(x.device)
        if mask is None:
            mask = dense_chunk_causal_mask(n_reg, n_anchor, n_text, chunk * tpf, kk, kk - 1 if n_clean else 0).to(
                x.device
            )

        out, taps = self.core(x, positions, mask, self.tap_blocks, attend, fuse_residual)
        if self.positive_zero_residual_add:
            # Not dead code: this turns every -0.0 the stack produced into
            # +0.0, which is what the original implementation's output
            # carries. It cannot change a finite value.
            out = out + 0.0
        video = out.narrow(1, n_global, n_noised + n_clean)
        split = [n_noised, n_clean] if n_clean else [n_noised]
        projected = self.output_projections[VIDEO](torch.cat(list(video.split(split, dim=1)), dim=1))
        cond = projected.narrow(1, 0, n_noised).view(*noised_shape[:-1], -1)

        features = TapFeatures(
            taps=list(taps),
            # Block 0's input, not block 0's tap.
            embeddings=x,
            frames_per_chunk=chunk,
            tokens_per_frame=tpf,
            rope_ids=positions,
            global_spans={
                "register": (0, n_reg),
                "anchor": (n_reg, n_anchor),
                "text": (n_reg + n_anchor, n_text),
            },
            noised_span=(n_global, n_noised),
            clean_span=(n_global + n_noised, n_clean) if n_clean else None,
            n_chunks=kk,
        )
        return cond, features
