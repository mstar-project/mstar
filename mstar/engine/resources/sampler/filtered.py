"""One launch for a sampled step with top-k / top-p and no penalty, at small vocabularies.

The full graph sampler runs a softmax, a top-k renorm, a top-p draw and a handful
of small kernels around them, about nine launches per call. A speech code
predictor calls it fifteen times a frame. A token drawn from the renormalised
kept set is the argmax of ``logits / T`` plus Gumbel noise over that set, so
``mstar::sample_filtered`` does it from one read of the logits: one program per
row holds the whole row in registers, finds the kept set by a bitwise search for
its cutoff, adds the noise and writes the token and the advanced offset.

Which tokens are kept follows ``filter_top_k_top_p``: top-k first, then top-p over
the renormalised rest, ties included. The noise is ``unfiltered._gumbel``'s, so
rows draw independently of their position in the batch. A row at temperature 0
takes the first index of the max, as the full sampler's greedy rows do.

A node opts in with ``SamplerSpec.fused_draw``, up to ``FUSED_FILTER_MAX_VOCAB``,
past which a row no longer fits a program's registers. Only ROCm has an
implementation, so elsewhere the node samples as before.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.engine.resources.sampler.unfiltered import _gumbel, gumbel_reference
from mstar.engine.resources.sampler.utils import _kth_largest_bits
from mstar.kernels.numerics import Tolerance
from mstar.kernels.registry import KernelOp, ShapePolicy

_WARPS = 4


@triton.jit
def _draw_kernel(
    logits_ptr, temperature_ptr, top_k_ptr, top_p_ptr, seed_ptr, offset_ptr, tokens_ptr,
    V, stride_b, STEP,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    valid = offs < V
    x = tl.load(logits_ptr + row.to(tl.int64) * stride_b + offs, mask=valid, other=-float("inf")).to(tl.float32)
    # a NaN logit never wins, so a row of them still draws an id inside the vocab
    x = tl.where(x == x, x, -float("inf"))  # noqa: PLR0124  (NaN test)
    temp = tl.load(temperature_ptr + row)
    off = tl.load(offset_ptr + row)
    score = x
    if temp > 0:
        scaled = x * (1.0 / temp)
        # the softmax up to a constant: same order, non-negative, so the float bits order the tokens
        p = tl.where(valid, tl.exp(scaled - tl.max(scaled, axis=0)), 0.0)
        p = tl.where(p == p, p, 0.0)  # noqa: PLR0124  (NaN test)
        bits = p.to(tl.int32, bitcast=True)
        top_k = tl.load(top_k_ptr + row)
        cutoff = 0
        if top_k > 0:  # branches, not tl.where: skip the search when nothing is cut
            if top_k < V:
                cutoff = _kth_largest_bits(bits, top_k)
        top_p = tl.load(top_p_ptr + row)
        if top_p < 1.0:
            # top-p keeps token i iff the kept mass strictly above it is < p * mass. That
            # mass falls as i's value rises, so search for the smallest such value.
            kept = bits >= cutoff
            target = top_p * tl.sum(tl.where(kept, p, 0.0), axis=0)
            lo = 0
            hi = 0x7F800000
            for _ in range(31):
                mid = lo + (hi - lo) // 2
                above = tl.sum(tl.where(kept & (bits > mid), p, 0.0), axis=0)
                ok = above < target
                hi = tl.where(ok, mid, hi)
                lo = tl.where(ok, lo, mid + 1)
            cutoff = tl.maximum(lo, cutoff)
        noise = _gumbel(tl.load(seed_ptr + row), 0, off, BLOCK)
        score = tl.where(valid & (bits >= cutoff), scaled + noise, -float("inf"))
    tl.store(tokens_ptr + row, tl.minimum(tl.argmax(score, axis=0), V - 1).to(tl.int64))
    tl.store(offset_ptr + row, off + STEP)


def _triton_impl():
    def sample_filtered(logits, temperature, top_k, top_p, seed, offset, tokens, offset_step) -> None:
        batch, vocab = logits.shape
        if logits.stride(1) != 1:
            logits = logits.contiguous()
        with torch.cuda.device(logits.device):
            _draw_kernel[(batch,)](
                logits, temperature, top_k, top_p, seed, offset, tokens,
                vocab, logits.stride(0), offset_step,
                BLOCK=triton.next_power_of_2(vocab), num_warps=_WARPS,
            )

    return sample_filtered


def kept_tokens(scaled: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """The tokens a row keeps, for scores ``scaled`` (``logits / T``); the reference's filter."""
    vocab = scaled.shape[0]
    p = torch.exp(scaled - scaled.max())
    kept = torch.ones(vocab, dtype=torch.bool, device=scaled.device)
    if 0 < top_k < vocab:
        kept = p >= torch.sort(p, descending=True).values[top_k - 1]
    if top_p < 1.0:
        mass = p[kept].sum()
        above = torch.stack([p[kept & (p > v)].sum() for v in p])
        kept = kept & (above < top_p * mass)
    return kept


def _reference(logits, temperature, top_k, top_p, seed, offset, tokens, offset_step) -> None:
    for r in range(logits.shape[0]):
        x = torch.nan_to_num(logits[r].float(), nan=-float("inf"))
        temp = float(temperature[r])
        score = x
        if temp > 0:
            scaled = x * (1.0 / temperature[r:r + 1].float())
            kept = kept_tokens(scaled, int(top_k[r]), float(top_p[r]))
            noise = gumbel_reference(int(seed[r]), int(offset[r]), x.shape[0], x.device)
            score = torch.where(kept, scaled + noise, -float("inf"))
        tokens[r] = int(torch.argmax(score))
        offset[r] = int(offset[r]) + offset_step


def _fake(logits, temperature, top_k, top_p, seed, offset, tokens, offset_step) -> None:
    return None


def _inputs(shape, dtype, device, generator=None):
    batch, vocab = shape["rows"], shape["vocab"]
    logits = torch.randn(batch, vocab, dtype=dtype, device=device, generator=generator) * 4
    # greedy rows only: the sampled rows' noise is checked against the reference in the sampler tests
    temperature = torch.zeros(batch, dtype=torch.float32, device=device)
    top_k = torch.full((batch,), 50, dtype=torch.int32, device=device)
    top_p = torch.ones(batch, dtype=torch.float32, device=device)
    seed = torch.arange(batch, dtype=torch.int64, device=device) + 7
    offset = torch.zeros(batch, dtype=torch.int64, device=device)
    tokens = torch.zeros(batch, dtype=torch.int64, device=device)
    return logits, temperature, top_k, top_p, seed, offset, tokens, 1


SAMPLE_FILTERED = KernelOp(
    "sample_filtered",
    "(Tensor logits, Tensor temperature, Tensor top_k, Tensor top_p, Tensor seed, Tensor(a!) offset, "
    "Tensor(b!) tokens, int offset_step) -> ()",
    fake=_fake,
    reference=_reference,
    impls={"rocm": _triton_impl},
    shapes=ShapePolicy(
        dims=("rows", "vocab"),
        key=lambda logits, *args, **kwargs: {"rows": logits.shape[0], "vocab": logits.shape[1]},
        inputs=_inputs,
        warm=({"rows": 1, "vocab": 2048}, {"rows": 64, "vocab": 2048}),
        dtypes=(torch.bfloat16, torch.float32),
    ),
    tolerance=Tolerance.exact(),
)


def available() -> bool:
    """Whether this platform runs the fused draw."""
    return SAMPLE_FILTERED.unavailable_reason() is None
