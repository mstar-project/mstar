"""Sampling rows that filter nothing: the argmax of ``logits / T`` plus Gumbel noise.

A token drawn from ``softmax(logits / T)`` with no top-k, top-p, min-p or
penalty is the argmax of ``logits / T + g``, ``g`` standard Gumbel noise per
token, so one read of the logits draws it. No probabilities are written and no
rejection sampler reads them back, which is most of what the full sampler costs
a decode step. A row at temperature 0 takes the argmax of its logits, the first
index among equal maxima, the token the full sampler's greedy rows get.

The noise is Philox keyed on the row's seed, counted by (vocab index / 4, the
row's offset), and a draw advances the offset by one as the full sampler's draws
do, so a request moving between the two paths never reuses a step's numbers.

``mstar::sample_unfiltered`` also does the step's bookkeeping around the draw,
so a captured step adds no kernels for it: given a slot index (in-graph
scatter), each row's offset is read from and written back to the slot masters
and the token goes to ``token_master`` by slot; without one, the step's own
offset row advances and the token goes to ``token_master`` by row.

A submodule opts a walk in with ``with_unfiltered_sampling`` (a second capture
of its batched configs, keyed ``UNFILTERED_SAMPLING``) and answers its
``cg_key_info`` from ``SamplerResource.unfiltered_key``; the sampler runs the
draw on the steps that replay that capture. Only ROCm has an implementation so
far, so elsewhere the helper adds nothing and every step samples as before.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from mstar.kernels.numerics import Tolerance
from mstar.kernels.registry import KernelOp, ShapePolicy

if TYPE_CHECKING:
    from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig
    from mstar.engine.resources.sampler.utils import SamplingConfig

# The capture key of a bucket whose batches filter nothing
UNFILTERED_SAMPLING = "unfiltered_sampling"

# A program reads one block of a row. Small batches cut the vocab finer so the
# whole GPU reads the logits; large ones use bigger blocks and fewer programs.
_MIN_BLOCK = 256
_MAX_BLOCK = 8192
_TARGET_PROGRAMS = 2048


def _block_for(batch: int, vocab: int) -> int:
    per_row = -(-_TARGET_PROGRAMS // max(batch, 1))
    return max(_MIN_BLOCK, min(_MAX_BLOCK, triton.next_power_of_2(-(-vocab // per_row))))


@triton.jit
def _gumbel(seed, start, off, BLOCK: tl.constexpr):
    """Standard Gumbel noise for vocab indices ``start + [0, BLOCK)`` at row
    offset ``off``. One Philox call gives four words, so index ``i`` takes word
    ``i % 4`` of counter ``i // 4``: a quarter of the integer work."""
    c0 = (start // 4 + tl.arange(0, BLOCK // 4)).to(tl.uint32)
    zero = c0 * 0
    lo = zero + (off & 0xFFFFFFFF).to(tl.uint32)
    hi = zero + ((off >> 32) & 0xFFFFFFFF).to(tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed, c0, lo, hi, zero)
    # [i, j, k] holds word 2j + k of counter i, so the flat order is the vocab order
    r = tl.reshape(tl.join(tl.join(r0, r2), tl.join(r1, r3)), [BLOCK])
    # 24 bits, centred, so u is never 0 or 1
    u = ((r >> 8).to(tl.float32) + 0.5) * (1.0 / 16777216.0)
    return -tl.log(-tl.log(u))


@triton.jit
def _partials_kernel(
    logits_ptr, temperature_ptr, seed_ptr, offset_ptr, slot_ptr, offset_master_ptr,
    part_val_ptr, part_idx_ptr,
    V, stride_b, NSPLIT,
    BLOCK: tl.constexpr,
    SLOTS: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    offs = split * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    x = tl.load(logits_ptr + row.to(tl.int64) * stride_b + offs, mask=mask, other=-float("inf")).to(tl.float32)
    temp = tl.load(temperature_ptr + row)
    if temp > 0:
        if SLOTS:
            off = tl.load(offset_master_ptr + tl.load(slot_ptr + row))
        else:
            off = tl.load(offset_ptr + row)
        score = x * (1.0 / temp) + _gumbel(tl.load(seed_ptr + row), split * BLOCK, off, BLOCK)
        score = tl.where(mask, score, -float("inf"))
    else:
        score = x
    # a NaN logit never wins, so a row of them still draws an id inside the vocab
    score = tl.where(score == score, score, -float("inf"))  # noqa: PLR0124  (NaN test)
    best = tl.max(score, axis=0)
    # the first index among equal maxima, as the full sampler's greedy rows
    idx = tl.argmax(score, axis=0)
    tl.store(part_val_ptr + row * NSPLIT + split, best)
    tl.store(part_idx_ptr + row * NSPLIT + split, tl.minimum(split * BLOCK + idx, V - 1))


@triton.jit
def _combine_kernel(
    part_val_ptr, part_idx_ptr, tokens_ptr, offset_ptr,
    slot_ptr, offset_master_ptr, token_master_ptr,
    NSPLIT, STEP,
    BLOCK_N: tl.constexpr,
    SLOTS: tl.constexpr,
    TOKEN_MASTER: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.arange(0, BLOCK_N)
    vals = tl.load(part_val_ptr + row * NSPLIT + n, mask=n < NSPLIT, other=-float("inf"))
    # the lowest block wins a tie, so the token is the first index of the max
    win = tl.argmax(vals, axis=0)
    token = tl.load(part_idx_ptr + row * NSPLIT + win).to(tl.int64)
    tl.store(tokens_ptr + row, token)
    if SLOTS:
        slot = tl.load(slot_ptr + row)
        off = tl.load(offset_master_ptr + slot) + STEP
        tl.store(offset_master_ptr + slot, off)
        where = slot
    else:
        off = tl.load(offset_ptr + row) + STEP
        where = row
    tl.store(offset_ptr + row, off)
    if TOKEN_MASTER:
        tl.store(token_master_ptr + where, token.to(token_master_ptr.dtype.element_ty))


def _triton_impl():
    def sample_unfiltered(
        logits: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        tokens: torch.Tensor,
        offset_step: int,
        slot_index: torch.Tensor | None = None,
        offset_master: torch.Tensor | None = None,
        token_master: torch.Tensor | None = None,
    ) -> None:
        batch, vocab = logits.shape
        if logits.stride(1) != 1:
            logits = logits.contiguous()
        block = _block_for(batch, vocab)
        nsplit = -(-vocab // block)
        slots = slot_index is not None
        part_val = torch.empty((batch, nsplit), device=logits.device, dtype=torch.float32)
        part_idx = torch.empty((batch, nsplit), device=logits.device, dtype=torch.int32)
        dummy = offset
        with torch.cuda.device(logits.device):
            _partials_kernel[(batch, nsplit)](
                logits, temperature, seed, offset,
                slot_index if slots else dummy, offset_master if slots else dummy,
                part_val, part_idx,
                vocab, logits.stride(0), nsplit,
                BLOCK=block, SLOTS=slots,
                num_warps=8 if block >= 4096 else 4,
            )
            _combine_kernel[(batch,)](
                part_val, part_idx, tokens, offset,
                slot_index if slots else dummy, offset_master if slots else dummy,
                token_master if token_master is not None else dummy,
                nsplit, offset_step,
                BLOCK_N=triton.next_power_of_2(nsplit), SLOTS=slots,
                TOKEN_MASTER=token_master is not None,
                num_warps=4,
            )

    return sample_unfiltered


# --- torch reference: the same Philox stream, for CPU and the parity tests ---

_MASK32 = 0xFFFFFFFF


def _mul32(a: int, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(high, low) 32-bit halves of ``a * b`` for a 32-bit constant and int64 tensors holding uint32."""
    a_hi, a_lo = a >> 16, a & 0xFFFF
    t_hi, t_lo = a_hi * b, a_lo * b  # each below 2**48
    high = (t_hi + (t_lo >> 16)) >> 16
    low = (((t_hi & 0xFFFF) << 16) + t_lo) & _MASK32
    return high, low


def philox4x32(seed: int, c0, c1, c2, c3, rounds: int = 10):
    """Philox-4x32 as ``triton.language.philox`` computes it, on int64 tensors holding uint32."""
    k0, k1 = seed & _MASK32, (seed >> 32) & _MASK32
    for _ in range(rounds):
        hi_b, lo_b = _mul32(0xCD9E8D57, c2)
        hi_a, lo_a = _mul32(0xD2511F53, c0)
        c0, c1, c2, c3 = hi_b ^ c1 ^ k0, lo_b, hi_a ^ c3 ^ k1, lo_a
        k0, k1 = (k0 + 0x9E3779B9) & _MASK32, (k1 + 0xBB67AE85) & _MASK32
    return c0, c1, c2, c3


def gumbel_reference(seed: int, offset: int, vocab: int, device=None) -> torch.Tensor:
    """The kernel's noise for one row: word ``i % 4`` of Philox counter ``i // 4``."""
    c0 = torch.arange(-(-vocab // 4), dtype=torch.int64, device=device)
    zero = torch.zeros_like(c0)
    words = philox4x32(seed & (2**64 - 1), c0, zero + (offset & _MASK32), zero + ((offset >> 32) & _MASK32), zero)
    r = torch.stack(words, dim=1).reshape(-1)[:vocab]
    u = ((r >> 8).to(torch.float32) + 0.5) * (1.0 / 16777216.0)
    return -torch.log(-torch.log(u))


def _reference(logits, temperature, seed, offset, tokens, offset_step, slot_index=None, offset_master=None,
               token_master=None) -> None:
    rows = slot_index.tolist() if slot_index is not None else None
    for r in range(logits.shape[0]):
        off = int(offset_master[rows[r]]) if rows is not None else int(offset[r])
        x = logits[r].float()
        temp = float(temperature[r])
        if temp > 0:
            # x * (1 / T) in fp32, as the kernel rounds it
            inv = 1.0 / temperature[r:r + 1].float()
            x = x * inv + gumbel_reference(int(seed[r]), off, x.shape[0], x.device)
        token = int(torch.argmax(torch.nan_to_num(x, nan=-float("inf"))))
        tokens[r] = token
        offset[r] = off + offset_step
        if rows is not None:
            offset_master[rows[r]] = off + offset_step
        if token_master is not None:
            token_master[rows[r] if rows is not None else r] = token


def _fake(logits, temperature, seed, offset, tokens, offset_step, slot_index=None, offset_master=None,
          token_master=None) -> None:
    return None


def _inputs(shape, dtype, device, generator=None):
    batch, vocab = shape["rows"], shape["vocab"]
    logits = torch.randn(batch, vocab, dtype=dtype, device=device, generator=generator) * 4
    # greedy rows only: the sampled rows' noise is checked against the reference in the sampler tests
    temperature = torch.zeros(batch, dtype=torch.float32, device=device)
    seed = torch.arange(batch, dtype=torch.int64, device=device) + 7
    offset = torch.zeros(batch, dtype=torch.int64, device=device)
    tokens = torch.zeros(batch, dtype=torch.int64, device=device)
    # the in-graph scatter's arguments too, so every argument the op writes is compared: rows on
    # slots 1..batch, slot 0 left as the trash row
    slots = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
    offset_master = torch.arange(batch + 1, dtype=torch.int64, device=device) * 3
    token_master = torch.zeros(batch + 1, dtype=torch.int64, device=device)
    return logits, temperature, seed, offset, tokens, 1, slots, offset_master, token_master


SAMPLE_UNFILTERED = KernelOp(
    "sample_unfiltered",
    "(Tensor logits, Tensor temperature, Tensor seed, Tensor(a!) offset, Tensor(b!) tokens, int offset_step, "
    "Tensor? slot_index=None, Tensor(c!)? offset_master=None, Tensor(d!)? token_master=None) -> ()",
    fake=_fake,
    reference=_reference,
    impls={"rocm": _triton_impl},
    shapes=ShapePolicy(
        dims=("rows", "vocab"),
        key=lambda logits, *args, **kwargs: {"rows": logits.shape[0], "vocab": logits.shape[1]},
        inputs=_inputs,
        warm=({"rows": 1, "vocab": 248320}, {"rows": 64, "vocab": 248320}),
        dtypes=(torch.bfloat16, torch.float32),
    ),
    tolerance=Tolerance.exact(),
)


def filters_nothing(cfg: "SamplingConfig") -> bool:
    """Whether the unfiltered draw samples ``cfg``'s request exactly as the full sampler would."""
    from mstar.engine.resources.sampler.utils import canonical_sampling_values

    _, top_k, top_p, min_p = canonical_sampling_values(cfg.temperature, cfg.top_k, cfg.top_p, cfg.min_p)
    return (
        top_k == 0 and top_p >= 1.0 and min_p == 0.0 and cfg.repetition_penalty == 1.0
        and not cfg.uses_history and not cfg.uses_filter_order
    )


def with_unfiltered_sampling(
    configs: Sequence["AcceleratorGraphConfig"], walks: Sequence[str],
) -> list["AcceleratorGraphConfig"]:
    """``configs`` plus, for each that captures one of ``walks`` under no other
    key, a copy keyed ``UNFILTERED_SAMPLING``; ``configs`` alone where the draw
    has no implementation. Call where the configs are built (on the device)."""
    if SAMPLE_UNFILTERED.unavailable_reason() is not None:
        return list(configs)
    extra = []
    for config in configs:
        if config.capture_graph_walk in walks and config.additional_key_info is None:
            keyed = copy.copy(config)
            keyed.additional_key_info = UNFILTERED_SAMPLING
            extra.append(keyed)
    return [*configs, *extra]
