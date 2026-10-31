"""A speculative step's bookkeeping as one kernel per phase.

What a verify step does around its layers is integer work on a few rows: read
the registers, address the step's writes, compare drafts, cut stop ids, address
the draft passes. Spelled in torch it is 60-odd small kernels at c=1, each paying
the launch floor of a replayed graph, so a Triton program per row does each
phase in one launch.

``mstar::spec_verify_inputs`` runs before the layers, ``mstar::spec_accept``
after the target's tokens are known. The torch references are the semantics,
and serve CPU tensors and devices without an implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.kernels.numerics import Tolerance
from mstar.kernels.registry import KernelOp, ShapePolicy


@triton.jit
def _verify_inputs_kernel(
    rows_ptr, real_ptr, seq_len_ptr, num_acc_ptr, tokens_ptr, block_table_ptr,
    ids_ptr, positions_ptr, slots_ptr, pages_ptr, offsets_ptr, kv_lens_ptr, num_accepted_ptr,
    registers, table_stride, page_size, T,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    row = tl.load(rows_ptr + r)
    real = tl.load(real_ptr + r) != 0
    base = tl.where(real, tl.load(seq_len_ptr + row), 0)
    tl.store(num_accepted_ptr + r, tl.where(real, tl.load(num_acc_ptr + row), 1))
    tl.store(kv_lens_ptr + r, base + T)

    slot = tl.arange(0, BLOCK)
    mask = slot < T
    position = base + slot
    entry = tl.load(block_table_ptr + r * table_stride + position // page_size, mask=mask, other=0)
    offset = position % page_size
    at = r * T + slot
    tl.store(ids_ptr + at, tl.load(tokens_ptr + slot * registers + row, mask=mask, other=0), mask=mask)
    tl.store(positions_ptr + at, position, mask=mask)
    tl.store(pages_ptr + at, entry, mask=mask)
    tl.store(offsets_ptr + at, offset, mask=mask)
    tl.store(slots_ptr + at, entry.to(tl.int64) * page_size + offset, mask=mask)


def _verify_inputs(
    rows, real, seq_len, num_acc, tokens, block_table, page_size, ids, positions, slots,
    write_pages, write_offsets, kv_lens, num_accepted,
) -> None:
    b, t = rows.shape[0], tokens.shape[0]
    _verify_inputs_kernel[(b,)](
        rows, real, seq_len, num_acc, tokens, block_table,
        ids, positions, slots, write_pages, write_offsets, kv_lens, num_accepted,
        tokens.shape[1], block_table.stride(0), page_size, t,
        BLOCK=triton.next_power_of_2(t), num_warps=1,
    )


def _verify_inputs_reference(
    rows, real, seq_len, num_acc, tokens, block_table, page_size, ids, positions, slots,
    write_pages, write_offsets, kv_lens, num_accepted,
) -> None:
    b, t = rows.shape[0], tokens.shape[0]
    rows_l, real_b = rows.long(), real.bool()
    base = torch.where(real_b, seq_len[rows_l], 0)
    num_accepted.copy_(torch.where(real_b, num_acc[rows_l], 1))
    kv_lens.copy_(base + t)
    position = base[:, None] + torch.arange(t, dtype=base.dtype, device=base.device)[None, :]
    page = torch.div(position, page_size, rounding_mode="floor")
    entry = block_table[:b].gather(1, page.long()).long()
    offset = (position - page * page_size).long()
    ids.copy_(tokens[:, rows_l].T.reshape(-1))
    positions.copy_(position.reshape(-1).long())
    write_pages.copy_(entry.reshape(-1))
    write_offsets.copy_(offset.reshape(-1))
    slots.copy_((entry * page_size + offset).reshape(-1))


def _verify_inputs_fake(*args) -> None:
    return None


def _verify_inputs_shape(rows, real, seq_len, num_acc, tokens, *rest):
    return {"rows": rows.shape[0], "tokens": tokens.shape[0]}


def _verify_inputs_args(shape, dtype, device, generator=None):
    b, t = shape["rows"], shape["tokens"]
    registers, page_size, width = b + 1, 8, 32

    def i32(values):
        return values.to(torch.int32).to(device)

    rows = i32(torch.arange(b, 0, -1))
    real = i32(torch.ones(b).long())
    # a padding row: it reads the sink row's registers as a fresh request
    real[-1] = 0
    seq_len = i32(torch.arange(registers) * 11 + 3)
    num_acc = i32(torch.arange(registers) % t + 1)
    tokens = (torch.arange(t * registers).view(t, registers) * 7).to(device)
    block_table = i32(1 + torch.arange(b * width).view(b, width))
    long = dict(dtype=torch.long, device=device)
    return (rows, real, seq_len, num_acc, tokens, block_table, page_size,
            torch.zeros(b * t, **long), torch.zeros(b * t, **long), torch.zeros(b * t, **long),
            torch.zeros(b * t, **long), torch.zeros(b * t, **long),
            torch.zeros(b, dtype=torch.int32, device=device), torch.zeros(b, dtype=torch.int32, device=device))


SPEC_VERIFY_INPUTS = KernelOp(
    "spec_verify_inputs",
    "(Tensor rows, Tensor real, Tensor seq_len, Tensor num_acc, Tensor tokens, Tensor block_table, int page_size, "
    "Tensor(a!) ids, Tensor(b!) positions, Tensor(c!) slots, Tensor(d!) write_pages, Tensor(e!) write_offsets, "
    "Tensor(f!) kv_lens, Tensor(g!) num_accepted) -> ()",
    fake=_verify_inputs_fake,
    reference=_verify_inputs_reference,
    impls={"rocm": lambda: _verify_inputs},
    fallback_to_reference=True,
    shapes=ShapePolicy(
        dims=("rows", "tokens"),
        key=_verify_inputs_shape,
        inputs=_verify_inputs_args,
        warm=({"rows": 4, "tokens": 3},),
        dtypes=(torch.int32,),
    ),
    tolerance=Tolerance.exact(),
)


@triton.jit
def _accept_kernel(
    rows_ptr, ids_ptr, targets_ptr, positions_ptr, block_table_ptr, stop_ids_ptr,
    seq_len_ptr, num_acc_ptr, tokens_ptr, emitted_ptr, budget_ptr, stop_on_ptr,
    out_tokens_ptr, accepted_ptr, done_ptr, keep_ptr,
    draft_slots_ptr, draft_positions_ptr, draft_lens_ptr, kept_ptr,
    table_stride, page_size, num_stops, num_rows, T, K,
    BLOCK: tl.constexpr, STOPS: tl.constexpr, PASSES: tl.constexpr, KEPT: tl.constexpr,
):
    r = tl.program_id(0)
    row = tl.load(rows_ptr + r)
    slot = tl.arange(0, BLOCK)
    mask = slot < T
    target = tl.load(targets_ptr + r * T + slot, mask=mask, other=-1)
    if KEPT:
        # a sampler has judged the drafts
        last = tl.load(kept_ptr + r)
    else:
        drafted = tl.load(ids_ptr + r * T + slot + 1, mask=slot + 1 < T, other=-2)
        # the first slot whose next draft the target refused ends the step: never
        # one past the last, which has no draft after it
        last = tl.argmin((drafted == target).to(tl.int32), axis=0)
    accepted = last + 1
    base = tl.load(positions_ptr + r * T).to(tl.int32)

    stop = tl.arange(0, STOPS)
    stop_ids = tl.load(stop_ids_ptr + stop, mask=stop < num_stops, other=-2)
    is_stop = tl.max((target[:, None] == stop_ids[None, :]).to(tl.int32), axis=1) > 0
    emit = slot <= last
    stops = is_stop & emit & (tl.load(stop_on_ptr + row) > 0)
    any_stop = tl.max(stops.to(tl.int32), axis=0) > 0
    first_stop = tl.where(any_stop, tl.argmax(stops.to(tl.int32), axis=0), T)
    emitted = tl.load(emitted_ptr + row)
    budget = tl.load(budget_ptr + row)
    emit = emit & (slot <= first_stop) & (slot < budget - emitted)
    emitted = emitted + tl.sum(emit.to(tl.int32), axis=0)

    tl.store(out_tokens_ptr + r * T + slot, tl.where(emit, target, -1), mask=mask)
    tl.store(accepted_ptr + r, accepted)
    tl.store(done_ptr + r, any_stop | (emitted >= budget))
    tl.store(keep_ptr + r, r * T + last)
    tl.store(seq_len_ptr + row, base + accepted)
    tl.store(num_acc_ptr + row, accepted)
    tl.store(emitted_ptr + row, emitted)
    tl.store(tokens_ptr + row, tl.load(targets_ptr + r * T + last))

    # draft pass j's one real query: the last accepted slot, j positions on
    j = tl.arange(0, PASSES)
    in_pass = j < K
    position = base + last + j
    entry = tl.load(block_table_ptr + r * table_stride + position // page_size, mask=in_pass, other=0)
    at = j * num_rows + r
    tl.store(draft_slots_ptr + at, entry.to(tl.int64) * page_size + position % page_size, mask=in_pass)
    tl.store(draft_positions_ptr + at, position, mask=in_pass)
    tl.store(draft_lens_ptr + at, position + 1, mask=in_pass)


def _accept(
    rows, ids, targets, positions, block_table, page_size, stop_ids,
    seq_len, num_acc, tokens, emitted, budget, stop_on,
    out_tokens, accepted, done, keep, draft_slots, draft_positions, draft_lens, kept=None,
) -> None:
    b, k = rows.shape[0], draft_slots.shape[0]
    t = out_tokens.shape[1]
    _accept_kernel[(b,)](
        rows, ids, targets, positions, block_table, stop_ids,
        seq_len, num_acc, tokens, emitted, budget, stop_on,
        out_tokens, accepted, done, keep, draft_slots, draft_positions, draft_lens,
        rows if kept is None else kept,
        block_table.stride(0), page_size, stop_ids.shape[0], b, t, k,
        BLOCK=triton.next_power_of_2(t), STOPS=triton.next_power_of_2(stop_ids.shape[0]),
        PASSES=triton.next_power_of_2(k), KEPT=kept is not None, num_warps=1,
    )


def _accept_reference(
    rows, ids, targets, positions, block_table, page_size, stop_ids,
    seq_len, num_acc, tokens, emitted, budget, stop_on,
    out_tokens, accepted, done, keep, draft_slots, draft_positions, draft_lens, kept=None,
) -> None:
    b, t, k = rows.shape[0], out_tokens.shape[1], draft_slots.shape[0]
    rows_l = rows.long()
    y = targets.view(b, t).long()
    x = ids.view(b, t)
    base = positions.view(b, t)[:, 0].int()
    count = ((x[:, 1:] == y[:, :-1]).int().cumprod(dim=1).sum(dim=1) if kept is None else kept) + 1
    last = (count - 1)[:, None].long()
    accepted.copy_(count.int())
    seq_len.index_copy_(0, rows_l, base + count.int())
    num_acc.index_copy_(0, rows_l, count.int())
    tokens[0].index_copy_(0, rows_l, y.gather(1, last)[:, 0])

    slots = torch.arange(t, dtype=torch.int32, device=rows.device)[None, :]
    emit = slots < count[:, None]
    stops = torch.isin(y, stop_ids) & emit & (stop_on[rows_l] > 0)[:, None]
    hit = stops.any(dim=1)
    first_stop = torch.where(hit, stops.int().argmax(dim=1), t)
    emit &= slots <= first_stop[:, None]
    so_far = emitted[rows_l]
    emit &= slots < (budget[rows_l] - so_far)[:, None]
    so_far = so_far + emit.sum(dim=1, dtype=torch.int32)
    emitted.index_copy_(0, rows_l, so_far)
    out_tokens.copy_(torch.where(emit, y, -1))
    done.copy_(hit | (so_far >= budget[rows_l]))
    keep.copy_(torch.arange(b, device=rows.device) * t + last[:, 0])

    shift = torch.arange(k, dtype=torch.int32, device=rows.device)[:, None]
    position = (base + count.int() - 1)[None, :] + shift
    page = torch.div(position, page_size, rounding_mode="floor")
    entry = block_table[:b].T.gather(0, page.long()).long()
    draft_slots.copy_(entry * page_size + (position - page * page_size).long())
    draft_positions.copy_(position.long())
    draft_lens.copy_(position + 1)


def _accept_fake(*args) -> None:
    return None


def _accept_shape(rows, ids, targets, positions, block_table, page_size, stop_ids, seq_len, num_acc, tokens, *rest):
    return {"rows": rows.shape[0], "tokens": tokens.shape[0]}


def _accept_args(shape, dtype, device, generator=None):
    b, t = shape["rows"], shape["tokens"]
    k = t - 1
    registers, page_size, width = b + 1, 8, 32

    def i32(values):
        return values.to(torch.int32).to(device)

    long = dict(dtype=torch.long, device=device)
    rows = i32(1 + torch.arange(b))
    ids = (torch.arange(b * t) % 5).to(device)
    # each row agrees with its drafts for a row-dependent stretch; every third row meets a stop id
    targets = ids.roll(-1).clone()
    for r in range(b):
        targets[r * t + r % t] = 99 if r % 3 else 7
    positions = (torch.arange(b)[:, None] * 13 + torch.arange(t)[None, :] + 20).reshape(-1).to(device)
    block_table = i32(1 + torch.arange(b * width).view(b, width))
    return (rows, ids, targets, positions, block_table, page_size, torch.tensor([7, 8], **long),
            torch.zeros(registers, dtype=torch.int32, device=device),
            torch.ones(registers, dtype=torch.int32, device=device),
            torch.zeros(t, registers, **long),
            i32(torch.arange(registers)),
            torch.full((registers,), 5, dtype=torch.int32, device=device),
            i32(torch.arange(registers) % 2),
            torch.zeros(b, t, **long), torch.zeros(b, dtype=torch.int32, device=device),
            torch.zeros(b, dtype=torch.bool, device=device), torch.zeros(b, **long),
            torch.zeros(k, b, **long), torch.zeros(k, b, **long),
            torch.zeros(k, b, dtype=torch.int32, device=device))


SPEC_ACCEPT = KernelOp(
    "spec_accept",
    "(Tensor rows, Tensor ids, Tensor targets, Tensor positions, Tensor block_table, int page_size, Tensor stop_ids, "
    "Tensor(a!) seq_len, Tensor(b!) num_acc, Tensor(c!) tokens, Tensor(d!) emitted, Tensor budget, Tensor stop_on, "
    "Tensor(e!) out_tokens, Tensor(f!) accepted, Tensor(g!) done, Tensor(h!) keep, Tensor(i!) draft_slots, "
    "Tensor(j!) draft_positions, Tensor(k!) draft_lens, Tensor? kept=None) -> ()",
    fake=_accept_fake,
    reference=_accept_reference,
    impls={"rocm": lambda: _accept},
    fallback_to_reference=True,
    shapes=ShapePolicy(
        dims=("rows", "tokens"),
        key=_accept_shape,
        inputs=_accept_args,
        warm=({"rows": 6, "tokens": 3}, {"rows": 6, "tokens": 4}),
        dtypes=(torch.int32,),
    ),
    tolerance=Tolerance.exact(),
)
