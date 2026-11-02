"""Ulysses sequence parallelism — model-agnostic primitives.

Sequence parallelism (SP) shards the token sequence across an ``sp_group`` and
replicates the weights, the mirror image of tensor parallelism (which shards the
weights and replicates the sequence). Attention is the only operator that needs
the whole sequence, so Ulysses brackets it with two all-to-alls: the first turns
a sequence-sharded ``[seq/P, heads, dim]`` tensor into a head-sharded
``[seq, heads/P, dim]`` one (full sequence, this rank's slice of heads), the
attention runs locally, and the second converts back. Net effect: attention runs
exactly as if at tensor-parallel degree ``tp*sp`` — the kernel (``run_attention``)
is unchanged — while every pointwise op (norms, MLP, residuals) runs on the
``seq/P`` shard for free.

These helpers take an ``sp_group`` (a :class:`CommGroup` over the SP axis of the
mesh; trivial when ``sp_size == 1``) and are model-independent: any attention that
projects to ``[tokens, heads, head_dim]`` and calls a ``run_attention(q, k, v)``
can use :func:`ulysses_attention`, which takes ``seq_dim`` / ``head_dim`` for other
layouts such as ``[batch, tokens, heads, head_dim]``; :func:`sp_head_slice` and
:func:`sp_head_gather` assume ``[tokens, heads, head_dim]``. ``seq_sizes`` is the
per-rank token count along the sequence (need not be equal — sequences indivisible
by ``sp`` are handled without padding); the scatter/gather of the residual stream
at the model boundary uses the same split.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from mstar.distributed.communication import CommGroup


def sp_seq_split(total: int, world_size: int) -> list[int]:
    """Even per-rank token counts summing to ``total`` (earlier ranks get the
    remainder). The sequence need not be divisible by ``world_size``."""
    base, rem = divmod(total, world_size)
    return [base + (1 if r < rem else 0) for r in range(world_size)]


def scatter_sequence(
    sp_group: CommGroup,
    x_full: torch.Tensor,
    seq_sizes: list[int],
    dim: int = 0,
) -> torch.Tensor:
    """Return this rank's contiguous slice of a sequence-replicated tensor. No
    communication — every SP rank holds the identical ``x_full`` (the denoise
    latent is replicated), so each simply narrows to its ``seq_sizes`` window."""
    if sp_group.world_size == 1:
        return x_full
    start = sum(seq_sizes[: sp_group.rank])
    return x_full.narrow(dim, start, seq_sizes[sp_group.rank]).contiguous()


def gather_sequence(
    sp_group: CommGroup,
    x_shard: torch.Tensor,
    seq_sizes: list[int],
    dim: int = 0,
) -> torch.Tensor:
    """All-gather the sequence shards back into the full tensor (rank order).

    Pads each shard to the max per-rank length so the underlying collective is a
    native equal-size all-gather, then trims — robust to sequences not divisible
    by the group size. For the common even split it is a plain all-gather."""
    world_size = sp_group.world_size
    if world_size == 1:
        return x_shard
    max_sz = max(seq_sizes)
    local = x_shard
    if local.size(dim) < max_sz:
        pad_shape = list(local.shape)
        pad_shape[dim] = max_sz - local.size(dim)
        pad = torch.zeros(pad_shape, dtype=local.dtype, device=local.device)
        local = torch.cat([local, pad], dim=dim)
    gathered = sp_group.all_gather(local.contiguous(), dim=dim)
    if all(s == max_sz for s in seq_sizes):
        return gathered
    chunks = list(torch.split(gathered, max_sz, dim=dim))
    return torch.cat(
        [chunks[r].narrow(dim, 0, seq_sizes[r]) for r in range(world_size)],
        dim=dim,
    ).contiguous()


def ulysses_attention(
    sp_group: CommGroup,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    run_attention: Callable[..., torch.Tensor],
    seq_sizes: list[int],
    seq_dim: int = 0,
    head_dim: int = 1,
) -> torch.Tensor:
    """Run attention under Ulysses SP.

    ``q``: ``[seq/P, Hq, D]``; ``k``/``v``: ``[seq/P, Hkv, D]`` (this rank's
    tokens, TP-local heads). Returns ``[seq/P, Hq, D]``. Any other layout names
    its sequence and head dims (``[B, seq/P, H, D]`` is ``seq_dim=1,
    head_dim=2``). The head counts must be divisible by ``sp`` (Ulysses shards
    heads). When the group is trivial this is a passthrough — byte-identical to
    the non-SP path.

    For one sequence (only ones before ``seq_dim``, the head dim right after it),
    ``run_attention`` gets q, k and v as strided views of the buffer they arrive
    in, with a dense head dim; the engine's attention resources take them. A batch
    interleaves its rows on arrival, so each tensor is gathered contiguous. The
    path depends only on the layout, so every rank of the group takes the same one."""
    if sp_group.world_size == 1:
        return run_attention(q=q, k=k, v=v)
    seq_dim %= q.dim()
    head_dim %= q.dim()
    if head_dim == seq_dim + 1 and math.prod(q.shape[:seq_dim]) == 1:
        return _ulysses_one_sequence(sp_group, q, k, v, run_attention, seq_sizes, seq_dim)
    # scatter heads, gather sequence: [seq/P, H, D] -> [seq, H/P, D], q, k and v in one
    # exchange (the exchanges are latency-bound at small sizes)
    q, k, v = sp_group.all_to_all((q, k, v), scatter_dim=head_dim, gather_dim=seq_dim, gather_sizes=seq_sizes)
    out = run_attention(q=q, k=k, v=v)
    # scatter sequence, gather heads: [seq, H/P, D] -> [seq/P, H, D]
    return sp_group.all_to_all(out, scatter_dim=seq_dim, gather_dim=head_dim, scatter_sizes=seq_sizes)


def _ulysses_one_sequence(
    sp_group: CommGroup,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    run_attention: Callable[..., torch.Tensor],
    seq_sizes: list[int] | None,
    seq_dim: int,
) -> torch.Tensor:
    """:func:`ulysses_attention` for one sequence, one copy each way. q, k and v leave
    in one buffer laid out by destination rank, then row, then head, and the blocks
    arrive in rank order, which is sequence order, so q, k and v are views of the
    received buffer. The output's row block for each rank goes back as it lies."""
    world, head_dim = sp_group.world_size, seq_dim + 1
    lead, rows, tail = q.shape[:seq_dim], q.shape[seq_dim], q.shape[head_dim + 1:]
    sizes = [rows] * world if seq_sizes is None else seq_sizes
    even = len(set(sizes)) == 1
    heads = [t.shape[head_dim] // world for t in (q, k, v)]
    send = q.new_empty(world, *lead, rows, sum(heads), *tail)
    for piece, t, n in zip(send.split(heads, dim=head_dim + 1), (q, k, v), heads, strict=True):
        piece.copy_(t.unflatten(head_dim, (world, n)).movedim(head_dim, 0))
    row = sum(heads) * math.prod(tail)
    recv = sp_group.all_to_all_single(
        send, send.new_empty(*lead, sum(sizes), sum(heads), *tail),
        None if even else [rows * row] * world, None if even else [n * row for n in sizes],
    )
    q, k, v = recv.split(heads, dim=head_dim)
    out = run_attention(q=q, k=k, v=v).contiguous()
    row = math.prod(out.shape[head_dim:])
    back = sp_group.all_to_all_single(
        out, out.new_empty(world, *lead, rows, *out.shape[head_dim:]),
        None if even else [n * row for n in sizes], None if even else [rows * row] * world,
    )
    # [P, .., rows, H/P, ..] -> [.., rows, H, ..]: rank r attended heads r*H/P .. (r+1)*H/P
    return back.movedim(0, head_dim).flatten(head_dim, head_dim + 1)


@torch.compiler.disable
def attention_over_sharded_keys(
    sp_group: CommGroup,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    run_attention: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """A query replicated on every rank attending keys sharded along the sequence:
    ``q`` ``[B, Lq, H, D]``, ``k``/``v`` ``[B, Lk/P, H, D]`` (this rank's tokens), returns
    ``[B, Lq, H, D]``. Each rank attends its own shard and the partial attentions merge
    exactly by their log-sum-exps, which moves query-sized results rather than the keys.

    ``run_attention(q, k, v, return_lse=True)`` returns ``(out, lse)`` with ``lse``
    ``[B, Lq, H]`` in natural log (``joint_attention`` over a ragged resource, or
    ``sdpa_attention``)."""
    if sp_group.world_size == 1:
        return run_attention(q, k, v)
    out, lse = run_attention(q, k, v, return_lse=True)
    # one gather for both: [P, B, Lq, H, D + 1] fp32, the log-sum-exp in the last column
    gathered = sp_group.all_gather(torch.cat([out.float(), lse.float().unsqueeze(-1)], dim=-1).unsqueeze(0), dim=0)
    outs, lses = gathered[..., :-1], gathered[..., -1:]
    return (outs * torch.softmax(lses, dim=0)).sum(dim=0).to(q.dtype)


def sp_head_slice(sp_group: CommGroup, x: torch.Tensor) -> torch.Tensor:
    """Keep this SP rank's contiguous head-group: ``[T, H, D] -> [T, H/P, D]``.

    The selected heads ``[rank*H/P : (rank+1)*H/P]`` are exactly those that
    :func:`ulysses_attention`'s all-to-all routes to this rank, so a tensor sliced
    here (e.g. the replicated UND prefix K/V) lands on the same head partition as
    the sequence-parallel GEN attention."""
    world_size = sp_group.world_size
    if world_size == 1:
        return x
    b = x.size(1) // world_size
    return x[:, sp_group.rank * b:(sp_group.rank + 1) * b, :].contiguous()


def sp_head_gather(sp_group: CommGroup, x: torch.Tensor) -> torch.Tensor:
    """Reassemble full TP-local heads from per-rank head-groups (the inverse of
    :func:`sp_head_slice`): ``[T, H/P, D] -> [T, H, D]`` via an all-gather over
    heads, restoring the layout the row-parallel output projection expects."""
    if sp_group.world_size == 1:
        return x
    return sp_group.all_gather(x, dim=1)

