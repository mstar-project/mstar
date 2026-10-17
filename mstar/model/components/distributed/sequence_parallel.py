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

from typing import Callable

import torch
import torch.distributed as dist

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
    the non-SP path."""
    if sp_group.world_size == 1:
        return run_attention(q=q, k=k, v=v)
    # scatter heads, gather sequence: [seq/P, H, D] -> [seq, H/P, D], q, k and v in one
    # exchange (the exchanges are latency-bound at small sizes)
    q, k, v = sp_group.all_to_all((q, k, v), scatter_dim=head_dim, gather_dim=seq_dim, gather_sizes=seq_sizes)
    out = run_attention(q=q, k=k, v=v)
    # scatter sequence, gather heads: [seq, H/P, D] -> [seq/P, H, D]
    return sp_group.all_to_all(out, scatter_dim=seq_dim, gather_dim=head_dim, scatter_sizes=seq_sizes)


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


@torch.compiler.disable
def _all_to_all(
    sp_group: CommGroup, send: torch.Tensor, recv: torch.Tensor,
    recv_splits: list[int] | None = None, send_splits: list[int] | None = None,
) -> torch.Tensor:
    dist.all_to_all_single(recv, send, recv_splits, send_splits, group=sp_group.device_group)
    return recv


def sequence_to_heads(
    sp_group: CommGroup, x: torch.Tensor, seq_sizes: list[int], seq_dim: int, head_dim: int,
) -> torch.Tensor:
    """Ulysses' first exchange for any layout: this rank's tokens over all heads to
    every token over this rank's head-group (``[.., S/P, .., H, ..] -> [.., S, .., H/P, ..]``).

    One ``all_to_all_single``, which replays from a CUDA graph. Only the collective is
    hidden from torch.compile: the layout changes on either side stay in the caller's
    graph, where inductor writes the send buffer from the producing kernel and folds the
    received layout into the consumer (two copies per exchange otherwise)."""
    world_size = sp_group.world_size
    if world_size == 1:
        return x
    # [.., H, ..] -> [P, .., H/P, ..]: piece r (head-group r) goes to rank r
    send = x.unflatten(head_dim, (world_size, -1)).movedim(head_dim, 0).contiguous()
    if len(set(seq_sizes)) == 1:
        recv = _all_to_all(sp_group, send, torch.empty_like(send))
        # piece r is rank r's tokens: concatenate them along the sequence, in rank order
        return recv.movedim(0, seq_dim).flatten(seq_dim, seq_dim + 1)
    # uneven: rank r's piece holds its own token count
    piece = list(send.shape[1:])
    shapes = [piece[:seq_dim] + [n] + piece[seq_dim + 1:] for n in seq_sizes]
    numels = [int(torch.Size(shape).numel()) for shape in shapes]
    recv = _all_to_all(sp_group, send.flatten(), x.new_empty(sum(numels)), numels, [send[0].numel()] * world_size)
    pieces = [chunk.view(shape) for chunk, shape in zip(recv.split(numels), shapes, strict=True)]
    return torch.cat(pieces, dim=seq_dim)


def heads_to_sequence(
    sp_group: CommGroup, x: torch.Tensor, seq_sizes: list[int], seq_dim: int, head_dim: int,
) -> torch.Tensor:
    """The inverse of :func:`sequence_to_heads`: ``[.., S, .., H/P, ..] -> [.., S/P, .., H, ..]``."""
    world_size = sp_group.world_size
    if world_size == 1:
        return x
    if len(set(seq_sizes)) == 1:
        # [.., S, ..] -> [P, .., S/P, ..]: piece r (rank r's tokens) goes to rank r
        send = x.unflatten(seq_dim, (world_size, -1)).movedim(seq_dim, 0).contiguous()
        recv = _all_to_all(sp_group, send, torch.empty_like(send))
    else:
        # uneven: piece r holds rank r's token count
        chunks = x.split(seq_sizes, dim=seq_dim)
        send = torch.cat([chunk.contiguous().flatten() for chunk in chunks])
        mine = chunks[sp_group.rank].shape
        recv = _all_to_all(sp_group, send, x.new_empty(world_size * mine.numel()),
                           [mine.numel()] * world_size, [c.numel() for c in chunks]).view(world_size, *mine)
    # piece r is head-group r of this rank's tokens
    return recv.movedim(0, head_dim).flatten(head_dim, head_dim + 1)
