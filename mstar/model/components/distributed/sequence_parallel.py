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

These helpers take an ``sp_group`` (a :class:`TPCommGroup` over the SP axis of the
mesh; trivial when ``sp_size == 1``) and are model-independent: any attention that
projects to ``[tokens, heads, head_dim]`` and calls a ``run_attention(q, k, v)``
can use :func:`ulysses_attention`. ``seq_sizes`` is the per-rank token count along
the sequence (need not be equal — sequences indivisible by ``sp`` are handled
without padding); the scatter/gather of the residual stream at the model boundary
uses the same split.
"""
from __future__ import annotations

from typing import Callable

import torch

from mstar.distributed.communication import TPCommGroup


def sp_seq_split(total: int, world_size: int) -> list[int]:
    """Even per-rank token counts summing to ``total`` (earlier ranks get the
    remainder). The sequence need not be divisible by ``world_size``."""
    base, rem = divmod(total, world_size)
    return [base + (1 if r < rem else 0) for r in range(world_size)]


def scatter_sequence(
    sp_group: TPCommGroup,
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
    sp_group: TPCommGroup,
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
    sp_group: TPCommGroup,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    run_attention: Callable[..., torch.Tensor],
    seq_sizes: list[int],
    prefer_all_gather: bool = False,
) -> torch.Tensor:
    """Run attention under Ulysses SP.

    ``q``: ``[seq/P, Hq, D]``; ``k``/``v``: ``[seq/P, Hkv, D]`` (this rank's
    tokens, TP-local heads). Returns ``[seq/P, Hq, D]``. The head counts must be
    divisible by ``sp`` (Ulysses shards heads). When the group is trivial this is
    a passthrough — byte-identical to the non-SP path.

    ``prefer_all_gather`` selects the all-gather collective instead of the
    all-to-all (see :func:`_ulysses_attention_via_all_gather`). The caller sets
    it on the denoise forward that the CUDA graph captures: the all-to-all is
    grouped point-to-point send/recv and does not replay from a captured graph,
    whereas all-gather (a true collective, like the TP all-reduce) does. It must
    be set consistently across warmup, capture and replay so the all-gather
    kernels are compiled and autotuned during eager warmup, not mid-capture
    (autotuning synchronizes, which is illegal while a graph is recording).
    Eager paths (video, uncaptured resolutions) leave it off for the lighter
    all-to-all. Both produce identical results."""
    if sp_group.world_size == 1:
        return run_attention(q=q, k=k, v=v)
    if prefer_all_gather:
        return _ulysses_attention_via_all_gather(sp_group, q, k, v, run_attention)
    # scatter heads, gather sequence: [seq/P, H, D] -> [seq, H/P, D]
    q = sp_group.all_to_all(q, scatter_dim=1, gather_dim=0, gather_sizes=seq_sizes)
    k = sp_group.all_to_all(k, scatter_dim=1, gather_dim=0, gather_sizes=seq_sizes)
    v = sp_group.all_to_all(v, scatter_dim=1, gather_dim=0, gather_sizes=seq_sizes)
    out = run_attention(q=q, k=k, v=v)  # [seq, Hq/P, D] (attends [UND-prefix | GEN])
    # scatter sequence, gather heads: [seq, H/P, D] -> [seq/P, H, D]
    out = sp_group.all_to_all(out, scatter_dim=0, gather_dim=1, scatter_sizes=seq_sizes)
    return out


def _ulysses_attention_via_all_gather(
    sp_group: TPCommGroup,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    run_attention: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """CUDA-graph-capturable Ulysses attention built from all-gather.

    Same result as :func:`ulysses_attention`'s all-to-all, different collective:
    each rank all-gathers the full sequence and attends over its own head-group,
    then all-gathers the full heads back and keeps its own sequence shard. The
    all-to-all would move fewer bytes, but it is grouped send/recv and does not
    replay from a CUDA graph; all-gather is a true collective and does. Assumes
    an even sequence split across the group (the captured resolutions guarantee
    it). Head counts are divisible by the group size (the Ulysses constraint)."""
    world_size, rank = sp_group.world_size, sp_group.rank
    # [seq/P, H, D] -> all-gather sequence -> [seq, H, D] -> keep head-group rank
    q_full = sp_group.all_gather(q, dim=0)
    k_full = sp_group.all_gather(k, dim=0)
    v_full = sp_group.all_gather(v, dim=0)
    hq, hkv = q_full.size(1) // world_size, k_full.size(1) // world_size
    q = q_full[:, rank * hq:(rank + 1) * hq, :].contiguous()
    k = k_full[:, rank * hkv:(rank + 1) * hkv, :].contiguous()
    v = v_full[:, rank * hkv:(rank + 1) * hkv, :].contiguous()
    out = run_attention(q=q, k=k, v=v)  # [seq, Hq/P, D]
    # [seq, Hq/P, D] -> all-gather heads -> [seq, Hq, D] -> keep sequence shard rank
    out = sp_group.all_gather(out, dim=1)
    sl = out.size(0) // world_size
    return out[rank * sl:(rank + 1) * sl, :, :].contiguous()


def sp_head_slice(sp_group: TPCommGroup, x: torch.Tensor) -> torch.Tensor:
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


def sp_head_gather(sp_group: TPCommGroup, x: torch.Tensor) -> torch.Tensor:
    """Reassemble full TP-local heads from per-rank head-groups (the inverse of
    :func:`sp_head_slice`): ``[T, H/P, D] -> [T, H, D]`` via an all-gather over
    heads, restoring the layout the row-parallel output projection expects."""
    if sp_group.world_size == 1:
        return x
    return sp_group.all_gather(x, dim=1)
