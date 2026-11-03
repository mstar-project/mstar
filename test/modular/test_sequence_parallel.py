"""Sequence parallelism on CPU processes over gloo.

``CommGroup.all_to_all`` against the layout it promises, on even and uneven
splits, and of several tensors at once against one at a time; and ``gather_merge``,
a replicated query over sharded keys merged by log-sum-exp, against one rank
attending all the keys.

The sequence shard resource's exchanges against one rank, and its rule for when a
step splits and into which ranges.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flashinfer_reference import reference_attention

from mstar.distributed.communication import CommGroup, JointGroups, sp_seq_split
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.attn.ragged.base import RaggedAttnManager
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig, RaggedAttentionSpec
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.context_parallel import gather_merge, merge, stripe_owners
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs, SequenceView, build_paged_indptrs
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.position.manager import PositionManager
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    Mamba2Geometry,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.engine.resources.sequence_shard import (
    SeqRange,
    SeqShardSpec,
    SeqShardStep,
    all_gather,
    head_slice,
    sharded_key_merge,
    ulysses,
)
from mstar.engine.resources.spec import apply_yaml_overrides, resolve_spec_dependencies
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext, SubmoduleStep
from mstar.model.components.diffusion.attention import sdpa_attention

HEADS = 6
CPU = torch.device("cpu")


def _group(rank: int, world: int) -> CommGroup:
    group = CommGroup(rank, rank, list(range(world)))
    group.device_group = dist.group.WORLD
    group.initialized = True
    return group


def _run(fn, world, store_path, *args):
    mp.spawn(_entry, args=(fn, world, str(store_path), args), nprocs=world)


def _entry(rank, fn, world, store_path, args):
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


def _layout(batched: bool) -> tuple[int, int]:
    """``(seq_dim, head_dim)``."""
    return (1, 2) if batched else (0, 1)


def _full(total: int, batched: bool, seed: int = 0, heads: int = HEADS) -> torch.Tensor:
    shape = (2, total, heads, 4) if batched else (total, heads, 4)
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _exchanges(rank, world, total, batched, negative):
    group = _group(rank, world)
    seq_dim, head_dim = _layout(batched)
    sizes = sp_seq_split(total, world)
    full = _full(total, batched)
    shard = full.narrow(seq_dim, sum(sizes[:rank]), sizes[rank])
    per_rank = HEADS // world
    seq, head = (seq_dim - full.dim(), head_dim - full.dim()) if negative else (seq_dim, head_dim)
    heads = group.all_to_all(shard, scatter_dim=head, gather_dim=seq, gather_sizes=sizes)
    assert torch.equal(heads, full.narrow(head_dim, rank * per_rank, per_rank))
    back = group.all_to_all(heads, scatter_dim=seq, gather_dim=head, scatter_sizes=sizes)
    assert torch.equal(back, shard)


@pytest.mark.parametrize("negative", [False, True], ids=["dims", "negative-dims"])
@pytest.mark.parametrize("batched", [False, True], ids=["SHD", "BSHD"])
@pytest.mark.parametrize(
    "world,total", [(2, 6), (2, 7), (3, 9), (3, 11)], ids=["2-even", "2-uneven", "3-even", "3-uneven"],
)
def test_all_to_all_exchanges(tmp_path, world, total, batched, negative):
    _run(_exchanges, world, tmp_path / "store", total, batched, negative)


def _several_at_once(rank, world, total, batched):
    group = _group(rank, world)
    seq_dim, head_dim = _layout(batched)
    sizes = sp_seq_split(total, world)
    start = sum(sizes[:rank])
    tensors = [_full(total, batched, seed, heads).narrow(seq_dim, start, sizes[rank])
               for seed, heads in enumerate((HEADS, world, 2 * world))]
    together = group.all_to_all(tensors, scatter_dim=head_dim, gather_dim=seq_dim, gather_sizes=sizes)
    for got, x in zip(together, tensors, strict=True):
        assert torch.equal(got, group.all_to_all(x, scatter_dim=head_dim, gather_dim=seq_dim, gather_sizes=sizes))


@pytest.mark.parametrize("batched", [False, True], ids=["SHD", "BSHD"])
@pytest.mark.parametrize("world,total", [(2, 6), (3, 11)], ids=["2-even", "3-uneven"])
def test_all_to_all_of_several_tensors_matches_one_at_a_time(tmp_path, world, total, batched):
    _run(_several_at_once, world, tmp_path / "store", total, batched)


def _sharded_keys(rank, world, total):
    group = _group(rank, world)
    sizes = sp_seq_split(total, world)
    q = _full(5, batched=True, seed=0)               # the same short query on every rank
    k, v = (_full(total, batched=True, seed=seed) for seed in (1, 2))
    want = sdpa_attention(q, k, v)
    start = sum(sizes[:rank])
    out = gather_merge(
        group, *sdpa_attention(q, k.narrow(1, start, sizes[rank]), v.narrow(1, start, sizes[rank]), return_lse=True),
    )
    torch.testing.assert_close(out, want)


@pytest.mark.parametrize("world,total", [(1, 7), (2, 7), (3, 11)], ids=["1", "2-uneven", "3-uneven"])
def test_gather_merge_of_sharded_keys_matches_one_rank(tmp_path, world, total):
    _run(_sharded_keys, world, tmp_path / "store", total)


def _bf16_partials(rank, world):
    # every rank's partial, built alike on each: a bf16 out (as LTX-2.5's kernels return) and fp32 lse
    parts = [(_full(5, True, seed=2 * r).to(torch.bfloat16), _full(5, True, seed=2 * r + 1)[..., 0])
             for r in range(world)]
    want = merge(torch.stack([out.float() for out, _ in parts]), torch.stack([lse for _, lse in parts]))
    assert torch.equal(gather_merge(_group(rank, world), *parts[rank]), want.to(torch.bfloat16))


def test_gather_merge_of_bf16_partials_is_the_fp32_merge_bit_for_bit(tmp_path):
    """The out travels at its own width and the lse in fp32, one byte tensor viewed back per dtype."""
    _run(_bf16_partials, 2, tmp_path / "store")


# --- the sequence shard's exchanges -------------------------------------------------------------


def _rows_attention(rows: int, causal: bool, lse: bool = False):
    """Attention within each of ``rows`` equal rows of packed ``[rows * S, H, D]`` tensors."""
    def run(q, k, v):
        parts = [reference_attention(*(t.unflatten(0, (rows, -1))[b] for t in (q, k, v)), causal) for b in range(rows)]
        out = torch.cat([part[0] for part in parts])
        return (out, torch.cat([part[1] for part in parts])) if lse else out
    return run


def _split_exchanges(rank, world, total, rows):
    """Each exchange of this rank's tokens of a ``rows``-row step (None: one packed label) against
    one rank attending them all, and a write alone (``q=None``), with the same collectives on every
    rank."""
    group = _group(rank, world)
    r = SeqRange(group, tuple(sp_seq_split(total, world)), rows)
    b = rows or 1
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(b * total, heads, 4, generator=gen) for heads in (HEADS, world, world))

    def local(x):
        return r.local(x.unflatten(0, (b, total)), 1).flatten(0, 1)

    def heads(x):
        per = x.shape[1] // world
        return x[:, rank * per:(rank + 1) * per]

    gathers = []
    gather_into = dist.all_gather_into_tensor

    def record(out, inp, *args, **kwargs):
        gathers.append(tuple(inp.shape))
        return gather_into(out, inp, *args, **kwargs)

    dist.all_gather_into_tensor = record
    run = _rows_attention(b, causal=True)
    want = run(q, k, v)
    torch.testing.assert_close(ulysses(run, local(q), local(k), local(v), r), local(want))
    if rows is None:
        torch.testing.assert_close(all_gather(run, local(q), local(k), local(v), r), local(want))
    torch.testing.assert_close(head_slice(run, q, k, v, group), want)
    # copied queries over this rank's keys
    q_short = torch.randn(b * 5, HEADS, 4, generator=gen)
    merged = sharded_key_merge(_rows_attention(b, causal=False, lse=True), q_short, local(k), local(v), r)
    torch.testing.assert_close(merged, _rows_attention(b, causal=False)(q_short, k, v))

    written = []

    def write(q, k, v):
        assert q is None
        written.append(k)

    assert ulysses(write, None, local(k), local(v), r) is None
    assert head_slice(write, None, k, v, group) is None
    for got, expect in zip(written, (heads(k), heads(k)), strict=True):
        assert torch.equal(got, expect)

    dist.all_gather_into_tensor = gather_into
    everyone = [None] * world
    dist.all_gather_object(everyone, gathers)
    assert gathers and all(g == gathers for g in everyone), everyone


@pytest.mark.parametrize("rows", [None, 2], ids=["packed", "rows"])
@pytest.mark.parametrize("world,total", [(2, 7), (3, 11)], ids=["2-uneven", "3-uneven"])
def test_sequence_shard_exchanges_match_one_rank(tmp_path, world, total, rows):
    _run(_split_exchanges, world, tmp_path / "store", total, rows)


def _striped(rank, world):
    """A step striped by stored KV position (rows with stored context below ``world``, of none, and
    longer, an odd total): each rank holds exactly the tokens it stores, ``whole`` undoes
    ``local``, and ``all_gather`` gives each rank its tokens' attention."""
    group = _group(rank, world)
    rows, stored = np.array([5, 3, 1, 2]), np.array([1, 0, 6, 3])
    qo = np.concatenate([[0], np.cumsum(rows)])
    owners = stripe_owners(qo, stored, world)
    order = torch.from_numpy(np.argsort(owners, kind="stable"))
    r = SeqRange(group, tuple(np.bincount(owners, minlength=world).tolist()), None, order)
    n = int(qo[-1])
    position = torch.from_numpy(np.repeat(stored - qo[:-1], rows) + np.arange(n))
    mine = r.local(position)
    assert (mine % world == rank).all() and mine.shape[0] == r.count and r.index.tolist() == sorted(r.index.tolist())
    assert torch.equal(r.whole(mine), position)

    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(n, heads, 4, generator=gen) for heads in (HEADS, world, world))

    def run(q, k, v):
        return torch.cat([reference_attention(q[a:b], k[a:b], v[a:b])[0] for a, b in zip(qo[:-1], qo[1:], strict=True)])

    torch.testing.assert_close(all_gather(run, r.local(q), r.local(k), r.local(v), r), r.local(run(q, k, v)))


@pytest.mark.parametrize("world", [2, 3])
def test_a_striped_step_gives_each_rank_the_tokens_it_stores(tmp_path, world):
    _run(_striped, world, tmp_path / "store")


def _rows_to_peers(rank, world):
    """``all_to_all_rows`` on SP groups whose ranks aren't the world's ({0, 2} and {1, 3} of 4),
    a ring's hops: at offset i each rank sends its rows to group rank ``r + i`` and receives from
    ``r - i``, each rank's count its own."""
    groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    members = [rank % 2, rank % 2 + 2]
    group = CommGroup(rank, members.index(rank), members)
    group.device_group, group.initialized = groups[rank % 2], True
    p, r = group.world_size, group.rank
    for i in range(1, p):
        to, back = [0] * p, [0] * p
        to[(r + i) % p], back[(r - i) % p] = r + 1, (r - i) % p + 1
        x = torch.full((r + 1, 3), float(rank))
        got = group.all_to_all_rows(x, to, back)
        assert torch.equal(got, torch.full((back[(r - i) % p], 3), float(members[(r - i) % p])))


def test_all_to_all_rows_counts_are_group_relative(tmp_path):
    _run(_rows_to_peers, 4, tmp_path / "store")


# --- a recurrent layer's exchanges ----------------------------------------------------------------


def test_row_pieces_cut_each_row_at_the_rank_bounds():
    r = SeqRange(CommGroup(0, 0, [0, 1, 2]), (7, 6, 6), None)
    assert r.row_pieces([7, 1, 2, 9]).tolist() == [[7, 0, 0, 0], [0, 1, 2, 3], [0, 0, 0, 6]]


def _conv_rows(x, weight, initial, spans):
    """A depthwise causal conv over each row of packed ``x`` ``[n, C]``, from its window
    ``initial[i]`` ``[C, W - 1]``: the output, and each row's window after it."""
    outs, finals, start = [], [], 0
    for window, span in zip(initial, spans, strict=True):
        seq = torch.cat([window, x[start:start + span].T], dim=1)
        outs.append(torch.nn.functional.conv1d(seq[None], weight[:, None], groups=weight.shape[0])[0].T)
        finals.append(seq[:, -window.shape[1]:])
        start += span
    return torch.cat(outs), torch.stack(finals)


def _halo(rank, world, spans):
    """``conv_halo`` over this rank's tokens of rows split ``spans``, some carrying a stored window:
    its output and final windows against the conv over whole rows on one rank."""
    from mstar.engine.resources.sequence_shard import conv_halo, conv_halo_plan

    group = _group(rank, world)
    channels, width = 5, 4
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(sum(spans), channels, generator=gen)
    weight = torch.randn(channels, width, generator=gen)
    stored = torch.randn(len(spans), channels, width - 1, generator=gen)
    stored[1::2] = 0  # rows without state read their window as zeros
    want, want_final = _conv_rows(x, weight, stored, spans)
    r = SeqRange(group, tuple(sp_seq_split(sum(spans), world)), None)
    halo = conv_halo_plan(r.row_pieces(spans), rank, width - 1).to(x.device)

    def conv(x, initial):
        return _conv_rows(x, weight, initial, halo.spans.tolist())[0]

    out, final = conv_halo(conv, r.local(x), stored, halo, group)
    torch.testing.assert_close(out, r.local(want))
    torch.testing.assert_close(final, want_final)


@pytest.mark.parametrize("world, spans", [
    (2, [4, 6]),
    (3, [7, 1, 2, 9]),
    (3, [1, 5, 1]),
], ids=["2", "3-a-row-on-rank-0-and-short-pieces", "3-a-window-over-three-ranks"])
def test_conv_halo_gives_each_rank_the_windows_of_the_whole_rows(tmp_path, world, spans):
    _run(_halo, world, tmp_path / "store", spans)


def test_conv_halo_indexes_on_the_device_its_tokens_are_on():
    """The manager uploads the plan's indices (``ConvHalo.to``), and the halo indexes with them as
    they are: a tensor on a device numpy can't read (``meta`` here, CUDA on a GPU) gets through,
    shapes intact."""
    from mstar.engine.resources.sequence_shard import conv_halo, conv_halo_plan

    meta, channels, taps, spans = torch.device("meta"), 5, 3, [4, 6]
    halo = conv_halo_plan(np.array([spans]), 0, taps).to(meta)
    x = torch.empty(sum(spans), channels, device=meta)
    out, final = conv_halo(lambda x, initial: x, x, torch.empty(2, channels, taps, device=meta), halo,
                           CommGroup.trivial())
    assert out.shape == x.shape and final.shape == (2, channels, taps) and final.device == meta


def _delta_rule(state, q, k, v, g, beta):
    """A gated delta rule over one row's tokens from ``state`` ``[H, V, K]``, decay ``g`` per head
    (``[n, H]``) or per head and key channel (``[n, H, K]``, KDA's): each token's output and the
    final state."""
    outs = []
    for t in range(q.shape[0]):
        decay = g[t][:, None, None] if g.dim() == 2 else g[t][:, None, :]
        state = state * decay
        state = state + beta[t][:, None, None] * torch.einsum(
            "hv,hk->hvk", v[t] - torch.einsum("hvk,hk->hv", state, k[t]), k[t],
        )
        outs.append(torch.einsum("hvk,hk->hv", state, q[t]))
    return torch.stack(outs) if outs else q.new_zeros(0, *v.shape[1:]), state


def _hand_off(rank, world, spans, per_key):
    """``state_passing`` for a recurrence whose transition is linear in the state: each rank runs
    the rows it holds from their start, hands off the last one, and runs a row continuing into it
    from the state it gets; outputs and final states against one rank."""
    from mstar.engine.resources.sequence_shard import state_passing

    group = _group(rank, world)
    heads, dim = 2, 3
    gen = torch.Generator().manual_seed(0)
    n = sum(spans)
    q, k, v = (torch.randn(n, heads, dim, generator=gen) for _ in range(3))
    k = torch.nn.functional.normalize(k, dim=-1)
    g = torch.rand((n, heads, dim) if per_key else (n, heads), generator=gen) * 0.5 + 0.5
    beta = torch.rand(n, heads, generator=gen)
    start_state = torch.randn(len(spans), heads, dim, dim, generator=gen)
    start_state[::2] = 0
    bounds = np.concatenate([[0], np.cumsum(spans)])
    want = [_delta_rule(start_state[i], *(t[a:b] for t in (q, k, v, g, beta))) for i, (a, b) in
            enumerate(zip(bounds[:-1], bounds[1:], strict=True))]

    r = SeqRange(group, tuple(sp_seq_split(n, world)), None)
    pieces = r.row_pieces(spans)
    first = bounds[:-1] + pieces[:rank].sum(0)
    rows = np.flatnonzero(pieces[rank])
    starts = {i: pieces[:rank, i].sum() == 0 for i in rows}
    ends = {i: pieces[rank + 1:, i].sum() == 0 for i in rows}

    def piece(i, state, values=v):
        a = first[i]
        return _delta_rule(state, *(t[a:a + pieces[rank, i]] for t in (q, k, values, g, beta)))

    outs, finals = {}, {}
    for i in rows:
        if starts[i]:
            outs[i], finals[i] = piece(i, start_state[i])
    eye = torch.eye(dim).expand(heads, dim, dim)
    transition, offset = eye, torch.zeros(heads, dim, dim)
    if len(rows) and not ends[rows[-1]]:
        last = rows[-1]
        if starts[last]:
            transition, offset = torch.zeros(heads, dim, dim), finals[last]
        else:
            # a middle piece: its map from the state it starts with, by linearity
            transition = piece(last, eye, torch.zeros_like(v))[1]
            offset = piece(last, torch.zeros(heads, dim, dim))[1]
    entering = state_passing(group, transition, offset, torch.matmul)
    for i in rows:
        if not starts[i]:
            outs[i], finals[i] = piece(i, entering)
    for i in rows:
        torch.testing.assert_close(outs[i], want[i][0][first[i] - bounds[i]:][:pieces[rank, i]])
        if ends[i]:
            torch.testing.assert_close(finals[i], want[i][1])


@pytest.mark.parametrize("per_key", [False, True], ids=["gdn", "kda"])
@pytest.mark.parametrize("world, spans", [
    (2, [3, 5, 4]),
    (2, [2, 4, 6]),
    (3, [2, 12, 2]),
], ids=["2", "2-a-row-ends-on-a-bound", "3-a-rank-holds-a-middle-piece"])
def test_state_passing_hands_each_row_on_as_one_rank_runs_it(tmp_path, world, spans, per_key):
    _run(_hand_off, world, tmp_path / "store", spans, per_key)


# --- when a step splits ---------------------------------------------------------------------------


def _shard(world: int = 2, rank: int = 0, labels=("main",), axis: int = 0, **spec):
    joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup(rank, rank, list(range(world))))
    spec = SeqShardSpec("shard", {"n"}, labels, axis, **spec)
    return spec.resource_class.build(spec, EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=joint))


def _split(shard, segments, walk="w", lease=None, force_prefill=False, combined=None, rows=None):
    """The shard's plan of a step of ``segments`` ((request, label, span)), as label -> (counts, rows)
    when it splits, None when copied."""
    rids = rows or sorted({rid for rid, _, _ in segments})
    ctx = StepContext(request_ids=rids, graph_walk=walk, slot=0, capture=False, slot_lease=lease,
                      force_prefill=force_prefill)
    # declared once on the submodule step, as a model does, which hands it to every resource step
    step = SubmoduleStep(
        segments=tuple(Segment(*seg) for seg in segments), combined_labels=combined or {},
        steps={"shard": SeqShardStep(), "kv": KVStep()},
    )
    assert step.steps["kv"].combined_labels == (combined or {})
    plan = shard.plan(step.steps["shard"], ctx)
    return {label: (r.counts, r.rows) for label, r in plan.ranges.items()} if plan.split else None


def _rows(label, span, n=2):
    return [(f"r{i}", label, span) for i in range(n)]


@pytest.mark.parametrize("shard, step, ranges", [
    (dict(min_tokens=16), dict(segments=[("r0", "main", 10)]), None),
    (dict(min_tokens=16), dict(segments=[("r0", "main", 16)]), {"main": ((8, 8), None)}),
    ({}, dict(segments=_rows("main", 1, 3)), None),
    (dict(walks={"prefill"}), dict(segments=[("r0", "main", 10)], walk="decode"), None),
    (dict(labels=("video",), axis=1), dict(segments=_rows("video", 5) + _rows("audio", 1)),
     {"video": ((3, 2), 2)}),
    (dict(labels=("a", "b")), dict(segments=[("r0", "a", 1), ("r0", "b", 9)]), None),
    (dict(world=4, labels=("video",), axis=1), dict(segments=_rows("video", 3)), None),
    # a packed replay of two real rows in a 16-token bucket: decided and split as the capture was
    ({}, dict(segments=[("r0", "main", 3), ("r1", "main", 2)], force_prefill=True,
              lease=SlotLease(0, BucketKey("w", bs=2, num_tokens=16))), {"main": ((8, 8), None)}),
    # a batched bucket declares its padding rows' spans already, and its other labels don't pad video
    (dict(labels=("video",), axis=1),
     dict(segments=_rows("video", 6) + _rows("audio", 2) + _rows("text", 3),
          lease=SlotLease(0, BucketKey("w", bs=2, num_tokens=22))), {"video": ((3, 3), 2)}),
    (dict(labels=("cond", "uncond")),
     dict(segments=[("r0", "cond", 5), ("r0", "uncond", 5)], combined={("cond", "uncond"): "cfg"}),
     {"cond": ((3, 2), None), "uncond": ((3, 2), None), "cfg": ((5, 5), None)}),
], ids=["below-min-tokens", "at-min-tokens", "one-token-a-row", "unlisted-walk", "axis1-per-row",
        "a-label-below-sp", "axis1-row-below-sp", "packed-lease", "batched-bucket", "combined-labels"])
def test_a_step_splits_only_where_every_rank_gets_tokens(shard, step, ranges):
    assert _split(_shard(**shard), **step) == ranges


def test_a_split_step_refuses_what_it_cant_range():
    with pytest.raises(ValueError, match="mixes split sources"):
        _split(_shard(labels=("cond",)), [("r0", "cond", 5), ("r0", "text", 5)], combined={("cond", "text"): "cfg"})
    with pytest.raises(ValueError, match="rows must be one length"):
        _split(_shard(labels=("video",), axis=1), [("r0", "video", 5), ("r1", "video", 6)])
    # a resource step can't group the step's labels differently from the shard
    with pytest.raises(ValueError, match="declare them once"):
        SubmoduleStep(combined_labels={("cond", "uncond"): "cfg"},
                      steps={"shard": SeqShardStep(), "kv": KVStep(combined_labels={("cond",): "c"})})
    shard = _shard()
    x = torch.arange(10.0)
    _split(shard, [("r0", "main", 10)])
    assert torch.equal(shard.scatter("main", x), x[:5])
    with pytest.raises(ValueError, match="not 'other'"):
        shard.scatter("other", x)
    # a copied step leaves every label whole
    _split(shard, [("r0", "main", 1)])
    assert shard.scatter("other", x) is x


def _kv_plan(*rows) -> KVPlanOutputs:
    """A pool's plan of label ``main``'s rows, each (stored, new)."""
    views = [SequenceView(f"r{i}", "main", list(range(1 + 4 * i, 1 + 4 * i - (-(c + n) // 16))), c + n, n)
             for i, (c, n) in enumerate(rows)]
    return KVPlanOutputs({"main": KVPlanOutput(build_paged_indptrs(views, 16), views)})


@pytest.mark.parametrize("world, rows, lease, counts", [
    # positions 1..4 and 0..2: rank 0 holds 2, 4, 0 and 2, step tokens 1, 3, 4 and 6
    (2, ((1, 4), (0, 3)), None, (4, 3)),
    # positions 1, 2, 5 and 6 over 4 ranks: ranks 0 and 3 would get none
    (4, ((1, 2), (5, 2)), None, None),
    # the ring runs eager: a leased step plans copied
    (2, ((1, 4), (0, 3)), SlotLease(0, BucketKey("w", bs=2, num_tokens=8)), None),
], ids=["split", "a-rank-gets-none", "leased"])
def test_a_striped_shard_gives_each_rank_its_stored_positions(world, rows, lease, counts):
    shard = _shard(world=world)
    # as a ring attention's shard derives them
    shard.stripe_pool, shard.eager_splits = "kv", True
    ctx = StepContext(request_ids=("r0", "r1"), graph_walk="w", slot=0, capture=False, slot_lease=lease,
                      force_prefill=lease is not None)
    ctx.plan_results["kv"] = _kv_plan(*rows)
    plan = shard.plan(SeqShardStep(segments=tuple(Segment(f"r{i}", "main", n) for i, (_, n) in enumerate(rows))), ctx)
    if counts is None:
        assert not plan.split
        return
    r = plan.ranges["main"]
    assert r.counts == counts and r.index.tolist() == [1, 3, 4, 6]


def test_a_striped_shard_refuses_a_step_it_cant_stripe():
    shard = _shard(world=2)
    shard.stripe_pool = "kv"
    ctx = StepContext(request_ids=("r0",), graph_walk="w", slot=0, capture=False)
    ctx.plan_results["kv"] = KVPlanOutputs({})
    step = SeqShardStep(segments=(Segment("r0", "main", 8),))
    with pytest.raises(ValueError, match="plans no such label there"):
        shard.plan(step, ctx)
    with pytest.raises(ValueError, match="which a combined label's rows"):
        _split(shard, [("r0", "main", 4), ("r0", "other", 4)], combined={("main", "other"): "cfg"})


def test_a_shard_that_never_splits_plans_every_step_copied():
    """Over KV stored by position the shard's exchange auto (a config from before such a pool could
    split runs as it did) or none: the shard derives that it splits nothing, so the node runs
    copied (min_tokens 0 and all), as one without a shard; all_gather splits. A ragged Ulysses
    attention there plans the heads the copied layer computes: over TP alone."""
    for choice, splits in ((None, False), ("none", False), ("all_gather", True)):
        specs = [
            KVSpec("kv", {"n"}, PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64,
                                              num_qo_heads=1, context_parallel=True)),
            AttentionSpec("attn", {"n"}, AttentionConfig(kv_cache="kv")),
            SeqShardSpec("shard", {"n"}, ("main",), 0),
            RaggedAttentionSpec("ragged", {"n"}, RaggedAttentionConfig(2, 2, 8), sp_exchange="ulysses"),
        ]
        apply_yaml_overrides(specs, {"resources": {} if choice is None else {"shard": {"exchange": choice}}})
        by_key = resolve_spec_dependencies(specs)
        spec = by_key["shard"]
        joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup(0, 0, [0, 1]))
        shard = spec.resource_class.build(spec, EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=joint))
        assert (spec.splits, shard.can_split, _split(shard, [("r0", "main", 10)]) is not None) == (splits,) * 3
        ragged = RaggedAttnManager.build(by_key["ragged"], EngineResourceInfo(
            device=torch.device("cpu"), joint_comm_group=joint, dependencies={"shard": spec},
        ))
        assert ragged._kwargs["num_qo_heads"] == (1 if splits else 2)


# --- the recurrent resources on a split step -----------------------------------------------------

LINEAR_SLOTS = 12
LINEAR_GEOMETRIES = {
    # heads that split two and three ways; equal key and value dims, as a middle piece's hand-off needs
    "gdn": DeltaNetGeometry(num_k_heads=6, num_v_heads=6, head_k_dim=2, head_v_dim=2, conv_kernel_size=4),
    # FlashInfer's kernels' head dim, for the GPU case
    "gdn-128": DeltaNetGeometry(num_k_heads=2, num_v_heads=4, head_k_dim=128, head_v_dim=128, conv_kernel_size=4),
    # one B/C group, fewer than the SP ranks
    "mamba2": Mamba2Geometry(num_heads=6, head_dim=2, state_size=3, n_groups=1, conv_kernel_size=4),
}


def _linear(variant, world_rank=None, exchange="ulysses", sink=True, device=CPU):
    """A linear-attention resource, its pool's two layers (seeded alike on every rank), and with
    ``world_rank`` the sequence shard it splits over: the resource as the engine builds it."""
    from mstar.engine.resources.linear_attn.base import LinearAttnManager
    from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnSpec, LinearAttnVariant
    from mstar.engine.resources.sequence_shard import SequenceShard
    from mstar.engine.resources.spec import CombinedSplit

    geometry = LINEAR_GEOMETRIES[variant]
    if isinstance(geometry, DeltaNetGeometry):
        config = LinearAttnConfig("state", LinearAttnVariant.GDN, splits=(CombinedSplit("mixed", ("prefill",)),))
    else:
        config = LinearAttnConfig("state", LinearAttnVariant.MAMBA2)
    pool = RecurrentStateSpec("state", {"n"}, RecurrentStateConfig(
        num_layers=1, blocks=geometry.to_blocks(conv_dtype=torch.float32), max_slots=LINEAR_SLOTS,
        disable_sink_slot=not sink,
    ))
    manager = LinearAttnManager.build(
        LinearAttnSpec("linear", {"n"}, config), EngineResourceInfo(device=device, dependencies={"state": pool}),
    )
    gen = torch.Generator().manual_seed(3)
    layers = {
        name: (torch.randn(LINEAR_SLOTS, *block.shape, generator=gen) * 0.3).to(device)
        for name, block in pool.config.blocks.items()
    }
    shard = None
    if world_rank is not None:
        shard = SequenceShard(("main",), 0, None, 0, _group(*world_rank))
        manager._shard_key, manager._split_exchange = "shard", exchange
        manager.bind_dependencies({"state": None, "shard": shard})
    return manager, layers, shard


def _linear_inputs(variant, n, device=CPU, dtype=torch.float32):
    geometry = LINEAR_GEOMETRIES[variant]
    gen = torch.Generator().manual_seed(4)
    if isinstance(geometry, DeltaNetGeometry):
        heads, gate_rows, names = geometry.num_v_heads, 2, ("a_log", "dt_bias")
    else:
        heads, gate_rows, names = geometry.num_heads, 1, ("a_log", "d", "dt_bias")
    params = {name: (torch.randn(heads, generator=gen) * 0.1).to(device) for name in names}
    x = torch.randn(n, geometry.conv_dim, generator=gen)
    weight = torch.randn(geometry.conv_dim, geometry.conv_kernel_size, generator=gen) * 0.5
    gates = torch.randn(n, gate_rows, heads, generator=gen)
    return x.to(device, dtype), weight.to(device, dtype), gates.to(device, dtype), params


def _linear_step(manager, layers, variant, layout, walk, x, weight, gates, params, shard=None):
    """Plan ``layout`` ((span, slot, has_state) a row) and run the layer: the conv, then the
    recurrence, on this rank's tokens when ``shard`` splits the step."""
    from mstar.engine.resources.linear_attn.config import LinearAttnStep
    from mstar.engine.resources.recurrent.pool import RecurrentAddressing

    spans, slots, has = (list(col) for col in zip(*layout, strict=True))
    rids = [f"r{i}" for i in range(len(layout))]
    segments = tuple(Segment(rid, "main", span) for rid, span in zip(rids, spans, strict=True))
    ctx = StepContext(request_ids=rids, graph_walk=walk, slot=0, capture=False)
    ctx.plan_results["state"] = {"main": RecurrentAddressing(
        torch.tensor(slots, dtype=torch.int32, device=x.device), torch.tensor(has, device=x.device), len(layout),
        tuple(slots), tuple(has),
    )}
    if shard is not None:
        ctx.plan_results["shard"] = shard.plan(SeqShardStep(segments=segments), ctx)
        x, gates = shard.scatter("main", x, gates)
    manager.plan(LinearAttnStep(segments=segments), ctx)
    manager.set_default_label("main")
    x = manager.run_conv(x, layers["conv"], weight, None)
    n, g = x.shape[0], LINEAR_GEOMETRIES[variant]
    if isinstance(g, DeltaNetGeometry):
        keys, values = g.num_k_heads * g.head_k_dim, g.num_v_heads * g.head_v_dim
        q, k, v = x.split([keys, keys, values], dim=-1)
        out = manager.run(
            q.reshape(n, g.num_k_heads, g.head_k_dim), k.reshape(n, g.num_k_heads, g.head_k_dim),
            v.reshape(n, g.num_v_heads, g.head_v_dim), gates[:, 0], gates[:, 1],
            layers["state"], params["a_log"], params["dt_bias"],
        )
    else:
        groups = g.n_groups * g.state_size
        xs, b, c = x.split([g.d_inner, groups, groups], dim=-1)
        out = manager.run(
            xs.reshape(n, g.num_heads, g.head_dim), gates[:, 0], b.reshape(n, g.n_groups, g.state_size),
            c.reshape(n, g.n_groups, g.state_size), layers["ssm"], params["a_log"], params["d"], params["dt_bias"],
        )
    if shard is not None:
        shard.commit(SeqShardStep(), ctx)
    return x, out


def _linear_split(rank, world, variant, exchange, layout, walk, sink, gpu=False):
    """The resource on this rank's share of a split step against the same resource running it
    whole on one rank: the conv and recurrence outputs (exact for ulysses on CPU, within tolerance
    for state_passing and the GPU's bf16), and the pool's state and conv windows, alike bit for
    bit on every rank."""
    device = torch.device("cuda", rank) if gpu else CPU
    n = sum(span for span, _, _ in layout)
    x, weight, gates, params = _linear_inputs(variant, n, device, torch.bfloat16 if gpu else torch.float32)
    ref, ref_layers, _ = _linear(variant, sink=sink, device=device)
    # a clone: the one-token conv update writes its output over its input
    want_conv, want = _linear_step(ref, ref_layers, variant, layout, walk, x.clone(), weight, gates, params)
    manager, layers, shard = _linear(variant, (rank, world), exchange, sink, device)
    got_conv, got = _linear_step(manager, layers, variant, layout, walk, x, weight, gates, params, shard)
    r = SeqRange(shard.group, tuple(sp_seq_split(n, world)), None)
    tol = dict(rtol=2e-2, atol=2e-2) if gpu else {} if exchange == "ulysses" else dict(rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got_conv, r.local(want_conv), **(tol if gpu else {}))
    torch.testing.assert_close(got, r.local(want), **tol)
    for name, layer in layers.items():
        torch.testing.assert_close(layer, ref_layers[name], **tol)
        everyone = shard.group.all_gather(layer[None].contiguous(), dim=0)
        assert all(torch.equal(other, layer) for other in everyone), f"{name} replicas differ"


# (span, slot, has_state) a row
_FRESH_AND_CONTINUING = [(7, 3, False), (1, 5, True), (2, 6, True), (9, 7, True)]
_MIXED = [(1, 2, True), (1, 4, True), (6, 5, True), (1, 8, False), (5, 9, False)]


@pytest.mark.parametrize("exchange", ["ulysses", "state_passing"])
@pytest.mark.parametrize("variant, world, layout, walk, sink", [
    ("gdn", 2, _FRESH_AND_CONTINUING, "prefill", True),
    # a row wholly on rank 0, pieces shorter than the conv window, a rank holding a middle piece
    ("gdn", 3, _FRESH_AND_CONTINUING, "prefill", True),
    ("gdn", 3, [(2, 3, True), (12, 4, True), (2, 5, False)], "prefill", True),
    # the decode run runs copied, the prompt rows split; the sink off, slot 0 a request's
    ("gdn", 2, _MIXED, "mixed", True),
    ("gdn", 3, [(1, 0, True), (1, 4, True), (9, 5, True), (4, 9, False)], "mixed", False),
    # one B/C group under three ranks
    ("mamba2", 3, _FRESH_AND_CONTINUING, "prefill", True),
    ("mamba2", 3, [(2, 3, True), (12, 4, True), (2, 5, False)], "prefill", False),
], ids=["gdn-2", "gdn-3", "gdn-3-middle-piece", "gdn-2-mixed", "gdn-3-mixed-sink-off", "mamba2-3",
        "mamba2-3-middle-piece-sink-off"])
def test_a_recurrent_layer_on_a_split_step_matches_one_rank(tmp_path, variant, world, layout, walk, sink, exchange):
    _run(_linear_split, world, tmp_path / "store", variant, exchange, layout, walk, sink)


def _nccl_entry(rank, fn, world, store_path, args):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{store_path}", rank=rank, world_size=world)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("exchange", ["ulysses", "state_passing"])
@pytest.mark.parametrize("layout, walk", [(_FRESH_AND_CONTINUING, "prefill"), (_MIXED, "mixed")],
                         ids=["prefill", "mixed"])
def test_two_gpus_run_a_gdn_layer_on_a_split_step_like_one(tmp_path, layout, walk, exchange):
    """The CPU cases above on FlashInfer's GDN kernels and the Triton conv, over NCCL."""
    pytest.importorskip("flashinfer")
    mp.spawn(_nccl_entry, args=(_linear_split, 2, str(tmp_path / "store"),
                                ("gdn-128", exchange, layout, walk, True, True)), nprocs=2)


def test_a_recurrent_layer_plans_after_its_nodes_shard_and_refuses_a_step_without_it():
    """At one SP rank too: the runner plans the shard first (the resource depends on it), and a
    step that doesn't declare it raises rather than planning copied."""
    from mstar.engine.resources.linear_attn.config import LinearAttnStep
    from mstar.engine.resources.recurrent.pool import RecurrentAddressing
    from mstar.engine.resources.runner import topo_sort

    manager, _, _ = _linear("gdn")
    manager._shard_key = "llm_shard"
    order = topo_sort({"linear_attn": manager, "llm_shard": _shard(world=1), "state": _shard(world=1)})
    assert order.index("llm_shard") < order.index("linear_attn")
    ctx = StepContext(request_ids=("r0",), graph_walk="prefill", slot=0, capture=False)
    ctx.plan_results["state"] = {"main": RecurrentAddressing(
        torch.tensor([3], dtype=torch.int32), torch.tensor([False]), 1, (3,), (False,),
    )}
    with pytest.raises(ValueError, match="doesn't declare"):
        manager.plan(LinearAttnStep(segments=(Segment("r0", "main", 4),)), ctx)


@pytest.mark.parametrize("placed, choice, want", [
    (True, None, (False, False, None, "none")),
    (True, "all_gather", (True, True, None, "ulysses")),
    (True, "ring", (True, True, "kv", "ulysses")),
    (False, None, (True, True, None, "ulysses")),
], ids=["placed-unset", "placed-all-gather", "ring", "by-head"])
def test_the_shard_reads_the_same_split_from_its_nodes_specs_in_any_order(placed, choice, want):
    """The shard derives from the specs' hooks alone, so whichever spec derives first (the shard
    first or last) it reads the same: whether the node splits, whether a leased step plans copied
    (a recurrent layer and the ring run split steps eagerly), the striped pool; and an unset
    linear spec follows the node."""
    import itertools

    from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnSpec, LinearAttnVariant

    blocks = DeltaNetGeometry(num_k_heads=2, num_v_heads=4, head_k_dim=8, head_v_dim=8, conv_kernel_size=4).to_blocks()
    for order in itertools.permutations(range(5)):
        specs = [
            KVSpec("kv", {"n"}, PagedKVConfig(num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=64,
                                              num_qo_heads=4, context_parallel=placed)),
            AttentionSpec("attn", {"n"}, AttentionConfig(kv_cache="kv")),
            RecurrentStateSpec("state", {"n"}, RecurrentStateConfig(num_layers=1, blocks=blocks)),
            LinearAttnSpec("linear", {"n"}, LinearAttnConfig(recurrent_state="state", variant=LinearAttnVariant.GDN)),
            SeqShardSpec("shard", {"n"}, ("main",), 0),
        ]
        specs = [specs[i] for i in order]
        apply_yaml_overrides(specs, {"resources": {} if choice is None else {"shard": {"exchange": choice}}})
        by_key = resolve_spec_dependencies(specs)
        shard = by_key["shard"]
        linear = by_key["linear"].resolve_exchange(by_key)
        assert (shard.splits, shard.eager_splits, shard.stripe_pool, linear) == want, order
        assert by_key["linear"].depends_on() == {"state", "shard"}


def test_a_walk_the_shard_never_splits_traces_the_bracket_straight_through():
    """On SP ranks the bracket breaks a compiled frame (it reads the step's plan), except given a
    walk the shard never splits (outside ``walks``), whose frame a model compiles apart: Qwen3.5's
    decode, which stays one graph."""
    shard = _shard(walks=("prefill",))

    def bracket(x, walk):
        return shard.gather("main", shard.scatter("main", x, walk=walk) * 2, walk=walk)

    x = torch.arange(4.0)
    decode = torch.compile(lambda x: bracket(x, "decode"), fullgraph=True, backend="eager")
    assert torch.equal(decode(x), 2 * x)
    with pytest.raises(torch._dynamo.exc.Unsupported):
        torch.compile(lambda x: bracket(x, "prefill"), fullgraph=True, backend="eager")(x)


@pytest.mark.parametrize("end", ["commit", "capture"])
def test_a_split_plan_ends_with_its_step(end):
    """Outside a step the bracket is the identity (Cosmos3's warmup compile runs after the
    captures), however the step ended: committed, or a capture's dummy rows reset."""
    shard = _shard()
    x = torch.arange(10.0)
    _split(shard, [("r0", "main", 10)])
    if end == "commit":
        shard.commit(SeqShardStep(), None)
    else:
        shard.reset_request("r0", free=True)
    assert shard.split_range("main") is None
    assert shard.scatter("main", x) is x and shard.scatter("other", x) is x


def test_an_exchange_of_a_label_the_stack_didnt_scatter_is_refused_on_every_rank():
    """Labels ``a`` (4, 4 a rank) and ``b`` (4, 3): rank 0 holds ``a``'s 4 tokens, as many as its
    share of ``b``, so only the label tells it the attention is over the wrong one; it refuses
    before the exchange's collective, as rank 1 does."""
    from mstar.engine.resources.attn.base import AttentionManager

    shard = _shard(labels=("a", "b"))
    _split(shard, [("r0", "a", 8), ("r0", "b", 7)])
    x = shard.scatter("a", torch.zeros(8, HEADS, 4))
    attn = types.SimpleNamespace(_shard=shard, _default_label="a", _split_exchange="ulysses")
    with pytest.raises(ValueError, match="before the layer stack scattered it"):
        AttentionManager.exchange(attn, lambda q, k, v: q, x, x, x, label="b")


def test_a_step_that_doesnt_declare_its_nodes_shard_is_refused_at_one_rank():
    """A dependent of the shard refuses a step without it at any SP size, so a declare_step branch
    that forgets it fails single-GPU tests rather than first on an SP box."""
    specs = [
        KVSpec("kv", {"n"}, PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, num_qo_heads=1)),
        PositionSpec("pos", {"n"}, PositionConfig(kv_cache="kv")),
        SeqShardSpec("shard", {"n"}, ("main",), 0),
    ]
    by_key = resolve_spec_dependencies(specs)
    assert by_key["pos"].depends_on() == {"kv", "shard"}
    info = EngineResourceInfo(device=torch.device("cpu"), dependencies={"kv": by_key["kv"]})
    pos = PositionManager.build(by_key["pos"], info)
    pos.bind_dependencies({"shard": _shard(world=1)})
    ctx = StepContext(request_ids=("r0",), graph_walk="w", slot=0, capture=False)
    ctx.plan_results["kv"] = KVPlanOutputs({})
    with pytest.raises(ValueError, match="doesn't declare"):
        pos.plan(PositionStep(segments=(Segment("r0", "main", 4),)), ctx)


@pytest.mark.parametrize("config, shard, piecewise, pattern", [
    (dict(kind="batched", walks=["a", "b"]), dict(walks={"a"}), {}, "replays both walks its sequence shard splits"),
    (dict(kind="packed", multiplier=2), {}, {}, "a packed capture"),
    (dict(kind="packed"), dict(labels=("cond", "uncond")), {}, "a packed capture"),
    (dict(kind="packed"), dict(axis=1), {}, "a packed capture"),
    (dict(kind="batched"), {}, {"region": object()}, "piecewise captures"),
    (dict(kind="batched", walks=["a", "b"]), dict(walks={"a", "b"}), {}, None),
    (dict(kind="packed"), {}, {}, None),
], ids=["walks-straddle", "multiplier", "two-labels", "axis1", "piecewise", "walks-closed", "packed"])
def test_captures_a_splitting_node_cant_replay_alike_are_refused(config, shard, piecewise, pattern):
    from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig, PackedAcceleratorGraphConfig

    walks = config.get("walks", ["a"])
    if config["kind"] == "batched":
        capture = BatchedAcceleratorGraphConfig("a", single_request_inputs=None, replay_graph_walks=walks)
    else:
        capture = PackedAcceleratorGraphConfig(
            "a", [16], make_node_input=lambda n: None, replay_graph_walks=walks,
            total_tokens_multiplier=config.get("multiplier", 1),
        )
    check = lambda: _shard(**shard).check_captures("n", [capture], piecewise)  # noqa: E731
    if pattern is None:
        check()
    else:
        with pytest.raises(ValueError, match=pattern):
            check()
