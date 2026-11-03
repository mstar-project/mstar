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
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.context_parallel import gather_merge, merge, stripe_owners
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs, SequenceView, build_paged_indptrs
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.position.manager import PositionManager
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
    shard.stripe_pool = "kv"
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
    copied (min_tokens 0 and all), as one without a shard; all_gather splits."""
    for choice, splits in ((None, False), ("none", False), ("all_gather", True)):
        specs = [
            KVSpec("kv", {"n"}, PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64,
                                              num_qo_heads=1, context_parallel=True)),
            AttentionSpec("attn", {"n"}, AttentionConfig(kv_cache="kv")),
            SeqShardSpec("shard", {"n"}, ("main",), 0),
        ]
        apply_yaml_overrides(specs, {"resources": {} if choice is None else {"shard": {"exchange": choice}}})
        spec = resolve_spec_dependencies(specs)["shard"]
        joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup(0, 0, [0, 1]))
        shard = spec.resource_class.build(spec, EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=joint))
        assert (spec.splits, shard.can_split, _split(shard, [("r0", "main", 10)]) is not None) == (splits,) * 3


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
    attn = types.SimpleNamespace(_shard=shard, _default_label="a")
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
