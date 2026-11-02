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

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flashinfer_reference import reference_attention

from mstar.distributed.communication import CommGroup, JointGroups, sp_seq_split
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.context_parallel import gather_merge, merge
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.position.manager import PositionManager
from mstar.engine.resources.sequence_shard import (
    SeqRange,
    SeqShardSpec,
    SeqShardStep,
    head_slice,
    sharded_key_merge,
    ulysses,
)
from mstar.engine.resources.spec import resolve_spec_dependencies
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext
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
    one rank attending them all; a write alone (``q=None``); and the entry check, on every rank."""
    group = _group(rank, world)
    r = SeqRange(group, tuple(sp_seq_split(total, world)), rows)
    b = rows or 1
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(b * total, heads, 4, generator=gen) for heads in (HEADS, world, world))

    def local(x):
        return x.unflatten(0, (b, total)).narrow(1, r.start, r.count).flatten(0, 1)

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

    # a forgotten scatter: every rank holds the whole step, and every rank refuses it
    with pytest.raises(ValueError, match="scatter the layer stack's input"):
        ulysses(run, q, k, v, r)
    dist.all_gather_into_tensor = gather_into
    everyone = [None] * world
    dist.all_gather_object(everyone, gathers)
    assert gathers and all(g == gathers for g in everyone), everyone


@pytest.mark.parametrize("rows", [None, 2], ids=["packed", "rows"])
@pytest.mark.parametrize("world,total", [(2, 7), (3, 11)], ids=["2-uneven", "3-uneven"])
def test_sequence_shard_exchanges_match_one_rank(tmp_path, world, total, rows):
    _run(_split_exchanges, world, tmp_path / "store", total, rows)


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
    step = SeqShardStep(segments=tuple(Segment(*seg) for seg in segments), combined_labels=combined or {})
    plan = shard.plan(step, ctx)
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
    shard = _shard()
    x = torch.arange(10.0)
    _split(shard, [("r0", "main", 10)])
    assert torch.equal(shard.scatter("main", x), x[:5])
    with pytest.raises(ValueError, match="not 'other'"):
        shard.scatter("other", x)
    # a copied step leaves every label whole
    _split(shard, [("r0", "main", 1)])
    assert shard.scatter("other", x) is x


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
