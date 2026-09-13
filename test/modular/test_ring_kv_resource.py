"""The ring KV resource: dispatch, ownership, isolation, and the things that
fail quietly.

Every test here pins a failure that produces no exception on its own. The ring
IS the world state, so a lost claim, a rebound buffer, a stale visibility row or
a world reading its neighbour's history does not raise — it produces plausible,
smoothly drifting video, or a request that hangs forever waiting on an evictor
with nothing to evict.

``num_worlds`` worlds share one buffer per layer, folded into its token
dimension (``kv/ring/cache.py``). Nothing physical separates them: ``upsert``
hands back K and V spanning every resident world and one bool row that is False
outside the caller's own span. That row is the whole isolation mechanism, so the
tests that check it (``test_the_flat_ring_matches_one_ring_per_world``,
``test_worlds_interleave_without_reaching_each_other``) are load-bearing in a
way the ownership tests are not: ownership failures lose capacity, isolation
failures corrupt video.

CPU-only and allocation-free at test scale: the geometry is 3 layers of 4 frames
at 128 tokens, not 24 x 17 x 512. The one GPU test is the capture test, which
cannot be anything else — the property it pins (``world_idx`` is *read* at
replay, not baked at capture) only exists inside a CUDA graph.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.cuda_graph_runner import DummyRowPool
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.config import (
    KVSpec,
    KVStep,
    PagedKVConfig,
    RingKVConfig,
    RingKVLayerConfig,
    RingKVStep,
)
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.ring import LayerRingCache, RingKVManager
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import (
    AdmitRuntimeError,
    AllocationFailed,
    RequestOffloading,
    StepContext,
)

TPF = 128           # the sparse block size; the smallest legal frame
RING_FRAMES = 4
N_LAYERS = 3
N_KV_HEADS = 2
D_HEAD = 8


# ── fixtures ────────────────────────────────────────────────────────────


def _ring_config(
    *,
    num_worlds: int = 1,
    ring_frames: int = RING_FRAMES,
    num_kv_heads: int = N_KV_HEADS,
) -> RingKVConfig:
    """Three layers, the last one dilated: a config with only stride-1 layers
    would never exercise the 1-in-8 commit redirect."""
    return RingKVConfig(
        num_layers=N_LAYERS,
        num_kv_heads=num_kv_heads,
        head_dim=D_HEAD,
        tokens_per_frame=TPF,
        num_worlds=num_worlds,
        layers=tuple(
            RingKVLayerConfig(
                ring_frames=ring_frames,
                ring_buckets=ring_frames,
                pinned_dilation=8 if i == N_LAYERS - 1 else 1,
            )
            for i in range(N_LAYERS)
        ),
    )


def _manager(
    config: RingKVConfig | None = None, name: str = "kv", device: str = "cpu"
) -> RingKVManager:
    spec = KVSpec(resource_key=name, nodes={"dit"}, config=config or _ring_config())
    info = EngineResourceInfo(device=torch.device(device), kv_dtype=torch.float32)
    return RingKVManager.build(spec, info)


def _ctx(*rids: str) -> StepContext:
    return StepContext(
        request_ids=tuple(rids), graph_walk="rollout", slot=0, capture=False,
    )


def _step(*rids: str, frame: int = 0) -> RingKVStep:
    """A step declaring ``frame`` as the ring clock for every rid in ``rids``.

    One clock per request, and never absent: ``RingKVStep`` has no shape that
    declines to answer, so there is no step a test can build that switches the
    continuity check off. Frame 0 by default, so the ownership tests below can
    say what they are about — they exercise who holds a world, not what frame it
    is on, and a clock they did not think about should not refuse them.
    """
    return RingKVStep(frames=tuple((rid, frame) for rid in rids))


def _open(kv: RingKVManager, *rids: str, frame: int = 0) -> None:
    """Register and admit each rid in turn, one step each — the shape the
    engine actually produces, since ``max_batch_size`` is 1 and each request
    gets its own step."""
    for rid in rids:
        kv.ingest_request(rid)
        outcome = kv.admit(_step(rid, frame=frame), _ctx(rid))
        assert outcome.ok, f"{rid} was refused a world: {outcome.reason}"


def _frame(kv: RingKVManager, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """One frame's K and V. Dim 0 is 1 and stays 1: it is FlexAttention's batch
    dim, and worlds live in the token dim, not here."""
    shape = (1, kv.config.num_kv_heads, kv.tokens_per_frame, kv.config.head_dim)
    return (
        torch.randn(shape, generator=gen),
        torch.randn(shape, generator=gen),
    )


def _rollout(
    kv: RingKVManager, frames: int = 6, seed: int = 0, start: int = 0
) -> None:
    """The 4+1 schedule, for as many frames as asked, into whichever world
    ``plan`` last staged. Past ``ring_frames`` it wraps, which is the only way
    the ring's contents stop being trivially position-ordered.

    ``start`` is the clock the first frame runs at, for the tests that drive
    the resource lifecycle alongside it and need the two to agree."""
    gen = torch.Generator().manual_seed(seed)
    for f in range(start, start + frames):
        frame_pos = torch.tensor(f, dtype=torch.int64)
        for commit in (False, False, False, False, True):
            for layer_idx in range(len(kv.layers)):
                k, v = _frame(kv, gen)
                kv.upsert(k, v, layer_idx, frame_pos, commit=commit)


def _ptrs(kv: RingKVManager) -> list[tuple[int, int, int, int]]:
    return [
        (id(layer.kv), layer.kv.data_ptr(), id(layer.written), layer.written.data_ptr())
        for layer in kv.layers
    ]


def _world_view(layer: LayerRingCache) -> torch.Tensor:
    """``written`` cut back into ``[num_worlds, capacity]``. A view, so it reads
    the live row rather than a snapshot of it."""
    return layer.written.view(layer.num_worlds, layer.capacity)


# ── spec dispatch ───────────────────────────────────────────────────────


def test_kv_spec_dispatches_on_the_config_it_was_handed():
    """The storage strategy is chosen by the config a model declares, not by a
    flag threaded through the manager — so a Waypoint spec cannot end up on the
    paged manager (which would allocate pages nobody reads and answer `plan`
    with a page table the ring has no use for)."""
    ring = KVSpec(resource_key="kv", nodes={"dit"}, config=_ring_config())
    paged = KVSpec(
        resource_key="kv", nodes={"decoder"},
        config=PagedKVConfig(
            num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=128,
        ),
    )

    assert ring.resource_class is RingKVManager
    assert paged.resource_class is KVManager


def test_building_a_ring_manager_from_a_paged_config_raises():
    """`resource_class` is the only thing that should ever pick the manager;
    a hand-built spec that skipped it gets a TypeError rather than an
    AttributeError forty lines into the constructor."""
    spec = KVSpec(
        resource_key="kv", nodes={"dit"},
        config=PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=128),
    )
    with pytest.raises(TypeError, match="RingKVConfig"):
        RingKVManager.build(spec, EngineResourceInfo(device=torch.device("cpu")))


@pytest.mark.parametrize(
    "override", [{"max_num_pages": 16}, {"page_size": 64}, {"max_seq_len": 8}, {"tokens_per_frame": 256}],
)
def test_ring_geometry_is_not_a_yaml_tunable(override):
    """The horizon is the one the weights were trained against. A deployment
    that "tunes" it gets a different model with every shape still valid."""
    spec = KVSpec(resource_key="kv", nodes={"dit"}, config=_ring_config())
    with pytest.raises(TypeError):
        spec.apply_yaml_overrides(**override)


def test_num_worlds_is_the_one_yaml_tunable():
    """The counterweight to the test above, and the reason it is a whitelist
    rather than a blanket refusal.

    `num_worlds` is a different kind of number from the geometry: how many
    concurrent sessions this box holds resident (~816 MiB of ring each at 720P)
    is a sizing decision about the box, exactly like `max_num_pages` on the
    paged config. It changes what the node can *serve* and never what it
    computes — a world's arithmetic is identical whether it is alone in the ring
    or one of eight, which is what the equivalence tests below pin.
    """
    config = _ring_config()
    spec = KVSpec(resource_key="kv", nodes={"dit"}, config=config)

    apply_yaml_overrides([spec], {"resources": {"kv": {"num_worlds": 4}}})

    assert config.num_worlds == 4
    kv = _manager(config)
    assert kv.num_worlds == 4
    assert kv.total_slots(0) == 4 * kv.capacity(0)


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, 1.9, "2"])
def test_invalid_num_worlds_is_refused_at_both_entry_points(bad):
    """A node sized for zero worlds refuses every request at admit — a
    deployment that boots, reports healthy, and serves nothing."""
    with pytest.raises(ValueError, match="num_worlds"):
        _ring_config(num_worlds=bad)
    with pytest.raises(ValueError, match="num_worlds"):
        _ring_config().apply_yaml_overrides(num_worlds=bad)


def test_applying_num_worlds_does_not_rebaseline_the_head_counts():
    """`apply_yaml_overrides` validates `num_worlds` inline rather than
    re-running `__post_init__`, and that is load-bearing rather than tidiness.

    `KVConfig.__post_init__` snapshots `_unsharded_kv_heads` from the CURRENT
    head count. Re-running it after `shard()` would make the already-sharded
    count the new baseline, and the next `shard()` — the KV resource and every
    attention resource planned against this same config object each shard it,
    relying on it being idempotent — would narrow the cache a second time. Every
    shape stays valid; the node just quietly holds a fraction of the heads.
    """
    config = _ring_config(num_kv_heads=8)
    config.shard(2)
    sharded = config.num_kv_heads
    assert sharded == 4

    config.apply_yaml_overrides(num_worlds=4)
    config.shard(2)

    assert config.num_kv_heads == sharded, "shard() stopped being idempotent"


def test_an_empty_yaml_block_is_not_an_override():
    """Refusing every *key* must not refuse the empty *block*.

    `apply_yaml_overrides` in spec.py skips the whole `resources:` map when it
    is empty, but not the individual blocks under it — so `resources: {kv: {}}`
    reaches the config with zero kwargs. Raising there fails a deployment over
    a block that asked for nothing, and names `[]` as the offending keys.
    """
    spec = KVSpec(resource_key="kv", nodes={"dit"}, config=_ring_config())

    apply_yaml_overrides([spec], {"resources": {"kv": {}}})


def test_a_real_override_still_raises_through_the_yaml_entry_point():
    """The counterweight to the above, on the same path: a block that does
    carry a key must still be refused where a deployment would hit it."""
    spec = KVSpec(resource_key="kv", nodes={"dit"}, config=_ring_config())

    with pytest.raises(TypeError, match="checkpoint fact"):
        apply_yaml_overrides([spec], {"resources": {"kv": {"tokens_per_frame": 256}}})


def test_paged_overrides_still_work_on_a_paged_config():
    """The counterweight: `apply_yaml_overrides` raising on the ring must not
    be a blanket refusal that also broke the paged path."""
    config = PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=128)
    spec = KVSpec(resource_key="kv", nodes={"decoder"}, config=config)

    spec.apply_yaml_overrides(max_num_pages=17, page_size=64)

    assert (config.max_num_pages, config.page_size) == (17, 64)


# ── ownership ───────────────────────────────────────────────────────────


def test_two_capture_configs_can_each_open_and_claim():
    """CUDA-graph capture, verbatim: `DummyRowPool.ensure` keys its rid pool by
    ``f"{config_idx}_slot{slot}"``, so each capture config opens its own dummy
    rid, and none is ever `remove_request`-ed. Waypoint declares two configs
    (prime + rollout).

    This is why the claim is in `admit` and not `ingest_request`. Under a
    claim-at-ingest design — where the claim is taken when the rid is opened and
    handed back only when its storage is freed — config 1's `ensure()` raises
    against config 0's still-held claim, and capture dies there. `release_all()`,
    the only `free=True` call, runs after ALL captures, so nothing in between
    would have let go. The node is deliberately sized for ONE world here: with
    room for two the bug this pins would sail through capture and reappear as a
    node that can serve one fewer request than it was sized for.

    It is also what pins the clock check against capture. Every admit below
    declares frame 0 — the capture template's `frame_pos` is `torch.zeros(1)`
    and replay re-stages it, so capture never advances — and `_capture_one`
    drives admit and plan but never `commit`. If the check were keyed on
    anything the manager could learn without a commit, the second warmup admit
    would refuse and capture would die here rather than in production.
    """
    kv = _manager(_ring_config(num_worlds=1))
    pool = DummyRowPool(
        prefix="dit",
        step_runner=SimpleNamespace(ingest_request=kv.ingest_request),
        resources={"kv": kv},
    )

    # `_capture_one`, in order: ensure, prepare (admit -> plan), NUM_WARMUP
    # forwards each followed by a free=False reset and a re-prepare, the capture
    # forward, and the `finally` reset. No commit anywhere in it.
    for config_idx in (0, 1):
        rids = pool.ensure(f"{config_idx}_slot0", 1)
        ctx = StepContext(
            request_ids=tuple(rids), graph_walk="rollout", slot=0, capture=True,
        )
        step = _step(*rids, frame=0)
        outcome = kv.admit(step, ctx)
        assert outcome.ok, (
            f"capture config {config_idx} was refused the ring: {outcome.reason}"
        )
        kv.plan(step, ctx)
        for _ in range(2):  # CudaGraphRunner.NUM_WARMUP
            _rollout(kv, frames=1)
            pool.reset(rids)
            assert kv.admit(step, ctx).ok
            kv.plan(step, ctx)
        _rollout(kv, frames=1)  # the capture forward itself
        pool.reset(rids)

    pool.release_all()
    kv.post_warmup_validate()


def test_ingest_request_registers_without_claiming():
    kv = _manager()

    kv.ingest_request("a")
    kv.ingest_request("b")
    kv.ingest_request("a")  # idempotent: one NewRequest per partition

    assert kv.world_of("a") is None and kv.world_of("b") is None
    # neither registration took a world, so either may still be admitted
    assert kv.admit(_step("b"), _ctx("b")).ok


def test_admit_refuses_a_request_that_was_never_ingested():
    """A world claimed here would never come back. Worlds are handed back by
    `remove_request`, which the engine only ever runs for a request it opened
    (`Engine.add_request` -> `ingest_request` and `Engine.remove_request` ->
    `remove_request` are symmetric across every resource), so a claim taken for
    a rid outside that pairing leaks a world permanently — the node loses
    concurrency one request at a time with nothing raised, until every admit
    fails terminally and the cause is long gone.

    With one world the distinction was invisible, because the single claim was
    always overwritten by whoever asked next. It is not invisible with a pool.
    """
    kv = _manager(_ring_config(num_worlds=4))

    outcome = kv.admit(_step("stranger"), _ctx("stranger"))

    assert not outcome.ok
    assert type(outcome.reason) is AdmitRuntimeError
    assert "never ingested" in outcome.reason.message
    assert kv.world_of("stranger") is None
    assert len(kv._free_worlds) == 4, "a refused admit still took a world"


@pytest.mark.parametrize("num_worlds", [1, 2, 4])
def test_the_n_plus_first_request_is_refused_with_a_terminal_reason(num_worlds):
    """The pool is finite and exhaustion is terminal. The reason class is the
    whole point: `AllocationFailed` sends the scheduler to evict, but
    `supports_eviction` is False so nothing is evictable and the request spins;
    `RequestOffloading` waits on a `reload` that never comes. Both hang instead
    of failing. That reasoning does not change with the pool size — an exhausted
    ring is exhausted for the same reason at N=4 as at N=1.
    """
    kv = _manager(_ring_config(num_worlds=num_worlds))
    _open(kv, *[f"r{i}" for i in range(num_worlds)])
    kv.ingest_request("extra")

    outcome = kv.admit(_step("extra"), _ctx("extra"))

    assert not outcome.ok
    assert not outcome.ready
    assert type(outcome.reason) is AdmitRuntimeError
    assert not isinstance(outcome.reason, (AllocationFailed, RequestOffloading))
    assert f"all {num_worlds}" in outcome.reason.message


@pytest.mark.parametrize("num_worlds", [2, 4])
def test_concurrent_requests_get_distinct_worlds(num_worlds):
    """The point of the whole change, and the thing that has no meaning at N=1.

    Two requests handed the same world index would share one span: each would
    write over the other's frames and read them back as its own history, with
    every shape still valid and nothing raised. Distinctness is checked as a
    set, not against an expected assignment, because which index a request gets
    is allocator business; that it gets its own is not.
    """
    kv = _manager(_ring_config(num_worlds=num_worlds))
    rids = [f"r{i}" for i in range(num_worlds)]

    _open(kv, *rids)

    worlds = [kv.world_of(rid) for rid in rids]
    assert None not in worlds
    assert len(set(worlds)) == num_worlds, (
        f"worlds collided: {dict(zip(rids, worlds, strict=True))}"
    )
    assert set(worlds) == set(range(num_worlds)), "a world was skipped"
    assert not kv._free_worlds


def test_admit_is_idempotent_for_the_holder():
    """A rollout admits once per frame for as many frames as it runs."""
    kv = _manager()
    kv.ingest_request("a")

    assert all(kv.admit(_step("a"), _ctx("a")).ok for _ in range(4))
    assert kv.world_of("a") == 0


def test_a_refused_admit_leaves_every_previous_holder_in_place():
    """The refusal must not half-claim: if it stole a world on the way out, the
    original holder's next admit would be refused by the request that was itself
    just refused. Checked across a *pool* rather than a single claim, since a
    refusal that half-claimed would now do it to whichever holder happened to
    own the index it grabbed.
    """
    kv = _manager(_ring_config(num_worlds=2))
    _open(kv, "a", "b")
    before = {rid: kv.world_of(rid) for rid in ("a", "b")}
    kv.ingest_request("c")

    assert not kv.admit(_step("c"), _ctx("c")).ok

    assert {rid: kv.world_of(rid) for rid in ("a", "b")} == before
    assert kv.admit(_step("a"), _ctx("a")).ok
    assert kv.admit(_step("b"), _ctx("b")).ok
    assert kv.world_of("c") is None


@pytest.mark.parametrize("free", [False, True])
def test_reset_request_releases_one_world_and_only_one(free):
    """`free=True` is indistinguishable from `free=False`: there is no physical
    allocation to hand back, so the only thing either can do is zero and let
    go. Capture calls `free=False` between warmups and `free=True` once at the
    end, and both have to leave a world claimable.

    The "only one" half is the bug this replaces. The single-world version
    dropped *the* claim, which was the same statement then and would now end
    every other rollout on the node every time any request was reset.
    """
    kv = _manager(_ring_config(num_worlds=2))
    _open(kv, "a", "b")
    b_world = kv.world_of("b")

    kv.reset_request("a", free=free)

    assert kv.world_of("a") is None
    assert kv.world_of("b") == b_world, "resetting `a` released `b`'s world"
    kv.ingest_request("c")
    assert kv.admit(_step("c"), _ctx("c")).ok
    assert kv.world_of("c") == 0, "the freed world was not the one handed on"


def test_remove_request_releases_the_world_and_the_registration():
    """Both, and neither anyone else's. Dropping the registration is what makes
    the world genuinely returned rather than reserved for a rid the engine has
    already forgotten."""
    kv = _manager(_ring_config(num_worlds=2))
    _open(kv, "a", "b")
    b_world = kv.world_of("b")

    kv.remove_request("a")

    assert kv.world_of("a") is None
    assert kv.world_of("b") == b_world, "removing `a` released `b`'s world"
    # the registration went with it: `a` is a stranger again
    assert not kv.admit(_step("a"), _ctx("a")).ok
    kv.ingest_request("a")
    assert kv.admit(_step("a"), _ctx("a")).ok


def test_supports_preplan_stays_false():
    """It keeps `CudaGraphRunner._num_slots` at 1. Two slots exist so a plan for
    step N+1 can write buffers replay N is not reading; the only thing planned
    here is one `[1]` world index, so the second slot would be an identical
    graph at double the capture cost — and the only reason `_static_world_idx`
    would have to become one buffer per slot."""
    assert _manager().supports_preplan is False


def test_plan_stages_the_world_index_in_place_as_a_device_tensor():
    """The single most important property in the resource, and the one nothing
    else can catch.

    A Python int here would be constant-folded into the graph at capture and
    every replay would serve the capture-time world — reading and writing
    somebody else's history with no shape error and no exception. A freshly
    allocated tensor is the mirror failure: the graph baked the address of the
    buffer that existed at capture, so a rebind leaves every replay reading the
    orphaned original.

    Hence both halves below: the staged value is a `[1]` int64 device tensor,
    and `plan` writes *through* it rather than replacing it.
    """
    kv = _manager(_ring_config(num_worlds=4))
    _open(kv, "a", "b")
    staged = kv._static_world_idx
    assert staged.shape == (1,) and staged.dtype == torch.int64

    kv.plan(_step("b"), _ctx("b"))

    assert kv._static_world_idx is staged, "plan rebound the buffer capture baked"
    assert staged.data_ptr() == kv._static_world_idx.data_ptr()
    assert int(staged) == kv.world_of("b")

    kv.plan(_step("a"), _ctx("a"))
    assert int(staged) == kv.world_of("a")


def test_plan_refuses_a_batch_it_cannot_stage():
    """One world index per step, so one request per step. `admit` refuses a
    mixed batch first; reaching here means it was bypassed, and staging one of
    the two rids arbitrarily would run the other request's frame into the wrong
    world."""
    kv = _manager(_ring_config(num_worlds=2))
    _open(kv, "a", "b")

    with pytest.raises(ValueError, match="one world index per step"):
        kv.plan(_step("a", "b"), _ctx("a", "b"))


def test_plan_refuses_a_request_holding_no_world():
    kv = _manager()

    with pytest.raises(KeyError, match="no world for request"):
        kv.plan(_step("a"), _ctx("a"))


def test_admit_refuses_a_mixed_batch_naming_the_step_limit():
    """A step advances one world. The message has to say which number capped it
    — this is `max_batch_size`, not the ring, and a reader who reads it as a
    ring limit raises `num_worlds` and sees nothing change."""
    kv = _manager(_ring_config(num_worlds=4))
    _open(kv, "a", "b")

    outcome = kv.admit(_step("a", "b"), _ctx("a", "b"))

    assert not outcome.ok
    assert type(outcome.reason) is AdmitRuntimeError
    assert "max_batch_size" in outcome.reason.message


# ── the ring clock ──────────────────────────────────────────────────────
#
# The clock has to advance by exactly one per committed frame, per world. Every
# failure below is silent on its own: a skipped or repeated frame picks a
# different ring slot to write and a different one to hide, so the video keeps
# coming out — from a history that is no longer the one that was generated.
# `admit` is the only point per frame where the engine holds both the clock the
# forward is about to run at and the frame the last one committed.


def _drive(kv: RingKVManager, rid: str, frame: int, seed: int | None = None) -> None:
    """One engine step at ``frame``: admit, plan, forward, commit. Asserts the
    admit, since a test that meant to reach `commit` and was refused on the way
    should say so there rather than fail three lines later."""
    step, ctx = _step(rid, frame=frame), _ctx(rid)
    outcome = kv.admit(step, ctx)
    assert outcome.ok, f"{rid} frame {frame} was refused: {outcome.reason}"
    kv.plan(step, ctx)
    _rollout(kv, frames=1, seed=frame if seed is None else seed, start=frame)
    kv.commit(step, ctx)


def test_the_clock_check_is_silent_until_the_first_commit():
    """A fresh claim may start its clock anywhere. Nothing has been committed,
    so there is no history for a frame number to be inconsistent *with* — and a
    world restored by `load_state` legitimately resumes at frame 37, not 0."""
    kv = _manager()
    kv.ingest_request("a")

    assert kv.admit(_step("a", frame=37), _ctx("a")).ok
    # still nothing committed: re-driving the same frame is not yet a repeat
    assert kv.admit(_step("a", frame=37), _ctx("a")).ok
    assert kv.admit(_step("a", frame=0), _ctx("a")).ok


def test_a_skipped_frame_is_refused_with_a_terminal_reason():
    """The reason class matters as much as the refusal: `AllocationFailed` sends
    the scheduler off to evict against `supports_eviction=False`, so the request
    would hang instead of failing. Nothing about a desynced clock is fixable by
    eviction or reload."""
    kv = _manager()
    kv.ingest_request("a")
    _drive(kv, "a", 0)

    outcome = kv.admit(_step("a", frame=2), _ctx("a"))

    assert not outcome.ok
    assert not outcome.ready
    assert type(outcome.reason) is AdmitRuntimeError
    assert not isinstance(outcome.reason, (AllocationFailed, RequestOffloading))
    # names both numbers, and whose clock they belong to
    assert "frame 0" in outcome.reason.message
    assert "declares frame 2" in outcome.reason.message
    assert "'a'" in outcome.reason.message


def test_a_repeated_frame_is_refused():
    """The other half of the invariant, and the one a stalled `postprocess`
    produces: the ring already holds frame 0, so re-running it would overwrite
    that slot with a different frame's K/V while the visibility row hides it."""
    kv = _manager()
    kv.ingest_request("a")
    _drive(kv, "a", 0)
    _drive(kv, "a", 1)

    assert not kv.admit(_step("a", frame=1), _ctx("a")).ok


def test_each_world_runs_its_own_clock():
    """Per rid, not per resource. A shared clock would refuse the second
    request's very first frame the moment the first request had committed one —
    and, worse under a hypothetical "just take the max", would let a lagging
    world skip forward into a slot it never wrote.
    """
    kv = _manager(_ring_config(num_worlds=3))
    _open(kv, "a", "b", "c")

    _drive(kv, "a", 0)
    _drive(kv, "a", 1)
    _drive(kv, "a", 2)
    # `b` has committed nothing, so it may still start wherever it likes
    assert kv.admit(_step("b", frame=0), _ctx("b")).ok
    _drive(kv, "b", 41)
    # and `c`'s clock is not `a`'s or `b`'s either
    _drive(kv, "c", 7)

    assert kv.admit(_step("a", frame=3), _ctx("a")).ok
    assert kv.admit(_step("b", frame=42), _ctx("b")).ok
    assert kv.admit(_step("c", frame=8), _ctx("c")).ok
    # each is still refused its neighbour's next frame
    assert not kv.admit(_step("a", frame=42), _ctx("a")).ok
    assert not kv.admit(_step("b", frame=3), _ctx("b")).ok


def test_a_step_that_declares_no_clock_for_an_admitted_request_is_refused():
    """`RingKVStep` has no shape that declines to answer, so this is only
    reachable by hand — which is the point. The continuity check is the only
    thing standing between a stalled clock and a world that rewrites its own
    history, and the singular `frame_pos: int | None` this replaced switched it
    off for any batch a submodule could not describe with one number.
    """
    kv = _manager()
    kv.ingest_request("a")

    outcome = kv.admit(RingKVStep(frames=()), _ctx("a"))

    assert not outcome.ok
    assert type(outcome.reason) is AdmitRuntimeError
    assert "no ring clock" in outcome.reason.message


def test_a_frame_that_never_committed_can_be_re_driven():
    """`commit` is called straight-line after the forward, not in a `finally`,
    so a forward that raised commits nothing. Re-driving that frame at the same
    clock has to be allowed — the ring write is idempotent per (frame, layer),
    since the slot a frame overwrites is the slot its own visibility row hides,
    so a frame torn halfway through the layer stack heals on the re-run."""
    kv = _manager()
    kv.ingest_request("a")
    _drive(kv, "a", 0)

    # frame 1 admitted, forward raised: no commit
    assert kv.admit(_step("a", frame=1), _ctx("a")).ok

    assert kv.admit(_step("a", frame=1), _ctx("a")).ok
    _drive(kv, "a", 1)
    assert kv.admit(_step("a", frame=2), _ctx("a")).ok


def test_commit_records_the_frame_and_moves_no_kv():
    """The division the whole design rests on, and the same one the paged
    manager keeps: `KVManager.write_kv` scatters the bytes from the layer body
    and `KVManager.commit` only advances `stored_len`. Here the bytes land in
    `upsert`. `commit` runs on the host after the forward's launch *returns*,
    outside any captured graph, so a hook that moved KV would add 24 layers of
    launches per frame to the path capture exists to shorten."""
    kv = _manager()
    kv.ingest_request("a")
    _drive(kv, "a", 0)
    before = [layer.kv.clone() for layer in kv.layers]
    written = [layer.written.clone() for layer in kv.layers]

    step, ctx = _step("a", frame=1), _ctx("a")
    assert kv.admit(step, ctx).ok
    kv.commit(step, ctx)  # no forward between them

    for layer, kv_t, w_t in zip(kv.layers, before, written, strict=True):
        assert torch.equal(layer.kv, kv_t), "commit moved KV"
        assert torch.equal(layer.written, w_t), "commit changed visibility"
    # but it did record, so frame 1 is now spent
    assert not kv.admit(_step("a", frame=1), _ctx("a")).ok


def test_a_paged_step_is_refused_rather_than_skipping_the_check():
    """A `KVStep` carries no `frames`, so it would admit and commit exactly
    as before while switching the check off — the wrong thing that raises
    nothing. Same guard, same reason, as `build` refusing a PagedKVConfig."""
    kv = _manager()

    with pytest.raises(TypeError, match="RingKVStep"):
        kv.admit(KVStep(), _ctx("a"))
    with pytest.raises(TypeError, match="RingKVStep"):
        kv.commit(KVStep(), _ctx("a"))


@pytest.mark.parametrize(
    "drop", [
        lambda kv: kv.reset_request("a"),
        lambda kv: kv.reset_request("a", free=True),
        lambda kv: kv.remove_request("a"),
    ],
    ids=["reset_request", "reset_request_free", "remove_request"],
)
def test_dropping_the_world_drops_the_clock(drop):
    """An empty world has committed no frames. Keeping the count across a reset
    would hold the next claim to a continuity it cannot satisfy — capture, which
    resets between every warmup, is the caller that would hit it first."""
    kv = _manager()
    kv.ingest_request("a")
    _drive(kv, "a", 0)
    _drive(kv, "a", 1)

    drop(kv)

    # frame 0 again, which frame 1 would otherwise have made a repeat. Every one
    # of these hands the world back, so re-register first — whether the *claim*
    # survives each of them is the ownership tests' question, not this one's.
    kv.ingest_request("a")
    assert kv.admit(_step("a", frame=0), _ctx("a")).ok


def test_load_state_clears_the_clock():
    """The snapshot does not carry the clock — `get_state` covers the ring and
    only the ring, and the clock lives in `PerRequestState`. Inventing one
    would be worse than having none, so the first frame after a load sets it."""
    kv = _manager()
    kv.ingest_request("a")
    _drive(kv, "a", 0)
    _drive(kv, "a", 1)
    state = kv.get_state("a")

    kv.load_state("a", state)

    assert kv.admit(_step("a", frame=9), _ctx("a")).ok


def test_post_warmup_validate_catches_a_committed_frame():
    """Capture drives admit and plan but never commit, so this should be
    unreachable — which is why it is pinned rather than assumed. A clock left
    set by capture refuses the first real admit."""
    kv = _manager()
    kv.post_warmup_validate()

    kv.commit(_step("a", frame=4), _ctx("a"))  # a commit capture should never have made

    with pytest.raises(RuntimeError, match="during CUDA graph capture"):
        kv.post_warmup_validate()


# ── the allocation never moves ──────────────────────────────────────────


def test_reset_request_zeroes_one_span_without_reallocating():
    """A captured graph baked the address of the buffer that existed at capture
    time. Every reset path has to write through the same storage: a fresh
    tensor would detach every replay from the ring the model reads, silently.

    The second world is not scenery. The assertion is byte equality against a
    snapshot taken with `b`'s history already in the buffer, so it pins both
    halves at once — `a`'s span went to zero, and `b`'s came through untouched.
    A reset that zeroed the whole buffer (which is what the single-world version
    did, indistinguishably) fails on `b`.
    """
    kv = _manager(_ring_config(num_worlds=2))
    before = _ptrs(kv)
    _open(kv, "a", "b")
    assert (kv.world_of("a"), kv.world_of("b")) == (0, 1)

    kv.plan(_step("b"), _ctx("b"))
    _rollout(kv, frames=6, seed=2)
    kept_kv = [layer.kv.clone() for layer in kv.layers]
    kept_written = [layer.written.clone() for layer in kv.layers]

    kv.plan(_step("a"), _ctx("a"))
    _rollout(kv, frames=6, seed=3)
    assert any(
        not torch.equal(layer.kv, snap)
        for layer, snap in zip(kv.layers, kept_kv, strict=True)
    ), "the second rollout wrote nothing, so the reset below proves nothing"

    kv.reset_request("a")

    assert _ptrs(kv) == before
    for layer, kv_t, w_t in zip(kv.layers, kept_kv, kept_written, strict=True):
        assert torch.equal(layer.kv, kv_t), "reset_request reached another world"
        assert torch.equal(layer.written, w_t)
    # the scratch tail stays visible through every reset, per world: the frame
    # being denoised must always be able to attend to itself
    for layer in kv.layers:
        assert bool(_world_view(layer)[:, layer.ring_len :].all())
        assert not bool(_world_view(layer)[0, : layer.ring_len].any())


def test_remove_request_zeroes_one_span_without_reallocating():
    """Same invariant on the other release path, which is the one a real
    rollout ending takes. The single-world version zeroed the whole buffer
    here: with one world that was the same statement, with N it ends every
    concurrent rollout on the node every time any one of them finishes."""
    kv = _manager(_ring_config(num_worlds=2))
    before = _ptrs(kv)
    _open(kv, "a", "b")

    kv.plan(_step("b"), _ctx("b"))
    _rollout(kv, frames=4, seed=4)
    kept_kv = [layer.kv.clone() for layer in kv.layers]
    kept_written = [layer.written.clone() for layer in kv.layers]
    kv.plan(_step("a"), _ctx("a"))
    _rollout(kv, frames=4, seed=5)

    kv.remove_request("a")

    assert _ptrs(kv) == before
    for layer, kv_t, w_t in zip(kv.layers, kept_kv, kept_written, strict=True):
        assert torch.equal(layer.kv, kv_t), "remove_request reached another world"
        assert torch.equal(layer.written, w_t)


def test_a_reused_world_starts_empty():
    """The pairing `_release_world` exists to enforce: a world handed back to
    the pool still holding a dead request's frames is handed to the next request
    as its history — neither empty nor its own, and attended to as real. There
    is no release path that does not zero first."""
    kv = _manager(_ring_config(num_worlds=1))
    _open(kv, "a")
    kv.plan(_step("a"), _ctx("a"))
    _rollout(kv, frames=5, seed=6)
    assert any(bool(layer.kv.any()) for layer in kv.layers)

    kv.remove_request("a")
    _open(kv, "b")

    assert kv.world_of("b") == 0, "this test needs the same index handed on"
    assert all(not bool(layer.kv.any()) for layer in kv.layers)
    for layer in kv.layers:
        assert not bool(layer.written[: layer.ring_len].any())


def test_build_cuda_graph_buffers_allocates_nothing():
    kv = _manager()
    before = _ptrs(kv)

    kv.build_cuda_graph_buffers([], max_bs=1, max_seq_len=4096)

    assert _ptrs(kv) == before


def test_post_warmup_validate_catches_capture_residue():
    """NUM_WARMUP=2 plus the capture forward is three committing passes
    into whatever world the dummy held. Left there, the first real rollout
    handed that world attends to them as history."""
    kv = _manager(_ring_config(num_worlds=2))
    kv.post_warmup_validate()

    _rollout(kv, frames=1)
    with pytest.raises(RuntimeError, match="capture-time frames"):
        kv.post_warmup_validate()


def test_post_warmup_validate_catches_a_lingering_claim():
    """A dummy rid still holding a world after capture is a world no request
    will ever get back: the node boots reporting healthy and serves one fewer
    session than it was sized for, forever."""
    kv = _manager(_ring_config(num_worlds=2))
    _open(kv, "dummy")

    with pytest.raises(RuntimeError, match="still claimed"):
        kv.post_warmup_validate()

    kv.reset_request("dummy", free=True)
    kv.post_warmup_validate()


# ── state ───────────────────────────────────────────────────────────────


def test_get_state_covers_the_ring_and_nothing_else():
    """Pinned so a later field cannot quietly make the name a lie. A Waypoint
    world is at least three things: the ring, the streaming VAE's temporal
    receptive field, and `frame_pos`/seed/iteration in `PerRequestState`. This
    covers one; what a complete world state should contain is still open."""
    kv = _manager()
    _open(kv, "a")
    kv.plan(_step("a"), _ctx("a"))
    _rollout(kv, frames=3)

    state = kv.get_state("a")

    assert set(state) == {"layers"}
    assert len(state["layers"]) == N_LAYERS
    assert all(len(entry) == 2 for entry in state["layers"])


@pytest.mark.parametrize("call", ["get_state", "load_state"])
def test_state_is_refused_for_a_request_holding_no_world(call):
    """Scoped to a rid because the buffer is not. An unscoped snapshot of a
    multi-world ring would carry every concurrent request's history, and loading
    it back would overwrite worlds the caller never asked about — so there is no
    unscoped spelling to fall back to, and a rid that owns nothing has to say
    so."""
    kv = _manager(_ring_config(num_worlds=2))
    _open(kv, "a")

    with pytest.raises(KeyError, match="no world for request"):
        if call == "get_state":
            kv.get_state("ghost")
        else:
            kv.load_state("ghost", kv.get_state("a"))


def test_get_state_covers_one_world_and_is_cloned_not_aliased():
    """Two properties, one setup, because they fail together in practice: the
    caller holds the snapshot across rollout steps that overwrite the rings in
    place, and an aliased snapshot silently tracks the live world instead. The
    span check is the multi-world half — a snapshot the size of the whole buffer
    would carry `b`'s history into `a`'s save file."""
    kv = _manager(_ring_config(num_worlds=2))
    _open(kv, "a", "b")
    kv.plan(_step("a"), _ctx("a"))
    _rollout(kv, frames=2)
    state = kv.get_state("a")
    saved = [(a.clone(), b.clone()) for a, b in state["layers"]]

    for layer, (kv_t, w_t) in zip(kv.layers, state["layers"], strict=True):
        assert kv_t.shape[-2] == layer.capacity, "the snapshot is not one world"
        assert w_t.shape == (layer.capacity,)

    kv.plan(_step("b"), _ctx("b"))
    _rollout(kv, frames=3, seed=7)
    kv.plan(_step("a"), _ctx("a"))
    _rollout(kv, frames=3, seed=8, start=2)

    for (kv_t, w_t), (kv_s, w_s) in zip(state["layers"], saved, strict=True):
        assert torch.equal(kv_t, kv_s)
        assert torch.equal(w_t, w_s)


def test_load_state_copies_into_one_span_of_the_fixed_allocation():
    """A `load_state` that assigned a fresh tensor would detach every captured
    CUDA graph from the pointer it baked, and nothing would raise. Writing into
    a *span* rather than the whole buffer is the second half of that: every
    other resident world has to come through untouched, or restoring one
    session resets its neighbours."""
    kv = _manager(_ring_config(num_worlds=2))
    _open(kv, "a", "b")
    kv.plan(_step("a"), _ctx("a"))
    _rollout(kv, frames=5)
    state = kv.get_state("a")
    before = _ptrs(kv)

    kv.plan(_step("b"), _ctx("b"))
    _rollout(kv, frames=4, seed=9)
    b_kv = [layer.kv.clone() for layer in kv.layers]
    b_written = [layer.written.clone() for layer in kv.layers]

    kv.plan(_step("a"), _ctx("a"))
    _rollout(kv, frames=3, seed=11, start=5)
    assert not torch.equal(kv.layers[0].kv, b_kv[0]), (
        "the second rollout has to actually move the ring, or the round trip "
        "below proves nothing"
    )

    kv.load_state("a", state)

    assert _ptrs(kv) == before
    for i, layer in enumerate(kv.layers):
        lo, hi = layer.world_span(kv.world_of("a"))
        assert torch.equal(layer.kv[:, :, :, lo:hi], state["layers"][i][0])
        assert torch.equal(layer.written[lo:hi], state["layers"][i][1])
        b_lo, b_hi = layer.world_span(kv.world_of("b"))
        assert torch.equal(layer.kv[:, :, :, b_lo:b_hi], b_kv[i][:, :, :, b_lo:b_hi]), (
            "load_state reached another world's span"
        )
        assert torch.equal(layer.written[b_lo:b_hi], b_written[i][b_lo:b_hi])


def test_a_state_is_portable_between_rings_of_different_widths():
    """The snapshot is one world's span, so how many neighbours it had is not
    part of it. This is not incidental: a node resized from 2 worlds to 4
    between restarts must still be able to load the sessions it wrote, and a
    guard that compared against the whole buffer would refuse them all."""
    small = _manager(_ring_config(num_worlds=2))
    big = _manager(_ring_config(num_worlds=4))
    _open(small, "a")
    small.plan(_step("a"), _ctx("a"))
    _rollout(small, frames=3, seed=13)
    _open(big, "x", "y")  # so `y` lands on a nonzero world index

    big.load_state("y", small.get_state("a"))

    for layer, (kv_t, w_t) in zip(big.layers, small.get_state("a")["layers"], strict=True):
        lo, hi = layer.world_span(big.world_of("y"))
        assert torch.equal(layer.kv[:, :, :, lo:hi], kv_t)
        assert torch.equal(layer.written[lo:hi], w_t)
    # and `x`'s world is still empty
    for layer in big.layers:
        lo, hi = layer.world_span(big.world_of("x"))
        assert not bool(layer.kv[:, :, :, lo:hi].any())


def test_load_state_refuses_a_geometry_that_would_broadcast():
    """The shape guard, not `copy_`, is what has to catch this. The failure it
    exists for is a state saved under one horizon loaded into another — a 360P
    state into a 720P ring, or a compacted state into a `full_global_ring` one.
    Where the dims happen to line up, `copy_` broadcasts a frame across the ring
    and produces a world made of one repeated moment, with no error anywhere."""
    small = _manager(_ring_config(ring_frames=2))
    big = _manager(_ring_config(ring_frames=RING_FRAMES))
    _open(small, "a")
    small.plan(_step("a"), _ctx("a"))
    _rollout(small, frames=2)
    _open(big, "a")

    with pytest.raises(ValueError, match="state shape"):
        big.load_state("a", small.get_state("a"))


def test_load_state_refuses_a_different_layer_count():
    kv = _manager()
    _open(kv, "a")
    state = kv.get_state("a")
    state["layers"] = state["layers"][:-1]

    with pytest.raises(ValueError, match="layers"):
        kv.load_state("a", state)


# ── the visible row ─────────────────────────────────────────────────────


def test_visible_is_the_layers_scratch_buffer_and_must_be_read_immediately():
    """`upsert` hands back `_mask_written` itself, not a copy: a fresh
    `[total_slots]` buffer on each of the 120 upserts per frame is allocation
    the compiled region does not need.

    The cost is an aliasing obligation the consumer has to honour — read it
    before the next `upsert` on the same layer. The 4+1 schedule does (each
    pass consumes the row inside the same attention call). A consumer that
    stashed it would attend under a *later* frame's visibility: the wrong ring
    slots, no exception, drifting video — and, with worlds resident, possibly
    another world's mask entirely.

    The clone below is what makes this test bite: it shows the two frames'
    visibility genuinely differ, so `visible0 is visible1` is a statement about
    aliasing and not a vacuous equality between identical rows.
    """
    kv = _manager()
    gen = torch.Generator().manual_seed(3)
    k, v = _frame(kv, gen)

    _, _, visible0 = kv.upsert(k, v, 0, torch.tensor(0, dtype=torch.int64), commit=True)
    snapshot = visible0.clone()
    _, _, visible1 = kv.upsert(k, v, 0, torch.tensor(1, dtype=torch.int64), commit=True)

    assert visible1 is visible0
    assert not torch.equal(snapshot, visible1), (
        "frames 0 and 1 must differ in visibility, or this test proves nothing"
    )
    # the buffer the caller was handed at frame 0 now reads as frame 1
    assert not torch.equal(snapshot, visible0)


def test_visible_hides_the_slot_this_frame_is_about_to_overwrite():
    """At the resource seam: without it the current frame attends to the
    stale frame still occupying its ring slot."""
    kv = _manager()
    gen = torch.Generator().manual_seed(4)
    for f in range(RING_FRAMES + 1):
        k, v = _frame(kv, gen)
        _, _, visible = kv.upsert(k, v, 0, torch.tensor(f, dtype=torch.int64), commit=True)
        slot = (f % RING_FRAMES) * TPF
        assert not bool(visible[slot : slot + TPF].any()), f"frame {f} sees its own slot"
        assert bool(visible[kv.layers[0].ring_len :].all()), "scratch must stay visible"


def test_upsert_returns_the_whole_buffer_and_delegates_by_layer():
    """K and V span every resident world — there is one buffer and no view is
    taken — and the visibility row is the same length, because it is what cuts
    the caller's world back out of it. `capacity` and `total_slots` are both
    named for this reason: using either where the other belongs is an off-by-N
    that produces a valid shape."""
    kv = _manager(_ring_config(num_worlds=3))
    gen = torch.Generator().manual_seed(5)
    k, v = _frame(kv, gen)

    for layer_idx in range(N_LAYERS):
        k_all, v_all, visible = kv.upsert(
            k, v, layer_idx, torch.tensor(0, dtype=torch.int64), commit=True
        )
        total = kv.total_slots(layer_idx)
        assert total == 3 * kv.capacity(layer_idx)
        assert k_all.shape[-2] == total and v_all.shape[-2] == total
        assert visible.shape == (total,)
        assert k_all.shape[0] == 1, "the world dim is folded into tokens, not dim 0"
        assert k_all.data_ptr() == kv.layers[layer_idx].kv.data_ptr()


def test_planned_attention_skips_per_upsert_visibility_reconstruction():
    """A planned block mask makes the token-level scratch row dead output."""
    kv = _manager()
    _open(kv, "r")
    kv.plan(_step("r"), _ctx("r"))
    layer = kv.layers[0]
    layer._mask_written.copy_(
        torch.arange(layer.total_slots).remainder(2).to(torch.bool)
    )
    sentinel = layer._mask_written.clone()
    k, v = _frame(kv, torch.Generator().manual_seed(91))

    _, _, returned = kv.upsert(
        k,
        v,
        0,
        torch.tensor(0, dtype=torch.int64),
        commit=False,
        build_visibility=False,
    )

    assert returned.data_ptr() == layer._mask_written.data_ptr()
    assert torch.equal(returned, sentinel)


def test_frozen_passes_leave_the_ring_byte_identical():
    """At the resource seam: `commit` is per call and the manager keeps no
    frozen state between calls, because all five passes of a frame sit inside
    one engine step and the step lifecycle never sees the boundary. If the
    manager ever started latching it, this is what would break."""
    kv = _manager()
    gen = torch.Generator().manual_seed(6)
    before = [layer.kv.clone() for layer in kv.layers]

    for _ in range(4):
        for layer_idx in range(N_LAYERS):
            k, v = _frame(kv, gen)
            kv.upsert(k, v, layer_idx, torch.tensor(0, dtype=torch.int64), commit=False)

    for layer, snapshot in zip(kv.layers, before, strict=True):
        assert torch.equal(layer.kv[:, :, :, : layer.ring_len], snapshot[:, :, :, : layer.ring_len])
        assert not bool(layer.written[: layer.ring_len].any())


# ── worlds are isolated ─────────────────────────────────────────────────
#
# The two tests below are the ones that matter most in this file. Nothing
# physical separates two resident worlds: they share one `kv` tensor, one
# `written` row and one `_mask_written` scratch, and `upsert` returns K and V
# spanning all of it. The only thing that stops world 1 attending to world 0's
# frames is a single `&=` in `cache.py`. Delete it and every shape is still
# valid, every test above still passes, and the video drifts between sessions.


def _w(idx: int) -> torch.Tensor:
    """A world index in the shape the forward path uses everywhere: a `[1]`
    int64 tensor, never a Python int. See `cache.py`'s module docstring."""
    return torch.tensor([idx], dtype=torch.int64)


@pytest.mark.parametrize("num_worlds", [2, 4])
@pytest.mark.parametrize("pinned_dilation", [1, 8])
def test_the_flat_ring_matches_one_ring_per_world(num_worlds, pinned_dilation):
    """N worlds folded into one token axis are bit-identical to N separate
    single-world rings, span for span.

    This is the equivalence the whole layout rests on and the reason
    `num_worlds` can be a deployment knob at all: a world's arithmetic must not
    depend on how many neighbours it has. Three things are compared and all
    three are necessary — the K/V bytes (the write landed in the right slots),
    `written` (the bookkeeping did too), and the visibility row restricted to
    the span (the mask agrees). The fourth assertion is the one with no
    single-world counterpart: everything OUTSIDE the span is False, which is
    isolation itself.

    The own-world term is checked as `_world_of_slot == world_idx` computed from
    the span arithmetic, not read back off the implementation, so a mask built
    from the wrong comparison cannot agree with it by construction.
    """
    kwargs = dict(
        n_kv_heads=N_KV_HEADS, ring_frames=RING_FRAMES, ring_buckets=RING_FRAMES,
        d_head=D_HEAD, tokens_per_frame=TPF, pinned_dilation=pinned_dilation,
        dtype=torch.float32, device="cpu",
    )
    flat = LayerRingCache(num_worlds=num_worlds, **kwargs)
    solo = [LayerRingCache(num_worlds=1, **kwargs) for _ in range(num_worlds)]
    assert flat.total_slots == num_worlds * flat.capacity
    assert all(s.capacity == flat.capacity for s in solo)

    # Independent clocks and a deliberately uneven interleave: with every world
    # on the same frame, a mask that ignored `world_idx` entirely would still
    # hide the same slots and this test would pass against it.
    gens = [torch.Generator().manual_seed(100 + w) for w in range(num_worlds)]
    clocks = [3 * w for w in range(num_worlds)]
    order = torch.Generator().manual_seed(41)

    for _ in range(20 * num_worlds):
        w = int(torch.randint(0, num_worlds, (1,), generator=order).item())
        frame_pos = torch.tensor(clocks[w], dtype=torch.int64)
        for commit in (False, False, False, False, True):
            kv = torch.randn(2, 1, N_KV_HEADS, TPF, D_HEAD, generator=gens[w])
            _, _, flat_vis = flat.upsert(kv, frame_pos, commit, _w(w))
            _, _, solo_vis = solo[w].upsert(kv, frame_pos, commit, _w(0))

            lo, hi = flat.world_span(w)
            assert torch.equal(flat_vis[lo:hi], solo_vis), (
                f"world {w} frame {clocks[w]}: visibility diverged from a solo ring"
            )
            outside = torch.ones_like(flat_vis)
            outside[lo:hi] = False
            assert not bool((flat_vis & outside).any()), (
                f"world {w} can see {int((flat_vis & outside).sum())} slots outside "
                "its own span"
            )
            # the own-world term, derived here rather than read from the cache
            own = (
                torch.arange(flat.total_slots) // flat.capacity
            ) == w
            assert torch.equal(flat_vis, flat_vis & own)
        clocks[w] += 1

    for w in range(num_worlds):
        lo, hi = flat.world_span(w)
        assert torch.equal(flat.kv[:, :, :, lo:hi], solo[w].kv), f"world {w} ring bytes"
        assert torch.equal(flat.written[lo:hi], solo[w].written), f"world {w} written"


def test_worlds_interleave_without_reaching_each_other():
    """The same claim one level up, driven through the resource lifecycle
    rather than the cache: three requests, three independent clocks starting at
    different frames, forty frames each at five passes, interleaved in an order
    none of them controls — against three managers each holding one world and
    driven in isolation.

    This is what a node actually does, and it is where a bug in the *lifecycle*
    would show up rather than a bug in the mask: a `plan` that staged the wrong
    index, a `commit` that recorded against the wrong rid, a claim that two
    requests shared. Every one of those is invisible at N=1.

    Each request's K/V comes from its own generator, so the interleave order
    cannot change what any request writes — only where it lands.
    """
    rids = ["a", "b", "c"]
    starts = {"a": 0, "b": 5, "c": 11}
    frames = 40

    flat = _manager(_ring_config(num_worlds=3))
    _open(flat, *rids)
    solo = {rid: _manager(_ring_config(num_worlds=1)) for rid in rids}
    for rid, mgr in solo.items():
        _open(mgr, rid)

    order = torch.Generator().manual_seed(53)
    schedule = [rid for rid in rids for _ in range(frames)]
    perm = torch.randperm(len(schedule), generator=order).tolist()
    schedule = [schedule[i] for i in perm]
    assert schedule[:3] != rids, "the schedule is round-robin; interleave nothing"

    clocks = dict(starts)
    seeds = {rid: 200 + i for i, rid in enumerate(rids)}
    for rid in schedule:
        frame = clocks[rid]
        seed = seeds[rid] * 1000 + frame
        _drive(flat, rid, frame, seed=seed)
        _drive(solo[rid], rid, frame, seed=seed)
        clocks[rid] += 1

    for rid in rids:
        world = flat.world_of(rid)
        for i, (layer, ref) in enumerate(zip(flat.layers, solo[rid].layers, strict=True)):
            lo, hi = layer.world_span(world)
            assert torch.equal(layer.kv[:, :, :, lo:hi], ref.kv), (
                f"{rid} layer {i}: interleaving changed what the world holds"
            )
            assert torch.equal(layer.written[lo:hi], ref.written), (
                f"{rid} layer {i}: interleaving changed what the world can see"
            )

    # and the mask hid every other world completely, on the last step of each
    gen = torch.Generator().manual_seed(59)
    for rid in rids:
        world = flat.world_of(rid)
        flat.plan(_step(rid), _ctx(rid))
        for layer_idx in range(N_LAYERS):
            k, v = _frame(flat, gen)
            _, _, visible = flat.upsert(
                k, v, layer_idx, torch.tensor(clocks[rid], dtype=torch.int64), commit=False
            )
            lo, hi = flat.layers[layer_idx].world_span(world)
            seen = visible.clone()
            seen[lo:hi] = False
            assert not bool(seen.any()), (
                f"{rid} layer {layer_idx} can see {int(seen.sum())} slots belonging "
                "to another world"
            )
            assert bool(visible[lo:hi].any()), (
                "the mask hid everything, including this world's own history; "
                "an all-False row would pass the check above vacuously"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to capture")
def test_the_captured_world_index_is_read_at_replay_not_baked_at_capture():
    """The property the entire layout exists for, and the only test that can
    see it.

    `world_idx` is a `[1]` int64 device tensor rather than a Python int for one
    reason: a host int — or a `kv[:, w]` view taken with one — has its value (or
    its `storage_offset`) folded into the graph at capture time, and every
    subsequent replay then serves whichever world capture happened to hold.
    Nothing raises. Shapes are identical. One session's frames land in another
    session's ring and both keep producing video.

    So: capture against world 0, then release it, push the real request onto a
    different index, restage, and replay. The write has to follow the staged
    tensor. Every other world, including the one capture used, has to be
    untouched.
    """
    kv = _manager(_ring_config(num_worlds=4), device="cuda")
    layer = kv.layers[0]

    _open(kv, "cap")
    kv.plan(_step("cap"), _ctx("cap"))
    assert kv.world_of("cap") == 0, "this test needs capture to hold world 0"

    static_k = torch.zeros(1, N_KV_HEADS, TPF, D_HEAD, dtype=torch.float32, device="cuda")
    static_v = torch.zeros_like(static_k)
    static_frame = torch.zeros((), dtype=torch.int64, device="cuda")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            kv.upsert(static_k, static_v, 0, static_frame, commit=True)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, _, captured_visible = kv.upsert(
            static_k, static_v, 0, static_frame, commit=True
        )
    torch.cuda.synchronize()

    kv.reset_request("cap", free=True)
    for world in range(layer.num_worlds):
        layer.reset(world)

    # push the real request off world 0: three placeholders take 0, 1, 2
    _open(kv, "pad0", "pad1", "pad2", "real")
    world = kv.world_of("real")
    assert world == 3
    kv.plan(_step("real"), _ctx("real"))

    static_k.fill_(1.5)
    static_v.fill_(-2.5)
    graph.replay()
    torch.cuda.synchronize()

    lo, hi = layer.world_span(world)
    span = layer.kv[:, :, :, lo:hi]
    assert bool((span == 1.5).any()), (
        "the replay wrote nothing into the world `plan` staged; `world_idx` was "
        "baked at capture"
    )
    assert bool(layer.written[lo : lo + TPF].all()), "the frame's ring slot went unmarked"
    for other in range(kv.num_worlds):
        if other == world:
            continue
        o_lo, o_hi = layer.world_span(other)
        assert not bool(layer.kv[:, :, :, o_lo:o_hi].any()), (
            f"the replay wrote into world {other}; it was staged to write world "
            f"{world}"
        )
        assert not bool(layer.written[o_lo : o_lo + layer.ring_len].any())
    # the visibility row the graph returns is the layer's scratch, restaged too
    assert bool(captured_visible[lo:hi].any())
    other_lo, other_hi = layer.world_span(0)
    assert not bool(captured_visible[other_lo:other_hi].any()), (
        "the captured mask still shows the capture-time world"
    )


# ── the two branches that do nothing under this schedule ────────────────


def test_the_bucket_rounding_is_unobservable_off_write_steps():
    """`bucket` rounds up, verbatim from the reference, and the rounding is
    dead arithmetic: `ring_idx` is discarded at both of its use sites off a
    write step, and on a write step `frame_pos` is a multiple of the dilation,
    where ceil and floor agree.

    What is pinned here is the *unobservability*, which is the load-bearing
    half — it holds for any rounding, not just floor, so it survives someone
    changing the expression as well as someone changing its consumers.

    Only the dilated layer can exercise it. At stride 1 every frame is a write
    step, the off-write-step case is empty, and a version of this test that ran
    layer 0 would assert nothing at all; hence the counter at the end.
    """
    layer = _manager(_ring_config(num_worlds=2)).layers[N_LAYERS - 1]
    d = layer.pinned_dilation
    assert d > 1, "this test needs a layer that has off-write-step frames"

    gen = torch.Generator().manual_seed(31)
    off_steps = 0
    for f in range(2 * d * RING_FRAMES):  # two full wraps of the bucket ring
        before_kv = layer.kv.clone()
        before_written = layer.written.clone()
        kv = torch.randn(2, 1, N_KV_HEADS, TPF, D_HEAD, generator=gen)

        # commit=True is the harder case: if `ring_idx` were live off a write
        # step, this is the call that would write through it. World 1, not 0, so
        # a placeholder address that forgot the world offset lands somewhere
        # this test can see.
        _, _, visible = layer.upsert(kv, torch.tensor(f, dtype=torch.int64), True, _w(1))

        lo, hi = layer.world_span(1)
        if f % d:
            off_steps += 1
            expected = before_written.clone()
            expected[: lo] = False
            expected[hi :] = False
            assert torch.equal(visible, expected), f"frame {f} hid a ring slot"
            assert torch.equal(
                layer.kv[:, :, :, lo : lo + layer.ring_len],
                before_kv[:, :, :, lo : lo + layer.ring_len],
            ), f"frame {f} wrote the ring through a slot it should not address"
            assert torch.equal(layer.written, before_written), f"frame {f} marked a slot"
        else:
            assert (f + d - 1) // d == f // d, f"ceil != floor on write step {f}"
        assert not bool(layer.kv[:, :, :, :lo].any()), f"frame {f} wrote world 0"

    assert off_steps, "no off-write-step frame ran; this test proved nothing"


def test_committing_on_every_pass_matches_the_4_plus_1_schedule():
    """`commit` is dead under the shipped schedule and live in general.

    Dead: on a write step the visibility row hides `ring_idx` from all five
    passes and the fifth writes it last, so committing on all five leaves the
    same visible K/V and a byte-identical ring. Off a write step `dst` is
    `current_idx`, which the unconditional write already filled with the same
    data, so the commit changes nothing anywhere.

    `raw_differs` is what stops the first half being vacuous: mid-frame the two
    rings genuinely diverge, and it is the mask — block-aligned, so the
    BlockMask drops the region whole — that makes the divergence unobservable.

    Live: committing on no pass diverges. Deleting the branch on the strength
    of the first half would pass everything above and produce a world that
    never remembers a frame.
    """
    def drive(kv, schedule):
        """One `(masked_k, masked_v, raw_k, visible)` per upsert, lazily, so
        the three schedules can be compared in lockstep without three full
        traces resident at once."""
        gen = torch.Generator().manual_seed(37)
        for f in range(3 * RING_FRAMES):
            frame_pos = torch.tensor(f, dtype=torch.int64)
            for commit in schedule:
                for layer_idx in range(N_LAYERS):
                    k, v = _frame(kv, gen)
                    k_all, v_all, visible = kv.upsert(
                        k, v, layer_idx, frame_pos, commit=commit
                    )
                    m = visible[None, None, :, None]
                    yield k_all * m, v_all * m, k_all, visible

    shipped, always, never = _manager(), _manager(), _manager()

    raw_differs = False
    for (sk, sv, s_raw, s_vis), (ak, av, a_raw, a_vis) in zip(
        drive(shipped, (False, False, False, False, True)),
        drive(always, (True,) * 5),
        strict=True,
    ):
        assert torch.equal(s_vis, a_vis), "visibility diverged"
        assert torch.equal(sk, ak) and torch.equal(sv, av), "visible K/V diverged"
        raw_differs |= not torch.equal(s_raw, a_raw)

    assert raw_differs, (
        "the rings never differed mid-frame, so the mask was never what made "
        "them agree and this test proves nothing"
    )
    for s_layer, a_layer in zip(shipped.layers, always.layers, strict=True):
        assert torch.equal(s_layer.kv, a_layer.kv), "end-of-frame ring diverged"
        assert torch.equal(s_layer.written, a_layer.written)

    for _ in drive(never, (False,) * 5):
        pass
    assert any(
        not torch.equal(s_layer.kv, n_layer.kv)
        for s_layer, n_layer in zip(shipped.layers, never.layers, strict=True)
    ), "suppressing every commit changed nothing; the branch is not live"


# ── the shapes the forward path insists on ──────────────────────────────


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(torch.tensor(0, dtype=torch.int64), id="scalar"),
        pytest.param(torch.tensor([0, 1], dtype=torch.int64), id="two"),
        pytest.param(torch.tensor([0], dtype=torch.int32), id="int32"),
    ],
)
def test_upsert_refuses_a_world_index_that_is_not_the_staged_shape(bad):
    """`[1]` int64, and nothing else — including `[]`, which would broadcast
    just as well here. This is not a shape necessity, it is the shape the
    resource stages into its static buffer and the shape a replay re-stages, and
    letting a second spelling through is how those two drift apart with nothing
    to catch it."""
    layer = _manager().layers[0]
    kv = torch.zeros(2, 1, N_KV_HEADS, TPF, D_HEAD)

    with pytest.raises(RuntimeError, match="world_idx must be a"):
        layer.upsert(kv, torch.tensor(0, dtype=torch.int64), True, bad)


def test_a_world_index_out_of_range_is_caught_on_the_host_paths():
    """Only on the host paths. The forward cannot check it — `world_idx` is a
    device tensor there and comparing it would cost a sync per upsert — so an
    out-of-range index in the graph silently writes past the buffer's last
    world or wraps into another's. What keeps it in range is that `admit` is the
    only thing that ever produces one."""
    layer = _manager(_ring_config(num_worlds=2)).layers[0]

    assert layer.world_span(1) == (layer.capacity, 2 * layer.capacity)
    for bad in (-1, 2):
        with pytest.raises(IndexError, match="out of range"):
            layer.world_span(bad)
        with pytest.raises(IndexError, match="out of range"):
            layer.reset(bad)
