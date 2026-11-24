"""Comprehensive tests for `lifetime="episode"` across all non-`DENSE_WINDOW` storages."""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

import mstar.engine.resources.kv.manager as kv_manager_module
from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.bounded.config import (
    BoundedKVConfig,
    BoundedKVSpec,
    BoundedKVStep,
    SinkWindow,
    StreamPosition,
)
from mstar.engine.resources.kv.bounded.manager import BoundedKVManager
from mstar.engine.resources.kv.config import (
    KVReqConfig,
    KVSpec,
    KVStep,
    PagedKVConfig,
    PrefixSpan,
    RingKVConfig,
    RingKVLayerConfig,
    RingKVStep,
    StreamLayout,
)
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.ring.manager import RingKVManager
from mstar.engine.resources.kv.rolling.config import (
    RollingKVConfig,
    RollingKVReqConfig,
    RollingKVSpec,
    RollingKVStep,
    RollingPosition,
)
from mstar.engine.resources.kv.rolling.manager import RollingKVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.position.config import LongRoPE, PositionConfig, PositionReqConfig, PositionStep
from mstar.engine.resources.position.manager import LongRopeManager, RopeManager
from mstar.engine.resources.recurrent.config import RecurrentBlockConfig, RecurrentStateConfig, RecurrentStep
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.sized.config import SizedField, SizedValueConfig, SizedValueReqConfig, SizedValueSpec
from mstar.engine.resources.sized.store import SizedValueStore
from mstar.engine.resources.step import ASKS, AdmitRuntimeError, AllocationFailed, Segment, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Bidirectional,
    Causal,
    Entry,
    FrameWindow,
    KVEntry,
    Positions,
    Read,
    Size,
    TensorType,
    Window,
    host,
)
from mstar.engine.state.adapters import (
    BOUNDED,
    HOST_VALUE,
    PAGED,
    PAGED_BY_WALK,
    PAGED_CROSS,
    RING,
    ROLLING,
    SIZED_VALUE,
    VALUE,
    VALUE_WITH_HOST,
    WITHIN_STEP,
    StateSetup,
    build_handle,
    build_specs,
    node_handles,
    position_key,
    state_steps,
    storage_of,
)
from mstar.engine.state.host_fields import HostFields
from mstar.model.base import RebuildPolicy
from mstar.model.submodule_base import NodeSubmodule
from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.testing_utils.model import make_model
from mstar.worker.rid_table import RidTable

CPU = torch.device("cpu")
PAGE_SIZE = 8


def _kv_entry(heads: int = 2, head_dim: int = 8, dtype: torch.dtype | None = None) -> Entry:
    if dtype is None:
        return KVEntry(heads=heads, head_dim=head_dim)
    t = TensorType((heads, head_dim), dtype=dtype, shard=0)
    return Entry({"k": t, "v": t})


def _paged_kv(
    name: str = "kv",
    *,
    lifetime: str = "episode",
    positions: Positions | None = None,
    reads: dict[str, Read] | None = None,
    max_len: int = 64,
) -> Array:
    return Array(
        name,
        nodes={"llm"},
        layers={0},
        entry=_kv_entry(),
        max_len=max_len,
        reads=reads or {"attn": Read(mask=Causal(), op=Attend(q_heads=2))},
        positions=positions,
        lifetime=lifetime,
    )


def _paged_kv_manager(
    *,
    max_num_pages: int = 16,
    page_size: int = PAGE_SIZE,
    prefix_cache: bool = False,
    episode_lifetime: bool = True,
) -> KVManager:
    cfg = PagedKVConfig(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=64,
        max_num_pages=max_num_pages,
        page_size=page_size,
        prefix_cache=prefix_cache,
    )
    kv = KVManager(
        cfg=cfg,
        name="kv",
        joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        device=CPU,
        dtype=torch.float32,
        episode_lifetime=episode_lifetime,
    )
    if prefix_cache:
        kv.enable_prefix_cache(b"episode-root")
    return kv


def _kv_req(tokens: list[int], *, cap: int = 0, walk: str = "prefill", keyed: bool = False) -> KVReqConfig:
    pages = [tokens[i:i + PAGE_SIZE] for i in range(0, len(tokens), PAGE_SIZE)] if tokens else []
    prefix_keys = {"main": chain(pages, root=b"")} if keyed and pages else None
    prefix_tail = {"main": pages[-1] if len(tokens) % PAGE_SIZE else []} if keyed else None
    return KVReqConfig(
        prefix_keys=prefix_keys,
        prefix_tail=prefix_tail,
        loop_caps={"decode": cap} if cap else {},
        kv_layout={
            "main": StreamLayout(
                spans=(PrefixSpan(len(tokens), len(tokens), walk),) if tokens else (),
                growth=(("decode", 1),) if cap else (),
                keyed_spans=1 if keyed and tokens else 0,
            )
        },
    )


def _kv_step(kv: KVManager, rid: int | str, span: int, *, label: str = "main", commit: bool = True) -> None:
    step = KVStep(segments=(Segment(rid, label, span),), commit=commit)
    ctx = StepContext(request_ids=[rid], graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    if span > 0:
        k = torch.full((span, 2, 8), float(rid if isinstance(rid, int) else 1), dtype=torch.float32)
        v = torch.full((span, 2, 8), float(rid if isinstance(rid, int) else 1) + 0.5, dtype=torch.float32)
        kv.write_kv(k, v, layer_idx=0, label=label)
    kv.commit(step, ctx)


# ---------------------------------------------------------------------------
# 1. Contract, storage dispatch, and startup capacity hooks
# ---------------------------------------------------------------------------


def test_build_specs_accepts_episode_lifetime_across_all_persistent_storages():
    paged = _paged_kv("p")
    paged_walk = _paged_kv(
        "pw",
        reads={
            "pre": Read(mask=Causal(), op=Attend(q_heads=2), walks={"prefill"}),
            "dec": Read(mask=Bidirectional(), op=Attend(q_heads=2), walks={"decode"}),
        },
    )
    paged_cross = _paged_kv(
        "pc",
        reads={"cross": Read(mask=Bidirectional(), op=Attend(q_heads=2), queries="p")},
    )
    paged_pos = _paged_kv("p_pos", positions=Positions(rope=None))
    ring = Array(
        "ring",
        nodes={"llm"},
        layers={0},
        entry=_kv_entry(),
        max_len=512,
        reads={"ring_read": Read(mask=FrameWindow(128, frames=4), op=Attend(q_heads=2), packed=False)},
        lifetime="episode",
    )
    rolling = Array(
        "roll",
        nodes={"llm"},
        layers={0},
        entry=_kv_entry(),
        max_len=32,
        reads={"roll_read": Read(mask=Bidirectional(kept_window=True), op=Attend(q_heads=2))},
        lifetime="episode",
    )
    val = Array(
        "val", nodes={"llm"}, layers={0}, entry=Entry({"s": TensorType((4,), dtype=torch.float32)}),
        max_len=1, rewritable=True, lifetime="episode",
    )
    val_host = Array(
        "vh",
        nodes={"llm"},
        layers={0},
        entry=Entry({"s": TensorType((4,), dtype=torch.float32)}, host={"flag": bool}),
        max_len=1,
        rewritable=True,
        lifetime="episode",
    )
    host_only = Array(
        "ho", nodes={"llm"}, layers={0}, entry=Entry({}, host={"flag": bool}),
        max_len=1, rewritable=True, lifetime="episode",
    )
    sized = Array(
        "sz",
        nodes={"llm"},
        layers={0},
        entry=Entry({"v": TensorType((Size("n"),), dtype=torch.float32)}),
        max_len=1,
        rewritable=True,
        largest={"n": 16},
        lifetime="episode",
    )
    bounded = Array(
        "bnd",
        nodes={"llm"},
        layers={0},
        entry=_kv_entry(head_dim=16, dtype=torch.float32),
        max_len=None,
        reads={"r": Read(mask=Window(8), op=Attend(q_heads=2), packed=False)},
        positions=Positions(rope=None),
        lifetime="episode",
    )

    assert storage_of(paged) is PAGED
    assert storage_of(paged_walk) is PAGED_BY_WALK
    assert storage_of(paged_cross) is PAGED_CROSS
    assert storage_of(ring) is RING
    assert storage_of(rolling) is ROLLING
    assert storage_of(val) is VALUE
    assert storage_of(val_host) is VALUE_WITH_HOST
    assert storage_of(host_only) is HOST_VALUE
    assert storage_of(sized) is SIZED_VALUE
    assert storage_of(bounded) is BOUNDED

    specs = build_specs([
        paged, paged_walk, paged_cross, paged_pos, ring, rolling, val, val_host, host_only, sized, bounded,
    ])
    by_key = {spec.resource_key: spec for spec in specs}
    for key in ("p", "pw", "pc", "p_pos", position_key(paged_pos), "ring", "roll", "val", "vh", "sz", "bnd"):
        assert by_key[key].episode_lifetime is True
    assert "ho" not in by_key  # HOST_VALUE emits no spec


def test_within_step_refuses_episode_lifetime_and_unbounded_keeps_dense_refusals():
    with pytest.raises(ValueError, match="max_len=0 keeps nothing between steps"):
        Array(
            "scratch",
            nodes={"llm"},
            layers={0},
            entry=_kv_entry(),
            max_len=0,
            reads={"r": Read(mask=Causal(), op=Attend(q_heads=2))},
            lifetime="episode",
        )
    # WITHIN_STEP.resolve also refuses as belt-and-suspenders if handed a fabricated episode array
    arr = Array(
        "scratch",
        nodes={"llm"},
        layers={0},
        entry=_kv_entry(),
        max_len=0,
        reads={"r": Read(mask=Causal(), op=Attend(q_heads=2))},
    )
    object.__setattr__(arr, "lifetime", "episode")
    with pytest.raises(ValueError, match="max_len=0 keeps nothing between steps"):
        WITHIN_STEP.resolve([arr], [arr], StateSetup())

    # Invalid KVEntry unbounded episode declarations remain refused
    with pytest.raises(ValueError, match="unpacked Attend read"):
        Array(
            "bad_bnd",
            nodes={"llm"},
            layers={0},
            entry=_kv_entry(),
            max_len=None,
            reads={"r": Read(mask=Window(8), op=None)},
            lifetime="episode",
        )
    with pytest.raises(ValueError, match="append-only tensor log"):
        Array(
            "bad_bnd_pos",
            nodes={"llm"},
            layers={0},
            entry=_kv_entry(),
            max_len=None,
            reads={"r": Read(mask=Window(8), op=Attend(q_heads=2), packed=False)},
            positions=object(),
            lifetime="episode",
        )


def test_usable_slots_and_capacity_problem_match_across_configs():
    paged_cfg = PagedKVConfig(
        num_layers=1, num_qo_heads=2, num_kv_heads=2, head_dim=8,
        max_seq_len=16, max_num_pages=9, page_size=8,
    )
    paged_spec = KVSpec(resource_key="p", nodes={"llm"}, config=paged_cfg, episode_lifetime=True)
    # 9 units -> 1 sink + 8 request pages; each 16-tok request uses 2 pages -> 4 usable slots
    assert paged_cfg.usable_slots == 4
    assert paged_spec.capacity_problem(4, None) is None
    assert paged_spec.capacity_problem(5, None) is not None

    ring_cfg = RingKVConfig(
        num_layers=1, num_qo_heads=2, num_kv_heads=2, head_dim=8, tokens_per_frame=4,
        layers=(RingKVLayerConfig(ring_frames=4, ring_buckets=4, pinned_dilation=1),), num_sessions=3,
    )
    assert ring_cfg.usable_slots == 3
    ring_spec = KVSpec(resource_key="ring", nodes={"llm"}, config=ring_cfg, episode_lifetime=True)
    assert ring_spec.capacity_problem(3, None) is None
    assert ring_spec.capacity_problem(4, None) is not None

    roll_cfg = RollingKVConfig(num_layers=1, num_heads=2, head_dim=8, pool_tokens=16)
    roll_spec = RollingKVSpec(resource_key="roll", nodes={"llm"}, config=roll_cfg, episode_lifetime=True)
    assert roll_cfg.usable_slots == 16
    assert roll_spec.capacity_problem(16, None) is None
    assert roll_spec.capacity_problem(17, None) is not None
    assert roll_spec.capacity_problem(2, RollingKVReqConfig(tokens=8)) is None
    assert roll_spec.capacity_problem(3, RollingKVReqConfig(tokens=8)) is not None

    bnd_cfg = BoundedKVConfig(
        num_layers=1, num_heads=2, head_dim=8, rows_per_request=1, max_source_len=0,
        retention=SinkWindow(0, 7), max_slots=5,
    )
    bnd_spec = BoundedKVSpec(resource_key="bnd", nodes={"llm"}, config=bnd_cfg, episode_lifetime=True)
    assert bnd_cfg.usable_slots == 5
    assert bnd_spec.capacity_problem(5, None) is None
    assert bnd_spec.capacity_problem(6, None) is not None

    sz_cfg = SizedValueConfig(
        num_layers=1, fields={"v": SizedField((("n", 1),), torch.float32)}, largest={"n": 4}, max_bytes=48,
    )
    sz_spec = SizedValueSpec(resource_key="sz", nodes={"llm"}, config=sz_cfg, episode_lifetime=True)
    # 4 * fp32 (4B) = 16 bytes at largest; budget 48 -> 3 usable slots
    assert sz_cfg.usable_slots == 3
    assert sz_spec.capacity_problem(3, None) is None
    assert sz_spec.capacity_problem(4, None) is not None


def test_engine_refuses_rebuild_policy_on_episode_kv_and_recurrent_pools():
    kv = _paged_kv_manager(episode_lifetime=True)
    assert kv.supports_drop is False
    assert "episode" in kv.check_drop("r", 1)
    with pytest.raises(RuntimeError, match="episode"):
        kv.drop("r", 1)

    engine = make_engine(_submodules={"llm": make_mgmt(resources={"kv": kv})})
    model = make_model(
        rebuild_policies=lambda: {
            "llm": RebuildPolicy(walks=frozenset({"prefill"}), ids_input="ids", sample_output="out", loop="decode"),
        },
        get_combined_graph_walks=lambda: {},
    )
    with pytest.raises(ValueError, match="can't drop"):
        engine._check_rebuild_policies(model)

    pool_cfg = RecurrentStateConfig(
        num_layers=1, blocks={"s": RecurrentBlockConfig(shape=(4,), dtype=torch.float32)}, max_slots=4,
    )
    pool = RecurrentStatePool(device=CPU, config=pool_cfg, name="val", episode_lifetime=True)
    assert pool.supports_drop is False
    assert "episode" in pool.check_drop("r", 1)
    with pytest.raises(RuntimeError, match="episode"):
        pool.drop("r", 1)
    engine2 = make_engine(_submodules={"llm": make_mgmt(resources={"val": pool})})
    with pytest.raises(ValueError, match="can't drop"):
        engine2._check_rebuild_policies(model)


# ---------------------------------------------------------------------------
# 2. KVManager (PAGED / PAGED_BY_WALK / PAGED_CROSS) + Positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix_cache", [False, True])
def test_paged_kv_episode_lifecycle_conservation_and_delta_allotment(prefix_cache):
    kv = _paged_kv_manager(max_num_pages=6, page_size=PAGE_SIZE, prefix_cache=prefix_cache, episode_lifetime=True)
    tensor_ptr = kv.kv_cache.tensor.data_ptr()
    # 6 pages total: 1 sink + 5 request pages
    assert kv._arena.num_free == 5

    # Cold call 1: 10 tokens -> 2 pages (1 full page indexed if prefix_cache=True + 1 partial tail page)
    kv.ingest_request(1, _kv_req(list(range(10)), keyed=prefix_cache))
    kv.bind_episode(1, 0, "ep-a", cold=True, reset=False)
    assert kv.ask(1, "llm", "prefill") is ASKS
    assert kv.reserve(1, "llm", "prefill", None) is None
    _kv_step(kv, 1, 10)
    kv.assert_pages_conserved()
    assert kv.held(1) == 10 and kv.logical_length(1, "main") == 10
    assert kv._arena.num_free == 3

    # Remove request 1 -> pages stay held under ("episode", 0), allotment unused cleared
    kv.remove_request(1)
    kv.assert_pages_conserved()
    assert kv._arena.num_free == 3
    owner = ("episode", 0)
    assert kv.reclaimable(owner) == 0 and kv.can_offload(owner) is False
    if prefix_cache:
        full_page = kv._streams[owner]["main"].page_indices[0]
        assert kv._arena.owners(full_page) == 2
        assert full_page not in kv._index.evictable()

    # Warm resume with recycled handle rid=1! 4 more tokens fit in partial tail page (10+4=14 <= 16 -> 0 extra pages!)
    kv.ingest_request(1, _kv_req(list(range(4)), keyed=prefix_cache))
    kv.bind_episode(1, 0, "ep-a", cold=False, reset=False)
    assert kv._streams[1]["main"].chain is None  # warm resume disabled offset-0 chain
    assert kv.ask(1, "llm", "prefill") is ASKS
    assert kv._asks[1] == 0
    assert kv.reserve(1, "llm", "prefill", None) is None
    assert kv._unused == 0
    _kv_step(kv, 1, 4)
    kv.assert_pages_conserved()
    assert kv.held(1) == 14 and kv._arena.num_free == 3

    # Remove 1, then warm call 3 adds 4 tokens (14+4=18 -> needs 1 extra page)
    kv.remove_request(1)
    kv.ingest_request(2, _kv_req(list(range(4)), keyed=prefix_cache))
    kv.bind_episode(2, 0, "ep-a", cold=False, reset=False)
    assert kv.ask(2, "llm", "prefill") is ASKS
    assert kv._asks[2] == 1
    assert kv.reserve(2, "llm", "prefill", None) is None
    assert kv._unused == 1
    # Unreserve restores delta ask without freeing existing pages
    kv.unreserve(2)
    kv.assert_pages_conserved()
    assert kv._unused == 0 and kv._asks[2] == 1 and kv.held(2) == 14
    assert kv.reserve(2, "llm", "prefill", None) is None
    _kv_step(kv, 2, 4)
    kv.assert_pages_conserved()
    assert kv.held(2) == 18 and kv._arena.num_free == 2
    kv.remove_request(2)

    # Concurrent temporary request borrows remaining 2 free pages and frees on remove_request
    kv.ingest_request(3, _kv_req(list(range(12))))
    kv.bind_episode(3, 1, None, cold=True, reset=False)
    assert kv.ask(3, "llm", "prefill") is ASKS
    assert kv.reserve(3, "llm", "prefill", None) is None
    _kv_step(kv, 3, 12)
    kv.assert_pages_conserved()
    assert kv._arena.num_free == 0
    kv.remove_request(3)
    kv.assert_pages_conserved()
    assert kv._arena.num_free == 2

    # Warm resume with reset=True frees old pages and starts at 0
    kv.ingest_request(4, _kv_req(list(range(6))))
    kv.bind_episode(4, 0, "ep-a", cold=False, reset=True)
    kv.assert_pages_conserved()
    assert kv.held(4) == 0
    assert kv.ask(4, "llm", "prefill") is ASKS
    assert kv.reserve(4, "llm", "prefill", None) is None
    _kv_step(kv, 4, 6)
    kv.assert_pages_conserved()
    assert kv.held(4) == 6
    kv.remove_request(4)

    # End episode frees remaining pages (and leaves indexed pages evictable if prefix_cache=True)
    kv.end_episode(0)
    kv.end_episode(0)  # idempotent
    kv.assert_pages_conserved()
    if prefix_cache:
        # At least the reset full page is still cached in PrefixIndex (evictable)
        assert kv._arena.num_free + len(kv._index.evictable()) == 5
    else:
        assert kv._arena.num_free == 5
    assert kv.kv_cache.tensor.data_ptr() == tensor_ptr


def test_paged_kv_episode_oversize_ask_refuses_and_rollback_admit_preserves_warm_pages():
    kv = _paged_kv_manager(max_num_pages=4, page_size=PAGE_SIZE, prefix_cache=False, episode_lifetime=True)
    # 3 request pages available (24 tokens)
    kv.ingest_request(1, _kv_req(list(range(16))))
    kv.bind_episode(1, 0, "ep-big", cold=True, reset=False)
    assert kv.ask(1, "llm", "prefill") is ASKS
    assert kv.reserve(1, "llm", "prefill", None) is None
    _kv_step(kv, 1, 16)
    kv.remove_request(1)

    # Warm call asking for 16 more tokens (2 + 2 = 4 > 3 request_pages) fails cleanly in ask()
    kv.ingest_request(2, _kv_req(list(range(16))))
    kv.bind_episode(2, 0, "ep-big", cold=False, reset=False)
    outcome = kv.ask(2, "llm", "prefill")
    assert outcome is not None and not outcome.ok
    kv.remove_request(2)

    # Warm rollback_admit frees newly appended tail page while preserving committed 2 pages
    kv.ingest_request(3, _kv_req(list(range(4))))
    kv.bind_episode(3, 0, "ep-big", cold=False, reset=False)
    step = KVStep(segments=(Segment(3, "main", 4),))
    ctx = StepContext(request_ids=[3], graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    assert kv._arena.num_free == 0
    kv.rollback_admit(step, ctx)
    kv.assert_pages_conserved()
    assert kv._arena.num_free == 1 and kv.held(3) == 16
    kv.remove_request(3)
    kv.end_episode(0)
    assert kv._arena.num_free == 3


def test_rope_and_longrope_episode_positions_persist_and_reset():
    pos_cfg = PositionConfig(kv_cache="kv", rotary_dim=8)
    rope = RopeManager(pos_cfg, CPU, max_seq_len=64, head_dim=8, episode_lifetime=True)
    assert rope.reports_episode_held is False and rope.supports_drop is False
    assert "episode" in rope.check_drop(1, 4)

    rope.ingest_request(1)
    rope.bind_episode(1, 0, "ep", cold=True, reset=False)
    rope.commit(PositionStep(segments=(Segment(1, "main", 5),)), StepContext([1], "prefill", 0, False))
    assert rope.position(1, "main") == 5
    rope.remove_request(1)

    rope.ingest_request(2)
    rope.bind_episode(2, 0, "ep", cold=False, reset=False)
    assert rope.position(2, "main") == 5
    rope.commit(PositionStep(segments=(Segment(2, "main", 3),)), StepContext([2], "decode", 0, False))
    assert rope.position(2, "main") == 8
    rope.remove_request(2)

    rope.ingest_request(3)
    rope.bind_episode(3, 0, "ep", cold=False, reset=True)
    assert rope.position(3, "main") == 0
    rope.remove_request(3)
    rope.end_episode(0)
    rope.end_episode(0)  # idempotent

    # LongRoPE preserves _long_rids membership across warm calls
    long_cfg = PositionConfig(
        kv_cache="kv",
        rotary_dim=8,
        longrope=LongRoPE(
            short_factor=(1.0, 1.0, 1.0, 1.0),
            long_factor=(2.0, 2.0, 2.0, 2.0),
            original_max_position_embeddings=16,
            max_position_embeddings=64,
        ),
    )
    lrope = LongRopeManager(long_cfg, CPU, max_seq_len=64, head_dim=8, episode_lifetime=True)
    lrope.ingest_request(1, PositionReqConfig(max_position=32))
    lrope.bind_episode(1, 0, "ep-long", cold=True, reset=False)
    assert 1 in lrope._long_rids
    lrope.commit(PositionStep(segments=(Segment(1, "main", 6),)), StepContext([1], "prefill", 0, False))
    lrope.remove_request(1)
    assert 1 not in lrope._long_rids and ("episode", 0) in lrope._long_rids

    lrope.ingest_request(2, None)
    lrope.bind_episode(2, 0, "ep-long", cold=False, reset=False)
    assert 2 in lrope._long_rids and lrope.position(2, "main") == 6
    lrope.remove_request(2)
    lrope.end_episode(0)
    assert ("episode", 0) not in lrope._long_rids


# ---------------------------------------------------------------------------
# 3. RingKVManager (RING)
# ---------------------------------------------------------------------------


def test_ring_kv_episode_retention_dilation_and_lazy_session():
    frame = 128
    cfg = RingKVConfig(
        num_layers=2,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=8,
        tokens_per_frame=frame,
        num_sessions=2,
        max_frames=6,
        layers=(
            RingKVLayerConfig(ring_frames=2, ring_buckets=2, pinned_dilation=1),
            RingKVLayerConfig(ring_frames=2, ring_buckets=2, pinned_dilation=2),
        ),
    )
    ring = RingKVManager(cfg, "ring", CPU, torch.float32, episode_lifetime=True)
    assert ring.supports_drop is False
    kv_ptrs = [layer.kv.data_ptr() for layer in ring.layers]
    sess_ptr = ring._static_session_idx.data_ptr()
    pos_ptr = ring._static_frame_pos.data_ptr()

    # Lazy claim edge case: rid=1 bound cold and removed before admit()
    ring.ingest_request(1)
    ring.bind_episode(1, 0, "ep-ring", cold=True, reset=False)
    ring.remove_request(1)
    assert len(ring._free_sessions) == 2

    # Warm resume rid=2 claims session on admit(), writes 2 frames
    ring.ingest_request(2)
    ring.bind_episode(2, 0, "ep-ring", cold=False, reset=False)
    for step_idx in range(2):
        step = RingKVStep(segments=(Segment(2, "main", frame),))
        ctx = StepContext([2], "step", 0, False)
        assert ring.admit(step, ctx).ok
        ring.plan(step, ctx)
        k = torch.full((1, 2, frame, 8), float(step_idx + 1), dtype=torch.float32)
        v = torch.full((1, 2, frame, 8), float(step_idx + 10), dtype=torch.float32)
        ring.write(k, v, 0, keep=True)
        ring.write(k, v, 1, keep=True)
        ring.commit(step, ctx)
    sess = ring.session_of(2)
    assert sess is not None and ring.position(2, "main") == 2 * frame and ring.held(2) == 2 * frame
    ring.remove_request(2)
    assert len(ring._free_sessions) == 1  # session still owned by episode

    # Temporary idless request borrows slot 1, writes 1 frame, frees on remove_request
    ring.ingest_request(3)
    ring.bind_episode(3, 1, None, cold=True, reset=False)
    step3 = RingKVStep(segments=(Segment(3, "main", frame),))
    ctx3 = StepContext([3], "step", 0, False)
    assert ring.admit(step3, ctx3).ok
    ring.plan(step3, ctx3)
    ring.write(torch.ones(1, 2, frame, 8), torch.ones(1, 2, frame, 8), 0, keep=True)
    ring.commit(step3, ctx3)
    assert len(ring._free_sessions) == 0
    ring.remove_request(3)
    assert len(ring._free_sessions) == 1

    # Warm resume rid=4 writes 3 more frames (5 total committed; layer0 keeps {3,4}, layer1 keeps {2,4} -> 3 distinct)
    ring.ingest_request(4)
    ring.bind_episode(4, 0, "ep-ring", cold=False, reset=False)
    assert ring.session_of(4) == sess
    for step_idx in range(2, 5):
        step = RingKVStep(segments=(Segment(4, "main", frame),))
        ctx = StepContext([4], "step", 0, False)
        assert ring.admit(step, ctx).ok
        ring.plan(step, ctx)
        k = torch.full((1, 2, frame, 8), float(step_idx + 1), dtype=torch.float32)
        v = torch.full((1, 2, frame, 8), float(step_idx + 10), dtype=torch.float32)
        ring.write(k, v, 0, keep=True)
        ring.write(k, v, 1, keep=True)
        ring.commit(step, ctx)
    assert ring.logical_length(4, "main") == 5 * frame
    assert ring.held(4) == 3 * frame  # 3 distinct frames * 128 tokens/frame
    # Commit 1 more frame -> hits max_frames=6 on next admit
    step = RingKVStep(segments=(Segment(4, "main", frame),))
    ctx = StepContext([4], "step", 0, False)
    assert ring.admit(step, ctx).ok
    ring.plan(step, ctx)
    ring.commit(step, ctx)
    assert not ring.admit(step, ctx).ok
    ring.remove_request(4)

    # Reset clears frames and clock while keeping session
    ring.ingest_request(5)
    ring.bind_episode(5, 0, "ep-ring", cold=False, reset=True)
    assert ring.session_of(5) == sess and ring.held(5) == 0 and ring.position(5, "main") == 0
    span_lo, span_hi = ring.layers[0].session_span(sess)
    assert not ring.layers[0].kv[:, :, :, span_lo:span_hi].any()
    ring.remove_request(5)
    ring.end_episode(0)
    ring.end_episode(0)
    assert len(ring._free_sessions) == 2
    assert [layer.kv.data_ptr() for layer in ring.layers] == kv_ptrs
    assert ring._static_session_idx.data_ptr() == sess_ptr and ring._static_frame_pos.data_ptr() == pos_ptr


# ---------------------------------------------------------------------------
# 4. RollingKVManager (ROLLING)
# ---------------------------------------------------------------------------


def test_rolling_kv_episode_compaction_rearrange_and_reset():
    cfg = RollingKVConfig(num_layers=1, num_heads=2, head_dim=8, pool_tokens=16, dtype=torch.float32)
    store = RollingKVManager(cfg, CPU, name="roll", episode_lifetime=True)
    assert store.supports_drop is False

    # Cold episode rid=1 reserves 8 tokens (sink=2, window=6), writes 6 then 4 (compacting past 8)
    store.ingest_request(1, RollingKVReqConfig(tokens=8, sink=2))
    store.bind_episode(1, 0, "ep-roll", cold=True, reset=False)
    assert store.ask(1, "llm", "step") is ASKS
    assert store.reserve(1, "llm", "step", None) is None

    def run_step(rid: int, span: int, fill_base: float):
        declared = store.declared(rid)
        rpos = RollingPosition(store.written(rid), declared.sink, declared.tokens - declared.sink)
        step = RollingKVStep(segments=(Segment(rid, "main", span),), positions={rid: rpos})
        ctx = StepContext([rid], "step", 0, False)
        assert store.admit(step, ctx).ok
        store.plan(step, ctx)
        k = torch.arange(fill_base, fill_base + span, dtype=torch.float32)[:, None, None].expand(span, 2, 8).clone()
        v = k + 100.0
        store.write(0, k[None], v[None])
        store.commit(step, ctx)

    run_step(1, 6, 1.0)
    assert store.held(1) == 6 and store.logical_length(1) == 6
    store.remove_request(1)

    # Temporary request rid=2 borrows remaining 8 tokens ahead/behind and frees on remove_request
    store.ingest_request(2, RollingKVReqConfig(tokens=8, sink=0))
    store.bind_episode(2, 1, None, cold=True, reset=False)
    assert store.reserve(2, "llm", "step", None) is None
    run_step(2, 4, 50.0)
    store.remove_request(2)
    assert store.num_free_tokens() == 8

    # Warm resume rid=3 with overrides=None succeeds, preserves sink/window & triggers compaction (6+4=10 > 8)
    store.ingest_request(3, None)
    store.bind_episode(3, 0, "ep-roll", cold=False, reset=False)
    assert store.ask(3, "llm", "step") is None
    run_step(3, 4, 7.0)
    k_win, _ = store.window(0)
    # Kept tokens: sink [1, 2] + latest window [5, 6, 7, 8, 9, 10]
    assert k_win[0, :, 0, 0].tolist() == [1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert store.held(3) == 8 and store.logical_length(3) == 10
    store.remove_request(3)

    # Mismatched RollingKVReqConfig on warm resume without reset raises ValueError without losing owner
    store.ingest_request(4, RollingKVReqConfig(tokens=10, sink=2))
    with pytest.raises(ValueError, match="without reset"):
        store.bind_episode(4, 0, "ep-roll", cold=False, reset=False)
    store.remove_request(4)
    assert store.held(("episode", 0)) == 8

    # Reset releases old region and lets new RollingKVReqConfig reserve/admit
    store.ingest_request(5, RollingKVReqConfig(tokens=10, sink=1))
    store.bind_episode(5, 0, "ep-roll", cold=False, reset=True)
    assert store.held(5) == 0 and store.num_free_tokens() == 16
    assert store.reserve(5, "llm", "step", None) is None
    run_step(5, 3, 20.0)
    assert store.held(5) == 3
    store.remove_request(5)
    store.end_episode(0)
    assert store.num_free_tokens() == 16


# ---------------------------------------------------------------------------
# 5. RecurrentStatePool (VALUE & VALUE_WITH_HOST) & HostFields (HOST_VALUE)
# ---------------------------------------------------------------------------


def test_value_recurrent_pool_episode_multilabel_partial_open_and_drop_refusal():
    cfg = RecurrentStateConfig(
        num_layers=1,
        blocks={"s": RecurrentBlockConfig(shape=(4,), dtype=torch.float32)},
        max_slots=4,  # 1 sink + 3 usable slots
        labels=("cond", "blank"),
    )
    pool = RecurrentStatePool(device=CPU, config=cfg, name="val", episode_lifetime=True)
    block_ptr = pool._blocks["s"].data_ptr()
    assert pool.supports_drop is False

    # Episode 0 cold call touches ONLY "cond" (1 slot used, 2 free)
    pool.ingest_request(1)
    pool.bind_episode(1, 0, "ep-val", cold=True, reset=False)
    assert pool.ask(1, "llm", "step") is ASKS
    assert pool.reserve(1, "llm", "step", None) is None
    step1 = RecurrentStep(segments=(Segment(1, "cond", 1),))
    ctx1 = StepContext([1], "step", 0, False)
    assert pool.admit(step1, ctx1).ok
    pool.plan(step1, ctx1)
    pool.scatter_("s", torch.tensor([[1.0, 2.0, 3.0, 4.0]]), 0, label="cond")
    pool.commit(step1, ctx1)
    assert pool.held(1) == 1 and pool.logical_length(1, "cond") == 1 and pool.logical_length(1, "blank") == 0
    pool.remove_request(1)
    assert pool.num_free_slots == 2 and pool._kept == 0

    # Warm resume with recycled handle rid=1 opens missing "blank" label -> ask() must ask for 1 slot!
    pool.ingest_request(1)
    pool.bind_episode(1, 0, "ep-val", cold=False, reset=False)
    assert pool.ask(1, "llm", "step") is ASKS
    assert pool.reserve(1, "llm", "step", None) is None
    assert pool._reserved[1] == 1
    step2 = RecurrentStep(segments=(Segment(1, "cond", 1), Segment(1, "blank", 1)))
    ctx2 = StepContext([1], "step", 0, False)
    assert pool.admit(step2, ctx2).ok
    pool.plan(step2, ctx2)
    assert torch.equal(pool.gather("s", 0, label="cond"), torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    pool.scatter_("s", torch.tensor([[5.0, 6.0, 7.0, 8.0]]), 0, label="blank")
    pool.commit(step2, ctx2)
    assert pool.held(1) == 2
    pool.remove_request(1)
    assert pool.num_free_slots == 1

    # Reset zeroes slots in place without freeing them
    pool.ingest_request(2)
    pool.bind_episode(2, 0, "ep-val", cold=False, reset=True)
    assert pool.held(2) == 0 and pool.num_free_slots == 1
    assert pool.ask(2, "llm", "step") is None  # both slots already held
    pool.remove_request(2)

    pool.end_episode(0)
    assert pool.num_free_slots == 3
    assert pool._blocks["s"].data_ptr() == block_ptr


def test_host_value_episode_persistence_clear_and_engine_lifecycle():
    arr = Array(
        "flags", nodes={"talker"}, layers={0}, entry=Entry({}, host={"step_count": int, "tag": str}),
        max_len=1, rewritable=True, lifetime="episode",
    )
    assert storage_of(arr) is HOST_VALUE
    assert build_specs([arr]) == []

    fields = HostFields()
    model = make_model(declare_state=lambda: [arr])
    handle = node_handles(model, "talker", {}, setup=StateSetup(host_fields=fields))["flags"]

    class Runner:
        publisher_nodes = {}

        def ingest_request(self, rid, overrides=None, arrival=None):
            return

        def remove_request(self, rid):
            return

        def publish_snapshot(self, *args, **kwargs):
            return {}

        def publish(self, *args, **kwargs):
            return {}

    engine = make_engine(
        _runner=Runner(),
        _state_setup=StateSetup(host_fields=fields),
        _submodules={"talker": make_mgmt(submodule=SimpleNamespace(cleanup_request=lambda rid: None))},
    )
    engine._episode_host_arrays = dict(fields.episode_arrays)

    with pytest.raises(ValueError, match="conductor-assigned episode_slot"):
        engine.add_request(1, episode_id="ep-host")

    engine.add_request(1, episode_id="ep-host", episode_slot=0)
    step1 = state_steps(
        SubmoduleStep(
            segments=(Segment(1, "main", 1),),
            state={"flags": ArrayStep(host={(1, "main", 0): {"step_count": 7, "tag": "warm"}})},
        ),
        {"flags": handle},
    )
    step1.set_ctx(StepContext([1], "decode", 0, False))
    step1.begin_calls()
    step1.commit_calls()
    step1.end_calls()
    batch1 = ExecutingBatch("talker", {}, step1.ctx)
    engine.finalize_batch(batch1)
    assert batch1.episode_held[1] == {"flags": 1}
    assert host(handle, 1, "main", 0) == {"step_count": 7, "tag": "warm"}
    engine.remove_request(1)

    # Warm resume reads prior host fields
    engine.add_request(2, episode_id="ep-host", episode_slot=0)
    assert host(handle, 2, "main", 0) == {"step_count": 7, "tag": "warm"}
    engine.remove_request(2)

    # episode_clear resets host array
    engine.add_request(3, episode_id="ep-host", episode_slot=0, episode_clear=["flags"])
    assert host(handle, 3, "main", 0) == {"step_count": 0, "tag": ""}
    batch3 = ExecutingBatch("talker", {}, StepContext([3], "decode", 0, False))
    engine.finalize_batch(batch3)
    assert batch3.episode_held[3] == {"flags": 0}
    engine.remove_request(3)
    assert engine.end_episode("ep-host", 0) is True


# ---------------------------------------------------------------------------
# 6. SizedValueStore (SIZED_VALUE)
# ---------------------------------------------------------------------------


def test_sized_value_episode_budget_warm_resume_mismatch_and_reset():
    cfg = SizedValueConfig(
        num_layers=1,
        fields={"v": SizedField((("n", 1),), torch.float32)},
        largest={"n": 8},
        max_bytes=32,  # exactly 8 floats -> fits one 8-float episode at a time
    )
    store = SizedValueStore(cfg, CPU, name="sz", episode_lifetime=True)
    assert store.supports_drop is False

    store.ingest_request(1, SizedValueReqConfig(sizes={"main": {"n": 8}}))
    store.bind_episode(1, 0, "ep-sz", cold=True, reset=False)
    assert store.ask(1, "dit", "rollout") is ASKS
    assert store.reserve(1, "dit", "rollout", None) is None
    assert store._used == 32

    t = torch.arange(8, dtype=torch.float32)
    taken = store.take(1, "main", 0)
    assert taken == {"v": None}
    store.put(1, "main", 0, {"v": t})
    assert store.held(1) == 1 and store.logical_length(1, "main") == 1
    store.remove_request(1)
    assert store._used == 32  # bytes stay reserved across calls

    # Competing episode 1 cannot reserve until ep-sz ends
    store.ingest_request(2, SizedValueReqConfig(sizes={"main": {"n": 4}}))
    store.bind_episode(2, 1, "ep-other", cold=True, reset=False)
    assert store.reserve(2, "dit", "rollout", None) == 0
    store.remove_request(2)
    store.end_episode(1)

    # Warm resume of ep-sz with overrides=None preserves tensors without double-charging _used
    store.ingest_request(3, None)
    store.bind_episode(3, 0, "ep-sz", cold=False, reset=False)
    assert store.ask(3, "dit", "rollout") is None
    assert store.reserve(3, "dit", "rollout", None) is None
    assert store._used == 32
    got = store.take(3, "main", 0)
    assert torch.equal(got["v"], t)
    store.put(3, "main", 0, {"v": got["v"] + 1})
    store.remove_request(3)

    # Size mismatch without reset raises without losing owner's reservation/tensors
    store.ingest_request(4, SizedValueReqConfig(sizes={"main": {"n": 4}}))
    with pytest.raises(ValueError, match="without reset"):
        store.bind_episode(4, 0, "ep-sz", cold=False, reset=False)
    store.remove_request(4)
    assert store._used == 32 and store.held(("episode", 0)) == 1

    # Reset releases old bytes and lets new sizes reserve
    store.ingest_request(5, SizedValueReqConfig(sizes={"main": {"n": 4}}))
    store.bind_episode(5, 0, "ep-sz", cold=False, reset=True)
    assert store._used == 0 and store.held(5) == 0
    assert store.reserve(5, "dit", "rollout", None) is None
    assert store._used == 16
    store.remove_request(5)
    store.end_episode(0)
    assert store._used == 0


# ---------------------------------------------------------------------------
# 7. BoundedKVManager (BOUNDED)
# ---------------------------------------------------------------------------


def test_bounded_kv_episode_retention_wrap_reset_and_data_ptr():
    arr = Array(
        "bnd",
        nodes={"llm"},
        layers={0},
        entry=_kv_entry(head_dim=16, dtype=torch.float32),
        max_len=None,
        reads={"enc": Read(mask=Window(4), op=Attend(q_heads=2), packed=False)},
        positions=Positions(rope=None),
        lifetime="episode",
    )
    (spec,) = build_specs([arr])
    assert spec.episode_lifetime is True and spec.config.max_retained == 3
    kv = BoundedKVManager.build(spec, EngineResourceInfo(device=CPU, kv_dtype=torch.float32))
    cache_ptr = kv._cache.data_ptr()
    assert kv.supports_drop is False

    handle = build_handle(arr, {"bnd": kv}, _resolve_bounded(arr))

    def run_step(rid: int, span: int) -> torch.Tensor:
        step = BoundedKVStep(
            segments=(Segment(rid, "main", span),),
            positions={rid: StreamPosition(0, kv.written(rid))},
        )
        ctx = StepContext([rid], "step", 0, False)
        assert kv.admit(step, ctx).ok
        kv.plan(step, ctx)
        q = torch.randn(1, span, 2, 16, dtype=torch.float32)
        k = torch.randn(1, span, 2, 16, dtype=torch.float32)
        v = torch.randn(1, span, 2, 16, dtype=torch.float32)
        handle._attend(0, q, k, v, None)
        kv.commit(step, ctx)
        return q

    kv.ingest_request(1)
    kv.bind_episode(1, 0, "ep-bnd", cold=True, reset=False)
    run_step(1, 2)
    assert kv.written(1) == 2 and kv.position(1, "main") == 2 and kv.held(1) == 2
    kv.remove_request(1)
    assert len(kv._free) == spec.config.max_slots - 1

    # Warm resume wraps window past max_retained=3
    kv.ingest_request(2)
    kv.bind_episode(2, 0, "ep-bnd", cold=False, reset=False)
    assert kv.written(2) == 2
    run_step(2, 3)
    assert kv.written(2) == 5 and kv.logical_length(2, "main") == 5 and kv.held(2) == 3
    kv.remove_request(2)

    # Temporary request borrows another slot and frees immediately
    kv.ingest_request(3)
    kv.bind_episode(3, 1, None, cold=True, reset=False)
    run_step(3, 2)
    assert len(kv._free) == spec.config.max_slots - 2
    kv.remove_request(3)
    assert len(kv._free) == spec.config.max_slots - 1

    # Reset clears written to 0 while keeping slot; end_episode returns slot
    kv.ingest_request(4)
    kv.bind_episode(4, 0, "ep-bnd", cold=False, reset=True)
    assert kv.written(4) == 0 and kv.held(4) == 0
    kv.remove_request(4)
    kv.end_episode(0)
    assert len(kv._free) == spec.config.max_slots
    assert kv._cache.data_ptr() == cache_ptr


def _resolve_bounded(arr: Array):
    from mstar.engine.state.adapters import _resolve

    return _resolve([arr], StateSetup())


# ---------------------------------------------------------------------------
# 8. Multi-storage Engine integration: finalize_batch, episode_clear, rollback
# ---------------------------------------------------------------------------


def test_engine_multi_storage_episode_held_selective_clear_and_rollback():
    kv = _paged_kv_manager(max_num_pages=8, page_size=PAGE_SIZE, prefix_cache=False, episode_lifetime=True)
    rope = RopeManager(
        PositionConfig(kv_cache="kv", rotary_dim=8), CPU, max_seq_len=64, head_dim=8, episode_lifetime=True,
    )
    ring_cfg = RingKVConfig(
        num_layers=1, num_qo_heads=2, num_kv_heads=2, head_dim=8, tokens_per_frame=128, num_sessions=2,
        layers=(RingKVLayerConfig(ring_frames=4, ring_buckets=4, pinned_dilation=1),),
    )
    ring = RingKVManager(ring_cfg, "ring", CPU, torch.float32, episode_lifetime=True)
    roll = RollingKVManager(
        RollingKVConfig(num_layers=1, num_heads=2, head_dim=8, pool_tokens=16, dtype=torch.float32),
        CPU,
        name="roll",
        episode_lifetime=True,
    )
    val = RecurrentStatePool(
        CPU,
        RecurrentStateConfig(
            num_layers=1, blocks={"s": RecurrentBlockConfig(shape=(4,), dtype=torch.float32)}, max_slots=4,
        ),
        name="val",
        episode_lifetime=True,
    )
    bnd = BoundedKVManager(
        BoundedKVConfig(
            num_layers=1, num_heads=2, head_dim=8, rows_per_request=1, max_source_len=0, retention=SinkWindow(0, 7),
        ),
        CPU,
        name="bnd",
        episode_lifetime=True,
    )
    sz = SizedValueStore(
        SizedValueConfig(num_layers=1, fields={"v": SizedField((("n", 1),), torch.float32)}, largest={"n": 4}),
        CPU,
        name="sz",
        episode_lifetime=True,
    )

    host_arr = Array(
        "ho", nodes={"node0"}, layers={0}, entry=Entry({}, host={"c": int}),
        max_len=1, rewritable=True, lifetime="episode",
    )
    fields = HostFields()
    fields.declare(host_arr)

    resources = {"kv": kv, "rope": rope, "ring": ring, "roll": roll, "val": val, "bnd": bnd, "sz": sz}
    runner = StepRunner(resources, node_resources={"node0": list(resources)})

    add_calls = []

    class Submod:
        def __init__(self, fail_on_add: bool = False):
            self.fail = fail_on_add

        def on_request_added(self, rid, info):
            add_calls.append(rid)
            if self.fail:
                raise RuntimeError("boom on add_request")

        def cleanup_request(self, rid):
            return

    submod = Submod()
    engine = make_engine(
        _resources=resources,
        _runner=runner,
        _state_setup=StateSetup(host_fields=fields),
        _submodules={"node0": make_mgmt(submodule=submod, resources=resources)},
        _resource_nodes={k: frozenset({"node0"}) for k in resources},
        _episode_pools={k: r for k, r in resources.items() if k != "rope"},
        _episode_aux={"rope": rope},
        _episode_host_arrays={"ho": host_arr},
    )

    overrides = {
        "kv": _kv_req(list(range(6))),
        "roll": RollingKVReqConfig(tokens=8, sink=1),
        "sz": SizedValueReqConfig(sizes={"main": {"n": 4}}),
    }

    # Partial failure rollback on cold bind cleans up all pools idempotently
    submod.fail = True
    with pytest.raises(RuntimeError, match="boom on add_request"):
        engine.add_request(
            99, overrides=overrides, request_info=SimpleNamespace(arrival=0), episode_id="ep-fail", episode_slot=0,
        )
    assert engine.episode_slot_available(0, "other")
    assert kv._arena.num_free == 7 and len(ring._free_sessions) == 2 and sz._used == 0
    submod.fail = False

    # Normal cold request 1 populates all storages
    engine.add_request(1, overrides=overrides, episode_id="ep-all", episode_slot=0)
    _kv_step(kv, 1, 6)
    rope.commit(PositionStep(segments=(Segment(1, "main", 6),)), StepContext([1], "step", 0, False))

    r_step = RingKVStep(segments=(Segment(1, "main", 128),))
    r_ctx = StepContext([1], "step", 0, False)
    assert ring.admit(r_step, r_ctx).ok
    ring.plan(r_step, r_ctx)
    ring.commit(r_step, r_ctx)

    assert roll.reserve(1, "node0", "step", None) is None
    rl_step = RollingKVStep(segments=(Segment(1, "main", 3),), positions={1: RollingPosition(0, 1, 7)})
    rl_ctx = StepContext([1], "step", 0, False)
    assert roll.admit(rl_step, rl_ctx).ok
    roll.plan(rl_step, rl_ctx)
    roll.commit(rl_step, rl_ctx)

    v_step = RecurrentStep(segments=(Segment(1, "main", 1),))
    v_ctx = StepContext([1], "step", 0, False)
    assert val.admit(v_step, v_ctx).ok
    val.commit(v_step, v_ctx)

    b_step = BoundedKVStep(segments=(Segment(1, "main", 2),), positions={1: StreamPosition(0, 0)})
    b_ctx = StepContext([1], "step", 0, False)
    assert bnd.admit(b_step, b_ctx).ok
    bnd.commit(b_step, b_ctx)

    assert sz.reserve(1, "node0", "step", None) is None
    sz.take(1, "main", 0)
    sz.put(1, "main", 0, {"v": torch.ones(4)})

    fields.commit(host_arr, ArrayStep(host={(1, "main", 0): {"c": 42}}), [1])

    batch = ExecutingBatch("node0", {}, StepContext([1], "step", 0, False))
    engine.finalize_batch(batch)
    # "rope" must NOT appear in episode_held; all 7 episode arrays (including HOST_VALUE "ho") appear!
    assert batch.episode_held[1] == {
        "kv": 6,
        "ring": 128,
        "roll": 3,
        "val": 1,
        "bnd": 2,
        "sz": 1,
        "ho": 1,
    }
    assert "rope" not in batch.episode_held[1]

    assert not engine.end_episode("ep-all", 0)
    engine.remove_request(1)

    # Selective episode_clear=["kv"] clears ONLY kv AND companion rope, leaving ring/roll/val/bnd/sz/ho warm!
    engine.add_request(2, episode_id="ep-all", episode_slot=0, episode_clear=["kv"])
    assert rope.position(2, "main") == 0
    batch2 = ExecutingBatch("node0", {}, StepContext([2], "step", 0, False))
    engine.finalize_batch(batch2)
    assert batch2.episode_held[2] == {
        "kv": 0,
        "ring": 128,
        "roll": 3,
        "val": 1,
        "bnd": 2,
        "sz": 1,
        "ho": 1,
    }
    assert fields.fields(2, "ho", "main", 0) == {"c": 42}
    engine.remove_request(2)

    assert engine.end_episode("ep-all", 0) is True
    assert kv._arena.num_free == 7
    assert len(ring._free_sessions) == 2
    assert roll.num_free_tokens() == 16
    assert val.num_free_slots == 3
    assert len(bnd._free) == 32
    assert sz._used == 0


def test_rolling_kv_rearrange_slides_idle_episode_without_corruption():
    from mstar.engine.resources.kv.rolling.config import RollingPosition

    cfg = RollingKVConfig(num_layers=1, num_heads=1, head_dim=4, pool_tokens=16, dtype=torch.float32)
    store = RollingKVManager(cfg, CPU, name="roll", episode_lifetime=True)

    # Temp request 1 occupies [0, 6); Episode A occupies [6, 12)
    store.ingest_request(1, RollingKVReqConfig(tokens=6, sink=0))
    store.bind_episode(1, 0, None, cold=True, reset=False)
    assert store.reserve(1, "llm", "step", None) is None
    st1 = RollingKVStep(segments=(Segment(1, "main", 4),), positions={1: RollingPosition(0, 0, 6)})
    ctx1 = StepContext([1], "step", 0, False)
    assert store.admit(st1, ctx1).ok
    store.plan(st1, ctx1)
    store.commit(st1, ctx1)

    store.ingest_request(2, RollingKVReqConfig(tokens=6, sink=2))
    store.bind_episode(2, 1, "ep-a", cold=True, reset=False)
    assert store.reserve(2, "llm", "step", None) is None
    st2 = RollingKVStep(segments=(Segment(2, "main", 4),), positions={2: RollingPosition(0, 2, 4)})
    ctx2 = StepContext([2], "step", 0, False)
    assert store.admit(st2, ctx2).ok
    store.plan(st2, ctx2)
    k_ep = torch.tensor([[[[1.0, 0, 0, 0]], [[2.0, 0, 0, 0]], [[3.0, 0, 0, 0]], [[4.0, 0, 0, 0]]]], dtype=torch.float32)
    store.write(0, k_ep, k_ep + 10.0)
    store.commit(st2, ctx2)
    assert store._requests[2].offset == 6
    store.remove_request(2)
    store.remove_request(1)  # frees [0, 6), leaving fragmented [0, 6) + [12, 16)

    # Request 3 wants 10 tokens -> triggers _rearrange(), sliding idle episode A to [0, 6)
    store.ingest_request(3, RollingKVReqConfig(tokens=10, sink=0))
    store.bind_episode(3, 0, None, cold=True, reset=False)
    assert store.reserve(3, "llm", "step", None) is None
    st3 = RollingKVStep(segments=(Segment(3, "main", 2),), positions={3: RollingPosition(0, 0, 10)})
    ctx3 = StepContext([3], "step", 0, False)
    assert store.admit(st3, ctx3).ok
    assert store._requests[("episode", 1)].offset == 0 and store._requests[3].offset == 6
    store.remove_request(3)

    # Warm resume episode A reads its intact values after sliding
    store.ingest_request(4, None)
    store.bind_episode(4, 1, "ep-a", cold=False, reset=False)
    st4 = RollingKVStep(segments=(Segment(4, "main", 2),), positions={4: RollingPosition(4, 2, 4)})
    ctx4 = StepContext([4], "step", 0, False)
    assert store.admit(st4, ctx4).ok
    store.plan(st4, ctx4)
    k_win, _ = store.window(0)
    assert k_win[0, :4, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    store.remove_request(4)
    store.end_episode(1)
    assert store.num_free_tokens() == 16


def test_value_with_host_episode_preserves_tensors_and_host_fields():
    arr = Array(
        "vh",
        nodes={"llm"},
        layers={0},
        entry=Entry({"s": TensorType((4,), dtype=torch.float32)}, host={"step": int}),
        max_len=1,
        rewritable=True,
        lifetime="episode",
    )
    assert storage_of(arr) is VALUE_WITH_HOST
    (spec,) = build_specs([arr])
    pool = RecurrentStatePool(CPU, spec.config, name="vh", episode_lifetime=True)
    fields = HostFields()
    model = make_model(declare_state=lambda: [arr])
    setup = StateSetup(host_fields=fields)
    handle = node_handles(model, "llm", {"vh": pool}, setup=setup)["vh"]

    pool.ingest_request(1)
    fields.open(1)
    pool.bind_episode(1, 0, "ep-vh", cold=True, reset=False)
    fields.bind_episode(1, 0, "ep-vh", cold=True)

    step1 = state_steps(
        SubmoduleStep(
            segments=(Segment(1, "main", 1),),
            state={"vh": ArrayStep(host={(1, "main", 0): {"step": 5}})},
        ),
        {"vh": handle},
    )
    step1.set_ctx(StepContext([1], "decode", 0, False))
    assert pool.admit(step1.steps["vh"], step1.ctx).ok
    pool.plan(step1.steps["vh"], step1.ctx)
    step1.begin_calls()
    handle.write(0, s=torch.tensor([[9.0, 8.0, 7.0, 6.0]]))
    step1.check_calls()
    pool.commit(step1.steps["vh"], step1.ctx)
    step1.commit_calls()
    step1.end_calls()
    assert pool.held(1) == 1 and host(handle, 1, "main", 0) == {"step": 5}

    pool.remove_request(1)
    fields.remove_request(1)

    pool.ingest_request(2)
    fields.open(2)
    pool.bind_episode(2, 0, "ep-vh", cold=False, reset=False)
    fields.bind_episode(2, 0, "ep-vh", cold=False)
    step2 = state_steps(SubmoduleStep(segments=(Segment(2, "main", 1),), state={"vh": ArrayStep()}), {"vh": handle})
    step2.set_ctx(StepContext([2], "decode", 0, False))
    assert pool.admit(step2.steps["vh"], step2.ctx).ok
    pool.plan(step2.steps["vh"], step2.ctx)
    step2.begin_calls()
    assert torch.equal(handle.read(0).fields["s"], torch.tensor([[9.0, 8.0, 7.0, 6.0]]))
    assert host(handle, 2, "main", 0) == {"step": 5}
    step2.end_calls()

    pool.remove_request(2)
    fields.remove_request(2)
    pool.end_episode(0)
    fields.end_episode(0)
    assert pool.num_free_slots == spec.config.usable_slots


def test_engine_load_model_registers_all_episode_storages_and_aux_positions():
    class DummyNode(NodeSubmodule):
        def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
            return inputs

        def forward(self, graph_walk, engine_inputs, **kwargs):
            return engine_inputs

    paged_pos = _paged_kv("p_pos", positions=Positions(rope=None))
    ring = Array(
        "ring",
        nodes={"llm"},
        layers={0},
        entry=_kv_entry(),
        max_len=512,
        reads={"ring_read": Read(mask=FrameWindow(128, frames=4), op=Attend(q_heads=2), packed=False)},
        lifetime="episode",
    )
    rolling = Array(
        "roll",
        nodes={"llm"},
        layers={0},
        entry=_kv_entry(),
        max_len=32,
        reads={"roll_read": Read(mask=Bidirectional(kept_window=True), op=Attend(q_heads=2))},
        lifetime="episode",
    )
    val_host = Array(
        "vh",
        nodes={"llm"},
        layers={0},
        entry=Entry({"s": TensorType((4,), dtype=torch.float32)}, host={"flag": bool}),
        max_len=1,
        rewritable=True,
        lifetime="episode",
    )
    host_only = Array(
        "ho",
        nodes={"llm"},
        layers={0},
        entry=Entry({}, host={"flag": bool}),
        max_len=1,
        rewritable=True,
        lifetime="episode",
    )
    sized = Array(
        "sz",
        nodes={"llm"},
        layers={0},
        entry=Entry({"v": TensorType((Size("n"),), dtype=torch.float32)}),
        max_len=1,
        rewritable=True,
        largest={"n": 16},
        lifetime="episode",
    )
    bounded = Array(
        "bnd",
        nodes={"llm"},
        layers={0},
        entry=_kv_entry(head_dim=16, dtype=torch.float32),
        max_len=None,
        reads={"r": Read(mask=Window(8), op=Attend(q_heads=2), packed=False)},
        positions=Positions(rope=None),
        lifetime="episode",
    )
    arrays = [paged_pos, ring, rolling, val_host, host_only, sized, bounded]
    specs = build_specs(arrays)
    model = make_model(
        declare_state=lambda: arrays,
        get_graph_walk_graphs=lambda: {"step": SimpleNamespace(get_nodes=lambda: {"llm": None})},
        get_combined_graph_walks=lambda: {},
        checkpoint_path=lambda: None,
    )
    joint = SimpleNamespace(
        rank=0,
        world_size=1,
        tp_group=SimpleNamespace(barrier=lambda: None, world_size=1),
        sp_group=SimpleNamespace(world_size=1),
    )
    groups = SimpleNamespace(
        get_joint_group_for_node=lambda name: joint,
        get_instance_world_size_for_node=lambda name: 1,
        get_instance_rank_for_node=lambda name: 0,
        resource_needs_remote_transfer=lambda spec_nodes, relevant: False,
    )
    engine = Engine(None, autocast_dtype=None)
    engine.load_model(
        submodules={"llm": DummyNode()},
        specs=specs,
        parallel_groups=groups,
        device=CPU,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        kv_cache_type=torch.float32,
        model=model,
    )
    pos_key = position_key(paged_pos)
    assert set(engine._episode_pools) == {"p_pos", "ring", "roll", "vh", "sz", "bnd"}
    assert set(engine._episode_aux) == {pos_key}
    assert set(engine._episode_host_arrays) == {"vh", "ho"}

    engine.add_request(1, episode_id="ep-full", episode_slot=0)
    batch = ExecutingBatch("llm", {}, StepContext([1], "step", 0, False))
    engine.finalize_batch(batch)
    assert set(batch.episode_held[1]) == {"p_pos", "ring", "roll", "vh", "ho", "sz", "bnd"}
    assert pos_key not in batch.episode_held[1]
    engine.remove_request(1)
    assert engine.end_episode("ep-full", 0) is True


def test_paged_kv_end_episode_with_debug_asserts_and_released_stream(monkeypatch):
    monkeypatch.setattr(kv_manager_module, "_DEBUG_ASSERTS", True)
    kv = _paged_kv_manager(max_num_pages=6, page_size=PAGE_SIZE, prefix_cache=True, episode_lifetime=True)

    # Full lifecycle under _DEBUG_ASSERTS=True: commit -> remove_request -> end_episode
    kv.ingest_request(1, _kv_req(list(range(10)), keyed=True))
    kv.bind_episode(1, 0, "ep-dbg", cold=True, reset=False)
    assert kv.ask(1, "llm", "prefill") is ASKS
    assert kv.reserve(1, "llm", "prefill", None) is None
    _kv_step(kv, 1, 10)
    kv.remove_request(1)
    kv.end_episode(0)
    kv.assert_pages_conserved()
    assert kv._arena.num_free == 4
    assert len(kv._index.evictable()) == 1

    # Un-admitted episode (no overrides on owner) -> remove_request -> end_episode
    kv.ingest_request(2, _kv_req(list(range(4)), keyed=True))
    kv.bind_episode(2, 1, "ep-dbg-uncommitted", cold=True, reset=False)
    kv.remove_request(2)
    kv.end_episode(1)
    kv.assert_pages_conserved()
    kv._report_underuse(("episode", 99))

    # Warm resume when stream has stored_len == 0 but released > 0 skips offset-0 key seeding
    kv.ingest_request(3, _kv_req(list(range(4)), keyed=True))
    kv.bind_episode(3, 2, "ep-dbg-released", cold=True, reset=False)
    kv.remove_request(3)
    kv._streams[("episode", 2)]["main"].released = PAGE_SIZE
    kv.ingest_request(4, _kv_req(list(range(4)), keyed=True))
    kv.bind_episode(4, 2, "ep-dbg-released", cold=False, reset=False)
    stream4 = kv._streams[4]["main"]
    assert stream4.span_cursor is None and stream4.chain is None
    kv.remove_request(4)
    kv.end_episode(2)
    kv.assert_pages_conserved()


def test_ring_kv_admit_full_pool_contention_with_mixed_int_and_tuple_keys():
    frame = 128
    cfg = RingKVConfig(
        num_layers=1,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=8,
        tokens_per_frame=frame,
        num_sessions=2,
        layers=(RingKVLayerConfig(ring_frames=4, ring_buckets=4, pinned_dilation=1),),
    )
    ring = RingKVManager(cfg, "ring", CPU, torch.float32, episode_lifetime=True)
    rids = RidTable()

    r0 = rids.intern("req-0")
    ring.ingest_request(r0)
    ring.bind_episode(r0, 0, "ep-0", cold=True, reset=False)
    st0 = RingKVStep(segments=(Segment(r0, "main", frame),))
    ctx0 = StepContext([r0], "step", 0, False)
    assert ring.admit(st0, ctx0).ok
    ring.commit(st0, ctx0)
    ring.remove_request(r0)  # leaves ("episode", 0) tuple in _sessions and _committed_frames

    r1 = rids.intern("req-1")
    ring.ingest_request(r1)
    ring.bind_episode(r1, 1, "ep-1", cold=True, reset=False)
    st1 = RingKVStep(segments=(Segment(r1, "main", frame),))
    ctx1 = StepContext([r1], "step", 0, False)
    assert ring.admit(st1, ctx1).ok
    ring.commit(st1, ctx1)  # leaves int r1 active in _sessions and _committed_frames

    r2 = rids.intern("req-2")
    ring.ingest_request(r2)
    ring.bind_episode(r2, 2, "ep-2", cold=True, reset=False)
    st2 = RingKVStep(segments=(Segment(r2, "main", frame),))
    ctx2 = StepContext([r2], "step", 0, False)
    outcome = ring.admit(st2, ctx2)
    assert not outcome.ok and not outcome.ready
    assert isinstance(outcome.reason, AllocationFailed)

    with pytest.raises(RuntimeError, match="still claimed"):
        ring.post_warmup_validate()

    # Check _committed_frames mixed-key formatting in post_warmup_validate too
    saved_sessions = dict(ring._sessions)
    ring._sessions.clear()
    ring._free_sessions.update({0, 1})
    try:
        with pytest.raises(RuntimeError, match="counted committed frames"):
            ring.post_warmup_validate()
    finally:
        ring._sessions.update(saved_sessions)
        ring._free_sessions.difference_update(saved_sessions.values())

    ring.remove_request(r2)
    ring.remove_request(r1)
    ring.end_episode(0)
    ring.end_episode(1)
    ring.end_episode(2)
    ring.post_warmup_validate()


def test_bounded_kv_post_warmup_validate_formats_mixed_int_and_tuple_keys():
    cfg = BoundedKVConfig(
        num_layers=1,
        num_heads=2,
        head_dim=16,
        rows_per_request=1,
        max_source_len=0,
        max_slots=3,
        retention=SinkWindow(0, 4),
        dtype=torch.float32,
    )
    bnd = BoundedKVManager(cfg, CPU, name="bnd", episode_lifetime=True)

    bnd.ingest_request(1)
    bnd.bind_episode(1, 0, "ep-0", cold=True, reset=False)
    st1 = BoundedKVStep(segments=(Segment(1, "main", 2),), positions={1: StreamPosition(0, 0)})
    ctx1 = StepContext([1], "step", 0, False)
    assert bnd.admit(st1, ctx1).ok
    bnd.commit(st1, ctx1)
    bnd.remove_request(1)  # ("episode", 0) in _requests

    bnd.ingest_request(2)
    bnd.bind_episode(2, 1, "ep-1", cold=True, reset=False)
    st2 = BoundedKVStep(segments=(Segment(2, "main", 2),), positions={2: StreamPosition(0, 0)})
    ctx2 = StepContext([2], "step", 0, False)
    assert bnd.admit(st2, ctx2).ok
    bnd.commit(st2, ctx2)  # int 2 in _requests

    with pytest.raises(RuntimeError, match="bounded KV slots still held after capture"):
        bnd.post_warmup_validate()

    bnd.remove_request(2)
    bnd.end_episode(0)
    bnd.end_episode(1)
    bnd.post_warmup_validate()


def test_rolling_kv_reserve_idempotent_on_warm_and_reserved_requests():
    cfg = RollingKVConfig(num_layers=1, num_heads=1, head_dim=4, pool_tokens=16, dtype=torch.float32)
    store = RollingKVManager(cfg, CPU, name="roll", episode_lifetime=True)

    # Cold request: reserve is idempotent while already reserved
    store.ingest_request(1, RollingKVReqConfig(tokens=8, sink=0))
    store.bind_episode(1, 0, "ep-roll", cold=True, reset=False)
    assert store.reserve(1, "llm", "step", None) is None
    assert store._kept == 8 and store._reserved == {1: 8}
    assert store.reserve(1, "llm", "step", None) is None
    assert store._kept == 8 and store._reserved == {1: 8}

    st1 = RollingKVStep(segments=(Segment(1, "main", 4),), positions={1: RollingPosition(0, 0, 8)})
    ctx1 = StepContext([1], "step", 0, False)
    assert store.admit(st1, ctx1).ok
    store.commit(st1, ctx1)
    store.remove_request(1)

    # Warm episode (half-full pool: room == 8): direct reserve() does not leak _kept/_reserved
    store.ingest_request(2, RollingKVReqConfig(tokens=8, sink=0))
    store.bind_episode(2, 0, "ep-roll", cold=False, reset=False)
    assert store.reserve(2, "llm", "step", None) is None
    assert store._kept == 0 and store._reserved == {}

    # Competing request can still reserve the remaining 8 tokens; then at full pool (room == 0)
    # direct reserve() on warm episode 2 still returns None cleanly
    store.ingest_request(3, RollingKVReqConfig(tokens=8, sink=0))
    store.bind_episode(3, 1, None, cold=True, reset=False)
    assert store.reserve(3, "llm", "step", None) is None
    assert store._room() == 0
    assert store.reserve(2, "llm", "step", None) is None
    assert store._kept == 8 and store._reserved == {3: 8}

    store.remove_request(3)
    store.remove_request(2)
    store.end_episode(0)
    assert store.num_free_tokens() == 16


def test_recurrent_pool_reserve_idempotent_on_reserved_and_partial_warm_episodes():
    cfg_half = RecurrentStateConfig(
        num_layers=1,
        blocks={"s": RecurrentBlockConfig((4,), torch.float32)},
        max_slots=6,  # sink + 5 usable slots
        labels=("cond", "blank"),
    )
    pool = RecurrentStatePool(CPU, cfg_half, name="val", episode_lifetime=True)
    initial_free = pool.num_free_slots

    # Repeated reserve() / reserve_past() on an already-reserved request is idempotent
    pool.ingest_request(100)
    pool.bind_episode(100, 0, "ep-half", cold=True, reset=False)
    assert pool.reserve(100, "llm", "step", None) is None
    assert pool._kept == 2 and pool._reserved == {100: 2}
    assert pool.reserve(100, "llm", "step", None) is None
    assert pool.reserve_past(100, "llm", "step", None) is None
    assert pool._kept == 2 and pool._reserved == {100: 2}
    pool.unreserve(100)
    assert pool._kept == 0 and pool._reserved == {}

    # reserve_past() refuses when room < need * 2, succeeds when room >= need * 2, and is idempotent
    assert pool.reserve_past(100, "llm", "step", None) is None
    assert pool._kept == 2 and pool._reserved == {100: 2}
    assert pool.reserve_past(100, "llm", "step", None) is None
    assert pool._kept == 2 and pool._reserved == {100: 2}
    pool.remove_request(100)
    pool.end_episode(0)
    assert pool._kept == 0 and pool._reserved == {}

    # Partial-label warm resume: rid1 allocates only "cond", rid2 reserves 1 remaining slot ("blank")
    pool.ingest_request(101)
    pool.bind_episode(101, 0, "ep-half", cold=True, reset=False)
    st1 = RecurrentStep(segments=(Segment(101, "cond", 1),))
    ctx1 = StepContext([101], "step", 0, False)
    assert pool.admit(st1, ctx1).ok
    pool.commit(st1, ctx1)
    pool.remove_request(101)
    assert pool._kept == 0 and pool.num_free_slots == initial_free - 1

    pool.ingest_request(102)
    pool.bind_episode(102, 0, "ep-half", cold=False, reset=False)
    assert pool.ask(102, "llm", "step") is ASKS
    assert pool.reserve(102, "llm", "step", None) is None
    assert pool._kept == 1 and pool._reserved == {102: 1}
    assert pool.reserve(102, "llm", "step", None) is None
    assert pool.reserve_past(102, "llm", "step", None) is None
    assert pool._kept == 1 and pool._reserved == {102: 1}
    pool.remove_request(102)
    assert pool._kept == 0 and pool._reserved == {}
    pool.end_episode(0)
    assert pool.num_free_slots == initial_free

    # Full-pool boundary (max_slots=4 -> sink + 3 usable slots):
    # room == 1 after reserving 2 slots; repeated reserve()/reserve_past() must return None, not room=1
    cfg_full = RecurrentStateConfig(
        num_layers=1,
        blocks={"s": RecurrentBlockConfig((4,), torch.float32)},
        max_slots=4,
        labels=("cond", "blank"),
    )
    full = RecurrentStatePool(CPU, cfg_full, name="val_full", episode_lifetime=True)
    full.ingest_request(1)
    full.bind_episode(1, 0, "ep-full", cold=True, reset=False)
    assert full.reserve_past(1, "llm", "step", None) == 3
    assert full._kept == 0 and full._reserved == {}
    assert full.reserve(1, "llm", "step", None) is None
    assert full._kept == 2 and full._reserved == {1: 2}
    assert full.reserve(1, "llm", "step", None) is None
    assert full.reserve_past(1, "llm", "step", None) is None
    assert full._kept == 2 and full._reserved == {1: 2}
    full.remove_request(1)
    full.end_episode(0)
    assert full._kept == 0 and full._reserved == {} and full.num_free_slots == 3


def test_sized_value_and_rolling_bind_episode_reset_without_overrides():
    # 1. Direct SizedValueStore: reset=True with overrides=None preserves prev.config
    sz = SizedValueStore(
        SizedValueConfig(
            num_layers=1,
            fields={"v": SizedField((("n", 1),), torch.float32)},
            largest={"n": 8},
            max_bytes=32,
        ),
        CPU,
        name="sz",
        episode_lifetime=True,
    )
    sz.ingest_request(1, SizedValueReqConfig(sizes={"main": {"n": 4}}))
    sz.bind_episode(1, 0, "ep-sz", cold=True, reset=False)
    assert sz.reserve(1, "dit", "step", None) is None
    assert sz.take(1, "main", 0) == {"v": None}
    sz.put(1, "main", 0, {"v": torch.arange(4, dtype=torch.float32)})
    assert sz._used == 16 and sz.held(1) == 1
    sz.remove_request(1)

    sz.ingest_request(2, None)
    sz.bind_episode(2, 0, "ep-sz", cold=False, reset=True)
    assert sz.sizes(2, "main") == {"n": 4}
    assert sz._used == 0 and sz.held(2) == 0
    assert sz.ask(2, "dit", "step") is ASKS
    assert sz.reserve(2, "dit", "step", None) is None
    assert sz._used == 16
    assert sz.take(2, "main", 0) == {"v": None}
    sz.put(2, "main", 0, {"v": torch.ones(4, dtype=torch.float32)})
    assert sz.held(2) == 1
    sz.remove_request(2)

    # Explicit overrides still take precedence over preserved config on reset=True
    sz.ingest_request(3, SizedValueReqConfig(sizes={"main": {"n": 8}}))
    sz.bind_episode(3, 0, "ep-sz", cold=False, reset=True)
    assert sz.sizes(3, "main") == {"n": 8}
    assert sz.reserve(3, "dit", "step", None) is None
    assert sz._used == 32
    sz.remove_request(3)
    sz.end_episode(0)
    assert sz._used == 0

    # 2. Direct RollingKVManager: reset=True with overrides=None preserves prev_wants
    roll = RollingKVManager(
        RollingKVConfig(num_layers=1, num_heads=1, head_dim=4, pool_tokens=16, dtype=torch.float32),
        CPU,
        name="roll",
        episode_lifetime=True,
    )
    roll.ingest_request(1, RollingKVReqConfig(tokens=8, sink=1))
    roll.bind_episode(1, 0, "ep-roll", cold=True, reset=False)
    assert roll.reserve(1, "llm", "step", None) is None
    st1 = RollingKVStep(segments=(Segment(1, "main", 3),), positions={1: RollingPosition(0, 1, 7)})
    ctx1 = StepContext([1], "step", 0, False)
    assert roll.admit(st1, ctx1).ok
    roll.plan(st1, ctx1)
    roll.commit(st1, ctx1)
    assert roll.written(1) == 3
    roll.remove_request(1)

    roll.ingest_request(2, None)
    roll.bind_episode(2, 0, "ep-roll", cold=False, reset=True)
    assert roll.declared(2) == RollingKVReqConfig(tokens=8, sink=1)
    assert roll.written(2) == 0 and roll.num_free_tokens() == 16
    assert roll.ask(2, "llm", "step") is ASKS
    assert roll.reserve(2, "llm", "step", None) is None
    assert roll._kept == 8
    st2 = RollingKVStep(segments=(Segment(2, "main", 2),), positions={2: RollingPosition(0, 1, 7)})
    ctx2 = StepContext([2], "step", 0, False)
    assert roll.admit(st2, ctx2).ok
    roll.plan(st2, ctx2)
    roll.commit(st2, ctx2)
    assert roll.written(2) == 2
    roll.remove_request(2)

    # Explicit overrides still take precedence over preserved wants on reset=True
    roll.ingest_request(3, RollingKVReqConfig(tokens=10, sink=2))
    roll.bind_episode(3, 0, "ep-roll", cold=False, reset=True)
    assert roll.declared(3) == RollingKVReqConfig(tokens=10, sink=2)
    assert roll.reserve(3, "llm", "step", None) is None
    assert roll._kept == 10
    roll.remove_request(3)
    roll.end_episode(0)
    assert roll._kept == 0 and roll.num_free_tokens() == 16


def test_engine_episode_clear_sized_and_rolling_without_overrides():
    roll = RollingKVManager(
        RollingKVConfig(num_layers=1, num_heads=2, head_dim=8, pool_tokens=16, dtype=torch.float32),
        CPU,
        name="roll",
        episode_lifetime=True,
    )
    sz = SizedValueStore(
        SizedValueConfig(num_layers=1, fields={"v": SizedField((("n", 1),), torch.float32)}, largest={"n": 4}),
        CPU,
        name="sz",
        episode_lifetime=True,
    )
    val = RecurrentStatePool(
        CPU,
        RecurrentStateConfig(
            num_layers=1, blocks={"s": RecurrentBlockConfig(shape=(4,), dtype=torch.float32)}, max_slots=4,
        ),
        name="val",
        episode_lifetime=True,
    )
    resources = {"roll": roll, "sz": sz, "val": val}
    runner = StepRunner(resources, node_resources={"node0": list(resources)})
    submod = SimpleNamespace(on_request_added=lambda rid, info: None, cleanup_request=lambda rid: None)
    engine = make_engine(
        _resources=resources,
        _runner=runner,
        _state_setup=StateSetup(),
        _submodules={"node0": make_mgmt(submodule=submod, resources=resources)},
        _resource_nodes={k: frozenset({"node0"}) for k in resources},
        _episode_pools=dict(resources),
    )

    overrides = {
        "roll": RollingKVReqConfig(tokens=8, sink=1),
        "sz": SizedValueReqConfig(sizes={"main": {"n": 4}}),
    }
    engine.add_request(1, overrides=overrides, episode_id="ep-clr", episode_slot=0)
    assert roll.reserve(1, "node0", "step", None) is None
    rl_step1 = RollingKVStep(segments=(Segment(1, "main", 3),), positions={1: RollingPosition(0, 1, 7)})
    rl_ctx1 = StepContext([1], "step", 0, False)
    assert roll.admit(rl_step1, rl_ctx1).ok
    roll.plan(rl_step1, rl_ctx1)
    roll.commit(rl_step1, rl_ctx1)
    assert sz.reserve(1, "node0", "step", None) is None
    sz.take(1, "main", 0)
    sz.put(1, "main", 0, {"v": torch.ones(4)})
    v_step1 = RecurrentStep(segments=(Segment(1, "main", 1),))
    v_ctx1 = StepContext([1], "step", 0, False)
    assert val.admit(v_step1, v_ctx1).ok
    val.commit(v_step1, v_ctx1)
    engine.remove_request(1)

    # Selective episode_clear=["sz", "roll"] with overrides=None keeps val warm and resets sz/roll cleanly
    engine.add_request(2, episode_id="ep-clr", episode_slot=0, episode_clear=["sz", "roll"])
    batch = ExecutingBatch("node0", {}, StepContext([2], "step", 0, False))
    engine.finalize_batch(batch)
    assert batch.episode_held[2] == {"roll": 0, "sz": 0, "val": 1}
    assert sz.sizes(2, "main") == {"n": 4}
    assert sz.ask(2, "node0", "step") is ASKS
    assert sz.reserve(2, "node0", "step", None) is None
    assert sz.take(2, "main", 0) == {"v": None}
    sz.put(2, "main", 0, {"v": torch.full((4,), 2.0)})

    assert roll.ask(2, "node0", "step") is ASKS
    assert roll.reserve(2, "node0", "step", None) is None
    rl_step2 = RollingKVStep(segments=(Segment(2, "main", 2),), positions={2: RollingPosition(0, 1, 7)})
    rl_ctx2 = StepContext([2], "step", 0, False)
    assert roll.admit(rl_step2, rl_ctx2).ok
    roll.plan(rl_step2, rl_ctx2)
    roll.commit(rl_step2, rl_ctx2)

    batch2 = ExecutingBatch("node0", {}, StepContext([2], "step", 0, False))
    engine.finalize_batch(batch2)
    assert batch2.episode_held[2] == {"roll": 2, "sz": 1, "val": 1}

    engine.remove_request(2)
    assert engine.end_episode("ep-clr", 0) is True
    assert roll.num_free_tokens() == 16 and sz._used == 0 and val.num_free_slots == 3


@pytest.mark.parametrize("debug_asserts", [False, True])
def test_paged_kv_warm_resume_without_layout_clears_footprint_and_span_cursor(monkeypatch, debug_asserts):
    monkeypatch.setattr(kv_manager_module, "_DEBUG_ASSERTS", debug_asserts)
    kv = _paged_kv_manager(max_num_pages=8, page_size=PAGE_SIZE, prefix_cache=False, episode_lifetime=True)

    # A. Committed kv_layout -> warm resume with overrides=None grows past Turn 1's 1-page footprint
    kv.ingest_request(1, _kv_req(list(range(PAGE_SIZE))))
    kv.bind_episode(1, 0, "ep-layout-grow", cold=True, reset=False)
    assert kv._streams[1]["main"].footprint == 1
    assert kv.ask(1, "llm", "prefill") is ASKS
    assert kv.reserve(1, "llm", "prefill", None) is None
    _kv_step(kv, 1, PAGE_SIZE)
    kv.remove_request(1)

    kv.ingest_request(2, None)
    kv.bind_episode(2, 0, "ep-layout-grow", cold=False, reset=False)
    stream2 = kv._streams[2]["main"]
    assert stream2.footprint is None and stream2.span_cursor is None
    assert kv.ask(2, "llm", "prefill") is ASKS
    assert kv.reserve(2, "llm", "prefill", None) is None
    _kv_step(kv, 2, PAGE_SIZE)
    assert kv.held(2) == 2 * PAGE_SIZE and not kv._overran
    kv.remove_request(2)
    kv.end_episode(0)
    kv.assert_pages_conserved()

    # B. Uncommitted kv_layout (stored_len == 0, released == 0) -> warm resume with overrides=None -> commit
    kv.ingest_request(3, _kv_req(list(range(PAGE_SIZE))))
    kv.bind_episode(3, 1, "ep-layout-uncommitted", cold=True, reset=False)
    assert kv._streams[3]["main"].span_cursor is not None
    kv.remove_request(3)

    kv.ingest_request(4, None)
    kv.bind_episode(4, 1, "ep-layout-uncommitted", cold=False, reset=False)
    stream4 = kv._streams[4]["main"]
    assert stream4.span_cursor is None and stream4.footprint is None
    assert kv.ask(4, "llm", "prefill") is ASKS
    assert kv.reserve(4, "llm", "prefill", None) is None
    _kv_step(kv, 4, PAGE_SIZE)
    assert kv.held(4) == PAGE_SIZE and not kv._overran
    kv.remove_request(4)
    kv.end_episode(1)
    kv.assert_pages_conserved()

    # C. Uncommitted kv_layout on "main" -> warm resume with kv_layout omitting "main"
    kv.ingest_request(5, _kv_req(list(range(PAGE_SIZE))))
    kv.bind_episode(5, 2, "ep-layout-other", cold=True, reset=False)
    assert kv._streams[5]["main"].span_cursor is not None
    kv.remove_request(5)

    other_layout = KVReqConfig(
        kv_layout={"other": StreamLayout(spans=(PrefixSpan(PAGE_SIZE, PAGE_SIZE, "prefill"),))}
    )
    kv.ingest_request(6, other_layout)
    kv.bind_episode(6, 2, "ep-layout-other", cold=False, reset=False)
    stream6 = kv._streams[6]["main"]
    assert stream6.span_cursor is None and stream6.footprint == 0
    _kv_step(kv, 6, PAGE_SIZE)
    assert ("main", False) in kv._overran
    kv.remove_request(6)
    kv.end_episode(2)
    kv.assert_pages_conserved()

    # D. _report_overrun defense-in-depth when overrides.kv_layout is None
    kv._overran.clear()
    kv.ingest_request(7, None)
    kv._streams[7]["main"].footprint = 0
    _kv_step(kv, 7, PAGE_SIZE)
    assert ("main", False) in kv._overran
    kv.remove_request(7)
    kv.assert_pages_conserved()


def test_ring_kv_admit_duplicate_mixed_int_and_tuple_keys_returns_admit_error():
    frame = 128
    cfg = RingKVConfig(
        num_layers=1,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=8,
        tokens_per_frame=frame,
        num_sessions=2,
        layers=(RingKVLayerConfig(ring_frames=2, ring_buckets=2, pinned_dilation=1),),
    )
    ring = RingKVManager(cfg, "ring", CPU, torch.float32, episode_lifetime=True)
    rids = [10, ("ep", 0), 10, ("ep", 0)]
    outcome = ring.admit(RingKVStep(segments=(Segment(10, "main", frame),)), StepContext(rids, "step", 0, False))
    assert not outcome.ok and not outcome.ready
    assert isinstance(outcome.reason, AdmitRuntimeError)
    assert "('ep', 0)" in outcome.reason.message and "10" in outcome.reason.message


def test_rolling_kv_remove_request_before_admit_after_reset_preserves_wants():
    cfg = RollingKVConfig(num_layers=1, num_heads=2, head_dim=8, pool_tokens=16, dtype=torch.float32)
    store = RollingKVManager(cfg, CPU, name="roll", episode_lifetime=True)

    # Turn 1 commits on slot 0
    store.ingest_request(1, RollingKVReqConfig(tokens=8, sink=1))
    store.bind_episode(1, 0, "ep-roll-cancel", cold=True, reset=False)
    assert store.reserve(1, "llm", "step", None) is None
    st1 = RollingKVStep(segments=(Segment(1, "main", 3),), positions={1: RollingPosition(0, 1, 7)})
    ctx1 = StepContext([1], "step", 0, False)
    assert store.admit(st1, ctx1).ok
    store.plan(st1, ctx1)
    store.commit(st1, ctx1)
    store.remove_request(1)

    # Turn 2 resets with overrides=None, then is removed BEFORE admit()
    store.ingest_request(2, None)
    store.bind_episode(2, 0, "ep-roll-cancel", cold=False, reset=True)
    assert store.declared(2) == RollingKVReqConfig(tokens=8, sink=1)
    store.remove_request(2)
    assert store._kept == 0 and store._reserved == {}
    assert store._owner_wants[("episode", 0)] == RollingKVReqConfig(tokens=8, sink=1)

    # Turn 3 with overrides=None still inherits preserved RollingKVReqConfig and runs cleanly
    store.ingest_request(3, None)
    store.bind_episode(3, 0, "ep-roll-cancel", cold=False, reset=False)
    assert store.declared(3) == RollingKVReqConfig(tokens=8, sink=1)
    assert store.ask(3, "llm", "step") is ASKS
    assert store.reserve(3, "llm", "step", None) is None
    st3 = RollingKVStep(segments=(Segment(3, "main", 3),), positions={3: RollingPosition(0, 1, 7)})
    ctx3 = StepContext([3], "step", 0, False)
    assert store.admit(st3, ctx3).ok
    store.plan(st3, ctx3)
    store.commit(st3, ctx3)
    assert store.held(3) == 3
    store.remove_request(3)
    store.end_episode(0)
    assert ("episode", 0) not in store._owner_wants
    assert store.num_free_tokens() == 16

    # Cold episode removed before admit() is cleaned up by end_episode()
    store.ingest_request(4, RollingKVReqConfig(tokens=6, sink=0))
    store.bind_episode(4, 1, "ep-roll-cold-cancel", cold=True, reset=False)
    store.remove_request(4)
    assert ("episode", 1) in store._owner_wants
    store.end_episode(1)
    assert ("episode", 1) not in store._owner_wants

    # Non-episode request removed before admit() never leaks into _owner_wants
    store.ingest_request(5, RollingKVReqConfig(tokens=6, sink=0))
    store.bind_episode(5, 2, None, cold=True, reset=False)
    store.remove_request(5)
    assert not store._owner_wants and not store._wants and store.num_free_tokens() == 16
