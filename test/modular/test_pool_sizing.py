"""Pool sizing from declarations (``mstar/engine/resources/sizing.py``).

Each pool's config says what one unit costs and what a request holds. These check
that arithmetic against what the resources allocate, the per-request bounds a model
declares, and the startup check that refuses a request cap a pool can't hold.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.engine.resources.dsa.config import DSAConfig, DSASpec
from mstar.engine.resources.kv.bounded.config import BoundedKVConfig, BoundedKVSpec, SinkWindow
from mstar.engine.resources.kv.config import KVLayout, KVSpec, PagedKVConfig, RingKVConfig, RingKVLayerConfig
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.sizing import (
    PoolDemand,
    concurrency,
    memory_allows,
    pool_demands,
    requests_for_unsized,
    requests_within,
    units_for,
)
from mstar.validation.resources import pools_hold_concurrency

CPU = torch.device("cpu")


def _paged(**kwargs) -> PagedKVConfig:
    return PagedKVConfig(**{
        "num_layers": 3, "num_kv_heads": 4, "num_qo_heads": 16, "head_dim": 16, "max_seq_len": 1000,
        "page_size": 16, "max_num_pages": 9, **kwargs,
    })


CTX_PAGES = 1 + 30 * 63  # 30 requests of 1000 tokens


def _model(labels=None, bound=None):
    """What the sizing reads off a model: its labels per pool and its own bound."""
    return SimpleNamespace(
        kv_labels=lambda: labels or {},
        pool_units_per_request=lambda spec: (bound or {}).get(spec.resource_key),
    )


# ── a unit's bytes are what the resource allocates ─────────────────────────


@pytest.mark.parametrize("shards", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_paged_page_is_what_the_cache_allocates(shards, dtype):
    """8 ranks over 4 KV heads replicate one head per rank, as ``shard`` does."""
    from mstar.engine.resources.kv.cache import KVCache

    cfg = _paged()
    page = cfg.unit_bytes(dtype, shards)
    cfg.shard(shards)
    assert KVCache(cfg, CPU, dtype).nbytes == page * cfg.max_num_pages


@pytest.mark.parametrize("shards", [1, 4])
def test_mla_page_is_one_latent_per_token_on_every_rank(shards):
    from mstar.engine.resources.kv.cache import KVCache

    cfg = PagedKVConfig(
        num_layers=2, num_kv_heads=1, head_dim=1, max_seq_len=256, max_num_pages=5,
        layout=KVLayout.MLA, kv_lora_rank=32, qk_rope_head_dim=8, num_qo_heads=8,
    )
    page = cfg.unit_bytes(torch.bfloat16, shards)
    assert page == 2 * 128 * 40 * 2
    cfg.shard(shards)
    assert KVCache(cfg, CPU, torch.bfloat16).nbytes == page * 5


@pytest.mark.parametrize("shards", [1, 2])
def test_recurrent_slot_is_what_the_pool_allocates(shards):
    """The state block is fp32 whatever the engine computes in; the conv block
    takes the engine's dtype."""
    from mstar.engine.resources.recurrent.pool import RecurrentStatePool

    cfg = RecurrentStateConfig(
        num_layers=2, max_slots=5,
        blocks=DeltaNetGeometry(
            num_k_heads=2, num_v_heads=4, head_k_dim=8, head_v_dim=8, conv_kernel_size=4,
        ).to_blocks(state_dtype=torch.float32, conv_dtype=None),
    )
    slot = cfg.unit_bytes(torch.bfloat16, shards)
    cfg.shard(shards)
    pool = RecurrentStatePool(CPU, cfg, kv_dtype=torch.bfloat16)
    assert sum(t.nbytes for t in pool._blocks.values()) == slot * 5
    assert (cfg.pool_units, cfg.reserved_units) == (5, 1)


def test_bounded_slot_is_what_the_pool_allocates():
    from mstar.engine.resources.kv.bounded.manager import BoundedKVManager

    cfg = BoundedKVConfig(
        num_layers=2, num_heads=2, head_dim=8, rows_per_request=2, max_source_len=4,
        retention=SinkWindow(6, 5), max_slots=3, dtype=None,
    )
    manager = BoundedKVManager(cfg, CPU, kv_dtype=torch.bfloat16)
    assert manager._cache.nbytes == cfg.unit_bytes(torch.bfloat16) * 3


def test_ring_session_is_what_the_rings_allocate():
    """The padding session is allocated and never handed out."""
    from mstar.engine.resources.kv.ring.manager import RingKVManager

    cfg = RingKVConfig(
        num_layers=2, num_kv_heads=2, head_dim=8, tokens_per_frame=128, num_sessions=2,
        layers=(RingKVLayerConfig(2, 2, 1), RingKVLayerConfig(3, 3, 1)),
    )
    manager = RingKVManager(cfg, "kv", CPU, torch.bfloat16)
    assert sum(layer.kv.nbytes for layer in manager.layers) == cfg.unit_bytes(torch.bfloat16) * 3
    assert (cfg.pool_units, cfg.reserved_units) == (3, 1)


# ── what a request holds ────────────────────────────────────────────────────


def test_a_request_holds_a_stream_per_declared_label_unless_the_model_bounds_it():
    specs = [KVSpec("kv", {"n"}, _paged())]
    [one] = pool_demands(specs, torch.bfloat16)
    assert one.per_request == 63  # ceil(1000 / 16)
    [cfg] = pool_demands(specs, torch.bfloat16, model=_model({"kv": {"cond": None, "uncond": None}}))
    assert (cfg.per_request, cfg.requests) == (126, 0)
    [bounded] = pool_demands(specs, torch.bfloat16, model=_model({"kv": {"cond": None}}, {"kv": 4}))
    assert (bounded.per_request, bounded.requests) == (4, 2)


def test_slot_pools_take_one_slot_and_index_keys_take_their_cache_length():
    state = RecurrentStateSpec("state", {"n"}, RecurrentStateConfig(
        num_layers=1, max_slots=7,
        blocks=DeltaNetGeometry(1, 1, 8, 8, 4).to_blocks(state_dtype=torch.float32),
    ))
    keys = DSASpec("dsa", {"n"}, DSAConfig(kv_cache="kv", topk=4, key_layers=3, key_dim=16))
    unsized = DSASpec("dsa2", {"n"}, DSAConfig(kv_cache="kv", topk=4))
    demands = {
        d.key: d for d in pool_demands(
            [KVSpec("kv", {"n"}, _paged()), state, keys, unsized], torch.bfloat16,
        )
    }
    assert sorted(demands) == ["dsa", "kv", "state"]  # undeclared key geometry is left out
    assert (demands["state"].per_request, demands["state"].requests) == (1, 6)
    assert demands["dsa"].units is None and demands["dsa"].per_request == 1000
    assert demands["dsa"].unit_bytes == 3 * 16 * 2


def test_concurrency_and_what_memory_holds():
    kv = PoolDemand("kv", "page", "max_num_pages", unit_bytes=100, units=21, reserved=1, per_request=4,
                    max_seq_len=64)
    state = PoolDemand("state", "slot", "max_slots", unit_bytes=50, units=9, reserved=1, per_request=1,
                       max_seq_len=None)
    keys = PoolDemand("keys", "token", "", unit_bytes=1, units=None, reserved=0, per_request=10,
                      max_seq_len=64)
    assert (kv.requests, state.requests, keys.requests) == (5, 8, None)
    assert concurrency([kv, state, keys]) == 5
    # a request takes 4 x 100 + 50 + 10 bytes, after the reserved 150
    assert requests_within([kv, state, keys], 150 + 7 * 460 + 459) == 7
    assert units_for(kv, 7) == 29
    # the pools' own 2100 + 450 bytes plus 450 spare hold 6 at 450 a request (keys left out)
    assert memory_allows([kv, state], 450) == "450 B spare would hold 6 at once (kv 25 pages, state 7 slots)"


def test_the_engine_logs_each_pool_and_the_requests_they_hold(caplog):
    engine = _engine()
    engine._pools = pool_demands([KVSpec("kv", {"n"}, _paged())], torch.bfloat16)
    with caplog.at_level(logging.INFO, logger="mstar.engine.engine"):
        engine._log_pools(None)
    assert "kv: 9 pages x 12.0 KiB = 108.0 KiB; a request takes 63 at max_seq_len 1000 -> 0 at once" in caplog.text
    assert "Pools[cpu]: 0 requests at once" in caplog.text


# ── the startup check ───────────────────────────────────────────────────────


def test_a_cap_a_pool_cannot_hold_is_refused_unless_the_pool_offloads(caplog):
    model = _model(bound={"kv": 4})
    specs = [KVSpec("kv", {"n"}, _paged())]
    assert pools_hold_concurrency(specs, model, {"max_concurrent_requests": 2}) == []
    [problem] = pools_hold_concurrency(specs, model, {"max_concurrent_requests": 3})
    assert problem.startswith("3 concurrent requests (max_concurrent_requests) need 13 pages of resources.kv")
    assert "raise its max_num_pages" in problem
    offloading = [KVSpec("kv", {"n"}, _paged(cpu_offload_pages=8))]
    with caplog.at_level(logging.WARNING, logger="mstar.validation.resources"):
        assert pools_hold_concurrency(offloading, model, {"max_concurrent_requests": 3}) == []
    assert "the worker offloads requests to the host" in caplog.text


def test_slot_pools_hold_a_request_each_past_their_sink():
    spec = RecurrentStateSpec("state", {"n"}, RecurrentStateConfig(
        num_layers=1, max_slots=5, blocks=DeltaNetGeometry(1, 1, 8, 8, 4).to_blocks(),
    ))
    assert pools_hold_concurrency([spec], _model(), {"max_concurrent_requests": 4}) == []
    [problem] = pools_hold_concurrency([spec], _model(), {"max_concurrent_requests": 5})
    assert "need 6 slots of resources.state" in problem
    bounded = BoundedKVSpec("enc", {"n"}, BoundedKVConfig(
        num_layers=1, num_heads=1, head_dim=8, rows_per_request=1, max_source_len=0,
        retention=SinkWindow(0, 4), max_slots=5,
    ))
    assert pools_hold_concurrency([bounded], _model(), {"max_concurrent_requests": 5}) == []


@pytest.mark.parametrize("cap", [0, -1, True, "64", 8.0])
def test_a_cap_that_is_not_a_positive_int_is_refused(cap):
    [problem] = pools_hold_concurrency([KVSpec("kv", {"n"}, _paged())], _model(), {"max_concurrent_requests": cap})
    assert "positive int" in problem


def test_no_cap_starts_with_a_warning_naming_what_the_pools_hold(caplog):
    with caplog.at_level(logging.WARNING, logger="mstar.validation.resources"):
        assert pools_hold_concurrency([KVSpec("kv", {"n"}, _paged())], _model(bound={"kv": 4}), {}) == []
    assert "sets no max_concurrent_requests: past what kv (2 at once) hold" in caplog.text


@pytest.mark.parametrize("disable_sink_slot, low", [(False, 2), (True, 1)])
def test_a_slot_count_with_none_to_hand_out_is_refused_from_yaml(disable_sink_slot, low):
    spec = RecurrentStateSpec("state", {"n"}, RecurrentStateConfig(
        num_layers=1, blocks=DeltaNetGeometry(1, 1, 8, 8, 4).to_blocks(), disable_sink_slot=disable_sink_slot,
    ))
    spec.apply_yaml_overrides(max_slots=low)
    with pytest.raises(ValueError, match=f"max_slots must be an int >= {low}"):
        spec.apply_yaml_overrides(max_slots=low - 1)


# ── sizing the unsized pools ────────────────────────────────────────────────


def test_an_unsized_pool_holds_its_default_until_the_engine_sizes_it():
    from mstar.engine.resources.kv.bounded.config import DEFAULT_MAX_SLOTS as BOUNDED_SLOTS
    from mstar.engine.resources.kv.config import DEFAULT_MAX_NUM_PAGES
    from mstar.engine.resources.recurrent.config import DEFAULT_MAX_SLOTS

    paged = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64)
    assert (paged.max_num_pages, paged.sized) == (DEFAULT_MAX_NUM_PAGES, False)
    spec = KVSpec("kv", {"n"}, paged)
    spec.apply_yaml_overrides(page_size=64)
    assert not paged.sized
    spec.apply_yaml_overrides(max_num_pages=100)
    assert (paged.max_num_pages, paged.sized) == (100, True)
    assert _paged().sized  # set by the model
    state = RecurrentStateConfig(num_layers=1, blocks=DeltaNetGeometry(1, 1, 8, 8, 4).to_blocks())
    assert (state.max_slots, state.sized) == (DEFAULT_MAX_SLOTS, False)
    bounded = BoundedKVConfig(
        num_layers=1, num_heads=1, head_dim=8, rows_per_request=1, max_source_len=0, retention=SinkWindow(0, 4),
    )
    assert (bounded.max_slots, bounded.sized) == (BOUNDED_SLOTS, False)
    # a ring's sessions are a real-time budget, so nothing sizes them from memory
    ring = RingKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, tokens_per_frame=128, layers=(RingKVLayerConfig(2, 2, 1),),
    )
    [demand] = pool_demands([KVSpec("ring", {"n"}, ring)], torch.bfloat16)
    assert not demand.unsized


def test_unsized_pools_split_what_the_sized_ones_leave():
    sized = PoolDemand("a", "page", "max_num_pages", unit_bytes=100, units=11, reserved=1, per_request=5,
                       max_seq_len=80)
    kv = PoolDemand("kv", "page", "max_num_pages", unit_bytes=100, units=2048, reserved=1, per_request=4,
                    max_seq_len=64, unsized=True)
    state = PoolDemand("state", "slot", "max_slots", unit_bytes=50, units=256, reserved=1, per_request=1,
                       max_seq_len=None, unsized=True)
    # 1100 bytes go to the sized pool and 150 to the reserved units; a request takes 450
    assert requests_for_unsized([sized, kv, state], 1100 + 150 + 9 * 450 + 449) == 9
    assert (units_for(kv, 9), units_for(state, 9)) == (37, 10)


def _unsized_specs():
    return [
        KVSpec("kv", {"n"}, PagedKVConfig(
            num_layers=3, num_kv_heads=4, num_qo_heads=16, head_dim=16, max_seq_len=1000, page_size=16,
        )),
        RecurrentStateSpec("state", {"n"}, RecurrentStateConfig(
            num_layers=1, blocks=DeltaNetGeometry(2, 2, 8, 8, 4).to_blocks(state_dtype=torch.float32),
        )),
        KVSpec("ctx", {"n"}, _paged(max_num_pages=CTX_PAGES)),  # sized by the model
    ]


def _engine() -> Engine:
    engine = Engine.__new__(Engine)
    engine._device = CPU
    engine.derived_concurrency = None
    return engine


def test_the_engine_sizes_the_unsized_pools_for_one_request_count(caplog):
    specs = _unsized_specs()
    kv, state, ctx = (spec.config for spec in specs)
    page, slot = kv.unit_bytes(torch.bfloat16), state.unit_bytes(torch.bfloat16)
    request = 63 * page + slot
    budget = CTX_PAGES * ctx.unit_bytes(torch.bfloat16) + page + slot + 20 * request + request - 1
    engine = _engine()
    with caplog.at_level(logging.INFO, logger="mstar.engine.engine"):
        engine._size_pools(specs, dict.fromkeys(("kv", "state", "ctx")), torch.bfloat16, None, lambda: budget)
    assert (kv.max_num_pages, state.max_slots, ctx.max_num_pages) == (1 + 20 * 63, 21, CTX_PAGES)
    assert engine.derived_concurrency == 20
    # a sized pool that holds fewer caps the count the conductor gets
    specs = _unsized_specs()
    specs[2].config.max_num_pages = 1 + 5 * 63
    engine._size_pools(specs, dict.fromkeys(("kv", "state", "ctx")), torch.bfloat16, None, lambda: budget)
    assert engine.derived_concurrency == 5
    assert "kv sized to 1261 pages for 20 requests at once" in caplog.text


def test_a_budget_that_holds_no_request_fails_startup():
    specs = _unsized_specs()
    with pytest.raises(RuntimeError, match=r"pools left unsized \(kv, state\) hold no request"):
        _engine()._size_pools(specs, dict.fromkeys(("kv", "state", "ctx")), torch.bfloat16, None, lambda: 10**5)


def _size_on_rank(rank, budgets, store, out):
    """One rank of a TP2 instance sizing its pools from its own budget."""
    import datetime
    import traceback

    import torch.distributed as dist

    from mstar.distributed.communication import CommGroup, JointGroups

    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{store}", rank=rank, world_size=2,
            timeout=datetime.timedelta(seconds=20),
        )
        tp = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
        tp.device_group = tp.cpu_group = dist.group.WORLD
        group = JointGroups(tp_group=tp, sp_group=CommGroup.trivial())
        specs = _unsized_specs()
        try:
            _engine()._size_pools(specs, dict.fromkeys(("kv", "state", "ctx"), group), torch.bfloat16, None,
                                  lambda: budgets[rank])
            out.put((rank, [spec.config.pool_units for spec in specs]))
        except RuntimeError as e:
            out.put((rank, f"refused: {e}"))
        dist.destroy_process_group()
    except BaseException:
        out.put((rank, "crash " + traceback.format_exc()))


def _two_ranks(tmp_path, budgets):
    import multiprocessing as mp
    import queue

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [ctx.Process(target=_size_on_rank, args=(r, budgets, tmp_path / "store", out)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, result = out.get(timeout=60)
            results[rank] = result
    except queue.Empty:
        pass
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
    assert sorted(results) == [0, 1], results
    return results


def test_lockstep_ranks_take_the_least_count_either_holds(tmp_path):
    """Rank 1 has less memory free; both ranks of the TP2 instance size their pools for
    what rank 1 holds, so the two pools match page for page. The pools shard across the
    two ranks, so a page here is half of one on a single rank."""
    kv, state, _ = (spec.config for spec in _unsized_specs())
    request = 63 * kv.unit_bytes(torch.bfloat16, 2) + state.unit_bytes(torch.bfloat16, 2)
    fixed = CTX_PAGES * _paged().unit_bytes(torch.bfloat16, 2) + kv.unit_bytes(torch.bfloat16, 2) + \
        state.unit_bytes(torch.bfloat16, 2)
    results = _two_ranks(tmp_path, [fixed + 30 * request, fixed + 12 * request])
    assert results[0] == results[1] == [1 + 12 * 63, 13, CTX_PAGES], results


def test_a_rank_that_holds_no_request_fails_the_whole_instance(tmp_path):
    """Rank 0 alone could hold requests; it still fails, after the reduction, rather
    than build pools rank 1 never will and wait on it at the first collective."""
    kv, state, _ = (spec.config for spec in _unsized_specs())
    request = 63 * kv.unit_bytes(torch.bfloat16, 2) + state.unit_bytes(torch.bfloat16, 2)
    results = _two_ranks(tmp_path, [10**9 + 30 * request, 10**3])
    assert all(str(result).startswith("refused: the pools left unsized") for result in results.values()), results


@pytest.mark.parametrize("rank_devices, device, expected", [
    (None, "cuda:0", 1),
    ({0: 0, 1: 0}, "cuda:0", 2),
    ({1: 0}, "cuda:1", 0 + 1),  # rank 1 moved to GPU 0, so nothing else is on GPU 1
    (None, "cpu", 1),
])
def test_workers_sharing_a_device_split_it(rank_devices, device, expected):
    from mstar.worker.engine_manager import workers_on_device

    config = {"node_groups": [{"node_names": ["a"], "ranks": [0]}, {"node_names": ["b"], "ranks": [1]}]}
    if rank_devices is not None:
        config["rank_devices"] = rank_devices
    assert workers_on_device(config, torch.device(device)) == expected


def test_the_conductor_caps_requests_at_what_the_sized_pools_hold(caplog):
    from mstar.conductor.conductor import Conductor
    from mstar.utils.ipc_format import ConductorMessage, ConductorMessageType, SetupDone

    def conductor(cap, counts):
        c = Conductor.__new__(Conductor)
        c.worker_ids = [f"w{i}" for i in range(len(counts))]
        c.max_concurrent_requests = cap
        c._startup_message_backlog = []
        c._poll_worker_liveness = lambda: None
        messages = [ConductorMessage(ConductorMessageType.SETUP_DONE, SetupDone(w, n))
                    for w, n in zip(c.worker_ids, counts, strict=True)]
        c.communicator = SimpleNamespace(get_all_new_messages=lambda: messages)
        c._wait_for_workers_ready()
        return c.max_concurrent_requests

    assert conductor(None, [40, None, 12]) == 12
    assert conductor(None, [None]) is None
    assert conductor(8, [40]) == 8
    with caplog.at_level(logging.WARNING, logger="mstar.conductor.conductor"):
        assert conductor(64, [40]) == 64
    assert "max_concurrent_requests 64 is more than the 40 requests" in caplog.text


def test_the_startup_check_leaves_an_unsized_pool_to_the_engine():
    spec = KVSpec("kv", {"n"}, PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=1 << 20))
    assert pools_hold_concurrency([spec], _model(), {"max_concurrent_requests": 64}) == []
