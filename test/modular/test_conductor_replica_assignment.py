"""Data-parallel replica choice: a new request goes to the replica where its cost
is lowest, the live requests on the replica plus ``routing_cost`` of the facts
the mirrors predict there (a KV cache: minus the share of the prompt it already
holds). With no mirror that is the replica serving the fewest live requests, so
concurrent requests spread over the replicas instead of landing on one at random."""
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.conductor.routing import ReplicaRouter, RoutingWeights, routing_cost  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import (  # noqa: E402
    KVReqConfig,
    KVSpec,
    KVStep,
    PagedKVConfig,
    PrefixSpan,
    StreamLayout,
)
from mstar.engine.resources.kv.keys import chain  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.spec import RoutingContext, RoutingFacts  # noqa: E402
from mstar.engine.resources.step import Segment, StepContext  # noqa: E402
from mstar.model.base import WorkerGraph  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs  # noqa: E402

PAGE = 4


@pytest.fixture(autouse=True, params=["first", "last"])
def tie(request, monkeypatch):
    """Ties go to the first replica, then to the last, so a case that a tie would
    pass by luck fails under one of them when what it tests is gone."""
    pick = (lambda seq: seq[0]) if request.param == "first" else (lambda seq: seq[-1])
    monkeypatch.setattr("mstar.conductor.routing.random.choice", pick)


def _wg(group_id, ranks, sp_size=1, nodes=("llm",), walk="prefill"):
    return WorkerGraph(
        section=SimpleNamespace(get_nodes=lambda: dict.fromkeys(nodes)), graph_walks={walk},
        ranks=list(ranks), sp_size=sp_size, _group_id=group_id,
    )


class _Model:
    """What the router reads of a model: its partitions, and the walks that key each KV label."""

    def __init__(self, keyed=None, partitions=None):
        self._keyed = {"kv": {"main": ("prefill",)}} if keyed is None else keyed
        self._partitions = partitions or [SimpleNamespace(name="default", graph_walks={"prefill"})]

    def get_partitions(self):
        return self._partitions

    def kv_labels(self):
        return {
            key: {label: SimpleNamespace(keys=SimpleNamespace(walks=walks)) for label, walks in labels.items()}
            for key, labels in self._keyed.items()
        }

    def checkpoint_path(self):
        return "/a/checkpoint"


def _conductor(worker_graphs, specs=(), weights=None, model=None):
    conductor = object.__new__(Conductor)
    conductor.worker_graphs = worker_graphs
    conductor.requests = {}
    conductor._router = ReplicaRouter(model or _Model(), worker_graphs.values(), specs=specs, weights=weights)
    return conductor


def _configs(model_kwargs: dict) -> dict:
    """The request's resource configs, opened as ``_do_ingest_request`` opens them."""
    configs = {}
    for key, keys in model_kwargs.get("prefix_keys", {}).items():
        configs[key] = KVReqConfig(needed_labels=["main"])
        configs[key].apply_conductor_config(
            prefix_keys=keys, kv_layout=model_kwargs["kv_layout"][key], prefix_cache=model_kwargs.get("prefix_cache"),
        )
    return configs


def _admit(conductor, rid, model_kwargs=None, passes=1):
    """What ``_do_ingest_request`` does for the pick, and nothing else. The
    request's partition has had ``passes`` passes come back when it ends."""
    configs = _configs(model_kwargs or {})
    router = conductor._router
    groups_run = router.groups_run(["default"])
    placement = router.pick(configs, groups_run)
    conductor._register_request(rid, SimpleNamespace(
        routing=router.add(placement, configs, groups_run),
        partition_states={"default": SimpleNamespace(fwd_pass_number=passes)},
    ))
    return conductor._assign_worker_graphs_to_workers(placement.replicas)


def test_requests_alternate_over_dp_replicas():
    # two node groups replicated on ranks 0 and 1, one wg each
    conductor = _conductor({0: _wg(0, [0, 1]), 1: _wg(0, [0, 1])})
    first = _admit(conductor, "a")
    second = _admit(conductor, "b")
    # a group's wgs stay on one replica, and the second request takes the idle one
    assert first[0] == first[1] and second[0] == second[1]
    assert first[0] != second[0]
    # a request leaving frees its replica's count, once
    conductor._drop_request("a")
    conductor._drop_request("a")
    assert sorted(conductor._router._load.values()) == [0, 1]
    assert _admit(conductor, "c")[0] == first[0]


def test_lockstep_instances_are_balanced_whole():
    # one sp2 node replicated as two instances: ranks (0, 1) and (2, 3)
    conductor = _conductor({0: _wg(0, [0, 1, 2, 3], sp_size=2)})
    picks = {tuple(_admit(conductor, rid)[0]) for rid in ("a", "b")}
    assert picks == {("worker_0", "worker_1"), ("worker_2", "worker_3")}


def test_groups_sharing_a_worker_balance_independently():
    # group 1 is DP2 on ranks 0 and 1; group 0 has one replica, on rank 0. Rank
    # 0's group-0 requests must not push every group-1 request onto rank 1.
    conductor = _conductor({0: _wg(0, [0]), 1: _wg(1, [0, 1])})
    picks = [_admit(conductor, rid)[1][0] for rid in "abcd"]
    assert sorted(picks) == ["worker_0", "worker_0", "worker_1", "worker_1"]


# ── the KV cache's mirror ────────────────────────────────────────────────


def _kv_spec():
    return KVSpec(resource_key="kv", nodes={"llm"}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, max_num_pages=64, page_size=PAGE,
    ))


def _kv_conductor(weights=None):
    spec = _kv_spec()
    conductor = _conductor({0: _wg(0, [0, 1])}, specs=[spec], weights=weights)
    conductor._resource_specs_cache = {"kv": spec}
    return conductor


def _prompt(prefix: str, pages: int = 4, unkeyed: int = 0, keys=("kv",)) -> dict:
    """The kwargs of a request whose prompt is ``pages`` whole pages keyed under
    ``prefix``, then ``unkeyed`` positions the data worker leaves unkeyed, in
    each KV resource of ``keys``."""
    spans = [(pages * PAGE, pages * PAGE, "prefill")] + [(unkeyed, unkeyed, "prefill")] * bool(unkeyed)
    return {
        "prefix_keys": {key: {"main": [f"{prefix}{i}".encode() for i in range(pages)]} for key in keys},
        "kv_layout": {key: {"main": (spans, (), 0, None, 1)} for key in keys},
    }


def _replica(conductor, rid, model_kwargs, passes=1) -> str:
    return _admit(conductor, rid, model_kwargs, passes)[0][0]


def _mirrors(router):
    return [m.mirror for unit in router._units for m in unit.mirrors]


def _held(mirror) -> int:
    """The pages live requests hold, over every replica: each indexed page has the index as one owner."""
    pools = zip(mirror._arenas, mirror._indexes, strict=True)
    return sum(sum(arena.num_owners) - index.num_pages() for arena, index in pools)


def _cached(mirror) -> int:
    return sum(index.num_pages() for index in mirror._indexes)


def _place_a(conductor) -> str:
    """Request "a" (prefix P) on the replica a tie would not pick, with the
    replicas left equally loaded and nothing of P's on the other one."""
    _replica(conductor, "x", _prompt("X"))
    a = _replica(conductor, "a", _prompt("P"))
    conductor._drop_request("x")
    return a


def test_same_prefix_requests_co_locate_before_any_completion():
    conductor = _kv_conductor()
    # a burst: no request has completed, so only placements fed the mirror
    a = _place_a(conductor)
    assert _replica(conductor, "b", _prompt("Q")) != a, (
        "a request with nothing cached anywhere went to the busier replica"
    )
    assert _replica(conductor, "c", _prompt("P")) == a, (
        "with equal load, a request did not follow its prefix to the replica placed with it"
    )


def test_a_hot_prefix_follows_its_hit_until_load_outweighs_it():
    conductor = _kv_conductor()
    a = _place_a(conductor)
    _replica(conductor, "b", _prompt("Q"))
    # each request matches 4 of its 5 pages on replica a, worth 0.8 * prefix live requests
    picks = [_replica(conductor, f"p{i}", _prompt("P", pages=5)) for i in range(16)]
    stayed = picks.index(next(p for p in picks if p != a))
    assert stayed == math.ceil(0.8 * RoutingWeights().prefix) > 1, (
        f"a hot prefix took {stayed} more requests on its replica before one spread"
    )


def test_two_caches_on_one_replica_credit_a_hit_less_than_one_request():
    # two keyed caches whose nodes share the replicas: their hits are one share of the prompt, not one each
    specs = [
        KVSpec(resource_key=key, nodes={node}, config=_kv_spec().config)
        for key, node in (("kv", "llm"), ("kv2", "llm2"))
    ]
    model = _Model(keyed={"kv": {"main": ("prefill",)}, "kv2": {"main": ("prefill",)}})
    conductor = _conductor({0: _wg(0, [0, 1], nodes=("llm", "llm2"))}, specs=specs, model=model,
                           weights={"prefix": 1})
    both = ("kv", "kv2")
    _replica(conductor, "x", _prompt("X", keys=both))
    a = _replica(conductor, "a", _prompt("P", keys=both))
    conductor._drop_request("x")
    _replica(conductor, "b", _prompt("Q", keys=both))
    assert _replica(conductor, "c", _prompt("P", keys=both)) == a
    assert _replica(conductor, "d", _prompt("P", keys=both)) != a, (
        "two caches' hits outweighed a live request, so a hot prefix stayed on the busier replica"
    )


def test_a_request_that_ends_before_a_pass_comes_back_leaves_nothing_cached():
    # aborted while queued, say: the worker never indexed its pages
    conductor = _kv_conductor()
    a = _place_a(conductor)
    _replica(conductor, "q", _prompt("Q"))
    # P's 4 pages, then 2 more that only b indexes
    assert _replica(conductor, "b", _prompt("P", pages=6), passes=0) == a
    conductor._drop_request("b")
    (mirror,) = _mirrors(conductor._router)
    assert mirror.facts(_configs(_prompt("P", pages=6))["kv"])[int(a[-1])].kv_hit == 4 * PAGE, (
        "the pages only the unrun request indexed are still predicted cached"
    )


def test_warmup_copies_of_one_prompt_spread_over_the_replicas():
    # as the startup warmup sends one copy per replica, before any completes,
    # with the prefix cache off so that no predicted hit gathers them
    conductor = _kv_conductor()
    picks = [_replica(conductor, rid, {**_prompt("P"), "prefix_cache": False}) for rid in "abcd"]
    assert sorted(picks) == ["worker_0", "worker_0", "worker_1", "worker_1"]


def test_a_hit_is_priced_as_its_share_of_the_whole_prompt():
    conductor = _kv_conductor()
    # as BAGEL lays out a generation: a keyed text span, then image positions the data worker leaves unkeyed
    kwargs = _prompt("P", pages=2, unkeyed=4000)
    a = int(_replica(conductor, "a", kwargs)[-1])
    (mirror,) = [m.mirror for unit in conductor._router._units for m in unit.mirrors]
    facts = mirror.facts(_configs(kwargs)["kv"])
    # the worker matches one of the two keyed pages, which saves PAGE of 2 * PAGE + 4000 positions
    assert facts[a] == RoutingFacts(work=2 * PAGE + 4000, kv_hit=PAGE)
    assert facts[1 - a] == RoutingFacts(work=2 * PAGE + 4000, kv_hit=0)
    weights = RoutingWeights()
    assert routing_cost(facts[a], weights) == pytest.approx(-weights.prefix * PAGE / (2 * PAGE + 4000))
    assert routing_cost(facts[1 - a], weights) == 0
    # a node no mirror reports on prices at 0
    assert routing_cost(RoutingFacts(), weights) == 0


def test_a_prefix_weight_of_0_picks_by_load_alone():
    picks = []
    for conductor in (_conductor({0: _wg(0, [0, 1])}), _kv_conductor(weights={"prefix": 0})):
        _place_a(conductor)
        # equal loads, then a request whose prefix only a's replica holds
        picks.append([_replica(conductor, rid, _prompt(prefix)) for rid, prefix in (("b", "Q"), ("c", "P"))])
    assert picks[0] == picks[1]


def test_a_dropped_request_releases_its_reservation_once_and_its_pages_stay_cached():
    conductor = _kv_conductor()
    a = _place_a(conductor)
    # dropped twice, as an abort racing a completion can
    conductor._drop_request("a")
    conductor._drop_request("a")
    assert not any(conductor._router._load.values())
    (mirror,) = _mirrors(conductor._router)
    assert not _held(mirror), "a dropped request still holds its pages"
    # with both replicas idle, the prefix it left cached decides
    assert _replica(conductor, "b", _prompt("P")) == a


def test_a_request_that_runs_none_of_a_resources_nodes_takes_no_part_in_its_mirror():
    # an encoder group and the LLM's on the same ranks pick together; the cache is the LLM's
    router = ReplicaRouter(
        _Model(partitions=[SimpleNamespace(name="encoder", graph_walks={"encode"}),
                           SimpleNamespace(name="default", graph_walks={"prefill"})]),
        [_wg(0, [0, 1], nodes=("enc",), walk="encode"), _wg(1, [0, 1])], specs=[_kv_spec()],
    )
    configs = _configs(_prompt("P"))
    groups_run = router.groups_run(["encoder"])
    router.add(router.pick(configs, groups_run), configs, groups_run)
    (mirror,) = _mirrors(router)
    assert not _held(mirror), (
        "a request that runs only the encoder reserved pages in the LLM's cache"
    )


def test_a_request_refused_at_ingest_leaves_nothing_in_the_mirrors():
    conductor = _kv_conductor()
    walk = SimpleNamespace(request_done=False, step_metadata={}, full_metadata=SimpleNamespace(
        graph_walk="prefill", kwargs={},
    ))
    conductor.model = SimpleNamespace(
        LENGTH_KWARGS=("max_output_tokens",),
        get_partitions=lambda: [SimpleNamespace(name="default", graph_walks={"prefill"})],
        get_partition_topology=lambda: SimpleNamespace(connections=[]),
        get_initial_forward_pass_args=lambda **kw: walk,
        get_max_output_tokens=lambda **kw: 8,
        get_loop_caps=lambda **kw: {},
        get_request_resource_configs=lambda **kw: {},
    )
    conductor._status_interval_s = None
    conductor._loop_max_iters = {}
    conductor._kv_labels = {"kv": ("main",)}
    conductor._build_request_sharding_config = lambda workers: None
    refused = []
    conductor._reject_request = lambda rid, error, status: refused.append(status)
    conductor._admit_request(SimpleNamespace(
        # 80 prompt positions, past the pool's max_seq_len of 64: a 400 in validate
        request_id="a", client_request_id=None, model_kwargs=_prompt("P", unkeyed=64), initial_signals={},
        initial_input_modalities=["text"], initial_output_modalities=["text"],
    ))
    assert refused == [400] and not conductor.requests
    (mirror,) = _mirrors(conductor._router)
    assert not _held(mirror) and not _cached(mirror), (
        "a request refused before dispatch left its prefix predicted cached"
    )
    assert not any(conductor._router._load.values())


@pytest.mark.parametrize("worker_graphs,nodes,walks,where", [
    # BAGEL with CFG-parallel nodes: its `kv` spans LLM, LLM_cfg_text and
    # LLM_cfg_img, each its own group, and only LLM runs the keyed walks
    ([_wg(0, [0, 1], nodes=("LLM",), walk="prefill_text"), _wg(0, [0, 1], nodes=("vit", "LLM"), walk="prefill_vit"),
      _wg(1, [2, 3], nodes=("LLM_cfg_text",), walk="image_gen_cfg"),
      _wg(2, [4, 5], nodes=("LLM_cfg_img",), walk="image_gen_cfg")],
     {"LLM", "LLM_cfg_text", "LLM_cfg_img"}, ("prefill_text", "prefill_vit"), [(0, 1)]),
    # the same, with the CFG nodes in LLM's group
    ([_wg(0, [0, 1], nodes=("LLM", "LLM_cfg_text"), walk="prefill_text"),
      _wg(0, [0, 1], nodes=("LLM_cfg_text",), walk="image_gen_cfg")],
     {"LLM", "LLM_cfg_text"}, ("prefill_text",), [(0, 1)]),
    # keyed walks on two of the spec's nodes, in groups on other ranks: a mirror on each
    ([_wg(0, [0, 1], nodes=("X",), walk="prefill_text"), _wg(1, [2, 3], nodes=("Y",), walk="prefill_vit")],
     {"X", "Y"}, ("prefill_text", "prefill_vit"), [(0, 1), (2, 3)]),
    # no walk keys the cache: no mirror to keep
    ([_wg(0, [0, 1], nodes=("LLM",), walk="prefill")], {"LLM"}, (), []),
], ids=["bagel_cfg_parallel", "cfg_nodes_in_the_llm_group", "keyed_walks_on_two_nodes", "unkeyed"])
def test_a_cache_has_a_mirror_where_a_walk_that_keys_it_runs(worker_graphs, nodes, walks, where):
    spec = KVSpec(resource_key="kv", nodes=nodes, config=_kv_spec().config)
    every_walk = set().union(*(wg.graph_walks for wg in worker_graphs))
    model = _Model(
        keyed={"kv": {"main": walks}} if walks else {},
        partitions=[SimpleNamespace(name="default", graph_walks=every_walk)],
    )
    router = ReplicaRouter(model, worker_graphs, specs=[spec])
    assert [unit.instances[0] for unit in router._units for _ in unit.mirrors] == [(rank,) for rank, _ in where]
    if not where:
        return
    groups_run = router.groups_run(["default"])

    def place(prefix):
        configs = _configs(_prompt(prefix))
        placement = router.pick(configs, groups_run)
        return placement.replicas[0], router.add(placement, configs, groups_run)

    # as _place_a, then one request on the other replica, so the loads are equal
    _, x = place("X")
    a, _ = place("P")
    router.remove(x, ["default"])
    assert place("Q")[0] != a
    assert place("P")[0] == a, "a request did not follow its prefix to the replica its node cached it on"


# ── the KV mirror beside the real cache ──────────────────────────────────


class _NoTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        pass

    def remove_request(self, request_id):
        pass

    def cleanup(self):
        pass


def test_the_kv_mirror_predicts_the_hits_the_real_cache_serves(monkeypatch):
    """One replica's ``KVManager`` and its mirror over the same requests: every
    prediction is the hit the cache serves. Each prompt is four keyed pages and a
    one-token tail page, in room for two, so ends and evictions interleave: the
    index evicts its oldest stamp first (a hit or an insert, not an end), and
    counts the pages no key names."""
    monkeypatch.setattr(manager_mod, "KVTransferManager", _NoTransfer)
    page, node, walk = 16, "LLM", "prefill"
    config = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, max_num_pages=2 * 5 + 1, page_size=page,
    )
    kv = KVManager(
        cfg=config, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(b"a root")
    mirror = KVSpec(resource_key="kv", nodes={node}, config=config).routing_mirror(
        RoutingContext(1, 1, frozenset({walk}), {"main": (walk,)}, True),
    )

    def request_config(first):
        tokens = list(range(first, first + 4 * page + 1))
        return tokens, KVReqConfig(
            # as the data worker keys a prompt: every page, the partial last one included
            prefix_keys={"main": chain([tokens[at:at + page] for at in range(0, len(tokens), page)])},
            prefix_tail={"main": tokens[4 * page:]},
            kv_layout={"main": StreamLayout(spans=(PrefixSpan(len(tokens), len(tokens), walk),), keyed_spans=1)},
        )

    live, predicted, served = {}, [], []
    for event, rid, first in [
        ("add", "a", 0), ("add", "b", 1000), ("end", "b", 0), ("end", "a", 0), ("add", "c", 2000),
        ("end", "c", 0), ("add", "a2", 0), ("end", "a2", 0), ("add", "b2", 1000), ("end", "b2", 0),
        ("add", "c2", 2000),
    ]:
        if event == "end":
            kv.remove_request(rid)
            mirror.remove_request(live.pop(rid), True)
            continue
        tokens, cfg = request_config(first)
        (facts,) = mirror.facts(cfg)
        predicted.append(facts.kv_hit)
        live[rid] = mirror.add_request(cfg, 0)
        kv.ingest_request(rid, request_config(first)[1])
        hit = kv.resolve_cached_prefix(rid, node, walk, ARNodeInputs(input_seq_len=len(tokens)))
        served.append(hit.tokens if hit else 0)
        # the walk runs what the cache does not hold
        step = KVStep(segments=(Segment(rid, "main", len(tokens) - served[-1]),))
        ctx = StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)
        assert kv.admit(step, ctx).ok
        kv.plan(step, ctx)
        kv.commit(step, ctx)
    assert any(served), "no request hit, so the sequence pins nothing"
    assert predicted == served
