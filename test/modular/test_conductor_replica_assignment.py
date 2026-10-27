"""Data-parallel replica choice: a new request goes to the replica where its cost
is lowest, the live requests on the replica plus what each
resource mirror predicts (a KV cache: minus the share of the prompt it already
holds). With no mirror that is the replica serving the fewest live requests, so
concurrent requests spread over the replicas instead of landing on one at random."""
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.conductor.routing import ReplicaRouter  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.routing import KVRoutingMirror  # noqa: E402
from mstar.model.base import WorkerGraph  # noqa: E402

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


def _conductor(worker_graphs, specs=()):
    conductor = object.__new__(Conductor)
    conductor.worker_graphs = worker_graphs
    conductor.requests = {}
    conductor._router = ReplicaRouter(
        worker_graphs.values(), [SimpleNamespace(name="default", graph_walks={"prefill"})],
        specs=specs, checkpoint_named=lambda: True,
    )
    return conductor


def _configs(model_kwargs: dict) -> dict:
    """The request's resource configs, opened as ``_do_ingest_request`` opens them."""
    if "prefix_keys" not in model_kwargs:
        return {}
    cfg = KVReqConfig(needed_labels=["main"])
    cfg.apply_conductor_config(prefix_keys=model_kwargs["prefix_keys"]["kv"], kv_layout=model_kwargs["kv_layout"]["kv"])
    return {"kv": cfg}


def _admit(conductor, rid, model_kwargs=None):
    """What ``_do_ingest_request`` does for the pick, and nothing else."""
    configs = _configs(model_kwargs or {})
    router = conductor._router
    groups_run = router.groups_run(["default"])
    placement = router.pick(configs, groups_run)
    conductor._register_request(rid, SimpleNamespace(routing=router.add(placement, configs, groups_run)))
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


def _kv_conductor():
    spec = _kv_spec()
    conductor = _conductor({0: _wg(0, [0, 1])}, specs=[spec])
    conductor._resource_specs_cache = {"kv": spec}
    return conductor


def _prompt(prefix: str, pages: int = 4, unkeyed: int = 0) -> dict:
    """The kwargs of a request whose prompt is ``pages`` whole pages keyed under
    ``prefix``, then ``unkeyed`` positions the data worker leaves unkeyed."""
    spans = [(pages * PAGE, pages * PAGE, "prefill")] + [(unkeyed, unkeyed, "prefill")] * bool(unkeyed)
    return {
        "prefix_keys": {"kv": {"main": [f"{prefix}{i}".encode() for i in range(pages)]}},
        "kv_layout": {"kv": {"main": (spans, (), 0, None, 1)}},
    }


def _replica(conductor, rid, model_kwargs) -> str:
    return _admit(conductor, rid, model_kwargs)[0][0]


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


def test_a_hot_prefix_spreads_once_load_outweighs_its_hit():
    conductor = _kv_conductor()
    a = _place_a(conductor)
    _replica(conductor, "b", _prompt("Q"))
    assert _replica(conductor, "c", _prompt("P")) == a
    # replica a now runs one more request than the other, which outweighs a
    # hit on 3 of 4 pages at the default weights
    assert _replica(conductor, "d", _prompt("P")) != a


def test_concurrent_copies_of_one_prompt_spread_over_the_replicas():
    # as the startup warmup sends one copy per replica, before any completes:
    # a hit never outweighs a live request at the default weights
    conductor = _kv_conductor()
    picks = [_replica(conductor, rid, _prompt("P")) for rid in "abcd"]
    assert sorted(picks) == ["worker_0", "worker_0", "worker_1", "worker_1"]


def test_a_hit_is_scored_as_its_share_of_the_whole_prompt():
    conductor = _kv_conductor()
    # as BAGEL lays out a generation: a keyed text span, then image positions the data worker leaves unkeyed
    kwargs = _prompt("P", pages=2, unkeyed=4000)
    _admit(conductor, "a", kwargs)
    (mirror,) = [m.mirror for unit in conductor._router._units for m in unit.mirrors]
    # the worker matches one of the two keyed pages, which saves PAGE of 2 * PAGE + 4000 positions
    assert sorted(mirror.score(_configs(kwargs)["kv"])) == [pytest.approx(-PAGE / (2 * PAGE + 4000)), 0.0]


def test_a_dropped_request_releases_its_reservation_once_and_its_pages_stay_cached():
    conductor = _kv_conductor()
    a = _place_a(conductor)
    # dropped twice, as an abort racing a completion can
    conductor._drop_request("a")
    conductor._drop_request("a")
    assert not any(conductor._router._load.values())
    (mirror,) = [m.mirror for unit in conductor._router._units for m in unit.mirrors]
    assert all(not pages.held for pages in mirror._replicas), "a dropped request still holds its pages"
    # with both replicas idle, the prefix it left cached decides
    assert _replica(conductor, "b", _prompt("P")) == a


def test_a_request_that_runs_none_of_a_resources_nodes_takes_no_part_in_its_mirror():
    # an encoder group and the LLM's on the same ranks pick together; the cache is the LLM's
    router = ReplicaRouter(
        [_wg(0, [0, 1], nodes=("enc",), walk="encode"), _wg(1, [0, 1])],
        [SimpleNamespace(name="encoder", graph_walks={"encode"}),
         SimpleNamespace(name="default", graph_walks={"prefill"})],
        specs=[_kv_spec()], checkpoint_named=lambda: True,
    )
    configs = _configs(_prompt("P"))
    groups_run = router.groups_run(["encoder"])
    router.add(router.pick(configs, groups_run), configs, groups_run)
    (mirror,) = [m.mirror for unit in router._units for m in unit.mirrors]
    assert not any(pages.held for pages in mirror._replicas), (
        "a request that runs only the encoder reserved pages in the LLM's cache"
    )


def test_a_request_with_two_labels_releases_each_chain_leaves_first():
    mirror = KVRoutingMirror(1, capacity=10, page_size=PAGE)
    chains = {label: [f"{label}{i}".encode() for i in range(4)] for label in ("a", "b")}
    two = KVReqConfig(needed_labels=["a", "b"])
    two.apply_conductor_config(prefix_keys=chains, kv_layout={
        label: ([(4 * PAGE, 4 * PAGE, "prefill")], (), 0, None, 1) for label in chains
    })
    mirror.remove_request(mirror.add_request(two, 0))
    # four more pages in a pool of ten: two of the eight idle ones go
    (fresh,) = _configs(_prompt("F")).values()
    mirror.add_request(fresh, 0)
    pages = mirror._replicas[0]
    assert [pages.matched(keys) for keys in chains.values()] == [3, 3], (
        "eviction took one label's prefix before the other label's leaves"
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
    (mirror,) = [m.mirror for unit in conductor._router._units for m in unit.mirrors]
    assert all(not pages.held and not pages.idle for pages in mirror._replicas), (
        "a request refused before dispatch left its prefix predicted cached"
    )
    assert not any(conductor._router._load.values())
