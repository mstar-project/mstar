"""``validate_deployment`` reports every problem a deployment has, in the
conductor, before it spawns a worker.

One test per check that ``test_startup_graph_validation.py`` does not already
cover (walk placement, stream-driven walks and Loop successors are there),
including the ones that used to fail on every worker after spawn, and two
that several problems come back together. The input limit is checked
against the Rust graph runtime's own compiler.
"""
import re
from dataclasses import dataclass, field

import pytest
import yaml
from test_graph_runtime import _sharding_config
from test_startup_graph_validation import (
    _loop_feeding_post,
    _prefill_decode,
    _start_conductor,
    _stream_model,
    _ToyModel,
    _validate,
    _vocoder,
)

from mstar.conductor.conductor import Conductor
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.engine.resources.base import EngineResourceInfo, Resource, build_resource
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.graph.base import GraphEdge, GraphNode, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import KVLabel, PrefixKeys, WorkerGraph
from mstar.validation import DeploymentError
from mstar.validation.placement import MAX_NODE_INPUTS, _sharding_groups


class _Stub(Resource):
    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError


@dataclass
class _Spec(NodeResourceSpec):
    deps: set[str] = field(default_factory=set)

    def depends_on(self):
        return set(self.deps)

    @property
    def resource_class(self):
        return _Stub


class _Model(_ToyModel):
    """A toy model with resources, KV labels and a YAML hook to taste."""

    def __init__(self, walks=_prefill_decode, specs=(), kv_labels=None, refuse_yaml=None,
                 sp_enabled_nodes=(), **kw):
        super().__init__(walks, **kw)
        self._specs = list(specs)
        self._kv_labels = kv_labels or {}
        self._refuse_yaml = refuse_yaml
        self._sp_enabled_nodes = set(sp_enabled_nodes)

    def get_default_sharding_config(self):
        sharding = super().get_default_sharding_config()
        sharding.sp_enabled_nodes = self._sp_enabled_nodes
        return sharding

    def get_node_resources(self):
        return list(self._specs)

    def kv_labels(self):
        return self._kv_labels

    def validate_config_yaml(self, config, config_path):
        if self._refuse_yaml:
            raise ValueError(self._refuse_yaml)


def _cfg(tmp_path, node_groups=None, **extra) -> str:
    if node_groups is None:
        node_groups = [{"node_names": ["llm"], "ranks": [0]}]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"max_seq_len": 64, "node_groups": node_groups, **extra}))
    return str(path)


def _problems(model, cfg) -> list[str]:
    with pytest.raises(DeploymentError) as refusal:
        _validate(model, cfg)
    return refusal.value.problems


def _refused(model, cfg, pattern):
    """Exactly one problem, matching ``pattern``."""
    problems = _problems(model, cfg)
    assert len(problems) == 1 and re.search(pattern, problems[0]), problems


def _a_then_b():
    return {"w": Sequential([
        GraphNode(name="a", input_names=["x"], outputs=[GraphEdge(next_node="b", name="h")]),
        GraphNode(name="b", input_names=["h"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="y", output_modality="text"),
        ]),
    ])}


# -- environment knobs --------------------------------------------------------


@pytest.mark.parametrize("env,pattern", [
    ({"MSTAR_RUST_GRAPH": "2"}, r"MSTAR_RUST_GRAPH must be 0, 1, or AUTO; got '2'"),
    ({"MSTAR_RUST_ZMQ": "yes"}, r"MSTAR_RUST_ZMQ must be 0, 1, or AUTO; got 'YES'"),
    ({"MSTAR_WIRE_CODEC": "pickle"}, r"MSTAR_WIRE_CODEC must be msgpack; got 'pickle'"),
    ({"MSTAR_RUST_GRAPH": "1", "MSTAR_RUST_ZMQ": "0"}, r"MSTAR_RUST_GRAPH=1 needs the Rust communicator"),
], ids=["graph_runtime", "transport", "codec", "rust_graph_on_pyzmq"])
def test_bad_runtime_knob_is_refused_in_the_conductor(env, pattern, tmp_path, monkeypatch):
    """Each worker read these as it started and raised there, once per rank."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(DeploymentError, match=pattern):
        _start_conductor(_Model(), _cfg(tmp_path), tmp_path, monkeypatch)


# -- the YAML's own keys ------------------------------------------------------


def test_missing_max_seq_len_is_refused(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("node_groups: [{node_names: [llm], ranks: [0]}]\n")
    _refused(_Model(), str(cfg), r"must define `max_seq_len`")


def test_mps_that_is_not_a_bool_is_refused(tmp_path):
    _refused(_Model(), _cfg(tmp_path, mps="yes"), r"mps must be true or false, got 'yes'")


def test_missing_node_groups_is_refused(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("max_seq_len: 64\n")
    _refused(_Model(), str(cfg), r"must define `node_groups`")


@pytest.mark.parametrize("node_groups,pattern", [
    ([{"node_names": ["llm"]}], r"node_groups\[0\] needs a `node_names` list and a non-empty `ranks` list"),
    ([{"node_names": ["llm"], "ranks": []}], r"node_groups\[0\] needs a `node_names` list and a non-empty"),
    ([{"node_names": ["llm"], "ranks": ["0"]}], r"node_groups\[0\]: ranks must be integers >= 0; got \['0'\]"),
    ([{"node_names": ["llm"], "ranks": [0], "tp_size": 0}], r"node_groups\[0\]: tp_size must be an integer >= 1"),
    ([{"node_names": ["llm"], "ranks": [0], "graph_walks": "decode"}], r"graph_walks must be a list of names"),
    ([{"node_names": ["llm"], "ranks": [0], "graph_walks": [["decode"]]}],
     r"node_groups\[0\]: graph_walks must be a list of names; got \[\['decode'\]\]"),
    ([{"node_names": ["llm", []], "ranks": [0]}], r"node_groups\[0\]: node_names must be a list of names"),
    ({"node_names": ["llm"], "ranks": [0]}, r"`node_groups` must be a list of entries"),
], ids=["no_ranks", "empty_ranks", "string_rank", "zero_tp", "walks_not_a_list", "nested_walk",
        "nested_node_name", "not_a_list"])
def test_unreadable_node_groups_are_reported_with_the_rest(node_groups, pattern, tmp_path):
    """Reported beside the other problems rather than crashing the pass."""
    problems = _problems(_Model(refuse_yaml="hook"), _cfg(tmp_path, node_groups))
    assert problems[0] == "hook" and len(problems) == 2 and re.search(pattern, problems[1]), problems


class _ReadsResources(_Model):
    """A YAML hook that reads ``resources:`` as a mapping, as Waypoint's does."""

    def validate_config_yaml(self, config, config_path):
        (config.get("resources") or {}).get("kv")


@pytest.mark.parametrize("resources,shown", [
    (["kv"], "['kv']"), ({1: {}, "typo": {}}, "{1: {}, 'typo': {}}"),
], ids=["list", "int_key"])
def test_resources_block_that_is_not_keyed_by_name_is_reported_with_the_rest(resources, shown, tmp_path):
    """The model's hook is skipped rather than crashing on it."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"node_groups": [{"node_names": ["llm"], "ranks": [0]}], "resources": resources}))
    assert _problems(_ReadsResources(specs=[_Spec("kv", {"llm"})]), str(cfg)) == [
        "the config must define `max_seq_len`",
        f"`resources:` must map resource keys to their overrides; got {shown}",
    ]


def test_model_hook_that_cannot_read_a_resources_block_is_reported_with_the_rest(tmp_path):
    """Waypoint's hook reads each block as a mapping."""
    class _ReadsBlocks(_Model):
        def validate_config_yaml(self, config, config_path):
            config["resources"]["kv"].get("num_sessions")

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "node_groups": [{"node_names": ["llm"], "ranks": [0]}], "resources": {"kv": ["num_sessions", 2]},
    }))
    problems = _problems(_ReadsBlocks(specs=[_Spec("kv", {"llm"})]), str(cfg))
    assert problems[:2] == [
        "the config must define `max_seq_len`",
        "validate_config_yaml could not read the config: AttributeError: 'list' object has no attribute 'get'",
    ], problems


def test_a_model_that_cannot_declare_its_resources_is_reported_with_the_rest(tmp_path):
    """LTX-2.5 reads checkpoint configs in ``get_node_resources``."""
    class _MissingCheckpoint(_Model):
        def get_node_resources(self):
            raise FileNotFoundError(2, "No such file or directory", "ckpt/vae/config.json")

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"node_groups": [{"node_names": ["llm"], "ranks": [0]}]}))
    problems = _problems(_MissingCheckpoint(), str(cfg))
    assert problems == [
        "the config must define `max_seq_len`",
        "declaring the node resources failed: FileNotFoundError: [Errno 2] No such file or directory: "
        "'ckpt/vae/config.json'; the `resources:` checks, which need the declarations, were skipped",
    ], problems


def test_a_checkpoint_read_failing_after_the_declarations_is_reported_with_the_rest(tmp_path):
    """FLUX.2 klein with a gated repo: the resource checks call model hooks
    (``kv_labels``, ``get_graph_walk_graphs``) that read the checkpoint."""
    class _GatedCheckpoint(_Model):
        def kv_labels(self):
            raise OSError("401 Client Error: access to model ckpt is restricted")

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"node_groups": [{"node_names": ["llm"], "ranks": [0]}]}))
    assert _problems(_GatedCheckpoint(), str(cfg)) == [
        "the config must define `max_seq_len`",
        "checking the node resources failed: OSError: 401 Client Error: access to model ckpt is restricted",
    ]


def test_worker_graph_checks_still_run_when_the_declarations_fail(tmp_path):
    """Only the checks that need the specs are skipped."""
    class _MissingCheckpoint(_Model):
        def get_node_resources(self):
            raise FileNotFoundError(2, "No such file or directory", "ckpt/vae/config.json")

    model = _MissingCheckpoint(lambda: {"w": _loop_feeding_post(accumulate=False)})
    problems = _problems(model, _cfg(tmp_path, [{"node_names": ["step", "post"], "ranks": [0]}]))
    assert len(problems) == 2, problems
    assert problems[0].startswith("declaring the node resources failed: FileNotFoundError")
    assert problems[0].endswith("the `resources:` checks, which need the declarations, were skipped")
    assert "node 'step' of Loop 'loop' sends 'y' to 'post'" in problems[1]


def test_unreadable_sharding_config_is_reported_with_the_rest(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "node_groups": [{"node_names": ["llm"], "ranks": [0]}], "sharding_config": {"shard_dim": 1},
    }))
    problems = _problems(_Model(), str(cfg))
    assert len(problems) == 2 and problems[0] == "the config must define `max_seq_len`", problems
    assert problems[1].startswith("reading node_groups and sharding_config could not read the config: TypeError")


def test_config_that_is_not_a_mapping_is_refused(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("")
    _refused(_Model(), str(cfg), r"the config must be a YAML mapping; got None")


def test_ranks_not_divisible_by_the_instance_size_are_refused(tmp_path):
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [0, 1, 2], "tp_size": 2}])
    _refused(_Model(tp_enabled_nodes=["llm"]), cfg,
             r"node_groups\[0\] \(nodes \['llm'\]\) has 3 ranks; not divisible by tp_size\*sp_size \(2\*1=2\)")


@pytest.mark.parametrize("axis", ["tp", "sp"])
def test_parallel_node_group_over_a_node_without_support_is_refused(axis, tmp_path):
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [0, 1], f"{axis}_size": 2}])
    _refused(_Model(), cfg, rf"node_groups\[0\] has {axis}_size 2 and contains nodes not in {axis}_enabled_nodes")


def test_model_yaml_hook_refusal_is_reported(tmp_path):
    _refused(_Model(refuse_yaml="max_concurrent_requests too high"), _cfg(tmp_path),
             r"max_concurrent_requests too high")


# -- graph shape --------------------------------------------------------------


def _wide(n):
    return lambda: {"w": GraphNode(name="wide", input_names=[f"x{i}" for i in range(n)], outputs=[
        GraphEdge(next_node=EMIT_TO_CLIENT, name="y", output_modality="text"),
    ])}


@pytest.mark.parametrize("n", [MAX_NODE_INPUTS, MAX_NODE_INPUTS + 1])
def test_input_limit_matches_the_rust_runtime(n, tmp_path):
    """Both runtimes take the same nodes: the Python one used to accept a node
    the Rust one refused when a worker compiled its graph."""
    model = _Model(_wide(n))
    cfg = _cfg(tmp_path, [{"node_names": ["wide"], "ranks": [0]}])
    if n > MAX_NODE_INPUTS:
        _refused(model, cfg, rf"node 'wide' of graph walk 'w' has {n} inputs; a node takes at most 64")
    else:
        _validate(model, cfg)

    pytest.importorskip("mstar_rust")
    from mstar.communication.tensor_store import RustTensorBookkeeping
    from mstar.graph.runtime.rust import RustGraphRuntime
    wg, = model.get_worker_graphs(cfg)
    wg.worker_graph_id = 0

    def compile_in_rust():
        RustGraphRuntime(
            bookkeeping=RustTensorBookkeeping(), my_worker_id="worker_0", my_worker_graphs=[wg],
            all_wg_ids_to_graph_walks={0: {"w"}}, all_wg_ids_to_dyn_loops={0: set()},
            all_wg_ids_to_nodes={0: {"wide"}}, node_to_partition={"wide": "default"},
            sharding_config=_sharding_config(),
        )
    if n > MAX_NODE_INPUTS:
        with pytest.raises(ValueError, match=f"has {n} inputs; readiness mask holds {MAX_NODE_INPUTS}"):
            compile_in_rust()
    else:
        compile_in_rust()


# -- sharding groups and parallel workers -------------------------------------


def test_streaming_consumer_in_a_sharding_group_with_custom_walks_is_refused(tmp_path):
    cfg = _cfg(tmp_path, [
        {"node_names": ["producer"], "ranks": [0]},
        {"node_names": ["vocoder"], "ranks": [1, 2], "tp_size": 2, "graph_walks": ["chunk"]},
    ])
    _refused(_stream_model(_vocoder, tp_enabled_nodes=["vocoder"]), cfg,
             r"Sharding group with nodes \['vocoder'\] includes a streaming consumer but has "
             r"custom graph walk configuration \['chunk'\]")


def _wg(nodes):
    section = Sequential([GraphNode(name=n, input_names=["x"], outputs=[]) for n in nodes])
    return WorkerGraph(section=section, graph_walks={"w"}, ranks=[0, 1], tp_size=2)


def test_worker_graph_claimed_by_two_sharding_groups_is_refused():
    """``node_groups`` puts all of a worker graph's nodes in one group, so
    only a hand-built sharding config reaches this."""
    groups = [ShardingGroup(nodes={"a"}, tp_size=2), ShardingGroup(nodes={"b"}, tp_size=2)]
    assert list(_sharding_groups([_wg(["a", "b"])], ShardingConfig(groups, set(), {}))) == [
        "the worker graph of ['w'] with nodes ['a', 'b'] on ranks [0, 1] is claimed by two sharding "
        "groups ('a' and 'b'). v1 requires one TP group per worker graph; split the worker graph "
        "or merge the groups."
    ]


def test_worker_graph_and_sharding_group_tp_mismatch_is_refused():
    groups = [ShardingGroup(nodes={"a"}, tp_size=4)]
    assert list(_sharding_groups([_wg(["a"])], ShardingConfig(groups, set(), {}))) == [
        "the worker graph of ['w'] with nodes ['a'] on ranks [0, 1] has tp_size 2, but its "
        "sharding group has tp_size 4. node_groups and sharding_config disagree."
    ]


_TP2 = {"ranks": [0, 1], "tp_size": 2}


def _kv(key, nodes):
    return KVSpec(resource_key=key, nodes=set(nodes), config=PagedKVConfig(
        num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=64,
    ))


@pytest.mark.parametrize("groups", [
    [{"node_names": ["a", "b"], **_TP2}],
    [{"node_names": ["a"], **_TP2}, {"node_names": ["b"], **_TP2}],
    # two replicas each: every worker is in the same block for both nodes
    [{"node_names": ["a"], "ranks": [0, 1, 2, 3], "tp_size": 2},
     {"node_names": ["b"], "ranks": [0, 1, 2, 3], "tp_size": 2}],
], ids=["one_group", "two_groups", "replicas"])
def test_lockstep_nodes_sharing_one_instance_are_accepted(groups, tmp_path, monkeypatch):
    """One leader and one step stream. Each node has a KV pool of its own."""
    model = _Model(
        _a_then_b, specs=[_kv("kv_a", ["a"]), _kv("kv_b", ["b"])],
        kv_labels={"kv_a": {"main": KVLabel()}, "kv_b": {"main": KVLabel()}}, tp_enabled_nodes=["a", "b"],
    )
    _start_conductor(model, _cfg(tmp_path, groups), tmp_path, monkeypatch)


@pytest.mark.parametrize("offload", [0, 4])
def test_lockstep_nodes_sharing_a_kv_resource_are_refused(offload, tmp_path):
    """Rank 0's page moves on a pool reach the followers per node, riding on
    one node's next step, so two lockstep nodes over one pool could admit out
    of step. The layout is refused at any offload setting until rank 0
    journals a shared pool in one stream."""
    model = _Model(
        _a_then_b, specs=[_kv("kv", ["a", "b"])], kv_labels={"kv": {"main": KVLabel()}}, tp_enabled_nodes=["a", "b"],
    )
    cfg = _cfg(tmp_path, [{"node_names": ["a", "b"], **_TP2}], resources={"kv": {"cpu_offload_pages": offload}})
    [problem] = _problems(model, cfg)
    assert problem.startswith("Resource kv is shared by lockstep-parallel nodes ['a', 'b'] on worker_0, worker_1;")


class _Trail(_Stub):
    """A resource whose page trail the followers check against rank 0's."""

    @classmethod
    def build(cls, spec, info):
        return cls()

    def page_trail(self):
        return (0, 0)


class _TrailSpec(_Spec):
    @property
    def resource_class(self):
        return _Trail


def test_a_resource_rank_0_decides_for_followers_needs_a_spec_that_says_so():
    """Validation refuses lockstep nodes sharing a resource only by its spec's
    journals_resident_moves, so a resource rank 0 decides for its followers
    can't be built from a spec that leaves it False."""
    with pytest.raises(ValueError, match="_TrailSpec.journals_resident_moves is False"):
        build_resource(_TrailSpec("trail", {"a"}), EngineResourceInfo(device="cpu"))


@pytest.mark.parametrize("b, placed", [
    ({"ranks": [1, 0], "tp_size": 2}, "TP ranks [1, 0]"),  # the same ranks, led by the other one
    ({"ranks": [0, 1, 2, 3], "tp_size": 4}, "TP ranks [0, 1, 2, 3]"),
    ({"ranks": [0, 1], "sp_size": 2}, "SP ranks [0, 1]"),  # the same ranks, the other axis
], ids=["other_leader", "other_size", "other_axis"])
def test_lockstep_nodes_in_different_instances_on_one_worker_are_refused(b, placed, tmp_path):
    """Two instances on a worker are two leaders' step streams with no order
    between them."""
    cfg = _cfg(tmp_path, [{"node_names": ["a"], **_TP2}, {"node_names": ["b"], **b}])
    model = _Model(_a_then_b, tp_enabled_nodes=["a", "b"], sp_enabled_nodes=["b"])
    problems = _problems(model, cfg)
    assert len(problems) == 2
    for rank, problem in enumerate(problems):
        assert f"on worker_{rank}" in problem and f"a on TP ranks [0, 1]; b on {placed}" in problem


def test_parallel_ranks_off_their_position_are_refused(tmp_path):
    """The conductor indexes workers by rank, so ranks [1, 2] mapped rank 1 to
    worker_2 and rank 2 off the end."""
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [1, 2], "tp_size": 2}])
    _refused(_Model(tp_enabled_nodes=["llm"]), cfg,
             r"the deployment uses ranks \[1, 2\]; ranks \[1, 2\] are in tensor- or sequence-parallel groups")


def test_an_unused_rank_above_the_parallel_ones_is_accepted(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, [
        {"node_names": ["a"], "ranks": [0, 1], "tp_size": 2},
        {"node_names": ["b"], "ranks": [3]},
    ])
    conductor = _start_conductor(_Model(_a_then_b, tp_enabled_nodes=["a"]), cfg, tmp_path, monkeypatch)
    assert conductor.worker_ids == ["worker_0", "worker_1", "worker_3"]


def test_node_in_two_parallel_groups_on_one_worker_is_refused(tmp_path):
    """The conductor raised this from GlobalParallelConfig, after the checks."""
    cfg = _cfg(tmp_path, [
        {"node_names": ["llm"], "ranks": [0, 1], "tp_size": 2, "graph_walks": ["prefill"]},
        {"node_names": ["llm"], "ranks": [0, 2], "tp_size": 2, "graph_walks": ["decode"]},
    ])
    _refused(_Model(tp_enabled_nodes=["llm"]), cfg, r"Node llm already has a comm group assigned for worker 0")


def test_resource_over_nodes_of_different_parallel_shapes_is_refused(tmp_path):
    """Engine.load_model raised this on the worker, after building the submodules."""
    cfg = _cfg(tmp_path, [
        {"node_names": ["a"], "ranks": [0, 1], "tp_size": 2},
        {"node_names": ["b"], "ranks": [2]},
    ])
    model = _Model(_a_then_b, specs=[_Spec("kv", {"a", "b"})], tp_enabled_nodes=["a"])
    _refused(model, cfg, r"Resource spec kv nodes \['a', 'b'\] must use the same TP x SP shape across replicas")


def test_resource_over_local_nodes_in_different_parallel_groups_is_refused(tmp_path):
    cfg = _cfg(tmp_path, [
        {"node_names": ["a"], "ranks": [0, 1], "tp_size": 2},
        {"node_names": ["b"], "ranks": [0, 2], "tp_size": 2},
    ])
    model = _Model(_a_then_b, specs=[_Spec("kv", {"a", "b"})], tp_enabled_nodes=["a", "b"])
    assert (
        "Resource spec kv local nodes ['a', 'b'] on worker_0 must all be in the same parallel (tp x sp) group"
        in _problems(model, cfg)
    )


# -- resources ----------------------------------------------------------------


@pytest.mark.parametrize("specs,extra,pattern", [
    ([_Spec("kv", {"llm"})], {"kv_cache": {"max_num_pages": 8}}, r"top-level `kv_cache:` .* no longer read"),
    ([_Spec("kv", {"llm"})], {"resources": {"vk": {}}},
     r"overrides unknown resource\(s\) \['vk'\]; this model declares \['kv'\]"),
    ([_Spec("kv", {"llm"})], {"resources": {"kv": {"pages": 8}}},
     r"resources.kv: resource 'kv' takes no YAML overrides; got \['pages'\]"),
    ([_Spec("kv", {"llm"}), _Spec("kv", {"llm"})], {}, r"two resources declared under the key 'kv'"),
    ([_Spec("attn", {"llm"}, deps={"kv"})], {},
     r"resource 'attn' depends on \['kv'\], which this model does not declare"),
    ([_Spec("kv", {"llm"}, deps={"kv"})], {}, r"resource 'kv' depends on itself"),
    ([_Spec("a", {"llm"}, deps={"b"}), _Spec("b", {"llm"}, deps={"a"})], {},
     r"dependency cycle among resources \['a', 'b'\]"),
], ids=["top_level_kv_cache", "unknown_resource", "bad_override", "duplicate_key", "missing_dependency",
        "self_dependency", "cycle"])
def test_resource_problem_is_refused_in_the_conductor(specs, extra, pattern, tmp_path, monkeypatch):
    """EngineManager.build and Engine.load_model raised these on every worker."""
    with pytest.raises(DeploymentError, match=pattern):
        _start_conductor(_Model(specs=specs), _cfg(tmp_path, **extra), tmp_path, monkeypatch)


def test_prefix_stream_naming_an_unknown_walk_is_refused_in_the_conductor(tmp_path, monkeypatch):
    """EngineManager.build ran the prefix-cache rules (each has its own test in
    test_kv_prefix_positions.py)."""
    model = _Model(kv_labels={"kv": {"main": KVLabel(keys=PrefixKeys("text_inputs", ("prefil", "decode")))}})
    with pytest.raises(DeploymentError, match=r"_Model declares 'kv'/'main' over \['prefil'\], which it never runs"):
        _start_conductor(model, _cfg(tmp_path), tmp_path, monkeypatch)


def test_prefix_stream_laying_out_over_an_unknown_walk_is_refused_in_the_conductor(tmp_path, monkeypatch):
    label = KVLabel(keys=PrefixKeys("text_inputs", ("prefill", "prefill_imag")))
    model = _Model(specs=[_Spec("kv", {"llm"})], kv_labels={"kv": {"main": label}})
    unknown = r"_Model declares 'kv'/'main' over \['prefill_imag'\], which it never runs"
    with pytest.raises(DeploymentError, match=unknown):
        _start_conductor(model, _cfg(tmp_path), tmp_path, monkeypatch)


def test_prefix_stream_with_a_walk_the_cache_cannot_serve_whole_is_refused_in_the_conductor(tmp_path, monkeypatch):
    """EngineManager.build refused the first of these on every worker (each
    case has its own test in test_kv_prefix_root.py)."""
    label = KVLabel(keys=PrefixKeys("text_inputs", ("prefill", "decode")))
    model = _Model(specs=[_Spec("kv", {"talker"})], kv_labels={"kv": {"main": label}})
    with pytest.raises(DeploymentError) as refusal:
        _start_conductor(model, _cfg(tmp_path), tmp_path, monkeypatch)
    assert [problem for problem in refusal.value.problems if " lays " in problem] == [
        f"_Model lays 'kv'/'main' out over {walk!r}, where none of ['talker'] runs"
        for walk in ("prefill", "decode")
    ]


def test_prefix_stream_laying_out_an_undeclared_resource_is_refused(tmp_path):
    """EngineManager.build hit a KeyError here."""
    label = KVLabel(keys=PrefixKeys("text_inputs", ("prefill", "decode")))
    model = _Model(specs=[_Spec("kv", {"llm"})], kv_labels={"vk": {"main": label}})
    assert (
        "_Model lays 'vk'/'main' out, but declares no resource 'vk'; its resources are ['kv']"
        in _problems(model, _cfg(tmp_path))
    )


# -- every problem at once ----------------------------------------------------


def test_client_stream_settings_are_reported_with_the_rest(tmp_path):
    """The API server reads these, but the conductor reports them."""
    cfg = _cfg(tmp_path, client_streams={"audio_in": {"max_items": 3}}, max_call_input_mb=0)
    assert _problems(_Model(refuse_yaml="no admit queue"), cfg) == [
        "no admit queue",
        "max_call_input_mb must be a positive number of MiB, not 0",
        "serving config limits unknown client stream(s) ['audio_in']; this model takes []",
    ]


def test_problems_from_every_stage_are_reported_together(tmp_path, monkeypatch):
    """A knob, a YAML key, the model's hook, placement and a resource override
    come back in one error, and no worker is started."""
    monkeypatch.setenv("MSTAR_RUST_GRAPH", "on")
    launched = []

    def launch(conductor):
        launched.append(conductor)
    monkeypatch.setattr(Conductor, "_launch_workers", launch)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "node_groups: [{node_names: [llm], ranks: [0], graph_walks: [prefill]}]\n"
        "resources: {vk: {}}\n"
    )
    model = _Model(specs=[_Spec("kv", {"llm"})], refuse_yaml="no admit queue")
    with pytest.raises(DeploymentError) as refusal:
        Conductor(model, str(cfg), socket_path_prefix=str(tmp_path / "s"))
    assert refusal.value.problems == [
        "MSTAR_RUST_GRAPH must be 0, 1, or AUTO; got 'on'",
        "the config must define `max_seq_len`",
        "no admit queue",
        "node 'llm' of graph walk 'decode' is placed by no node_groups entry; place it in exactly "
        "one node_groups entry for that walk (use graph_walks to give one node different ranks per walk).",
        "serving config overrides unknown resource(s) ['vk']; this model declares ['kv']",
    ]
    assert str(refusal.value).startswith(f"{cfg}: 5 problems; no worker was started:\n  - MSTAR_RUST_GRAPH")
    assert not launched


def test_worker_graph_problems_are_reported_together(tmp_path):
    """Once placement holds, the graph shape, parallelism and resource checks
    all run: a Loop feeding its successor, a resource over two parallel
    shapes, and an unknown override."""
    def walks():
        loop = _loop_feeding_post(accumulate=False).sections[0]  # step -> post from inside the Loop
        return {"w": Sequential([
            loop,
            GraphNode(name="post", input_names={"y"}, outputs=[GraphEdge(next_node="llm", name="x")]),
            GraphNode(name="llm", input_names={"x"}, outputs=[
                GraphEdge(next_node=EMIT_TO_CLIENT, name="out", output_modality="text"),
            ]),
        ])}

    cfg = _cfg(tmp_path, [
        {"node_names": ["step", "post"], "ranks": [0]},
        {"node_names": ["llm"], "ranks": [1, 2], "tp_size": 2},
    ], resources={"vk": {}})
    model = _Model(walks, specs=[_Spec("kv", {"post", "llm"})], tp_enabled_nodes=["llm"])
    problems = _problems(model, cfg)
    assert len(problems) == 3, problems
    assert problems[0] == "serving config overrides unknown resource(s) ['vk']; this model declares ['kv']"
    assert problems[1].startswith("graph walk 'w': node 'step' of Loop 'loop' sends 'y' to 'post'")
    assert problems[2] == (
        "Resource spec kv nodes ['llm', 'post'] must use the same TP x SP shape across replicas"
    )
