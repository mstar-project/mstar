"""``validate_deployment`` reports every problem a deployment has, in the
conductor, before it spawns a worker.

One test per check that ``test_startup_graph_validation.py`` does not already
cover (walk placement, stream-driven walks and Loop successors are there),
including the ones that used to fail on every worker after spawn, and two
that several problems come back together. The input limit is checked
against the Rust graph runtime's own compiler.
"""
import dataclasses
import re
from dataclasses import dataclass, field

import pytest
import yaml
from test_graph_runtime import _sharding_config
from test_startup_graph_validation import (
    ROOT,
    _loop_feeding_post,
    _prefill_decode,
    _qwen3_5,
    _start_conductor,
    _stream_model,
    _ToyModel,
    _validate,
    _vocoder,
)

from mstar.conductor.conductor import Conductor
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttnBackend
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedCrossAttentionSpec,
)
from mstar.engine.resources.kv.config import KVLayout, KVSpec, PagedKVConfig
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnSpec, LinearAttnVariant
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.engine.resources.sequence_shard import SeqShardSpec, sequence_parallel_refusals
from mstar.engine.resources.spec import NodeResourceSpec, apply_yaml_overrides, resolve_spec_dependencies
from mstar.graph.base import GraphEdge, GraphNode, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import KVLabel, PrefixKeys, WorkerGraph
from mstar.model.components.moe_parallel import MoeParallelConfig
from mstar.testing_utils.resource import StubResource
from mstar.validation import DeploymentError
from mstar.validation.placement import MAX_NODE_INPUTS, _sharding_groups


class _Stub(StubResource):
    pass


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

    def __init__(self, walks=_prefill_decode, specs=(), kv_labels=None, refuse_yaml=None, **kw):
        super().__init__(walks, **kw)
        self._specs = list(specs)
        self._kv_labels = kv_labels or {}
        self._refuse_yaml = refuse_yaml

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
    ({"MSTAR_WIRE_CODEC": "pickle"}, r"MSTAR_WIRE_CODEC must be msgpack; got 'pickle'"),
], ids=["codec"])
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


def test_unpausable_outputs_that_are_not_a_list_are_refused(tmp_path):
    _refused(
        _Model(), _cfg(tmp_path, unpausable_outputs="audio"),
        r"unpausable_outputs must be a list of output modalities from \[.*\], got 'audio'",
    )


@pytest.mark.parametrize("outputs", [["Audio"], ["speech"]])
def test_unpausable_outputs_that_name_no_output_modality_are_refused(tmp_path, outputs):
    """A misspelled modality matched no request's outputs, so every voice stream was preemptible."""
    _refused(
        _Model(), _cfg(tmp_path, unpausable_outputs=outputs),
        rf"unpausable_outputs must be a list of output modalities from \[.*\], got \['{outputs[0]}'\]",
    )


def test_missing_node_groups_is_refused(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("max_seq_len: 64\n")
    _refused(_Model(), str(cfg), r"must define `node_groups`")


def test_unknown_autocast_dtype_is_refused(tmp_path):
    """Each worker refused it after spawn."""
    _refused(_Model(), _cfg(tmp_path, autocast_dtype="bf16"), r"autocast_dtype 'bf16' is not a torch dtype")


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
             r"node_groups\[0\] \(nodes \['llm'\]\) has 3 ranks; not divisible by its instance size")


@pytest.mark.parametrize("axis, pattern", [
    ("tp", r"node_groups\[0\] has tp_size 2 and contains nodes not in tp_enabled_nodes"),
    ("sp", r"node_groups\[0\]: node 'llm' runs on 2 SP ranks but neither splits tokens"),
    ("cfg", r"node_groups\[0\] has cfg_size 2 and contains nodes not in cfg_enabled_nodes"),
], ids=["tp", "sp", "cfg"])
def test_parallel_node_group_over_a_node_without_support_is_refused(axis, pattern, tmp_path):
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [0, 1], f"{axis}_size": 2}])
    _refused(_Model(), cfg, pattern)


def test_the_sp_checks_run_when_the_lora_specs_fail_to_build(tmp_path):
    """Both problems are reported, not the build failure alone, and the LoRA spec that failed
    isn't read again."""
    from mstar.engine.resources.lora.config import LoraSpec

    def declare():
        raise OSError("the checkpoint's modules can't be read")

    lora = LoraSpec(resource_key="lora", nodes={"llm"}, declare=declare)
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [0, 1], "sp_size": 2}], resources={"lora": {}})
    problems = _problems(_Model(specs=[lora]), cfg)
    assert any("building the node resources failed: OSError" in p for p in problems), problems
    assert any("node 'llm' runs on 2 SP ranks but neither splits tokens" in p for p in problems), problems


def _paged_llm(*extra, node="llm", kv_heads=2, **kw) -> _Model:
    """A model whose ``node`` holds a paged KV pool, and ``extra`` specs."""
    kv = KVSpec(resource_key="kv", nodes={node}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=kv_heads, head_dim=8, max_seq_len=64, num_qo_heads=4, page_size=16,
    ))
    return _Model(specs=[kv, *extra], kv_labels={"kv": {"main": KVLabel()}}, **kw)


def _shard(node="llm", **kw) -> SeqShardSpec:
    return SeqShardSpec("shard", {node}, ("main",), 0, **kw)


def _attention(nodes=("llm",)) -> AttentionSpec:
    return AttentionSpec("attn", set(nodes), AttentionConfig(kv_cache="kv"))


def _ulysses(heads: int) -> RaggedAttentionSpec:
    # over the label the shard splits, so it attends it by Ulysses
    return RaggedAttentionSpec("ragged", {"llm"}, RaggedAttentionConfig(heads, heads, 8), labels={"main"})


def _cross() -> RaggedCrossAttentionSpec:
    return RaggedCrossAttentionSpec("xattn", {"llm"}, RaggedAttentionConfig(4, 4, 8))


def _mla_llm() -> _Model:
    """A model whose ``llm`` attends an MLA pool, and declares a shard to split its steps."""
    kv = KVSpec(resource_key="kv", nodes={"llm"}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, num_qo_heads=4, page_size=16,
        layout=KVLayout.MLA, kv_lora_rank=8, qk_rope_head_dim=4,
    ))
    attn = AttentionSpec("attn", {"llm"}, AttentionConfig(kv_cache="kv", backend=AttnBackend.FLASHINFER_MLA,
                                                          sm_scale=0.1))
    return _Model(specs=[kv, attn, _shard()], kv_labels={"kv": {"main": KVLabel()}})


def _linear(k_heads=2, v_heads=4, k_dim=8, v_dim=8, variant=LinearAttnVariant.GDN) -> list[NodeResourceSpec]:
    """A gated delta rule layer stack's pool and resource on ``llm``."""
    geometry = DeltaNetGeometry(
        num_k_heads=k_heads, num_v_heads=v_heads, head_k_dim=k_dim, head_v_dim=v_dim, conv_kernel_size=4,
    )
    return [
        RecurrentStateSpec("state", {"llm"}, RecurrentStateConfig(num_layers=1, blocks=geometry.to_blocks())),
        LinearAttnSpec("linear", {"llm"}, LinearAttnConfig(recurrent_state="state", variant=variant)),
    ]


@pytest.mark.parametrize("model,group,kv,pattern", [
    ({}, dict(sp_size=4), dict(page_size=6),
     r"node_groups\[0\]: KV 'kv': context_parallel stripes each page over 4 ranks, which doesn't divide page_size 6"),
    (dict(tp_enabled_nodes=["llm"]), dict(tp_size=2), {},
     r"node_groups\[0\]: KV 'kv': .* 2 KV head\(s\) over 2 TP rank\(s\) leave none sharing one"),
], ids=["page_size", "kv_heads"])
def test_a_kv_placement_the_node_cannot_take_is_refused(model, group, kv, pattern, tmp_path):
    """The engine refused these on every worker, after loading the weights."""
    size = group.get("tp_size", 1) * group.get("sp_size", 1)
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": list(range(size)), **group}],
               resources={"kv": {"context_parallel": True, **kv}})
    _refused(_paged_llm(**model), cfg, pattern)


@pytest.mark.parametrize("specs, group, resources, pattern", [
    ([_shard()], dict(sp_size=2), {},
     r"node_groups\[0\]: node 'llm' runs on 2 SP ranks, and its steps never split"),
    ([_shard(), _attention()], dict(sp_size=4), {},
     r"node_groups\[0\]: node 'llm' splits tokens over 4 SP ranks, and KV 'kv' stores 4 query and 2 KV heads by "
     r"head, which don't split 4 ways per TP rank"),
    # each row split over KV stored by position
    ([SeqShardSpec("shard", {"llm"}, ("main",), 1), _attention()], dict(sp_size=2),
     {"kv": {"context_parallel": True}, "shard": {"exchange": "all_gather"}},
     r"node_groups\[0\]: node 'llm' splits each row of its steps \(axis 1\) over 2 SP ranks, and KV 'kv' stores "
     r"them by position"),
    # the shard's exchange against the node's storage, at any SP size
    ([_shard(), _attention()], dict(sp_size=1), {"shard": {"exchange": "ring"}},
     r"resources.shard: exchange 'ring' attends split steps over KV stored by position, and no KV pool on nodes "
     r"\['llm'\] sets context_parallel: true"),
    ([_shard(), _attention()], dict(sp_size=2), {"shard": {"exchange": "all_gather"}},
     r"resources.shard: exchange 'all_gather' attends split steps over KV stored by position"),
    ([_shard(), _attention()], dict(sp_size=2), {"kv": {"context_parallel": True}, "shard": {"exchange": "ulysses"}},
     r"resources.shard: exchange 'ulysses' trades a split step's tokens for KV heads, and nodes \['llm'\] store KV "
     r"by position"),
    ([_shard(), _attention()], dict(sp_size=2), {"shard": {"exchange": "zigzag"}},
     r"resources.shard: exchange must be one of auto, none, ulysses, all_gather, ring; got 'zigzag'"),
    # never split over KV stored by head: its SP ranks would only share heads
    ([_shard(), _attention()], dict(sp_size=2), {"shard": {"exchange": "none"}},
     r"node_groups\[0\]: node 'llm' runs on 2 SP ranks, and its steps never split"),
    # a ring stripes steps by stored position, which Ulysses' ragged plan doesn't read
    ([_shard(), _attention(), _ulysses(heads=2)], dict(sp_size=2),
     {"kv": {"context_parallel": True}, "shard": {"exchange": "ring"}},
     r"node_groups\[0\]: ragged attention 'ragged' attends labels its sequence shard splits \(Ulysses\), whose "
     r"plan reads a split label's tokens in step order, and node 'llm''s sequence shard stripes them by KV "
     r"'kv''s positions"),
    ([_shard(), _ulysses(heads=3)], dict(sp_size=2), {},
     r"node_groups\[0\]: ragged attention 'ragged' attends labels its sequence shard splits \(Ulysses\), and its 3 "
     r"query and 3 KV heads "
     r"don't split over 1 TP x 2 SP ranks"),
    # a packed split can leave a rank none of a row's queries or keys
    ([_shard(), _attention(), _cross()], dict(sp_size=2), {},
     r"node_groups\[0\]: ragged cross-attention 'xattn' on node 'llm', whose sequence shard 'shard' splits "
     r"packed tokens \(axis 0\)"),
    # the attention spans a node the shard doesn't, so it would split on one and not the other: at any SP size
    ([_shard(), _attention(nodes=("llm", "post"))], dict(sp_size=1), {},
     r"resources.attn: spans nodes \['post'\] without the sequence shard 'shard'"),
    # a recurrent layer's split exchange: its heads don't split 3 ways, so it hands state on
    ([_shard(), _attention(), *_linear(v_dim=16)], dict(sp_size=3),
     {"kv": {"context_parallel": True, "page_size": 18}, "shard": {"exchange": "all_gather"}},
     r"linear attention 'linear''s heads don't split 3 ways per TP rank, so it hands state on over 3 SP ranks "
     r"\(state_passing\), whose middle pieces' transition needs equal key and value head dims; they are 8 and 16"),
    ([SeqShardSpec("shard", {"llm"}, ("main",), 1), _attention(), *_linear()], dict(sp_size=2), {},
     r"linear attention 'linear' splits a step's rows into contiguous pieces, and node 'llm''s shard splits each "
     r"row \(axis 1\)"),
    ([_shard(), _attention(), *_linear()], dict(sp_size=2),
     {"kv": {"context_parallel": True}, "shard": {"exchange": "ring"}},
     r"linear attention 'linear' splits a step's rows into contiguous pieces, and node 'llm''s shard splits by "
     r"KV 'kv''s stored positions"),
    # whatever its heads: no exchange of the linear resource's serves it
    ([_shard(), _attention(), *_linear(k_heads=3, v_heads=3, variant=LinearAttnVariant.KDA)], dict(sp_size=2), {},
     r"linear attention 'linear' is Kimi delta attention, which takes no split steps"),
], ids=["no_attention", "heads_by_head", "axis1_over_positions", "ring_by_head", "all_gather_by_head",
        "ulysses_by_position", "unknown_exchange", "none_by_head", "ring_beside_ragged_ulysses", "ulysses_heads",
        "axis0_cross", "shard_misses_a_node", "linear_state_passing_head_dims",
        "linear_axis1", "linear_beside_a_ring", "kda"])
def test_a_node_whose_sp_ranks_cant_split_its_steps_is_refused(specs, group, resources, pattern, tmp_path):
    size = group.get("sp_size", 1)
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": list(range(size)), **group}], resources=resources)
    _refused(_paged_llm(*specs), cfg, pattern)


@pytest.mark.parametrize("specs, resources", [
    ([_shard(), _attention()], {}),
    ([_attention()], {"kv": {"context_parallel": True}}),
    ([_shard(), _attention()], {"shard": {"exchange": "ulysses"}}),
    # a strategy for split steps is a no-op on a node that never splits
    ([_ulysses(heads=3), _attention()], {"kv": {"context_parallel": True}}),
    ([_shard(), _attention()], {"kv": {"context_parallel": True}, "shard": {"exchange": "all_gather"}}),
    ([_shard(), _attention()], {"kv": {"context_parallel": True}, "shard": {"exchange": "ring"}}),
    # DCP alone: the node runs copied, as one without a shard
    ([_shard(), _attention()], {"kv": {"context_parallel": True}}),
    ([_shard(), _attention()], {"kv": {"context_parallel": True}, "shard": {"exchange": "none"}}),
    # a model's shard over a node with no attention that takes split steps is a no-op too
    ([_shard()], {"kv": {"context_parallel": True}}),
], ids=["heads", "positions_copied", "ulysses", "ulysses_without_a_shard", "all_gather", "ring", "positions_auto",
        "none", "shard_without_an_attention"])
def test_a_node_that_splits_its_steps_or_stores_positions_on_sp_ranks_validates(specs, resources, tmp_path):
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [0, 1], "sp_size": 2}], resources=resources)
    assert _validate(_paged_llm(*specs), cfg).worker_graphs


def test_an_mla_node_cant_split_its_steps(tmp_path):
    """MLA attention takes no split steps: each rank planning its own queries over gathered latents
    needs admission bounding every rank's plan, which lands with MLA over placed pools. Its pool's
    one KV head is refused before spawn."""
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [0, 1], "sp_size": 2}])
    _refused(_mla_llm(), cfg, r"KV 'kv' stores 4 query and 1 KV heads by head, which don't split 2 ways per TP rank")


def _windowed_llm(window_layers=(1,)) -> _Model:
    """A model whose ``llm`` attends a pool of a full layer and ``window_layers`` keeping a window,
    and splits its steps."""
    kv = KVSpec(resource_key="kv", nodes={"llm"}, config=PagedKVConfig(
        num_layers=1 + len(window_layers), num_kv_heads=2, head_dim=8, max_seq_len=256, num_qo_heads=4,
    ))
    full = AttentionSpec("attn", {"llm"}, AttentionConfig(kv_cache="kv"))
    window = AttentionSpec("window", {"llm"}, AttentionConfig(kv_cache="kv", sliding_window=64,
                                                              layers=tuple(window_layers)))
    return _Model(specs=[kv, full, window, _shard()], kv_labels={"kv": {"main": KVLabel()}})


@pytest.mark.parametrize("resources, window_layers, refused", [
    # the full layer split by position, the window layers whole on each rank (kv_gather)
    ({"kv": {"context_parallel": True}, "shard": {"exchange": "all_gather"}}, (1,), None),
    # DCP alone: copied steps, no decode collective on the window layers
    ({"kv": {"context_parallel": True}}, (1,), None),
    # kv_gather plans contiguous pieces, which a ring's stripe doesn't give
    ({"kv": {"context_parallel": True}, "shard": {"exchange": "ring"}}, (1,),
     r"attention 'window' gathers a split step's keys into KV 'kv', which every rank stores whole \(kv_gather\)"),
    # 3-token window pages (6-token full ones) don't split over the 2 ranks
    ({"kv": {"context_parallel": True, "page_size": 6}, "shard": {"exchange": "all_gather"}}, (1, 2),
     r"keeps its sliding-window layers whole on each of its 2 partners in 3-token pages"),
], ids=["all_gather", "copied", "ring", "window_page"])
def test_a_node_storing_kv_by_position_keeps_its_window_layers_whole(resources, window_layers, refused, tmp_path):
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [0, 1], "sp_size": 2}], resources=resources)
    if refused is None:
        assert _validate(_windowed_llm(window_layers), cfg).worker_graphs
    else:
        _refused(_windowed_llm(window_layers), cfg, refused)


@pytest.mark.parametrize("attention, pattern", [
    (_attention, r"exchanges=False, which can hand a rank no tokens, beside an attention that exchanges"),
    (_cross, r"ragged cross-attention 'xattn' .* declares exchanges=False, which can leave a rank no tokens"),
], ids=["attention", "cross_attention"])
def test_a_split_without_exchanges_beside_an_attention_is_refused(attention, pattern):
    """A split nothing exchanges can give a rank no tokens, which no attention over them can take."""
    shard = SeqShardSpec("shard", {"llm"}, ("main",), 1, exchanges=False)
    refusals = sequence_parallel_refusals(_paged_llm(shard, attention()).get_node_resources(), "llm", 1, 4)
    assert any(re.search(pattern, refusal) for refusal in refusals), refusals


def test_expert_parallelism_on_a_node_that_splits_its_steps_is_refused():
    specs = _paged_llm(_shard(), _attention()).get_node_resources()
    assert sequence_parallel_refusals(specs, "llm", 1, 2, MoeParallelConfig(parallel="ep")) == [
        "node 'llm' splits tokens over SP ranks, which expert parallelism (ep_config) can't",
    ]
    assert sequence_parallel_refusals(specs, "llm", 1, 2, MoeParallelConfig()) == []
    # a node that never splits (auto over KV stored by position) takes what one without a shard does
    apply_yaml_overrides(specs, {"resources": {"kv": {"context_parallel": True}}})
    resolve_spec_dependencies(specs)
    assert sequence_parallel_refusals(specs, "llm", 1, 2, MoeParallelConfig(parallel="ep")) == []


def test_a_resource_over_nodes_that_split_heads_differently_is_refused(tmp_path):
    """Engine.load_model raised this on the worker, after loading the weights: 'a' places its
    positions over the SP ranks (heads over TP), 'b' splits its steps (heads over TP x SP)."""
    def pool(key, node):
        return KVSpec(resource_key=key, nodes={node}, config=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=64, num_qo_heads=4, page_size=16,
        ))

    specs = [pool("kv", "a"), pool("kv_b", "b"), _shard("b"),
             AttentionSpec("attn", {"b"}, AttentionConfig(kv_cache="kv_b")), _Spec("shared", {"a", "b"})]
    model = _Model(_a_then_b, specs=specs, kv_labels={"kv": {"main": KVLabel()}, "kv_b": {"main": KVLabel()}})
    cfg = _cfg(tmp_path, [{"node_names": ["a", "b"], "ranks": [0, 1], "sp_size": 2}],
               resources={"kv": {"context_parallel": True}})
    _refused(model, cfg, r"node_groups\[0\]: \['a'\] place their KV positions over the SP ranks and "
                         r"\['b'\] keep KV by head over them, .* share resources \['shared'\]")


@pytest.mark.parametrize("tp, placed, shard", [
    (4, True, None), (2, True, {"exchange": "all_gather"}), (1, False, None), (1, False, {"exchange": "ulysses"}),
], ids=["copied", "split", "by_head", "by_head_ulysses"])
def test_qwen3_5_on_sp_ranks_runs_with_its_kv_positions_placed_over_them(tp, placed, shard, tmp_path, monkeypatch):
    """As placed, its steps run copied (DCP); with its shard's exchange set they split, and its
    recurrent layers follow by Ulysses (at a TP size leaving their heads two SP ranks' worth).
    Over KV stored by head it splits only where the deployment opts into Ulysses, which no GPU
    has run: refused by default."""
    config = yaml.safe_load((ROOT / "configs/qwen3_5_27b_tp4.yaml").read_text())
    llm, = (group for group in config["node_groups"] if group["node_names"] == ["LLM"])
    llm.update(ranks=list(range(2 * tp)), tp_size=tp, sp_size=2)
    if placed:
        config["resources"]["kv_cache"] = {"context_parallel": True}
    if shard is not None:
        config["resources"]["llm_shard"] = shard
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config))
    model = _qwen3_5(config["model"], {}, tmp_path, monkeypatch)
    if not placed and shard is None:
        _refused(model, str(path), r"node 'LLM' runs on 2 SP ranks, and its steps never split")
        return
    assert _validate(model, str(path)).worker_graphs


class _CfgModel(_Model):
    """A model whose ``llm`` splits two guidance branches over ``cfg_size``."""

    def get_default_sharding_config(self):
        return ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}, cfg_enabled_nodes={"llm": 2})


def test_cfg_size_that_does_not_divide_the_branches_is_refused(tmp_path):
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [0, 1, 2, 3], "cfg_size": 4}])
    _refused(_CfgModel(), cfg, r"cfg_size 4, which does not divide the 2 guidance branches 'llm' runs")


def test_a_cfg_instance_is_one_lockstep_group(tmp_path):
    """Its ranks route as one instance, not as cfg_size data-parallel replicas
    (which would leave each rank's gather waiting for a peer that never runs)."""
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [0, 1], "cfg_size": 2}])
    _validate(_CfgModel(), cfg)
    (group,) = _CfgModel().get_sharding_config(cfg).groups
    assert (group.nodes, group.tp_size) == ({"llm"}, 2)


@pytest.mark.parametrize("samplers, node_groups, pattern", [
    (("sampler", "talker"), None, r"enable_grammar is on samplers \['sampler', 'talker'\]"),
    (("sampler",), [{"node_names": ["llm"], "ranks": [0], "graph_walks": ["prefill"]},
                    {"node_names": ["llm"], "ranks": [1], "graph_walks": ["decode"]}],
     r"sampler sampler: enable_grammar is not served on a node whose walks run on different ranks"),
], ids=["two samplers", "prefill/decode split"])
def test_enable_grammar_where_one_grammar_cannot_follow_the_draws_is_refused(tmp_path, samplers, node_groups, pattern):
    """A request's grammar lives with one sampler on one worker."""
    specs = [SamplerSpec(resource_key=key, nodes={"llm"}, vocab_size=8) for key in samplers]
    cfg = _cfg(tmp_path, node_groups, resources={key: {"enable_grammar": True} for key in samplers})
    _refused(_GrammarModel(specs=specs), cfg, pattern)


class _GrammarModel(_Model):
    STRUCTURED_OUTPUTS = True


def test_enable_grammar_on_a_model_that_does_not_declare_it_is_refused(tmp_path):
    """Structured outputs are served only on a model that declares them (Qwen3.5 for now)."""
    specs = [SamplerSpec(resource_key="sampler", nodes={"llm"}, vocab_size=8)]
    cfg = _cfg(tmp_path, resources={"sampler": {"enable_grammar": True}})
    _refused(_Model(specs=specs), cfg, r"_Model does not declare STRUCTURED_OUTPUTS")
    _validate(_GrammarModel(specs=specs), cfg)


def test_qwen3_5_declares_structured_outputs():
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    assert Qwen3_5DenseModel.STRUCTURED_OUTPUTS


def test_a_paged_kv_cache_on_a_cfg_parallel_node_is_refused(tmp_path):
    """Its rank math takes every instance rank for a head shard, and a CFG rank
    runs another branch on every head. Refused before any worker loads weights."""
    kv = KVSpec(resource_key="kv", nodes={"llm"}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, max_num_pages=4, page_size=16,
    ))
    model = _CfgModel(specs=[kv], kv_labels={"kv": {"main": KVLabel()}})
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [0, 1], "cfg_size": 2}])
    _refused(model, cfg, r"KV kv: a node with cfg_size > 1 can't hold a paged KV cache")


class _WideModel(_Model):
    """A model whose ``llm`` takes TP beside SP or CFG."""

    def get_default_sharding_config(self):
        return ShardingConfig(groups=[], tp_enabled_nodes={"llm"}, shard_dim={}, cfg_enabled_nodes={"llm": 2})


@pytest.mark.parametrize("axis", ["sp", "cfg"])
def test_a_sharded_signal_into_an_instance_wider_than_tp_is_refused(axis, tmp_path):
    """The fanout splits a shard_dim signal over every instance rank, while
    only the TP ranks split the weights."""
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [0, 1, 2, 3], "tp_size": 2, f"{axis}_size": 2}],
               sharding_config={"shard_dim": {"x": 1}})
    # a node SP ranks can run: a sequence shard and an attention that takes it
    model = _WideModel(specs=[SeqShardSpec("shard", {"llm"}, ("main",), 0), _ulysses(4)])
    _refused(model, cfg, r"node 'llm' takes or emits \['x'\], which have a shard_dim")


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
    """The conductor refuses a node the graph runtime would refuse when a
    worker compiled its graph."""
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
        "the worker graph of ['w'] with nodes ['a'] on ranks [0, 1] has instance size 2, but its "
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


class _Trail(_Stub):
    """Keeps a page trail the followers check against rank 0's, and nothing else."""

    def page_trail(self):
        return (0, 0)


class _TrailSpec(_Spec):
    @property
    def resource_class(self):
        return _Trail


@pytest.mark.parametrize("spec, resources", [
    (_kv("kv", ["a", "b"]), {"kv": {"cpu_offload_pages": 0}}),  # no host pool: can't evict
    (_TrailSpec("kv", {"a", "b"}), {}),
], ids=["kv_offload_off", "page_trail_only"])
def test_lockstep_nodes_sharing_a_journaled_resource_are_refused(spec, resources, tmp_path):
    """Rank 0's prefix leases and page trails on a resource reach the followers
    per node, riding on one node's next step, so two lockstep nodes over one
    resource could admit out of step."""
    labels = {"kv": {"main": KVLabel()}} if isinstance(spec, KVSpec) else None
    model = _Model(_a_then_b, specs=[spec], kv_labels=labels, tp_enabled_nodes=["a", "b"])
    cfg = _cfg(tmp_path, [{"node_names": ["a", "b"], **_TP2}], resources=resources)
    [problem] = _problems(model, cfg)
    assert problem.startswith("Resource kv is shared by lockstep-parallel nodes ['a', 'b'] on worker_0, worker_1;")


@pytest.mark.parametrize("b, placed", [
    ({"ranks": [0, 1, 2, 3], "tp_size": 4}, "TP ranks [0, 1, 2, 3]"),
    ({"ranks": [0, 1], "sp_size": 2}, "SP ranks [0, 1]"),  # the same ranks, the other axis
], ids=["other_size", "other_axis"])
def test_lockstep_nodes_in_different_instances_on_one_worker_are_refused(b, placed, tmp_path):
    """Two instances on a worker are two leaders' step streams with no order
    between them."""
    cfg = _cfg(tmp_path, [{"node_names": ["a"], **_TP2}, {"node_names": ["b"], **b}])
    model = _paged_llm(_shard(node="b"), _attention(nodes=("b",)), node="b", walks=_a_then_b,
                       tp_enabled_nodes=["a", "b"])
    problems = _problems(model, cfg)
    assert len(problems) == 2
    for rank, problem in enumerate(problems):
        assert f"on worker_{rank}" in problem and f"a on TP ranks [0, 1]; b on {placed}" in problem


def test_parallel_ranks_off_their_position_are_refused(tmp_path):
    """The conductor indexes workers by rank, so ranks [1, 2] mapped rank 1 to
    worker_2 and rank 2 off the end."""
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [1, 2], "tp_size": 2}])
    _refused(_Model(tp_enabled_nodes=["llm"]), cfg,
             r"the deployment uses ranks \[1, 2\]; ranks \[1, 2\] are in parallel groups")


@pytest.mark.parametrize(("ranks", "refused"), [([1, 0], True), ([2, 3, 0, 1], False)])
def test_parallel_ranks_out_of_order_are_refused(tmp_path, ranks, refused):
    """torch orders a group by global rank and the mesh by the config's list,
    so a gather would come back in a different order than the ranks split by.
    Only each instance's own ranks need the order: [2, 3, 0, 1] is two
    ascending replicas."""
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": ranks, "tp_size": 2}])
    if refused:
        _refused(_Model(tp_enabled_nodes=["llm"]), cfg, r"out of order")
    else:
        _validate(_Model(tp_enabled_nodes=["llm"]), cfg)


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
    _refused(_Model(tp_enabled_nodes=["llm"]), cfg, r"Node llm already has a tp comm group assigned for worker 0")


def test_resource_over_nodes_of_different_parallel_shapes_is_refused(tmp_path):
    """Engine.load_model raised this on the worker, after building the submodules."""
    cfg = _cfg(tmp_path, [
        {"node_names": ["a"], "ranks": [0, 1], "tp_size": 2},
        {"node_names": ["b"], "ranks": [2]},
    ])
    model = _Model(_a_then_b, specs=[_Spec("kv", {"a", "b"})], tp_enabled_nodes=["a"])
    _refused(model, cfg, r"Resource spec kv nodes \['a', 'b'\] must use the same parallel shape across replicas")


def test_resource_over_local_nodes_in_different_parallel_groups_is_refused(tmp_path):
    cfg = _cfg(tmp_path, [
        {"node_names": ["a"], "ranks": [0, 1], "tp_size": 2},
        {"node_names": ["b"], "ranks": [0, 2], "tp_size": 2},
    ])
    model = _Model(_a_then_b, specs=[_Spec("kv", {"a", "b"})], tp_enabled_nodes=["a", "b"])
    assert (
        "Resource spec kv local nodes ['a', 'b'] on worker_0 must all be in the same parallel group"
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
    # NaN would make every replica's cost NaN, and every DP ingest fail
    ([_Spec("kv", {"llm"})], {"routing_weights": {"prefix": float("nan")}},
     r"routing_weights.prefix must be a number from 0 to 1e\+06; got nan"),
    # finite, but a busy replica's cost overflows to inf, and inf - inf is NaN
    ([_Spec("kv", {"llm"})], {"routing_weights": {"load": 1e308, "prefix": 1e308}},
     r"routing_weights.load must be a number from 0 to 1e\+06; got 1e\+308"),
    # a weight is per term of the cost, not per resource
    ([_Spec("kv", {"llm"})], {"routing_weights": {"kv": 2.0}},
     r"routing_weights names \['kv'\]; it takes \['load', 'prefix'\]"),
    # named on its own, not as a config the resource checks could not read, empty too
    ([_Spec("kv", {"llm"})], {"routing_weights": []}, r"routing_weights must be a mapping"),
    # nothing would spread a hot prefix
    ([_Spec("kv", {"llm"})], {"routing_weights": {"load": 0}}, r"routing_weights.load must be above 0"),
], ids=["top_level_kv_cache", "unknown_resource", "bad_override", "duplicate_key", "missing_dependency",
        "self_dependency", "cycle", "routing_weight_not_finite", "routing_weights_overflow",
        "routing_weight_per_resource", "routing_weights_not_a_mapping", "routing_load_0"])
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


# -- LoRA ---------------------------------------------------------------------


def _lora_spec():
    """One LoRA spec over a (its qkv target) and b (its o target)."""
    from test_lora_pool import QKV, O, make_spec  # skips the test without safetensors

    targets = {"qkv": dataclasses.replace(QKV, node="a"), "o": dataclasses.replace(O, node="b")}
    return dataclasses.replace(make_spec({}), nodes={"a", "b"}, targets=targets)


# a tensor-parallel LLM (a) and its vision tower (b), in different lockstep instances
_A_TP_THEN_B = [{"node_names": ["a"], "ranks": [0, 1], "tp_size": 2}, {"node_names": ["b"], "ranks": [2]}]


def test_a_lora_deployment_passes(tmp_path):
    """The pass checks the specs the YAML turns on, as the workers build them: the opt-in
    LoRA spec, for the `lora:` block to override, split into a pool per node."""
    cfg = _cfg(tmp_path, _A_TP_THEN_B, resources={"lora": {"max_slots": 2}})
    _validate(_Model(_a_then_b, specs=[_lora_spec()], tp_enabled_nodes=["a"]), cfg)


@pytest.mark.parametrize("exchange", ["all_gather", "ring", None])
def test_lora_over_kv_stored_by_position_is_refused_where_the_steps_split(exchange, tmp_path):
    """Storing KV by position doesn't make a node's steps run copied: the shard's exchange decides,
    and a split step holds a shard of each row's tokens, which LoRA can't lay out."""
    from test_lora_pool import QKV, make_spec  # skips the test without safetensors

    lora = dataclasses.replace(make_spec({}), nodes={"llm"}, targets={"qkv": dataclasses.replace(QKV, node="llm")})
    model = _paged_llm(_shard(), _attention(), lora)
    resources = {"lora": {"max_slots": 2}, "kv": {"context_parallel": True}}
    if exchange is not None:
        resources["shard"] = {"exchange": exchange}
    cfg = _cfg(tmp_path, [{"node_names": ["llm"], "ranks": [0, 1], "sp_size": 2}], resources=resources)
    if exchange is None:
        assert _validate(model, cfg).worker_graphs
    else:
        _refused(model, cfg, r"node_groups\[0\]: LoRA resource 'lora@llm' serves node 'llm', which splits tokens "
                             r"over 2 SP ranks")


@pytest.mark.parametrize("block, refusal", [
    ({"max_slots": 0}, r"resources\.lora: max_slots must be an integer >= 1; got 0"),
    (["sql"], r"resources\.lora: .*argument after \*\* must be a mapping, not list"),
], ids=["bad_value", "not_a_mapping"])
def test_a_bad_lora_value_is_reported_once_with_the_other_problems(tmp_path, block, refusal):
    """A refused `lora:` block leaves the spec as declared, so splitting it per node
    reports nothing again, and the other resources' placement checks still run."""
    cfg = _cfg(tmp_path, _A_TP_THEN_B, resources={"lora": block})
    model = _Model(_a_then_b, specs=[_lora_spec(), _Spec("kv", {"a", "b"})], tp_enabled_nodes=["a"])
    problems = _problems(model, cfg)
    assert len(problems) == 2 and re.fullmatch(refusal, problems[0]), problems
    assert problems[1] == "Resource spec kv nodes ['a', 'b'] must use the same parallel shape across replicas"


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
    monkeypatch.setenv("MSTAR_WIRE_CODEC", "pickle")
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
        "MSTAR_WIRE_CODEC must be msgpack; got 'pickle'",
        "the config must define `max_seq_len`",
        "no admit queue",
        "node 'llm' of graph walk 'decode' is placed by no node_groups entry; place it in exactly "
        "one node_groups entry for that walk (use graph_walks to give one node different ranks per walk).",
        "serving config overrides unknown resource(s) ['vk']; this model declares ['kv']",
    ]
    assert str(refusal.value).startswith(f"{cfg}: 5 problems; no worker was started:\n  - MSTAR_WIRE_CODEC")
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
        "Resource spec kv nodes ['llm', 'post'] must use the same parallel shape across replicas"
    )


class _StateModel(_Model):
    def __init__(self, arrays, **kw):
        super().__init__(**kw)
        self._arrays = arrays

    def declare_state(self):
        return list(self._arrays)


def test_an_sdpa_read_over_a_split_label_is_reported_only_where_the_node_splits(tmp_path):
    """The decision is made from the specs after the YAML's overrides and `derive`, as the engine does."""
    from test_state_cross_reads import _array

    # other heads, so the two reads don't share one resource
    video, audio = _array("video", "video_self", nodes=("llm",)), _array("audio", nodes=("llm",), heads=2)
    model = _StateModel([video, audio], specs=[SeqShardSpec("shard", {"llm"}, ("video", "audio"), 0)])
    groups = [{"node_names": ["llm"], "ranks": [0, 1], "sp_size": 2}]
    sdpa = {"video_self": {"backend": "sdpa"}}
    _refused(model, _cfg(tmp_path, groups, resources=sdpa), r"backend sdpa over labels \['video'\]")
    # the YAML's exchange turns the split off, so SDPA reads the label whole; the node's other problem remains
    copied = _problems(model, _cfg(tmp_path, groups, resources={**sdpa, "shard": {"exchange": "none"}}))
    assert copied and not any("backend sdpa over labels" in problem for problem in copied), copied
