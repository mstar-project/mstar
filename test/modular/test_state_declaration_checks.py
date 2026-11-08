"""A declaration is checked against the model that made it, and every process can compare what it built.

Layer code still picks its resource key by name, so a read naming a walk the model doesn't have, or two reads over one
layer and walk, would otherwise run the wrong mask silently. The digest is how the conductor and the workers will
compare declarations, so it can't rest on Python's ``hash``, which randomizes strings per process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, replace
from typing import Tuple

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.state import (
    Array,
    Attend,
    Bidirectional,
    Block,
    Causal,
    CausalConv,
    Entry,
    GatedDeltaNet,
    KVEntry,
    Positions,
    Read,
    Rope,
    Sampler,
    TensorType,
    Value,
    Window,
)
from mstar.engine.state.analysis import declaration_digest, state_declaration_problems
from mstar.engine.state.contract import Mask
from mstar.graph.base import GraphNode
from mstar.model.base import KVLabel, PrefixKeys
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import ChunkingPolicy

_ATTEND = Attend(q_heads=8)


class _StubModel:
    """Serves declarations and walks without building anything."""

    def __init__(self, *arrays, combined=None, kv_labels=None, rebuild=None):
        self.arrays = list(arrays)
        self.combined = combined or {}
        self.labels = kv_labels or {}
        self.rebuild = rebuild or {}

    def kv_labels(self):
        return self.labels

    def declare_state(self):
        return self.arrays

    def get_chunking_policies(self):
        return {}

    def rebuild_policies(self):
        return self.rebuild

    def get_graph_walk_graphs(self):
        return {
            "prefill": GraphNode(name="LLM", input_names=[], outputs=[]),
            "decode": GraphNode(name="LLM", input_names=[], outputs=[]),
            "encode": GraphNode(name="vision", input_names=[], outputs=[]),
        }

    def get_combined_graph_walks(self):
        return self.combined


class _NoWalks(_StubModel):
    """Fails the way a model whose walk graphs read a checkpoint does."""

    def get_graph_walk_graphs(self):
        raise OSError("no checkpoint here")


def _array(name="kv", **overrides) -> Array:
    return Array(name, **{"nodes": {"LLM"}, "layers": {0, 1, 2}, "entry": KVEntry(heads=4, head_dim=8),
                          "max_len": 16, **overrides})


def _read(**overrides) -> Read:
    return Read(**{"mask": Causal(), "op": _ATTEND, **overrides})


def _one_problem(*arrays, **kwargs) -> str:
    problems = state_declaration_problems(_StubModel(*arrays, **kwargs))
    assert len(problems) == 1, problems
    return problems[0]


def test_a_model_that_declares_nothing_never_builds_its_walk_graphs():
    assert state_declaration_problems(_NoWalks()) == []


def test_the_walk_graphs_are_built_once_since_building_them_can_read_checkpoints():
    class _Counting(_StubModel):
        builds = 0

        def get_graph_walk_graphs(self):
            self.builds += 1
            return super().get_graph_walk_graphs()

    model = _Counting(_array())
    assert state_declaration_problems(model) == [] and model.builds == 1


def test_a_good_declaration_has_no_problems():
    assert state_declaration_problems(_StubModel(
        _array(reads={"local": _read(mask=Window(4), layers={0, 1}), "global": _read(layers={2})}),
    )) == []


def test_two_arrays_may_not_share_a_name():
    assert _one_problem(_array(), _array()).startswith("Array 'kv'.name: the name 'kv'")


def test_two_arrays_may_not_derive_one_positions_key():
    problem = _one_problem(
        _array(positions=Positions()), _array("mem", reads={"other": _read()}, positions=Positions()),
    )
    assert problem == "Array 'mem'.positions: the name 'rope' is declared twice, also by Array 'kv'.positions"


def test_a_positions_key_may_not_equal_another_declared_name():
    problem = _one_problem(_array("thinker_kv", reads={"thinker_pos": _read()}, positions=Positions()))
    assert problem == (
        "Array 'thinker_kv'.positions: the name 'thinker_pos' is declared twice, also by Array 'thinker_kv'.reads"
    )


def test_a_read_may_not_reuse_an_array_name():
    problem = _one_problem(_array(), _array("other", reads={"kv": _read()}))
    assert problem.startswith("Array 'other'.reads: the name 'kv'")


def test_an_update_may_not_reuse_a_read_key():
    gdn = Entry({"conv": TensorType((32, 3), dtype=None)})
    problem = _one_problem(
        _array(reads={"attn": _read()}),
        Value("gdn", nodes={"LLM"}, layers={0}, entry=gdn, updates={"attn": CausalConv()}),
    )
    assert problem.startswith("Array 'gdn'.updates: the name 'attn'")


def test_a_node_must_be_in_the_model():
    assert _one_problem(_array(nodes={"LLM", "dit"})).startswith("Array 'kv'.nodes: 'dit'")


def test_a_read_must_name_a_real_walk():
    problem = _one_problem(_array(reads={"r": _read(walks={"nope"})}))
    assert problem.startswith("Array 'kv'.reads['r'].walks: 'nope' is not a walk of the model")


def test_a_read_must_name_a_walk_of_the_array_nodes():
    problem = _one_problem(_array(reads={"r": _read(walks={"encode"})}))
    assert problem.startswith("Array 'kv'.reads['r'].walks: 'encode' runs none")


def test_a_combined_walk_is_refused_because_its_rows_read_by_their_own_walk():
    problem = _one_problem(
        _array(reads={"r": _read(walks={"mixed"})}), combined={"LLM": {"mixed": {"prefill", "decode"}}},
    )
    assert problem.startswith("Array 'kv'.reads['r'].walks: 'mixed' is a combined walk")


def test_two_reads_may_not_cover_one_layer_and_walk():
    problem = _one_problem(_array(reads={"a": _read(layers={0, 1}), "b": _read(layers={1, 2})}))
    assert problem.startswith("Array 'kv'.reads['b'].layers: layer 1"), "layer 1 reads through both"


def test_layers_none_expands_to_every_layer_of_the_array():
    problem = _one_problem(_array(reads={"a": _read(), "b": _read(layers={2})}))
    assert problem.startswith("Array 'kv'.reads['b'].layers: layer 2"), "a has every layer, so 2 is read twice"


def test_walks_none_expands_to_every_walk_of_the_nodes():
    problem = _one_problem(_array(reads={"a": _read(), "b": _read(walks={"decode"})}))
    assert "on walk 'decode'" in problem, "a has every walk of LLM, so decode is read twice"


def test_two_reads_may_share_a_layer_on_different_walks():
    assert state_declaration_problems(_StubModel(_array(reads={
        "p": _read(walks={"prefill"}), "d": _read(walks={"decode"}),
    }))) == []


def test_a_layer_with_no_read_is_allowed():
    assert state_declaration_problems(_StubModel(_array(reads={"r": _read(layers={0})}))) == []


def _gdn_entry() -> Entry:
    return Entry({"state": TensorType((4, 8, 8), dtype=torch.float32), "conv": TensorType((32, 3), dtype=None)})


def test_a_sampler_may_not_reuse_a_declared_name():
    problem = _one_problem(_array(reads={"r": _read()}), Sampler("r", nodes={"LLM"}, vocab_size=8))
    assert problem.startswith("Sampler 'r'.name") and "declared twice" in problem


def test_a_block_read_needs_max_len_zero():
    problem = _one_problem(_array(reads={"r": _read(mask=Block())}))
    assert problem.startswith("Array 'kv'.reads['r'].mask") and "Block needs max_len 0" in problem


def test_every_read_of_an_array_with_max_len_zero_is_a_block_read():
    problem = _one_problem(_array(max_len=0, reads={"r": _read()}))
    assert problem.startswith("Array 'kv'.reads['r'].mask") and "is Causal" in problem


def test_a_block_read_on_max_len_zero_has_no_problem():
    assert state_declaration_problems(_StubModel(_array(max_len=0, reads={"r": _read(mask=Block())}))) == []


@pytest.mark.parametrize("array", [
    Value("kv", nodes={"LLM"}, layers={0}, entry=_gdn_entry()),
    Array("kv", nodes={"LLM"}, layers={0}, entry=KVEntry(heads=4, head_dim=8), max_len=0),
])
def test_positions_sit_only_on_an_append_only_array_that_keeps_entries(array):
    with_positions = replace(array, positions=Positions())
    problem = _one_problem(with_positions)
    assert problem.startswith("Array 'kv'.positions") and "append-only" in problem


@pytest.mark.parametrize("rope", [
    Rope(low_freq_factor=1.0),
    Rope(low_freq_factor=1.0, high_freq_factor=1.0),
    Rope(old_context_len=8192),
])
def test_a_rope_sets_all_three_llama_fields_or_none(rope):
    problem = _one_problem(_array(positions=Positions(rope=rope)))
    assert problem.startswith("Array 'kv'.positions.rope") and "all set or all None" in problem


@pytest.mark.parametrize("positions", [
    Positions(),
    Positions(rope=Rope()),
    Positions(rope=Rope(low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192)),
])
def test_a_good_rope_has_no_problem(positions):
    assert state_declaration_problems(_StubModel(_array(positions=positions))) == []


def test_an_array_may_not_carry_two_updates_of_one_type():
    value = Value("gdn", nodes={"LLM"}, layers={0}, entry=_gdn_entry(),
                  updates={"a": CausalConv(), "b": CausalConv(activation=None)})
    problem = _one_problem(value)
    assert problem.startswith("Array 'gdn'.updates") and "2 updates of type CausalConv" in problem


def test_kv_labels_may_name_only_declared_labels():
    problem = _one_problem(_array(), kv_labels={"kv": {"main": KVLabel(), "cfg": KVLabel()}})
    assert problem.startswith("kv_labels()['kv']['cfg']") and "Array 'kv'.labels" in problem


def test_kv_labels_may_name_only_declared_arrays():
    problem = _one_problem(_array(), kv_labels={"other": {"main": None}})
    assert problem.startswith("kv_labels()['other']") and "no declared array" in problem


def test_a_sampler_node_must_be_in_the_model():
    problem = _one_problem(Sampler("s", nodes={"talker"}, vocab_size=8))
    assert problem.startswith("Sampler 's'.nodes") and "'talker' is not a node" in problem


@pytest.mark.parametrize("field, value", [("vocab_size", 0), ("draws", 0)])
def test_a_sampler_field_must_be_at_least_one(field, value):
    problem = _one_problem(Sampler("s", **{"nodes": {"LLM"}, "vocab_size": 8, field: value}))
    assert problem.startswith(f"Sampler 's'.{field}") and "at least 1" in problem


def test_every_problem_is_reported():
    problems = state_declaration_problems(_StubModel(_array(nodes={"dit"}), _array(nodes={"sit"})))
    assert len(problems) == 3, f"the clashing name and both unknown nodes: {problems}"


def _declarations() -> list[Array]:
    entry = Entry({
        "state": TensorType((4, 8, 8), dtype=torch.float32, shard=0),
        "conv": TensorType((32, 3), dtype=None, shard=0),
    }, host={"seen": tuple[int, ...], "done": bool})
    return [
        _array(reads={"local": _read(mask=Window(4), layers={0, 1}, walks={"prefill", "decode"}),
                      "global": _read(layers={2})}),
        Value("gdn", nodes={"LLM", "vision"}, layers={3, 5}, entry=entry,
              updates={"conv": CausalConv(), "delta": GatedDeltaNet(qk_l2norm=True, scale=0.5)}),
    ]


def _empty_value() -> Array:
    entry = Entry({"state": TensorType((4, 8, 8), dtype=torch.float32, shard=0)})
    return Value("gdn", nodes={"LLM"}, layers=set(), entry=entry, updates={"delta": GatedDeltaNet()})


def _within(name, **overrides) -> Array:
    return Array(name, **{"nodes": {"LLM"}, "layers": {0, 1}, "entry": KVEntry(heads=4, head_dim=8), "max_len": 0,
                          "reads": {f"{name}_attn": Read(mask=Block(), op=_ATTEND)}, **overrides})


def _cross(name="text", queries="video", **overrides) -> Array:
    return _within(name, reads={f"{name}_attn": Read(mask=Block(), op=_ATTEND, queries=queries)}, **overrides)


def test_a_cross_read_over_an_array_on_its_node_has_no_problems():
    assert state_declaration_problems(_StubModel(_within("video"), _cross())) == []


def test_a_cross_read_over_an_array_that_keeps_entries_has_no_problems():
    assert state_declaration_problems(_StubModel(_array("video", reads={"r": _read()}), _cross())) == []


def test_a_kept_cross_read_over_an_array_that_keeps_entries_has_no_problems():
    context = _array("context", reads={"context_attn": _read(mask=Bidirectional(), queries="kv")})
    assert state_declaration_problems(_StubModel(_array(reads={"r": _read()}), context)) == []


def test_a_bidirectional_read_needs_an_array_that_keeps_entries():
    problem = _one_problem(_within("vision", reads={"r": _read(mask=Bidirectional())}))
    assert "every read of an array with max_len 0 is a Block read, and this one is Bidirectional" in problem


def _chunked(*arrays, chunkable=True) -> _StubModel:
    model = _StubModel(*arrays)
    model.get_chunking_policies = lambda: {"LLM": (ChunkingPolicy(walks=("prefill",), chunkable=chunkable),)}
    return model


@pytest.mark.parametrize("array, mask", [
    (_array(reads={"r": _read(mask=Bidirectional())}), "Bidirectional"),
    (_within("vision", reads={"r": _read(mask=Block())}), "Block"),
], ids=["kept", "within_step"])
def test_a_chunked_walk_cant_read_this_steps_adds_as_a_whole(array, mask):
    """Splitting a row's adds across steps changes what they see, with no error from any kernel."""
    problems = state_declaration_problems(_chunked(array))
    assert len(problems) == 1
    assert problems[0].startswith(f"Array {array.name!r}.reads['r']: {mask} on walk 'prefill'")
    assert "which a chunking policy splits across steps" in problems[0]


@pytest.mark.parametrize("array", [
    _array(reads={"r": _read(mask=Bidirectional())}),
    _within("vision", reads={"r": _read(mask=Block())}),
], ids=["kept", "within_step"])
def test_a_policy_that_doesnt_split_rows_can_read_this_steps_adds_as_a_whole(array):
    assert state_declaration_problems(_chunked(array, chunkable=False)) == []


def test_a_chunked_walk_reads_causally_and_another_nodes_reads_stay_free():
    causal = _array(reads={"r": _read()})
    vision = _within("vision", nodes={"vision"}, reads={"v": _read(mask=Block())})
    assert state_declaration_problems(_chunked(causal, vision)) == []


def test_a_chunked_walk_outside_a_reads_walks_is_free():
    array = _array(reads={"r": _read(mask=Bidirectional(), walks={"decode"})})
    assert state_declaration_problems(_chunked(array)) == []


_REBUILD = {"LLM": RebuildPolicy(walk="prefill", ids_input="text_inputs", sample_output="token", loop="loop")}


def _keyed(growth=False) -> dict:
    keys = PrefixKeys("text_inputs", ("prefill",), growth=growth)
    return {"kv": {"main": KVLabel(growth=(("decode", "loop"),), keys=keys)}}


@pytest.mark.parametrize("kwargs, cause", [
    ({"kv_labels": _keyed()}, "kv_labels()['kv']['main'] keys for prefix reuse"),
    ({"rebuild": _REBUILD}, "LLM's rebuild policy runs again in one step"),
], ids=["prefix_keys", "rebuild"])
def test_a_walk_a_prefix_hit_or_a_rebuild_regroups_cant_read_this_steps_adds_as_a_whole(kwargs, cause):
    """A hit can end inside one step's adds, and a rebuild runs the prompt again beside the ids sampled after it."""
    problem = _one_problem(_array(reads={"r": _read(mask=Bidirectional())}), **kwargs)
    assert problem.startswith("Array 'kv'.reads['r']: Bidirectional on walk 'prefill'") and cause in problem


@pytest.mark.parametrize("growth, refused", [(True, 1), (False, 0)])
def test_a_keyed_growth_walk_cant_read_this_steps_adds_as_a_whole(growth, refused):
    """A growth step can add several entries per row, and with growth its sampled ids extend the keys."""
    array = _array(reads={"p": _read(walks={"prefill"}), "d": _read(mask=Bidirectional(), walks={"decode"})})
    problems = state_declaration_problems(_StubModel(array, kv_labels=_keyed(growth)))
    assert len(problems) == refused
    assert all(problem.startswith("Array 'kv'.reads['d']: Bidirectional on walk 'decode'") for problem in problems)


@pytest.mark.parametrize("other, mask", [
    (_array("image_kv", reads={"o": _read(mask=Bidirectional())}), "Bidirectional"),
    (_within("vision", reads={"o": _read(mask=Block())}), "Block"),
], ids=["kept", "within_step"])
def test_another_arrays_read_on_a_keyed_node_cant_read_this_steps_adds_as_a_whole(other, mask):
    """A hit trims the step for every array on the node, and the keyed entries came from layers that read the rest."""
    problem = _one_problem(_array(reads={"r": _read()}), other, kv_labels=_keyed())
    assert problem.startswith(f"Array {other.name!r}.reads['o']: {mask} on walk 'prefill'")


def test_keyed_and_rebuilt_walks_read_causally_have_no_problems():
    model = _StubModel(_array(reads={"r": _read()}), kv_labels=_keyed(growth=True), rebuild=_REBUILD)
    assert state_declaration_problems(model) == []


@pytest.mark.parametrize("arrays, where, cause", [
    ((_cross(queries="nothing"),), "reads['text_attn'].queries", "'nothing' is not a declared array"),
    ((_within("video", nodes={"vision"}), _cross()), "reads['text_attn'].queries", "isn't on nodes ['LLM']"),
])
def test_a_cross_read_is_refused_naming_its_array_and_the_cause(arrays, where, cause):
    problem = _one_problem(*arrays)
    assert problem.startswith("Array 'text'.") and where in problem and cause in problem


def _flag() -> Array:
    return Value("flags", nodes={"LLM"}, layers={0}, entry=Entry({}, host={"sent": bool}))


def test_a_value_of_host_fields_has_no_problems_on_a_node_without_prefix_keys():
    labels = {"kv": {"main": KVLabel()}}
    assert state_declaration_problems(_StubModel(_array(reads={"r": _read()}), _flag(), kv_labels=labels)) == []


def test_host_fields_are_refused_on_a_node_with_prefix_keys():
    labels = {"kv": {"main": KVLabel(keys=PrefixKeys("text_inputs", ("prefill",)))}}
    problem = _one_problem(_array(reads={"r": _read()}), _flag(), kv_labels=labels)
    assert problem.startswith("Array 'flags'.entry.host") and "prefix keys" in problem


def test_a_value_with_no_layers_has_no_problems():
    assert state_declaration_problems(_StubModel(_array(), _empty_value())) == []


def test_the_digest_tells_a_value_with_no_layers_from_one_with_a_layer():
    empty = _empty_value()
    with_layer = Value(empty.name, nodes=empty.nodes, layers={0}, entry=empty.entry, updates=empty.updates)
    assert declaration_digest([empty]) != declaration_digest([with_layer])


def test_the_digest_ignores_the_order_of_arrays_and_keys():
    a, b = _declarations()
    swapped = Array(a.name, nodes=a.nodes, layers=a.layers, entry=a.entry, max_len=a.max_len,
                    reads=dict(reversed(list(a.reads.items()))))
    assert declaration_digest([b, swapped]) == declaration_digest([a, b])


def test_the_digest_matches_across_processes_with_different_hash_seeds():
    script = (
        "import sys; sys.path.insert(0, 'test/modular'); sys.path.insert(0, '.')\n"
        "from test_state_declaration_checks import _declarations\n"
        "from mstar.engine.state.analysis import declaration_digest\n"
        "print(declaration_digest(_declarations()))\n"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        for seed in ("0", "1", "2024")
    }
    assert digests == {declaration_digest(_declarations())}, "the digest moved with PYTHONHASHSEED"


@dataclass(frozen=True)
class _OtherMask(Mask):
    """Has no fields, as `Causal` has none, so only its class tells them apart."""


_T = TensorType((4, 8), dtype=None, shard=0)
_READ = Read(mask=Causal(), op=_ATTEND)
_FIELD_CHANGES = {
    "mask class": _array(reads={"r": Read(mask=_OtherMask(), op=_ATTEND)}),
    "name": _array("kv2", reads={"r": _READ}),
    "nodes": _array(nodes={"vision"}, reads={"r": _READ}),
    "layers": _array(layers={0, 1, 3}, reads={"r": _READ}),
    "max_len": _array(max_len=17, reads={"r": _READ}),
    "rewritable": _array(rewritable=True, reads={"r": _READ}),
    "read key": _array(reads={"s": _READ}),
    "no reads": _array(),
    "mask": _array(reads={"r": Read(mask=Window(4), op=_ATTEND)}),
    "window size": _array(reads={"r": Read(mask=Window(5), op=_ATTEND)}),
    "op": _array(reads={"r": Read(mask=Causal(), op=Attend(q_heads=4))}),
    "no op": _array(reads={"r": Read(mask=Causal())}),
    "read layers": _array(reads={"r": Read(mask=Causal(), op=_ATTEND, layers={0})}),
    "other read layers": _array(reads={"r": Read(mask=Causal(), op=_ATTEND, layers={1})}),
    "read walks": _array(reads={"r": Read(mask=Causal(), op=_ATTEND, walks={"decode"})}),
    "other read walks": _array(reads={"r": Read(mask=Causal(), op=_ATTEND, walks={"prefill"})}),
    "tensor shape": _array(entry=KVEntry(heads=4, head_dim=16), reads={"r": _READ}),
    "tensor dtype": _array(entry=Entry({"k": TensorType((4, 8), dtype=torch.bfloat16, shard=0), "v": _T}),
                           reads={"r": Read(mask=Causal())}),
    "tensor shard": _array(entry=Entry({"k": TensorType((4, 8), dtype=None), "v": _T}),
                           reads={"r": Read(mask=Causal())}),
    "tensor name": _array(entry=Entry({"k": _T, "w": _T}), reads={"r": Read(mask=Causal())}),
    "labels": _array(labels={"main", "cfg"}, reads={"r": _READ}),
    "a block read": _array(max_len=0, reads={"r": Read(mask=Block(), op=_ATTEND)}),
    "a cross read": _array(max_len=0, reads={"r": Read(mask=Block(), op=_ATTEND, queries="video")}),
    "seq_sharded": _array(max_len=0, seq_sharded=True, reads={"r": Read(mask=Block(), op=_ATTEND)}),
    "a counter": _array(positions=Positions(), reads={"r": _READ}),
    "a rope": _array(positions=Positions(rope=Rope()), reads={"r": _READ}),
    **{
        f"rope {field}": _array(positions=Positions(rope=Rope(**{field: value})), reads={"r": _READ})
        for field, value in (
            ("theta", 5.0), ("rotary_dim", 8), ("interleave", True), ("scale", 2.0), ("low_freq_factor", 1.0),
            ("high_freq_factor", 4.0), ("old_context_len", 8192), ("dtype", torch.bfloat16),
            ("table_dtype", torch.float32),
        )
    },
}


def test_every_field_change_moves_the_digest():
    base = declaration_digest([_array(reads={"r": _READ})])
    digests = {name: declaration_digest([array]) for name, array in _FIELD_CHANGES.items()}
    assert all(digest != base for digest in digests.values()), [n for n, d in digests.items() if d == base]
    assert len(set(digests.values())) == len(digests), "two changes landed on one digest"


@pytest.mark.parametrize("host", [{"done": bool}, {"done": int}, {"sent": bool}, {"done": tuple[bool, ...]},
                                  {"done": Tuple[int, ...]}])
def test_a_host_field_change_moves_the_digest(host):
    def value(fields):
        return Value("v", nodes={"LLM"}, layers={0}, entry=Entry({}, host=fields))

    assert declaration_digest([value({})]) != declaration_digest([value(host)])
    assert declaration_digest([value(host)]) == declaration_digest([value(dict(host))])


def _sampler(**overrides) -> Sampler:
    return Sampler("s", **{"nodes": {"LLM"}, "vocab_size": 8, **overrides})


@pytest.mark.parametrize("changed", [
    _sampler(nodes={"vision"}), _sampler(vocab_size=9), _sampler(penalty=False), _sampler(min_p=True),
    _sampler(eos_ids=(1,)), _sampler(draws=2),
])
def test_a_sampler_field_change_moves_the_digest(changed):
    assert declaration_digest([_sampler()]) != declaration_digest([changed])


@pytest.mark.parametrize("first, second", [
    (CausalConv(), CausalConv(activation=None)),
    (GatedDeltaNet(), GatedDeltaNet(qk_l2norm=False)),
    (GatedDeltaNet(), GatedDeltaNet(scale=0.5)),
    (CausalConv(), GatedDeltaNet()),
])
def test_an_update_field_change_moves_the_digest(first, second):
    entry = Entry({
        "state": TensorType((4, 8, 8), dtype=torch.float32), "conv": TensorType((32, 3), dtype=None),
    })

    def value(update):
        return Value("gdn", nodes={"LLM"}, layers={0}, entry=entry, updates={"u": update})
    assert declaration_digest([value(first)]) != declaration_digest([value(second)])
