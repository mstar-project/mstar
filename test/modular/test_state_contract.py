"""A model's state declarations are values, and a bad one is refused where it is written.

The conductor, the data worker and the workers each call ``declare_state``, so a declaration has to be immutable and
hashable: a dict the model changes later must not change the array. A refusal names the array and the field, or a
model author reads a shape error from a worker minutes later.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

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
    Fork,
    GatedDeltaNet,
    KVEntry,
    PagedNHD,
    Positions,
    Read,
    Rope,
    Sampler,
    TensorType,
    Value,
    Window,
)
from mstar.engine.state.contract import Op

_ENTRY = KVEntry(heads=4, head_dim=8)
_GDN = Entry({
    "state": TensorType((4, 8, 8), dtype=torch.float32, shard=0),
    "conv": TensorType((32, 3), dtype=None, shard=0),
})


def _array(**overrides) -> Array:
    return Array("kv", **{"nodes": {"LLM"}, "layers": {0, 1, 2}, "entry": _ENTRY, "max_len": 16, **overrides})


def _read(**overrides) -> Read:
    return Read(**{"mask": Causal(), "op": Attend(q_heads=8), **overrides})


def _value(**overrides) -> Array:
    return Value("gdn", **{"nodes": {"LLM"}, "layers": {0}, "entry": _GDN, **overrides})


def _entry(k: TensorType, v: TensorType) -> Entry:
    return Entry({"k": k, "v": v})


def _holding(tensor: TensorType) -> Array:
    return _array(entry=Entry({"k": tensor}))


REFUSALS = {
    "list shape": (lambda: _holding(TensorType([4, 8], dtype=None)), "Array 'kv'.entry.tensors['k'].shape"),
    "zero in shape": (lambda: _holding(TensorType((4, 0), dtype=None)), "Array 'kv'.entry.tensors['k'].shape"),
    "bool in shape": (lambda: _holding(TensorType((True, 8), dtype=None)), "Array 'kv'.entry.tensors['k'].shape"),
    "shard past the shape": (lambda: _holding(TensorType((4, 8), dtype=None, shard=2)),
                             "Array 'kv'.entry.tensors['k'].shard"),
    "negative shard": (lambda: _holding(TensorType((4, 8), dtype=None, shard=-1)),
                       "Array 'kv'.entry.tensors['k'].shard"),
    "a value's tensor": (lambda: _value(entry=Entry({"conv": TensorType((32, 0), dtype=None)})),
                         "Array 'gdn'.entry.tensors['conv'].shape"),
    "host object": (lambda: _value(entry=Entry({}, host={"gen": torch.Generator})), "Array 'gdn'.entry.host['gen']"),
    "host list": (lambda: _value(entry=Entry({}, host={"ids": list})), "Array 'gdn'.entry.host['ids']"),
    "host tuple of objects": (lambda: _value(entry=Entry({}, host={"ids": tuple[object, ...]})),
                              "Array 'gdn'.entry.host['ids']"),
    "window of zero": (lambda: _array(reads={"r": _read(mask=Window(0))}), "Array 'kv'.reads['r'].mask.size"),
    "q_heads of zero": (lambda: _array(reads={"r": _read(op=Attend(q_heads=0))}), "Array 'kv'.reads['r'].op.q_heads"),
    "empty read layers": (lambda: _array(reads={"r": _read(layers=set())}), "Array 'kv'.reads['r'].layers"),
    "empty read walks": (lambda: _array(reads={"r": _read(walks=set())}), "Array 'kv'.reads['r'].walks"),
    "read layers from an empty generator": (lambda: _array(reads={"r": _read(layers=(i for i in ()))}),
                                            "Array 'kv'.reads['r'].layers"),
    "a mask as a read": (lambda: _array(reads={"r": Causal()}), "Array 'kv'.reads['r']"),
    "no nodes": (lambda: _array(nodes=set()), "Array 'kv'.nodes"),
    "no layers": (lambda: _array(layers=set()), "Array 'kv'.layers is empty, and only a Value"),
    "no layers on a rewritable log": (lambda: _array(layers=set(), rewritable=True),
                                      "Array 'kv'.layers is empty, and only a Value"),
    "no layers on one append-only entry": (lambda: _array(layers=set(), max_len=1),
                                           "Array 'kv'.layers is empty, and only a Value"),
    "negative axis with no layers": (
        lambda: _value(layers=set(), entry=Entry({"conv": TensorType((32, -1), dtype=None)})),
        "Array 'gdn'.entry.tensors['conv'].shape",
    ),
    "float axis with no layers": (
        lambda: _value(layers=set(), entry=Entry({"conv": TensorType((32, 0.0), dtype=None)})),
        "Array 'gdn'.entry.tensors['conv'].shape",
    ),
    "an axis of 0 under a read with no layers": (
        lambda: _array(layers=set(), max_len=1, rewritable=True, reads={"r": _read()},
                       entry=_entry(TensorType((0, 8), dtype=None), TensorType((0, 8), dtype=None))),
        "Array 'kv'.entry.tensors['k'].shape",
    ),
    "negative layer": (lambda: _array(layers={-1}), "Array 'kv'.layers"),
    "negative max_len": (lambda: _array(max_len=-1), "Array 'kv'.max_len"),
    "no labels": (lambda: _array(labels=set()), "Array 'kv'.labels"),
    "sampler with no nodes": (lambda: Sampler("s", nodes=set(), vocab_size=8), "Sampler 's'.nodes"),
    "read layer outside": (lambda: _array(reads={"r": _read(layers={7})}), "Array 'kv'.reads['r'].layers"),
    "attend without v": (
        lambda: _array(entry=Entry({"k": TensorType((4, 8), dtype=None)}), reads={"r": _read()}),
        "Array 'kv'.reads['r'].op",
    ),
    "attend on rank 3": (
        lambda: _array(entry=_entry(TensorType((4, 8, 8), dtype=None), TensorType((4, 8), dtype=None)),
                       reads={"r": _read()}),
        "Array 'kv'.reads['r'].op",
    ),
    "attend on unequal heads": (
        lambda: _array(entry=_entry(TensorType((4, 8), dtype=None), TensorType((2, 8), dtype=None)),
                       reads={"r": _read()}),
        "Array 'kv'.reads['r'].op",
    ),
    "attend on two dtypes": (
        lambda: _array(entry=_entry(TensorType((4, 8), dtype=torch.float32), TensorType((4, 8), dtype=None)),
                       reads={"r": _read()}),
        "Array 'kv'.reads['r'].op",
    ),
    "attend on two shards": (
        lambda: _array(entry=_entry(TensorType((4, 8), dtype=None, shard=0), TensorType((4, 8), dtype=None)),
                       reads={"r": _read()}),
        "Array 'kv'.reads['r'].op",
    ),
    "attend sharded off the heads": (
        lambda: _array(entry=_entry(TensorType((4, 8), dtype=None, shard=1), TensorType((4, 8), dtype=None, shard=1)),
                       reads={"r": _read()}),
        "Array 'kv'.reads['r'].op",
    ),
    "q_heads off the heads": (lambda: _array(reads={"r": _read(op=Attend(q_heads=6))}), "Array 'kv'.reads['r'].op"),
    "update off a Value": (lambda: _array(updates={"u": CausalConv()}), "Array 'kv'.updates['u']"),
    "host fields off a Value": (lambda: _array(entry=Entry({}, host={"sent": bool})), "Array 'kv'.entry.host"),
    "queries with a mask that uses the query's position": (
        lambda: _array(reads={"r": _read(queries="video")}), "Array 'kv'.reads['r'].queries",
    ),
    "queries with no op": (
        lambda: _array(reads={"r": Read(mask=Bidirectional(), queries="video")}),
        "Array 'kv'.reads['r'].queries: a read with no op",
    ),
    "queries naming the read's own array": (
        lambda: _array(reads={"r": _read(mask=Bidirectional(), queries="kv")}),
        "Array 'kv'.reads['r'].queries names this array",
    ),
    "queries naming the read's own array, within the step": (
        lambda: _array(max_len=0, reads={"r": _read(mask=Block(), queries="kv")}),
        "Array 'kv'.reads['r'].queries names this array",
    ),
    "update on one entry that isn't rewritable": (
        lambda: _array(entry=_GDN, max_len=1, updates={"u": CausalConv()}),
        "Array 'kv'.updates['u']: only a Value",
    ),
    "update of another kind": (lambda: _value(updates={"u": Attend(q_heads=8)}), "Array 'gdn'.updates['u']"),
    "conv without its field": (
        lambda: _value(entry=Entry({"state": _GDN.tensors["state"]}), updates={"u": CausalConv()}),
        "Array 'gdn'.updates['u']",
    ),
    "conv of rank 3": (
        lambda: _value(entry=Entry({"conv": TensorType((4, 8, 8), dtype=None)}), updates={"u": CausalConv()}),
        "Array 'gdn'.updates['u']",
    ),
    "delta net without its field": (
        lambda: _value(entry=Entry({"conv": _GDN.tensors["conv"]}), updates={"u": GatedDeltaNet()}),
        "Array 'gdn'.updates['u']",
    ),
    "delta net of rank 2": (
        lambda: _value(entry=Entry({"state": TensorType((4, 8), dtype=None)}), updates={"u": GatedDeltaNet()}),
        "Array 'gdn'.updates['u']",
    ),
}


@pytest.mark.parametrize("name", REFUSALS)
def test_a_bad_declaration_is_refused_naming_the_array_and_field(name):
    build, where = REFUSALS[name]
    with pytest.raises(ValueError) as refused:
        build()
    assert str(refused.value).startswith(where), f"the refusal should open with {where}: {refused.value}"


@pytest.mark.parametrize("build, where", [
    (lambda: Read(mask="nope"), "Read.mask"),
    (lambda: Read(mask=Causal(), op="garbage"), "Read.op"),
    (lambda: Read(mask=Causal(), op=CausalConv()), "Read.op: a read changes no field"),
])
def test_a_bad_mask_or_op_is_refused_when_the_read_is_built(build, where):
    with pytest.raises(ValueError) as refused:
        build()
    assert str(refused.value).startswith(where), f"the refusal should open with {where}: {refused.value}"


@dataclass(frozen=True)
class _Decay(Op):
    """An update kernel the contract doesn't define yet, on a state of rank 3."""

    writes = ("state",)

    def check(self, where, entry):
        if len(entry.tensors["state"].shape) != 3:
            raise ValueError(f"{where}: _Decay needs a state of rank 3")


def test_a_new_op_brings_its_own_check():
    _value(updates={"u": _Decay()})
    with pytest.raises(ValueError, match=re.escape("Array 'gdn'.updates['u']: _Decay needs a state of rank 3")):
        _value(entry=Entry({"state": TensorType((4, 8), dtype=None)}), updates={"u": _Decay()})


@pytest.mark.parametrize("build", [
    lambda: _array(reads={"local": _read(mask=Window(4), layers={0, 1}), "global": _read(layers={2})}),
    lambda: _value(updates={"conv": CausalConv(), "delta": GatedDeltaNet(qk_l2norm=True)}),
    lambda: Entry({}, host={"eos_sent": bool, "seen": tuple[int, ...], "pair": tuple[int, str]}),
    lambda: _array(labels={"main", "cfg"}, positions=Positions(rope=Rope(theta=5.0, dtype=torch.bfloat16))),
    lambda: _array(max_len=0, reads={"r": _read(mask=Block())}),
    lambda: Sampler("s", nodes={"LLM"}, vocab_size=8, eos_ids=(1, 2), draws=15),
])
def test_a_declaration_hashes_and_equals_one_built_the_same(build):
    assert hash(build()) == hash(build())
    assert build() == build(), "two processes building one declaration should get equal values"


def test_changing_a_dict_after_passing_it_changes_nothing():
    tensors = {"k": TensorType((4, 8), dtype=None, shard=0), "v": TensorType((4, 8), dtype=None, shard=0)}
    reads = {"r": _read()}
    array = _array(entry=Entry(tensors), reads=reads)
    tensors["w"] = tensors["k"]
    reads["other"] = _read()
    assert set(array.entry.tensors) == {"k", "v"} and set(array.reads) == {"r"}, "the array kept a live dict"


def test_changing_a_set_after_passing_it_changes_nothing():
    layers, nodes = {0, 1}, {"LLM"}
    read = Read(mask=Causal(), layers=layers)
    array = _array(nodes=nodes, layers=layers)
    layers.add(9)
    nodes.add("other")
    assert read.layers == {0, 1} and array.layers == {0, 1} and array.nodes == {"LLM"}, "a live set was kept"


def test_a_declaration_cannot_be_assigned_to():
    array = _array()
    with pytest.raises(AttributeError):
        array.max_len = 1
    with pytest.raises(TypeError):
        array.reads["r"] = _read()


def test_an_array_opens_main_unless_it_declares_other_labels():
    assert _array().labels == {"main"}
    assert _array(labels=["main", "cfg"]).labels == frozenset({"main", "cfg"})


def test_a_value_is_a_rewritable_array_of_one_entry():
    value = _value()
    assert value.rewritable and value.max_len == 1


def test_a_value_with_no_layers_may_hold_an_empty_axis():
    conv = TensorType((32, 0), dtype=None, shard=0)
    assert _value(layers=set(), entry=Entry({"conv": conv})).entry.tensors["conv"] == conv, (
        "a model with no linear layers would be refused at a conv kernel of 1"
    )


def test_kv_entry_is_k_and_v_sharded_on_the_heads():
    t = TensorType((4, 8), dtype=None, shard=0)
    assert KVEntry(heads=4, head_dim=8) == Entry({"k": t, "v": t})


def test_an_entry_may_hold_only_host_fields():
    assert Entry({}, host={"eos_sent": bool}).tensors == {}, "a Value of host fields has no tensors"


def test_a_paged_format_is_a_value_whatever_set_it_is_given():
    assert PagedNHD(devices=["cuda"]) == PagedNHD(devices={"cuda"})
    assert hash(PagedNHD(dtypes={torch.bfloat16})) == hash(PagedNHD(dtypes=[torch.bfloat16]))


def test_a_fork_refuses_a_time_it_doesnt_have():
    assert Fork(src="main", dst="cfg", at="end").at == "end"
    with pytest.raises(ValueError, match="Fork.at must be 'start' or 'end', got 'begin'"):
        Fork(src="main", dst="cfg", at="begin")


def test_a_sampler_with_a_list_of_eos_ids_stays_hashable():
    sampler = Sampler("s", nodes={"LLM"}, vocab_size=8, eos_ids=[1, 2])
    assert hash(sampler) == hash(Sampler("s", nodes={"LLM"}, vocab_size=8, eos_ids=(1, 2)))


def test_a_scalar_field_declares_an_empty_shape():
    assert _value(entry=Entry({"count": TensorType((), dtype=torch.int64)})).entry.tensors["count"].shape == ()


def test_a_scalar_field_has_no_axis_to_shard():
    with pytest.raises(ValueError, match=r"shard must be None or an axis of shape \(\)"):
        _value(entry=Entry({"count": TensorType((), dtype=torch.int64, shard=0)}))


def test_a_scalar_field_has_no_kv_layout():
    with pytest.raises(ValueError, match="Attend needs entry fields"):
        _array(entry=_entry(TensorType((), dtype=None), TensorType((), dtype=None)), reads={"attn": _read()})
