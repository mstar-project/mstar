"""The engine builds today's specs from a model's declarations, and refuses what today's specs can't hold.

Specs stay the boundary in this series: the pool, the attention backends, the prefix cache and admission all read them.
So a declared field with no spec field must be refused, or the engine would silently run something else, and a window's
layers must convert to KV layer indices, or a model with KV on some of its layers would window the wrong ones.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, replace

sys.path.insert(0, ".")
sys.path.insert(0, "test/modular")

import pytest
import torch
import yaml
from test_startup_graph_validation import _prefill_decode, _ToyModel, _validate
from test_state_ports import QWEN3_5, _qwen3_5

from mstar.engine.resources.attn.config import AttentionSpec
from mstar.engine.resources.kv.bounded.config import BoundedKVSpec, SinkWindow
from mstar.engine.resources.kv.config import KVSpec
from mstar.engine.resources.linear_attn.config import LinearAttnSpec, LinearAttnVariant
from mstar.engine.resources.recurrent.config import RecurrentBlockConfig, RecurrentStateSpec
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.engine.resources.spec import CombinedSplit, apply_yaml_overrides, spec_dependency_problems
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
    Mamba2,
    Positions,
    Read,
    Rope,
    Sampler,
    TensorType,
    Value,
    Window,
    position,
)
from mstar.engine.state.adapters import (
    BINDINGS,
    VALUE,
    BoundedHandle,
    LinearAttnBinding,
    StateSetup,
    _resolve,
    build_specs,
    node_handles,
    node_resources,
    state_setup,
)
from mstar.engine.state.contract import Mask, Op
from mstar.model.base import Model, enabled_resources, serving_resources
from mstar.model.qwen3_5.config import GDN_STATE
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.validation import DeploymentError

_SPLIT = CombinedSplit("mixed", prefill_walks=("prefill",))


def _read(**overrides) -> Read:
    return Read(**{"mask": Causal(), "op": Attend(q_heads=8), **overrides})


def _kv(**overrides) -> Array:
    return Array("kv", **{
        "nodes": {"llm"}, "layers": {2, 5, 7, 9}, "entry": KVEntry(heads=4, head_dim=16), "max_len": 512,
        "reads": {"attn": _read()}, **overrides,
    })


def _gdn_entry(state_dtype=torch.float32, **host) -> Entry:
    return Entry({
        "state": TensorType((8, 128, 64), dtype=state_dtype, shard=0),
        "conv": TensorType((1024, 3), dtype=None, shard=0),
    }, **host)


def _gdn(**overrides) -> Array:
    return Value("gdn", **{
        "nodes": {"llm"}, "layers": {0, 1, 3}, "entry": _gdn_entry(),
        "updates": {"conv": CausalConv(), "delta": GatedDeltaNet()}, **overrides,
    })


def _ssm_entry(ssm_dtype=torch.float32, conv_channels=4 * 16 + 2 * 16, **extra) -> Entry:
    """A Mamba-2 Value's entry, 4 heads of 16 over a state of 16 in one group, which a conv of 96 channels matches."""
    return Entry({
        "ssm": TensorType((4, 16, 16), dtype=ssm_dtype, shard=0),
        "conv": TensorType((conv_channels, 3), dtype=None, shard=0),
        **extra,
    })


def _mamba(**overrides) -> Array:
    return Value("mamba", **{
        "nodes": {"llm"}, "layers": {2, 4}, "entry": _ssm_entry(),
        "updates": {"conv": CausalConv(), "ssm": Mamba2()}, **overrides,
    })


def _by_key(*arrays, **kwargs) -> dict:
    return {spec.resource_key: spec for spec in build_specs(arrays, **kwargs)}


def test_an_append_only_array_becomes_one_pool_and_one_attention_per_read():
    specs = _by_key(_kv(reads={"local": _read(mask=Window(4), layers={2}), "global": _read(layers={5, 7, 9})}))
    assert set(specs) == {"kv", "local", "global"}
    assert isinstance(specs["kv"], KVSpec)
    assert isinstance(specs["local"], AttentionSpec) and isinstance(specs["global"], AttentionSpec)
    assert specs["kv"].nodes == specs["local"].nodes == {"llm"}, "a spec's nodes are the array's"


def test_a_pool_takes_its_geometry_from_the_entry_and_its_query_heads_from_the_reads():
    config = _by_key(_kv())["kv"].config
    assert (config.num_layers, config.num_kv_heads, config.head_dim, config.max_seq_len, config.num_qo_heads) == (
        4, 4, 16, 512, 8,
    )


def test_an_attention_names_the_array_as_its_cache():
    assert _by_key(_kv())["attn"].config.kv_cache == "kv"


def test_a_causal_read_sets_no_window_and_no_layers():
    config = _by_key(_kv(reads={"attn": _read(layers={2, 5})}))["attn"].config
    assert config.sliding_window is None and config.layers is None, "only a window names its layers"


def test_a_window_read_sets_its_size():
    assert _by_key(_kv(reads={"attn": _read(mask=Window(6), layers={2})}))["attn"].config.sliding_window == 6


def test_a_window_read_counts_its_layers_among_the_array_layers():
    config = _by_key(_kv(reads={"attn": _read(mask=Window(6), layers={9, 5})}))["attn"].config
    assert config.layers == (1, 3), "model layers 5 and 9 are the pool's layers 1 and 3"


def test_a_window_read_of_every_layer_lists_them_all():
    assert _by_key(_kv(reads={"attn": _read(mask=Window(6))}))["attn"].config.layers == (0, 1, 2, 3)


def test_a_value_becomes_one_pool_and_one_linear_attention_under_the_delta_key():
    specs = _by_key(_kv(), _gdn())
    assert set(specs) == {"kv", "attn", "gdn", "delta"}, "a CausalConv runs through the delta's spec and has none"
    assert isinstance(specs["gdn"], RecurrentStateSpec) and isinstance(specs["delta"], LinearAttnSpec)


def test_a_value_may_declare_no_conv_update():
    assert set(_by_key(_kv(), _gdn(updates={"delta": GatedDeltaNet()}))) == {"kv", "attn", "gdn", "delta"}


def test_a_value_with_no_layers_builds_a_pool_of_no_layers_and_its_linear_attention():
    specs = _by_key(_kv(), _gdn(layers=set()), kernel_settings={"delta": {"splits": (_SPLIT,)}})
    assert specs["gdn"].config.num_layers == 0 and specs["gdn"].config.kv_cache == "kv"
    assert specs["delta"].config.recurrent_state == "gdn" and specs["delta"].config.splits == (_SPLIT,), (
        "a model's kernel settings don't depend on its having linear layers"
    )


def test_a_pool_with_no_layers_fingerprints_as_one_with_layers_does():
    def fingerprint(layers) -> bytes:
        config = replace(_by_key(_kv(), _gdn(layers=layers))["gdn"].config, max_slots=4)
        return RecurrentStatePool(torch.device("cpu"), config, kv_dtype=torch.bfloat16).fingerprint()

    assert fingerprint(set()) == fingerprint({0, 1}), "the fingerprint names only each block's shape and stored dtype"


def test_a_pool_links_to_the_one_append_only_array_on_its_nodes():
    config = _by_key(_kv(), _gdn())["gdn"].config
    assert config.num_layers == 3 and config.kv_cache == "kv"


def test_the_order_of_the_declarations_does_not_change_the_link():
    assert _by_key(_gdn(), _kv())["gdn"].config.kv_cache == "kv"


def test_an_append_only_array_on_other_nodes_does_not_stop_the_link():
    other = Array("vision_kv", nodes={"vision"}, layers={0}, entry=KVEntry(heads=4, head_dim=16), max_len=512,
                  reads={"vision_attn": _read()})
    assert _by_key(other, _kv(), _gdn())["gdn"].config.kv_cache == "kv", "only an array on the value's nodes can link"


def test_a_cross_read_array_on_the_value_nodes_does_not_stop_the_link():
    context = Array("context", nodes={"llm"}, layers={2, 5}, entry=KVEntry(heads=4, head_dim=16), max_len=64,
                    reads={"cross": _read(mask=Bidirectional(), queries="kv")})
    assert _by_key(_kv(), context, _gdn())["gdn"].config.kv_cache == "kv", "a context holds no row's own entries"


def test_a_value_beside_an_array_whose_reads_differ_by_walk_links_to_its_pages():
    by_walk = _kv(reads={"attn": _read(walks={"prefill"}), "attn_decode": _read(walks={"decode"})})
    assert _by_key(by_walk, _gdn())["gdn"].config.kv_cache == "kv", "its pages hold each row's own entries"


def test_a_value_specs_sit_on_the_value_nodes_not_the_pool_nodes():
    specs = _by_key(_kv(nodes={"llm", "audio"}), _gdn())
    assert specs["gdn"].nodes == specs["delta"].nodes == {"llm"}, "the state would be built where the value isn't"


def test_the_state_block_is_stored_k_last():
    block = _by_key(_kv(), _gdn())["gdn"].config.blocks["state"]
    assert block == RecurrentBlockConfig(shape=(8, 64, 128), dtype=torch.float32, shard_dims=(0,))


@pytest.mark.parametrize("shard, stored", [(1, 2), (2, 1)])
def test_a_state_sharded_on_a_swapped_axis_follows_the_swap(shard, stored):
    entry = Entry({
        "state": TensorType((8, 128, 64), dtype=torch.float32, shard=shard), "conv": TensorType((1024, 3), dtype=None),
    })
    assert _by_key(_kv(), _gdn(entry=entry))["gdn"].config.blocks["state"].shard_dims == (stored,)


def test_a_block_with_no_declared_dtype_keeps_none_for_the_pool_to_resolve():
    block = _by_key(_kv(), _gdn())["gdn"].config.blocks["conv"]
    assert block == RecurrentBlockConfig(shape=(1024, 3), dtype=None, shard_dims=(0,))


@pytest.mark.parametrize("kv_dtype, stored", [
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16),
    (None, torch.get_default_dtype()),    # autocast off
])
def test_the_pool_stores_a_none_block_in_the_engine_dtype(kv_dtype, stored):
    config = replace(_by_key(_kv(), _gdn())["gdn"].config, max_slots=4)
    assert RecurrentStatePool(torch.device("cpu"), config, kv_dtype=kv_dtype)._blocks["conv"].dtype is stored


def test_a_block_with_a_declared_dtype_is_stored_in_it():
    entry = Entry({
        "state": TensorType((8, 128, 64), dtype=torch.float32), "conv": TensorType((1024, 3), dtype=torch.float16),
    })
    assert _by_key(_kv(), _gdn(entry=entry))["gdn"].config.blocks["conv"].dtype is torch.float16


def test_a_replicated_tensor_has_no_shard_dims():
    entry = Entry({
        "state": TensorType((8, 128, 64), dtype=torch.float32), "conv": TensorType((1024, 3), dtype=None),
    })
    blocks = _by_key(_kv(), _gdn(entry=entry))["gdn"].config.blocks
    assert blocks["state"].shard_dims == () and blocks["conv"].shard_dims == ()


@pytest.mark.parametrize("declared, head_k_dim, stored", [
    (torch.bfloat16, 128, torch.bfloat16),
    (torch.bfloat16, 64, torch.float32),    # no bf16 decode kernel off K = V = 128
    (torch.float16, 128, torch.float32),    # no fp16 decode kernel at all
])
def test_a_state_is_stored_in_its_declared_dtype_only_where_the_decode_kernel_takes_it(declared, head_k_dim, stored):
    entry = Entry({
        "state": TensorType((8, head_k_dim, 128), dtype=declared), "conv": TensorType((1024, 3), dtype=None),
    })
    assert _by_key(_kv(), _gdn(entry=entry))["gdn"].config.blocks["state"].dtype is stored


def test_a_state_with_no_declared_dtype_is_refused_naming_the_array_and_the_field():
    with pytest.raises(ValueError, match=re.escape("Array 'gdn'.entry.tensors['state'].dtype must be declared")):
        _by_key(_kv(), _gdn(entry=_gdn_entry(state_dtype=None)))


def test_the_yaml_state_dtype_applies_after_build_specs():
    specs = build_specs([_kv(), _gdn()])
    apply_yaml_overrides(specs, {"resources": {"gdn": {"state_dtype": "bfloat16", "max_slots": 9}}})
    config = next(spec for spec in specs if spec.resource_key == "gdn").config
    assert config.blocks["state"].dtype is torch.bfloat16 and config.max_slots == 9


def test_a_linear_attention_takes_its_fields_from_the_update():
    config = _by_key(_kv(), _gdn(updates={"delta": GatedDeltaNet(qk_l2norm=False, scale=0.25)}))["delta"].config
    assert (config.recurrent_state, config.variant) == ("gdn", LinearAttnVariant.GDN)
    assert config.qk_l2norm is False and config.sm_scale == 0.25


def test_a_gated_delta_net_defaults_to_the_config_defaults():
    config = _by_key(_kv(), _gdn())["delta"].config
    assert config.qk_l2norm is True and config.sm_scale is None


@dataclass(frozen=True)
class _OtherMask(Mask):
    """A mask `build_specs` has no layout for."""


@dataclass(frozen=True)
class _OtherUpdate(Op):
    """An update kernel `build_specs` has no layout for."""

    writes = ("state",)


_MIXED = _read(op=Attend(q_heads=4))
# The array each refusal names, its declarations, and what the message says is missing.
NO_LAYOUT = {
    "a read through no op": ("kv", lambda: [_kv(reads={"r": _read(op=None)})], "reads['r'].op"),
    "a cross read of one walk": ("ctx", lambda: [_kv(), Array(
        "ctx", nodes={"llm"}, layers={2, 5, 7, 9}, entry=KVEntry(heads=4, head_dim=16), max_len=512,
        reads={"x": _read(mask=Bidirectional(), queries="kv", walks={"decode"})},
    )], "reads['x'].walks"),
    "a mask of no known name": ("kv", lambda: [_kv(reads={"r": _read(mask=_OtherMask())})], "reads['r'].mask"),
    "two query head counts": ("kv", lambda: [_kv(reads={"a": _read(layers={2}), "b": _MIXED})], "q_heads"),
    "no reads": ("kv", lambda: [_kv(reads={})], "reads is empty"),
    "an entry of other fields": (
        "kv", lambda: [_kv(entry=Entry({"x": TensorType((4, 16), dtype=None)}), reads={})], "entry",
    ),
    "a third field": (
        "kv",
        lambda: [_kv(entry=Entry({
            "k": TensorType((4, 16), dtype=None, shard=0), "v": TensorType((4, 16), dtype=None, shard=0),
            "w": TensorType((4,), dtype=None),
        }))],
        "k and v",
    ),
    "k and v of two head dims": (
        "kv",
        lambda: [_kv(entry=Entry({
            "k": TensorType((4, 16), dtype=None, shard=0), "v": TensorType((4, 32), dtype=None, shard=0),
        }))],
        "one shape",
    ),
    "a declared dtype": (
        "kv",
        lambda: [_kv(entry=Entry({
            "k": TensorType((4, 16), dtype=torch.bfloat16, shard=0),
            "v": TensorType((4, 16), dtype=torch.bfloat16, shard=0),
        }))],
        "no dtype",
    ),
    "replicated heads": (
        "kv",
        lambda: [_kv(entry=Entry({"k": TensorType((4, 16), dtype=None), "v": TensorType((4, 16), dtype=None)}))],
        "sharded on",
    ),
    "a rewritable log": ("kv", lambda: [_kv(rewritable=True)], "rewritable"),
    "a read on a value": ("gdn", lambda: [_kv(), Array(
        "gdn", nodes={"llm"}, layers={0}, entry=_gdn_entry(), max_len=1, rewritable=True,
        reads={"r": Read(mask=Causal())},
    )], "today's resources run no read on this array"),
    "a value of one tensor": (
        "gdn",
        lambda: [_kv(), _gdn(entry=Entry({"state": TensorType((8, 128, 64), dtype=None)}),
                             updates={"delta": GatedDeltaNet()})],
        "entry",
    ),
    "a value with a host field": ("gdn", lambda: [_kv(), _gdn(entry=_gdn_entry(host={"done": bool}))], "updates"),
    "a value with only a conv": ("gdn", lambda: [_kv(), _gdn(updates={"conv": CausalConv()})], "GatedDeltaNet"),
    "an update of another kernel": (
        "gdn", lambda: [_kv(), _gdn(updates={"delta": GatedDeltaNet(), "x": _OtherUpdate()})], "updates ['x']",
    ),
    "two delta nets": (
        "gdn", lambda: [_kv(), _gdn(updates={"a": GatedDeltaNet(), "b": GatedDeltaNet()})], "GatedDeltaNet",
    ),
    "two convs": (
        "gdn",
        lambda: [_kv(), _gdn(updates={"a": CausalConv(), "b": CausalConv(), "delta": GatedDeltaNet()})],
        "CausalConv",
    ),
    "a conv with no activation": (
        "gdn",
        lambda: [_kv(), _gdn(updates={"a": CausalConv(activation=None), "delta": GatedDeltaNet()})],
        "LinearAttnConfig has no activation",
    ),
    "no append-only array": ("gdn", lambda: [_gdn()], "append-only"),
    "two append-only arrays": ("gdn", lambda: [_kv(), _kv(reads={"attn2": _read()}), _gdn()], "append-only"),
    "an append-only array on other nodes": ("gdn", lambda: [_kv(nodes={"other"}), _gdn()], "append-only"),
    "an append-only array on some nodes": ("gdn", lambda: [_kv(), _gdn(nodes={"llm", "other"})], "append-only"),
}


@pytest.mark.parametrize("name", NO_LAYOUT)
def test_what_todays_specs_cant_hold_is_refused_naming_what_is_missing(name):
    array, arrays, missing = NO_LAYOUT[name]
    with pytest.raises(ValueError, match=f"Array '{array}': no layout yet") as refused:
        build_specs(arrays())
    assert missing in str(refused.value), f"should name {missing!r}: {refused.value}"


def test_a_second_update_of_one_type_is_refused_as_no_layout_yet():
    with pytest.raises(ValueError, match=re.escape(
        "Array 'gdn': no layout yet: updates['a'] and updates['b'] are both GatedDeltaNet, and an update call finds "
        "its op by type",
    )):
        build_specs([_kv(), _gdn(updates={"a": GatedDeltaNet(), "b": GatedDeltaNet()})])


@dataclass(frozen=True)
class _OtherRecurrence(Op):
    """A second recurrence on the state, so a Value can carry two linear attention updates."""

    reads = writes = ("state",)


class _OtherLinear(LinearAttnBinding):
    """Binds `_OtherRecurrence` as a linear attention update, as `GDN` binds a `GatedDeltaNet`."""


def test_an_op_with_no_binding_on_its_storage_is_refused_as_no_layout_yet():
    refusal = "Array 'gdn': no layout yet: updates ['other'] must be CausalConv or GatedDeltaNet or Mamba2"
    with pytest.raises(ValueError, match=re.escape(refusal)):
        build_specs([_kv(), _gdn(updates={"delta": GatedDeltaNet(), "other": _OtherRecurrence()})])


def test_resolve_records_the_resource_every_op_runs_on():
    resolved = _resolve([_kv(positions=Positions()), _gdn()], StateSetup())
    assert resolved.resource_of == {"attn": "attn", "rope": "rope", "delta": "delta", "conv": "delta"}, (
        "an op with no resource here falls back to its own key, and the conv has no resource of its own"
    )


@pytest.mark.parametrize("field, refusal", [
    ({"seq_sharded": True}, "seq_sharded takes max_len 0"),
    ({"positions": Positions()}, "positions, which only an append-only array that keeps entries holds"),
])
def test_a_host_value_is_refused_what_only_another_storage_holds(field, refusal):
    flags = replace(Value("flags", nodes={"talker"}, layers={0}, entry=Entry({}, host={"sent": bool})), **field)
    with pytest.raises(ValueError, match=f"Array 'flags': no layout yet: {refusal}"):
        build_specs([flags])


def test_a_within_step_read_of_another_mask_is_refused_as_no_layout_yet():
    within = Array("vision", nodes={"llm"}, layers={0}, entry=KVEntry(heads=4, head_dim=16), max_len=0,
                   reads={"vision_attn": _read(mask=Causal())})
    refusal = "Array 'vision': no layout yet: reads['vision_attn'].mask must be Block"
    with pytest.raises(ValueError, match=re.escape(refusal)):
        build_specs([within])


@pytest.mark.parametrize("updates, linear", [
    ({"conv": CausalConv()}, "[]"),
    ({"conv": CausalConv(), "delta": GatedDeltaNet(), "other": _OtherRecurrence()}, "['delta', 'other']"),
], ids=["none", "two"])
def test_a_conv_needs_exactly_one_linear_attention_update_to_run_inside(monkeypatch, updates, linear):
    monkeypatch.setitem(BINDINGS, (VALUE, _OtherRecurrence), _OtherLinear())
    with pytest.raises(ValueError, match=re.escape(
        f"Array 'gdn': no layout yet: updates['conv'] is a CausalConv, which today's resources run inside one linear "
        f"attention update (GatedDeltaNet, Mamba2, _OtherRecurrence), and the array has {linear}",
    )):
        build_specs([_kv(), _gdn(updates=updates)])


def test_kernel_settings_set_the_two_fields_on_the_spec_under_the_key():
    specs = _by_key(_kv(), _gdn(), kernel_settings={
        "attn": {"splits": (_SPLIT,), "decode_fmha": True}, "delta": {"splits": (_SPLIT,)},
    })
    assert specs["attn"].config.splits == (_SPLIT,) and specs["attn"].config.decode_fmha is True
    assert specs["delta"].config.splits == (_SPLIT,)


def test_with_no_kernel_settings_the_fields_keep_their_defaults():
    specs = _by_key(_kv(), _gdn())
    assert specs["attn"].config.splits == () and specs["attn"].config.decode_fmha is False
    assert specs["delta"].config.splits == ()


@pytest.mark.parametrize("settings, word", [
    ({"nope": {"splits": ()}}, "'nope'"),
    ({"kv": {"splits": ()}}, "'kv'"),    # a kernel choice under an array's name
    ({"conv": {"splits": ()}}, "'conv'"),    # a CausalConv has no spec of its own
    ({"attn": {"backend": "sdpa"}}, "backend"),
    ({"attn": {"sliding_window": 4}}, "sliding_window"),
    ({"delta": {"decode_fmha": True}}, "decode_fmha"),    # only on an Attend read
    ({"delta": {"sm_scale": 1.0}}, "sm_scale"),
])
def test_kernel_settings_refuse_anything_else(settings, word):
    with pytest.raises(ValueError, match="kernel_settings") as refused:
        build_specs([_kv(), _gdn()], kernel_settings=settings)
    assert word in str(refused.value)


@pytest.mark.parametrize("key, array", [("attn", "kv"), ("delta", "gdn")])
def test_a_refused_kernel_setting_names_the_array_its_key_belongs_to(key, array):
    with pytest.raises(ValueError, match=f"Array '{array}'"):
        build_specs([_kv(), _gdn()], kernel_settings={key: {"sm_scale": 1.0}})


def _within(**overrides) -> Array:
    return Array("vision", **{
        "nodes": {"llm"}, "layers": {0}, "entry": KVEntry(heads=4, head_dim=16), "max_len": 0,
        "reads": {"vision_attn": _read(mask=Block())}, **overrides,
    })


_FLAG = Value("flag", nodes={"llm"}, layers={0}, entry=Entry({}, host={"seen": bool}))


def test_a_hook_page_count_builds_the_spec_the_same_key_under_resources_builds():
    hooked = build_specs([_kv()], kernel_settings={"kv": {"max_num_pages": 256}})
    written = build_specs([_kv()])
    apply_yaml_overrides(written, {"resources": {"kv": {"max_num_pages": 256}}})
    apply_yaml_overrides(hooked, {})
    assert hooked[0].config == written[0].config and hooked[0].config.max_num_pages == 256


def test_a_hook_slot_count_builds_the_spec_the_same_key_under_resources_builds():
    hooked = build_specs([_kv(), _gdn()], kernel_settings={"gdn": {"max_slots": 64}})
    written = build_specs([_kv(), _gdn()])
    apply_yaml_overrides(written, {"resources": {"gdn": {"max_slots": 64}}})
    apply_yaml_overrides(hooked, {})
    assert hooked[2].config == written[2].config and hooked[2].config.max_slots == 64


def test_a_hook_slot_count_sizes_qwen3_5s_value_and_the_yaml_overrides_it():
    class _Sized(Qwen3_5DenseModel):
        def kernel_settings(self):
            return {**super().kernel_settings(), GDN_STATE: {"max_slots": 64}}

    model = _Sized.__new__(_Sized)
    model.config = _qwen3_5(**QWEN3_5["9B"]).config
    model.vision_config = None

    def slots(config: dict) -> int:
        return next(s for s in serving_resources(model, config) if s.resource_key == GDN_STATE).config.max_slots

    assert slots({}) == 64, "the hook is the default"
    assert slots({"resources": {GDN_STATE: {"max_slots": 128}}}) == 128, "a deployment overrides the hook"


def test_a_yaml_page_count_overrides_the_hook_as_a_server_builds_the_specs():
    model = _Declaring(_kv(), settings={"kv": {"max_num_pages": 256}})
    specs = serving_resources(model, {"resources": {"kv": {"max_num_pages": 1024}}})
    assert specs[0].config.max_num_pages == 1024, "a deployment overrides the hook"
    assert serving_resources(model, {})[0].config.max_num_pages == 256, "the hook is the default"


@pytest.mark.parametrize("arrays, key, settings, allowed", [
    (lambda: [_kv()], "kv", {"max_slots": 8}, "['max_num_pages']"),
    (lambda: [_kv()], "kv", {"page_size": 64}, "['max_num_pages']"),
    (lambda: [_kv(), _gdn()], "gdn", {"max_num_pages": 8}, "['max_slots']"),
    (lambda: [_kv(), _gdn()], "gdn", {"prefix_slots": 8}, "['max_slots']"),
    (lambda: [_FLAG], "flag", {"max_slots": 8}, "[]"),
    (lambda: [_FLAG], "flag", {"max_num_pages": 8}, "[]"),
    (lambda: [_within()], "vision", {"max_slots": 8}, "[]"),
    (lambda: [_within()], "vision", {"max_num_pages": 8}, "[]"),
])
def test_a_size_its_built_spec_doesnt_take_is_refused_naming_the_array_the_key_and_what_it_may_set(
    arrays, key, settings, allowed,
):
    with pytest.raises(ValueError) as refused:
        build_specs(arrays(), kernel_settings={key: settings})
    message = str(refused.value)
    assert f"Array {key!r}: kernel_settings[{key!r}] sets" in message and allowed in message, message


@pytest.mark.parametrize("value", [0, -1, 1.5, "8", True])
@pytest.mark.parametrize("arrays, key, field", [
    (lambda: [_kv()], "kv", "max_num_pages"),
    (lambda: [_kv(), _gdn()], "gdn", "max_slots"),
])
def test_a_size_that_isnt_a_positive_int_is_refused_naming_the_array_and_the_key(arrays, key, field, value):
    with pytest.raises(ValueError) as refused:
        build_specs(arrays(), kernel_settings={key: {field: value}})
    message = str(refused.value)
    assert f"Array {key!r}: kernel_settings[{key!r}]" in message and field in message, message


@pytest.mark.parametrize("arrays, key", [(lambda: [_FLAG], "flag"), (lambda: [_within()], "vision")])
def test_empty_settings_on_an_array_with_no_pool_set_nothing(arrays, key):
    build_specs(arrays(), kernel_settings={key: {}})


def test_a_key_naming_nothing_lists_the_arrays_among_the_names():
    with pytest.raises(ValueError) as refused:
        build_specs([_kv()], kernel_settings={"nope": {"max_num_pages": 8}})
    assert "names no read" in str(refused.value) and "['attn', 'kv']" in str(refused.value), refused.value


class _Declaring(_ToyModel):
    """Declares ``arrays`` beside the hand-written ``specs``."""

    def __init__(self, *arrays, specs=(), settings=None):
        super().__init__(_prefill_decode)
        self.arrays, self.specs, self.settings = list(arrays), list(specs), settings or {}

    def declare_state(self):
        return self.arrays

    def kernel_settings(self):
        return self.settings

    def get_node_resources(self):
        return list(self.specs)


_SAMPLER = SamplerSpec(resource_key="sampler", nodes={"llm"}, vocab_size=16)


def test_a_model_declares_nothing_and_sets_no_kernels_by_default():
    assert Model.declare_state(None) == [] and Model.kernel_settings(None) == {}


def test_node_resources_are_the_built_specs_and_then_the_hand_written_ones():
    specs = node_resources(_Declaring(_kv(), specs=[_SAMPLER]))
    assert [spec.resource_key for spec in specs] == ["kv", "attn", "sampler"]


def test_node_resources_pass_the_model_kernel_settings():
    specs = node_resources(_Declaring(_kv(), settings={"attn": {"decode_fmha": True}}))
    assert specs[1].config.decode_fmha is True


def test_a_declared_name_clashing_with_a_hand_written_key_is_caught_on_the_joined_specs():
    clash = SamplerSpec(resource_key="kv", nodes={"llm"}, vocab_size=16)
    assert spec_dependency_problems(node_resources(_Declaring(_kv(), specs=[clash]))) == [
        "two resources declared under the key 'kv'",
    ]


def test_enabled_resources_builds_from_the_declarations():
    specs = enabled_resources(_Declaring(_kv(), specs=[_SAMPLER]), {}, StateSetup())
    assert {spec.resource_key for spec in specs} == {"kv", "attn", "sampler"}


def test_kernel_settings_with_no_declarations_are_refused():
    with pytest.raises(ValueError, match="names no read"):
        node_resources(_Declaring(settings={"attn": {"decode_fmha": True}}))


def test_enabled_resources_raises_the_first_problem():
    model = _Declaring(_kv(nodes={"dit"}), _kv(nodes={"sit"}))
    with pytest.raises(ValueError, match="the name 'kv' is declared twice"):
        enabled_resources(model, {}, StateSetup())


def _validation_problems(model, tmp_path, **config) -> list[str]:
    """What ``validate_deployment`` reports for a config that has one other problem of its own."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"node_groups": [{"node_names": ["llm"], "ranks": [0]}], **config}))
    with pytest.raises(DeploymentError) as refusal:
        _validate(model, str(path))
    problems = refusal.value.problems
    assert problems[0] == "the config must define `max_seq_len`", f"an earlier problem was dropped: {problems}"
    return problems[1:]


def test_validate_deployment_reports_every_declaration_problem(tmp_path):
    model = _Declaring(_kv(nodes={"dit"}), _kv(nodes={"sit"}))
    problems = _validation_problems(model, tmp_path)
    assert len(problems) == 5 and all(problem.startswith("Array 'kv'") for problem in problems[:4]), problems


def test_validate_deployment_says_that_a_declaration_problem_skips_the_resources_checks(tmp_path):
    problems = _validation_problems(_Declaring(_kv(nodes={"dit"})), tmp_path, resources={"nope": {}})
    assert problems[-1] == "the `resources:` checks, which need the declarations, were skipped", problems
    assert not any("nope" in problem for problem in problems), problems


def test_validate_deployment_refuses_log_probs_on_several_draws(tmp_path):
    class _TwoDraws(_Declaring):
        def declare_state(self):
            return [Sampler("sampler", nodes={"llm"}, vocab_size=8, draws=2)]

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({
        "max_seq_len": 8, "node_groups": [{"node_names": ["llm"], "ranks": [0]}],
        "resources": {"sampler": {"enable_logprobs": True}},
    }))
    with pytest.raises(DeploymentError) as refusal:
        _validate(_TwoDraws(), str(path))
    assert any("Sampler 'sampler'" in problem and "enable_logprobs" in problem for problem in refusal.value.problems), (
        refusal.value.problems
    )


def test_validate_deployment_skips_building_specs_after_a_declaration_problem(tmp_path):
    class _NoSpecs(_Declaring):
        def get_node_resources(self):
            raise AssertionError("specs were built from a declaration with a problem")

    problems = _validation_problems(_NoSpecs(_kv(nodes={"dit"})), tmp_path)
    assert len(problems) == 2 and problems[0].startswith("Array 'kv'.nodes"), problems


def test_validate_deployment_reports_a_refusal_at_construction_with_the_rest(tmp_path):
    class _BadWindow(_Declaring):
        def declare_state(self):
            return [_kv(reads={"attn": _read(mask=Window(0))})]

    problems = _validation_problems(_BadWindow(), tmp_path)
    assert len(problems) == 1 and "Array 'kv'.reads['attn'].mask.size" in problems[0], problems


def test_validate_deployment_reports_no_layout_yet_with_the_rest(tmp_path):
    problems = _validation_problems(_Declaring(_kv(reads={"attn": _read(op=None)})), tmp_path)
    assert len(problems) == 1 and "Array 'kv': no layout yet" in problems[0], problems


def _tokens(packed: bool) -> Array:
    return Array("tokens", nodes={"llm"}, layers={0}, entry=KVEntry(heads=2, head_dim=4), max_len=0,
                 reads={"tokens_attn": Read(mask=Block(), op=Attend(q_heads=2), packed=packed)})


_ON_SDPA = {"tokens_attn": {"backend": "sdpa"}}


def test_validate_deployment_refuses_sdpa_on_a_read_of_packed_queries(tmp_path):
    problems = _validation_problems(_Declaring(_tokens(True)), tmp_path, resources=_ON_SDPA)
    (problem,) = problems
    assert "reads['tokens_attn'] runs SDPA, which can't tell one segment" in problem, "a worker would bind it and fail"


def test_validate_deployment_takes_sdpa_on_a_read_of_rows(tmp_path):
    problems = _validation_problems(_Declaring(_tokens(False)), tmp_path, resources=_ON_SDPA)
    assert problems == [], "the block chose a binding, which builds no spec for it to name"


def test_validate_deployment_reports_a_refused_sdpa_block_with_the_rest_of_the_block_checks(tmp_path):
    problems = _validation_problems(_Declaring(_tokens(True)), tmp_path, resources={**_ON_SDPA, "nope": {}})
    assert len(problems) == 2 and "runs SDPA, which can't tell" in problems[0] and "['nope']" in problems[1], (
        "a refused binding would hide the deployment's other problems"
    )


def test_a_worker_builds_no_spec_for_a_resource_on_sdpa():
    assert serving_resources(_Declaring(_tokens(False)), {"resources": _ON_SDPA}) == [], (
        "the overrides would name a resource SDPA doesn't build"
    )


def test_validate_deployment_refuses_attention_backend_on_a_node_group_naming_the_new_key(tmp_path):
    groups = [{"node_names": ["llm"], "ranks": [0], "attention_backend": "sdpa"}]
    problems = _validation_problems(_Declaring(_tokens(False)), tmp_path, node_groups=groups)
    assert problems == [
        "node_groups[0].attention_backend: the backend is a resource's setting, as "
        "`resources: {<the read's resource key>: {backend: sdpa}}`",
    ], "a config that asks for SDPA there would run the ragged kernel"


def test_a_config_half_moved_to_the_new_key_gets_one_problem(tmp_path):
    groups = [{"node_names": ["llm"], "ranks": [0], "attention_backend": "sdpa"}]
    problems = _validation_problems(_Declaring(_tokens(False)), tmp_path, node_groups=groups, resources=_ON_SDPA)
    assert len(problems) == 1 and problems[0].startswith("node_groups[0].attention_backend"), (
        "the backend block it already moved would be refused too"
    )


def test_validate_deployment_labels_a_declaration_it_cannot_read(tmp_path):
    class _Unreadable(_Declaring):
        def declare_state(self):
            return [_kv(nodes=5)]

    problems = _validation_problems(_Unreadable(), tmp_path)
    assert len(problems) == 1 and problems[0].startswith("declaring the node resources could not read"), problems


def test_validate_deployment_reports_a_state_with_no_declared_dtype_with_the_rest(tmp_path):
    problems = _validation_problems(_Declaring(_kv(), _gdn(entry=_gdn_entry(state_dtype=None))), tmp_path)
    assert len(problems) == 1 and "Array 'gdn'.entry.tensors['state'].dtype" in problems[0], problems
    assert problems[0].startswith("declaring the node resources failed"), problems


def test_validate_deployment_refuses_kernel_settings_with_the_rest(tmp_path):
    problems = _validation_problems(_Declaring(_kv(), settings={"attn": {"backend": "sdpa"}}), tmp_path)
    assert len(problems) == 1 and "kernel_settings['attn']" in problems[0], problems


def _plain(**overrides) -> Array:
    return Value("state", **{
        "nodes": {"llm"}, "layers": {0, 2}, "entry": Entry({
            "h": TensorType((2, 8), dtype=torch.float32),
            "count": TensorType((), dtype=torch.int64),
            "conv": TensorType((4, 3), dtype=None),
            "sharded": TensorType((4, 3), dtype=torch.float32, shard=0),
        }), **overrides,
    })


def test_a_plain_value_builds_one_block_per_field_and_no_kernel():
    specs = _by_key(_plain())
    assert set(specs) == {"state"} and isinstance(specs["state"], RecurrentStateSpec), "it needs no KV or LinearAttn"
    config = specs["state"].config
    assert config.num_layers == 2 and config.kv_cache is None
    assert list(config.blocks) == ["h", "count", "conv", "sharded"], "a pool's blocks keep the entry's order"
    assert config.blocks["count"].shape == () and config.blocks["count"].dtype is torch.int64, "0-d reaches the pool"
    assert config.blocks["conv"].dtype is None, "None is the compute dtype, resolved by the pool"
    assert config.blocks["sharded"].shard_dims == (0,) and config.blocks["h"].shard_dims == ()


def test_a_values_labels_reach_its_spec_on_both_paths():
    plain = replace(_plain(), labels=frozenset({"main", "cfg"}))
    gdn = replace(_gdn(), labels=frozenset({"main", "cfg"}))
    assert _by_key(plain)["state"].config.labels == ("cfg", "main")
    assert _by_key(_kv(), gdn)["gdn"].config.labels == ("cfg", "main")
    assert _by_key(_kv(), _gdn())["gdn"].config.labels == ("main",), "a pool's default, as main's"


def test_a_field_named_state_on_a_plain_value_is_an_ordinary_field():
    entry = Entry({"state": TensorType((2, 8, 4), dtype=torch.float16)})
    block = _by_key(_plain(entry=entry))["state"].config.blocks["state"]
    assert block.shape == (2, 8, 4) and block.dtype is torch.float16, "only a GatedDeltaNet's state is stored K-last"


def test_kernel_settings_sizes_a_plain_value_and_yaml_overrides_it():
    (hooked,) = build_specs([_plain()], kernel_settings={"state": {"max_slots": 5}})
    assert hooked.config.max_slots == 5, "the hook's size reaches a plain Value's pool"
    apply_yaml_overrides([hooked], {"resources": {"state": {"max_slots": 9}}})
    assert hooked.config.max_slots == 9, "YAML wins over the hook"


@pytest.mark.parametrize("build, cause", [
    (lambda: replace(_plain(), reads={"r": Read(mask=Causal())}), r"reads\['r'\]\.op: today's resources run no read"),
    (lambda: replace(_plain(), seq_sharded=True), "seq_sharded"),
])
def test_a_plain_value_refuses_what_a_pool_cant_hold(build, cause):
    with pytest.raises(ValueError, match=cause):
        build_specs([build()])


def test_kernel_settings_picks_a_kept_reads_decode_kernel_and_yaml_overrides_it():
    specs = build_specs([_kv()], kernel_settings={"attn": {"decode_tensor_cores": False}})
    assert specs[1].config.decode_tensor_cores is False, "Qwen3-TTS's two query heads per KV head want the CUDA kernel"
    apply_yaml_overrides(specs, {"resources": {"attn": {"decode_tensor_cores": True}}})
    assert specs[1].config.decode_tensor_cores is True, "YAML wins over the hook"


@pytest.mark.parametrize("field", ["decode_tensor_cores", "decode_fmha"])
def test_a_kept_reads_decode_setting_takes_the_yaml_paths_check(field):
    """The string "false" would otherwise be stored as set, and read as true."""
    with pytest.raises(ValueError, match=rf"Array 'kv': kernel_settings\['attn'\]: {field} must be true or false"):
        build_specs([_kv()], kernel_settings={"attn": {field: "false"}})


def test_decode_tensor_cores_on_a_within_step_read_is_refused():
    array = Array("a", nodes={"llm"}, layers={0}, entry=KVEntry(heads=4, head_dim=16), max_len=0,
                  reads={"r": _read(mask=Block(), op=Attend(q_heads=4))})
    with pytest.raises(ValueError, match=r"Array 'a': kernel_settings\['r'\] sets \['decode_tensor_cores'\]"):
        build_specs([array], kernel_settings={"r": {"decode_tensor_cores": False}})


_EXTRA = {"extra": TensorType((2,), dtype=None)}


@pytest.mark.parametrize("value, refusal", [
    (lambda: _gdn(entry=Entry({**_gdn_entry().tensors, **_EXTRA})),
     "Array 'gdn': no layout yet: entry must be exactly state and conv, the blocks DeltaNetGeometry reads"),
    (lambda: _mamba(entry=_ssm_entry(**_EXTRA)),
     "Array 'mamba': no layout yet: entry must be exactly ssm and conv, the blocks Mamba2Geometry reads"),
], ids=["gdn", "mamba2"])
def test_a_linear_value_holds_its_op_s_field_and_a_conv(value, refusal):
    with pytest.raises(ValueError, match=re.escape(refusal)):
        build_specs([_kv(), value()])


@pytest.mark.parametrize("value, field", [
    (lambda: _gdn(entry=_gdn_entry(state_dtype=None)), "Array 'gdn'.entry.tensors['state']"),
    (lambda: _mamba(entry=_ssm_entry(ssm_dtype=None)), "Array 'mamba'.entry.tensors['ssm']"),
], ids=["gdn", "mamba2"])
def test_a_linear_value_declares_its_state_dtype(value, field):
    with pytest.raises(ValueError, match=re.escape(f"{field}.dtype must be declared")):
        build_specs([_kv(), value()])


def test_a_mamba2_conv_matches_its_heads_and_groups():
    """A conv one channel short of ``heads * head_dim + 2 * groups * state_size`` matches no group count."""
    with pytest.raises(ValueError, match=re.escape("Array 'mamba': no layout yet: conv block (95, 3) does not match")):
        build_specs([_kv(), _mamba(entry=_ssm_entry(conv_channels=95))])


def test_a_mamba2_update_takes_no_kernel_settings():
    with pytest.raises(ValueError, match=re.escape(
        "kernel_settings['ssm'] names no read or update of the declarations that takes kernel settings",
    )):
        build_specs([_kv(), _mamba()], kernel_settings={"ssm": {"splits": (_SPLIT,)}})


def test_only_a_snapshot_writing_value_links_its_pool_to_the_kv():
    specs = _by_key(_kv(), _gdn(), _mamba())
    assert specs["gdn"].config.kv_cache == "kv", "GDN's kernels write the snapshots a prefix hit restores"
    assert specs["mamba"].config.kv_cache is None, "no Mamba-2 kernel writes a snapshot, so the pool keeps none"
    assert specs["ssm"].config.variant is LinearAttnVariant.MAMBA2
    alone = _by_key(_mamba())
    assert alone["mamba"].config.kv_cache is None, "a Mamba-2 Value needs no paged array on its nodes"


def _context(read: Read) -> Array:
    return Array("ctx", nodes={"llm"}, layers={2, 5}, entry=KVEntry(heads=4, head_dim=16), max_len=64,
                 reads={"x": read})


@pytest.mark.parametrize("arrays, key", [
    (lambda: [_kv(reads={"r": _read(packed=False)})], "r"),
    (lambda: [_kv(reads={"r": _read(packed=False, walks={"decode"}), "p": _read(walks={"prefill"})})], "r"),
    (lambda: [_kv(), _context(_read(mask=Bidirectional(), queries="kv", packed=False))], "x"),
], ids=["paged", "paged by walk", "paged cross"])
def test_a_paged_read_takes_packed_queries(arrays, key):
    """Padded rows reach no paged kernel, which would read the padding as tokens."""
    with pytest.raises(ValueError, match=re.escape(
        f"no layout yet: reads[{key!r}].packed is False, and the paged kernels take packed queries",
    )):
        build_specs(arrays())


# Unbounded arrays


def _window(**overrides) -> Read:
    return Read(**{"mask": Window(5), "op": Attend(q_heads=4), "packed": False, **overrides})


def _unbounded(**overrides) -> Array:
    return Array("enc", **{
        "nodes": {"llm"}, "layers": {0, 1}, "entry": KVEntry(heads=4, head_dim=16), "max_len": None,
        "reads": {"enc_attn": _window()}, **overrides,
    })


def _no_layout(refusal: str, *arrays, **kwargs) -> None:
    with pytest.raises(ValueError, match=re.escape(f"Array 'enc': no layout yet: {refusal}")):
        build_specs(arrays, **kwargs)


def test_an_unbounded_array_is_one_bounded_kv_holding_its_window():
    (spec,) = build_specs([_unbounded()])
    config = spec.config
    assert isinstance(spec, BoundedKVSpec) and spec.resource_key == "enc" and spec.nodes == {"llm"}
    assert (config.num_layers, config.num_heads, config.head_dim, config.dtype) == (2, 4, 16, None)
    assert (config.rows_per_request, config.max_source_len) == (1, 0), "one row per request, with no source"
    assert config.retention == SinkWindow(0, 4) and config.query_window == 4, "Window(5) reaches distances 0 to 4"


@pytest.mark.parametrize("reads, masks", [
    ({"enc_attn": _window(mask=Causal())}, ["Causal"]),
    ({"enc_attn": _window(layers={0}), "more": _window(layers={1})}, ["Window", "Window"]),
    ({}, []),
], ids=["causal", "two reads", "no read"])
def test_an_unbounded_array_reads_through_one_window(reads, masks):
    _no_layout(f"an unbounded array reads through one Window read, whose window the bounded KV keeps, and its reads "
               f"have masks {masks}", _unbounded(reads=reads))


def test_an_unbounded_array_has_no_reach_read_yet():
    _no_layout("reads['enc_attn'].op must be Attend, the reads today's resources run on this array",
               _unbounded(reads={"enc_attn": Read(mask=Window(5), packed=False)}))


@pytest.mark.parametrize("read, refusal", [
    (_window(packed=True), "reads['enc_attn'] takes packed queries, and the bounded kernels take rows padded"),
    (_window(walks={"decode"}), "reads['enc_attn'].walks must be None"),
    (_window(layers={0}), "reads['enc_attn'].layers must be the array's"),
], ids=["packed", "walks", "some layers"])
def test_a_bounded_read_takes_padded_rows_and_no_walks(read, refusal):
    _no_layout(refusal, _unbounded(reads={"enc_attn": read}))


def test_a_bounded_read_has_as_many_query_heads_as_its_entry():
    _no_layout("reads['enc_attn'].op.q_heads is 8, and the bounded kernels run one query head per head of the "
               "entry's 4", _unbounded(reads={"enc_attn": _window(op=Attend(q_heads=8))}))


def test_a_bounded_read_takes_a_head_dim_the_kernels_tile():
    _no_layout("entry head_dim 72, and the bounded kernels tile a power of two of at least 16",
               _unbounded(entry=KVEntry(heads=4, head_dim=72)))


@pytest.mark.parametrize("field, refusal", [
    ({"labels": {"main", "cfg"}}, "an unbounded array takes one label, the bounded KV's one row per request, and it "
                                  "has ['cfg', 'main']"),
    ({"seq_sharded": True}, "seq_sharded takes max_len 0"),
], ids=["two labels", "seq_sharded"])
def test_a_bounded_array_has_one_label_and_no_seq_sharding(field, refusal):
    _no_layout(refusal, _unbounded(**field))


def test_a_bounded_array_takes_no_rope():
    assert len(build_specs([_unbounded(positions=Positions())])) == 1, "a count of entries builds no position spec"
    _no_layout("positions.rope, since the bounded KV keeps a count and no ids to rotate at",
               _unbounded(positions=Positions(rope=Rope())))


def test_the_bounded_read_takes_no_kernel_settings():
    with pytest.raises(ValueError, match=re.escape(
        "kernel_settings['enc_attn'] names no read or update of the declarations that takes kernel settings, and no "
        "array, of ['enc']",
    )):
        build_specs([_unbounded()], kernel_settings={"enc_attn": {"max_slots": 4}})


def test_kernel_settings_size_the_bounded_array():
    (spec,) = build_specs([_unbounded()], kernel_settings={"enc": {"max_slots": 7}})
    assert spec.config.max_slots == 7, "max_slots under the array's name sizes its bounded KV"
    with pytest.raises(ValueError, match=re.escape("kernel_settings['enc'] sets ['max_num_pages']; it may set only "
                                                   "['max_slots']")):
        build_specs([_unbounded()], kernel_settings={"enc": {"max_num_pages": 7}})


def test_a_bounded_read_runs_on_no_sdpa():
    """The read runs on the array's own resource, so a deployment names that."""
    _, problems = state_setup({"resources": {"enc": {"backend": "sdpa"}}}, [_unbounded()])
    assert problems == ["resources.enc: backend sdpa runs only an Attend read within a step, and Array "
                        "'enc'.reads['enc_attn'] isn't one"]


def test_a_cross_read_of_an_unbounded_array_s_queries_is_refused():
    context = Array("ctx", nodes={"llm"}, layers={0, 1}, entry=KVEntry(heads=4, head_dim=16), max_len=64,
                    reads={"x": _read(mask=Bidirectional(), op=Attend(q_heads=4), queries="enc")})
    with pytest.raises(ValueError, match=re.escape(
        "Array 'ctx': no layout yet: reads['x'] keeps entries and reads the queries of 'enc', which isn't an "
        "append-only array that keeps entries in pages",
    )):
        build_specs([_unbounded(), context])


class _StubBounded:
    """Counts as the bounded KV does, without a cache."""

    def written(self, rid):
        return {1: 9}.get(rid, 0)


def test_node_handles_build_a_bounded_handle():
    (h,) = node_handles(_Declaring(_unbounded(positions=Positions())), "llm", {"enc": _StubBounded()}).values()
    assert isinstance(h, BoundedHandle) and list(h.reads) == ["enc_attn"], "the handle takes its read by its key"
    assert [bound.key for bound in h._bindings] == ["enc_attn", "rope"], "the read, then the positions entry"
    assert position(h, 1, "main") == 9 and h.written(1) == 9, "position() and the step builder read one count"
