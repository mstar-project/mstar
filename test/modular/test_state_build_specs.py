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
    Read,
    Sampler,
    TensorType,
    Value,
    Window,
)
from mstar.engine.state.adapters import build_specs, node_resources
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
    )], "reads must be empty"),
    "a value of one tensor": (
        "gdn",
        lambda: [_kv(), _gdn(entry=Entry({"state": TensorType((8, 128, 64), dtype=None)}),
                             updates={"delta": GatedDeltaNet()})],
        "entry",
    ),
    "a value with a host field": ("gdn", lambda: [_kv(), _gdn(entry=_gdn_entry(host={"done": bool}))], "entry"),
    "a value with no fields": ("gdn", lambda: [_gdn(entry=Entry({}), updates={})], "entry"),
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
    specs = enabled_resources(_Declaring(_kv(), specs=[_SAMPLER]), {})
    assert {spec.resource_key for spec in specs} == {"kv", "attn", "sampler"}


def test_kernel_settings_with_no_declarations_are_refused():
    with pytest.raises(ValueError, match="names no read"):
        node_resources(_Declaring(settings={"attn": {"decode_fmha": True}}))


def test_enabled_resources_raises_the_first_problem():
    model = _Declaring(_kv(nodes={"dit"}), _kv(nodes={"sit"}))
    with pytest.raises(ValueError, match="the name 'kv' is declared twice"):
        enabled_resources(model, {})


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
    (lambda: replace(_plain(), reads={"r": Read(mask=Causal())}), "reads must be empty"),
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
