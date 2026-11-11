"""A ported model's specs equal the ones it wrote by hand, field for field.

Nothing in the engine changes when a model moves to declarations, because it sees the same specs. The hand-written
lists below are the reference, copied from the models as they stood before the port, so the adapter can't pass by
agreeing with itself. Each is compared per key with dataclass equality, with a block's dtype as the pool stores it,
before and after a deployment's overrides.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, ".")

import pytest
import torch
import yaml

from mstar.engine.node_dtype import resolve_model_dtype
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnSpec, LinearAttnVariant
from mstar.engine.resources.lora.config import LORA_RESOURCE_KEY
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.engine.resources.spec import CombinedSplit, apply_yaml_overrides
from mstar.engine.state.adapters import node_resources
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.base import enabled_resources, parse_autocast_dtype, serving_resources
from mstar.model.command_a_plus import config as command_a_plus_config
from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel
from mstar.model.components.request_kwargs import sampler_capabilities
from mstar.model.keys import MIXED
from mstar.model.qwen3_5.config import (
    ATTN,
    CONV,
    FULL_ATTENTION,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTENTION,
    LINEAR_ATTN,
    ROPE,
    SAMPLER,
    VISION_ATTN,
    Qwen3_5Config,
)
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.registry import MODEL_REGISTRY

_MIXED_SPLIT = CombinedSplit(MIXED, prefill_walks=("prefill_text",))

# The published geometries (Qwen/Qwen3.5-0.8B, -9B and -27B `config.json`), so these tests need no download.
QWEN3_5 = {
    "0.8B": dict(
        num_hidden_layers=24, layer_types=([LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]) * 6,
        num_attention_heads=8, num_key_value_heads=2, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=16, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, max_position_embeddings=262144, mamba_ssm_dtype="float32",
    ),
    "9B": dict(
        num_hidden_layers=32, layer_types=([LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]) * 8,
        num_attention_heads=16, num_key_value_heads=4, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, max_position_embeddings=262144, mamba_ssm_dtype="float32",
    ),
    "27B": dict(
        num_hidden_layers=64, layer_types=([LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]) * 16,
        num_attention_heads=24, num_key_value_heads=4, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, max_position_embeddings=262144, mamba_ssm_dtype="float32",
    ),
}
# 2B has 0.8B's geometry, 4B has 9B's, and Qwen3.8-27B has Qwen3.5-27B's.
_QWEN3_5_GEOMETRY = {
    "qwen3_5_0.8b": "0.8B", "qwen3_5_2b": "0.8B", "qwen3_5_4b": "9B", "qwen3_5_9b": "9B", "qwen3_5_27b": "27B",
    "qwen3_8_27b": "27B",
}


def _qwen3_5(**geometry) -> Qwen3_5DenseModel:
    model = Qwen3_5DenseModel.__new__(Qwen3_5DenseModel)
    model.config = Qwen3_5Config(
        hidden_size=1024, intermediate_size=2048, vocab_size=1024, rms_norm_eps=1e-6, **geometry,
    )
    model.vision_config = None
    return model


def _qwen3_5_state_dtype(config) -> torch.dtype:
    dtype = getattr(torch, config.mamba_ssm_dtype, None)
    head_dims_ok = config.linear_key_head_dim == 128 and config.linear_value_head_dim == 128
    return torch.bfloat16 if dtype is torch.bfloat16 and head_dims_ok else torch.float32


# Less the opt-in LoRA spec, which `enabled_resources` builds only when a deployment names it. It holds callables, so
# a second build wouldn't equal it.
def _qwen3_5_reference(config) -> list:
    return [
        KVSpec(
            resource_key=KV_CACHE, nodes={"LLM"},
            config=PagedKVConfig(
                num_layers=len(config.full_layer_indices),
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                max_seq_len=config.max_position_embeddings,
                num_qo_heads=config.num_attention_heads,
            ),
        ),
        AttentionSpec(
            resource_key=ATTN, nodes={"LLM"},
            config=AttentionConfig(kv_cache=KV_CACHE, splits=(_MIXED_SPLIT,), decode_fmha=True),
        ),
        PositionSpec(resource_key=ROPE, nodes={"LLM"}, config=PositionConfig(kv_cache=KV_CACHE)),
        RecurrentStateSpec(
            resource_key=GDN_STATE, nodes={"LLM"},
            config=RecurrentStateConfig(
                num_layers=len(config.linear_layer_indices),
                blocks=DeltaNetGeometry(
                    num_k_heads=config.linear_num_key_heads,
                    num_v_heads=config.linear_num_value_heads,
                    head_k_dim=config.linear_key_head_dim,
                    head_v_dim=config.linear_value_head_dim,
                    conv_kernel_size=config.linear_conv_kernel_dim,
                ).to_blocks(state_dtype=_qwen3_5_state_dtype(config)),
                kv_cache=KV_CACHE,
            ),
        ),
        LinearAttnSpec(
            resource_key=LINEAR_ATTN, nodes={"LLM"},
            config=LinearAttnConfig(
                recurrent_state=GDN_STATE, variant=LinearAttnVariant.GDN, splits=(_MIXED_SPLIT,),
            ),
        ),
        SamplerSpec(
            resource_key=SAMPLER, nodes={"LLM"}, vocab_size=config.vocab_size, enable_repetition_penalty=True,
        ),
    ]


def _stored(spec, kv_dtype: torch.dtype | None):
    """``spec`` with each block's dtype as the pool stores it, which is how a hand-written spec states it."""
    if not isinstance(spec, RecurrentStateSpec):
        return spec
    blocks = {name: replace(block, dtype=block.resolved_dtype(kv_dtype)) for name, block in spec.config.blocks.items()}
    return replace(spec, config=replace(spec.config, blocks=blocks))


def _assert_equal_per_key(specs, reference, kv_dtype: torch.dtype | None = torch.bfloat16) -> None:
    """``kv_dtype`` is the deployment's autocast dtype, bf16 for a ported model with no YAML."""
    by_key = {spec.resource_key: spec for spec in specs}
    assert len(by_key) == len(specs), "two specs under one key"
    expected = {spec.resource_key: spec for spec in reference}
    assert by_key.keys() == expected.keys()
    for key, spec in expected.items():
        assert _stored(by_key[key], kv_dtype) == spec, f"{key} differs from the hand-written spec"


def _published(model_class: str) -> list[Path]:
    """Every config in ``configs/`` that serves ``model_class``."""
    paths = sorted((Path(__file__).parents[2] / "configs").glob("*.yaml"))
    models = {path: yaml.safe_load(path.read_text()).get("model") for path in paths}
    return [path for path, model in models.items() if MODEL_REGISTRY.get(model, (None, None))[1] == model_class]


def test_the_published_configs_are_found():
    assert _published("Qwen3_5DenseModel") and _published("CommandAPlusModel"), "an empty list would skip their tests"


@pytest.mark.parametrize("size", QWEN3_5)
def test_qwen3_5_specs_equal_the_hand_written_ones(size):
    model = _qwen3_5(**QWEN3_5[size])
    _assert_equal_per_key(enabled_resources(model, {}), _qwen3_5_reference(model.config))


# The `resources:` blocks of configs/qwen3_5_9b.yaml, with its commented-out state_dtype on.
QWEN3_5_OVERRIDES = {"resources": {GDN_STATE: {"max_slots": 129, "prefix_slots": 128, "state_dtype": "bfloat16"}}}


@pytest.mark.parametrize("size", QWEN3_5)
def test_qwen3_5_specs_equal_the_hand_written_ones_after_yaml_overrides(size):
    model = _qwen3_5(**QWEN3_5[size])
    reference = _qwen3_5_reference(model.config)
    apply_yaml_overrides(reference, QWEN3_5_OVERRIDES)
    specs = serving_resources(model, QWEN3_5_OVERRIDES)
    _assert_equal_per_key(specs, reference)
    state = next(spec for spec in specs if spec.resource_key == GDN_STATE).config.blocks["state"]
    assert state.dtype is torch.bfloat16, "the YAML's state_dtype should still reach the stored block"


@pytest.mark.parametrize("path", _published("Qwen3_5DenseModel"), ids=lambda path: path.stem)
def test_qwen3_5_specs_equal_the_hand_written_ones_on_every_published_config(path):
    deployment = yaml.safe_load(path.read_text())
    model = _qwen3_5(**QWEN3_5[_QWEN3_5_GEOMETRY[deployment["model"]]])
    reference = _qwen3_5_reference(model.config)
    apply_yaml_overrides(reference, deployment)
    kv_dtype = parse_autocast_dtype(resolve_model_dtype(model, deployment))
    _assert_equal_per_key(serving_resources(model, deployment), reference, kv_dtype)


@pytest.mark.parametrize("ssm_dtype, head_k_dim, head_v_dim", [
    ("bfloat16", 128, 128),
    ("bfloat16", 64, 128),
    ("bfloat16", 128, 64),
    ("float16", 128, 128),
    ("no_such_dtype", 128, 128),
])
def test_qwen3_5_stored_dtype_equals_the_hand_written_rule(ssm_dtype, head_k_dim, head_v_dim):
    model = _qwen3_5(**{
        **QWEN3_5["9B"], "mamba_ssm_dtype": ssm_dtype, "linear_key_head_dim": head_k_dim,
        "linear_value_head_dim": head_v_dim,
    })
    _assert_equal_per_key(enabled_resources(model, {}), _qwen3_5_reference(model.config))


def test_qwen3_5_conv_follows_the_engine_dtype_and_its_state_does_not():
    blocks = {spec.resource_key: spec for spec in node_resources(_qwen3_5(**QWEN3_5["9B"]))}[GDN_STATE].config.blocks
    assert blocks["conv"].resolved_dtype(torch.float16) is torch.float16, "the conv would run on bf16 in an fp16 model"
    assert blocks["state"].resolved_dtype(torch.float16) is torch.float32, "the state's dtype comes from the checkpoint"


# Qwen/Qwen3.5-9B cut to 4 layers, all full attention, so no layer reads the GDN pool, whose axes may then be 0.
_ALL_FULL = {**QWEN3_5["9B"], "num_hidden_layers": 4, "layer_types": [FULL_ATTENTION] * 4}
_ALL_FULL_SHAPES = pytest.mark.parametrize(
    "overrides", [{}, {"linear_conv_kernel_dim": 1}, {"linear_value_head_dim": 0}],
    ids=["kernel 4", "kernel 1", "value dim 0"],
)


@_ALL_FULL_SHAPES
def test_qwen3_5_all_full_specs_equal_the_hand_written_ones(overrides):
    model = _qwen3_5(**{**_ALL_FULL, **overrides})
    specs = enabled_resources(model, {})
    _assert_equal_per_key(specs, _qwen3_5_reference(model.config))
    assert {spec.resource_key: spec for spec in specs}[GDN_STATE].config.num_layers == 0, "not the all-full shape"


@_ALL_FULL_SHAPES
def test_qwen3_5_all_full_specs_equal_the_hand_written_ones_after_yaml_overrides(overrides):
    model = _qwen3_5(**{**_ALL_FULL, **overrides})
    reference = _qwen3_5_reference(model.config)
    apply_yaml_overrides(reference, QWEN3_5_OVERRIDES)
    _assert_equal_per_key(serving_resources(model, QWEN3_5_OVERRIDES), reference)


@pytest.mark.parametrize("overrides, where", [
    ({"layer_types": [LINEAR_ATTENTION] * 32}, "Array 'kv_cache'.layers"),
    ({"linear_conv_kernel_dim": 1}, "Array 'gdn_state'.entry.tensors['conv'].shape"),
    ({"num_attention_heads": 15}, "Array 'kv_cache'.reads['attn'].op"),
    ({"max_position_embeddings": 262144.0}, "Array 'kv_cache'.max_len"),
], ids=["all linear", "conv kernel of 1", "query heads off the KV heads", "float length"])
def test_qwen3_5_refuses_a_shape_that_cant_run(overrides, where):
    with pytest.raises(ValueError, match=re.escape(where)):
        _qwen3_5(**{**QWEN3_5["9B"], **overrides}).declare_state()


def test_qwen3_5_writes_only_positions_the_sampler_and_lora_by_hand():
    model = _qwen3_5(**QWEN3_5["9B"])
    assert {spec.resource_key for spec in model.get_node_resources()} == {ROPE, SAMPLER, LORA_RESOURCE_KEY}


def test_qwen3_5_keeps_its_vision_attention_by_hand(monkeypatch):
    vision = SamplerSpec(resource_key=VISION_ATTN, nodes={"vision_encoder"}, vocab_size=1)
    monkeypatch.setattr(Qwen3_5DenseModel, "_vision_resources", lambda self: [vision])
    assert vision in node_resources(_qwen3_5(**QWEN3_5["9B"]))


@pytest.mark.parametrize("size", QWEN3_5)
def test_qwen3_5_declarations_have_no_problems(size):
    assert state_declaration_problems(_qwen3_5(**QWEN3_5[size])) == []


def test_qwen3_5_declares_the_conv_update_beside_the_delta_net_with_no_spec_of_its_own():
    model = _qwen3_5(**QWEN3_5["9B"])
    gdn = next(array for array in model.declare_state() if array.name == GDN_STATE)
    assert set(gdn.updates) == {CONV, LINEAR_ATTN}
    assert CONV not in {spec.resource_key for spec in node_resources(model)}


_FIXTURE = Path("mstar/model/command_a_plus/tests/fixtures/tiny_config.json")
_LOCAL_ATTN, _GLOBAL_ATTN = command_a_plus_config.LOCAL_ATTN, command_a_plus_config.GLOBAL_ATTN
# The published geometry (CohereLabs/command-a-plus-05-2026-bf16 `config.json`), so these tests need no download.
_COMMAND_A_PLUS = dict(
    num_hidden_layers=32, layer_types=(["sliding_attention"] * 3 + ["full_attention"]) * 8,
    num_attention_heads=128, num_key_value_heads=8, head_dim=128, sliding_window=4096,
    max_position_embeddings=200000, rope_theta=50000.0,
)


def _command_a_plus(tmp_path, text=None, **kwargs) -> CommandAPlusModel:
    raw = json.loads(_FIXTURE.read_text())
    raw["text_config"].update(text or {})
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return CommandAPlusModel(str(tmp_path), **kwargs)


def _command_a_plus_reference(model: CommandAPlusModel) -> list:
    config = model.config
    local = tuple(i for i, kind in enumerate(config.layer_types) if kind == "sliding_attention")
    return [
        KVSpec(resource_key=KV_CACHE, nodes={"LLM"}, config=PagedKVConfig(
            num_layers=config.num_hidden_layers, num_qo_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads, head_dim=config.head_dim,
            max_seq_len=model.max_seq_len,
        )),
        AttentionSpec(resource_key=_LOCAL_ATTN, nodes={"LLM"}, config=AttentionConfig(
            kv_cache=KV_CACHE, sliding_window=config.sliding_window, layers=local,
        )),
        AttentionSpec(resource_key=_GLOBAL_ATTN, nodes={"LLM"}, config=AttentionConfig(
            kv_cache=KV_CACHE,
        )),
        PositionSpec(resource_key=ROPE, nodes={"LLM"}, config=PositionConfig(
            kv_cache=KV_CACHE, rotary_dim=config.head_dim, interleave=True,
            rope_theta=config.rope_theta,
        )),
        SamplerSpec(resource_key=SAMPLER, nodes={"LLM"}, vocab_size=config.vocab_size,
                    **sampler_capabilities(model.sampling_defaults)),
    ]


@pytest.mark.parametrize("text", [None, _COMMAND_A_PLUS], ids=["tiny", "published"])
def test_command_a_plus_specs_equal_the_hand_written_ones(tmp_path, text):
    model = _command_a_plus(tmp_path, text)
    _assert_equal_per_key(node_resources(model), _command_a_plus_reference(model))


# configs/command_a_plus_tp4.yaml's block, and a page size the window derives its groups from.
@pytest.mark.parametrize("overrides", [
    {KV_CACHE: {"max_num_pages": 8192}},
    {KV_CACHE: {"page_size": 6, "max_num_pages": 16}},
])
def test_command_a_plus_specs_equal_the_hand_written_ones_after_yaml_overrides(tmp_path, overrides):
    model = _command_a_plus(tmp_path)
    reference = _command_a_plus_reference(model)
    apply_yaml_overrides(reference, {"resources": overrides})
    specs = serving_resources(model, {"resources": overrides})
    _assert_equal_per_key(specs, reference)
    kv = {spec.resource_key: spec for spec in specs}[KV_CACHE].config
    expected = {spec.resource_key: spec for spec in reference}[KV_CACHE].config
    assert kv.groups == expected.groups, "groups don't compare in the dataclass, and they hold the window's layers"
    assert len(kv.groups) == 2 and kv.prefix_cache is True, (
        "the local layers should keep only their window, and the cache still prefix caches"
    )


@pytest.mark.parametrize("path", _published("CommandAPlusModel"), ids=lambda path: path.stem)
def test_command_a_plus_specs_equal_the_hand_written_ones_on_every_published_config(tmp_path, path):
    deployment = yaml.safe_load(path.read_text())
    model = _command_a_plus(tmp_path, _COMMAND_A_PLUS, max_seq_len=deployment["model_kwargs"]["max_seq_len"])
    reference = _command_a_plus_reference(model)
    apply_yaml_overrides(reference, deployment)
    _assert_equal_per_key(serving_resources(model, deployment), reference)


def test_command_a_plus_refuses_query_heads_off_its_kv_heads(tmp_path):
    model = _command_a_plus(tmp_path, {"num_attention_heads": 4, "num_key_value_heads": 3})
    with pytest.raises(ValueError, match=re.escape("Array 'kv_cache'.reads['local_attn'].op")):
        model.declare_state()


def test_command_a_plus_keeps_the_window_on_its_local_read_only(tmp_path):
    specs = {spec.resource_key: spec for spec in node_resources(_command_a_plus(tmp_path, _COMMAND_A_PLUS))}
    local, global_ = specs[_LOCAL_ATTN].config, specs[_GLOBAL_ATTN].config
    assert local.sliding_window == 4096 and local.layers == tuple(i for i in range(32) if i % 4 != 3)
    assert global_.sliding_window is None and global_.layers is None


def test_command_a_plus_global_read_lists_the_layers_outside_the_window(tmp_path):
    [kv] = _command_a_plus(tmp_path, _COMMAND_A_PLUS).declare_state()
    assert kv.reads[_GLOBAL_ATTN].layers == {i for i in range(32) if i % 4 == 3}
    assert kv.reads[_LOCAL_ATTN].layers | kv.reads[_GLOBAL_ATTN].layers == kv.layers, "a layer with no read"


def test_command_a_plus_writes_only_positions_and_the_sampler_by_hand(tmp_path):
    assert {spec.resource_key for spec in _command_a_plus(tmp_path).get_node_resources()} == {ROPE, SAMPLER}


def test_command_a_plus_passes_a_shorter_max_seq_len_to_the_array(tmp_path):
    model = _command_a_plus(tmp_path, max_seq_len=32)
    assert model.declare_state()[0].max_len == 32
    _assert_equal_per_key(node_resources(model), _command_a_plus_reference(model))


@pytest.mark.parametrize("text", [None, _COMMAND_A_PLUS], ids=["tiny", "published"])
def test_command_a_plus_declarations_have_no_problems(tmp_path, text):
    assert state_declaration_problems(_command_a_plus(tmp_path, text)) == []
