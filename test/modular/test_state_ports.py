"""A ported model's specs, steps and layouts equal the ones it wrote by hand, field for field.

Nothing in the engine changes when a model moves to declarations, because it sees the same objects. The hand-written
specs and steps below are the reference, copied from the models as they stood before the port, so the adapter can't
pass by agreeing with itself. Each is compared per key with dataclass equality, with a block's dtype as the pool stores
it, before and after a deployment's overrides.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, ".")

import pytest
import torch
import yaml
from test_gdn_tensor_parallel import tiny_config

from mstar.distributed.communication import CommGroup
from mstar.engine.accelerator_graph_config import PiecewisePackedConfig
from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner
from mstar.engine.node_dtype import resolve_model_dtype
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig, RaggedAttentionSpec
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
)
from mstar.engine.resources.lora.config import LORA_RESOURCE_KEY
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.spec import CombinedSplit, apply_yaml_overrides
from mstar.engine.resources.step import ResourceStep, Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Layout, Span
from mstar.engine.state.adapters import (
    PAGED,
    VALUE,
    WITHIN_STEP,
    StateSetup,
    build_handle,
    kv_layouts,
    node_handles,
    node_resources,
    sampling_configs,
    storage_of,
)
from mstar.engine.state.analysis import state_declaration_problems
from mstar.engine.state.calls import causal_conv
from mstar.model.base import (
    LabelLayout,
    ProcessPromptOutput,
    enabled_resources,
    parse_autocast_dtype,
    serving_resources,
)
from mstar.model.command_a_plus import config as command_a_plus_config
from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel
from mstar.model.command_a_plus.submodules import CommandAPlusLLMSubmodule
from mstar.model.components.block_loop_encoder import VISION_BLOCK_LOOP
from mstar.model.components.request_kwargs import sampler_capabilities
from mstar.model.keys import MIXED, ROPE
from mstar.model.multimodal import vision_encoder
from mstar.model.multimodal_decoder import MultimodalDecoderModel
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM, Qwen3_5LanguageModel
from mstar.model.qwen3_5.config import (
    ATTN,
    CONV,
    FULL_ATTENTION,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTENTION,
    LINEAR_ATTN,
    SAMPLER,
    VISION_ATTN,
    VISION_TOKENS,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.qwen3_5.submodules import LLMSubmodule, VisionEncoderSubmodule
from mstar.model.registry import MODEL_REGISTRY
from mstar.model.submodule_base import ARNodeInputs, engine_step

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


class _StubResource:
    """Stands in for any built resource a handle reads at bind."""

    def layer_view(self, i):
        return torch.empty(0)

    def block(self, name, i):
        return torch.empty(0)

    def run(self, q, k=None, v=None, label=None):
        return q


def _bind(node, model, node_name="LLM"):
    resources = {spec.resource_key: _StubResource() for spec in node_resources(model)}
    node.bind_node_state(node_handles(model, node_name, resources))
    return node


def _assert_steps_equal(step: SubmoduleStep, reference: SubmoduleStep) -> None:
    """Per key, with the tracked tokens apart, because dataclass equality can't compare a dict of tensors."""
    assert step.state, "a ported walk should declare its state, not write today's step classes"
    assert step.steps.keys() == reference.steps.keys()
    assert step.segments == reference.segments and step.cg_key_info == reference.cg_key_info
    for key, expected in reference.steps.items():
        actual = step.steps[key]
        if isinstance(expected, SamplerStep):
            tokens, want = actual.prefill_tracked_tokens, expected.prefill_tracked_tokens
            assert tokens.keys() == want.keys()
            assert all(torch.equal(tokens[rid], want[rid]) for rid in want)
            actual, expected = replace(actual, prefill_tracked_tokens={}), replace(expected, prefill_tracked_tokens={})
        assert actual == expected, f"{key} differs from the hand-written step"


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
    _assert_equal_per_key(enabled_resources(model, {}, StateSetup()), _qwen3_5_reference(model.config))


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
    _assert_equal_per_key(enabled_resources(model, {}, StateSetup()), _qwen3_5_reference(model.config))


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
    specs = enabled_resources(model, {}, StateSetup())
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


def test_qwen3_5_writes_only_lora_by_hand():
    model = _qwen3_5(**QWEN3_5["9B"])
    assert {spec.resource_key for spec in model.get_node_resources()} == {LORA_RESOURCE_KEY}


# The tower of Qwen/Qwen3.5-9B's `config.json`
_VISION = dict(
    depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16, in_channels=3, patch_size=16,
    temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=576, out_hidden_size=4096,
)


def _qwen3_5_vision(monkeypatch) -> Qwen3_5DenseModel:
    model = _qwen3_5(**QWEN3_5["9B"])
    model.vision_config = Qwen3_5VisionConfig(**_VISION)
    model.media = (SimpleNamespace(encoder=vision_encoder()),)
    # the image processor's numbers, which `Qwen3_5DenseModel.image_processor` reads off the checkpoint
    processor = SimpleNamespace(size={"shortest_edge": 65536}, patch_size=16)
    monkeypatch.setattr(Qwen3_5DenseModel, "image_processor", processor)
    return model


def _qwen3_5_vision_reference(model) -> RaggedAttentionSpec:
    vision, processor = model.vision_config, model.image_processor
    min_patches = processor.size["shortest_edge"] // processor.patch_size ** 2
    return RaggedAttentionSpec(
        resource_key=VISION_ATTN, nodes={"vision_encoder"},
        config=RaggedAttentionConfig(
            num_qo_heads=vision.num_heads,
            num_kv_heads=vision.num_heads,
            head_dim=vision.head_dim,
            max_segments_per_step=VisionEncoderSubmodule.BLOCK_LOOP_PATCH_BUCKETS[-1] // min_patches,
        ),
    )


def test_qwen3_5_vision_spec_equals_the_hand_written_one(monkeypatch):
    model = _qwen3_5_vision(monkeypatch)
    reference = [*_qwen3_5_reference(model.config), _qwen3_5_vision_reference(model)]
    _assert_equal_per_key(enabled_resources(model, {}, StateSetup()), reference)
    assert state_declaration_problems(model) == []


def test_a_yaml_override_of_the_vision_ceiling_still_holds(monkeypatch):
    model = _qwen3_5_vision(monkeypatch)
    overrides = {"resources": {VISION_ATTN: {"max_segments_per_step": 3}}}
    reference = [*_qwen3_5_reference(model.config), _qwen3_5_vision_reference(model)]
    apply_yaml_overrides(reference, overrides)
    specs = serving_resources(model, overrides)
    _assert_equal_per_key(specs, reference)
    assert {spec.resource_key: spec for spec in specs}[VISION_ATTN].config.max_segments_per_step == 3


def _qwen3_5_reference_step(rids, inputs, prefill_tokens, advance) -> SubmoduleStep:
    return SubmoduleStep(
        segments=[
            Segment(request_id=rid, label="main", span=inp.input_seq_len)
            for rid, inp in zip(rids, inputs, strict=True)
        ],
        steps={
            KV_CACHE: KVStep(),
            ATTN: AttentionStep(causal=True),
            GDN_STATE: RecurrentStep(),
            LINEAR_ATTN: LinearAttnStep(),
            SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=prefill_tokens),
            ROPE: PositionStep(advance=advance),
        },
    )


def _qwen3_5_llm(model, encoders=()) -> LLMSubmodule:
    llm = LLMSubmodule(None, SimpleNamespace(stop_token_ids=frozenset()), encoders=encoders)
    return _bind(llm, model)


def test_qwen3_5_decode_step_equals_the_hand_written_one():
    llm = _qwen3_5_llm(_qwen3_5(**QWEN3_5["9B"]))
    rids, inputs = [1, 2], [ARNodeInputs(input_seq_len=1), ARNodeInputs(input_seq_len=1)]
    step = engine_step(llm, graph_walk="decode", request_ids=rids, inputs=inputs)
    _assert_steps_equal(step, _qwen3_5_reference_step(rids, inputs, {}, None))


def test_qwen3_5_prefill_text_step_equals_the_hand_written_one():
    llm = _qwen3_5_llm(_qwen3_5(**QWEN3_5["9B"]))
    rids = [1, 2]
    inputs = [ARNodeInputs(input_seq_len=n, input_ids=torch.arange(n)) for n in (5, 3)]
    step = engine_step(llm, graph_walk="prefill_text", request_ids=rids, inputs=inputs)
    tokens = {1: inputs[0].input_ids, 2: inputs[1].input_ids}
    _assert_steps_equal(step, _qwen3_5_reference_step(rids, inputs, tokens, None))


def test_qwen3_5_mixed_step_equals_the_hand_written_one():
    llm = _qwen3_5_llm(_qwen3_5(**QWEN3_5["9B"]))
    rids = [1, 2, 3]
    inputs = [
        ARNodeInputs(input_seq_len=1, graph_walk="decode"),
        ARNodeInputs(input_seq_len=3, input_ids=torch.arange(3), graph_walk="prefill_text"),
        ARNodeInputs(input_seq_len=2, input_ids=torch.arange(2), graph_walk="prefill_text"),
    ]
    step = engine_step(llm, graph_walk=MIXED, request_ids=rids, inputs=inputs)
    tokens = {2: inputs[1].input_ids, 3: inputs[2].input_ids}
    _assert_steps_equal(step, _qwen3_5_reference_step(rids, inputs, tokens, None))


@_ALL_FULL_SHAPES
def test_qwen3_5_all_full_steps_equal_the_hand_written_ones(overrides):
    llm = _qwen3_5_llm(_qwen3_5(**{**_ALL_FULL, **overrides}))
    rids, inputs = [1, 2], [ARNodeInputs(input_seq_len=1), ARNodeInputs(input_seq_len=1)]
    step = engine_step(llm, graph_walk="decode", request_ids=rids, inputs=inputs)
    _assert_steps_equal(step, _qwen3_5_reference_step(rids, inputs, {}, None))
    inputs = [ARNodeInputs(input_seq_len=n, input_ids=torch.arange(n)) for n in (5, 3)]
    step = engine_step(llm, graph_walk="prefill_text", request_ids=rids, inputs=inputs)
    tokens = {1: inputs[0].input_ids, 2: inputs[1].input_ids}
    _assert_steps_equal(step, _qwen3_5_reference_step(rids, inputs, tokens, None))


@_ALL_FULL_SHAPES
def test_a_step_opens_and_closes_an_all_full_gdn_handle_with_no_layer_to_update(overrides):
    model = _qwen3_5(**{**_ALL_FULL, **overrides})
    resources = {spec.resource_key: _StubResource() for spec in node_resources(model)}
    handle = node_handles(model, "LLM", resources)[GDN_STATE]
    step = SubmoduleStep(segments=[Segment(1, "main", 1)], state={GDN_STATE: ArrayStep()})
    step.set_ctx(StepContext((1,), "decode", 0, False))
    handle.begin_step(step)
    handle.check_step(step)
    with pytest.raises(ValueError, match=r"Array 'gdn_state': no field 'conv' on layer 0, of layers \[\]"):
        causal_conv(handle, 0, torch.zeros(1), weight=torch.zeros(1))
    handle.end_step(step)


@_ALL_FULL_SHAPES
def test_qwen3_5_all_full_layout_converts_to_the_hand_written_one(overrides):
    model = _qwen3_5(**{**_ALL_FULL, **overrides})
    span = Span("ids", "prefill_text", 7, 7, "text_inputs")
    layout = Layout(spans=(span,), growth=(("decode_loop", 1),))
    converted = kv_layouts(model.declare_state(), {KV_CACHE: {"main": layout}, GDN_STATE: {"main": Layout()}}, {})
    reference = {KV_CACHE: {"main": LabelLayout(spans=(span,), growth=(("decode_loop", 1),))}}
    assert converted == reference, "a Value with no layers should still drop from the layouts"


def test_qwen3_5_prefill_vision_step_equals_the_hand_written_one():
    llm = _qwen3_5_llm(_qwen3_5(**QWEN3_5["9B"]), encoders=(vision_encoder(),))
    rids = [1]
    tokens = torch.arange(4)
    inputs = [ARNodeInputs(input_seq_len=4, input_ids=tokens, resource_step_info=6)]
    step = engine_step(llm, graph_walk="prefill_vision", request_ids=rids, inputs=inputs)
    _assert_steps_equal(step, _qwen3_5_reference_step(rids, inputs, {1: tokens}, (6,)))


def _qwen3_5_encoder(monkeypatch) -> VisionEncoderSubmodule:
    model = _qwen3_5_vision(monkeypatch)
    encoder = VisionEncoderSubmodule(
        torch.nn.Linear(1, 1), model.vision_config, "vision_embeds", (torch.zeros(1), torch.ones(1)),
    )
    return _bind(encoder, model, "vision_encoder")


def test_qwen3_5_vision_step_equals_the_hand_written_one(monkeypatch):
    encoder = _qwen3_5_encoder(monkeypatch)
    rids = [1, 2]
    inputs = [SimpleNamespace(resource_step_info={"main": lengths}) for lengths in ((4, 4), (8,))]
    step = engine_step(encoder, graph_walk="prefill_vision", request_ids=rids, inputs=inputs)
    reference = SubmoduleStep(
        segments=[Segment(1, "main", 4), Segment(1, "main", 4), Segment(2, "main", 8)],
        steps={VISION_ATTN: AttentionStep(causal=False)},
    )
    _assert_steps_equal(step, reference)


def test_qwen3_5_leased_vision_step_declares_none_as_it_did(monkeypatch):
    encoder = _qwen3_5_encoder(monkeypatch)
    leases = {VISION_BLOCK_LOOP: object()}
    step = engine_step(encoder, graph_walk="prefill_vision", request_ids=[1], inputs=[], piecewise_leases=leases)
    assert step is None


def test_qwen3_5_leased_vision_step_beside_a_lora_pool_declares_its_spans_and_no_attention(monkeypatch):
    """The replayed block loop's region declares the attention, and the patch embed and merger outside it still need
    the spans planned for a resource whose step is the segments (a LoRA pool), as `main` declares them."""
    encoder = _qwen3_5_encoder(monkeypatch)
    encoder.bind_node_resources({"lora": SimpleNamespace(steps_from_segments=True)})
    inputs = [SimpleNamespace(resource_step_info={"main": (4, 4)})]
    leases = {VISION_BLOCK_LOOP: object()}
    step = engine_step(encoder, graph_walk="prefill_vision", request_ids=[1], inputs=inputs, piecewise_leases=leases)
    segments = [Segment(1, "main", 4), Segment(1, "main", 4)]
    assert not step.state, "a leased step's attention is its region's, which would plan it twice"
    assert step.segments == segments and step.steps == {"lora": ResourceStep(segments=tuple(segments))}


def test_qwen3_5_vision_region_steps_equal_the_hand_written_ones(monkeypatch):
    encoder = _qwen3_5_encoder(monkeypatch)
    config = encoder.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.bfloat16)[VISION_BLOCK_LOOP]
    runner = object.__new__(PiecewiseAcceleratorGraphRunner)
    runner._config, runner._segment_keys, runner._state_handles = config, (), encoder.state_handles
    step = runner._region_step([7], [3, 5])
    reference = PiecewisePackedConfig(
        capture_fn=config.capture_fn, make_static_inputs=config.make_static_inputs, attn_key=VISION_ATTN,
        lease_before_step=True, one_row_per_step=True, total_tokens=config.total_tokens, capture_batch_sizes=[1],
    ).region_step([7], [3, 5])
    _assert_steps_equal(step, reference)


def test_qwen3_5_vision_capture_check_reads_the_ceiling_in_effect(monkeypatch):
    encoder = _qwen3_5_encoder(monkeypatch)
    encoder.vision.reads[VISION_ATTN].max_segments_per_step = 2
    assert encoder._fits_replay(16, 2) and not encoder._fits_replay(16, 3), "a YAML override should hold"


def test_qwen3_5_process_prompt_declares_the_hand_written_layout(monkeypatch):
    model = _qwen3_5(**QWEN3_5["9B"])
    model.vision_config = SimpleNamespace(spatial_merge_size=2)
    model.tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: "hello", convert_tokens_to_ids=lambda token: None, unk_token_id=0,
    )
    monkeypatch.setattr("mstar.model.qwen3_5.qwen3_5_model.encode_ids", lambda tokenizer, text: [1, 2, 3])
    metadata = model.process_prompt("hello", ["text"], ["text"]).metadata
    span = Span("ids", "prefill_text", 3, 3, "text_inputs")
    assert metadata["state_layout"] == {
        KV_CACHE: {"main": Layout(spans=(span,), growth=(("decode_loop", 1),))},
        GDN_STATE: {"main": Layout()},
    }
    assert "keyed_spans" not in metadata, "a text prompt keys every span"


def test_qwen3_5_text_layout_converts_to_the_hand_written_one():
    model = _qwen3_5(**QWEN3_5["9B"])
    span = Span("ids", "prefill_text", 7, 7, "text_inputs")
    layout = Layout(spans=(span,), growth=(("decode_loop", 1),))
    converted = kv_layouts(model.declare_state(), {KV_CACHE: {"main": layout}, GDN_STATE: {"main": Layout()}}, {})
    reference = LabelLayout(spans=(span,), growth=(("decode_loop", 1),))
    assert converted == {KV_CACHE: {"main": reference}}, "a Value's layout should drop, and the KV's convert"
    assert converted[KV_CACHE]["main"].row() == reference.row()


def test_qwen3_5_vision_layout_converts_to_the_hand_written_one_with_no_keyed_spans():
    model = _qwen3_5(**QWEN3_5["9B"])
    spans = (Span("rows", "prefill_vision", 8, 6, "text_inputs"),)
    layout = Layout(spans=spans, growth=(("decode_loop", 1),))
    converted = kv_layouts(
        model.declare_state(), {KV_CACHE: {"main": layout}, GDN_STATE: {"main": Layout()}}, {},
    )[KV_CACHE]["main"]
    reference = LabelLayout(spans=spans, growth=(("decode_loop", 1),))
    assert converted == reference
    assert converted.keyed_count() == 0, "the data worker refuses a keyed span from any walk but prefill_text"
    assert converted.row() == reference.row(), "the one-walk check runs in row(), and the conductor reads what it gives"


def test_qwen3_5_hands_on_the_scaffolds_keyed_spans(monkeypatch):
    """The scaffold keys no span of a media prompt, whose leading text the data worker would refuse to key on its
    `prefill_vision` walk. The state layout holds no count, so it rides beside it."""
    spans = (
        Span("ids", "prefill_vision", 3, 3, "text_inputs"), Span("rows", "prefill_vision", 8, 6, "text_inputs"),
        Span("ids", "prefill_vision", 2, 2, "text_inputs"),
    )
    scaffold = LabelLayout(spans=spans, growth=(("decode_loop", 1),), keyed_spans=0)
    monkeypatch.setattr(
        MultimodalDecoderModel, "process_prompt",
        lambda *args, **kwargs: ProcessPromptOutput({}, {"kv_layout": {KV_CACHE: {"main": scaffold}}}),
    )
    model = _qwen3_5(**QWEN3_5["9B"])
    metadata = model.process_prompt("hello", ["text", "image"], ["text"]).metadata
    converted = kv_layouts(model.declare_state(), metadata["state_layout"], metadata.get("keyed_spans", {}))
    assert converted == {KV_CACHE: {"main": scaffold}}, "the leading text span would be keyed"


def test_qwen3_5_request_settings_equal_the_hand_written_config():
    model = _qwen3_5(**QWEN3_5["9B"])
    kwargs = {"temperature": 0.0, "top_p": 0.5, "top_k": 3, "repetition_penalty": 1.3, "ignore_eos": True, "seed": 9}
    configs = sampling_configs(model.get_request_resource_configs([], [], kwargs), model.declare_state())
    reference = SamplingReqConfig(temperature=0.0, top_p=0.5, top_k=3, repetition_penalty=1.3, ignore_eos=True)
    assert configs == {SAMPLER: reference}


@pytest.mark.parametrize("size", QWEN3_5)
def test_qwen3_5_declarations_have_no_problems(size):
    assert state_declaration_problems(_qwen3_5(**QWEN3_5[size])) == []


def test_qwen3_5_declares_the_conv_update_beside_the_delta_net_with_no_spec_of_its_own():
    model = _qwen3_5(**QWEN3_5["9B"])
    gdn = next(declared for declared in model.declare_state() if declared.name == GDN_STATE)
    assert set(gdn.updates) == {CONV, LINEAR_ATTN}
    assert CONV not in {spec.resource_key for spec in node_resources(model)}


# Qwen3.5's real stack on handles


class _QwenKV:
    def __init__(self, array):
        self.pages = torch.zeros(len(array.layers), 8, 2, *array.entry.tensors["k"].shape)

    def layer_view(self, i):
        return self.pages[i]

    def write_kv(self, k, v, layer_idx=None, label=None):
        pass

    def opens_label(self, rid, label):
        return True


class _QwenPool:
    def __init__(self, array):
        self.blocks = {
            field: torch.zeros(len(array.layers), 8, *tensor.shape) for field, tensor in array.entry.tensors.items()
        }

    def block(self, field, i):
        return self.blocks[field][i]


class _QwenAttn:
    requires_kv_write = True

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        return q.clone()


class _QwenLinear:
    def run_conv(self, x, conv_layer, weight, bias, label):
        return x.clone()

    def run(self, q, k, v, a, b, state_layer, a_log, dt_bias, label):
        return v.clone()


def _next_step(*handles) -> None:
    step = SubmoduleStep(segments=[Segment(1, "main", 3)], state={h.name: ArrayStep() for h in handles})
    step.set_ctx(StepContext((1,), "decode", 0, False))
    for handle in handles:
        handle.end_step(step)
        handle.begin_step(step)


def _qwen3_5_stack(blocks: int = 2):
    """``blocks`` of three delta-net layers and one full-attention layer, so a frame reused across layers shows."""
    layer_types = ([LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]) * blocks
    config = replace(
        tiny_config(), num_hidden_layers=len(layer_types), layer_types=layer_types, mrope_section=[6, 5, 5],
    )
    model = Qwen3_5ForCausalLM(config, CommGroup(0, 0, [0])).eval()
    generator = torch.Generator().manual_seed(0)
    for parameter in model.parameters():  # the model's own parameters start uninitialized
        parameter.data.normal_(0, 0.02, generator=generator)
    stack = Qwen3_5DenseModel.__new__(Qwen3_5DenseModel)
    stack.config, stack.vision_config = config, None
    declared = {d.name: d for d in stack.declare_state()}
    kv_array, gdn_array = declared[KV_CACHE], declared[GDN_STATE]
    kv = build_handle(kv_array, {KV_CACHE: _QwenKV(kv_array), ATTN: _QwenAttn(), ROPE: None})
    gdn = build_handle(gdn_array, {GDN_STATE: _QwenPool(gdn_array), CONV: None, LINEAR_ATTN: _QwenLinear()})
    for module in model.modules():
        if hasattr(module, "bind_state"):
            module.bind_state({KV_CACHE: kv, GDN_STATE: gdn})
    _next_step(kv, gdn)
    return config, model, kv, gdn


def _tables(model, ids) -> tuple[torch.Tensor, torch.Tensor]:
    positions_3d = torch.arange(len(ids)).unsqueeze(0).expand(3, -1)
    return model.model.build_cos_sin(positions_3d, dtype=model.embed_dtype)


@torch.no_grad()
def test_each_qwen3_5_layer_names_itself_to_its_handle_by_its_id_in_the_stack():
    config, model, kv, gdn = _qwen3_5_stack()
    seen = []
    kv._record_add = lambda layer, keep: seen.append(("add", layer))
    kv._record_read = lambda layer: seen.append(("read", layer))
    gdn._record_update = lambda field, layer: seen.append((field, layer))
    ids = torch.tensor([1, 2, 3])
    model(model.embed(ids), label="main", cos_sin=_tables(model, ids))
    expected = []
    for layer, kind in enumerate(config.layer_types):
        calls = ("conv", "state") if kind == LINEAR_ATTENTION else ("add", "read")
        expected += [(call, layer) for call in calls]
    assert seen == expected


@torch.no_grad()
def test_a_qwen3_5_stack_compiled_as_a_capture_compiles_no_frame_per_layer():
    """A capture compiles with `dynamic=False`, where a module's int specializes per layer and hits the recompile limit,
    and the frames then run uncompiled."""
    torch._dynamo.reset()
    config, model, kv, gdn = _qwen3_5_stack()
    guards = []
    frames = []

    def backend(graph_module, inputs):
        frames.append(graph_module)
        return graph_module.forward

    optimize = torch._dynamo.optimize(backend, dynamic=False, guard_fail_fn=lambda f: guards.append(str(f.reason)))
    compiled = optimize(model)
    ids = torch.tensor([1, 2, 3])
    tables, embeds = _tables(model, ids), model.embed(ids)
    expected = model(embeds, label="main", cos_sin=tables)
    _next_step(kv, gdn)
    actual = compiled(embeds, label="main", cos_sin=tables)
    assert torch.allclose(actual, expected), "the compiled stack should run what the eager one does"
    assert not [guard for guard in guards if "layer" in guard], "one frame should serve every layer"
    first_call = len(frames)
    _next_step(kv, gdn)
    with torch.compiler.set_stance("fail_on_recompile"):
        compiled(embeds, label="main", cos_sin=tables)
    assert len(frames) == first_call, "a capture's stance raises on any new compile"


def _compiled_frames(blocks: int) -> int:
    torch._dynamo.reset()
    _, model, _, _ = _qwen3_5_stack(blocks)
    frames = []

    def backend(graph_module, inputs):
        frames.append(graph_module)
        return graph_module.forward

    ids = torch.tensor([1, 2, 3])
    torch._dynamo.optimize(backend, dynamic=False)(model)(model.embed(ids), label="main", cos_sin=_tables(model, ids))
    return len(frames)


@torch.no_grad()
def test_a_longer_qwen3_5_stack_compiles_no_more_frames():
    """The loop stays eager and each layer's frame is reused, as `run_layers` needs its cursor to make them."""
    assert _compiled_frames(blocks=4) == _compiled_frames(blocks=2) > 1


def test_the_frame_around_qwen3_5s_stack_is_marked_for_dynamo_to_skip():
    """It holds no tensor work, and would otherwise recompile on every shape. Read from the source, since
    `torch._dynamo.reset` clears a mark. `test_a_frame_marked_eager_is_skipped_and_what_it_calls_compiles` shows what
    the mark does."""
    source = ast.parse(inspect.getsource(Qwen3_5LanguageModel))
    forward = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == "forward")
    assert [decorator.id for decorator in forward.decorator_list] == ["eager_frame"]


@torch.no_grad()
@pytest.mark.parametrize("walk, last_token", [("prefill_text", True), ("decode", False)])
def test_qwen3_5_takes_each_rows_last_token_on_a_prefill_only(monkeypatch, walk, last_token):
    llm = LLMSubmodule(None, SimpleNamespace(stop_token_ids=frozenset()))
    llm.model = Mock(return_value=torch.randn(3, 4), embed=lambda ids: ids, logits=lambda x: x)
    llm.kv, llm.sampler = object(), object()
    monkeypatch.setattr(LLMSubmodule, "_rope_in_graph", lambda *args: False)
    taken = []
    monkeypatch.setattr("mstar.model.qwen3_5.submodules.at_last_token", lambda h, x: taken.append(h) or x[-1:])
    monkeypatch.setattr("mstar.model.qwen3_5.submodules.sample", lambda s, logits: logits)
    logits = llm._forward(walk, None, torch.tensor([1, 2, 3]), cos_3d=torch.ones(1), sin_3d=torch.ones(1))
    assert (taken == [llm.kv]) is last_token
    assert logits.shape[0] == (1 if last_token else 3)


def _qwen3_5_open_step(llm) -> SubmoduleStep:
    step = engine_step(llm, graph_walk="decode", request_ids=[1], inputs=[ARNodeInputs(input_seq_len=1)])
    step.set_ctx(StepContext((1,), "decode", 0, True))
    step.begin_calls()
    return step


def test_a_qwen3_5_step_that_never_samples_through_its_handle_is_refused():
    llm = _qwen3_5_llm(_qwen3_5(**QWEN3_5["9B"]))
    step = _qwen3_5_open_step(llm)
    # no forward runs, so only the sampler's count is checked: the arrays would refuse a step with no adds
    with pytest.raises(ValueError, match=r"Sampler 'sampler': a step listing it sampled 0 times, and draws is 1"):
        llm.sampler.check_step(step)
    step.end_calls()


def test_a_qwen3_5_step_counts_the_scaffolds_draw_through_its_handle():
    llm = _qwen3_5_llm(_qwen3_5(**QWEN3_5["9B"]))
    llm.sampler.resource = SimpleNamespace(sample=lambda rows, logits: logits)
    step = _qwen3_5_open_step(llm)
    llm.sample_tokens(None, torch.zeros(1, 4))
    llm.sampler.check_step(step)
    assert llm.sampler._drawn == 1, "the scaffold's forward should sample through the handle"
    step.end_calls()


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
    assert len(kv.groups) == 2 and kv.prefix_cache is False, "the local layers should keep only their window"


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
    kv = _command_a_plus(tmp_path, _COMMAND_A_PLUS).declare_state()[0]
    assert kv.reads[_GLOBAL_ATTN].layers == {i for i in range(32) if i % 4 == 3}
    assert kv.reads[_LOCAL_ATTN].layers | kv.reads[_GLOBAL_ATTN].layers == kv.layers, "a layer with no read"


def test_command_a_plus_writes_no_spec_by_hand(tmp_path):
    assert _command_a_plus(tmp_path).get_node_resources() == []


def test_command_a_plus_passes_a_shorter_max_seq_len_to_the_array(tmp_path):
    model = _command_a_plus(tmp_path, max_seq_len=32)
    assert model.declare_state()[0].max_len == 32
    _assert_equal_per_key(node_resources(model), _command_a_plus_reference(model))


@pytest.mark.parametrize("text", [None, _COMMAND_A_PLUS], ids=["tiny", "published"])
def test_command_a_plus_declarations_have_no_problems(tmp_path, text):
    assert state_declaration_problems(_command_a_plus(tmp_path, text)) == []


def test_each_ported_array_goes_to_the_storage_that_holds_its_kind(tmp_path, monkeypatch):
    def storages(model) -> dict:
        arrays = [declared for declared in model.declare_state() if isinstance(declared, Array)]
        return {array.name: storage_of(array) for array in arrays}

    assert storages(_qwen3_5_vision(monkeypatch)) == {KV_CACHE: PAGED, GDN_STATE: VALUE, VISION_TOKENS: WITHIN_STEP}, (
        "the array's shape decides its storage"
    )
    assert storages(_command_a_plus(tmp_path)) == {KV_CACHE: PAGED}, "the array's shape decides its storage"


def _command_a_plus_reference_step(walk, rids, inputs) -> SubmoduleStep:
    return SubmoduleStep(
        segments=[
            Segment(request_id=rid, label="main", span=inp.input_seq_len)
            for rid, inp in zip(rids, inputs, strict=True)
        ],
        steps={
            KV_CACHE: KVStep(),
            _LOCAL_ATTN: AttentionStep(causal=True),
            _GLOBAL_ATTN: AttentionStep(causal=True),
            ROPE: PositionStep(),
            SAMPLER: SamplerStep(
                apply_penalty=True,
                prefill_tracked_tokens={
                    rid: inp.input_ids for rid, inp in zip(rids, inputs, strict=True)
                } if walk != "decode" else {},
            ),
        },
    )


@pytest.mark.parametrize("walk, lengths", [
    ("prefill", [5, 2]), ("prefill_image", [9]), ("decode", [1, 1, 1]),
])
def test_command_a_plus_steps_equal_the_hand_written_ones(tmp_path, walk, lengths):
    model = _command_a_plus(tmp_path)
    node = _bind(CommandAPlusLLMSubmodule(torch.nn.Module(), model.config, (model.image_encoder,)), model)
    rids = list(range(len(lengths)))
    inputs = [ARNodeInputs(input_ids=torch.arange(n), input_seq_len=n) for n in lengths]
    step = engine_step(node, graph_walk=walk, request_ids=rids, inputs=inputs)
    _assert_steps_equal(step, _command_a_plus_reference_step(walk, rids, inputs))


def test_command_a_plus_layout_converts_to_the_hand_written_one(tmp_path):
    model = _command_a_plus(tmp_path)
    span = Span("ids", "prefill", 7, 7, "text_inputs")
    converted = kv_layouts(
        model.declare_state(), {KV_CACHE: {"main": Layout(spans=(span,), growth=(("decode_loop", 1),))}}, {},
    )
    reference = {KV_CACHE: {"main": LabelLayout(spans=(span,), growth=(("decode_loop", 1),))}}
    assert converted == reference
    assert converted[KV_CACHE]["main"].row() == reference[KV_CACHE]["main"].row()
