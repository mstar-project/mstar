"""The speech models on the state contract build the specs, steps and layouts they wrote by hand, field for field.

Nothing in the engine changes when a model moves to declarations, because it sees the same objects. The hand-written
specs and steps below are the reference, copied from the models as they stood before the port, so the adapter can't pass
by agreeing with itself. Higgs's position spec is the one named difference, and its reference says why.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

sys.path.insert(0, ".")
sys.path.insert(0, "test/modular")

import pytest
import torch
import transformers
import yaml
from test_startup_graph_validation import _bare
from test_state_ports import (
    _assert_equal_per_key,
    _assert_steps_equal,
    _bind,
    _next_step,
    _published,
    _QwenAttn,
    _QwenKV,
    _StubResource,
)
from test_voxcpm2_model import TINY, _Tokenizer
from torch import nn

from mstar.engine.accelerator_graph_config import PiecewisePackedConfig
from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig, RaggedAttentionSpec
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.engine.state import Array, Span
from mstar.engine.state.adapters import (
    StateSetup,
    build_handle,
    kv_layouts,
    node_handles,
    node_resources,
    sampling_configs,
)
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.base import LabelLayout, keyed_labels, serving_resources
from mstar.model.higgs_audio.components.llm import HiggsAudioLLM
from mstar.model.higgs_audio.config import HiggsAudioModelConfig
from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel
from mstar.model.higgs_audio.submodules import HiggsAudioLLMSubmodule
from mstar.model.kokoro.kokoro_model import KokoroModel
from mstar.model.omnilingual_asr.config import ctc_config
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
from mstar.model.omnilingual_asr.submodules import ENCODER_LOOP, OmniASRCTCSubmodule
from mstar.model.omnivoice.omnivoice_model import OmniVoiceModel
from mstar.model.omnivoice.submodules import BACKBONE_LAYERS, OmniVoiceBackboneSubmodule
from mstar.model.orpheus.components.language_model import OrpheusLanguageModel
from mstar.model.orpheus.config import OrpheusModelConfig
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
from mstar.model.registry import model_init_kwargs
from mstar.model.submodule_base import ARNodeInputs, NodeInputs, engine_step
from mstar.model.voxcpm2.components.minicpm import MiniCPMLM
from mstar.model.voxcpm2.config import VoxCPM2Config
from mstar.model.voxcpm2.submodules import TalkerSubmodule
from mstar.model.voxcpm2.voxcpm2_model import VoxCPM2Model

KV_CACHE, ATTN, ROPE, SAMPLER = "kv_cache", "attn", "rope", "sampler"
BASE_KV, BASE_ATTN, BASE_POS, RES_KV, RES_ATTN = "base_kv", "base_attn", "base_pos", "res_kv", "res_attn"
ENCODER_ATTN, BACKBONE_ATTN = "encoder_attn", "backbone_attn"
PROMPT = Span("ids", "prefill", 7, 7, "text_inputs")


def _orpheus() -> OrpheusModel:
    return _bare(OrpheusModel, config=OrpheusModelConfig())


def _higgs() -> HiggsAudioModel:
    return _bare(HiggsAudioModel, config=HiggsAudioModelConfig())


def _omnilingual(dtype: torch.dtype = torch.bfloat16, size: str = "300m") -> OmnilingualASRModel:
    return _bare(OmnilingualASRModel, config=ctc_config(size, "v2"), dtype=dtype, max_batch_size=16)


def _omnivoice() -> OmniVoiceModel:
    return OmniVoiceModel("k2-fsa/OmniVoice", skip_weight_loading=True)


def _voxcpm2() -> VoxCPM2Model:
    return _bare(VoxCPM2Model, config=VoxCPM2Config())


def _orpheus_reference(model) -> list:
    c = model.config
    return [
        KVSpec(
            resource_key=KV_CACHE, nodes={"LLM"},
            config=PagedKVConfig(
                num_layers=c.num_hidden_layers, num_kv_heads=c.num_key_value_heads, head_dim=c.head_dim,
                max_seq_len=c.max_position_embeddings, num_qo_heads=c.num_attention_heads,
            ),
        ),
        AttentionSpec(resource_key=ATTN, nodes={"LLM"}, config=AttentionConfig(kv_cache=KV_CACHE)),
        SamplerSpec(resource_key=SAMPLER, nodes={"LLM"}, vocab_size=c.vocab_size, enable_repetion_penalty=True),
        PositionSpec(
            resource_key=ROPE, nodes={"LLM"},
            config=PositionConfig(
                kv_cache=KV_CACHE, rope_theta=c.rope_theta, rope_scale=c.rope_scaling["factor"],
                low_freq_factor=c.rope_scaling["low_freq_factor"], high_freq_factor=c.rope_scaling["high_freq_factor"],
                old_context_len=c.rope_scaling["original_max_position_embeddings"],
            ),
        ),
    ]


def _higgs_reference(model) -> list:
    c = model.config
    return [
        KVSpec(
            resource_key=KV_CACHE, nodes={"LLM"},
            config=PagedKVConfig(
                num_layers=c.num_hidden_layers, num_kv_heads=c.num_key_value_heads, head_dim=c.head_dim,
                max_seq_len=c.max_position_embeddings, num_qo_heads=c.num_attention_heads, max_num_pages=256,
            ),
        ),
        AttentionSpec(resource_key=ATTN, nodes={"LLM"}, config=AttentionConfig(kv_cache=KV_CACHE)),
        # The hand-written config set only rope_theta, and its layers passed these three on every call, which call
        # arguments win over the config. Layers on the contract pass nothing, so the declaration carries them, and
        # flashinfer's Llama 3.1 kernel stays.
        PositionSpec(
            resource_key=ROPE, nodes={"LLM"},
            config=PositionConfig(
                kv_cache=KV_CACHE, rope_theta=c.rope_theta, low_freq_factor=1.0, high_freq_factor=1.0,
                old_context_len=8192,
            ),
        ),
        SamplerSpec(resource_key=SAMPLER, nodes={"LLM"}, vocab_size=c.vocab_size, enable_repetion_penalty=False),
    ]


def _omnivoice_reference(model) -> list:
    body = model.config.backbone
    return [RaggedAttentionSpec(
        resource_key=BACKBONE_ATTN, nodes={"backbone"},
        config=RaggedAttentionConfig(
            num_qo_heads=body.num_attention_heads, num_kv_heads=body.num_key_value_heads, head_dim=body.head_dim,
            max_segments_per_request=2, dtype=getattr(torch, model.config.load_dtype),
        ),
    )]


def _omnilingual_reference(model) -> list:
    if model.dtype not in (torch.bfloat16, torch.float16):
        return []
    c = model.config
    return [RaggedAttentionSpec(
        resource_key=ENCODER_ATTN, nodes={"encoder"},
        config=RaggedAttentionConfig(
            num_qo_heads=c.num_heads, num_kv_heads=c.num_heads, head_dim=c.model_dim // c.num_heads, dtype=model.dtype,
            max_segments_per_step=model.max_batch_size,
        ),
    )]


def _voxcpm2_reference(model) -> list:
    def kv(stack) -> PagedKVConfig:
        config = PagedKVConfig(
            num_layers=stack.num_hidden_layers, num_kv_heads=stack.num_key_value_heads, head_dim=stack.head_dim,
            num_qo_heads=stack.num_attention_heads, max_seq_len=model.config.max_length,
        )
        config.max_num_pages = 1 + model.DEFAULT_MAX_CONCURRENT * -(-config.max_seq_len // config.tokens_per_page)
        return config

    nodes = {"Talker"}
    return [
        KVSpec(resource_key=BASE_KV, nodes=nodes, config=kv(model.config.lm)),
        AttentionSpec(resource_key=BASE_ATTN, nodes=nodes, config=AttentionConfig(kv_cache=BASE_KV)),
        PositionSpec(resource_key=BASE_POS, nodes=nodes, config=PositionConfig(kv_cache=BASE_KV)),
        KVSpec(resource_key=RES_KV, nodes=nodes, config=kv(model.config.residual)),
        AttentionSpec(resource_key=RES_ATTN, nodes=nodes, config=AttentionConfig(kv_cache=RES_KV)),
    ]


_PAGES = {"page_size": 64, "max_num_pages": 512}
MODELS = {
    "orpheus": (_orpheus, _orpheus_reference, {KV_CACHE: _PAGES}),
    "higgs": (_higgs, _higgs_reference, {KV_CACHE: _PAGES}),
    "omnivoice": (_omnivoice, _omnivoice_reference, {BACKBONE_ATTN: {"max_segments_per_request": 4}}),
    "omnilingual": (_omnilingual, _omnilingual_reference, {ENCODER_ATTN: {"max_segments_per_step": 8}}),
    "voxcpm2": (_voxcpm2, _voxcpm2_reference, {BASE_KV: _PAGES, RES_KV: _PAGES}),
}
_COMPARED = {
    "OrpheusModel": "orpheus", "HiggsAudioModel": "higgs", "OmniVoiceModel": "omnivoice",
    "OmnilingualASRModel": "omnilingual", "VoxCPM2Model": "voxcpm2",
}
_PUBLISHED = [(path, name) for cls, name in _COMPARED.items() for path in _published(cls)]


def test_the_published_configs_are_found():
    assert [sum(name == other for _, name in _PUBLISHED) for other in MODELS] == [4, 1, 2, 4, 1]
    assert len(_published("KokoroModel")) == 2, "an empty list would skip their tests"


@pytest.mark.parametrize("name", MODELS)
def test_specs_equal_the_hand_written_ones(name):
    build, reference, _ = MODELS[name]
    model = build()
    _assert_equal_per_key(node_resources(model), reference(model))


@pytest.mark.parametrize("name", MODELS)
def test_specs_equal_the_hand_written_ones_after_a_deployments_overrides(name):
    build, reference, overrides = MODELS[name]
    model = build()
    deployment = {"resources": overrides}
    expected = reference(model)
    apply_yaml_overrides(expected, deployment)
    specs = serving_resources(model, deployment)
    _assert_equal_per_key(specs, expected)
    for key, settings in overrides.items():
        config = next(spec for spec in specs if spec.resource_key == key).config
        assert all(getattr(config, field) == value for field, value in settings.items()), "the YAML should reach it"


@pytest.mark.parametrize("path, name", _PUBLISHED, ids=lambda value: getattr(value, "stem", None))
def test_specs_equal_the_hand_written_ones_on_every_published_config(path, name):
    deployment = yaml.safe_load(path.read_text())
    build, reference, _ = MODELS[name]
    size = model_init_kwargs(deployment["model"]).get("size")
    model = _omnilingual(size=size) if size else build()
    expected = reference(model)
    apply_yaml_overrides(expected, deployment)
    _assert_equal_per_key(serving_resources(model, deployment), expected)


def test_omnilingual_in_fp32_declares_nothing_and_sets_no_kernel():
    model = _omnilingual(torch.float32)
    assert node_resources(model) == [] and model.kernel_settings() == {}


def test_kokoro_declares_nothing():
    model = _bare(KokoroModel)
    assert node_resources(model) == [] and model.kernel_settings() == {}


def _segments(rids, inputs) -> list[Segment]:
    return [Segment(rid, "main", inp.input_seq_len) for rid, inp in zip(rids, inputs, strict=True)]


def _orpheus_llm() -> OrpheusLLMSubmodule:
    language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=None), lm_head=None)
    return _bind(OrpheusLLMSubmodule(language_model, OrpheusModelConfig()), _orpheus())


def _orpheus_reference_step(rids, inputs, prefill_tokens) -> SubmoduleStep:
    return SubmoduleStep(
        segments=_segments(rids, inputs),
        steps={
            KV_CACHE: KVStep(),
            ATTN: AttentionStep(causal=True),
            SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=prefill_tokens),
            ROPE: PositionStep(),
        },
    )


@pytest.mark.parametrize("walk", ["prefill", "decode", "mixed"])
def test_orpheus_steps_equal_the_hand_written_ones(walk):
    rows = {
        "prefill": [("prefill", 5), ("prefill", 3)],
        "decode": [("decode", 1), ("decode", 1)],
        "mixed": [("prefill", 5), ("decode", 1)],
    }[walk]
    rids = [1, 2]
    inputs = [
        ARNodeInputs(input_ids=torch.arange(n), input_seq_len=n, graph_walk=None if walk != "mixed" else own)
        for own, n in rows
    ]
    tokens = {rid: inp.input_ids for rid, inp, (own, _) in zip(rids, inputs, rows, strict=True) if own == "prefill"}
    step = engine_step(_orpheus_llm(), graph_walk=walk, request_ids=rids, inputs=inputs)
    _assert_steps_equal(step, _orpheus_reference_step(rids, inputs, tokens))


def _higgs_llm() -> HiggsAudioLLMSubmodule:
    return _bind(HiggsAudioLLMSubmodule(None, HiggsAudioModelConfig()), _higgs())


def _higgs_reference_step(rids, inputs, kept_rids) -> SubmoduleStep:
    return SubmoduleStep(
        segments=_segments(rids, inputs),
        steps={
            KV_CACHE: KVStep(),
            ATTN: AttentionStep(causal=True),
            SAMPLER: SamplerStep(apply_penalty=False, kept_rids=kept_rids),
            ROPE: PositionStep(),
        },
    )


@pytest.mark.parametrize("last", [frozenset(), frozenset({2})])
def test_higgs_prefill_step_equals_the_hand_written_one(last):
    rids, inputs = [1, 2], [ARNodeInputs(input_seq_len=5), ARNodeInputs(input_seq_len=3)]
    info = {rid: SimpleNamespace(step_metadata={"is_last_prefill": rid in last}) for rid in rids}
    step = engine_step(_higgs_llm(), graph_walk="prefill_audio", request_ids=rids, inputs=inputs, per_request_info=info)
    _assert_steps_equal(step, _higgs_reference_step(rids, inputs, last))


def test_higgs_decode_step_equals_the_hand_written_one():
    rids, inputs = [1, 2], [ARNodeInputs(input_seq_len=1), ARNodeInputs(input_seq_len=1)]
    step = engine_step(_higgs_llm(), graph_walk="decode", request_ids=rids, inputs=inputs)
    _assert_steps_equal(step, _higgs_reference_step(rids, inputs, None))


def _within_step_reference(segments, key) -> SubmoduleStep:
    return SubmoduleStep(segments=segments, steps={key: AttentionStep(causal=False)})


def _omnivoice_backbone() -> OmniVoiceBackboneSubmodule:
    model = _omnivoice()
    return _bind(OmniVoiceBackboneSubmodule(None, model.config), model, node_name="backbone")


def _region_step(node, config, rids, lengths) -> SubmoduleStep:
    runner = object.__new__(PiecewiseAcceleratorGraphRunner)
    runner._config, runner._lora_keys, runner._state_handles = config, (), node.state_handles
    return runner._region_step(rids, lengths)


def test_omnivoice_eager_step_equals_the_hand_written_one_with_an_empty_unconditional_document():
    backbone = _omnivoice_backbone()
    inputs = [NodeInputs(kwargs={"item": SimpleNamespace(doc_pair=pair)}) for pair in ((5, 3), (4, 0))]
    step = engine_step(backbone, graph_walk="speech_gen", request_ids=[1, 2], inputs=inputs)
    segments = [Segment(1, "main", 5), Segment(1, "main", 3), Segment(2, "main", 4), Segment(2, "main", 0)]
    _assert_steps_equal(step, _within_step_reference(segments, BACKBONE_ATTN))


def test_omnivoice_leased_step_declares_none_as_it_did():
    backbone = _omnivoice_backbone()
    inputs = [NodeInputs(kwargs={"item": SimpleNamespace(doc_pair=(5, 3))})]
    step = engine_step(
        backbone, graph_walk="speech_gen", request_ids=[1], inputs=inputs, piecewise_leases={BACKBONE_LAYERS: object()},
    )
    assert step is None, "a leased replay plans its own"


@pytest.mark.parametrize("rids, lengths, segments", [
    # a padded replay: two documents per real request and one zero per padding row, spread over the ids in order
    ([1, 2, 91, 92], [5, 3, 4, 0, 0, 0], [(1, 5), (1, 3), (2, 4), (2, 0), (91, 0), (91, 0)]),
    # a capture: one length per row
    ([91, 92], [8, 8], [(91, 8), (92, 8)]),
], ids=["padded-replay", "capture"])
def test_omnivoice_region_step_equals_the_hand_written_one(rids, lengths, segments):
    backbone = _omnivoice_backbone()
    region = backbone.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.bfloat16)[BACKBONE_LAYERS]
    step = _region_step(backbone, region, rids, lengths)
    reference = _within_step_reference([Segment(rid, "main", n) for rid, n in segments], BACKBONE_ATTN)
    _assert_steps_equal(step, reference)


class _OneParameter(nn.Module):
    """As much of a model as a submodule's ``__init__`` reads, its dtype and device from a parameter."""

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.config = SimpleNamespace(model_dim=64)

    def frame_lengths(self, lengths: list[int]) -> list[int]:
        return lengths


def _omnilingual_encoder() -> OmniASRCTCSubmodule:
    node = OmniASRCTCSubmodule(_OneParameter(torch.bfloat16), max_batch_size=16, max_batch_seconds=320.0)
    return _bind(node, _omnilingual(), node_name="encoder")


def test_omnilingual_eager_step_equals_the_hand_written_one():
    rids, inputs = [1, 2], [NodeInputs(input_seq_len=5), NodeInputs(input_seq_len=3)]
    step = engine_step(_omnilingual_encoder(), graph_walk="transcribe", request_ids=rids, inputs=inputs)
    _assert_steps_equal(step, _within_step_reference(_segments(rids, inputs), ENCODER_ATTN))


def test_omnilingual_leased_step_declares_none_as_it_did():
    step = engine_step(
        _omnilingual_encoder(), graph_walk="transcribe", request_ids=[1], inputs=[NodeInputs(input_seq_len=5)],
        piecewise_leases={ENCODER_LOOP: object()},
    )
    assert step is None, "a leased replay plans its own"


def test_omnilingual_region_step_equals_the_hand_written_one():
    encoder = _omnilingual_encoder()
    region = encoder.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.bfloat16)[ENCODER_LOOP]
    reference = PiecewisePackedConfig(
        capture_fn=region.capture_fn, make_static_inputs=region.make_static_inputs, attn_key=ENCODER_ATTN,
        lease_before_step=True, can_lease=region.can_lease, one_row_per_step=True,
        total_tokens=region.total_tokens, capture_batch_sizes=[1],
    ).region_step([7], [3, 5])
    _assert_steps_equal(_region_step(encoder, region, [7], [3, 5]), reference)


def test_omnilingual_in_fp32_declares_no_step_and_no_region():
    node = OmniASRCTCSubmodule(_OneParameter(torch.float32), max_batch_size=16, max_batch_seconds=320.0)
    step = engine_step(node, graph_walk="transcribe", request_ids=[1], inputs=[NodeInputs(input_seq_len=5)])
    assert step is None and node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32) == {}


def _talker() -> TalkerSubmodule:
    model = _voxcpm2()
    node = TalkerSubmodule(_OneParameter(torch.bfloat16), model.config, 4)
    resources = {spec.resource_key: _StubResource() for spec in node_resources(model)}
    node.bind_node_state(node_handles(model, "Talker", resources, setup=StateSetup()))
    return node


@pytest.mark.parametrize("walk, lengths", [
    ("talker_prefill", (5, 3)), ("talker_prefill_clone", (9, 4)), ("talker_decode", (1, 1)),
])
def test_voxcpm2_steps_equal_the_hand_written_ones(walk, lengths):
    rids = [1, 2]
    inputs = [ARNodeInputs(input_seq_len=n, kwargs={"step": 0, "rng": b"state"}) for n in lengths]
    step = engine_step(_talker(), graph_walk=walk, request_ids=rids, inputs=inputs)
    reference = SubmoduleStep(
        segments=_segments(rids, inputs),
        steps={
            BASE_KV: KVStep(),
            BASE_ATTN: AttentionStep(causal=True),
            BASE_POS: PositionStep(),
            RES_KV: KVStep(),
            RES_ATTN: AttentionStep(causal=True),
        },
    )
    _assert_steps_equal(step, reference)


def test_orpheus_layout_converts_to_the_hand_written_one_with_its_keys():
    model = _orpheus()
    model.tokenizer = lambda prompt, return_tensors: SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))
    output = model.process_prompt("hello", ["text"], ["audio"])
    converted = kv_layouts(model.declare_state(), output.metadata["state_layout"], {})[KV_CACHE]["main"]
    n = output.new_input_tensors["text_inputs"][0].numel()
    reference = LabelLayout(spans=(Span("ids", "prefill", n, n, "text_inputs"),), growth=(("decode_loop", 1),))
    assert converted == reference and converted.row() == reference.row()
    assert converted.keyed_count() == 1, "the prompt span is keyed"
    keys = keyed_labels(model)[KV_CACHE]["main"].keys
    assert keys.tensor == "text_inputs" and keys.walks == ("prefill",)


def test_higgs_layout_converts_to_the_hand_written_one_with_no_keys():
    from test_model_kv_layouts import _higgs as _higgs_with_a_processor

    model = _higgs_with_a_processor()
    output = model.process_prompt("transcribe", ["audio"], ["text"], tensors={"audio_inputs": [torch.zeros(16000)]})
    [layout] = output.metadata["state_layout"][KV_CACHE].values()
    converted = kv_layouts(model.declare_state(), output.metadata["state_layout"], {})[KV_CACHE]["main"]
    (span,) = layout.spans
    assert span.kind == "rows" and span.walk == "prefill_audio" and span.source == "audio_features"
    reference = LabelLayout(spans=(span,), growth=(("decode_loop", 1),))
    assert converted == reference and converted.row() == reference.row()
    assert converted.keyed_count() == 0 and model.kv_labels()[KV_CACHE]["main"].keys is None


@pytest.fixture
def tiny_voxcpm2(tmp_path, monkeypatch) -> VoxCPM2Model:
    (tmp_path / "config.json").write_text(json.dumps(TINY))
    monkeypatch.setattr(transformers.LlamaTokenizerFast, "from_pretrained", classmethod(lambda cls, path: _Tokenizer()))
    return VoxCPM2Model(model_path_hf=str(tmp_path))


def test_voxcpm2_layout_converts_to_the_hand_written_one_for_both_arrays(tiny_voxcpm2):
    output = tiny_voxcpm2.process_prompt("hello world", ["text"], ["audio"])
    converted = kv_layouts(tiny_voxcpm2.declare_state(), output.metadata["state_layout"], {})
    (span,) = output.metadata["state_layout"][BASE_KV]["main"].spans
    assert span.kind == "ids" and span.walk == "talker_prefill"
    reference = LabelLayout(spans=(span,), growth=(("talker_decode_loop", 1),))
    assert converted == {BASE_KV: {"main": reference}, RES_KV: {"main": reference}}, "the Value's fields have no layout"


@pytest.mark.parametrize("build, kwargs, reference", [
    (_orpheus, {"temperature": 0.0, "top_p": 0.5, "repetition_penalty": 1.3, "ignore_eos": True},
     SamplingReqConfig(temperature=0.0, top_p=0.5, repetition_penalty=1.3, ignore_eos=True)),
    (_orpheus, {}, SamplingReqConfig(
        temperature=OrpheusModelConfig().temperature, top_p=OrpheusModelConfig().top_p,
        repetition_penalty=OrpheusModelConfig().repetition_penalty, ignore_eos=OrpheusModelConfig().ignore_eos,
    )),
    (_higgs, {"temperature": 0.7, "top_p": 0.5, "ignore_eos": True},
     SamplingReqConfig(temperature=0.7, top_p=0.5, ignore_eos=True)),
    (_higgs, {}, SamplingReqConfig(temperature=0.0, top_p=1.0, ignore_eos=False)),
], ids=["orpheus", "orpheus-defaults", "higgs", "higgs-greedy-by-default"])
def test_request_settings_equal_the_hand_written_config(build, kwargs, reference):
    model = build()
    configs = sampling_configs(model.get_request_resource_configs({}, kwargs), model.declare_state())
    assert configs == {SAMPLER: reference}


@pytest.mark.parametrize("build", [_orpheus, _higgs, _omnivoice, _omnilingual], ids=lambda build: build.__name__)
def test_declarations_have_no_problems(build):
    assert state_declaration_problems(build()) == []


def test_voxcpm2_declarations_have_no_problems_with_host_fields_on_a_node_with_no_prefix_keys(tiny_voxcpm2):
    assert state_declaration_problems(tiny_voxcpm2) == []


def _kernel_arguments(manager: RopeManager, monkeypatch, **call) -> tuple:
    """What ``manager`` hands the rotation op: everything but the tensors it rotates."""
    seen = []

    def rotate(*args, **kwargs) -> None:
        seen.append((args[3:], kwargs))

    monkeypatch.setattr(torch.ops.mstar, "rope_apply_qk_inplace", rotate)
    q, k = torch.zeros(3, 4, 16), torch.zeros(3, 2, 16)
    manager.apply_qk(q, k, label="main", pos_ids=torch.arange(3), rope_dtype=q.dtype, **call)
    ((arguments, kwargs),) = seen
    return arguments, kwargs


@pytest.mark.parametrize("build, passed", [
    (_orpheus, lambda c: dict(
        rope_theta=c.rope_theta, rope_scale=c.rope_scaling["factor"],
        low_freq_factor=c.rope_scaling["low_freq_factor"], high_freq_factor=c.rope_scaling["high_freq_factor"],
        old_context_len=c.rope_scaling["original_max_position_embeddings"],
    )),
    (_higgs, lambda c: dict(
        rope_theta=c.rope_theta, rope_scale=1.0, low_freq_factor=1.0, high_freq_factor=1.0, old_context_len=8192,
    )),
], ids=["orpheus", "higgs"])
def test_a_declared_rope_reaches_the_kernel_with_what_the_layers_passed_on_every_call(build, passed, monkeypatch):
    """The layers named these five values per call, and call arguments win over the config, so a declaration that left
    any out would pick another kernel with no error."""
    model = build()
    spec = next(spec for spec in node_resources(model) if spec.resource_key == ROPE)
    declared = RopeManager(spec.config, torch.device("cpu"))
    hand_written = RopeManager(PositionConfig(kv_cache=KV_CACHE), torch.device("cpu"))
    assert _kernel_arguments(declared, monkeypatch) == _kernel_arguments(
        hand_written, monkeypatch, **passed(model.config),
    )


class _Positions:
    """Multiplies q and k by 1 in place, since the kernel rotates in place. These tests read which frames compile."""

    def __init__(self, length: int = 3):
        self.ids = torch.arange(length)

    def apply_qk(self, q, k, label, rope_dtype):
        q.mul_(1.0)
        k.mul_(1.0)
        return q, k

    def pos_ids(self, label):
        return self.ids


def _bound(stack: nn.Module, declared: list, resources: dict):
    handles = {}
    for array in declared:
        if isinstance(array, Array) and array.max_len:
            handles[array.name] = build_handle(array, resources)
    for module in stack.modules():
        if hasattr(module, "bind_state"):
            module.bind_state(handles)
    return list(handles.values())


def _initialized(stack: nn.Module) -> nn.Module:
    generator = torch.Generator().manual_seed(0)
    for parameter in stack.parameters():    # a model's own parameters start uninitialized
        parameter.data.normal_(0, 0.02, generator=generator)
    return stack.eval()


def _paged_stack(stack_class, model_class, config_class, layers: int):
    config = config_class(
        num_hidden_layers=layers, hidden_size=64, num_attention_heads=4, head_dim=16, num_key_value_heads=2,
        intermediate_size=128, vocab_size=64,
    )
    stack = _initialized(stack_class(config))
    declared = model_class.declare_state(SimpleNamespace(config=config))
    kv = next(array for array in declared if array.name == KV_CACHE)
    handles = _bound(stack, declared, {KV_CACHE: _QwenKV(kv), ATTN: _QwenAttn(), ROPE: _Positions()})
    return stack, handles, (torch.zeros(3, 64),)


_orpheus_stack = partial(_paged_stack, OrpheusLanguageModel, OrpheusModel, OrpheusModelConfig)
_higgs_stack = partial(_paged_stack, HiggsAudioLLM, HiggsAudioModel, HiggsAudioModelConfig)


def _voxcpm2_stack(layers: int):
    config = VoxCPM2Config(lm=replace(
        VoxCPM2Config().lm, num_hidden_layers=layers, hidden_size=64, intermediate_size=96, num_attention_heads=4,
        head_dim=16,
    ))
    stack = _initialized(MiniCPMLM(config.lm, kv_key=BASE_KV, positions_from=BASE_KV, max_len=32))
    stack.materialize_buffers()
    declared = VoxCPM2Model.declare_state(SimpleNamespace(config=config))
    kv = next(array for array in declared if array.name == BASE_KV)
    handles = _bound(stack, [kv], {BASE_KV: _QwenKV(kv), BASE_ATTN: _QwenAttn(), BASE_POS: _Positions()})
    return stack, handles, (torch.zeros(3, 64),)


STACKS = {"orpheus": _orpheus_stack, "higgs": _higgs_stack, "voxcpm2": _voxcpm2_stack}


def _ops(graph_module) -> set[str]:
    return {getattr(node.target, "__name__", str(node.target)) for node in graph_module.graph.nodes}


@torch.no_grad()
@pytest.mark.parametrize("build", [_orpheus_stack, _higgs_stack], ids=["orpheus", "higgs"])
def test_the_rotation_compiles_in_the_frame_that_projects_q_and_k(build):
    """Without a QK norm, q and k are views of one projection. In a frame of their own they are aliased inputs to an
    in-place kernel, and a compiled capture fails Inductor's size check at its first run, at boot."""
    frames, _ = _compiled_frames(*build(layers=2))
    rotating = [frame for frame in frames if "mul_" in _ops(frame)]
    assert rotating and all("linear" in _ops(frame) for frame in rotating), "the rotation left the projection's frame"


def _compiled_frames(stack, handles, args, guards: list | None = None) -> tuple[list, nn.Module]:
    torch._dynamo.reset()
    frames = []

    def backend(graph_module, inputs):
        frames.append(graph_module)
        return graph_module.forward

    fail = None if guards is None else (lambda failure: guards.append(str(failure.reason)))
    compiled = torch._dynamo.optimize(backend, dynamic=False, guard_fail_fn=fail)(stack)
    _next_step(*handles)
    compiled(*args)
    return frames, compiled


@torch.no_grad()
@pytest.mark.parametrize("build", STACKS.values(), ids=STACKS)
def test_each_layer_names_itself_to_its_handle_by_its_id_in_the_stack(build):
    stack, handles, args = build(layers=2)
    seen = []
    handles[0]._record_add = lambda layer, keep: seen.append(("add", layer))
    handles[0]._record_read = lambda layer: seen.append(("read", layer))
    _next_step(*handles)
    stack(*args)
    assert seen == [("add", 0), ("read", 0), ("add", 1), ("read", 1)]


@torch.no_grad()
@pytest.mark.parametrize("build", STACKS.values(), ids=STACKS)
def test_a_stack_compiled_as_a_capture_compiles_no_frame_per_layer(build):
    """A capture compiles with `dynamic=False`, where a module's int specializes per layer and hits the recompile
    limit, and the frames then run uncompiled."""
    stack, handles, args = build(layers=2)
    guards = []
    frames, compiled = _compiled_frames(stack, handles, args, guards)
    assert not [guard for guard in guards if "layer" in guard], "one frame should serve every layer"
    first_call = len(frames)
    _next_step(*handles)
    with torch.compiler.set_stance("fail_on_recompile"):
        compiled(*args)
    assert len(frames) == first_call, "a capture's stance raises on any new compile"


@torch.no_grad()
@pytest.mark.parametrize("build", STACKS.values(), ids=STACKS)
def test_a_longer_stack_compiles_no_more_frames(build):
    assert len(_compiled_frames(*build(layers=4))[0]) == len(_compiled_frames(*build(layers=2))[0]) > 1
