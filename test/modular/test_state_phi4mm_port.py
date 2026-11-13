"""Phi-4-multimodal builds the specs, steps and layout it wrote by hand, field for field.

Nothing in the engine changes when a model moves to declarations, because it sees the same objects. The hand-written
specs and steps below are the reference, copied from the model as it stood before the port, so the adapter can't pass
by agreeing with itself. Two of its fields had no declaration before: the LLM's LongRoPE factors, which pick a
request's table by its prompt length, and the vision tower's FA3 kernel, since FlashInfer pads the tower's head_dim of
72 to 128.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")
sys.path.insert(0, "test/modular")

import torch
import yaml
from test_startup_graph_validation import _FACTORIES
from test_state_ports import _assert_equal_per_key, _assert_steps_equal, _bind

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig, RaggedAttentionSpec, RaggedAttnBackend
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.lora.config import LORA_RESOURCE_KEY
from mstar.engine.resources.position.config import MAX_POSITION_KWARG, PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.engine.state import Span
from mstar.engine.state.adapters import StateSetup, kv_layouts
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.base import LabelLayout, ProcessPromptOutput, enabled_resources, serving_resources
from mstar.model.keys import MIXED
from mstar.model.multimodal_decoder import MIXED_SPLIT, MultimodalDecoderModel
from mstar.model.phi4mm.components.language_model import Phi4MiniLM
from mstar.model.phi4mm.submodules import Phi4MMLLMSubmodule
from mstar.model.phi4mm.vision import Phi4MMVisionEncoderSubmodule
from mstar.model.submodule_base import ARNodeInputs, engine_step

PUBLISHED = Path(__file__).parents[2] / "configs" / "phi4mm.yaml"
# What a deployment with no `resources:` block of its own could set.
OVERRIDES = {"resources": {
    "kv_cache": {"max_num_pages": 2048}, "vision_attn": {"backend": "flashinfer", "max_segments_per_step": 64},
}}


def _phi4mm():
    return _FACTORIES["Phi4MMModel"]("phi4mm", {}, None, None)


# Less the opt-in LoRA spec, which `enabled_resources` builds only when a deployment names it. It holds callables, so
# a second build wouldn't equal it.
def _reference(model) -> list:
    cfg, vision = model.config, model.vision_config
    return [
        KVSpec(
            resource_key="kv_cache", nodes={"LLM"},
            config=PagedKVConfig(
                num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
                max_seq_len=cfg.max_position_embeddings, num_qo_heads=cfg.num_attention_heads,
            ),
        ),
        AttentionSpec(
            resource_key="attn", nodes={"LLM"}, config=AttentionConfig(kv_cache="kv_cache", splits=(MIXED_SPLIT,)),
        ),
        PositionSpec(
            resource_key="rope", nodes={"LLM"},
            config=PositionConfig(
                kv_cache="kv_cache", rotary_dim=cfg.rotary_dim, rope_theta=cfg.rope_theta, longrope=cfg.longrope,
                rope_table_dtype=torch.bfloat16,
            ),
        ),
        SamplerSpec(
            resource_key="sampler", nodes={"LLM"}, vocab_size=cfg.vocab_size, enable_repetition_penalty=True,
            enable_min_p=bool(model.sampling_defaults.get("min_p")),
        ),
        # The hand-written spec left `labels` empty. Only a sequence shard on the node reads them, and the vision node
        # has none.
        RaggedAttentionSpec(
            resource_key="vision_attn", nodes={"vision_encoder"}, labels={"main"},
            config=RaggedAttentionConfig(
                num_qo_heads=vision.num_attention_heads, num_kv_heads=vision.num_attention_heads,
                head_dim=vision.head_dim,
                max_segments_per_step=Phi4MMVisionEncoderSubmodule.MAX_BATCH_SIZE * (1 + vision.max_crops),
                backend=RaggedAttnBackend.DENSE,
            ),
        ),
    ]


def test_phi4mm_specs_equal_the_hand_written_ones():
    model = _phi4mm()
    _assert_equal_per_key(enabled_resources(model, {}, StateSetup()), _reference(model))
    assert state_declaration_problems(model) == []


def test_phi4mm_specs_equal_the_hand_written_ones_after_a_deployments_overrides():
    model = _phi4mm()
    expected = _reference(model)
    apply_yaml_overrides(expected, OVERRIDES)
    _assert_equal_per_key(serving_resources(model, OVERRIDES), expected)


def test_phi4mm_specs_equal_the_hand_written_ones_on_its_published_config():
    """``configs/phi4mm.yaml`` overrides only the LoRA pool, which the model still writes by hand."""
    model = _phi4mm()
    specs = serving_resources(model, yaml.safe_load(PUBLISHED.read_text()))
    lora = {spec.resource_key for spec in specs if spec.resource_key.startswith(LORA_RESOURCE_KEY)}
    assert len(lora) == 1, "the published config serves both adapters from one LoRA pool"
    _assert_equal_per_key([spec for spec in specs if spec.resource_key not in lora], _reference(model))


def test_phi4mm_writes_only_lora_by_hand():
    assert {spec.resource_key for spec in _phi4mm().get_node_resources()} == {LORA_RESOURCE_KEY}


def test_phi4mm_llm_layers_each_carry_their_own_kv_layer():
    """The other CPU tests never build the LLM, so a missing or wrong index would first fail on a GPU."""
    cfg = replace(
        _phi4mm().config, hidden_size=32, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
        num_hidden_layers=3,
    )
    layers = [None if layer.self_attn.layer is None else int(layer.self_attn.layer) for layer in Phi4MiniLM(cfg).layers]
    assert layers == list(range(cfg.num_hidden_layers)), "a layer would add and read another layer's KV, or none"


def _llm(model) -> Phi4MMLLMSubmodule:
    llm = Phi4MMLLMSubmodule(None, SimpleNamespace(stop_token_ids=frozenset()), encoders=model.encoders)
    return _bind(llm, model)


def _reference_step(rids, inputs, prefill_tokens, advance) -> SubmoduleStep:
    return SubmoduleStep(
        segments=[
            Segment(request_id=rid, label="main", span=inp.input_seq_len) for rid, inp in zip(rids, inputs, strict=True)
        ],
        steps={
            "kv_cache": KVStep(),
            "attn": AttentionStep(causal=True),
            "sampler": SamplerStep(apply_penalty=True, prefill_tracked_tokens=prefill_tokens),
            "rope": PositionStep(advance=advance),
        },
    )


def test_phi4mm_decode_step_equals_the_hand_written_one():
    llm = _llm(_phi4mm())
    rids, inputs = [1, 2], [ARNodeInputs(input_seq_len=1), ARNodeInputs(input_seq_len=1)]
    step = engine_step(llm, graph_walk="decode", request_ids=rids, inputs=inputs)
    _assert_steps_equal(step, _reference_step(rids, inputs, {}, None))


def test_phi4mm_prefill_text_step_equals_the_hand_written_one():
    llm = _llm(_phi4mm())
    rids = [1, 2]
    inputs = [ARNodeInputs(input_seq_len=n, input_ids=torch.arange(n)) for n in (5, 3)]
    step = engine_step(llm, graph_walk="prefill_text", request_ids=rids, inputs=inputs)
    _assert_steps_equal(step, _reference_step(rids, inputs, {1: inputs[0].input_ids, 2: inputs[1].input_ids}, None))


def test_phi4mm_mixed_step_equals_the_hand_written_one():
    llm = _llm(_phi4mm())
    rids = [1, 2]
    inputs = [
        ARNodeInputs(input_seq_len=1, graph_walk="decode"),
        ARNodeInputs(input_seq_len=3, input_ids=torch.arange(3), graph_walk="prefill_text"),
    ]
    step = engine_step(llm, graph_walk=MIXED, request_ids=rids, inputs=inputs)
    _assert_steps_equal(step, _reference_step(rids, inputs, {2: inputs[1].input_ids}, None))


def test_phi4mm_prefill_vision_step_equals_the_hand_written_one():
    llm = _llm(_phi4mm())
    tokens = torch.arange(4)
    inputs = [ARNodeInputs(input_seq_len=4, input_ids=tokens, resource_step_info=6)]
    step = engine_step(llm, graph_walk="prefill_vision", request_ids=[1], inputs=inputs)
    _assert_steps_equal(step, _reference_step([1], inputs, {1: tokens}, (6,)))


def _encoder(model) -> Phi4MMVisionEncoderSubmodule:
    return _bind(Phi4MMVisionEncoderSubmodule(torch.nn.Linear(1, 1), model.vision_config), model, "vision_encoder")


def test_phi4mm_vision_step_equals_the_hand_written_one():
    encoder = _encoder(_phi4mm())
    inputs = [SimpleNamespace(resource_step_info={"main": lengths}) for lengths in ((1024, 256), (1024,))]
    step = engine_step(encoder, graph_walk="prefill_vision", request_ids=[1, 2], inputs=inputs)
    reference = SubmoduleStep(
        segments=[Segment(1, "main", 1024), Segment(1, "main", 256), Segment(2, "main", 1024)],
        steps={"vision_attn": AttentionStep(causal=False)},
    )
    _assert_steps_equal(step, reference)


def test_phi4mm_vision_capture_check_reads_the_ceiling_in_effect():
    encoder = _encoder(_phi4mm())
    encoder.vision.reads["vision_attn"].max_segments_per_step = 2
    assert encoder.max_segments_per_step() == 2, "a YAML override of the ceiling should hold"


def test_phi4mm_process_prompt_hands_on_the_scaffolds_layout_and_the_prompt_length(monkeypatch):
    model = _phi4mm()
    span = Span("ids", "prefill_text", 7, 7, "text_inputs")
    scaffold = LabelLayout(spans=(span,), growth=(("decode_loop", 1),))
    out = ProcessPromptOutput({"text_inputs": [torch.arange(7)]}, {"kv_layout": {"kv_cache": {"main": scaffold}}})
    monkeypatch.setattr(MultimodalDecoderModel, "process_prompt", lambda self, *args, **kwargs: out)
    metadata = model.process_prompt("hello", ["text"], ["text"], {"text_inputs": []}).metadata
    assert "kv_layout" not in metadata, "a ported model returns its layout under state_layout"
    assert kv_layouts(model.declare_state(), metadata["state_layout"], {}) == {"kv_cache": {"main": scaffold}}
    assert metadata[MAX_POSITION_KWARG] == 7, "the request's LongRoPE table would no longer follow its prompt length"
