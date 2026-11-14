"""Step-Audio2 on the state contract builds the specs, steps and layouts it wrote by hand, field for field.

The hand-written specs and steps below are the reference, copied from the model as it stood before the port, so the
adapter can't pass by agreeing with itself.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "test/modular")

import pytest
import torch
import yaml
from test_startup_graph_validation import _step_audio2
from test_state_ports import _assert_equal_per_key, _assert_steps_equal, _bind, _published

from mstar.engine.resources import RaggedAttentionConfig, RaggedKeyPrefixAttentionSpec
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.kv.bounded import BoundedKVSpec
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.engine.state import Layout, Span
from mstar.engine.state.adapters import StateSetup, kv_layouts, sampling_configs
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.base import LabelLayout, enabled_resources, serving_resources
from mstar.model.components.token2wav.resources import token2wav_resources
from mstar.model.step_audio2.config import (
    AUDIO_ATTN,
    DECODE,
    DECODE_LOOP,
    DECODE_MIXED,
    DECODE_SPOKEN,
    DEFAULT_SAMPLING,
    LLM_ATTN,
    LLM_KV,
    LLM_POS,
    LLM_SAMPLER,
    PREFILL_AUDIO,
    PREFILL_TEXT,
)
from mstar.model.step_audio2.step_audio2_model import AUDIO, LLM, T2W_DEFAULT_SLOTS, T2W_KEYS, TOKEN2WAV
from mstar.model.step_audio2.submodules import LLMSubmodule
from mstar.model.step_audio2.token2wav import T2W_STREAMING
from mstar.model.submodule_base import ARNodeInputs, engine_step


def _model():
    return _step_audio2("step_audio2.yaml", {}, None, None)


def _reference(model) -> list:
    """The model's specs as it wrote them before the port."""
    llm, audio = model.config.llm, model.config.audio
    return [
        KVSpec(
            resource_key=LLM_KV, nodes={LLM},
            config=PagedKVConfig(
                num_layers=llm.num_hidden_layers, num_kv_heads=llm.num_key_value_heads, head_dim=llm.head_dim,
                max_seq_len=model.max_seq_len, num_qo_heads=llm.num_attention_heads,
            ),
        ),
        AttentionSpec(resource_key=LLM_ATTN, nodes={LLM}, config=AttentionConfig(kv_cache=LLM_KV)),
        PositionSpec(
            resource_key=LLM_POS, nodes={LLM}, config=PositionConfig(kv_cache=LLM_KV, rope_theta=llm.rope_theta),
        ),
        SamplerSpec(resource_key=LLM_SAMPLER, nodes={LLM}, vocab_size=llm.vocab_size, enable_repetition_penalty=True),
        RaggedKeyPrefixAttentionSpec(
            resource_key=AUDIO_ATTN, nodes={AUDIO},
            config=RaggedAttentionConfig(
                num_qo_heads=audio.n_audio_head, num_kv_heads=audio.n_audio_head, head_dim=audio.head_dim,
                max_segments_per_request=8,
            ),
        ),
        *token2wav_resources(T2W_KEYS, TOKEN2WAV, model._voice_seconds(), T2W_DEFAULT_SLOTS, T2W_STREAMING),
    ]


def _assert_specs_equal(specs: list, reference: list) -> None:
    """Per key, each bounded cache's retention compared as the policies it resolves to: the hand-written one is a
    lambda made afresh per build, which compares by identity."""
    def split(each):
        bounded = {
            spec.resource_key: (spec.nodes, {k: v for k, v in vars(spec.config).items() if k != "retention"})
            for spec in each if isinstance(spec, BoundedKVSpec)
        }
        return [spec for spec in each if not isinstance(spec, BoundedKVSpec)], bounded

    (rest, bounded), (want_rest, want_bounded) = split(specs), split(reference)
    _assert_equal_per_key(rest, want_rest)
    assert bounded == want_bounded


def test_the_published_config_is_found():
    assert [path.name for path in _published("StepAudio2Model")] == ["step_audio2.yaml"]


def test_specs_equal_the_hand_written_ones():
    model = _model()
    _assert_specs_equal(enabled_resources(model, {}, StateSetup()), _reference(model))


def test_specs_equal_the_hand_written_ones_on_the_published_config():
    (path,) = _published("StepAudio2Model")
    deployment = yaml.safe_load(path.read_text())
    model = _model()
    expected = _reference(model)
    apply_yaml_overrides(expected, deployment)
    _assert_specs_equal(serving_resources(model, deployment), expected)


def test_declarations_have_no_problems():
    assert state_declaration_problems(_model()) == []


def _llm() -> LLMSubmodule:
    model = _model()
    return _bind(LLMSubmodule(torch.nn.Identity(), model.config), model, node_name=LLM)


def _reference_step(rids, inputs, prompt_tokens) -> SubmoduleStep:
    return SubmoduleStep(
        segments=[Segment(rid, "main", inp.input_seq_len) for rid, inp in zip(rids, inputs, strict=True)],
        steps={
            LLM_KV: KVStep(),
            LLM_ATTN: AttentionStep(causal=True),
            LLM_POS: PositionStep(),
            LLM_SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=prompt_tokens),
        },
    )


@pytest.mark.parametrize("walk, lengths", [
    (PREFILL_TEXT, [5, 3]), (PREFILL_AUDIO, [9]), (DECODE, [1, 1]), (DECODE_SPOKEN, [1, 1]), (DECODE_MIXED, [1, 1]),
])
def test_llm_steps_equal_the_hand_written_ones(walk, lengths):
    rids = list(range(1, len(lengths) + 1))
    inputs = [ARNodeInputs(input_ids=torch.arange(n), input_seq_len=n) for n in lengths]
    prompt = {rid: inp.input_ids for rid, inp in zip(rids, inputs, strict=True)} if walk.startswith("prefill") else {}
    step = engine_step(_llm(), graph_walk=walk, request_ids=rids, inputs=inputs)
    _assert_steps_equal(step, _reference_step(rids, inputs, prompt))


def test_the_prompt_layout_converts_to_the_hand_written_one():
    spans = (Span("rows", PREFILL_TEXT, 12, 12, "text_inputs"),)
    layout = Layout(spans=spans, growth=((DECODE_LOOP, 1),))
    converted = kv_layouts(_model().declare_state(), {LLM_KV: {"main": layout}}, {})
    assert converted == {LLM_KV: {"main": LabelLayout(spans=spans, growth=((DECODE_LOOP, 1),))}}


@pytest.mark.parametrize("kwargs", [{}, {"temperature": 0.0, "top_p": 0.5, "repetition_penalty": 1.3}])
def test_request_settings_equal_the_hand_written_config(kwargs):
    model = _model()
    configs = sampling_configs(model.get_request_resource_configs([], [], kwargs), model.declare_state())
    assert configs == {LLM_SAMPLER: SamplingReqConfig(**{**DEFAULT_SAMPLING, **kwargs})}


def test_the_spoken_decode_walks_are_held_to_one_loop():
    # the kv label's growth names both decodes, which the declaration's one array keeps
    assert {DECODE, DECODE_SPOKEN} <= set(_model().kv_labels()[LLM_KV]["main"].growth[i][0] for i in range(2))
