"""MiniCPM-o's specs and steps on the contract equal the ones it wrote by hand.

The hand-written specs and steps below are the reference, copied from the model as it stood before the port, so the
adapter can't pass by agreeing with itself. One edit: the vision read's spec lists its label, as every declared
within-step read's does for a sequence shard. The TTS's position spec is checked by its effect instead: its layers
passed Llama 3.1's factors on every call, which picked FlashInfer's llama31 kernel, and the declared `Rope` now holds
them.
Token2Wav keeps its hand-written specs.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_minicpm_o import a_model
from test_state_omni_port import _StubResource
from test_state_ports import _assert_steps_equal

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedBlockCausalAttentionSpec,
    RaggedCrossAttentionSpec,
    RaggedCrossAttentionStep,
)
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.engine.state.adapters import StateSetup, node_handles, node_resources, state_steps
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.components.distributed import ParallelAttention
from mstar.model.minicpm_o.config import (
    AUDIO_ATTN,
    LLM_ATTN,
    LLM_KV,
    LLM_SAMPLER,
    PATCHES,
    QUERIES,
    RESAMPLER_ATTN,
    T2W_KV,
    T2W_STATE,
    TTS_ATTN,
    TTS_DECODE,
    TTS_KV,
    TTS_PREFILL,
    TTS_SAMPLER,
    VISION_ATTN,
)
from mstar.model.minicpm_o.minicpm_o_model import AUDIO, LLM, TTS, TTS_SAMPLING, VISION
from mstar.model.minicpm_o.submodules import AudioEncoderSubmodule, LLMSubmodule, TTSSubmodule, VisionEncoderSubmodule
from mstar.model.submodule_base import ARNodeInputs, NodeInputs


def _reference(model) -> list:
    llm, vision, resampler, audio, tts = (
        model.config.llm, model.config.vision, model.config.resampler, model.config.audio, model.tts_config,
    )
    return [
        KVSpec(resource_key=LLM_KV, nodes={LLM}, config=PagedKVConfig(
            num_layers=llm.num_hidden_layers, num_kv_heads=llm.num_key_value_heads, head_dim=llm.head_dim,
            max_seq_len=model.max_seq_len, num_qo_heads=llm.num_attention_heads,
        )),
        AttentionSpec(resource_key=LLM_ATTN, nodes={LLM}, config=AttentionConfig(kv_cache=LLM_KV)),
        PositionSpec(resource_key="llm_pos", nodes={LLM},
                     config=PositionConfig(kv_cache=LLM_KV, rope_theta=llm.rope_theta)),
        SamplerSpec(resource_key=LLM_SAMPLER, nodes={LLM}, vocab_size=llm.vocab_size, enable_repetition_penalty=True,
                    enable_min_p=True),
        RaggedAttentionSpec(resource_key=VISION_ATTN, nodes={VISION}, labels={PATCHES}, config=RaggedAttentionConfig(
            num_qo_heads=vision.num_attention_heads, num_kv_heads=vision.num_attention_heads, head_dim=vision.head_dim,
        )),
        RaggedCrossAttentionSpec(resource_key=RESAMPLER_ATTN, nodes={VISION}, config=RaggedAttentionConfig(
            num_qo_heads=resampler.num_heads, num_kv_heads=resampler.num_heads, head_dim=resampler.head_dim,
        )),
        RaggedBlockCausalAttentionSpec(resource_key=AUDIO_ATTN, nodes={AUDIO}, block_size=audio.chunk_frames,
                                       config=RaggedAttentionConfig(
            num_qo_heads=audio.encoder_attention_heads, num_kv_heads=audio.encoder_attention_heads,
            head_dim=audio.head_dim, max_segments_per_request=4,
        )),
        KVSpec(resource_key=TTS_KV, nodes={TTS}, config=PagedKVConfig(
            num_layers=tts.num_hidden_layers, num_kv_heads=tts.num_key_value_heads, head_dim=tts.head_dim,
            max_seq_len=tts.max_position_embeddings, num_qo_heads=tts.num_attention_heads,
        )),
        AttentionSpec(resource_key=TTS_ATTN, nodes={TTS}, config=AttentionConfig(kv_cache=TTS_KV)),
        SamplerSpec(resource_key=TTS_SAMPLER, nodes={TTS}, vocab_size=tts.num_audio_tokens,
                    enable_repetition_penalty=False, max_repetition_window=TTS_SAMPLING.penalty_window,
                    min_tokens_stop_ids=(tts.eos_code,), enable_top_p_first=True),
    ]


def test_specs_equal_the_hand_written_ones(tmp_path):
    model = a_model(tmp_path)
    skipped = {"tts_pos", *(spec.resource_key for spec in model.get_node_resources())}
    declared = {spec.resource_key: spec for spec in node_resources(model) if spec.resource_key not in skipped}
    assert declared == {spec.resource_key: spec for spec in _reference(model)}


def test_the_port_writes_only_token2wav_by_hand(tmp_path):
    assert {spec.resource_key for spec in a_model(tmp_path).get_node_resources()} == {T2W_STATE, *T2W_KV.values()}


def test_the_declarations_have_no_problems(tmp_path):
    assert state_declaration_problems(a_model(tmp_path)) == []


def test_the_tts_rotates_through_the_kernel_its_layers_picked(tmp_path, monkeypatch):
    """The arguments the RoPE kernel gets from the declared positions, with nothing passed per call, are the ones it got
    from the hand-written spec and the factors each TTS layer passed."""
    model = a_model(tmp_path)
    tts = model.tts_config
    (declared,) = [spec for spec in node_resources(model) if spec.resource_key == "tts_pos"]
    layer = ParallelAttention(
        hidden_size=tts.hidden_size, num_heads=tts.num_attention_heads, num_kv_heads=tts.num_key_value_heads,
        head_dim=tts.head_dim, rope_theta=tts.rope_theta,
    )
    calls = []

    def record(q, k, ids, table, *args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(torch.ops.mstar, "rope_apply_qk_inplace", record)
    q, k = torch.zeros(3, 2, tts.head_dim, dtype=torch.bfloat16), torch.zeros(3, 1, tts.head_dim, dtype=torch.bfloat16)
    ids = torch.arange(3)
    RopeManager(PositionConfig(kv_cache=TTS_KV, rope_theta=tts.rope_theta), torch.device("cpu")).apply_qk(
        q, k, pos_ids=ids, rope_theta=layer.rope_theta, rope_scale=layer.rope_scale,
        low_freq_factor=layer.rope_low_freq_factor, high_freq_factor=layer.rope_high_freq_factor,
        old_context_len=layer.rope_old_context_len,
    )
    RopeManager(declared.config, torch.device("cpu")).apply_qk(q, k, pos_ids=ids, rope_dtype=q.dtype)
    hand, contract = calls
    assert hand[1], "the layers' factors pick FlashInfer's llama31 kernel"
    assert contract == hand, "the TTS would rotate through another kernel than its layers picked"


def _handles(model, node):
    resources = {spec.resource_key: _StubResource() for spec in node_resources(model)}
    return node_handles(model, node, resources, setup=StateSetup())


def test_an_llm_step_equals_the_hand_written_one(tmp_path):
    model = a_model(tmp_path)
    llm = LLMSubmodule(torch.nn.Identity(), model.config)
    llm.bind_node_state(_handles(model, LLM))
    rows = [ARNodeInputs(input_seq_len=5), ARNodeInputs(input_seq_len=1)]
    step = state_steps(llm.declare_step("decode_mixed", [1, 2], rows), llm.state_handles)
    _assert_steps_equal(step, SubmoduleStep(
        segments=[Segment(1, "main", 5), Segment(2, "main", 1)],
        steps={LLM_KV: KVStep(), LLM_ATTN: AttentionStep(causal=True), "llm_pos": PositionStep(),
               LLM_SAMPLER: SamplerStep(apply_penalty=True)},
    ))


@pytest.mark.parametrize("walk, filters", [(TTS_PREFILL, False), (TTS_DECODE, True)])
def test_a_tts_step_equals_the_hand_written_one(tmp_path, walk, filters):
    model = a_model(tmp_path)
    tts = TTSSubmodule(SimpleNamespace(config=model.tts_config), model.config, TTS_SAMPLING)
    tts.bind_node_state(_handles(model, TTS))
    step = state_steps(tts.declare_step(walk, [1], [ARNodeInputs(input_seq_len=3)]), tts.state_handles)
    _assert_steps_equal(step, SubmoduleStep(
        segments=[Segment(1, "main", 3)],
        steps={TTS_KV: KVStep(), TTS_ATTN: AttentionStep(causal=True), "tts_pos": PositionStep(),
               TTS_SAMPLER: SamplerStep(apply_penalty=False, apply_filters=filters)},
    ))


def test_a_vision_step_equals_the_hand_written_one(tmp_path):
    model = a_model(tmp_path)
    vision = VisionEncoderSubmodule(torch.nn.Identity(), model.config)
    vision.bind_node_state(_handles(model, VISION))
    step = state_steps(
        vision.declare_step("prefill_image", [1], [NodeInputs(input_seq_len=13, kwargs={"seq_lengths": (9, 4)})]),
        vision.state_handles,
    )
    q = model.config.resampler.num_queries
    patches = (Segment(1, PATCHES, 9), Segment(1, PATCHES, 4))
    queries = (Segment(1, QUERIES, q), Segment(1, QUERIES, q))
    assert step.state, "a ported walk declares its state, not today's step classes"
    assert step.steps.keys() == {VISION_ATTN, RESAMPLER_ATTN}
    assert step.steps[VISION_ATTN] == AttentionStep(segments=patches, causal=False)
    cross = step.steps[RESAMPLER_ATTN]
    assert isinstance(cross, RaggedCrossAttentionStep) and cross.pairs == ((QUERIES, PATCHES),), cross
    assert sorted(cross.segments, key=lambda s: s.label) == sorted(patches + queries, key=lambda s: s.label), cross


def test_an_audio_step_equals_the_hand_written_one(tmp_path):
    model = a_model(tmp_path)
    audio = AudioEncoderSubmodule(torch.nn.Identity(), model.config)
    audio.bind_node_state(_handles(model, AUDIO))
    inputs = [NodeInputs(input_seq_len=150, kwargs={"frames": [100, 50]})]
    step = state_steps(audio.declare_step("prefill_audio", [1], inputs), audio.state_handles)
    segments = [Segment(1, "main", 100), Segment(1, "main", 50)]
    assert step.state, "a ported walk declares its state, not today's step classes"
    assert step.steps.keys() == {AUDIO_ATTN} and step.segments == segments
    assert list(step.steps[AUDIO_ATTN].segments) == segments and not step.steps[AUDIO_ATTN].causal
