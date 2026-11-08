"""Qwen3-Omni's specs and steps on the contract equal the ones it wrote by hand, and the Talker sends ``tts_eos`` once.

The hand-written specs and steps below are the reference, copied from the model as it stood before the port, so the
adapter can't pass by agreeing with itself. One edit: the Talker's position spec holds the four RoPE fields its layers
passed per call before. ``tts_eos`` conditions exactly one step after the Thinker finishes: its flag
applies at commit, so a pre-plan's second declaration can't count twice, and a step that never ran doesn't count at all.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_state_ports import _assert_steps_equal

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig, RaggedAttentionSpec
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import host
from mstar.engine.state.adapters import node_handles, node_resources, sampling_configs, state_steps
from mstar.engine.state.analysis import state_declaration_problems
from mstar.engine.state.host import HostFields
from mstar.model.components.aut_encoder import AuTEncoderConfig
from mstar.model.components.aut_submodule import AuTEncoderSubmodule
from mstar.model.components.code_predictor_kernels import DepthKV
from mstar.model.components.norm import RMSNorm
from mstar.model.qwen3_omni.components.attention import _norm_rotate_attend
from mstar.model.qwen3_omni.config import (
    AUDIO_ATTN,
    AUDIO_TOKENS,
    CODE_PRED_SAMPLER,
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    TALKER_SAMPLER,
    TALKER_TEXT,
    THINKER_ATTN,
    THINKER_KV,
    THINKER_POS,
    THINKER_SAMPLER,
    VISION_ATTN,
    VISION_TOKENS,
    Qwen3OmniModelConfig,
    VisionEncoderConfig,
)
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
from mstar.model.qwen3_omni.submodules import (
    AudioEncoderSubmodule,
    TalkerSubmodule,
    ThinkerSubmodule,
    VisionEncoderSubmodule,
)
from mstar.model.submodule_base import ARNodeInputs, NodeInputs

AUDIO = {
    "num_mel_bins": 16, "d_model": 64, "encoder_attention_heads": 4, "encoder_layers": 2, "encoder_ffn_dim": 128,
    "downsample_hidden_size": 8, "output_dim": 32, "max_source_positions": 64, "n_window": 8, "n_window_infer": 16,
    "conv_chunksize": 8,
}
VISION = {
    "depth": 2, "hidden_size": 32, "num_heads": 4, "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2,
    "temporal_patch_size": 2, "out_hidden_size": 16, "intermediate_size": 64, "num_position_embeddings": 16,
    "deepstack_visual_indexes": [0],
}


def _model() -> Qwen3OmniModel:
    model = object.__new__(Qwen3OmniModel)
    model.config = Qwen3OmniModelConfig(
        audio_encoder=AuTEncoderConfig.from_hf(AUDIO), vision=VisionEncoderConfig.from_dict(VISION),
    )
    return model


def _reference(c: Qwen3OmniModelConfig) -> list:
    tt, kt = c.thinker_text, c.talker_text
    return [
        KVSpec(resource_key=THINKER_KV, nodes={"Thinker"}, config=PagedKVConfig(
            num_layers=tt.num_hidden_layers, num_kv_heads=tt.num_key_value_heads, head_dim=c.thinker_head_dim,
            max_seq_len=tt.max_position_embeddings, num_qo_heads=tt.num_attention_heads,
        )),
        KVSpec(resource_key=TALKER_KV, nodes={"Talker"}, config=PagedKVConfig(
            num_layers=kt.num_hidden_layers, num_kv_heads=kt.num_key_value_heads, head_dim=c.talker_head_dim,
            max_seq_len=tt.max_position_embeddings, num_qo_heads=kt.num_attention_heads,
        )),
        AttentionSpec(resource_key=THINKER_ATTN, nodes={"Thinker"}, config=AttentionConfig(kv_cache=THINKER_KV)),
        AttentionSpec(resource_key=TALKER_ATTN, nodes={"Talker"}, config=AttentionConfig(kv_cache=TALKER_KV)),
        PositionSpec(resource_key=THINKER_POS, nodes={"Thinker"}, config=PositionConfig(kv_cache=THINKER_KV)),
        # the four fields its layers passed per call, which the contract's `Rope` holds
        PositionSpec(resource_key=TALKER_POS, nodes={"Talker"}, config=replace(
            PositionConfig(kv_cache=TALKER_KV), rope_theta=1_000_000.0, low_freq_factor=1.0, high_freq_factor=1.0,
            old_context_len=8192,
        )),
        SamplerSpec(resource_key=THINKER_SAMPLER, nodes={"Thinker"}, vocab_size=tt.vocab_size,
                    enable_repetion_penalty=True),
        SamplerSpec(resource_key=TALKER_SAMPLER, nodes={"Talker"}, vocab_size=kt.vocab_size,
                    enable_repetion_penalty=True),
        SamplerSpec(resource_key=CODE_PRED_SAMPLER, nodes={"Talker"}, vocab_size=c.code_predictor.vocab_size,
                    enable_repetion_penalty=False),
        RaggedAttentionSpec(resource_key=AUDIO_ATTN, nodes={"audio_encoder"}, config=RaggedAttentionConfig(
            num_qo_heads=c.audio_encoder.encoder_attention_heads, num_kv_heads=c.audio_encoder.encoder_attention_heads,
            head_dim=c.audio_encoder.head_dim,
            max_segments_per_request=AuTEncoderSubmodule.max_windows_per_replay(c.audio_encoder),
        )),
        RaggedAttentionSpec(resource_key=VISION_ATTN, nodes={"vision_encoder"}, config=RaggedAttentionConfig(
            num_qo_heads=c.vision.num_heads, num_kv_heads=c.vision.num_heads, head_dim=c.vision.head_dim,
            max_segments_per_step=VisionEncoderSubmodule.BLOCK_LOOP_PATCH_BUCKETS[-1] // 256,
        )),
    ]


def _assert_specs_equal(specs, reference) -> None:
    assert {s.resource_key: s for s in specs} == {s.resource_key: s for s in reference}


def test_specs_equal_the_hand_written_ones():
    model = _model()
    _assert_specs_equal(node_resources(model), _reference(model.config))


def test_specs_equal_the_hand_written_ones_after_yaml_overrides():
    model = _model()
    overrides = {"resources": {
        THINKER_KV: {"max_seq_len": 1024}, VISION_ATTN: {"max_segments_per_step": 9},
        THINKER_SAMPLER: {"enable_logprobs": True},
    }}
    specs, reference = node_resources(model), _reference(model.config)
    apply_yaml_overrides(specs, overrides)
    apply_yaml_overrides(reference, overrides)
    _assert_specs_equal(specs, reference)


def test_the_port_writes_no_spec_by_hand():
    assert _model().get_node_resources() == []


def test_the_value_of_host_fields_is_on_a_node_with_no_prefix_keys():
    labels = _model().kv_labels()
    assert all(label.keys is None for label in labels[TALKER_KV].values()), "the Talker's host field has no snapshot"


def test_the_code_predictor_declares_its_fifteen_draws():
    sampler = next(d for d in _model().declare_state() if d.name == CODE_PRED_SAMPLER)
    assert sampler.draws == 15 and not sampler.penalty


@pytest.mark.parametrize("kwargs, thinker", [
    ({}, {"temperature": 0.7, "top_p": 0.9, "ignore_eos": False, "repetition_penalty": 1.0}),
    ({"thinker_temperature": 0.1, "ignore_eos": True}, {"temperature": 0.1, "top_p": 0.9, "ignore_eos": True,
                                                        "repetition_penalty": 1.0}),
])
def test_request_settings_equal_the_hand_written_ones(kwargs, thinker):
    model = _model()
    settings = sampling_configs(model.get_request_resource_configs({}, kwargs), model.declare_state())
    assert settings == {
        THINKER_SAMPLER: SamplingReqConfig(**thinker),
        TALKER_SAMPLER: SamplingReqConfig(temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05),
        CODE_PRED_SAMPLER: SamplingReqConfig(temperature=1.0, top_k=50, top_p=0.8),
    }


# The steps


class _StubResource:
    """Stands in for any built resource a handle reads at bind."""

    def layer_view(self, i):
        return torch.empty(0)

    def opens_label(self, rid, label):
        return True

    def run(self, q, k=None, v=None, label=None):
        return q


def _handles(model, node, fields=None):
    resources = {spec.resource_key: _StubResource() for spec in node_resources(model)}
    return node_handles(model, node, resources, host_fields=fields or HostFields())


def _thinker(model=None) -> ThinkerSubmodule:
    model = model or _model()
    sub = ThinkerSubmodule(SimpleNamespace(), model.config)
    sub.bind_node_state(_handles(model, "Thinker"))
    return sub


def _thinker_reference(rids, spans, *, tokens=None, kept=None, advance=None) -> SubmoduleStep:
    return SubmoduleStep(
        segments=[Segment(request_id=rid, label="main", span=span) for rid, span in zip(rids, spans, strict=True)],
        steps={
            THINKER_KV: KVStep(),
            THINKER_ATTN: AttentionStep(causal=True),
            THINKER_SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=tokens or {}, kept_rids=kept),
            THINKER_POS: PositionStep(advance=advance),
        },
    )


def test_a_thinker_decode_step_equals_the_hand_written_one():
    sub = _thinker()
    rows = [ARNodeInputs(input_seq_len=1), ARNodeInputs(input_seq_len=1)]
    step = state_steps(sub.declare_step("thinker_decode", [1, 2], rows), sub.state_handles)
    _assert_steps_equal(step, _thinker_reference([1, 2], [1, 1]))


def test_a_text_prefill_step_equals_the_hand_written_one():
    sub = _thinker()
    ids = torch.arange(4)
    rows = [ARNodeInputs(input_seq_len=4, input_ids=ids)]
    info = {1: SimpleNamespace(step_metadata={"is_last_prefill": True})}
    declared = sub.declare_step("prefill_text", [1], rows, per_request_info=info)
    step = state_steps(declared, sub.state_handles)
    _assert_steps_equal(step, _thinker_reference([1], [4], tokens={1: ids}, kept=frozenset({1})))


def test_a_vision_prefill_step_equals_the_hand_written_one_and_keeps_its_capture_key():
    sub = _thinker()
    rows = [ARNodeInputs(input_seq_len=6, tensor_inputs={"mrope_pos_advance": 9, "text_token_ids": torch.arange(2)})]
    step = state_steps(sub.declare_step("prefill_vision", [1], rows, per_request_info={}), sub.state_handles)
    reference = replace(
        _thinker_reference([1], [6], tokens={1: torch.arange(2)}, kept=frozenset(), advance=(9,)),
        cg_key_info=sub.DEEPSTACK_KEY,
    )
    _assert_steps_equal(step, reference)


def test_a_mixed_step_equals_the_hand_written_one_row_by_row():
    sub = _thinker()
    rows = [
        ARNodeInputs(input_seq_len=1, graph_walk="thinker_decode"),
        ARNodeInputs(input_seq_len=3, graph_walk="prefill_vision", tensor_inputs={"mrope_pos_advance": 7}),
        ARNodeInputs(input_seq_len=2, graph_walk="prefill_text", input_ids=torch.arange(2)),
    ]
    step = state_steps(sub.declare_step("thinker_mixed", [1, 2, 3], rows, per_request_info={}), sub.state_handles)
    reference = replace(
        _thinker_reference([1, 2, 3], [1, 3, 2], tokens={3: torch.arange(2)}, kept=frozenset({1}), advance=(1, 7, 2)),
        cg_key_info=sub.DEEPSTACK_KEY,
    )
    _assert_steps_equal(step, reference)


def _talker(model=None, fields=None) -> TalkerSubmodule:
    model = model or _model()
    body = SimpleNamespace(model=SimpleNamespace(codec_embedding=None))
    sub = TalkerSubmodule(body, None, model.config)
    sub.bind_node_state(_handles(model, "Talker", fields))
    sub.get_device = lambda: torch.device("cpu")
    sub._tts_eos_raw, sub._tts_pad_raw = torch.full((1, 2), 7.0), torch.full((1, 2), 3.0)
    return sub


def _talker_reference(rids, spans, *, samples: bool) -> SubmoduleStep:
    steps = {TALKER_KV: KVStep(), TALKER_ATTN: AttentionStep(causal=True), TALKER_POS: PositionStep()}
    if samples:
        steps[TALKER_SAMPLER] = SamplerStep(apply_penalty=True)
        steps[CODE_PRED_SAMPLER] = SamplerStep(apply_penalty=False)
    return SubmoduleStep(
        segments=[Segment(request_id=rid, label="main", span=span) for rid, span in zip(rids, spans, strict=True)],
        steps=steps,
    )


@pytest.mark.parametrize("walk, samples", [("talker_prefill", False), ("talker_last_prefill", True),
                                           ("talker_decode", True)])
def test_a_talker_step_equals_the_hand_written_one(walk, samples):
    sub = _talker()
    rows = [ARNodeInputs(input_seq_len=2), ARNodeInputs(input_seq_len=2)]
    step = state_steps(sub.declare_step(walk, [1, 2], rows), sub.state_handles)
    _assert_steps_equal(step, _talker_reference([1, 2], [2, 2], samples=samples))


def test_a_talker_step_writing_its_flag_holds_no_resource_step_for_it():
    sub = _talker()
    sub.state_handles[TALKER_TEXT].host_fields.open(1)
    rows = [ARNodeInputs(input_seq_len=1, tensor_inputs={"text_raw": sub._tts_eos_raw})]
    step = state_steps(sub.declare_step("talker_decode", [1], rows), sub.state_handles)
    _assert_steps_equal(step, _talker_reference([1], [1], samples=True))
    assert TALKER_TEXT in step.state and TALKER_TEXT not in step.steps


def test_an_audio_encoder_step_equals_the_hand_written_one():
    model = _model()
    encoder = SimpleNamespace(config=model.config.audio_encoder)
    sub = AudioEncoderSubmodule.__new__(AudioEncoderSubmodule)
    torch.nn.Module.__init__(sub)
    handles = _handles(model, "audio_encoder")
    sub.__dict__.update(encoder=encoder, attn_key=None, array=AUDIO_TOKENS, state_handles=handles)
    tokens = 2 * model.config.audio_encoder.window_tokens + 1
    declared = sub.declare_step("prefill_audio", [1], [NodeInputs(input_seq_len=tokens)])
    step = state_steps(declared, sub.state_handles)
    windows = model.config.audio_encoder.window_lengths(tokens)
    reference = SubmoduleStep(
        segments=[Segment(1, "main", w) for w in windows], steps={AUDIO_ATTN: AttentionStep(causal=False)},
    )
    assert step.steps.keys() == reference.steps.keys() and step.segments == reference.segments
    assert list(step.steps[AUDIO_ATTN].segments) == reference.segments and not step.steps[AUDIO_ATTN].causal


def test_a_vision_encoder_step_equals_the_hand_written_one():
    model = _model()
    sub = VisionEncoderSubmodule(None, model.config)
    sub.bind_node_state(_handles(model, "vision_encoder"))
    row = NodeInputs(input_seq_len=8, resource_step_info={"main": (4, 4)})
    step = state_steps(sub.declare_step("prefill_vision", [1], [row]), sub.state_handles)
    assert list(step.steps[VISION_ATTN].segments) == [Segment(1, "main", 4)] * 2
    assert not step.steps[VISION_ATTN].causal


def test_a_leased_vision_step_declares_none_as_it_did():
    sub = VisionEncoderSubmodule(None, _model().config)
    assert sub.declare_step("prefill_vision", [1], [], piecewise_leases={"vision_block_loop": object()}) is None


def test_the_vision_capture_check_reads_the_ceiling_in_effect():
    model = _model()
    sub = VisionEncoderSubmodule(None, model.config)
    resources = {spec.resource_key: _StubResource() for spec in node_resources(model)}
    resources[VISION_ATTN].max_segments_per_step = 2
    sub.bind_node_state(node_handles(model, "vision_encoder", resources, host_fields=HostFields()))
    frames = [SimpleNamespace(input_seq_len=4 * n, resource_step_info={"main": (4,) * n}) for n in (2, 3)]
    assert sub._fits_capture(frames[:1]) and not sub._fits_capture(frames[1:]), "a YAML override must still hold"


def test_each_region_declares_its_array():
    model = _model()
    sub = VisionEncoderSubmodule(None, model.config)
    region = sub.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.bfloat16)["vision_block_loop"]
    step = region.declare_step([1], [4, 4])
    assert step.state.keys() == {VISION_TOKENS} and step.segments == [Segment(1, "main", 4)] * 2


# tts_eos


def _decode_input(sub, rid, states=None) -> ARNodeInputs:
    inputs = {"talker_input_embeds": [torch.zeros(1, 2)]}
    if states is not None:
        inputs["thinker_states"] = [states]
    return sub.prepare_inputs("talker_decode", SimpleNamespace(rid_handle=rid), inputs)


def _run_step(sub, rows: dict[int, ARNodeInputs], padded=()) -> None:
    rids = [*rows, *padded]
    inputs = [*rows.values(), *(ARNodeInputs(input_seq_len=1, tensor_inputs={"text_raw": torch.zeros(1, 2)})
                                for _ in padded)]
    step = state_steps(sub.declare_step("talker_decode", rids, inputs), sub.state_handles)
    ctx = StepContext(request_ids=list(rows), graph_walk="talker_decode", slot=0, capture=False)
    ctx.set_padded_rids(rids)
    step.set_ctx(ctx)
    for handle in (h for h in step.state_handles if not hasattr(h, "sampler")):
        handle.begin_step(step)
    step.commit_calls()
    for handle in (h for h in step.state_handles if not hasattr(h, "sampler")):
        handle.end_step(step)


@pytest.fixture
def talker():
    fields = HostFields()
    sub = _talker(fields=fields)
    fields.open(1)
    fields.open(2)
    return sub


def _sent(sub, rid) -> bool:
    return host(sub.text, rid, "main", 0)["eos_sent"]


def test_a_request_gets_exactly_one_tts_eos_and_then_pad(talker):
    seen = []
    for _ in range(4):
        row = _decode_input(talker, 1)
        seen.append(row.tensor_inputs["text_raw"] is talker._tts_eos_raw)
        _run_step(talker, {1: row})
    assert seen == [True, False, False, False]
    assert _decode_input(talker, 1).tensor_inputs["text_raw"] is talker._tts_pad_raw


def test_a_step_declared_twice_for_one_step_counts_once(talker):
    """A pre-plan declares the step, and exec declares it again."""
    row = _decode_input(talker, 1)
    state_steps(talker.declare_step("talker_decode", [1], [row]), talker.state_handles)
    assert not _sent(talker, 1), "the pre-plan's declaration must not count"
    _run_step(talker, {1: row})
    assert _sent(talker, 1)
    assert _decode_input(talker, 1).tensor_inputs["text_raw"] is talker._tts_pad_raw


def test_a_step_that_never_committed_sends_tts_eos_again(talker):
    """A refused admit prepares the step again, and that step is the one the Talker actually runs."""
    first = _decode_input(talker, 1)
    state_steps(talker.declare_step("talker_decode", [1], [first]), talker.state_handles)
    assert _decode_input(talker, 1).tensor_inputs["text_raw"] is talker._tts_eos_raw


def test_a_row_still_fed_by_the_thinker_writes_nothing(talker):
    states = torch.ones(1, 2)
    row = _decode_input(talker, 1, states)
    assert row.tensor_inputs["text_raw"] is states
    _run_step(talker, {1: row})
    assert not _sent(talker, 1), "tts_eos comes after the Thinker's last state"


def test_each_row_tracks_its_own_flag(talker):
    _run_step(talker, {1: _decode_input(talker, 1), 2: _decode_input(talker, 2, torch.ones(1, 2))})
    assert _sent(talker, 1) and not _sent(talker, 2)


def test_padding_rows_never_write(talker):
    inputs = [_decode_input(talker, 1), *(ARNodeInputs(input_seq_len=1, tensor_inputs={"text_raw": torch.zeros(1, 2)})
                                          for _ in range(2))]
    declared = talker.declare_step("talker_decode", [1, -1, -2], inputs)
    assert list(declared.state[TALKER_TEXT].host) == [(1, "main", 0)]


def test_the_store_is_empty_after_a_removal_and_a_recycled_handle_starts_clean(talker):
    _run_step(talker, {1: _decode_input(talker, 1)})
    fields = talker.text.host_fields
    fields.remove_request(1)
    fields.remove_request(2)
    assert not fields._fields, "a leaked handle would attach to the next request that gets it"
    fields.open(1)
    assert _decode_input(talker, 1).tensor_inputs["text_raw"] is talker._tts_eos_raw


# The fused write's reference


def test_the_reference_equals_the_unfused_norm_and_rotation():
    """What a format the kernel can't write runs, written out as the fused op's comment states it."""
    q, k, v = torch.randn(3, 4, 8), torch.randn(3, 2, 8), torch.randn(3, 2, 8)
    norm = RMSNorm(8)
    torch.nn.init.normal_(norm.weight)
    cos, sin = torch.randn(5, 8), torch.randn(5, 8)
    calls = []

    def read(layer, q):
        calls.append(("read", layer, q.clone()))
        return q * 2

    kv = SimpleNamespace(add=lambda layer, k, v: calls.append(("add", layer, k, v)), read=read)

    def rotate(x):
        half = x.shape[-1] // 2
        n = norm(x.reshape(-1, 8)).view(x.shape)
        return n * cos[:3, None] + torch.cat((-n[..., half:], n[..., :half]), dim=-1) * sin[:3, None]

    want_q, want_k = rotate(q), rotate(k)
    k_before, v_before = k.clone(), v
    out = _norm_rotate_attend(
        kv, 2, q, k, v, norm.weight, norm.weight, cos, sin, norm.variance_epsilon, norm.variance_epsilon,
    )
    assert [call[0] for call in calls] == ["add", "read"], "the read comes after the add"
    (_, layer, added_k, added_v), (_, read_layer, read_q) = calls
    assert torch.equal(q, want_q), "the fused op mutates q, so its reference does too"
    assert torch.equal(added_k, want_k) and added_v is v_before and layer == 2
    assert torch.equal(k, k_before), "the fused op leaves k as it was"
    assert read_layer == 2 and torch.equal(read_q, want_q), "the read takes the rotated q"
    assert torch.equal(out, want_q * 2), "the reference returns what the read returns"


# Scratch at load


def test_the_depth_scratch_can_be_made_at_load_and_a_step_reuses_it():
    depth = DepthKV((2, 4, 2, 8, 2, 16))
    depth.make(torch.bfloat16, "cpu")
    cache, partials = depth.kv_cache, depth.partials
    assert cache.dtype is torch.bfloat16 and cache.shape == (2, 4, 2, 8, 2, 16)
    assert depth.scratch(torch.zeros(1, dtype=torch.float32)) is partials and depth.kv_cache is cache


def test_a_depth_scratch_nothing_made_at_load_still_comes_at_the_first_step():
    depth = DepthKV((2, 4, 2, 8, 2, 16))
    depth.scratch(torch.zeros(1, dtype=torch.float16))
    assert depth.kv_cache.dtype is torch.float16, "Qwen3-TTS still makes its buffers at the first step"


def test_the_declarations_have_no_problems_on_the_real_walks(monkeypatch):
    model = _model()
    monkeypatch.setattr(model, "get_graph_walk_graphs", lambda: {
        "talker_decode": SimpleNamespace(get_nodes=lambda: ["Talker"]),
        "thinker_decode": SimpleNamespace(get_nodes=lambda: ["Thinker"]),
        "prefill_audio": SimpleNamespace(get_nodes=lambda: ["audio_encoder"]),
        "prefill_vision": SimpleNamespace(get_nodes=lambda: ["vision_encoder"]),
    })
    monkeypatch.setattr(model, "get_combined_graph_walks", dict)
    assert state_declaration_problems(model) == []
