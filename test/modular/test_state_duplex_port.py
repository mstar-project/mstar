"""Nemotron Duplex plans on the contract what it planned by hand.

The specs, steps and layouts below are the reference, copied from the model as it stood before the port, so the
adapter can't pass by agreeing with itself. Each difference is asserted where it shows, with its reason: the nano's
conv stored in the compute dtype, the talker's positions with no rotary dim, the talker's combined label, the
``talker_prev`` Value for the codes the talker kept in request state, and the encoder's conv rows at each stream's new
frames, its all-steady step and its regions' padding rows.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import yaml
from test_fastconformer import CONV, FRONT, KV, LAYERS, STEP, STREAM, _Encoder, _encoder_regions
from test_nemotron_duplex_codec import _Talker
from test_nemotron_duplex_submodules import _make_nano
from test_nemotron_duplex_talker_batch import TINY
from test_nemotron_duplex_talker_pool import fwd as _fwd
from test_nemotron_duplex_talker_pool import make_sub as _make_talker_sub
from test_state_ports import _assert_equal_per_key, _assert_steps_equal, _bind, _published, _stored, _StubResource

from mstar.engine.node_dtype import resolve_model_dtype
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    KVSpec,
    KVStep,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    PositionStep,
    SamplerSpec,
    SamplerStep,
    Segment,
    SubmoduleStep,
)
from mstar.engine.resources.kv.bounded import BoundedKVConfig, BoundedKVSpec, BoundedKVStep, StreamPosition
from mstar.engine.resources.kv.bounded.config import SinkWindow
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
)
from mstar.engine.resources.recurrent import Mamba2Geometry, RecurrentStateConfig, RecurrentStateSpec, RecurrentStep
from mstar.engine.resources.recurrent.config import RecurrentBlockConfig
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import StepContext
from mstar.engine.state import ArrayStep
from mstar.engine.state.adapters import kv_layouts, node_handles, state_steps
from mstar.engine.state.analysis import declaration_digests, state_declaration_problems
from mstar.model.base import LabelLayout, Span, parse_autocast_dtype, serving_resources
from mstar.model.components.fastconformer import (
    ConvModule,
    FastConformerConfig,
    FastConformerEncoder,
    RelPositionAttention,
)
from mstar.model.nemotron_duplex.components.eartts_talker import _Attn
from mstar.model.nemotron_duplex.components.nemotron_h import Mamba2Mixer, NemotronHAttention, NemotronHLLM
from mstar.model.nemotron_duplex.config import (
    CODEC_STATE,
    ENCODER_FRONT,
    ENCODER_KV,
    ENCODER_STATE,
    MAMBA,
    MAMBA_STATE,
    NANO_ATTN,
    NANO_KV,
    NANO_SAMPLER,
    TALKER_ATTN,
    TALKER_KV,
    TALKER_PREV,
    NanoConfig,
    NemotronDuplexConfig,
)
from mstar.model.nemotron_duplex.nemotron_duplex_model import NemotronDuplexModel
from mstar.model.submodule_base import engine_step

# `position_key("talker_kv")`, the key main declared the talker's positions under
TALKER_POS = "talker_pos"


def _model(**model_kwargs) -> NemotronDuplexModel:
    return NemotronDuplexModel("unused", **model_kwargs)


# Specs


def _reference_specs(model: NemotronDuplexModel) -> list:
    """`NemotronDuplexModel.get_node_resources` and `stream_node_resources` before the port, and the Value the talker's
    previous codes are now."""
    nano, eartts, enc = model.config.nano, model.config.eartts, model.config.stt.encoder
    hop, sessions, f32 = model.config.stt.hop_length, model.DEFAULT_SESSIONS, torch.float32
    codec = model.config.codec
    codec_blocks = {f"stage{i}": RecurrentBlockConfig(shape=(codec.num_blocks, ch, codec.kernel_size - 1), dtype=None)
                    for i, ch in enumerate(reversed(codec.channels))}
    codec_blocks["overlap"] = RecurrentBlockConfig(shape=(2, codec.n_fft - codec.hop_length), dtype=f32)

    def slots(key, node, num_layers, blocks):
        return RecurrentStateSpec(resource_key=key, nodes={node}, config=RecurrentStateConfig(
            num_layers=num_layers, blocks=blocks, max_slots=sessions + 1))

    return [
        KVSpec(resource_key=NANO_KV, nodes={"nano_llm"}, config=PagedKVConfig(
            num_layers=nano.num_attention_layers, num_kv_heads=nano.num_key_value_heads, head_dim=nano.head_dim,
            max_seq_len=nano.max_position_embeddings, num_qo_heads=nano.num_attention_heads, max_num_pages=641)),
        AttentionSpec(resource_key=NANO_ATTN, nodes={"nano_llm"}, config=AttentionConfig(kv_cache=NANO_KV)),
        slots(MAMBA_STATE, "nano_llm", nano.num_mamba_layers, Mamba2Geometry(
            num_heads=nano.mamba_num_heads, head_dim=nano.mamba_head_dim, state_size=nano.ssm_state_size,
            n_groups=nano.n_groups, conv_kernel_size=nano.conv_kernel).to_blocks()),
        LinearAttnSpec(resource_key=MAMBA, nodes={"nano_llm"}, config=LinearAttnConfig(
            recurrent_state=MAMBA_STATE, variant=LinearAttnVariant.MAMBA2, time_step_limit=nano.time_step_limit)),
        SamplerSpec(resource_key=NANO_SAMPLER, nodes={"nano_llm"}, vocab_size=model.config.vocab_size,
                    enable_repetition_penalty=True),
        KVSpec(resource_key=TALKER_KV, nodes={"eartts_talker"}, config=PagedKVConfig(
            num_layers=eartts.num_hidden_layers, num_kv_heads=eartts.num_key_value_heads, head_dim=eartts.kv_head_dim,
            max_seq_len=eartts.sliding_window + 512, num_qo_heads=eartts.num_attention_heads, max_num_pages=1025)),
        AttentionSpec(resource_key=TALKER_ATTN, nodes={"eartts_talker"}, config=AttentionConfig(kv_cache=TALKER_KV)),
        PositionSpec(resource_key=TALKER_POS, nodes={"eartts_talker"}, config=PositionConfig(
            kv_cache=TALKER_KV, rotary_dim=eartts.head_dim, interleave=False, rope_theta=eartts.rope_theta_local)),
        slots(CODEC_STATE, "eartts_talker", 1, codec_blocks),
        # Named: the talker kept each call's previous codes in request state. They are a Value now.
        slots(TALKER_PREV, "eartts_talker", 1,
              {"codes": RecurrentBlockConfig(shape=(eartts.num_quantizers,), dtype=torch.long)}),
        slots(ENCODER_STATE, "conformer_encoder", enc.num_layers,
              {"conv": RecurrentBlockConfig(shape=(enc.hidden_size, enc.conv_kernel_size - 1), dtype=None)}),
        slots(ENCODER_FRONT, "conformer_encoder", 1, {
            "sig": RecurrentBlockConfig(shape=(hop * enc.subsampling_factor,), dtype=f32),
            "mel": RecurrentBlockConfig(shape=(2 * enc.subsampling_factor, enc.num_mel_bins), dtype=f32),
            "last": RecurrentBlockConfig(shape=(1,), dtype=f32),
        }),
        BoundedKVSpec(resource_key=ENCODER_KV, nodes={"conformer_encoder"}, config=BoundedKVConfig(
            num_layers=enc.num_layers, num_heads=enc.num_heads, head_dim=enc.head_dim, rows_per_request=1,
            max_source_len=0, retention=SinkWindow(0, enc.left_context), query_window=enc.left_context,
            max_slots=sessions, dtype=None)),
    ]


SYNTHETIC = {"resources": {
    NANO_KV: {"max_num_pages": 777}, TALKER_KV: {"page_size": 64, "max_num_pages": 900}, MAMBA_STATE: {"max_slots": 9},
    CODEC_STATE: {"max_slots": 9}, TALKER_PREV: {"max_slots": 9}, ENCODER_STATE: {"max_slots": 9},
    ENCODER_FRONT: {"max_slots": 9}, ENCODER_KV: {"max_slots": 7},
}}


@pytest.mark.parametrize("deployment", [
    *(yaml.safe_load(path.read_text()) for path in _published("NemotronDuplexModel")), SYNTHETIC,
], ids=[*(path.stem for path in _published("NemotronDuplexModel")), "synthetic"])
def test_duplex_specs_equal_main_s(deployment):
    model = _model(**deployment.get("model_kwargs", {}))
    by_key = {spec.resource_key: spec for spec in serving_resources(model, deployment)}
    assert by_key[MAMBA_STATE].config.blocks["conv"].dtype is None, \
        "the conv is stored in the compute dtype, bf16 at the default autocast, which main wrote out"
    pos = by_key[TALKER_POS]
    assert pos.config.rotary_dim is None, "a counter: the talker rotates in torch, at a theta per layer kind"
    expected = _reference_specs(model)
    apply_yaml_overrides(expected, deployment)
    main_pos = next(spec for spec in expected if spec.resource_key == TALKER_POS)
    by_key[TALKER_POS] = replace(pos, config=replace(pos.config, rotary_dim=main_pos.config.rotary_dim))
    kv_dtype = parse_autocast_dtype(resolve_model_dtype(model, deployment))
    # main left the codec's and the encoder's conv in the compute dtype too (dtype None), so both sides are stored
    _assert_equal_per_key(list(by_key.values()), [_stored(spec, kv_dtype) for spec in expected], kv_dtype)


def test_declarations_have_no_problems_and_one_digest():
    model = _model()
    assert {d.name for d in model.declare_state()} == {
        NANO_KV, MAMBA_STATE, NANO_SAMPLER, TALKER_KV, TALKER_PREV, CODEC_STATE, ENCODER_KV, ENCODER_STATE,
        ENCODER_FRONT, STREAM,
    }
    assert state_declaration_problems(model) == []
    assert declaration_digests(model.declare_state()) == declaration_digests(_model().declare_state())


# Steps


def _nano_steps():
    model = _model()
    nano = _bind(_make_nano(), model, "nano_llm")
    frame = nano.prepare_inputs("decode", None, {"audio_frame": [torch.ones(8)]})
    prompt = nano.prepare_inputs("decode", None, {"text_inputs": [torch.arange(4)], "audio_frame": [torch.ones(8)]})
    for inputs, key in (([frame, frame], nano.DECODE_KEY), ([prompt, frame], None)):
        main = SubmoduleStep(segments=[Segment(request_id=r, label="main", span=i.input_seq_len)
                                       for r, i in zip([1, 2], inputs, strict=True)], cg_key_info=key, steps={
            NANO_KV: KVStep(), NANO_ATTN: AttentionStep(causal=True), MAMBA_STATE: RecurrentStep(),
            MAMBA: LinearAttnStep(), NANO_SAMPLER: SamplerStep(apply_penalty=True),
        })
        yield engine_step(nano, graph_walk="decode", request_ids=[1, 2], inputs=inputs), main


def _talker_steps():
    model = _model()
    sub = _make_talker_sub()
    resources = {spec.resource_key: _StubResource() for spec in serving_resources(model, {})}
    sub.state_handles = node_handles(model, "eartts_talker", resources)
    first = sub.prepare_inputs("talker_decode", _fwd("a"), {"new_token": [torch.tensor([5])]})
    later = sub.prepare_inputs("talker_decode", _fwd("b", 3), {"new_token": [torch.tensor([7])]})
    for rids, inputs, key in ((["a", "b"], [first, later], None), (["b"], [later], sub.DECODE_KEY)):
        step = engine_step(sub, graph_walk="talker_decode", request_ids=rids, inputs=inputs)
        prev = step.steps.pop(TALKER_PREV)
        assert prev == RecurrentStep(segments=tuple(Segment(rid, "main", 1) for rid in rids)), \
            "the previous codes, a Value read and written on the main rows, which main kept in request state"
        kv = step.steps[TALKER_KV]
        assert kv.combined_labels == {("main", "uncond"): "main+uncond"}, \
            "the combined label is the layer's labels joined, where main named it _cfg"
        step.steps[TALKER_KV] = replace(kv, combined_labels={("main", "uncond"): "_cfg"})
        main = SubmoduleStep(segments=[Segment(request_id=rid, label=label, span=inp.input_seq_len)
                                       for label in ("main", "uncond") for rid, inp in zip(rids, inputs, strict=True)],
                             cg_key_info=key, steps={
            TALKER_KV: KVStep(combined_labels={("main", "uncond"): "_cfg"}), TALKER_ATTN: AttentionStep(causal=True),
            TALKER_POS: PositionStep(),
            CODEC_STATE: RecurrentStep(segments=tuple(Segment(rid, "main", 1) for rid in rids)),
        })
        yield step, main


def _main_encoder_step(rids, inputs, enc) -> SubmoduleStep:
    """Main's eager or mixed step: the pools over the step's rows at span 1, the KV over each stream's new frames at
    its position."""
    # main's RecurrentStep() took the step's list of segments; the same rows
    rows = tuple(Segment(rid, "main", 1) for rid in rids)
    return SubmoduleStep(segments=list(rows), steps={
        FRONT: RecurrentStep(segments=rows), CONV: RecurrentStep(segments=rows),
        KV: BoundedKVStep(
            segments=tuple(Segment(rid, "main", inp.kwargs["n"]) for rid, inp in zip(rids, inputs, strict=True)),
            positions={rid: StreamPosition(0, enc.counts(rid)["frames"]) for rid in rids},
        ),
    })


def _main_region_step(rids, keys, enc) -> SubmoduleStep:
    """Main's region step: every row at span 1, and a position only for a stream."""
    return SubmoduleStep(segments=[Segment(rid, "main", 1) for rid in rids], steps={
        **{key: RecurrentStep() for key in keys},
        KV: BoundedKVStep(positions={rid: StreamPosition(0, enc.counts(rid)["frames"]) for rid in rids
                                     if rid in enc.opened}),
    })


def _conv_rows_at_the_new_frames(step, inputs):
    conv = step.steps[CONV]
    assert [s.span for s in conv.segments] == [inp.kwargs["n"] for inp in inputs], \
        "the conv Value's rows at each stream's new frames, where main took 1: a row at 0 reads its start and writes " \
        "the sink, so a stream with no new frame keeps its state"
    step.steps[CONV] = replace(conv, segments=tuple(Segment(s.request_id, s.label, 1) for s in conv.segments))


def _encoder_steps():
    enc = _Encoder()
    wavs = {h: torch.randn(1280 * 8) * 0.1 for h in (1, 2, 3)}
    for k in range(3):
        enc.step([1, 2], [enc.prepare(h, wavs[h][1280 * k:1280 * (k + 1)]) for h in (1, 2)])
    # eager: a frame on, under a frame, a stream's opening chunk
    rids = [1, 2, 3]
    inputs = [enc.prepare(1, wavs[1][3840:5120]), enc.prepare(2, wavs[2][3840:4000]), enc.prepare(3, wavs[3][:9000])]
    assert [inp.kwargs["n"] for inp in inputs] == [1, 0, 7]
    step = enc.declare(rids, inputs)
    assert step.state[STREAM].host == {(rid, "main", 0): inp.kwargs["after"]
                                       for rid, inp in zip(rids, inputs, strict=True)}
    _conv_rows_at_the_new_frames(step, inputs)
    yield step, _main_encoder_step(rids, inputs, enc)

    # mixed: two streams one frame on under the whole-step region's lease, the third eager
    for k in range(3, 5):
        enc.step([1, 2], [enc.prepare(h, wavs[h][1280 * k:1280 * (k + 1)]) for h in (1, 2)])
    enc.regions = _encoder_regions(enc, STEP, LAYERS)
    inputs = [enc.prepare(1, wavs[1][6400:7680]), enc.prepare(2, wavs[2][6400:7680]), enc.prepare(3, wavs[3][:640])]
    assert [inp.kwargs["steady"] for inp in inputs] == [True, True, False]
    step = enc.declare(rids, inputs)
    _conv_rows_at_the_new_frames(step, inputs)
    yield step, _main_encoder_step(rids, inputs, enc)

    step = enc.declare([1, 2], inputs[:2])
    assert STEP in step.ctx.piecewise_leases
    assert set(step.state) == {STREAM} and step.steps == {}, \
        "main declared nothing for an all-steady step; the streams' counts build no resource step"

    padded = [1, 2, "pad"]
    for label, keys in ((STEP, (FRONT, CONV)), (LAYERS, (CONV,))):
        step = enc.regions[label]._region_step(padded, [1, 1, 1])
        assert step.steps[KV].positions.pop("pad") == StreamPosition(0, 0), \
            "a padding row's position, where main gave none: the manager plans and commits no padding row"
        yield step, _main_region_step(padded, keys, enc)


@pytest.mark.parametrize("steps", [_nano_steps, _talker_steps, _encoder_steps], ids=["nano", "talker", "encoder"])
def test_duplex_steps_equal_main_s(steps):
    for step, main in steps():
        _assert_steps_equal(step, main)


# Layouts


def _main_layouts(model: NemotronDuplexModel, prompt_len: int) -> dict:
    frames = model.max_session_frames + 1
    nano = LabelLayout(spans=(Span("ids", "decode", prompt_len, prompt_len, "text_inputs"),) if prompt_len else (),
                       growth=(("decode_loop", 1),), max_growth=frames)
    warmup = model.TALKER_WARMUP_POSITIONS
    talker = LabelLayout(spans=(Span("rows", "talker_decode", warmup, warmup, "audio_prompt_latents"),),
                         growth=(("talker_decode_loop", 1),), max_growth=frames)
    return {NANO_KV: {"main": nano}, TALKER_KV: {"main": talker, "uncond": talker}}


@pytest.mark.parametrize("prompt, length", [(None, 0), ("a prompt", 5)])
def test_duplex_layouts_equal_main_s(prompt, length):
    model = _model()
    ids = torch.zeros(1, 5, dtype=torch.long)
    model._tokenizer = lambda prompt, return_tensors: SimpleNamespace(input_ids=ids)
    out = model.process_prompt(prompt, ["audio", "text"], ["audio", "text"],
                               tensors={"audio_inputs": [torch.zeros(16000)]})
    assert kv_layouts(model.declare_state(), out.metadata["state_layout"], {}) == _main_layouts(model, length)


# The talker's noise and first step


def test_the_noise_memo_s_row_k_is_a_fresh_generator_s_kth_draw():
    sub = _make_talker_sub()
    e = sub.config.eartts
    sub._noise([3], torch.device("cpu"))
    for k in range(4):
        gen = torch.Generator().manual_seed(e.inference_seed)
        for _ in range(k + 1):
            u = torch.rand((e.inference_num_iter, 1, e.mog_num_predictions), generator=gen)
            eps = torch.randn((e.inference_num_iter, 1, e.code_dim), generator=gen)
        assert torch.equal(sub._noise_u[k], u) and torch.equal(sub._noise_eps[k], eps), f"row {k}"


def test_the_noise_memo_holds_the_talker_loop_s_cap_plus_one(monkeypatch):
    from test_nemotron_duplex_model import _talker_from

    sub = _talker_from(monkeypatch, NemotronDuplexConfig())
    (loop,) = (section for section in _model().get_graph_walk_graphs().values()
               if getattr(section, "name", None) == "talker_decode_loop")
    assert sub.noise_rows == loop.max_iters + 1


def test_the_talker_refuses_a_noise_row_past_the_loop_s_cap():
    sub = _make_talker_sub()    # 8 rows: a loop cap of 7
    engine = SimpleNamespace(request_ids=["a"], resources={}, per_request_states=None)
    last = sub.prepare_inputs("talker_decode", _fwd("a", 7), {"new_token": [torch.tensor([5])]})
    sub.preprocess("talker_decode", engine, [last])
    past = sub.prepare_inputs("talker_decode", _fwd("a", 8), {"new_token": [torch.tensor([5])]})
    with pytest.raises(ValueError, match="talker step 8 has no noise row: the memo holds 8, the talker loop's cap"):
        sub.preprocess("talker_decode", engine, [past])


def test_the_talker_s_first_step_reads_the_warm_up_codes():
    talker = _Talker(captured=False)
    for pool in talker.pools.values():
        pool.ingest_request("a")
    ctx = StepContext(request_ids=("a",), graph_walk="talker_decode", slot=0, capture=False)
    step = SubmoduleStep(segments=[Segment("a", "main", 1)], state={TALKER_PREV: ArrayStep()})
    step.set_ctx(ctx)
    step = state_steps(step, talker.handles)
    step.set_ctx(ctx)
    assert talker.runner.admit(step).ok
    talker.runner.plan(step)
    step.begin_calls()
    assert torch.equal(talker.node.prev.read(0).fields["codes"], talker.node.warmup()[2].unsqueeze(0))


def test_duplex_layers_run_standalone_with_no_handle():
    """With no handle bound, each stateful layer keeps ``state`` None and runs the single-sequence math that
    ``offline_inference`` uses: the nano's Mamba-2 and attention layers, the causal encoder's attention and convs, and
    the talker's attention."""
    torch.manual_seed(0)
    nano = NanoConfig(
        num_hidden_layers=2, hybrid_override_pattern="M*", hidden_size=32, intermediate_size=64, vocab_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, mamba_num_heads=4, mamba_head_dim=16,
        ssm_state_size=8, n_groups=2, time_step_rank=8, combined_embed_dim=32,
    )
    llm = NemotronHLLM(nano)
    enc_config = FastConformerConfig(hidden_size=32, num_heads=4, num_layers=2, intermediate_size=64, num_mel_bins=16,
                                     subsampling_channels=8, causal=True, left_context=6)
    keyed = FastConformerEncoder(enc_config, kv_key=ENCODER_KV, conv_key=ENCODER_STATE)
    plain = FastConformerEncoder(enc_config)
    plain.load_state_dict(keyed.state_dict())
    attn = _Attn(TINY, qk_norm=True, layer=0)
    stateful = [m for module in (llm, keyed, attn) for m in module.modules() if hasattr(m, "bind_state")]
    for layer in stateful:
        layer.bind_state({})
    assert {type(layer) for layer in stateful} == {
        Mamba2Mixer, NemotronHAttention, RelPositionAttention, ConvModule, _Attn}
    assert all(layer.state is None for layer in stateful)

    x = torch.randn(5, nano.hidden_size)
    mixer = next(m for m in llm.modules() if isinstance(m, Mamba2Mixer))
    h = torch.randn(2, 7, enc_config.hidden_size)
    with torch.no_grad():
        torch.testing.assert_close(mixer(x), mixer.prefill(x)[0])
        torch.testing.assert_close(llm.offline_forward(x), llm.prefill(x)[0])
        torch.testing.assert_close(keyed.blocks(h), plain.blocks(h))
        assert attn(torch.randn(1, 5, TINY.hidden_size)).shape == (1, 5, TINY.hidden_size)


# Session checks


def test_max_session_s_past_the_talker_window_is_refused():
    """The talker reads every layer causally, which Gemma3's sliding layers match only within their window: at the
    default 7500 positions, 596 s gives 7489 and 597 s gives 7502."""
    _model(max_session_s=596).validate_config_yaml({"max_concurrent_requests": 32}, "c.yaml")
    with pytest.raises(ValueError, match="max_session_s 597 gives a talker 7502 positions, past its 7500-position "
                                         "sliding window, which the talker reads causally"):
        _model(max_session_s=597).validate_config_yaml({"max_concurrent_requests": 32}, "c.yaml")


PER_SESSION_POOLS = (MAMBA_STATE, ENCODER_STATE, ENCODER_FRONT, CODEC_STATE, TALKER_PREV, ENCODER_KV, NANO_KV,
                     TALKER_KV)


@pytest.mark.parametrize("key", PER_SESSION_POOLS)
def test_duplex_refuses_a_cap_past_any_per_session_pool(key):
    """Every session holds a slot in each Value's pool and in ``encoder_kv``, and its nano and talker pages, from start
    to end, through the layouts ``process_prompt`` builds with both loops at their cap. A cap of 33 needs each pool
    raised past its default 32 sessions, and a deployment that misses one is refused at startup, naming it."""
    model = _model()
    raised = {pool: {"max_slots": 34} for pool in (MAMBA_STATE, ENCODER_STATE, ENCODER_FRONT, CODEC_STATE, TALKER_PREV)}
    raised[ENCODER_KV] = {"max_slots": 33}
    raised[NANO_KV] = {"max_num_pages": 1 + 33 * 20}
    raised[TALKER_KV] = {"max_num_pages": 1 + 33 * 32}
    assert raised.keys() == set(PER_SESSION_POOLS)
    model.validate_config_yaml({"max_concurrent_requests": 33, "resources": raised}, "c.yaml")
    del raised[key]
    with pytest.raises(ValueError, match=f"33 concurrent requests .* resources.{key}"):
        model.validate_config_yaml({"max_concurrent_requests": 33, "resources": raised}, "c.yaml")
