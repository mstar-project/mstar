"""Chatterbox plans on the contract what it planned by hand, and S3Gen streams the audio it streamed before.

T3's specs, steps, layouts and request configs below are the reference, copied from the model as it stood before the
port, so the adapter can't pass by agreeing with itself. Two differences are named in the reference: a guided step
plans under ``main+uncond`` where it planned under ``cfg``, and S3Gen declares a Value and a step for the streams it
kept in a dict. Moving a stream there moved every noise draw out of the forward into ``prepare_inputs``, and a draw
taken in another order changes the audio with nothing failing. So S3Gen's chunks are checked sample for sample against
its dict-backed path, copied here too.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest
import torch
import yaml
from test_chatterbox_model import _bind_stream, _make_model, _s3_submodule, _s3gen_step, _t3_submodule, _tone
from test_startup_graph_validation import _FACTORIES
from test_state_ports import _assert_equal_per_key, _assert_steps_equal, _published

from mstar.conductor.conductor import Conductor
from mstar.engine.node_dtype import resolve_model_dtype
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.lora.config import LORA_RESOURCE_KEY
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.engine.state import Span, host
from mstar.engine.state.adapters import StateSetup, kv_layouts
from mstar.engine.state.analysis import declaration_digests, state_declaration_problems
from mstar.model.base import LabelLayout, enabled_resources, parse_autocast_dtype, serving_resources
from mstar.model.chatterbox.chatterbox_model import ChatterboxModel
from mstar.model.chatterbox.components.audio_frontend import resample, trim_reference
from mstar.model.chatterbox.components.s3_tokenizer import S3Tokenizer
from mstar.model.chatterbox.components.s3gen import FlowRow, ReferenceConditioning, S3Gen
from mstar.model.chatterbox.config import (
    COND_LABEL,
    S3_SR,
    S3GEN_NODE,
    S3GEN_STREAM,
    T3_ATTN,
    T3_KV,
    T3_NODE,
    T3_POS,
    T3_SAMPLER,
    UNCOND_LABEL,
    ChatterboxConfig,
    S3GenCFMConfig,
    S3GenConfig,
    S3GenEncoderConfig,
    S3GenEstimatorConfig,
    S3GenHiFTConfig,
    T3Config,
)
from mstar.model.chatterbox.submodules import (
    DECODE_LOOP,
    PREFILL_WALKS,
    PREV_TOKEN,
    PROMPT_TOKENS,
    REF_AUDIO,
    SPEECH_TOKENS,
    TEXT_INPUTS,
    VOICE_KEY,
    S3GenSubmodule,
    T3Submodule,
    max_reference_tokens,
    reference_tokens,
)
from mstar.model.components.audio_features import MatchaLogMel
from mstar.model.submodule_base import ARNodeInputs, engine_step
from mstar.streaming.chunk_policy import RampChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.worker.executing_batch import input_metadata_and_fwd_info

VARIANTS = ("chatterbox", "multilingual", "turbo")


def _reference_specs(model: ChatterboxModel) -> list:
    """`ChatterboxModel.get_node_resources` before the port, and the stream Value S3Gen declares now. Less the opt-in
    LoRA spec, which `enabled_resources` builds only when a deployment names it."""
    t3 = model.config.t3
    bb = t3.backbone
    kv = PagedKVConfig(
        num_layers=bb.num_hidden_layers,
        num_kv_heads=bb.num_key_value_heads,
        head_dim=bb.head_dim,
        max_seq_len=t3.cond_len + model.config.max_text_tokens + t3.max_speech_tokens,
        num_qo_heads=bb.num_attention_heads,
    )
    if bb.is_gpt2:
        position = PositionConfig(kv_cache=T3_KV)
    else:
        position = PositionConfig(
            kv_cache=T3_KV,
            rope_theta=bb.rope_theta,
            rope_scale=bb.rope_scaling["factor"],
            low_freq_factor=bb.rope_scaling["low_freq_factor"],
            high_freq_factor=bb.rope_scaling["high_freq_factor"],
            old_context_len=bb.rope_scaling["original_max_position_embeddings"],
        )
    f32 = torch.float32
    return [
        KVSpec(resource_key=T3_KV, nodes={T3_NODE}, config=kv),
        AttentionSpec(resource_key=T3_ATTN, nodes={T3_NODE}, config=AttentionConfig(kv_cache=T3_KV)),
        PositionSpec(resource_key=T3_POS, nodes={T3_NODE}, config=position),
        SamplerSpec(
            resource_key=T3_SAMPLER, nodes={T3_NODE}, vocab_size=t3.speech_vocab_size,
            enable_repetition_penalty=True, enable_min_p=not model.config.is_turbo,
        ),
        # Named: S3Gen kept each stream in a dict on its node. The stream is now a Value of the vocoder's 8-frame tail
        # and the reference, padded to the 250 tokens of a 10 s clip.
        RecurrentStateSpec(
            resource_key=S3GEN_STREAM, nodes={S3GEN_NODE},
            config=RecurrentStateConfig(num_layers=1, labels=("main",), blocks={
                "hift_mel": RecurrentBlockConfig(shape=(80, 8), dtype=f32),
                "hift_source": RecurrentBlockConfig(shape=(1, 3840), dtype=f32),
                "hift_speech": RecurrentBlockConfig(shape=(3840,), dtype=f32),
                "ref_tokens": RecurrentBlockConfig(shape=(250,), dtype=torch.long),
                "ref_feat": RecurrentBlockConfig(shape=(500, 80), dtype=f32),
                "ref_embedding": RecurrentBlockConfig(shape=(192,), dtype=f32),
            }),
        ),
    ]


@pytest.mark.parametrize("variant", VARIANTS)
def test_specs_equal_the_hand_written_ones(variant):
    model = _make_model(variant)
    _assert_equal_per_key(enabled_resources(model, {}, StateSetup()), _reference_specs(model))


@pytest.mark.parametrize("variant", VARIANTS)
def test_specs_equal_the_hand_written_ones_under_a_resources_block(variant):
    model = _make_model(variant)
    deployment = {"resources": {
        T3_KV: {"page_size": 32, "max_num_pages": 777}, T3_ATTN: {"decode_tensor_cores": False},
        S3GEN_STREAM: {"max_slots": 65},
    }}
    expected = _reference_specs(model)
    apply_yaml_overrides(expected, deployment)
    _assert_equal_per_key(serving_resources(model, deployment), expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_port_writes_only_lora_by_hand(variant):
    hand_written = {spec.resource_key for spec in _make_model(variant).get_node_resources()}
    assert hand_written == (set() if variant == "turbo" else {LORA_RESOURCE_KEY})


def test_every_published_config_is_covered():
    stems = {path.stem for path in _published("ChatterboxModel")}
    assert stems == {
        "chatterbox", "chatterbox_split", "chatterbox_multilingual", "chatterbox_multilingual_split",
        "chatterbox_turbo", "chatterbox_turbo_split",
    }, "a config here that the port never compares would change with no test noticing"


@pytest.mark.parametrize("path", _published("ChatterboxModel"), ids=lambda path: path.stem)
def test_specs_equal_the_hand_written_ones_on_every_published_config(path, tmp_path, monkeypatch):
    deployment = yaml.safe_load(path.read_text())
    model = _FACTORIES["ChatterboxModel"](deployment["model"], deployment["model_kwargs"], tmp_path, monkeypatch)
    _assert_equal_per_key(enabled_resources(model, {}, StateSetup()), _reference_specs(model))
    expected = _reference_specs(model)
    apply_yaml_overrides(expected, deployment)
    kv_dtype = parse_autocast_dtype(resolve_model_dtype(model, deployment))
    _assert_equal_per_key(serving_resources(model, deployment), expected, kv_dtype)


@pytest.mark.parametrize("variant", VARIANTS)
def test_declarations_have_no_problems_and_one_digest(variant):
    model = _make_model(variant)
    assert state_declaration_problems(model) == []
    assert declaration_digests(model.declare_state()) == declaration_digests(_make_model(variant).declare_state())


# Steps

# Named: a guided step planned under ``cfg``, and a step over several labels now plans under its labels joined by
# ``+``. Only the KV and attention resources read a plan's name.
GUIDED_PLAN = "main+uncond"


def _reference_t3_step(walk: str, rids: list[int], spans: tuple[int, ...], guided: bool) -> SubmoduleStep:
    """`T3Submodule.declare_step` before the port, with the guided plan's name."""
    labels = (COND_LABEL, UNCOND_LABEL) if guided else (COND_LABEL,)
    segments = [
        Segment(request_id=rid, label=label, span=span)
        for label in labels for rid, span in zip(rids, spans, strict=True)
    ]
    tracked = {}
    if walk in PREFILL_WALKS:
        bos = torch.tensor([ChatterboxConfig.chatterbox().t3.start_speech_token], dtype=torch.long)
        tracked = {rid: bos for rid in rids}
    return SubmoduleStep(
        segments=segments,
        combined_labels={labels: GUIDED_PLAN} if guided else {},
        steps={
            T3_KV: KVStep(),
            T3_ATTN: AttentionStep(causal=True),
            T3_POS: PositionStep(),
            T3_SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=tracked),
        },
        cg_key_info=guided,
    )


@pytest.mark.parametrize("guided", [True, False])
@pytest.mark.parametrize("walk, spans", [("prefill", (40, 37)), ("prefill_voice", (52, 9)), ("decode", (1, 1))])
def test_t3_steps_equal_the_hand_written_ones(walk, spans, guided):
    rows = [ARNodeInputs(input_seq_len=span, resource_step_info=guided) for span in spans]
    step = engine_step(_t3_submodule(), graph_walk=walk, request_ids=[1, 2], inputs=rows)
    _assert_steps_equal(step, _reference_t3_step(walk, [1, 2], spans, guided))


def test_s3gen_declares_its_streams_value_where_it_declared_no_step():
    """Named: S3Gen declared no step, since its streams lived in a dict. Each row now steps its stream's one entry."""
    sub, _ = _s3_submodule()
    rows = [_prepare(sub, rid, torch.arange(1, 21), is_final=False) for rid in (1, 2)]
    step = engine_step(sub, graph_walk="s3gen_chunk", request_ids=[1, 2], inputs=rows)
    _assert_steps_equal(step, SubmoduleStep(
        segments=[Segment(request_id=1, label="main", span=1), Segment(request_id=2, label="main", span=1)],
        steps={S3GEN_STREAM: RecurrentStep()},
    ))


# Layouts and request configs


def _reference_layout(model: ChatterboxModel, text: int, clip: torch.Tensor | None, guided: bool) -> dict:
    """``kv_layout`` as `ChatterboxModel.process_prompt` returned it before the port."""
    t3 = model.config.t3
    walk = "prefill" if clip is None else "prefill_voice"
    cond = t3.cond_len
    if not t3.use_perceiver_resampler:
        cond += reference_tokens(clip.numel(), model.config) - t3.speech_cond_prompt_len
    bos = 2 if t3.duplicate_bos_in_prefill else 1
    layout = LabelLayout(
        spans=(
            Span("rows", walk, cond, cond, PROMPT_TOKENS),
            Span("ids", walk, text, text, TEXT_INPUTS),
            Span("rows", walk, bos, bos, PREV_TOKEN),
        ),
        growth=(("decode_loop", 1),),
    )
    labels = (COND_LABEL, UNCOND_LABEL) if guided else (COND_LABEL,)
    return {T3_KV: dict.fromkeys(labels, layout)}


@pytest.mark.parametrize("variant, clip, cfg_weight", [
    ("chatterbox", False, 0.5), ("chatterbox", True, 0.5), ("chatterbox", False, 0.0), ("turbo", True, 0.0),
])
def test_layouts_convert_to_the_hand_written_ones(variant, clip, cfg_weight):
    model = _make_model(variant)
    tensors = {"audio_inputs": [_tone(6.0)]} if clip else None
    out = model.process_prompt("hello", ["text", "audio"] if clip else ["text"], ["audio"], tensors=tensors,
                               cfg_weight=cfg_weight)
    converted = kv_layouts(model.declare_state(), out.metadata["state_layout"], {})
    text = out.new_input_tensors[TEXT_INPUTS][0].numel()
    reference_clip = out.new_input_tensors[REF_AUDIO][0] if clip else None
    assert converted == _reference_layout(model, text, reference_clip, cfg_weight > 0)


def _reference_configs(model: ChatterboxModel, kwargs: dict) -> dict:
    """`ChatterboxModel.get_request_resource_configs` before the port."""
    knobs = model.resolve_generation_kwargs(kwargs)
    # Named: the model returned this KVReqConfig, and the conductor now fills it with the labels the layout names
    labels = [COND_LABEL, UNCOND_LABEL] if knobs["cfg_weight"] > 0 else [COND_LABEL]
    return {
        T3_SAMPLER: SamplingReqConfig(
            temperature=knobs["temperature"], top_k=knobs["top_k"], top_p=knobs["top_p"],
            repetition_penalty=knobs["repetition_penalty"], min_p=knobs["min_p"], ignore_eos=knobs["ignore_eos"],
            penalize_prompt=knobs["penalize_prompt"],
        ),
        T3_KV: KVReqConfig(needed_labels=labels),
    }


@pytest.mark.parametrize("variant, kwargs", [
    ("chatterbox", {}), ("chatterbox", {"cfg_weight": 0.0, "temperature": 0.3, "ignore_eos": True}),
    ("chatterbox", {"min_p": 0.1, "top_k": 40}), ("turbo", {"top_p": 0.5}),
])
def test_request_configs_equal_the_hand_written_ones(variant, kwargs):
    model = _make_model(variant)
    # Turbo's built-in voice is in the checkpoint, which these tests don't load
    out = model.process_prompt("hello", ["text", "audio"], ["audio"], tensors={"audio_inputs": [_tone(6.0)]}, **kwargs)
    kv_layout = kv_layouts(model.declare_state(), out.metadata["state_layout"], {})
    conductor = SimpleNamespace(
        model=model, _declarations=model.declare_state, _node_resources=list,
        _kv_labels={key: tuple(labels) for key, labels in model.kv_labels().items()},
    )
    configs = Conductor._get_resource_configs(
        conductor, ["text", "audio"], ["audio"], {**kwargs, "kv_layout": kv_layout},
    )
    assert configs == _reference_configs(model, kwargs)


# T3's speech position


def _decode_embeds(sub: T3Submodule, metadata) -> torch.Tensor:
    fwd_info = SimpleNamespace(step_metadata={"cfg_weight": 0.5})
    inputs = {PREV_TOKEN: [torch.tensor([17])]}
    return sub.prepare_inputs("decode", fwd_info, inputs, input_metadata=metadata).input_embeds


def _decode_metadata(iteration: int | None, continues: bool = False):
    """The row's input metadata as the worker builds it, from the loop's count; ``continues`` for a row continuing
    from the step in flight, which the worker prepares before that step's count lands."""
    counts = {} if iteration is None else {DECODE_LOOP: iteration}
    fwd_info = SimpleNamespace(rid_handle=7, dynamic_loop_iter_counts=counts)
    metadata, _ = input_metadata_and_fwd_info(
        SimpleNamespace(get_fwd_info=lambda rid, partition: fwd_info), SimpleNamespace(possible=lambda: False),
        T3_NODE, "T3", {7: {}}, increment_loop_rids=[7] if continues else None, increment_loop_name=DECODE_LOOP,
    )
    return metadata[7]


def _at_position(sub: T3Submodule, position: int) -> torch.Tensor:
    return sub.model.speech_emb(torch.tensor([17])) + sub.model.speech_pos_emb.weight[position]


def test_the_first_decode_step_reads_speech_position_one():
    sub = _t3_submodule()
    embeds = _decode_embeds(sub, _decode_metadata(None))
    assert torch.equal(embeds, _at_position(sub, 1)), "the prefill's token sits at 0, so the first decode is at 1"


def test_a_decode_step_prepared_again_keeps_its_speech_position():
    sub = _t3_submodule()
    first = _decode_embeds(sub, _decode_metadata(4))
    again = _decode_embeds(sub, _decode_metadata(4))
    assert torch.equal(first, again) and torch.equal(first, _at_position(sub, 5)), (
        "a step prepared again after a refused admit moved its speech position"
    )


def test_a_speculative_continuation_reads_the_position_after_the_step_in_flight():
    sub = _t3_submodule()
    speculative = _decode_embeds(sub, _decode_metadata(3, continues=True))
    landed = _decode_embeds(sub, _decode_metadata(4))
    assert torch.equal(speculative, landed), "a row continuing from the step in flight would repeat its position"


# The chunk policy


def _chunks(policy, tokens: int, done_first: bool = False) -> list:
    """Every chunk of a ``tokens``-long stream, T3 putting one token at a time, then the flush at its end. S3Gen pops
    each chunk as soon as it's ready, or with ``done_first`` T3 finishes before S3Gen takes the last one."""
    buffer = StreamBuffer(request_id=0, edge_name=SPEECH_TOKENS, from_partition="T3", policy=policy)
    chunks = []
    for i in range(tokens):
        buffer.pre_read_register(str(i))
        buffer.put(str(i), torch.tensor([i]))
        if done_first and i == tokens - 1:
            break
        while buffer.has_chunk_ready():
            chunks.append(buffer.pop_chunk())
    buffer.signal_done(tokens)
    while buffer.has_chunk_ready():
        chunks.append(buffer.pop_chunk())
    return chunks


def _items(chunk) -> list[int]:
    data = chunk.data["data"]
    return [] if data is None else data.reshape(-1).tolist()


@pytest.mark.parametrize("done_first", [False, True], ids=["s3gen_pops_first", "t3_finishes_first"])
@pytest.mark.parametrize("first, chunk, growth, cap", [
    (15, 50, 2.0, 200), (35, 50, 2.0, 200), (15, 50, 1.0, 200), (10, 30, 1.5, 100), (15, 50, 2.0, 50),
])
def test_each_chunk_carries_the_ramps_new_tokens_behind_the_whole_history(first, chunk, growth, cap, done_first):
    """Read against the ramp's chunks when S3Gen pops before T3 finishes. The port keeps every token in its buffer, so
    only the flush at the end of a stream marks a chunk final, whether T3 or S3Gen finishes first. The ramp's buffer
    empties when a pop lands on a chunk boundary, so if T3 has already finished, the ramp ends the stream one solve
    sooner."""
    model = _make_model()
    model.config.stream_first_chunk_tokens, model.config.stream_chunk_tokens = first, chunk
    model.config.stream_chunk_growth, model.config.stream_max_chunk_tokens = growth, cap
    for tokens in [*range(1, 400), 999, 1000, 1001]:
        # the policy before the port
        ramp = RampChunkPolicy(first_chunk=first, chunk_size=chunk, growth=growth, max_chunk=max(cap, chunk))
        before_chunks, now_chunks = _chunks(ramp, tokens), _chunks(model._chunk_policy(), tokens, done_first)
        assert len(now_chunks) == len(before_chunks), f"a {tokens}-token stream would take another number of solves"
        seen = []
        for before, now in zip(before_chunks, now_chunks, strict=True):
            new = _items(now)[now.context_items:]
            assert _items(now) == seen + new and now.is_final == before.is_final
            seen += new
            assert new == _items(before), f"a {tokens}-token stream's chunk would release other tokens than the ramp's"


# S3Gen's chunks against the path it took before the port


def _tiny_config(meanflow: bool) -> ChatterboxConfig:
    s3 = S3GenConfig(
        meanflow=meanflow,
        encoder=S3GenEncoderConfig(linear_units=128, num_blocks=1, num_up_blocks=1),
        estimator=S3GenEstimatorConfig(n_blocks=1, num_mid_blocks=1, attention_head_dim=32, num_heads=2,
                                       channels=(64,)),
        cfm=S3GenCFMConfig(n_timesteps=2),
        hift=S3GenHiFTConfig(base_channels=32, f0_cond_channels=32),
    )
    return ChatterboxConfig(s3gen=s3, t3=T3Config(max_speech_tokens=400), trailing_silence_tokens=3 if meanflow else 0)


def _reference(seed: int, tokens: int) -> ReferenceConditioning:
    gen = torch.Generator().manual_seed(seed)
    return ReferenceConditioning(
        prompt_tokens=torch.randint(0, 6561, (1, tokens), generator=gen),
        prompt_feat=torch.randn(1, 2 * tokens, 80, generator=gen) * 0.1, embedding=torch.randn(1, 192, generator=gen),
    )


@dataclass
class _StreamState:
    """`StreamState` before the port: one request's progress through chunked synthesis."""

    tokens: torch.Tensor
    token_offset: int = 0
    noise: torch.Tensor | None = None
    hift_mel: torch.Tensor | None = None
    hift_source: torch.Tensor | None = None
    hift_speech: torch.Tensor | None = None
    chunks_emitted: int = 0
    done: bool = False


@dataclass
class _Plan:
    state: _StreamState
    generator: torch.Generator
    is_final: bool
    usable: int = 0
    window_start: int = 0
    row: FlowRow | None = None
    ready: torch.Tensor | None = None


class _BeforeThePort:
    """`S3GenSubmodule`'s synthesis before the port, each stream and its generator in a dict as `request_state` held
    them, a chunk's new tokens in. ``sub`` lends the helpers the port left as they were."""

    def __init__(self, sub: S3GenSubmodule):
        self.sub = sub
        self.streams: dict[int, tuple[_StreamState, torch.Generator]] = {}

    def _noise_field(self, state, ref, generator):
        sub = self.sub
        if state.noise is None:
            prompt_frames = ref.num_prompt_tokens * sub.frames_per_token
            max_tokens = sub.config.t3.max_speech_tokens + sub.config.trailing_silence_tokens
            frames = prompt_frames + max_tokens * sub.frames_per_token
            state.noise = torch.randn(
                (1, sub.config.s3gen.output_size, frames), dtype=sub.s3gen.dtype, generator=generator,
            )
        return state.noise

    def _plan_chunk(self, state, new_tokens, is_final, ref, generator):
        sub = self.sub
        plan = _Plan(state=state, generator=generator, is_final=is_final)
        if state.done:
            plan.ready = torch.zeros(0)
            return plan
        state.tokens = torch.cat([state.tokens, new_tokens])
        if is_final:
            state.tokens = sub._with_trailing_silence(state.tokens)
        n = state.tokens.numel()
        usable = n if is_final else max(n - sub.lookahead_tokens, 0)
        if usable <= state.token_offset:
            if not is_final:
                plan.ready = torch.zeros(0)
                return plan
            state.done = True
            tail = state.hift_speech
            plan.ready = tail[0] if tail is not None else torch.zeros(0)
            return plan
        plan.usable = usable
        whole = is_final and state.token_offset == 0 and state.noise is None
        if whole:
            # the frames the encoder keeps for the prompt and the utterance, as the old solve drew them
            full = torch.cat([ref.prompt_tokens[0], state.tokens.reshape(-1)])
            _, masks = sub.s3gen._run_encoder(full[None], torch.tensor([full.numel()]))
            mel_prompt = ref.num_prompt_tokens * sub.s3gen.config.token_mel_ratio
            valid = int(masks.sum())
            assert valid - mel_prompt == n * sub.frames_per_token, (
                "the port would draw the noise for other frames than the encoder keeps"
            )
            noise = sub.s3gen.draw_noise(1, mel_prompt, valid - mel_prompt, generator)
            plan.row = FlowRow(tokens=state.tokens, ref=ref, finalize=True, noise=noise)
            return plan
        start = 0
        if sub.context_tokens > 0:
            start = max(state.token_offset - sub.context_tokens, 0)
        plan.window_start = start
        noise = self._noise_field(state, ref, generator)
        if start > 0:
            prompt_frames = ref.num_prompt_tokens * sub.frames_per_token
            gen = noise[:, :, prompt_frames + start * sub.frames_per_token : prompt_frames + n * sub.frames_per_token]
            noise = torch.cat([noise[:, :, :prompt_frames], gen], dim=2)
        plan.row = FlowRow(tokens=state.tokens[start:], ref=ref, finalize=is_final, noise=noise)
        return plan

    def _emit_chunk(self, plan, mel):
        sub, state = self.sub, plan.state
        new_mel = mel[:, :, (state.token_offset - plan.window_start) * sub.frames_per_token:]
        state.token_offset = plan.usable
        first = state.hift_mel is None
        mel_in = new_mel if first else torch.cat([state.hift_mel, new_mel], dim=2)
        wav, source = sub.s3gen.vocode(mel_in, generator=plan.generator, cache_source=state.hift_source, fade_in=first)
        if not first:
            n_fade = sub.cache_samples
            wav[:, :n_fade] = (
                wav[:, :n_fade] * sub.fade_window[:n_fade] + state.hift_speech[:, -n_fade:] * sub.fade_window[n_fade:]
            )
        if plan.is_final:
            state.done = True
            state.hift_mel = state.hift_source = state.hift_speech = None
            emitted = wav
        else:
            state.hift_mel = mel_in[:, :, -sub.cache_frames:]
            state.hift_source = source[:, :, -sub.cache_samples:]
            state.hift_speech = wav[:, -sub.cache_samples:]
            emitted = wav[:, : -sub.cache_samples]
        state.chunks_emitted += 1
        return emitted[0]

    @torch.no_grad()
    def step(self, chunks: dict) -> dict:
        """One step of `forward_batched`: ``chunks`` maps a rid to its seed, new tokens, final flag and reference."""
        plans = []
        for rid, (seed, tokens, is_final, ref) in chunks.items():
            if rid not in self.streams:
                self.streams[rid] = (_StreamState(tokens=torch.empty(0, dtype=torch.long)),
                                     torch.Generator().manual_seed(seed))
            state, generator = self.streams[rid]
            speech = tokens[tokens < self.sub.config.s3gen.vocab_size]
            plans.append(self._plan_chunk(state, speech, is_final, ref, generator))
        outputs = [plan.ready for plan in plans]
        pending = [i for i, plan in enumerate(plans) if plan.row is not None]
        if pending:
            mels = self.sub.s3gen.tokens_to_mel_rows(
                [plans[i].row for i in pending], n_timesteps=2, frame_bucket=self.sub.frame_bucket,
            )
            for i, mel in zip(pending, mels, strict=True):
                outputs[i] = self._emit_chunk(plans[i], mel)
        return {rid: self.sub._finish(wav, False) for rid, wav in zip(chunks, outputs, strict=True)}


class _Both:
    """The ported node and the path before the port over one tiny random-weight S3Gen, the node's streams in a stub
    of their Value. Built-in voice ``_reference(0, 12)``; voice key 1 and 2 are clips."""

    def __init__(self, meanflow: bool = False):
        torch.manual_seed(0)
        config = _tiny_config(meanflow)
        self.voices = {1: _reference(1, 31), 2: _reference(2, 50)}
        self.node = S3GenSubmodule(S3Gen(config.s3gen).eval(), None, config, builtin_voice=_reference(0, 12))
        self.node.condition = lambda wav: self.voices[int(wav.numel())]
        _bind_stream(self.node)
        self.before = _BeforeThePort(self.node)
        self.histories: dict[int, torch.Tensor] = {}

    def step(self, chunks: dict, voices: dict | None = None) -> tuple[dict, dict]:
        """``chunks`` maps a rid to its new tokens and final flag, and ``voices`` a rid to the voice key whose clip
        rides with its first chunk. Each side's PCM by rid."""
        voices = voices or {}
        prepared, before = {}, {}
        for rid, (tokens, is_final) in chunks.items():
            first = rid not in self.histories
            delivered = self.histories.get(rid, torch.empty(0, dtype=torch.long))
            history = self.histories[rid] = torch.cat([delivered, tokens])
            seed = 1000 + rid
            edges = {}
            if rid in voices:
                key = voices[rid]
                clip = [torch.zeros(key)] if first else []
                edges = {REF_AUDIO: clip, VOICE_KEY: [torch.tensor([key])] if first else []}
            prepared[rid] = _prepare(self.node, rid, history, is_final, seed=seed, edges=edges)
            ref = self.voices[voices[rid]] if rid in voices else self.node.builtin_voice
            before[rid] = (seed, tokens, is_final, ref)
        return _s3gen_step(self.node, prepared), self.before.step(before)


def _prepare(sub, rid, history, is_final, seed=7, edges=None):
    fields = sub.stream.host_fields
    if not fields.is_open(rid):
        fields.open(rid)
    info = SimpleNamespace(request_id=f"wire-{rid}", rid_handle=rid, random_seed=seed,
                           step_metadata={"n_cfm_timesteps": 2, "watermark": False})
    return sub.prepare_inputs("s3gen_chunk", info, {SPEECH_TOKENS: [history], **(edges or {})},
                              is_final_stream_chunk=is_final)


def _assert_same_audio(ported: dict, before: dict, what: str) -> None:
    for rid, pcm in before.items():
        assert ported[rid].shape == pcm.shape and torch.equal(ported[rid], pcm), (
            f"{what}: request {rid}'s audio differs from the path before the port"
        )


@pytest.mark.parametrize("meanflow", [False, True])
def test_a_streamed_request_gives_its_audio_with_the_noise_field_drawn_again_each_solve(meanflow):
    both = _Both(meanflow)
    tokens = torch.randint(0, 6561, (130,), generator=torch.Generator().manual_seed(3))
    tokens[[20, 70]] = 6562
    for start, end, is_final in ((0, 15, False), (15, 65, False), (65, 66, False), (66, 130, True)):
        ported, before = both.step({4: (tokens[start:end], is_final)})
        _assert_same_audio(ported, before, f"tokens {start} to {end}")


@pytest.mark.parametrize("meanflow", [False, True])
def test_a_final_chunk_arriving_alone_gives_the_offline_audio(meanflow):
    both = _Both(meanflow)
    tokens = torch.randint(0, 6561, (90,), generator=torch.Generator().manual_seed(4))
    ported, before = both.step({4: (tokens, True)})
    _assert_same_audio(ported, before, "the whole utterance")


@pytest.mark.parametrize("meanflow", [False, True])
def test_a_chunk_prepared_again_after_a_refused_admit_draws_the_same_noise(meanflow):
    """A refused admit drops a prepared row, and a later step prepares it again. Each prepare draws from the generator
    state the last committed step left, so the step that runs draws what the dropped prepare drew."""
    both = _Both(meanflow)
    tokens = torch.randint(0, 6561, (130,), generator=torch.Generator().manual_seed(3))
    for start, end, is_final in ((0, 15, False), (15, 65, False), (65, 130, True)):
        history = torch.cat([both.histories.get(4, torch.empty(0, dtype=torch.long)), tokens[start:end]])
        _prepare(both.node, 4, history, is_final, seed=1004)
        ported, before = both.step({4: (tokens[start:end], is_final)})
        _assert_same_audio(ported, before, f"tokens {start} to {end}, prepared twice")


def test_a_call_after_the_final_chunk_emits_nothing_and_draws_nothing():
    both = _Both()
    tokens = torch.randint(0, 6561, (40,), generator=torch.Generator().manual_seed(5))
    both.step({4: (tokens[:20], False)})
    both.step({4: (tokens[20:], True)})
    held = host(both.node.stream, 4, "main", 0)
    ported, before = both.step({4: (torch.tensor([9]), True)})
    _assert_same_audio(ported, before, "the call after the end")
    assert ported[4].numel() == 0 and host(both.node.stream, 4, "main", 0) == held, (
        "a stream that ended moved its state on a later call"
    )


def test_an_uploaded_voice_is_held_for_the_chunks_without_its_clip():
    both = _Both()
    tokens = torch.randint(0, 6561, (100,), generator=torch.Generator().manual_seed(6))
    for start, end, is_final in ((0, 15, False), (15, 65, False), (65, 100, True)):
        ported, before = both.step({4: (tokens[start:end], is_final)}, voices={4: 2})
        _assert_same_audio(ported, before, f"tokens {start} to {end} in the uploaded voice")


def test_two_requests_in_one_padded_solve_give_their_own_audio():
    both = _Both()
    tokens = torch.randint(0, 6561, (2, 90), generator=torch.Generator().manual_seed(7))
    steps = (
        {1: (tokens[0, :15], False), 2: (tokens[1, :35], False)},
        {1: (tokens[0, 15:65], False), 2: (tokens[1, 35:90], True)},
        {1: (tokens[0, 65:90], True), 2: (tokens[1, :0], True)},
    )
    for i, chunks in enumerate(steps):
        ported, before = both.step(chunks, voices={1: 1})
        _assert_same_audio(ported, before, f"step {i}")


def test_the_store_is_empty_after_a_removal_and_a_recycled_handle_starts_clean():
    both = _Both()
    tokens = torch.randint(0, 6561, (60,), generator=torch.Generator().manual_seed(8))
    both.step({4: (tokens[:30], False)}, voices={4: 1})
    fields = both.node.stream.host_fields
    fields.remove_request(4)
    both.node.cleanup_request(4)
    assert not fields._fields and not both.node.request_states
    # the handle comes back for another request, over the entry the last one left in the Value's stub
    del both.histories[4], both.before.streams[4]
    ported, before = both.step({4: (tokens[30:], True)})
    assert torch.equal(ported[4], before[4]), "a recycled handle would read the last request's voice or tail"


# The reference the stream holds


@pytest.mark.parametrize("seconds", [10.0, 12.0])
def test_the_stream_holds_every_reference_token_of_a_clip_at_least_dec_cond_seconds_long(seconds):
    config = ChatterboxConfig.chatterbox()
    wav24 = torch.randn(round(seconds * config.sample_rate), generator=torch.Generator().manual_seed(0)) * 0.1
    wav24 = trim_reference(wav24, config.sample_rate, config.dec_cond_seconds)
    with torch.no_grad():
        _, lens = S3Tokenizer(config.s3_tokenizer).eval()([resample(wav24, config.sample_rate, S3_SR)])
        frames = MatchaLogMel(**asdict(config.s3gen.mel))(wav24[None]).shape[-1]
    # `S3Gen.embed_reference`'s cut
    kept = min(int(lens[0]), frames // config.s3gen.token_mel_ratio)
    assert kept == max_reference_tokens(config) == 250, "the stream's reference fields would cut or pad a full clip"


def test_a_reference_past_the_bound_is_refused_naming_it():
    sub, _ = _s3_submodule()
    sub.condition = lambda wav: SimpleNamespace(num_prompt_tokens=251)
    edges = {REF_AUDIO: [torch.zeros(10)], VOICE_KEY: [torch.tensor([3])]}
    with pytest.raises(ValueError, match="the reference has 251 S3 tokens, past the 250 its stream holds"):
        _prepare(sub, 1, torch.arange(1, 5), False, edges=edges)
