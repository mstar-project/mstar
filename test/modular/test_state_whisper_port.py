"""Whisper plans on the contract what it planned by hand, and its align walk forces the transcript it forced before.

Whisper's transcripts and word timings compare token for token, so a context planned under another label, or a decode
step that loses its rule state, moves them with no error. The hand-written specs, steps and layouts below are the
reference, copied from the model as it stood before the port, so the adapter can't pass by agreeing with itself. Each
states where the port differs, and why.

Before the port, the stop check recorded the transcript on the host: the prompt walk's token, then each decode step's up
to the one that stopped the loop. Now every walk writes its token into the decoder state, and the align walk cuts that
row where the recording stopped. A cut one slot off would time words the client never got, or drop ones it did, and
nothing would fail.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
import yaml
from test_state_ports import _assert_equal_per_key, _assert_steps_equal, _bind, _published
from test_whisper_decoder_node import _Driver
from test_whisper_model import EOT, NOTS, SOT, TRANSCRIBE, _make_model, _tiny_config

from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    CrossAttentionConfig,
    CrossAttentionSpec,
)
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig, Span
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, SubmoduleStep
from mstar.engine.state.adapters import kv_layouts, node_resources, sampling_configs
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.base import LabelLayout, serving_resources
from mstar.model.submodule_base import ARNodeInputs, engine_step
from mstar.model.utils import kv_pages
from mstar.model.whisper.components.timestamps import inactive_state, rule_state
from mstar.model.whisper.config import (
    ALIGN_WALK,
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    DECODE_LOOP,
    DECODE_WALK,
    DECODER_NODE,
    DECODER_STATE,
    DETECT_LANGUAGE_WALK,
    KV_CACHE,
    PREFILL_PROMPT_WALK,
    PREFILL_WALK,
    SAMPLER,
    WhisperModelConfig,
)
from mstar.model.whisper.submodules import WhisperDecoderSubmodule
from mstar.model.whisper.whisper_model import WhisperModel

ROPE = "rope"
# The decoder of each published model, by its registry key: large-v3's, and turbo's with 4 layers.
GEOMETRY = {"whisper_large": {}, "whisper_large_v3_turbo": {"decoder_layers": 4}}
# A `resources:` block a deployment could write, each key one a spec's YAML takes.
SYNTHETIC = {
    KV_CACHE: {"page_size": 64, "max_num_pages": 33, "cpu_offload_pages": 8},
    CROSS_KV_CACHE: {"page_size": 64, "max_num_pages": 97, "cpu_offload_pages": 8},
    ATTN: {"decode_tensor_cores": True},
    CROSS_ATTN: {"decode_tensor_cores": True},
    DECODER_STATE: {"max_slots": 9},
}


def _model(name: str = "whisper_large") -> WhisperModel:
    model = object.__new__(WhisperModel)
    model.config = WhisperModelConfig(**GEOMETRY[name])
    return model


def _reference(model: WhisperModel) -> list:
    c = model.config
    nodes = {DECODER_NODE}

    def pool(max_seq_len: int) -> PagedKVConfig:
        # The model set page_size=128, the engine's default page: see the test on its page and arena.
        return PagedKVConfig(
            num_layers=c.decoder_layers, num_kv_heads=c.decoder_attention_heads, head_dim=c.head_dim,
            max_seq_len=max_seq_len, num_qo_heads=c.decoder_attention_heads,
            max_num_pages=kv_pages(max_seq_len, 128, model.MAX_CONCURRENT_REQUESTS),
        )

    return [
        KVSpec(resource_key=KV_CACHE, nodes=nodes, config=pool(c.max_target_positions)),
        AttentionSpec(
            resource_key=ATTN, nodes=nodes, config=AttentionConfig(kv_cache=KV_CACHE, decode_tensor_cores=False),
        ),
        KVSpec(resource_key=CROSS_KV_CACHE, nodes=nodes, config=pool(c.max_source_positions)),
        CrossAttentionSpec(
            resource_key=CROSS_ATTN, nodes=nodes,
            config=CrossAttentionConfig(
                kv_cache=CROSS_KV_CACHE, query_kv_cache=KV_CACHE, context_label=CONTEXT_LABEL,
                decode_tensor_cores=False,
            ),
        ),
        # Keyed `positions` by the model. `position_key` names an array's counter, and gives `rope` to this one. No
        # published config names either key, which the per-config test would refuse.
        PositionSpec(resource_key=ROPE, nodes=nodes, config=PositionConfig(kv_cache=KV_CACHE)),
        # The model also declared `loopback_state={"ts_rules": ...}`, the decode loop's rule state kept by sampler
        # slot, which is a field of the decoder state now.
        SamplerSpec(resource_key=SAMPLER, nodes=nodes, vocab_size=c.vocab_size, enable_repetion_penalty=False),
        # New: the rule state, and the transcript the stop check recorded in the node's request state.
        RecurrentStateSpec(resource_key=DECODER_STATE, nodes=nodes, config=RecurrentStateConfig(
            num_layers=1, labels=("main",), blocks={
                "ts_rules": RecurrentBlockConfig(shape=(4,), dtype=torch.long),
                "transcript": RecurrentBlockConfig(shape=(c.max_target_positions + 1,), dtype=torch.long),
            },
        )),
    ]


def test_the_published_configs_are_found():
    assert _published("WhisperModel"), "an empty list would skip their tests"


@pytest.mark.parametrize("name", sorted(GEOMETRY))
def test_specs_equal_the_hand_written_ones(name):
    model = _model(name)
    _assert_equal_per_key(node_resources(model), _reference(model))


@pytest.mark.parametrize("synthetic", [False, True], ids=["as_published", "with_a_synthetic_resources_block"])
@pytest.mark.parametrize("path", _published("WhisperModel"), ids=lambda path: path.stem)
def test_specs_equal_the_hand_written_ones_on_every_published_config(path, synthetic):
    deployment = yaml.safe_load(path.read_text())
    if synthetic:
        deployment["resources"] = SYNTHETIC
    model = _model(deployment["model"])
    reference = _reference(model)
    apply_yaml_overrides(reference, deployment)
    specs = serving_resources(model, deployment)
    _assert_equal_per_key(specs, reference)
    by_key = {spec.resource_key: spec for spec in specs}
    reached = (by_key[CROSS_KV_CACHE].config.page_size, by_key[DECODER_STATE].config.max_slots)
    assert not synthetic or reached == (64, 9), "the YAML should still reach the pools"


def test_the_pools_leave_their_page_size_unset_with_the_page_and_arena_the_model_set():
    """With no window a page of None and of 128 are one page, so dropping the argument moves nothing."""
    for spec in node_resources(_model()):
        if isinstance(spec, KVSpec):
            written = replace(spec.config, page_size=128)
            assert spec.config.tokens_per_page == written.tokens_per_page == 128
            assert spec.config.num_units == written.num_units, f"{spec.resource_key} has another arena"


@pytest.mark.parametrize("name", sorted(GEOMETRY))
def test_the_declaration_has_no_load_problems(name):
    assert state_declaration_problems(_model(name)) == []


# Steps


def _node(model: WhisperModel) -> WhisperDecoderSubmodule:
    node = WhisperDecoderSubmodule(None, model.config)
    # a parameter for the device and dtype the capture templates read
    node.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1)))
    return _bind(node, model, DECODER_NODE)


def _reference_step(walk: str, spans, context) -> SubmoduleStep:
    """What the model's `declare_step` returned on ``walk`` before the port, for rows 0, 1, ... of ``spans`` tokens and
    ``context`` encoder positions."""
    contexts = tuple(Segment(rid, CONTEXT_LABEL, span) for rid, span in enumerate(context))
    return SubmoduleStep(segments=[Segment(rid, "main", span) for rid, span in enumerate(spans)], steps={
        KV_CACHE: KVStep(),
        ATTN: AttentionStep(causal=True),
        CROSS_KV_CACHE: KVStep(segments=contexts, eager_write=walk in (PREFILL_WALK, DETECT_LANGUAGE_WALK)),
        CROSS_ATTN: AttentionStep(segments=contexts, causal=False),
        # The model's decode step also named `loopback_writes={"ts_rules"}`, the forward's rule state stored by sampler
        # slot, which the decoder state holds now. Detection takes an argmax and draws nothing, so it keeps no row's
        # sample, where the model's step kept every row, a commit that does nothing on its eager step.
        SAMPLER: SamplerStep(apply_penalty=False, kept_rids=frozenset() if walk == DETECT_LANGUAGE_WALK else None),
        # keyed `positions` by the model, as its spec was
        ROPE: PositionStep(),
        # new: every walk reads the rule state and the transcript, and writes them back with its token
        DECODER_STATE: RecurrentStep(),
    })


def _rows(spans, context) -> list[ARNodeInputs]:
    """Rows of ``spans`` tokens, each carrying ``context`` encoder positions, of which only the length is read."""
    return [
        ARNodeInputs(
            input_ids=torch.zeros(span, dtype=torch.long), input_seq_len=span,
            tensor_inputs={"encoder_states": torch.zeros(positions, 4)} if positions else {},
        )
        for span, positions in zip(spans, context, strict=True)
    ]


@pytest.mark.parametrize("walk, spans, context", [
    # a forced prompt, and one carrying the previous window's text as <|startofprev|> context
    (PREFILL_WALK, (4, 9), (1500, 1500)),
    (DETECT_LANGUAGE_WALK, (1, 1), (1500, 1500)),
    # the detected language and the rest of the forced prompt
    (PREFILL_PROMPT_WALK, (3, 2), (0, 0)),
    (DECODE_WALK, (1, 1), (0, 0)),
])
def test_an_eager_step_equals_the_hand_written_one(walk, spans, context):
    model = _model()
    step = engine_step(_node(model), graph_walk=walk, request_ids=[0, 1], inputs=_rows(spans, context))
    _assert_steps_equal(step, _reference_step(walk, spans, context))


@pytest.mark.parametrize("walk, bs, tokens", [(PREFILL_WALK, 2, 8), (DECODE_WALK, 4, 4)])
def test_a_captured_step_equals_the_hand_written_one(walk, bs, tokens):
    """At capture the runner declares the bucket's template rows under a slot lease, as `_warmed` does."""
    model = _model()
    node = _node(model)
    (config,) = [c for c in node.get_accelerator_graph_configs(torch.device("cpu")) if c.capture_graph_walk == walk]
    inputs = config.get_node_inputs(bs, tokens)
    lease = SlotLease(slot=0, bucket=BucketKey(walk, bs, tokens))
    step = engine_step(node, graph_walk=walk, request_ids=list(range(bs)), inputs=inputs, slot_lease=lease)
    spans = [inp.input_seq_len for inp in inputs]
    context = [model.config.max_source_positions if walk == PREFILL_WALK else 0] * bs
    _assert_steps_equal(step, _reference_step(walk, spans, context))


def test_the_align_step_lists_only_the_decoder_state_at_its_one_entry():
    model = _model()
    node = _node(model)
    info = SimpleNamespace(step_metadata={}, resource_configs={SAMPLER: SamplingReqConfig()})
    row = node.prepare_inputs(ALIGN_WALK, info, {"encoder_states": [torch.zeros(1500, 4)]})
    step = engine_step(node, graph_walk=ALIGN_WALK, request_ids=[0], inputs=[row])
    # The model listed nothing, and spanned the teacher-forced sequence's rows for a LoRA pool, which Whisper doesn't
    # serve. The walk now reads the transcript off the decoder state and writes it back.
    _assert_steps_equal(step, SubmoduleStep(segments=[Segment(0, "main", 1)], steps={DECODER_STATE: RecurrentStep()}))


# Layouts and request settings


@pytest.mark.parametrize("kwargs", [
    {"language": "en"},
    {},
    {"language": "en", "initial_prompt": "so far"},
    {"initial_prompt": "so far", "timestamps": True},
], ids=["forced", "detected", "forced_with_context", "detected_with_context_and_timestamps"])
def test_the_layout_converts_to_the_hand_written_one(kwargs):
    model = _make_model()
    out = model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": [torch.zeros(8000)]}, **kwargs)
    prompt = len(out.new_input_tensors["text_inputs"][0])
    if "language" in kwargs:
        spans, encoder_walk = (Span("ids", PREFILL_WALK, prompt, prompt, "text_inputs"),), PREFILL_WALK
    else:
        tail = 1 + len(out.new_input_tensors["prompt_tail"][0])
        spans = (
            Span("ids", DETECT_LANGUAGE_WALK, prompt, prompt, "text_inputs"),
            Span("rows", PREFILL_PROMPT_WALK, tail, tail, "prompt_tail"),
        )
        encoder_walk = DETECT_LANGUAGE_WALK
    context = model.config.max_source_positions
    assert kv_layouts(model.declare_state(), out.metadata["state_layout"], {}) == {
        KV_CACHE: {"main": LabelLayout(spans=spans, growth=((DECODE_LOOP, 1),))},
        CROSS_KV_CACHE: {CONTEXT_LABEL: LabelLayout(
            spans=(Span("rows", encoder_walk, context, context, "encoder_states"),),
        )},
    }, "the request would reserve and commit other pages than it did"


@pytest.mark.parametrize("kwargs, expected", [
    ({}, SamplingReqConfig(temperature=0.0, top_p=1.0, ignore_eos=False)),
    (
        {"temperature": "0.4", "top_p": 0.9, "ignore_eos": True},
        SamplingReqConfig(temperature=0.4, top_p=0.9, ignore_eos=True),
    ),
])
def test_sample_settings_convert_to_the_hand_written_ones(kwargs, expected):
    model = _model()
    settings = sampling_configs(model.get_request_resource_configs({}, kwargs), model.declare_state())
    assert settings == {SAMPLER: expected}, "the request would sample with other settings than it did"


# The align walk's transcript

# A short position table, so a request runs to its cap in a few steps. A word-timestamp prompt is 3 tokens on both
# paths, <|sot|><|en|><|transcribe|>, or <|sot|> then the detected language and <|transcribe|>.
MAXPOS = 24
TIMESTAMP = 50365


def _forced(language: int, first: int, decoded: list[int], ignore_eos: bool) -> list[int]:
    """The ids the align walk forced before the port. The stop check recorded the prompt walk's token, then each decode
    step's up to the one whose end-of-text stopped the loop, and the walk kept the tokens below end-of-text that fit."""
    recorded = [first]
    for token in decoded:
        recorded.append(token)
        if token == EOT and not ignore_eos:
            break
    text = [token for token in recorded if token < EOT][: MAXPOS - 5]
    return [SOT, language, TRANSCRIBE, NOTS, *text, EOT]


class _WordRequest:
    """Word-timestamp request ``rid`` on ``driver``, routed as the conductor routes it."""

    def __init__(self, driver: _Driver, rid: int, **kwargs):
        self.driver, self.rid = driver, rid
        self.model = model = _make_model(driver.node.config)
        kwargs = {"timestamps": "word", **kwargs}
        prompt = model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": [torch.zeros(8000)]}, **kwargs)
        self.tensors = prompt.new_input_tensors
        self.sampling = sampling_configs(model.get_request_resource_configs({}, kwargs), model.declare_state())
        signals = {name: [object()] for name in self.tensors}
        self.route = model.get_initial_forward_pass_args(
            "default", ["audio"], ["text"], signals, {**kwargs, **prompt.metadata},
        )
        config = driver.node.config
        self.encoder = {"encoder_states": [torch.zeros(config.max_source_positions, config.d_model)]}

    def info(self) -> SimpleNamespace:
        return SimpleNamespace(step_metadata=self.route.step_metadata, resource_configs=self.sampling)

    def next_walk(self) -> None:
        self.route = self.model.get_partition_forward_pass_args("default", self.route.full_metadata, {})

    def step(self, walk: str, inputs: dict):
        return self.driver.step(walk, {self.rid: (self.info(), inputs)})


def _serve(
    driver: _Driver, rid: int, first: int, decoded: list[int], *, language: str | None = "en",
    ignore_eos: bool = False, initial_prompt: str | None = None, after_step: Callable[[str], None] | None = None,
) -> tuple[list[int], int]:
    """Serve word-timestamp request ``rid`` through its walks, the prompt walk sampling ``first`` and each decode step
    the next of ``decoded``. Every step commits, a speculative one past a stop too, and then calls ``after_step`` with
    its walk. Returns the align walk's teacher-forced ids, and the request's language token."""
    request = _WordRequest(driver, rid, language=language, initial_prompt=initial_prompt, ignore_eos=ignore_eos)
    tensors, encoder = request.tensors, request.encoder
    driver.sampler.tokens = [first, *decoded]

    def step(walk: str, inputs: dict) -> dict:
        out = request.step(walk, inputs)
        if after_step is not None:
            after_step(walk)
        return out

    if language is None:
        out = step(DETECT_LANGUAGE_WALK, {"text_inputs": tensors["text_inputs"], **encoder})
        detected = out.row_outputs["new_token"]
        request.next_walk()
        step(PREFILL_PROMPT_WALK, {"text_inputs": [detected], "prompt_tail": tensors["prompt_tail"]})
        language_token = int(detected)
    else:
        step(PREFILL_WALK, {"text_inputs": tensors["text_inputs"], **encoder})
        language_token = driver.node.config.language_token(language)
    request.next_walk()
    for token in [first, *decoded[:-1]]:
        step(DECODE_WALK, {"text_inputs": [torch.tensor([token])]})
    request.next_walk()
    step(ALIGN_WALK, encoder)
    return driver.batch["input_ids"].tolist(), language_token


@pytest.mark.parametrize("language", ["en", None], ids=["forced", "detected"])
@pytest.mark.parametrize("first, decoded, ignore_eos", [
    # a speculative step ran past the step that stopped the loop
    (10, [11, 12, EOT, 13], False),
    # every position of the table holds a token, more text than fits beside the forced prompt
    (7, [8] * (MAXPOS - 3), False),
    (10, [11, EOT, 12, 13], True),
    (EOT, [11, 12, EOT, 13], False),
], ids=["stopped_by_end_of_text", "stopped_by_its_cap", "past_an_end_of_text_under_ignore_eos",
        "after_an_end_of_text_from_the_prompt_walk"])
def test_the_align_walk_forces_the_transcript_the_stop_check_recorded(language, first, decoded, ignore_eos):
    got, token = _serve(_Driver(_tiny_config(max_target_positions=MAXPOS)), 0, first, decoded, language=language,
                        ignore_eos=ignore_eos)
    assert got == _forced(token, first, decoded, ignore_eos), "the align walk would time other words than it did"


def test_the_align_walk_forces_the_transcript_after_a_prompt_carried_over_as_context():
    got, token = _serve(_Driver(_tiny_config(max_target_positions=MAXPOS)), 0, 10, [11, EOT], initial_prompt="so far")
    assert got == _forced(token, 10, [11, EOT], False), "the transcript would start inside the carried-over context"


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["eager_draw", "captured_draw"])
def test_a_token_of_either_drawn_dtype_lands_in_the_transcript(dtype):
    """An eager draw returns int32 tokens and a captured one int64, and the transcript holds int64. Every step that
    draws eagerly writes one: the detected language's prompt walk, which no graph captures, and any eager prefill or
    decode."""
    driver = _Driver(_tiny_config(max_target_positions=MAXPOS))
    driver.sampler.dtype = dtype
    got, token = _serve(driver, 0, 10, [11, EOT], language=None)
    assert got == _forced(token, 10, [11, EOT], False), f"a draw of {dtype} tokens would not reach the transcript"


def _rules_row(driver: _Driver, rid: int) -> list[int]:
    return driver.pool.block("ts_rules", 0)[driver.pool._slots[rid]["main"].index].tolist()


@pytest.mark.parametrize("language", ["en", None], ids=["forced", "detected"])
def test_each_step_carries_the_rule_state_its_tokens_lead_to(language):
    """Word timestamps keep the timestamp rules on, and a decode step takes their state from the decoder state, which
    the step before it advanced by the token it sampled. A step that wrote back the state it read would mask every later
    token as the request's first, and the transcript tests, whose sampler ignores the logits, would still pass."""
    driver = _Driver(_tiny_config(max_target_positions=MAXPOS))
    config = driver.node.config
    first, decoded = 10, [TIMESTAMP + 5, 11, TIMESTAMP + 9, EOT]
    tokens = [first, *decoded]
    walks, rows, received = [], [], {}
    apply = driver.node.timestamp_rules.apply

    def recording_apply(logits, state):
        received[len(walks)] = state[0].tolist()
        return apply(logits, state)

    def read_row(walk: str) -> None:
        walks.append(walk)
        rows.append(_rules_row(driver, 0))

    driver.node.timestamp_rules.apply = recording_apply
    _serve(driver, 0, first, decoded, language=language, after_step=read_row)
    if language is None:
        assert walks[:2] == [DETECT_LANGUAGE_WALK, PREFILL_PROMPT_WALK]
        assert rows[0] == inactive_state(), "language detection decodes with the rules off"
    prompt = walks.index(DECODE_WALK) - 1
    assert rows[prompt] == rule_state(tokens[:1], config), (
        f"{walks[prompt]} would leave the first decode step a state its token didn't advance"
    )
    for k, i in enumerate(i for i, walk in enumerate(walks) if walk == DECODE_WALK):
        assert received[i] == rule_state(tokens[: k + 1], config), f"decode step {k} would mask by another state"
        assert rows[i] == rule_state(tokens[: k + 2], config), (
            f"decode step {k} would leave the next step a state its token didn't advance"
        )


def test_a_step_reads_a_looped_rows_token_off_the_sampler_beside_a_routed_rows_own():
    """A decode row whose last step here kept its token on the device reads it off the sampler, and a row whose last
    step ran elsewhere brings the token that step drew, which `preprocess` splices in. The other tests route every
    token, so neither the read nor the splice ran."""
    driver = _Driver(_tiny_config(max_target_positions=MAXPOS))
    driver.node.node_resources = {SAMPLER: driver.sampler}
    looped, routed = _WordRequest(driver, 0, language="en"), _WordRequest(driver, 1, language="en")
    driver.sampler.tokens = [10, 20, 11, 12, 30]
    for request in (looped, routed):
        request.step(PREFILL_WALK, {"text_inputs": request.tensors["text_inputs"], **request.encoder})
        request.next_walk()
    looped.step(DECODE_WALK, {"text_inputs": [torch.tensor([10])]})
    driver.step(DECODE_WALK, {0: (looped.info(), {}), 1: (routed.info(), {"text_inputs": [torch.tensor([21])]})},
                looped={0})
    assert driver.batch["input_ids"].tolist() == [11, 21], "a row would decode from another row's or a stale token"
    transcript = driver.pool.block("transcript", 0)
    rows = [transcript[driver.pool._slots[rid]["main"].index].tolist() for rid in (0, 1)]
    # the 3-token prompt, then each drawn token at the position it is fed back at
    assert rows[0][3:6] == [10, 11, 12] and rows[1][3:5] == [20, 30], "a token would land in another row's slot"


def test_requests_sharing_the_decoder_state_each_force_their_own_transcript():
    """Random requests through one pool, as the designer's script drew them: either prompt path, a prompt carried over
    as context or not, text and timestamp tokens, an end-of-text the prompt walk or a decode step samples, steps past
    it, and ``ignore_eos``."""
    rng = random.Random(0)
    driver = _Driver(_tiny_config(max_target_positions=MAXPOS))
    for rid in range(100):
        language, initial_prompt = rng.choice(["en", None]), rng.choice([None, "so far"])
        ignore_eos = rng.random() < 0.3
        # the longest prompt is 11 tokens: <|startofprev|>, 7 of context, and 3
        decoded = [rng.choice([rng.randint(0, EOT - 1), rng.randint(TIMESTAMP, TIMESTAMP + 1500)])
                   for _ in range(rng.randint(1, MAXPOS - 11))]
        if rng.random() < 0.7:
            stop = rng.randrange(len(decoded))
            decoded[stop] = EOT
            if not ignore_eos:
                decoded = decoded[: stop + 1 + rng.randint(0, 2)]
        first = EOT if rng.random() < 0.1 else rng.randint(0, EOT - 1)
        got, token = _serve(driver, rid, first, decoded, language=language, ignore_eos=ignore_eos,
                            initial_prompt=initial_prompt)
        assert got == _forced(token, first, decoded, ignore_eos), (
            f"request {rid} ({language=}, {initial_prompt=}, {ignore_eos=}, {first=}, {decoded=}) would time other "
            "words than it did"
        )
