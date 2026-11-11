"""The shared VLM scaffolds: the media walk set in ``multimodal``, the
`SampledTokenDecoder` LLM node and the `PackedBlockLoopEncoder` vision node.

No checkpoint: the pad expansion, the splice and the prefill walk run on small
ids, and the nodes on stubs. The per-model files test each model's own pieces.
"""
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import FINAL_PREFILL, WalkStep
from mstar.graph.base import Parallel
from mstar.model.components.sampled_token_decoder import MEDIA_EMBEDS, MEDIA_MASK
from mstar.model.keys import KV_CACHE, MIXED, ROPE, SAMPLER
from mstar.model.multimodal import (
    MEDIA_LAYOUT,
    MediaEncoder,
    MediaRun,
    expand_media_pads,
    media_layout,
    media_prefill_step,
    media_walk_names,
    media_walks,
    vision_encoder,
)

IMAGE_PAD, VIDEO_PAD, AUDIO_PAD = 90, 91, 80
VISION = vision_encoder()
AUDIO = MediaEncoder("audio", "audio_encoder", ("audio_values",), "audio_embeds")


# ---------------------------------------------------------------------------
# Pad expansion
# ---------------------------------------------------------------------------


def test_each_pad_expands_to_its_tokens_in_prompt_order():
    ids = torch.tensor([1, IMAGE_PAD, 2, AUDIO_PAD, IMAGE_PAD])
    pads = expand_media_pads(ids, {
        IMAGE_PAD: [MediaRun(2, "vision_embeds", torch.tensor([1, 4, 4])),
                    MediaRun(1, "vision_embeds", torch.tensor([1, 2, 2]))],
        AUDIO_PAD: [MediaRun(3, "audio_embeds")],
    }, sentinels={AUDIO_PAD: (70, 71)})
    assert pads.ids.tolist() == [1, 90, 90, 2, 70, 80, 80, 80, 71, 90]
    assert media_layout(pads) == {
        "offset": [1, 5, 9], "tokens": [2, 3, 1],
        "output": ["vision_embeds", "audio_embeds", "vision_embeds"],
        "grid": [[1, 4, 4], [], [1, 2, 2]],
    }


@pytest.mark.parametrize("ids", [[IMAGE_PAD, IMAGE_PAD], []])
def test_a_pad_without_a_count_or_a_count_without_a_pad_is_refused(ids):
    # user text that spells a placeholder; an attachment the prompt never placed
    with pytest.raises(ValueError, match="placeholder"):
        expand_media_pads(torch.tensor(ids, dtype=torch.long), {IMAGE_PAD: [MediaRun(1, "vision_embeds")]})


# ---------------------------------------------------------------------------
# The splice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compiled", [False, True])
def test_the_splice_fills_each_pad_with_the_next_media_row(compiled):
    """Also traced whole, as inside a compiled forward: no host sync or graph break."""
    from mstar.model.multimodal import splice_media

    splice = torch.compile(splice_media, fullgraph=True, backend="aot_eager") if compiled else splice_media
    ids = torch.tensor([1, IMAGE_PAD, IMAGE_PAD, 2, VIDEO_PAD, 3])
    media = torch.tensor([[100.0], [101.0], [102.0]])
    out = splice(ids.float()[:, None], (ids == IMAGE_PAD) | (ids == VIDEO_PAD), media)
    assert out[:, 0].tolist() == [1, 100, 101, 2, 102, 3]


def test_a_replay_tail_cannot_shift_a_real_row():
    """Real tokens precede the tail, so stale pads there index past the real
    media without moving a real row's index; the buffer holds a row per
    token, so their index stays in bounds."""
    from mstar.model.multimodal import splice_media

    real = [5, IMAGE_PAD, 6]
    ids = torch.tensor(real + [IMAGE_PAD, IMAGE_PAD, IMAGE_PAD])  # a previous replay's tail
    media = torch.zeros(len(ids), 1)
    media[0] = 100.0
    out = splice_media(ids.float()[:, None], ids == IMAGE_PAD, media)
    assert out[:3, 0].tolist() == [5, 100, 6]


# ---------------------------------------------------------------------------
# Walks, prefill and forward-pass args
# ---------------------------------------------------------------------------


def test_one_media_walk_per_subset_of_encoders():
    assert media_walk_names([VISION]) == ["prefill_vision"]
    assert media_walk_names([VISION, AUDIO]) == ["prefill_vision", "prefill_audio", "prefill_vision_audio"]
    walks = media_walks(8, [VISION, AUDIO])
    assert set(walks) == {"prefill_text", "prefill_vision", "prefill_audio", "prefill_vision_audio", "decode"}
    both = walks["prefill_vision_audio"]
    assert isinstance(both.sections[0], Parallel)
    llm = both.sections[1]
    assert set(llm.input_names) == {"text_inputs", "vision_embeds", "audio_embeds"}


def _signals(**counts: int) -> dict[str, list[str]]:
    # the conductor passes tensor handles; their identity is all that matters
    return {name: [f"{name}{i}" for i in range(n)] for name, n in counts.items()}


def test_a_text_prompt_is_one_prefill_text_walk():
    signals = _signals(text_inputs=1)
    step = media_prefill_step(signals, [VISION], model_name="m")
    assert step == WalkStep("prefill_text", step_metadata={FINAL_PREFILL: True})
    # encoder inputs with no layout to place them by are refused, not dropped
    with pytest.raises(ValueError, match="no media layout"):
        media_prefill_step(_signals(text_inputs=1, pixel_values=1), [VISION], model_name="m")
    # a graph with no text-only walk (TEXT_PROMPTS False) refuses the prompt by what it needs
    with pytest.raises(ValueError, match="m needs audio"):
        media_prefill_step(signals, [AUDIO], model_name="m", text=False)


@pytest.mark.parametrize(("counts", "walk"), [
    ({"pixel_values": 2}, "prefill_vision"),
    ({"audio_values": 1}, "prefill_audio"),
    ({"pixel_values": 1, "audio_values": 2}, "prefill_vision_audio"),
])
def test_a_media_prompt_is_one_walk_over_the_encoders_it_needs(counts, walk):
    signals = _signals(text_inputs=1, **counts)
    encoder = {"pixel_values": VISION, "audio_values": AUDIO}
    # the layout's runs name the encoders, as they do the KV layout's walk
    layout = {"offset": [3] * len(counts), "output": [encoder[name].output for name in counts]}
    step = media_prefill_step(signals, [VISION, AUDIO], layout, model_name="m")
    assert step == WalkStep(walk, step_metadata={MEDIA_LAYOUT: layout, FINAL_PREFILL: True})


@pytest.mark.parametrize(("texts", "match"), [(0, "nothing to prefill"), (2, "2 text tensors")])
def test_a_prompt_is_one_text_tensor(texts, match):
    with pytest.raises(ValueError, match=match):
        media_prefill_step(_signals(text_inputs=texts), [VISION], model_name="m")


# ---------------------------------------------------------------------------
# The decoder node
# ---------------------------------------------------------------------------


def _qwen_llm(encoders=(VISION,), model=None):
    from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    llm = object.__new__(LLMSubmodule)
    model = model or SimpleNamespace(embed_dtype=torch.bfloat16)
    SampledTokenDecoder.__init__(llm, model, SimpleNamespace(hidden_size=8, stop_token_ids=()), encoders)
    return llm


def test_a_media_prompt_hands_its_advance_and_ids_to_the_resources():
    """The position resource takes the row's advance; the repetition penalty
    tracks the whole prompt, pads and sentinels included."""
    from mstar.model.submodule_base import ARNodeInputs

    llm = _qwen_llm()
    rows = [
        ARNodeInputs(input_seq_len=n, input_ids=torch.full((n,), IMAGE_PAD), resource_step_info=advance)
        for n, advance in ((10, 6), (4, 4))
    ]
    step = llm.declare_step("prefill_vision", [0, 1], rows)
    assert step.state[KV_CACHE].advance == (6, 4)
    assert step.state[SAMPLER].prompt_tokens[0] is rows[0].input_ids
    # a mixed step never carries media rows
    assert llm.declare_step(MIXED, [0], [ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([1]))]) \
        .state[KV_CACHE].advance is None


def test_a_padded_media_step_moves_no_position_for_its_padding_rows():
    """A replay pads a 3-row media batch to the 4-row capture with the
    capture's own rows; those advance by their (zero) length beside real rows
    that place their own positions."""
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
    from mstar.engine.resources.step import BucketKey, SlotLease
    from mstar.model.submodule_base import ARNodeInputs

    llm = _qwen_llm()
    (config,) = [
        c for c in llm.get_accelerator_graph_configs(torch.device("cpu"))
        if c.capture_graph_walk == "prefill_vision"
    ]
    rows = [
        ARNodeInputs(input_seq_len=n, input_ids=torch.full((n,), IMAGE_PAD), resource_step_info=advance)
        for n, advance in ((10, 6), (4, 4), (7, 5))
    ]
    bucket = BucketKey("prefill_vision", bs=4, num_tokens=32)
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._buckets = {bucket: SimpleNamespace(config=config)}
    padded = runner.pad_inputs(SlotLease(slot=0, bucket=bucket), rows)
    step = llm.declare_step("prefill_vision", [0, 1, 2, 3], padded)
    assert step.state[KV_CACHE].advance == (6, 4, 5, 0)


def test_a_media_row_pads_with_the_captures_rows_on_the_device():
    """A replay pads a real media row (its mask built on the host from the
    layout) with the capture's own rows, built for the accelerator: the step's
    mask still concatenates, then goes to the device in one copy."""
    from mstar.model.components.sampled_token_decoder import SampledTokenDecoder

    device = torch.device("meta")  # a device other than the host, as an accelerator is
    llm = _qwen_llm()
    llm.prompt_positions = lambda start, layout, n: (None, None)
    llm.next_position = lambda rid_handle: 0
    (config,) = [
        c for c in llm.get_accelerator_graph_configs(device) if c.capture_graph_walk == "prefill_vision"
    ]
    layout = {"offset": [1], "tokens": [2], "output": ["vision_embeds"], "grid": [[]]}
    fwd_info = SimpleNamespace(request_id="r", rid_handle=0, step_metadata={MEDIA_LAYOUT: layout})
    row = llm.prepare_inputs("prefill_vision", fwd_info, {
        "text_inputs": [torch.zeros(4, dtype=torch.long, device=device)],
        "vision_embeds": [torch.zeros(2, 8, dtype=torch.bfloat16, device=device)],
    })
    engine_inputs = SimpleNamespace(resources={}, request_ids=[0, 1])
    out = SampledTokenDecoder.preprocess(llm, "prefill_vision", engine_inputs, [row, config.make_node_input(0)])
    assert out[MEDIA_MASK].device == device and out[MEDIA_MASK].shape == (4,)


def test_a_media_row_with_the_wrong_row_count_fails_its_request():
    """Checked on host values: the media tensor's shape against the layout."""
    llm = _qwen_llm((AUDIO,))
    starts = []
    llm.prompt_positions = lambda start, layout, n: starts.append(start) or (None, None)
    # the stream's counter is keyed by the worker's handle, not the wire id
    llm.kv = SimpleNamespace(positions=SimpleNamespace(position=lambda rid, label: {(7, "main"): 5}[rid, label]))
    ids = torch.tensor([1, AUDIO_PAD, AUDIO_PAD])
    layout = {"offset": [1], "tokens": [2], "output": ["audio_embeds"], "grid": [[]]}
    fwd_info = SimpleNamespace(request_id="r", rid_handle=7, step_metadata={MEDIA_LAYOUT: layout})
    row = llm.prepare_inputs("prefill_audio", fwd_info, {"text_inputs": [ids], "audio_embeds": [torch.zeros(2, 8)]})
    assert row.input_ids is ids and row.tensor_inputs[MEDIA_EMBEDS].shape == (2, 8)
    assert starts == [5]
    with pytest.raises(ValueError, match="1 rows for the prompt's 2 pads"):
        llm.prepare_inputs("prefill_audio", fwd_info, {"text_inputs": [ids], "audio_embeds": [torch.zeros(1, 8)]})
    # no output at all: a replay would splice in the previous step's rows
    with pytest.raises(ValueError, match="0 rows for the prompt's 2 pads"):
        llm.prepare_inputs("prefill_audio", fwd_info, {"text_inputs": [ids]})


@pytest.mark.parametrize("tile", [{"tile_embeds": [torch.zeros(0, 1)]}, {}], ids=["empty", "absent"])
def test_encoders_sharing_a_placeholder_take_their_rows_from_the_layout(tile):
    """The layout, not the pad id, says whose row each token takes, so two
    media may share one placeholder (MiniCPM-o's ``<unk>``). A run of no
    tokens takes no rows, whether its encoder sent an empty output or none."""
    from mstar.model.multimodal import splice_media

    tiles = MediaEncoder("tile", "tile_encoder", ("tile_values",), "tile_embeds")
    llm = _qwen_llm((VISION, AUDIO, tiles))
    llm.prompt_positions = lambda start, layout, n: (None, None)
    llm.next_position = lambda rid_handle: 0
    unk = 0
    ids = torch.tensor([1, unk, unk, 2, unk, 3, unk])
    layout = {"offset": [1, 4, 5, 6], "tokens": [2, 1, 0, 1],
              "output": ["vision_embeds", "audio_embeds", "tile_embeds", "vision_embeds"], "grid": [[], [], [], []]}
    fwd_info = SimpleNamespace(request_id="r", rid_handle=0, step_metadata={MEDIA_LAYOUT: layout})
    vision, audio = torch.tensor([[10.0], [11.0], [12.0]]), torch.tensor([[20.0]])
    row = llm.prepare_inputs(
        "prefill_vision_audio_tile", fwd_info,
        {"text_inputs": [ids], "vision_embeds": [vision], "audio_embeds": [audio], **tile},
    )
    spliced = splice_media(ids.float()[:, None], row.tensor_inputs[MEDIA_MASK], row.tensor_inputs[MEDIA_EMBEDS])
    assert spliced[:, 0].tolist() == [1, 10, 11, 2, 20, 3, 12]


U = 0  # a placeholder two intakes share (MiniCPM-o's <unk>)


@pytest.mark.parametrize("rendered, opens, expanded, offsets", [
    ([1, U, 2, U, 3], (), [1, U, U, U, 2, U, U, 3], [1, 5]),
    # each intake's placeholders follow its own openers (1 audio's, 2 the
    # image's); the pads the user typed follow neither and stay tokens
    ([U, 1, U, 2, U, 3, U], ({1}, {2}), [U, 1, U, U, U, 2, U, U, 3, U], [2, 6]),
], ids=["every-pad", "after-openers"])
def test_two_intakes_may_share_a_placeholder(rendered, opens, expanded, offsets):
    """Each placeholder goes to the attachment the prompt renders there, and
    the layout names that attachment's encoder. An intake declaring the ids
    its placeholders follow leaves its pad elsewhere as text."""
    from mstar.model.multimodal_decoder import MediaIntake, MultimodalDecoderModel, PreparedMedia

    class Model(MultimodalDecoderModel):
        NAME = "shared"

        def _prompt_ids(self, parts, raw, **kwargs):
            return torch.tensor(rendered)

        def get_node_resources(self):
            return []

    model = object.__new__(Model)
    model.media = tuple(
        MediaIntake(
            m, encoder, U, lambda attachment, metadata, request, n=n: PreparedMedia({}, [n]),
            opens=frozenset(after),
        )
        for m, encoder, n, after in zip(("audio", "image"), (AUDIO, VISION), (3, 2), opens or ((), ()), strict=True)
    )
    attachments = {"image_inputs": [torch.zeros(1)], "audio_inputs": [torch.zeros(1)]}
    prompt = model.intake(None, ["audio", "image"], attachments)
    layout = model.prompt_layout(prompt)
    assert prompt.pads.ids.tolist() == expanded
    assert (layout["offset"], layout["tokens"]) == (offsets, [3, 2])
    assert layout["output"] == ["audio_embeds", "vision_embeds"]


def test_every_media_walk_replays_one_capture_that_caps_it():
    """Audio-only and vision-only steps replay the one media capture, and its
    largest batch is each walk's cap, from the node itself too, so it holds
    where nothing is captured."""
    llm = _qwen_llm((VISION, AUDIO))
    configs = llm.get_accelerator_graph_configs(torch.device("cpu"))
    for walk in ("prefill_vision", "prefill_audio", "prefill_vision_audio"):
        (config,) = [c for c in configs if walk in c.replay_graph_walks]
        assert config.caps_eager_batch_size
        assert config.input_seq_dims == {MEDIA_EMBEDS: 0, MEDIA_MASK: 0}
        assert llm.max_batch_size(walk) == max(config.capture_batch_sizes)
    assert llm.max_batch_size("decode") is None


def test_a_node_keeps_its_walk_name_and_can_capture_decode_only():
    """A lone encoder's declared walk names its media walk (configs and
    metrics read walk names), and CAPTURE_PREFILL = False leaves decode as the
    only capture."""
    from dataclasses import replace

    from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    named = replace(AUDIO, walk="prefill")
    assert media_walk_names([named]) == ["prefill"]
    assert media_walk_names([VISION, named]) == ["prefill_vision", "prefill", "prefill_vision_audio"]

    class DecodeOnly(LLMSubmodule):
        CAPTURE_PREFILL = False

    llm = object.__new__(DecodeOnly)
    SampledTokenDecoder.__init__(
        llm, SimpleNamespace(embed_dtype=torch.bfloat16), SimpleNamespace(hidden_size=8, stop_token_ids=()), (named,),
    )
    assert [c.capture_graph_walk for c in llm.get_accelerator_graph_configs(torch.device("cpu"))] == ["decode"]


def test_a_media_only_model_has_no_text_walk_mixed_step_or_prefix_keys():
    """TEXT_PROMPTS = False (an audio-only STT model) drops the text prefill
    walk, the mixed step and the prefix keys from the model, and the text and
    mixed captures from its LLM node."""
    from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    walks = media_walks(8, [AUDIO], text=False)
    assert set(walks) == {"prefill_audio", "decode"}
    llm = object.__new__(LLMSubmodule)
    SampledTokenDecoder.__init__(
        llm, SimpleNamespace(embed_dtype=torch.bfloat16), SimpleNamespace(hidden_size=8, stop_token_ids=()),
        (AUDIO,), text_prompts=False,
    )
    captured = {w for c in llm.get_accelerator_graph_configs(torch.device("cpu")) for w in c.replay_graph_walks}
    assert captured == {"prefill_audio", "decode"}


def test_a_second_decode_walk_in_the_graph_is_a_decode_walk_everywhere():
    """A walk whose graph runs the LLM in the decode loop, which nothing else
    lists, stops on the stop ids, replays the decode capture, joins the mixed
    step and grows the KV label, as ``decode`` does: the node the model builds
    takes its graph's walks however the model constructs it."""
    from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
    from mstar.model.keys import KV_CACHE
    from mstar.model.multimodal_decoder import MultimodalDecoderModel
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    class Model(MultimodalDecoderModel):
        NAME = "two decodes"

        def get_graph_walk_graphs(self):
            walks = media_walks(8, self.encoders)
            walks["decode_other"] = walks["decode"]
            return walks

        def get_node_resources(self):
            return []

        def _create_submodule(self, node_name, device, dtype, tp_group):
            llm = object.__new__(LLMSubmodule)
            SampledTokenDecoder.__init__(
                llm, SimpleNamespace(embed_dtype=torch.bfloat16), SimpleNamespace(hidden_size=8, stop_token_ids=(7,)),
            )
            return llm

    model = object.__new__(Model)
    model._submodule_cache = {}
    assert model.get_combined_graph_walks() == {"LLM": {MIXED: {"prefill_text", "decode", "decode_other"}}}
    (label,) = model.kv_labels()[KV_CACHE].values()
    assert label.growth == (("decode", "decode_loop"), ("decode_other", "decode_loop"))

    llm = model.get_submodule("LLM")
    assert llm.get_stop_policies("decode_other")["decode_loop"].eos_tokens == frozenset({7})
    (decode,) = [c for c in llm.get_accelerator_graph_configs(torch.device("cpu")) if c.capture_graph_walk == "decode"]
    # and a mixed step of its rows and decode's alone
    assert decode.replay_graph_walks == ["decode", "decode_other", MIXED]
    # each decoder node takes its own graph's walks: one in no decode loop is refused
    with pytest.raises(ValueError, match="no walk runs Drafter"):
        model.get_submodule("Drafter")


def test_qwen_media_capture_stages_every_media_buffer():
    """The captured media step's static inputs are `preprocess`'s outputs:
    the ids, a row per token for every encoder's media, and the cos/sin built
    from the position resource's broadcast ids (a replay restages real ones)."""
    seen = {}

    def build_cos_sin(pos, dtype):
        seen["pos"] = tuple(pos.shape)
        table = torch.zeros(pos.shape[1], 4, dtype=dtype)
        return table, table

    llm = _qwen_llm(model=SimpleNamespace(
        embed_dtype=torch.bfloat16, model=SimpleNamespace(build_cos_sin=build_cos_sin),
    ))
    llm.kv = SimpleNamespace(
        _check_open=lambda call: None, label="main",
        positions=SimpleNamespace(pos_ids=lambda label: torch.arange(8192)),
    )
    (config,) = [
        c for c in llm.get_accelerator_graph_configs(torch.device("cpu"))
        if c.capture_graph_walk == "prefill_vision"
    ]
    for n in config.get_total_tokens(1):
        rows = [config.make_node_input(n)]
        engine_inputs = SimpleNamespace(resources={}, request_ids=[0], captured=True)
        out = type(llm).preprocess(llm, "prefill_vision", engine_inputs, rows)
        assert list(out) == ["input_ids", MEDIA_EMBEDS, MEDIA_MASK, "cos_3d", "sin_3d"]
        assert out[MEDIA_EMBEDS].shape == (n, 8) and out[MEDIA_EMBEDS].dtype == torch.bfloat16
        assert seen["pos"] == (3, n)


def test_the_forward_reads_only_tensors_from_preprocess():
    llm = _qwen_llm()
    engine_inputs = SimpleNamespace(resources={SAMPLER: SimpleNamespace()}, request_ids=[0])
    with pytest.raises(AssertionError, match="must be tensors"):
        type(llm)._forward(llm, "decode", engine_inputs, input_ids=torch.tensor([1]), cos_3d=(1, 2))


# ---------------------------------------------------------------------------
# Muse Glimmer on the scaffolds
# ---------------------------------------------------------------------------


def _glimmer_llm():
    from mstar.model.muse.submodules import LLMSubmodule

    model = SimpleNamespace(embed_dtype=torch.bfloat16)
    return LLMSubmodule(model, SimpleNamespace(hidden_size=8, stop_token_ids=()), (VISION,))


def test_glimmer_writes_its_steps_by_hand_on_the_scaffolds_default():
    from mstar.engine.resources.position.config import PositionStep
    from mstar.engine.resources.sampler.config import SamplerStep
    from mstar.model.muse.config import ATTN_FULL
    from mstar.model.submodule_base import ARNodeInputs

    step = _glimmer_llm().declare_step(
        "prefill_vision", [0], [ARNodeInputs(input_seq_len=4, input_ids=torch.arange(4), resource_step_info=3)],
    )
    assert not step.state, "Muse Glimmer has no handles to expand a declared step"
    assert isinstance(step.steps[SAMPLER], SamplerStep) and isinstance(step.steps[ROPE], PositionStep)
    assert step.steps[ROPE].advance == (3,) and ATTN_FULL in step.steps


def test_glimmer_samples_and_loops_back_off_its_resources():
    from mstar.model.muse.config import ATTN_FULL

    hidden = torch.arange(6.0).reshape(3, 2)

    class _Decoder:
        def embed(self, ids):
            return hidden

        def __call__(self, embeds, label):
            return embeds

        def logits(self, x):
            return x

    llm = _glimmer_llm()
    llm.model = _Decoder()
    engine_inputs = SimpleNamespace(request_ids=["r"], resources={
        SAMPLER: SimpleNamespace(
            loopback_tokens=lambda rids: torch.tensor([9]) if rids == ["r"] else None,
            sample=lambda rids, logits: (rids, logits),
        ),
        ATTN_FULL: SimpleNamespace(select_last_hidden=lambda x: x[-1:]),
    })
    rows, logits = llm._forward("prefill_text", engine_inputs)
    assert rows == ["r"] and torch.equal(logits, hidden[-1:]), "a prefill feeds the head each row's last token"
    assert llm.loopback_tokens(engine_inputs).tolist() == [9]


def test_glimmer_reads_its_position_off_its_resource():
    llm = _glimmer_llm()
    llm.node_resources = {ROPE: SimpleNamespace(position=lambda rid, label: {(7, "main"): 5}[rid, label])}
    assert llm.next_position(7) == 5


def test_glimmer_encoder_reads_its_segment_ceiling_off_its_resource():
    from mstar.model.keys import VISION_ATTN
    from mstar.model.muse.submodules import VisionEncoderSubmodule

    encoder = VisionEncoderSubmodule(torch.nn.Linear(1, 1), SimpleNamespace(), "vision_embeds")
    encoder.node_resources = {VISION_ATTN: SimpleNamespace(max_segments_per_step=2)}
    assert encoder._fits_replay(16, 2) and not encoder._fits_replay(16, 3), "a YAML override should hold"


def test_glimmer_encoder_declares_windows_and_frames():
    from mstar.model.muse.config import VISION_FULL, VISION_WINDOW
    from mstar.model.muse.submodules import VisionEncoderSubmodule
    from mstar.model.submodule_base import NodeInputs

    encoder = object.__new__(VisionEncoderSubmodule)
    row = NodeInputs(input_seq_len=8, resource_step_info={VISION_WINDOW: (4, 4), VISION_FULL: (8,)})
    step = encoder.declare_step("prefill_vision", ["r"], [row])
    assert [(seg.label, seg.span) for seg in step.segments] == [
        (VISION_WINDOW, 4), (VISION_WINDOW, 4), (VISION_FULL, 8),
    ]


@pytest.mark.parametrize("knobs", [
    {"return_logprobs": True}, {"logprobs_mode": "processed"},
])
def test_a_request_sets_its_sampling_knobs(knobs):
    """Log-probs (#914) reach the sampler's config like every other knob."""
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    model = object.__new__(Qwen3_5DenseModel)
    model.sampling_defaults = {}
    config = model.get_request_resource_configs([], [], knobs)[SAMPLER]
    assert {k: getattr(config, k) for k in knobs} == knobs


def test_qwen3_5_floors_a_float_images_levels():
    """Qwen3.5 (and Qwen3-Omni) hand the processor floored levels, as their
    parity was measured with; the shared helper rounds by default."""
    from mstar.model.multimodal import as_uint8_chw
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    half = torch.full((3, 2, 2), 0.5)  # 127.5 levels
    assert as_uint8_chw(half).max() == 128 and as_uint8_chw(half, truncate=True).max() == 127
    model = object.__new__(Qwen3_5DenseModel)
    model.vision_config = SimpleNamespace(spatial_merge_size=2)
    seen = []

    def processor(images, return_tensors, **kwargs):
        seen.extend(images)
        return {"pixel_values": torch.zeros(4, 8), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    model._image_processor = processor
    model._image_intake(half, {})
    assert int(seen[0].max()) == 127


def test_a_two_label_tower_refuses_a_resource_whose_step_is_its_segments():
    """A LoRA pool reads the segments as the rows the tower's linears see; Glimmer's
    windows and frames each cover every patch, so binding one would count them twice."""
    from mstar.model.muse.submodules import VisionEncoderSubmodule

    encoder = object.__new__(VisionEncoderSubmodule)
    torch.nn.Module.__init__(encoder)
    with pytest.raises(ValueError, match="one row per segment"):
        encoder.bind_node_resources({"lora": SimpleNamespace(steps_from_segments=True)})


def _glimmer_encoder(max_segments: int):
    from mstar.model.keys import VISION_ATTN
    from mstar.model.muse.submodules import VisionEncoderSubmodule

    encoder = object.__new__(VisionEncoderSubmodule)
    encoder.config = SimpleNamespace(hidden_size=8, head_dim=4)
    encoder.node_resources = {VISION_ATTN: SimpleNamespace(max_segments_per_step=max_segments)}
    return encoder


def test_glimmer_leases_and_replays_both_layouts():
    """A replay's plan holds the resource's ceiling under each label, frames
    as well as windows, and plans both; the frames pick the bucket."""
    from mstar.model.components.block_loop_encoder import VISION_BLOCK_LOOP
    from mstar.model.muse.config import VISION_FULL, VISION_WINDOW
    from mstar.model.submodule_base import NodeInputs

    encoder = _glimmer_encoder(max_segments=4)
    region = encoder.get_piecewise_accelerator_graph_configs(
        torch.device("cpu"), torch.bfloat16,
    )[VISION_BLOCK_LOOP]

    def row(windows: int, frames: int) -> NodeInputs:
        return NodeInputs(
            input_seq_len=windows,
            resource_step_info={VISION_WINDOW: (1,) * windows, VISION_FULL: (1,) * frames},
        )

    assert region.can_lease([row(4, 4)])
    assert not region.can_lease([row(1, 5)])
    assert not region.can_lease([row(5, 1)])

    ran = {}
    runner = SimpleNamespace(
        run=lambda **kwargs: ran.update(kwargs) or SimpleNamespace(get_view=lambda name: "replayed"),
    )
    hidden = torch.zeros(8, 8)
    engine_inputs = SimpleNamespace(
        piecewise_runners={VISION_BLOCK_LOOP: runner}, piecewise_leases={VISION_BLOCK_LOOP: object()},
        request_ids=["r"],
    )
    lengths = {VISION_WINDOW: (4, 4), VISION_FULL: (8,)}
    replayed = encoder._replay_block_loop(engine_inputs, hidden, hidden, hidden, lengths)
    assert replayed.get_view("hidden") == "replayed"
    assert ran["seq_lens"] == [8]
    assert ran["label_seq_lens"] == {VISION_WINDOW: [4, 4], VISION_FULL: [8]}
