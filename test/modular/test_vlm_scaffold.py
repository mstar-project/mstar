"""The shared VLM scaffolds: the media walk set in ``multimodal``, the
`SampledTokenDecoder` LLM node and the `PackedBlockLoopEncoder` vision node.

No checkpoint: the pad expansion, the splice and the prefill walk run on small
ids, and the nodes on stubs. The per-model files test each model's own pieces.
"""
from types import SimpleNamespace

import pytest
import torch

from mstar.graph.base import Parallel
from mstar.model.keys import KV_CACHE, MIXED, ROPE, SAMPLER
from mstar.model.multimodal import (
    MEDIA_LAYOUT,
    MediaEncoder,
    PrefillStep,
    expand_media_pads,
    initial_prefill_args,
    media_layout,
    media_prefill_step,
    media_walk_names,
    media_walks,
    next_prefill_args,
    vision_encoder,
    walk_input_edges,
)

IMAGE_PAD, VIDEO_PAD, AUDIO_PAD = 90, 91, 80
VISION = vision_encoder((IMAGE_PAD, VIDEO_PAD))
AUDIO = MediaEncoder("audio", "audio_encoder", ("audio_values",), "audio_embeds", (AUDIO_PAD,))


# ---------------------------------------------------------------------------
# Pad expansion
# ---------------------------------------------------------------------------


def test_each_pad_expands_to_its_tokens_in_prompt_order():
    ids = torch.tensor([1, IMAGE_PAD, 2, AUDIO_PAD, IMAGE_PAD])
    pads = expand_media_pads(
        ids, {IMAGE_PAD: [2, 1], AUDIO_PAD: [3]}, sentinels={AUDIO_PAD: (70, 71)},
    )
    assert pads.ids.tolist() == [1, 90, 90, 2, 70, 80, 80, 80, 71, 90]
    assert (pads.pad, pads.index, pads.offset, pads.tokens) == (
        [IMAGE_PAD, AUDIO_PAD, IMAGE_PAD], [0, 0, 1], [1, 5, 9], [2, 3, 1],
    )
    layout = media_layout(
        pads, {IMAGE_PAD: "vision_embeds", AUDIO_PAD: "audio_embeds"},
        {IMAGE_PAD: [torch.tensor([1, 4, 4]), torch.tensor([1, 2, 2])]},
    )
    assert layout == {
        "offset": [1, 5, 9], "tokens": [2, 3, 1],
        "output": ["vision_embeds", "audio_embeds", "vision_embeds"],
        "grid": [[1, 4, 4], [], [1, 2, 2]],
    }


@pytest.mark.parametrize("ids", [[IMAGE_PAD, IMAGE_PAD], []])
def test_a_pad_without_a_count_or_a_count_without_a_pad_is_refused(ids):
    # user text that spells a placeholder; an attachment the prompt never placed
    with pytest.raises(ValueError, match="placeholder"):
        expand_media_pads(torch.tensor(ids, dtype=torch.long), {IMAGE_PAD: [1]})


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
    assert step == PrefillStep("prefill_text", {("LLM", "text_inputs"): ["text_inputs0"]})
    # encoder inputs with no layout to place them by are refused, not dropped
    with pytest.raises(ValueError, match="no media layout"):
        media_prefill_step(_signals(text_inputs=1, pixel_values=1), [VISION], model_name="m")


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
    assert step.walk == walk and step.metadata == {MEDIA_LAYOUT: layout}
    edges = {(e.next_node, e.name): e.tensor_info for e in walk_input_edges(step)}
    assert edges == {
        ("LLM", "text_inputs"): ["text_inputs0"],
        **{(encoder[name].node, name): signals[name] for name in counts},
    }


def test_the_walks_run_the_prefill_then_decode_then_finish():
    signals = _signals(text_inputs=1)
    first = initial_prefill_args(media_prefill_step(signals, [VISION], model_name="m"), ["text"], ["text"])
    assert first.step_metadata == {"is_prefill": True}
    meta = first.full_metadata
    assert meta.graph_walk == "prefill_text"
    decode = next_prefill_args(meta, {"new_token": ["tok"]})
    assert meta.graph_walk == "decode" and not meta.is_prefill
    assert [(e.next_node, e.name, e.tensor_info) for e in decode.inputs] == [("LLM", "text_inputs", ["tok"])]
    assert next_prefill_args(meta, {}).request_done


@pytest.mark.parametrize("knobs", [
    {"return_logprobs": True}, {"logprobs_mode": "processed"},
])
def test_a_request_sets_its_sampling_knobs(knobs):
    """Log-probs (#914) reach the sampler's config like every other knob."""
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    model = object.__new__(Qwen3_5DenseModel)
    model.sampling_defaults = {}
    config = model.get_request_resource_configs({}, knobs)[SAMPLER]
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


def test_a_media_walk_carries_its_layout_in_step_metadata():
    signals = _signals(text_inputs=1, pixel_values=1)
    layout = {"offset": [1], "tokens": [4], "output": ["vision_embeds"], "grid": [[1, 4, 4]]}
    step = media_prefill_step(signals, [VISION], layout, model_name="m")
    first = initial_prefill_args(step, ["image", "text"], ["text"])
    assert first.step_metadata == {"is_prefill": True, MEDIA_LAYOUT: layout}


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
    assert row.input_ids is ids and row.tensor_inputs["audio_embeds"].shape == (2, 8)
    assert starts == [5]
    with pytest.raises(ValueError, match="1 rows for the prompt's 2 pads"):
        llm.prepare_inputs("prefill_audio", fwd_info, {"text_inputs": [ids], "audio_embeds": [torch.zeros(1, 8)]})
    # no output at all: a replay would splice in the previous step's rows
    with pytest.raises(ValueError, match="0 rows for the prompt's 2 pads"):
        llm.prepare_inputs("prefill_audio", fwd_info, {"text_inputs": [ids]})


def test_every_media_walk_replays_one_capture_that_caps_it():
    """Audio-only and vision-only steps replay the one media capture, and its
    largest batch is each walk's cap, from the node itself too, so it holds
    where nothing is captured."""
    llm = _qwen_llm((VISION, AUDIO))
    configs = llm.get_accelerator_graph_configs(torch.device("cpu"))
    for walk in ("prefill_vision", "prefill_audio", "prefill_vision_audio"):
        (config,) = [c for c in configs if walk in c.replay_graph_walks]
        assert config.caps_eager_batch_size
        assert config.input_seq_dims == {"vision_embeds": 0, "audio_embeds": 0}
        assert llm.max_batch_size(walk) == max(config.capture_batch_sizes)
    assert llm.max_batch_size("decode") is None


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
    for n in config.capture_token_lengths:
        rows = [config.make_node_input(n)]
        engine_inputs = SimpleNamespace(resources={}, request_ids=[0], captured=True)
        out = type(llm).preprocess(llm, "prefill_vision", engine_inputs, rows)
        assert list(out) == ["input_ids", "vision_embeds", "cos_3d", "sin_3d"]
        assert out["vision_embeds"].shape == (n, 8) and out["vision_embeds"].dtype == torch.bfloat16
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
