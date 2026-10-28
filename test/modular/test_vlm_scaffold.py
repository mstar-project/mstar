"""The shared VLM scaffolds: the media walk set in ``multimodal``, the
`SampledTokenDecoder` LLM node and the `PackedBlockLoopEncoder` vision node.

No checkpoint: the pad expansion, the splice and the schedule run on small
ids, and the nodes on stubs. The per-model files test each model's own pieces.
"""
from types import SimpleNamespace

import pytest
import torch

from mstar.graph.base import Parallel
from mstar.model.keys import MIXED, ROPE, SAMPLER
from mstar.model.multimodal import (
    MEDIA_LAYOUT,
    MediaEncoder,
    PrefillStep,
    expand_media_pads,
    initial_prefill_args,
    media_layout,
    media_prefill_schedule,
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
    from mstar.model.components.sampled_token_decoder import splice_media

    splice = torch.compile(splice_media, fullgraph=True, backend="aot_eager") if compiled else splice_media
    ids = torch.tensor([1, IMAGE_PAD, IMAGE_PAD, 2, VIDEO_PAD, 3])
    media = torch.tensor([[100.0], [101.0], [102.0]])
    out = splice(ids.float()[:, None], ids, media, (IMAGE_PAD, VIDEO_PAD))
    assert out[:, 0].tolist() == [1, 100, 101, 2, 102, 3]


def test_a_replay_tail_cannot_shift_a_real_row():
    """Real tokens precede the tail, so stale pads there index past the real
    media without moving a real row's index; the buffer holds a row per
    token, so their index stays in bounds."""
    from mstar.model.components.sampled_token_decoder import splice_media

    real = [5, IMAGE_PAD, 6]
    ids = torch.tensor(real + [IMAGE_PAD, IMAGE_PAD, IMAGE_PAD])  # a previous replay's tail
    media = torch.zeros(len(ids), 1)
    media[0] = 100.0
    out = splice_media(ids.float()[:, None], ids, media, (IMAGE_PAD,))
    assert out[:3, 0].tolist() == [5, 100, 6]


# ---------------------------------------------------------------------------
# Walks, schedule and forward-pass args
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


def test_text_only_prompts_prefill_a_walk_per_text():
    signals = _signals(text_inputs=2)
    schedule = media_prefill_schedule(signals, [VISION], model_name="m")
    assert schedule == [
        PrefillStep("prefill_text", {("LLM", "text_inputs"): [f"text_inputs{i}"]}) for i in range(2)
    ]


@pytest.mark.parametrize(("counts", "walk"), [
    ({"pixel_values": 2, "image_grid_thw": 2}, "prefill_vision"),
    ({"audio_values": 1}, "prefill_audio"),
    ({"pixel_values": 1, "image_grid_thw": 1, "audio_values": 2}, "prefill_vision_audio"),
])
def test_a_media_prompt_is_one_walk_over_the_encoders_it_needs(counts, walk):
    signals = _signals(text_inputs=1, **counts)
    layout = {"offset": [3]}
    (step,) = media_prefill_schedule(signals, [VISION, AUDIO], layout, model_name="m")
    assert step.walk == walk and step.metadata == {MEDIA_LAYOUT: layout}
    edges = {(e.next_node, e.name): e.tensor_info for e in walk_input_edges(step)}
    node = {"pixel_values": "vision_encoder", "image_grid_thw": "vision_encoder", "audio_values": "audio_encoder"}
    assert edges == {
        ("LLM", "text_inputs"): ["text_inputs0"],
        **{(node[name], name): signals[name] for name in counts},
    }


def test_the_walks_run_the_schedule_then_decode_then_finish():
    signals = _signals(text_inputs=2)
    schedule = media_prefill_schedule(signals, [VISION], model_name="m")
    first = initial_prefill_args(schedule, ["text"], ["text"], signals, model_name="m")
    assert first.step_metadata == {"is_prefill": True, "last_prefill": False}
    meta = first.full_metadata
    second = next_prefill_args(meta, {})
    assert meta.graph_walk == "prefill_text" and second.step_metadata["last_prefill"]
    decode = next_prefill_args(meta, {"new_token": ["tok"]})
    assert meta.graph_walk == "decode" and not meta.is_prefill
    assert [(e.next_node, e.name, e.tensor_info) for e in decode.inputs] == [("LLM", "text_inputs", ["tok"])]
    assert next_prefill_args(meta, {}).request_done


def test_a_media_walk_carries_its_layout_in_step_metadata():
    signals = _signals(text_inputs=1, pixel_values=1, image_grid_thw=1)
    layout = {"offset": [1], "tokens": [4], "output": ["vision_embeds"], "grid": [[1, 4, 4]]}
    schedule = media_prefill_schedule(signals, [VISION], layout, model_name="m")
    first = initial_prefill_args(schedule, ["image", "text"], ["text"], signals, model_name="m")
    assert first.step_metadata == {"is_prefill": True, "last_prefill": True, MEDIA_LAYOUT: layout}


def test_nothing_to_prefill_is_a_client_error():
    with pytest.raises(ValueError, match="nothing to prefill"):
        initial_prefill_args([], ["text"], ["text"], {}, model_name="m")


# ---------------------------------------------------------------------------
# The decoder node
# ---------------------------------------------------------------------------


def test_a_token_is_real_on_decode_and_on_the_last_prefill():
    from mstar.model.components.sampled_token_decoder import token_is_real

    last = SimpleNamespace(step_metadata={"last_prefill": True})
    early = SimpleNamespace(step_metadata={"last_prefill": False})
    assert token_is_real("decode", None)
    assert token_is_real("prefill_text", last)
    assert not token_is_real("prefill_text", early)
    # a capture's padding rows carry no info
    assert not token_is_real("prefill_vision", None)


def _qwen_llm(encoders=(VISION,), model=None):
    from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    llm = object.__new__(LLMSubmodule)
    model = model or SimpleNamespace(embed_dtype=torch.bfloat16)
    SampledTokenDecoder.__init__(llm, model, SimpleNamespace(hidden_size=8, stop_token_ids=()), encoders)
    return llm


def test_the_step_is_the_layers_then_the_sampler_and_position():
    from mstar.model.qwen3_5.config import ATTN, GDN_STATE, KV_CACHE, LINEAR_ATTN
    from mstar.model.submodule_base import ARNodeInputs

    llm = _qwen_llm()
    rows = [ARNodeInputs(input_seq_len=3, input_ids=torch.tensor([5, 6, 7]))]
    step = llm.declare_step("prefill_text", [0], rows)
    assert list(step.steps) == [KV_CACHE, ATTN, GDN_STATE, LINEAR_ATTN, SAMPLER, ROPE]
    assert step.steps[ROPE].advance is None


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
    assert step.steps[ROPE].advance == (6, 4)
    assert step.steps[SAMPLER].prefill_tracked_tokens[0] is rows[0].input_ids
    # a mixed step never carries media rows
    assert llm.declare_step(MIXED, [0], [ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([1]))]) \
        .steps[ROPE].advance is None


def test_a_media_row_with_the_wrong_row_count_fails_its_request():
    """Checked on host values: the media tensor's shape against the layout."""
    llm = _qwen_llm((AUDIO,))
    llm.prompt_positions = lambda fwd_info, layout, n: (None, None)
    ids = torch.tensor([1, AUDIO_PAD, AUDIO_PAD])
    layout = {"offset": [1], "tokens": [2], "output": ["audio_embeds"], "grid": [[]]}
    fwd_info = SimpleNamespace(request_id="r", step_metadata={MEDIA_LAYOUT: layout})
    row = llm.prepare_inputs("prefill_audio", fwd_info, {"text_inputs": [ids], "audio_embeds": [torch.zeros(2, 8)]})
    assert row.input_ids is ids and row.tensor_inputs["audio_embeds"].shape == (2, 8)
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
        assert max(config.capture_batch_sizes) == max(llm.PREFILL_MEDIA_CAPTURE_BATCH_SIZES)
        assert config.input_seq_dims == {"vision_embeds": 0, "audio_embeds": 0}
        assert llm.max_batch_size(walk) == max(llm.PREFILL_MEDIA_CAPTURE_BATCH_SIZES)
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
    llm.node_resources = {ROPE: SimpleNamespace(pos_ids=lambda label: torch.arange(8192))}
    (config,) = [
        c for c in llm.get_accelerator_graph_configs(torch.device("cpu"))
        if c.capture_graph_walk == "prefill_vision"
    ]
    for n in config.capture_token_lengths:
        rows = [config.make_node_input(n)]
        engine_inputs = SimpleNamespace(resources={}, request_ids=[0])
        out = type(llm).preprocess(llm, "prefill_vision", engine_inputs, rows)
        assert list(out) == ["input_ids", "vision_embeds", "cos_3d", "sin_3d"]
        assert out["vision_embeds"].shape == (n, 8) and out["vision_embeds"].dtype == torch.bfloat16
        assert seen["pos"] == (3, n)


def test_the_forward_reads_only_tensors_from_preprocess():
    llm = _qwen_llm()
    engine_inputs = SimpleNamespace(resources={SAMPLER: SimpleNamespace()}, request_ids=[0])
    with pytest.raises(AssertionError, match="must be tensors"):
        type(llm)._forward(llm, "decode", engine_inputs, input_ids=torch.tensor([1]), cos_3d=(1, 2))
