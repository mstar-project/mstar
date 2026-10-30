"""Qwen3-Omni prefills a request in one Thinker step and one Talker step.

The Thinker's walk runs the encoders the prompt's attachments need, then one
row with text spans and encoder outputs spliced in prompt order; that row's
MRoPE positions are HF's for the whole prompt. The step also streams the first
sampled token's embedding, so the Talker prefills the prompt and its assistant
prefix in one step, and the Thinker's first decode step sends nothing.
"""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

# Qwen3OmniModel's submodules import flashinfer at module level.
pytest.importorskip("flashinfer")
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoePreTrainedModelForConditionalGeneration,
    Qwen3OmniMoeVisionEncoder,
    _get_feat_extract_output_lengths,
)

from mstar.engine.accelerator_graph_config import PackedAcceleratorGraphConfig  # noqa: E402
from mstar.model.multimodal import parts_from_modalities, prefill_plan  # noqa: E402
from mstar.model.qwen3_omni.config import (  # noqa: E402
    AUDIO_PART,
    CODE_PRED_SAMPLER,
    IMAGE_PART,
    TALKER_KV,
    TALKER_SAMPLER,
    TEXT_PART,
    THINKER_KV,
    THINKER_POS,
    THINKER_PREFILL_WALKS,
    VIDEO_PART,
    Qwen3OmniModelConfig,
)
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel  # noqa: E402
from mstar.model.qwen3_omni.submodules import (  # noqa: E402
    AudioEncoderSubmodule,
    TalkerSubmodule,
    ThinkerSubmodule,
    VisionEncoderSubmodule,
)

HIDDEN = 8


@pytest.fixture(scope="module")
def config():
    return Qwen3OmniModelConfig()


@pytest.fixture
def model(config):
    model = Qwen3OmniModel.__new__(Qwen3OmniModel)
    model.config = config
    return model


# --- the walks and what the conductor sends -----------------------------------


def test_each_prefill_walk_reads_what_the_conductor_sends(model):
    walks = model.get_graph_walk_graphs()
    for walk in THINKER_PREFILL_WALKS:
        expected = {(name, node) for node, name in Qwen3OmniModel.PREFILL_INPUTS[walk]}
        assert walks[walk].get_inputs_outputs().ext_inputs == expected, walk
    assert {"talker_prefill", "talker_decode"} <= set(walks)
    assert "talker_last_prefill" not in walks


@pytest.mark.parametrize("mods,walk", [
    (["text"], "prefill_text"),
    (["audio", "text"], "prefill_audio"),
    (["text", "image"], "prefill_vision"),
    (["video", "text"], "prefill_vision"),
    (["image", "text", "audio"], "prefill_audio_vision"),
])
def test_the_walk_runs_the_encoders_the_attachments_need(model, mods, walk):
    signals = {
        "text_inputs": ["t0", "t1", "t2", "t3"],
        "pixel_values": ["px0"], "image_grid_thw": ["g0"],
        "pixel_values_videos": ["pv0"], "video_grid_thw": ["vg0"], "video_second_per_grid": ["s0"],
        "audio_features": ["af0"], "audio_seqlens": ["as0"],
    }
    assert model._thinker_prefill(mods, signals).walk == walk


def test_one_prefill_sends_every_part_in_prompt_order(model):
    mods = ["text", "video", "text", "image", "audio"]
    signals = {
        "text_inputs": ["t0", "t1", "t2", "t3"],
        "pixel_values": ["px0"], "image_grid_thw": ["g0"],
        "pixel_values_videos": ["pv0"], "video_grid_thw": ["vg0"], "video_second_per_grid": ["s0"],
        "audio_features": ["af0"], "audio_seqlens": ["as0"],
    }
    prefill = model._thinker_prefill(mods, signals)

    assert prefill.walk == "prefill_audio_vision"
    # no text between the image and the clip, so no span there
    assert prefill.order == (TEXT_PART, VIDEO_PART, TEXT_PART, IMAGE_PART, AUDIO_PART, TEXT_PART)
    sent = {(e.next_node, e.name): e.tensor_info for e in prefill.inputs}
    assert sent == {
        ("Thinker", "text_inputs"): ["t0", "t1", "t2"],
        # the video was written first, so its tensors come first
        ("vision_encoder", "pixel_values"): ["pv0", "px0"],
        ("vision_encoder", "image_grid_thw"): ["vg0", "g0"],
        ("Thinker", "image_grid_thw"): ["vg0", "g0"],
        ("Thinker", "video_second_per_grid"): ["s0"],
        ("audio_encoder", "audio_features"): ["af0"],
        ("audio_encoder", "audio_seqlens"): ["as0"],
    }


def test_an_image_prompt_still_sends_the_empty_video_edge(model):
    """The Thinker node waits for every input it names, an empty one included."""
    prefill = model._thinker_prefill(
        ["image", "text"], {"text_inputs": ["t0", "t1"], "pixel_values": ["px0"], "image_grid_thw": ["g0"]},
    )
    sent = {(e.next_node, e.name): e.tensor_info for e in prefill.inputs}
    assert sent["Thinker", "video_second_per_grid"] == []


def test_an_attachment_without_its_tensors_is_skipped(model):
    prefill = model._thinker_prefill(
        ["image", "text", "audio"],
        {"text_inputs": ["t0", "t1", "t2"], "pixel_values": ["px0"], "image_grid_thw": ["g0"]},
    )
    assert prefill.walk == "prefill_vision"
    assert prefill.order == (TEXT_PART, IMAGE_PART, TEXT_PART, TEXT_PART)


# --- the state machines -------------------------------------------------------


def test_the_thinker_prefills_once_then_decodes(model):
    signals = {"text_inputs": [object(), object()], "audio_features": [object()], "audio_seqlens": [object()]}
    first = model.get_initial_forward_pass_args("Thinker", ["audio", "text"], ["text", "audio"], signals)
    assert first.full_metadata.graph_walk == "prefill_audio"
    assert first.step_metadata["prefill_order"] == (TEXT_PART, AUDIO_PART, TEXT_PART)

    token = object()
    decode = model.get_partition_forward_pass_args("Thinker", first.full_metadata, {"new_token": [token]})
    assert decode.full_metadata.graph_walk == "thinker_decode"
    assert [(e.next_node, e.name, e.tensor_info) for e in decode.inputs] == [("Thinker", "text_inputs", [token])]
    assert not decode.request_done

    done = model.get_partition_forward_pass_args("Thinker", decode.full_metadata, {})
    assert done.request_done


def test_the_talker_prefills_once_then_decodes(model):
    first = model.get_initial_forward_pass_args("Talker", ["text"], ["text", "audio"], {}, {"voice": "Ethan"})
    assert first.full_metadata.graph_walk == "talker_prefill"
    # the Thinker's stream starts it
    assert first.inputs == []

    embeds = object()
    decode = model.get_partition_forward_pass_args(
        "Talker", first.full_metadata, {"talker_input_embeds": [embeds]},
    )
    assert decode.full_metadata.graph_walk == "talker_decode"
    assert [(e.name, e.tensor_info) for e in decode.inputs] == [("talker_input_embeds", [embeds])]

    done = model.get_partition_forward_pass_args("Talker", decode.full_metadata, {})
    assert done.request_done


# --- the Thinker's prefill row ------------------------------------------------


# past every special token id the prompts below use
VOCAB = 152_000


class _Embeds(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)


def _thinker(config) -> ThinkerSubmodule:
    thinker = ThinkerSubmodule(_Embeds(), config)
    thinker.node_resources = {THINKER_POS: SimpleNamespace(position=lambda rid, label: 0)}
    return thinker


def _hf_positions(config, input_ids, image_grids, video_grids, audio_seqlens, seconds) -> torch.Tensor:
    """HF's MRoPE positions for the whole prompt, pads included."""
    cls = Qwen3OmniMoePreTrainedModelForConditionalGeneration
    tc = config.thinker
    hf = SimpleNamespace(
        spatial_merge_size=config.vision.spatial_merge_size,
        config=SimpleNamespace(
            image_token_id=tc.image_token_id, video_token_id=tc.video_token_id, audio_token_id=tc.audio_token_id,
            vision_start_token_id=tc.vision_start_token_id, audio_start_token_id=tc.audio_start_token_id,
            position_id_per_seconds=tc.position_id_per_seconds, audio_config=SimpleNamespace(n_window=50),
        ),
    )
    hf.get_llm_pos_ids_for_vision = lambda *args: cls.get_llm_pos_ids_for_vision(hf, *args)
    positions, _ = cls.get_rope_index(
        hf, input_ids[None], image_grid_thw=image_grids, video_grid_thw=video_grids,
        attention_mask=torch.ones_like(input_ids)[None], audio_seqlens=audio_seqlens, second_per_grids=seconds,
    )
    return positions[:, 0, :]


def _prompt(config):
    """A system turn and a user turn holding a clip, an image and a video between text."""
    c, tc = config, config.thinker
    texts = [
        torch.tensor([c.im_start_token_id, c.system_token_id, 10, 11, c.im_end_token_id, c.im_start_token_id, 12, 13]),
        torch.tensor([20, 21]),
        torch.tensor([22]),
        torch.tensor([23, c.im_end_token_id, c.im_start_token_id, c.assistant_token_id, 198]),
    ]
    seqlens = torch.tensor([250])
    audio_rows = int(_get_feat_extract_output_lengths(seqlens))
    image_grid, video_grid = torch.tensor([1, 4, 6]), torch.tensor([2, 4, 4])
    image_rows = int(image_grid.prod()) // config.vision.spatial_merge_size ** 2
    video_rows = int(video_grid.prod()) // config.vision.spatial_merge_size ** 2
    ids = torch.cat([
        texts[0], torch.tensor([tc.audio_start_token_id, *[tc.audio_token_id] * audio_rows, tc.audio_end_token_id]),
        texts[1], torch.tensor([tc.vision_start_token_id, *[tc.image_token_id] * image_rows, tc.vision_end_token_id]),
        texts[2], torch.tensor([tc.vision_start_token_id, *[tc.video_token_id] * video_rows, tc.vision_end_token_id]),
        texts[3],
    ])
    return SimpleNamespace(
        texts=texts, seqlens=seqlens, audio_rows=audio_rows, image_grid=image_grid, video_grid=video_grid,
        image_rows=image_rows, video_rows=video_rows, ids=ids, seconds=torch.tensor([1.0]),
        mods=["text", "audio", "text", "image", "text", "video"],
        order=(TEXT_PART, AUDIO_PART, TEXT_PART, IMAGE_PART, TEXT_PART, VIDEO_PART, TEXT_PART),
    )


def _row(config, prompt):
    layers = len(config.vision.deepstack_visual_indexes)
    deepstack = (
        [torch.full((prompt.image_rows, HIDDEN), 1.0 + i) for i in range(layers)]
        + [torch.full((prompt.video_rows, HIDDEN), 10.0 + i) for i in range(layers)]
    )
    inputs = {
        "text_inputs": prompt.texts,
        "audio_embeds": [torch.zeros(prompt.audio_rows, HIDDEN)],
        "vision_embeds": [torch.zeros(prompt.image_rows, HIDDEN), torch.zeros(prompt.video_rows, HIDDEN)],
        "deepstack": deepstack,
        "image_grid_thw": [prompt.image_grid, prompt.video_grid],
        "video_second_per_grid": [prompt.seconds],
    }
    info = SimpleNamespace(rid_handle=0, step_metadata={"prefill_order": prompt.order})
    with torch.no_grad():
        return _thinker(config).prepare_inputs("prefill_audio_vision", info, inputs)


def test_the_prefill_row_has_hfs_positions_for_the_whole_prompt(config):
    prompt = _prompt(config)
    row = _row(config, prompt)
    hf = _hf_positions(
        config, prompt.ids, prompt.image_grid[None], prompt.video_grid[None], prompt.seqlens, prompt.seconds,
    )

    assert row.input_seq_len == len(prompt.ids)
    assert torch.equal(row.custom_pos_ids, hf.float())
    # decode goes on from one past the prompt's largest position
    assert row.tensor_inputs["mrope_pos_advance"] == int(hf.max()) + 1


def test_the_prefill_row_marks_and_fills_the_encoder_rows(config):
    prompt = _prompt(config)
    row = _row(config, prompt)
    tc = config.thinker
    pads = torch.isin(prompt.ids, torch.tensor([tc.audio_token_id, tc.image_token_id, tc.video_token_id]))
    vision = torch.isin(prompt.ids, torch.tensor([tc.image_token_id, tc.video_token_id]))

    assert torch.equal(row.tensor_inputs["masks_for_talker"][0], pads)
    for layer in row.tensor_inputs["deepstack"]:
        assert torch.equal(layer.abs().sum(-1) > 0, vision)
    # the image's features, then the video's, each on its own rows
    first = row.tensor_inputs["deepstack"][0][vision][:, 0]
    assert first.tolist() == [1.0] * prompt.image_rows + [10.0] * prompt.video_rows
    assert torch.equal(row.tensor_inputs["text_token_ids"], torch.cat(prompt.texts))


def test_the_layout_counts_what_the_prefill_row_and_the_talker_write(config, model):
    prompt = _prompt(config)
    row = _row(config, prompt)
    tensors = {
        "text_inputs": prompt.texts, "audio_seqlens": [prompt.seqlens], "image_grid_thw": [prompt.image_grid],
        "video_grid_thw": [prompt.video_grid], "video_second_per_grid": [prompt.seconds],
    }
    plan = prefill_plan(parts_from_modalities(prompt.mods))
    layout = model._kv_layout(plan, tensors, audio_output=True)

    thinker = layout[THINKER_KV]["main"].spans
    assert {span.walk for span in thinker} == {"prefill_audio_vision"}
    assert sum(span.length for span in thinker) == row.input_seq_len
    assert sum(span.advance for span in thinker) == row.tensor_inputs["mrope_pos_advance"]

    multimodal, text = row.tensor_inputs["masks_for_talker"].bool()
    kept = int((multimodal | text).sum())
    talker = layout[TALKER_KV]["main"].spans
    assert {span.walk for span in talker} == {"talker_prefill"}
    assert sum(span.length for span in talker) == kept + TalkerSubmodule.ASSISTANT_PREFIX_ROWS


# --- the Thinker's states for the Talker --------------------------------------


# a prefill's masks: an encoder row, a system-turn row the Talker drops, two kept text rows
PREFILL_MASK = torch.tensor([[True, False, False, False], [False, False, True, True]])
# a decode step's: one text row
DECODE_MASK = torch.tensor([[False], [True]])


def _postprocess(config, walk, *, iteration=0, audio_output=True, first=True):
    thinker = _thinker(config)
    hidden = config.thinker_hidden_size
    mask = DECODE_MASK if walk == "thinker_decode" else PREFILL_MASK
    rows = mask.shape[1]
    outputs = {
        "new_token": [torch.tensor([5])],
        "thinker_states": [torch.arange(rows * 2 * hidden, dtype=torch.float).view(rows, 2 * hidden)],
        "thinker_mask": [mask],
    }
    if first:
        outputs["first_token_embed"] = [torch.full((1, hidden), -1.0)]
    info = SimpleNamespace(
        graph_walk=walk, step_metadata={"audio_output": audio_output},
        dynamic_loop_iter_counts={"thinker_decode_loop": iteration} if iteration else {},
    )
    thinker.postprocess(0, info, outputs)
    return outputs


def test_the_prefill_sends_its_kept_rows_then_the_first_tokens(config):
    out = _postprocess(config, "prefill_text")
    states, mask = out["thinker_states"][0], out["thinker_mask"][0]

    assert "first_token_embed" not in out
    assert states.shape[0] == 3 + 1
    assert torch.equal(states[-1], torch.full((config.thinker_hidden_size,), -1.0))
    assert mask.tolist() == [True, False, False, False]
    assert out["text_inputs"] == out["new_token"]


def test_the_first_decode_step_sends_nothing_the_prefill_sent(config):
    first = _postprocess(config, "thinker_decode", first=False)
    later = _postprocess(config, "thinker_decode", iteration=1, first=False)

    assert "thinker_states" not in first and "thinker_mask" not in first
    assert later["thinker_states"][0].shape[0] == 1
    assert "new_token" in first


def test_a_decode_row_of_a_mixed_step_drops_the_first_token_embed(config):
    out = _postprocess(config, "thinker_decode", iteration=3)
    assert "first_token_embed" not in out
    assert out["thinker_states"][0].shape[0] == 1


def test_a_text_only_request_sends_the_talker_nothing(config):
    out = _postprocess(config, "prefill_text", audio_output=False)
    assert set(out) == {"new_token", "text_inputs"}


# --- the Talker's one prefill step --------------------------------------------


class _TalkerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.codec_embedding = nn.Embedding(config.talker_text.vocab_size, config.talker_hidden_size)
        self.text_projection = nn.Linear(config.thinker_hidden_size, config.talker_hidden_size)
        self.hidden_projection = nn.Linear(config.thinker_hidden_size, config.talker_hidden_size)


def _talker(config) -> TalkerSubmodule:
    config = copy.deepcopy(config)
    config.talker.speaker_id = {"ethan": 2}
    talker = TalkerSubmodule(_TalkerModel(config), nn.Module(), config)
    talker._tts_pad_embed_cached = torch.randn(config.talker_hidden_size)
    talker._tts_bos_embed_cached = torch.randn(config.talker_hidden_size)
    return talker


def test_the_talker_prefills_the_prompt_and_its_assistant_prefix_in_one_step(config):
    talker = _talker(config)
    states = torch.randn(5, config.thinker_hidden_size)
    multimodal = torch.tensor([False, True, True, False, False])
    info = SimpleNamespace(rid_handle=0, step_metadata={"voice": "Ethan"})

    with torch.no_grad():
        row = talker.prepare_inputs("talker_prefill", info, {"thinker_states": [states], "thinker_mask": [multimodal]})
        prompt = talker._get_talker_embeds(states[:-1], multimodal[:-1])
        prefix = talker._assistant_prefix("Ethan", states[-1:])

    assert row.input_seq_len == 4 + TalkerSubmodule.ASSISTANT_PREFIX_ROWS
    assert torch.equal(row.input_embeds, torch.cat([prompt, prefix]))
    # the first token's row is projected as text, last in the prefix
    assert torch.equal(prefix[-1:], talker.talker_code_emb(torch.tensor([config.talker.codec_bos_id]))
                       + talker.model.text_projection(states[-1:]))

    step = talker.declare_step("talker_prefill", [0], [row])
    assert {TALKER_SAMPLER, CODE_PRED_SAMPLER} <= set(step.steps)
    assert [segment.span for segment in step.segments] == [row.input_seq_len]


def test_the_talker_prefill_is_one_packed_capture(config):
    configs = _talker(config).get_accelerator_graph_configs(torch.device("cpu"))
    walks = {c.capture_graph_walk: c for c in configs}

    assert set(walks) == {"talker_decode", "talker_prefill"}
    assert isinstance(walks["talker_prefill"], PackedAcceleratorGraphConfig)
    # requests that reach their prefill together share the step
    assert _talker(config).can_batch(SimpleNamespace(graph_walk="talker_prefill"), [object(), object()])


# --- the encoders, one call per attachment --------------------------------------


def test_the_audio_encoder_returns_one_embedding_per_clip():
    encoder = Qwen3OmniMoeAudioEncoder._from_config(
        Qwen3OmniMoeAudioEncoderConfig(
            encoder_layers=1, d_model=32, encoder_attention_heads=2, encoder_ffn_dim=64, output_dim=16,
            downsample_hidden_size=8,
        ),
        attn_implementation="sdpa",
    ).eval()
    audio = AudioEncoderSubmodule(encoder, config=None)
    clips = [torch.randn(128, 60), torch.randn(128, 600)]
    seqlens = [torch.tensor([60]), torch.tensor([600])]
    inputs = audio.prepare_inputs("prefill_audio", None, {"audio_features": clips, "audio_seqlens": seqlens})

    with torch.no_grad():
        out = audio.forward("prefill_audio", None, **inputs.tensor_inputs)
        alone = [encoder(clip, feature_lens=lens).last_hidden_state for clip, lens in zip(clips, seqlens, strict=True)]

    assert len(out["audio_embeds"]) == 2
    assert all(torch.equal(ours, one) for ours, one in zip(out["audio_embeds"], alone, strict=True))


def test_the_vision_encoder_returns_each_ones_embeddings_and_deepstack_in_order():
    encoder = Qwen3OmniMoeVisionEncoder._from_config(
        Qwen3OmniMoeVisionEncoderConfig(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=16,
            num_position_embeddings=64, deepstack_visual_indexes=[0, 1],
        ),
        attn_implementation="sdpa",
    ).eval()
    vision = VisionEncoderSubmodule(encoder, config=None)
    grids = [torch.tensor([1, 4, 6]), torch.tensor([1, 2, 2])]
    pixels = [torch.randn(int(grid.prod()), 3 * 2 * 16 * 16) for grid in grids]
    inputs = vision.prepare_inputs("prefill_vision", None, {"pixel_values": pixels, "image_grid_thw": grids})

    with torch.no_grad():
        out = vision.forward("prefill_vision", None, **inputs.tensor_inputs)
        alone = [encoder(item, grid_thw=grid[None]) for item, grid in zip(pixels, grids, strict=True)]

    assert all(torch.equal(ours, one.pooler_output) for ours, one in zip(out["vision_embeds"], alone, strict=True))
    expected = [layer for one in alone for layer in one.deepstack_features]
    assert len(out["deepstack"]) == len(expected)
    assert all(torch.equal(ours, layer) for ours, layer in zip(out["deepstack"], expected, strict=True))

