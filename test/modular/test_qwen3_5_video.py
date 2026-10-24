"""Qwen3.5 video input against transformers, on CPU with no weights.

transformers' ``Qwen3VLProcessor`` / ``Qwen3VLVideoProcessor`` and
``Qwen3_5Model.get_rope_index`` are the reference: the expanded prompt ids
(timestamps, sentinels, pads), the video patches and grids, and the 3D MRoPE
positions of a mixed text + image + video prompt must equal theirs exactly.

The checkpoint is a synthetic one built in ``tmp_path`` (a byte-level
tokenizer with Qwen's special tokens, a minimal chat template, the released
processor configs, a tiny model config), so the suite runs offline. Set
``QWEN3_5_CKPT`` to a Qwen3.5 / Qwen3.8 checkpoint directory (metadata files
suffice) to also run it on the real tokenizer and template.

The model-level tests import the engine, so they need Triton (Linux CI).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.model.multimodal import (
    MEDIA_LAYOUT,
    TEXT,
    VIDEO,
    PromptPart,
    parts_from_modalities,
)
from mstar.model.qwen3_5.video import (
    expand_video_pads,
    ffmpeg_unavailable,
    group_timestamps,
)

transformers = pytest.importorskip("transformers")

# transformers 5.10 changed how Qwen3VLProcessor lays out a video: it replaces
# only the pad and keeps the template's outer sentinels (vLLM and SGLang do the
# same); earlier versions replaced the whole triple. mstar follows >= 5.10, and
# `Qwen3_5Model` (for `get_rope_index`) only exists from 5.x.
_NEW_LAYOUT = int(transformers.__version__.split(".")[0]) >= 6 or tuple(
    int(v) for v in transformers.__version__.split(".")[:2]
) >= (5, 10)
needs_new_layout = pytest.mark.skipif(
    not _NEW_LAYOUT,
    reason=f"transformers {transformers.__version__} predates 5.10's video layout and Qwen3.5 classes",
)

SPECIAL_TOKENS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
    "<think>", "</think>",
]

# What Qwen3.5's template does with content parts, and nothing else.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{ '<|im_start|>' + message.role + '\n' }}"
    "{%- for item in message.content -%}"
    "{%- if item.type == 'image' -%}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{%- elif item.type == 'video' -%}{{ '<|vision_start|><|video_pad|><|vision_end|>' }}"
    "{%- elif item.type == 'text' -%}{{ item.text }}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{{ '<|im_end|>\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\n<think>\n' }}{%- endif -%}"
)

# The released checkpoints' processor configs (Qwen/Qwen3.5-0.8B).
_PROCESSOR = {
    "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
}
PREPROCESSOR_CONFIG = {
    **_PROCESSOR, "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "image_processor_type": "Qwen2VLImageProcessorFast",
}
VIDEO_PREPROCESSOR_CONFIG = {
    **_PROCESSOR, "size": {"longest_edge": 25165824, "shortest_edge": 4096},
    "video_processor_type": "Qwen3VLVideoProcessor",
}


def tiny_config(token_ids: dict[str, int], vocab_size: int) -> dict:
    """A Qwen3.5 ``config.json`` both mstar and transformers parse; tiny so
    ``Qwen3_5Model`` builds on CPU for ``get_rope_index``."""
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "image_token_id": token_ids["<|image_pad|>"],
        "video_token_id": token_ids["<|video_pad|>"],
        "vision_start_token_id": token_ids["<|vision_start|>"],
        "vision_end_token_id": token_ids["<|vision_end|>"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "attn_output_gate": True,
            "eos_token_id": token_ids["<|endoftext|>"],
            "head_dim": 32, "hidden_size": 64, "intermediate_size": 128,
            "layer_types": ["linear_attention", "full_attention"],
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 16,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_value_head_dim": 16,
            "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
            "vocab_size": vocab_size,
            "rope_parameters": {
                "mrope_interleaved": True, "mrope_section": [2, 1, 1],
                "rope_type": "default", "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        },
        "vision_config": {
            "model_type": "qwen3_5", "deepstack_visual_indexes": [], "depth": 1,
            "hidden_act": "gelu_pytorch_tanh", "hidden_size": 32, "in_channels": 3,
            "intermediate_size": 64, "num_heads": 2, "num_position_embeddings": 16,
            "out_hidden_size": 64, "patch_size": 16, "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }


def write_synthetic_checkpoint(root: Path) -> Path:
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    byte_chars = pre_tokenizers.ByteLevel.alphabet()
    vocab = {c: i for i, c in enumerate(sorted(byte_chars))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in SPECIAL_TOKENS])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(root)

    ids = {t: fast.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
    (root / "config.json").write_text(json.dumps(tiny_config(ids, len(fast))))
    (root / "preprocessor_config.json").write_text(json.dumps(PREPROCESSOR_CONFIG))
    (root / "video_preprocessor_config.json").write_text(json.dumps(VIDEO_PREPROCESSOR_CONFIG))
    return root


CHECKPOINTS = ["synthetic"] + (["real"] if os.environ.get("QWEN3_5_CKPT") else [])


@pytest.fixture(scope="module", params=CHECKPOINTS)
def ckpt(request, tmp_path_factory) -> Path:
    if request.param == "real":
        return Path(os.environ["QWEN3_5_CKPT"])
    return write_synthetic_checkpoint(tmp_path_factory.mktemp("qwen3_5_synthetic"))


@pytest.fixture(scope="module")
def model(ckpt):
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    return Qwen3_5DenseModel(str(ckpt))


@pytest.fixture(scope="module")
def processor(ckpt):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(ckpt))


@pytest.fixture(scope="module")
def rope_model():
    """``get_rope_index`` is a method; only the vision config's merge size
    is read, so a tiny random model stands in for any checkpoint."""
    from transformers import Qwen3_5Config, Qwen3_5Model

    config = Qwen3_5Config(**tiny_config(
        {t: i for i, t in enumerate(SPECIAL_TOKENS)}, len(SPECIAL_TOKENS) + 256,
    ))
    return Qwen3_5Model(config).eval()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# (frames, source fps, height, width): sampled at 2 fps, clamped to [4, 768]
VIDEOS = {
    # 2.5 s -> 5 frames: an odd count, so the last group repeats a frame
    "odd": (25, 10.0, 64, 96),
    # 5 s -> 10 frames, 5 groups
    "even": (40, 8.0, 48, 80),
    # 0.3 s -> under min_frames, so 4 frames, 2 groups
    "short": (9, 30.0, 70, 50),
}


def a_video(name: str) -> tuple[torch.Tensor, float]:
    n, fps, h, w = VIDEOS[name]
    gen = torch.Generator().manual_seed(len(name))
    return torch.randint(0, 256, (n, 3, h, w), dtype=torch.uint8, generator=gen), fps


def sampled(video: torch.Tensor, fps: float, video_processor) -> tuple[torch.Tensor, dict]:
    """What `Qwen3_5DenseModel.load_video` returns for a file of these frames
    (`test_mp4_matches_processor` checks the decode itself)."""
    from transformers.video_utils import VideoMetadata

    from mstar.model.base import video_metadata_dict

    metadata = VideoMetadata(total_num_frames=video.shape[0], fps=fps)
    indices = video_processor.sample_frames(
        metadata, num_frames=video_processor.num_frames, fps=video_processor.fps,
    )
    metadata.frames_indices = indices
    return video[torch.as_tensor(indices)], video_metadata_dict(metadata)


def an_image() -> torch.Tensor:
    # what the data worker sends: (C, H, W) float32 in [0, 1]
    return torch.rand(3, 40, 56, generator=torch.Generator().manual_seed(1))


LAYOUTS = {
    "video_then_question": (["video", "text"], ["odd"]),
    "question_then_video": (["text", "video"], ["even"]),
    "mixed": (["text", "image", "text", "video", "text"], ["odd"]),
    "two_videos_around_image": (["video", "image", "video", "text"], ["short", "even"]),
}


def run_both(model, processor, layout: str):
    """mstar's `process_prompt` and the transformers processor on one prompt."""
    from mstar.model.qwen3_5.qwen3_5_model import _as_hwc_uint8

    modalities, video_names = LAYOUTS[layout]
    texts = [f"Question {i}: what happens?" for i in range(modalities.count(TEXT))]
    parts = parts_from_modalities(modalities, texts)
    images = [an_image() for _ in range(modalities.count("image"))]
    videos = [a_video(name) for name in video_names]
    frames, metadata = zip(*(sampled(v, fps, model.video_processor) for v, fps in videos), strict=True)

    out = model.process_prompt(
        "\n".join(texts), modalities, ["text"],
        tensors={"image_inputs": images, "video_inputs": list(frames)},
        prompt_parts=parts,
        input_metadata={"video_inputs": list(metadata)},
    )

    content = [
        {"type": TEXT, "text": p.text} if p.modality == TEXT else {"type": p.modality}
        for p in parts
    ]
    text = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    ref = processor(
        text=[text],
        images=[_as_hwc_uint8(i) for i in images] or None,
        videos=[v for v, _ in videos] or None,
        video_metadata=[{"total_num_frames": v.shape[0], "fps": fps} for v, fps in videos] or None,
        return_tensors="pt",
    )
    return parts, out, ref


def by_modality(modalities, tensors) -> dict[str, list]:
    """The prompt-ordered tower inputs split back into images and videos."""
    split = {"image": ([], []), VIDEO: ([], [])}
    ordered = zip(tensors.get("pixel_values", []), tensors.get("image_grid_thw", []), strict=True)
    for m in modalities:
        if m != TEXT:
            pixels, grid = next(ordered)
            split[m][0].append(pixels)
            split[m][1].append(grid)
    return split


# ---------------------------------------------------------------------------
# Pure helpers (no engine import)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_group_timestamps_match_processor(n):
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    indices = [round(i * 7.3) for i in range(n)]
    ref = Qwen3VLProcessor._calculate_timestamps(None, list(indices), 29.97, 2)
    assert group_timestamps(indices, 29.97, 2) == ref


@pytest.mark.parametrize("grid", [[(1, 6, 10)], [(1, 4, 4), (2, 8, 2)], [(1, 2, 2)]], ids=str)
def test_the_position_resample_matches_interpolate(grid):
    """The learned table resampled onto each frame's grid, as upstream's
    ``F.interpolate(align_corners=True)`` (edge reads, no zero padding),
    in the merger's block order."""
    import torch.nn.functional as F

    from mstar.model.qwen3_5.components.vision import vision_interpolation

    side, merge, dim = 4, 2, 3
    table = torch.randn(side * side, dim, dtype=torch.float64)
    idx, w = vision_interpolation(grid, side, merge, torch.device("cpu"))
    ours = (table[idx] * w[:, :, None].double()).sum(1)
    refs = []
    for t, h, wd in grid:
        img = F.interpolate(
            table.T.reshape(1, dim, side, side), size=(h, wd), mode="bilinear", align_corners=True,
        )[0]
        # raster -> merge blocks: (block row, block col, row in block, col in block)
        blocks = img.reshape(dim, h // merge, merge, wd // merge, merge).permute(1, 3, 2, 4, 0)
        refs.append(blocks.reshape(-1, dim).repeat(t, 1))
    assert torch.allclose(ours, torch.cat(refs), atol=1e-6)


def test_expand_video_pads_writes_a_triple_per_group():
    text = "a<|vision_start|><|video_pad|><|vision_end|>b<|video_pad|>c"
    out = expand_video_pads(
        text, "<|video_pad|>", "<|vision_start|>", "<|vision_end|>", [[0.25, 1.5], [3.0]],
    )
    assert out == (
        "a<|vision_start|><0.2 seconds><|vision_start|><|video_pad|><|vision_end|>"
        "<1.5 seconds><|vision_start|><|video_pad|><|vision_end|><|vision_end|>b"
        "<3.0 seconds><|vision_start|><|video_pad|><|vision_end|>c"
    )
    with pytest.raises(ValueError, match="2 video placeholders for 1 videos"):
        expand_video_pads(text, "<|video_pad|>", "<|vision_start|>", "<|vision_end|>", [[0.0]])


# ---------------------------------------------------------------------------
# Against transformers
# ---------------------------------------------------------------------------


@needs_new_layout
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_prompt_ids_match_processor(model, processor, layout):
    _, out, ref = run_both(model, processor, layout)
    (ids,) = out.new_input_tensors["text_inputs"]
    assert torch.equal(ids, ref["input_ids"][0])


def test_an_attachment_prompt_keys_nothing_for_the_prefix_cache(model, processor):
    """An attachment prompt's one `prefill_vision` step carries tensor inputs, which the prefix
    cache never cuts, and a keyed span in that walk makes the preprocess worker refuse the request."""
    from mstar.model.qwen3_5.config import KV_CACHE

    _, out, _ = run_both(model, processor, "mixed")
    assert out.metadata["kv_layout"][KV_CACHE]["main"].keyed_count() == 0


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_patches_and_grids_match_processor(model, processor, layout):
    _, out, ref = run_both(model, processor, layout)
    tensors = out.new_input_tensors
    split = by_modality(LAYOUTS[layout][0], tensors)
    assert torch.equal(torch.stack(split[VIDEO][1]), ref["video_grid_thw"])
    assert torch.equal(torch.cat(split[VIDEO][0]), ref["pixel_values_videos"])
    if split["image"][0]:
        assert torch.equal(torch.stack(split["image"][1]), ref["image_grid_thw"])
        assert torch.equal(torch.cat(split["image"][0]), ref["pixel_values"])
    # the raw frames are not kept past preprocessing
    assert tensors["video_inputs"] == []


IMAGE_MARK, VIDEO_MARK = -1.0, -2.0


@needs_new_layout
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_vision_walk_matches_get_rope_index(model, processor, rope_model, layout):
    """The schedule and the LLM's media row, against transformers: the
    spliced row's token order and its 3D positions, plus where decode
    resumes."""
    from mstar.model.components.sampled_token_decoder import splice_media
    from mstar.model.qwen3_5.config import KV_CACHE, ROPE
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    _, out, ref = run_both(model, processor, layout)
    tensors = out.new_input_tensors
    modalities, _ = LAYOUTS[layout]
    (step,) = model._prefill_schedule(modalities, tensors, out.metadata[MEDIA_LAYOUT])
    assert step.walk == "prefill_vision"
    # the encoder gets each image and video once, in prompt order
    assert all(a is b for a, b in zip(
        step.input_tensors[("vision_encoder", "pixel_values")], tensors["pixel_values"], strict=True,
    ))

    merge = model.vision_config.spatial_merge_size ** 2
    kinds = [m for m in modalities if m != TEXT]
    marks = torch.cat([
        torch.full((int(g.prod()) // merge, 1), VIDEO_MARK if kind == VIDEO else IMAGE_MARK)
        for g, kind in zip(tensors["image_grid_thw"], kinds, strict=True)
    ])
    (encoder,) = model.encoders
    llm = LLMSubmodule(nn.Linear(1, 1), model.config, model.vision_config, model.encoders)
    llm.node_resources = {ROPE: SimpleNamespace(position=lambda rid, label: 0)}
    fwd_info = SimpleNamespace(request_id="r", step_metadata=step.metadata)
    (ids,) = step.input_tensors[("LLM", "text_inputs")]
    row = llm._media_inputs(fwd_info, {"text_inputs": [ids], "vision_embeds": [marks]})

    expected_row = ref["input_ids"][0].float()
    expected_row[ref["input_ids"][0] == model.config.image_token_id] = IMAGE_MARK
    expected_row[ref["input_ids"][0] == model.config.video_token_id] = VIDEO_MARK
    spliced = splice_media(row.input_ids.float()[:, None], row.input_ids, marks, encoder.pad_ids)
    assert torch.equal(spliced[:, 0], expected_row)

    positions, _ = rope_model.get_rope_index(
        ref["input_ids"], ref["mm_token_type_ids"],
        image_grid_thw=ref.get("image_grid_thw"), video_grid_thw=ref["video_grid_thw"],
    )
    assert torch.equal(row.custom_pos_ids.long(), positions[:, 0])
    # decode resumes where transformers' rope delta puts it
    assert row.resource_step_info == int(positions.max()) + 1
    spans = out.metadata["kv_layout"][KV_CACHE]["main"].spans
    assert sum(span.length for span in spans) == row.input_seq_len
    assert sum(span.advance for span in spans) == row.resource_step_info


def test_mp4_matches_processor(model, processor, tmp_path):
    """The decode itself: an mp4 through `load_video`, against the processor
    reading the same file."""
    if reason := ffmpeg_unavailable():
        pytest.skip(f"torchcodec with FFmpeg is the decoder under test; {reason}")
    av = pytest.importorskip("av")

    video, fps = a_video("odd")
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.height, stream.width = video.shape[-2:]
        stream.pix_fmt = "yuv420p"
        for frame in video.permute(0, 2, 3, 1).numpy():
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())

    loaded = model.load_video(str(path), "cpu")
    out = model.process_prompt(
        "What happens?", ["video", "text"], ["text"],
        tensors={"video_inputs": [loaded.data]},
        input_metadata={"video_inputs": [loaded.metadata]},
    ).new_input_tensors
    ref = processor.video_processor(videos=[str(path)], return_tensors="pt")
    assert torch.equal(out["image_grid_thw"][0], ref["video_grid_thw"][0])
    assert torch.equal(out["pixel_values"][0], ref["pixel_values_videos"])


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_more_frame_groups_than_a_replay_holds_is_accepted(model):
    """Many tiny frame groups fit the top patch bucket but overflow a replay's
    segment ceiling; the encoder runs them eagerly, so intake takes them."""
    from mstar.model.qwen3_5.submodules import VisionEncoderSubmodule

    (spec,) = model._vision_resources()
    limit = spec.config.max_segments_per_request
    n = 2 * (limit + 1)
    video = torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8)
    frames, metadata = sampled(video, 2.0, model.video_processor)
    out = model.process_prompt(
        "", ["video"], ["text"],
        tensors={"video_inputs": [frames]},
        input_metadata={"video_inputs": [metadata]},
    )
    (grid,) = out.new_input_tensors["image_grid_thw"]
    assert int(grid[0]) > limit
    assert int(grid.prod()) <= VisionEncoderSubmodule.BLOCK_LOOP_PATCH_BUCKETS[-1]


def _encoder(max_segments: int):
    """A vision encoder bound to a ragged attention resource with this ceiling."""
    from mstar.model.multimodal import vision_encoder
    from mstar.model.qwen3_5.config import VISION_ATTN, Qwen3_5VisionConfig
    from mstar.model.qwen3_5.submodules import VisionEncoderSubmodule

    config = Qwen3_5VisionConfig(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3,
        patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
        num_position_embeddings=16, out_hidden_size=64,
    )
    encoder = VisionEncoderSubmodule(nn.Linear(1, 1), config, vision_encoder((0,)))
    encoder.node_resources = {VISION_ATTN: SimpleNamespace(max_segments_per_request=max_segments)}
    return encoder


def test_vision_block_loop_leases_only_what_a_replay_holds():
    """The captured block loop declares `can_lease`: a request whose
    segments exceed a replay's plan gets no slot."""
    from mstar.model.components.block_loop_encoder import VISION_BLOCK_LOOP
    from mstar.model.submodule_base import NodeInputs

    encoder = _encoder(max_segments=4)
    config = encoder.get_piecewise_accelerator_graph_configs(
        torch.device("cpu"), torch.bfloat16,
    )[VISION_BLOCK_LOOP]

    def request(segments: int) -> NodeInputs:
        return NodeInputs(input_seq_len=16 * segments, resource_step_info={"main": (16,) * segments})

    assert config.lease_before_step
    assert config.can_lease([request(4)])
    assert not config.can_lease([request(5)])


def test_vision_block_loop_replays_exactly_when_leased():
    """A leased step declares nothing of its own, so the forward replays
    then and runs eagerly otherwise, whatever the runner's buckets allow; a
    replay plans the frames under the one label."""
    from mstar.model.components.block_loop_encoder import VISION_BLOCK_LOOP

    class Runner:
        has_bucket = True

        def can_run(self, batch_size, total_tokens):
            return self.has_bucket

        def run(self, **kwargs):
            self.ran = kwargs
            return SimpleNamespace(get_view=lambda name: "replayed")

    encoder, runner = _encoder(max_segments=4), Runner()
    hidden = cos = sin = torch.zeros(8, 32)

    def replay(step):
        engine_inputs = SimpleNamespace(
            piecewise_runners={VISION_BLOCK_LOOP: runner}, request_ids=["r"], step=step,
        )
        return encoder._replay_block_loop(engine_inputs, hidden, cos, sin, {"main": (4, 4)})

    assert replay(step=object()) is None
    assert not hasattr(runner, "ran")
    # no declaration but no bucket either (a forward driven outside the engine)
    runner.has_bucket = False
    assert replay(step=None) is None
    assert not hasattr(runner, "ran")
    runner.has_bucket = True
    assert replay(step=None) == "replayed"
    assert runner.ran["seq_lens"] == [4, 4]
    assert runner.ran["label_seq_lens"] == {"main": [4, 4]}


def test_the_replayed_region_declares_what_the_eager_step_would():
    """One declaration serves the eager step and the region: at capture
    (no labels) each label plans the bucket's span, at replay the frames."""
    from mstar.model.components.block_loop_encoder import VISION_BLOCK_LOOP
    from mstar.model.qwen3_5.config import VISION_ATTN
    from mstar.model.submodule_base import NodeInputs

    encoder = _encoder(max_segments=4)
    region = encoder.get_piecewise_accelerator_graph_configs(
        torch.device("cpu"), torch.bfloat16,
    )[VISION_BLOCK_LOOP]

    def spans(step):
        return [(seg.request_id, seg.label, seg.span) for seg in step.segments]

    eager = encoder.declare_step(
        "prefill_vision", ["r"], [NodeInputs(input_seq_len=8, resource_step_info={"main": (4, 4)})],
    )
    replayed = region.declare_step(["r"], [8], label_seq_lens={"main": [4, 4]})
    assert spans(eager) == spans(replayed) == [("r", "main", 4), ("r", "main", 4)]
    assert set(eager.steps) == set(replayed.steps) == {VISION_ATTN}
    assert spans(region.declare_step(["r"], [256])) == [("r", "main", 256)]
    # leased, the region declares the node's whole step
    leased = encoder.declare_step(
        "prefill_vision", ["r"], [NodeInputs(input_seq_len=8, resource_step_info={"main": (4, 4)})],
        piecewise_leases={VISION_BLOCK_LOOP: object()},
    )
    assert leased is None


def test_a_bad_layout_is_refused_before_the_video_is_preprocessed(model, monkeypatch):
    """A literal `<|video_pad|>` in the user's text leaves the prompt with
    more video placeholders than videos; that 400s before the processor
    spends any time resizing frames."""
    video, fps = a_video("odd")
    frames, metadata = sampled(video, fps, model.video_processor)

    def must_not_run(*args, **kwargs):
        raise AssertionError("the video was preprocessed before the layout was checked")

    monkeypatch.setattr(model, "_preprocess_video", must_not_run)
    with pytest.raises(ValueError, match="video placeholders"):
        model.process_prompt(
            "look: <|video_pad|>", ["text", "video"], ["text"],
            tensors={"video_inputs": [frames]},
            input_metadata={"video_inputs": [metadata]},
        )


def test_an_undecodable_video_is_a_client_error(model, tmp_path):
    """A corrupt upload is the client's input: ValueError (a 400), not the
    decoder's RuntimeError (a 500)."""
    if reason := ffmpeg_unavailable():
        pytest.skip(f"torchcodec with FFmpeg is the decoder under test; {reason}")
    from mstar.model.qwen3_5.video import sample_video

    path = tmp_path / "broken.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"not a video" * 64)
    with pytest.raises(ValueError, match="could not decode"):
        sample_video(str(path), model.video_processor)


def _header_only_loader(monkeypatch, height, width, total=600, fps=30.0) -> list[int]:
    """transformers' torchcodec reader without torchcodec: a header of this
    size, then the sampler, then a decode recorded as its frame count."""
    from transformers import video_utils
    from transformers.video_utils import VideoMetadata

    from mstar.model.qwen3_5 import video as video_module

    decoded = []

    def load_video(path, backend, sample_indices_fn):
        metadata = VideoMetadata(total_num_frames=total, fps=fps, height=height, width=width)
        indices = sample_indices_fn(metadata=metadata)
        decoded.append(len(indices))
        metadata.frames_indices = indices
        return torch.zeros(len(indices), 3, 1, 1, dtype=torch.uint8), metadata

    monkeypatch.setattr(video_utils, "load_video", load_video)
    monkeypatch.setattr(video_module, "ffmpeg_unavailable", lambda: None)
    return decoded


def test_the_video_pixel_cap_holds_before_decoding(model, tmp_path, monkeypatch):
    """`MSTAR_MAX_VIDEO_PIXELS` counts the frames the processor samples, from
    the header, so a video over it is refused before any frame is decoded."""
    from transformers.video_utils import VideoMetadata

    from mstar.model.qwen3_5.video import sample_video

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"")
    decoded = _header_only_loader(monkeypatch, height=1080, width=1920)
    processor = model.video_processor
    n = len(processor.sample_frames(
        VideoMetadata(total_num_frames=600, fps=30.0), num_frames=processor.num_frames, fps=processor.fps,
    ))

    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(n * 1080 * 1920))
    sample_video(str(path), processor)
    assert decoded == [n]

    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(n * 1080 * 1920 - 1))
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
        sample_video(str(path), processor)
    assert decoded == [n]


def test_a_video_with_no_frame_size_is_refused(model, tmp_path, monkeypatch):
    """Without a frame size the decode can't be bounded, so it doesn't start."""
    from mstar.model.qwen3_5.video import sample_video

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"")
    decoded = _header_only_loader(monkeypatch, height=None, width=None)
    with pytest.raises(ValueError, match="no frame size"):
        sample_video(str(path), model.video_processor)
    assert decoded == []


def test_still_refuses_audio(model):
    with pytest.raises(ValueError, match="no audio path"):
        model.process_prompt(
            "hi", ["audio", "text"], ["text"], tensors={"audio_inputs": [torch.zeros(16)]},
        )


def test_video_needs_its_sampling_metadata(model):
    video, fps = a_video("odd")
    frames, _ = sampled(video, fps, model.video_processor)
    with pytest.raises(ValueError, match="sampling metadata"):
        model.process_prompt("hi", ["video", "text"], ["text"], tensors={"video_inputs": [frames]})


def test_the_layout_places_every_pad(model, processor):
    """Each laid-out run covers exactly its pads in the expanded prompt, and
    the runs' tokens sum to the tower's merged tokens."""
    _, out, _ = run_both(model, processor, "two_videos_around_image")
    (ids,) = out.new_input_tensors["text_inputs"]
    layout = out.metadata[MEDIA_LAYOUT]
    pads = set(model.encoders[0].pad_ids)
    covered = torch.zeros_like(ids, dtype=torch.bool)
    for offset, n in zip(layout["offset"], layout["tokens"], strict=True):
        assert all(int(t) in pads for t in ids[offset:offset + n])
        covered[offset:offset + n] = True
    assert torch.equal(covered, torch.isin(ids, torch.tensor(sorted(pads))))
    merge = model.vision_config.spatial_merge_size ** 2
    assert sum(layout["tokens"]) == sum(int(g.prod()) // merge for g in out.new_input_tensors["image_grid_thw"])


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


def test_adapter_maps_video_url_to_a_video_attachment(tmp_path):
    import base64

    from mstar.api_server.openai.adapters import Qwen3_5Adapter
    from mstar.api_server.openai.protocol import ChatCompletionRequest

    data = base64.b64encode(b"\x00\x00\x00\x18ftypmp42").decode()
    req = ChatCompletionRequest(messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens?"},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{data}"}},
            {"type": "text", "text": "Answer briefly."},
        ],
    }])
    args = Qwen3_5Adapter().chat_to_request(req, tmp_path)
    assert args.input_modalities == ["text", "video", "text"]
    assert list(args.file_paths) == ["video"]
    (path,) = args.file_paths["video"]
    assert Path(path).suffix == ".mp4" and Path(path).read_bytes().startswith(b"\x00\x00\x00\x18")
    assert [p.modality for p in args.prompt_parts] == ["text", "video", "text"]
    assert args.prompt_parts[1] == PromptPart(modality="video", index=0, role="user")
