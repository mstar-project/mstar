"""Qwen3-Omni's Thinker keys its prompt past images and audio, and a hit cuts a walk short."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch._dynamo.utils import counters
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeVisionEncoder,
)

from mstar.engine.engine import Engine
from mstar.engine.resources import PositionStep, Resource, StepContext, StepRunner
from mstar.engine.resources.base import CachedPrefix
from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager
from mstar.model.qwen3_omni.components.rope import get_rope_index_vision
from mstar.model.qwen3_omni.config import THINKER_KV, THINKER_MIXED, THINKER_POS
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel, image_span
from mstar.model.qwen3_omni.submodules import AudioEncoderSubmodule, ThinkerSubmodule, VisionEncoderSubmodule

RID = "r0"
NODE = "Thinker"
HIDDEN = 8

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="flash-attn runs on a GPU")


class _StubThinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, HIDDEN)


class _Holding(Resource):

    supports_preplan = True

    def __init__(self, tokens: int, position: int):
        self._prefix = CachedPrefix("main", tokens, position)

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def resolve_cached_prefix(self, rid, node_name, graph_walk, inputs):
        return self._prefix


@pytest.fixture
def thinker():
    config = SimpleNamespace(
        im_start_token_id=1, system_token_id=2, assistant_token_id=3, thinker_hidden_size=HIDDEN,
        thinker=SimpleNamespace(
            audio_start_token_id=4, audio_end_token_id=5, vision_start_token_id=6, vision_end_token_id=7,
            position_id_per_seconds=25,
        ),
        vision=SimpleNamespace(spatial_merge_size=2, deepstack_visual_indexes=[1, 2]),
    )
    thinker = ThinkerSubmodule(thinker_model=_StubThinker(), config=config)
    thinker.node_resources = {THINKER_POS: RopeManager(
        config=PositionConfig(kv_cache=THINKER_KV), device=torch.device("cpu"), dtype=torch.float32,
    )}
    return thinker


def _prepare(thinker, walk: str, inputs: dict, held: CachedPrefix | None = None, rid: str = RID):
    info = SimpleNamespace(rid_handle=rid)
    prepared = thinker.prepare_inputs(walk, info, inputs)
    if held is None:
        return prepared
    engine = Engine.__new__(Engine)
    engine._runner = StepRunner(
        {THINKER_KV: _Holding(held.tokens, held.position), THINKER_POS: thinker.node_resources[THINKER_POS]},
        node_resources={NODE: [THINKER_KV, THINKER_POS]},
    )
    engine._submodules = {NODE: SimpleNamespace(submodule=thinker)}
    engine._keyed_walks = {NODE: {"prefill_text", "prefill_vision", "prefill_audio"}}
    batch = SimpleNamespace(
        node_name=NODE, cached_rids=set(), per_request_info_wrapped={rid: info},
        step_context=StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False),
    )
    return engine._skip_cached_prefix(batch, rid, prepared)


def test_a_cut_text_walk_prepared_again_is_planned_where_the_cache_keyed_it(thinker):
    rope = thinker.node_resources[THINKER_POS]
    rope._counters[RID] = {"main": 10}
    text = {"text_inputs": [torch.arange(8, 14).repeat(2)]}

    _prepare(thinker, "prefill_text", text, CachedPrefix("main", 6, 16))
    # a refused admit prepares the walk again, after the probe moved the counter
    again = _prepare(thinker, "prefill_text", text, CachedPrefix("main", 6, 16))
    kv_plan = KVPlanOutputs({"main": SimpleNamespace(rows=[(RID, "main", again.input_seq_len)])})
    rope.plan(PositionStep(), StepContext((RID,), "prefill_text", 0, False, plan_results={THINKER_KV: kv_plan}))

    assert again.custom_pos_ids is None and thinker._position_ids_3d([again]).tolist() == [list(range(16, 22))] * 3, (
        "the retried walk's tail runs past the positions the cache keyed it at"
    )


def test_a_cut_vision_walk_cuts_its_masks_and_deepstack_with_the_embeds(thinker):
    image = {
        "vision_embeds": [torch.randn(6, HIDDEN)], "deepstack": [torch.randn(6, HIDDEN)] * 2,
        "image_grid_thw": [torch.tensor([1, 4, 6])],
    }
    whole = _prepare(thinker, "prefill_vision", image)

    cut = thinker.split_inputs("prefill_vision", SimpleNamespace(rid_handle=RID), whole, 3, whole.input_seq_len)

    assert torch.equal(cut.tensor_inputs["masks_for_talker"], whole.tensor_inputs["masks_for_talker"][:, 3:]) and all(
        torch.equal(layer, full[3:])
        for layer, full in zip(cut.tensor_inputs["deepstack"], whole.tensor_inputs["deepstack"], strict=True)
    ), "the Talker masks or deepstack rows kept rows the embeds lost, so they would land on other tokens"


def test_each_cut_row_of_a_mixed_step_moves_its_own_counter_past_its_whole_walk(thinker):
    # The cache holds 6 of 12 text tokens, 28 of a clip's 30 slots, and 3 of an image's 8 slots,
    # which advance the counter by 5. A partly held item leaves the counter at its start.
    rope = thinker.node_resources[THINKER_POS]
    rows = {
        "text": ("prefill_text", {"text_inputs": [torch.arange(8, 14).repeat(2)]}, 10, CachedPrefix("main", 6, 16)),
        "clip": ("prefill_audio", {"audio_embeds": [torch.zeros(28, HIDDEN)]}, 20, CachedPrefix("main", 28, 20)),
        "image": ("prefill_vision", {
            "vision_embeds": [torch.randn(6, HIDDEN)], "deepstack": [torch.randn(6, HIDDEN)] * 2,
            "image_grid_thw": [torch.tensor([1, 4, 6])],
        }, 30, CachedPrefix("main", 3, 30)),
    }
    inputs = []
    for rid, (walk, tensors, counter, held) in rows.items():
        rope._counters[rid] = {"main": counter}
        cut = _prepare(thinker, walk, tensors, held, rid=rid)
        cut.graph_walk = walk
        inputs.append(cut)

    step = thinker.declare_step(THINKER_MIXED, list(rows), inputs)
    rope.commit(step.steps[THINKER_POS], StepContext(tuple(rows), THINKER_MIXED, slot=0, capture=False))

    assert [cut.input_seq_len for cut in inputs] == [6, 2, 5], "a row ran slots the cache already holds"
    assert [rope.position(rid, "main") for rid in rows] == [22, 50, 35], (
        "a row moved its counter by the slots it ran, so its later walks would run at wrong positions"
    )


def test_an_images_span_is_what_its_positions_place():
    grid, start = (1, 30, 40), 7
    positions = get_rope_index_vision(torch.tensor(grid), start + 1, position_id_per_seconds=25, spatial_merge_size=2)

    # the end sentinel sits one past the image's last position, and the next walk one past that
    assert image_span(torch.tensor(grid), 2) == (positions.shape[1] + 2, int(positions.max()) + 2 - start), (
        "the layout's image span disagrees with the positions its walk places"
    )


_SPECIALS = {
    "<|vision_start|>": 151652, "<|vision_end|>": 151653, "<|image_pad|>": 151655, "<|video_pad|>": 151656,
    "<|audio_start|>": 151669, "<|audio_end|>": 151670, "<|audio_pad|>": 151675,
}
_PLACEHOLDERS = {
    "image": "<|vision_start|><|image_pad|><|vision_end|>",
    "audio": "<|audio_start|><|audio_pad|><|audio_end|>",
}


class _Tokenizer:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return _SPECIALS.get(token, 0)

    def __call__(self, text, return_tensors=None):
        for token, id_ in _SPECIALS.items():
            text = text.replace(token, chr(id_))
        return {"input_ids": torch.tensor([[ord(c) for c in text]])}


class _Processor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        turns = "".join(
            f"<|im_start|>{m['role']}\n" + (m["content"] if isinstance(m["content"], str) else "".join(
                part["text"] if part["type"] == "text" else _PLACEHOLDERS[part["type"]] for part in m["content"]
            )) + "<|im_end|>\n"
            for m in messages
        )
        return turns + "<|im_start|>assistant\n"

    @staticmethod
    def image_processor(images, return_tensors):
        height, width, _ = images[0].shape
        return {"pixel_values": torch.zeros(1), "image_grid_thw": torch.tensor([[1, height // 16, width // 16]])}

    @staticmethod
    def feature_extractor(audio, sampling_rate, **kwargs):
        frames = len(audio) // 160
        return {"input_features": torch.zeros(1, 128, frames), "attention_mask": torch.ones(1, frames)}


@pytest.fixture
def model():
    model = Qwen3OmniModel.__new__(Qwen3OmniModel)
    model.tokenizer = _Tokenizer()
    model._processor = _Processor()
    model.config = SimpleNamespace(
        vision=SimpleNamespace(spatial_merge_size=2), audio_encoder=SimpleNamespace(n_window=50),
    )
    return model


def _prompt(model, modalities: list[str], tensors: dict | None, **kwargs):
    texts = ["Say what you see."] * modalities.count("text")
    return model.process_prompt(texts, modalities, ["text"], tensors=tensors, **kwargs)


# A 64x96 image is a 4x6 patch grid, 2x3 after the merge, so it writes 8 slots with its sentinels and advances by 5.
_IMAGE = ("prefill_vision", 8, 5)
# A second of audio is 100 feature frames, which the encoder turns into 13, so it writes 15 slots with its sentinels.
_AUDIO = ("prefill_audio", 15, 15)
_PROMPTS = {
    "text, image, text": (["text", "image", "text"], {"image_inputs": [torch.rand(3, 64, 96)]}, [_IMAGE]),
    "audio then text": (["audio", "text"], {"audio_inputs": [torch.zeros(16000)]}, [_AUDIO]),
}


@pytest.mark.parametrize(("modalities", "tensors", "items"), _PROMPTS.values(), ids=_PROMPTS.keys())
def test_the_layout_lists_each_walk_the_schedule_runs_with_the_slots_it_writes(model, modalities, tensors, items):
    out = _prompt(model, modalities, tensors)
    layout = out.metadata["prefix_layout"][THINKER_KV]["main"]
    signals = {name: [object()] * len(value) for name, value in out.new_input_tensors.items()}

    walks = [walk for walk, _ in model._build_thinker_prefill_schedule(modalities, signals)]
    segments = iter(out.new_input_tensors["text_inputs"])
    media = iter(items)
    expected = [
        ("prefill_text", len(next(segments)), None) if walk == "prefill_text" else next(media) for walk in walks
    ]
    assert [(span.walk, span.length, None if span.kind == "ids" else span.advance) for span in layout] == expected, (
        "the layout disagrees with the walks the schedule runs, so the cache would drop it partway through"
    )


def _vision(attn_implementation: str = "sdpa") -> VisionEncoderSubmodule:
    # The tensors `prepare_inputs` builds depend only on the grid, the merge size and
    # `num_grid_per_side`, so a small encoder is enough.
    config = Qwen3OmniMoeVisionEncoderConfig(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=16,
        num_position_embeddings=64, deepstack_visual_indexes=[0, 1],
    )
    encoder = Qwen3OmniMoeVisionEncoder._from_config(config, attn_implementation=attn_implementation).eval()
    return VisionEncoderSubmodule(encoder, config=None)


def _image(vision):
    grid, pixels = torch.tensor([1, 4, 6]), torch.randn(24, 3 * 2 * 16 * 16)
    inputs = {"pixel_values": [pixels], "image_grid_thw": [grid]}
    return grid, pixels, vision.prepare_inputs("prefill_vision", None, inputs)


def test_the_vision_encoder_embeds_an_image_from_its_prepared_inputs_as_it_would_alone():
    vision = _vision()
    grid, pixels, prepared = _image(vision)
    encoder = vision.vision_encoder
    with torch.no_grad():
        ours = vision.forward("prefill_vision", None, **prepared.tensor_inputs)
        alone = encoder(pixels, grid_thw=grid[None])

    assert torch.equal(ours["vision_embeds"][0], alone.pooler_output) and all(
        torch.equal(layer, own) for layer, own in zip(ours["deepstack"], alone.deepstack_features, strict=True)
    ), "the inputs built before the compiled call embed the image otherwise than the encoder does"


def _audio(attn_implementation: str = "sdpa") -> AudioEncoderSubmodule:
    # The tensors `prepare_inputs` builds depend only on the clip's length and the encoder's
    # windows, so a small encoder is enough.
    config = Qwen3OmniMoeAudioEncoderConfig(
        encoder_layers=1, d_model=32, encoder_attention_heads=2, encoder_ffn_dim=64, output_dim=16,
        downsample_hidden_size=8,
    )
    encoder = Qwen3OmniMoeAudioEncoder._from_config(config, attn_implementation=attn_implementation).eval()
    return AudioEncoderSubmodule(encoder, config=None)


def _clip(frames: int, device: str = "cpu") -> dict:
    return {"audio_features": [torch.randn(128, frames, device=device)],
            "audio_seqlens": [torch.tensor([frames], device=device)]}


# shorter than one chunk, shorter than one attention window, and three windows
@pytest.mark.parametrize("frames", [60, 600, 1773])
def test_the_audio_encoder_embeds_a_clip_from_its_prepared_inputs_as_it_would_alone(frames):
    audio, clip = _audio(), _clip(frames)
    prepared = audio.prepare_inputs("prefill_audio", None, clip)
    # sdpa ignores `max_seqlen`, so comparing the outputs can't test it
    assert audio._max_seqlen >= prepared.tensor_inputs["cu_seqlens"].diff().max(), (
        "flash-attn would be told a bound shorter than the clip's longest attention window"
    )
    with torch.no_grad():
        ours = audio.forward("prefill_audio", None, **prepared.tensor_inputs)
        alone = audio.audio_encoder(clip["audio_features"][0], feature_lens=clip["audio_seqlens"][0])

    assert torch.equal(ours["audio_embeds"][0], alone.last_hidden_state), (
        "the inputs built before the compiled call embed the clip otherwise than the encoder does"
    )


def test_the_audio_encoder_compiles_with_the_default_while_its_output_is_cached(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "compile", lambda fn, dynamic, **kwargs: dynamic)
    engine = Engine.__new__(Engine)
    engine._device = torch.device("cuda")
    engine._item_encoders = {"audio_encoder"}
    engine._submodules = {"audio_encoder": SimpleNamespace(submodule=_audio(), forward="eager")}

    engine._compile_submodules()

    assert engine._submodules["audio_encoder"].forward is None, (
        "the audio encoder compiled with dynamic=True or stayed eager while its output is cached"
    )

@requires_cuda
@pytest.mark.parametrize(("node", "seen", "unseen"), [
    ("vision", [(1, 4, 6)], [(1, 8, 10)]),
    # The clips of 2, 60 and 50100 frames have a dimension of size 1 inside the encoder, so they compile
    # graphs of their own.
    ("audio", [1773, 60, 50100, 2], [600, 35]),
])
def test_an_encoder_compiles_no_graph_for_a_size_inside_the_ranges_it_has_compiled(node, seen, unseen):
    pytest.importorskip("flash_attn")
    torch._dynamo.reset()
    if node == "vision":
        submodule, walk = _vision("flash_attention_2"), "prefill_vision"

        def inputs(grid):
            return {"pixel_values": [torch.randn(grid[0] * grid[1] * grid[2], 3 * 2 * 16 * 16, device="cuda")],
                    "image_grid_thw": [torch.tensor(grid, device="cuda")]}
    else:
        submodule, walk = _audio("flash_attention_2"), "prefill_audio"

        def inputs(frames):
            return _clip(frames, "cuda")

    submodule = submodule.to(device="cuda", dtype=torch.bfloat16)
    engine = Engine.__new__(Engine)
    engine._device = torch.device("cuda")
    engine._item_encoders = {node}
    engine._submodules = {node: SimpleNamespace(submodule=submodule)}
    engine._compile_submodules()
    forward = engine._submodules[node].forward

    def encode(size):
        prepared = submodule.prepare_inputs(walk, None, inputs(size))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            forward(walk, None, **prepared.tensor_inputs)

    for size in seen:
        encode(size)
    frames = counters["frames"]["ok"]
    with torch._dynamo.config.patch(error_on_recompile=True):
        for size in unseen:
            encode(size)

    assert counters["frames"]["ok"] == frames, (
        f"the {node} encoder compiled a graph for a size inside a range it had compiled"
    )
