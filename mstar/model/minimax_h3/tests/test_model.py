"""MiniMax-H3 in dummy mode, on a pinned config: no weights, GPU or network."""

from __future__ import annotations

import pytest
import torch

from mstar.graph.base import GraphNode, Loop, Parallel, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.components.diffusion.attention import sdpa_attention
from mstar.model.minimax_h3.components.layout import (
    T_AUDIO,
    T_COND,
    T_VIDEO,
    H3Layout,
    modality_tags,
    patchify_video,
    timestep_classes,
    unpatchify_video,
)
from mstar.model.minimax_h3.config import (
    AUDIO_TAG,
    DENOISE_LOOP,
    DIT_ATTN,
    REFINER_ATTN,
    TEXT_TAG,
    VIDEO_TAG,
    H3Config,
)
from mstar.model.minimax_h3.keyframes import fit_keyframes, presentation, vision_spans
from mstar.model.minimax_h3.minimax_h3_model import (
    ENCODE_KEYFRAMES_WALK,
    ENCODE_TEXT_WALK,
    GENERATE_AUDIO_WALK,
    GENERATE_AV_WALK,
    GENERATE_VIDEO_WALK,
    KEYFRAMES_SUFFIX,
    H3Model,
)
from mstar.model.minimax_h3.schedule import h3_schedule
from mstar.model.minimax_h3.submodules import LATENTS, TEXT_INPUTS, H3DenoiseSubmodule, layout_from_metadata


def _model(**kwargs) -> H3Model:
    model = H3Model(model_path_hf="test/minimax_h3", skip_weight_loading=True, **kwargs)
    model.set_config(H3Config())
    return model


def _tensor_info(dims: list[int]) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=dims, dtype="torch.int64", nbytes=8, address=0, stride=[1] * len(dims),
        uuid="uuid-text", source_session_id="test:0", source_entity="test",
    )


class _Tokenizer:
    """``tokenizers.Tokenizer``'s two calls: a token per character, the vision tokens
    named (and matched in the text, as the real tokenizer's added tokens are)."""

    SPECIAL = {"<|vision_start|>": 900, "<|image_pad|>": 901, "<|vision_end|>": 902, "<|video_pad|>": 903}

    class _Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            special = next((t for t in self.SPECIAL if text.startswith(t, i)), None)
            if special:
                ids.append(self.SPECIAL[special])
                i += len(special)
            else:
                ids.append(ord(text[i]) % 800)
                i += 1
        return self._Encoding(ids)

    def token_to_id(self, token):
        return self.SPECIAL.get(token)


def test_graph_walks_and_nodes():
    model = _model()
    walks = model.get_graph_walk_graphs()
    generate = {GENERATE_AV_WALK, GENERATE_VIDEO_WALK, GENERATE_AUDIO_WALK}
    assert set(walks) == {ENCODE_TEXT_WALK, ENCODE_KEYFRAMES_WALK} | generate | {w + KEYFRAMES_SUFFIX for w in generate}
    assert model.nodes == ["audio_decoder", "dit", "text_encoder", "vae_decoder", "vae_encoder"]
    keyframes = walks[ENCODE_KEYFRAMES_WALK]
    assert isinstance(keyframes, Parallel)
    assert {n.name: set(n.input_names) for n in keyframes.sections} == {
        "text_encoder": {"text_inputs", "vision_pixels"}, "vae_encoder": {"keyframe_pixels"}}
    assert all(e.persist for n in keyframes.sections for e in n.outputs)
    kf_loop = walks[GENERATE_AV_WALK + KEYFRAMES_SUFFIX].sections[0]
    assert "cond_latents" in kf_loop.section.input_names
    assert "cond_latents" not in walks[GENERATE_AV_WALK].sections[0].section.input_names
    text = walks[ENCODE_TEXT_WALK]
    assert isinstance(text, GraphNode)
    (embeds,) = text.outputs
    assert embeds.persist and embeds.next_node == EMPTY_DESTINATION
    av = walks[GENERATE_AV_WALK]
    assert isinstance(av, Sequential)
    loop, decoders = av.sections
    assert isinstance(loop, Loop) and loop.name == DENOISE_LOOP
    assert {e.next_node for e in loop.section.outputs} == {"dit"}
    assert isinstance(decoders, Parallel)
    emitted = {e.output_modality for d in decoders.sections for e in d.outputs if e.next_node == EMIT_TO_CLIENT}
    assert emitted == {"video", "audio"}


def test_resources_per_backend():
    assert _model(attention_backend="sdpa").get_node_resources() == []
    specs = {s.resource_key: s for s in _model().get_node_resources()}
    assert set(specs) == {DIT_ATTN, REFINER_ATTN}
    assert specs[DIT_ATTN].nodes == {"dit"} and specs[DIT_ATTN].config.shard_over_sp
    assert specs[REFINER_ATTN].nodes == {"text_encoder"} and not specs[REFINER_ATTN].config.shard_over_sp


def test_request_defaults_and_validation():
    model = _model()
    assert model.resolve_request({}) == {
        "height": 768, "width": 1344, "num_frames": 124, "num_inference_steps": 50,
        "video_shift": 12.0, "audio_shift": 3.0,
    }
    # snapped up to 17 n + 5, as the reference does
    assert model.resolve_request({"num_frames": 120})["num_frames"] == 124
    for bad in (
        {"height": 768},                          # width missing
        {"height": 770, "width": 1344},           # not a multiple of 32
        {"height": 256, "width": 1344 * 4},       # aspect beyond 4:1
        {"num_frames": 100},                      # under 5 s
        {"num_frames": 400},                      # over 15 s
        {"num_inference_steps": 1},
        {"flow_shift": 0},
        {"height": "tall", "width": 1344},
        # non-finite, truncated or overflowing numbers: a 400, never NaN sigmas or a 500
        {"flow_shift": float("inf")},
        {"audio_flow_shift": float("nan")},
        {"flow_shift": 1e39},                     # finite, but inf in fp32
        {"flow_shift": True},
        {"num_inference_steps": float("inf")},
        {"num_frames": 1e999},
        {"height": 768.5, "width": 1344},
        {"num_inference_steps": 2.7},
        {"height": 768, "width": "1344"},
        {"flow_shift": "3"},
    ):
        with pytest.raises(ValueError):
            model.resolve_request(bad)
    assert model.resolve_request({"height": 768.0, "width": 1344})["height"] == 768
    for anchors in (5, [["first"]], [None], {"first": 1}):
        with pytest.raises(ValueError, match="keyframe_anchors"):
            model.keyframe_anchors({"keyframe_anchors": anchors}, 1)


def _keyframe(height: int = 64, width: int = 96) -> torch.Tensor:
    return torch.full((3, height, width), 128, dtype=torch.uint8)


def test_prompts_refused_before_any_work(monkeypatch):
    from mstar.model.minimax_h3 import minimax_h3_model

    def no_resize(*args, **kwargs):
        raise AssertionError("keyframes resized before the size was checked")

    monkeypatch.setattr(minimax_h3_model, "fit_keyframes", no_resize)
    model = _model(max_prompt_tokens=8)
    model.tokenizer = _Tokenizer()
    image = {"image_inputs": [_keyframe()]}
    for prompt, inputs, outputs, extra, match in (
        ("a dog", ["text", "image"], ["video", "audio"], {}, "image"),
        ("  ", ["text"], ["video", "audio"], {}, "prompt"),
        ("a dog", ["text"], ["text"], {}, "output"),
        ("a dog runs far", ["text"], ["video", "audio"], {}, "max_prompt_tokens"),
        ("a <|image_pad|>", ["text"], ["video", "audio"], {}, "vision tokens"),
        ("<|vision_start|>a", ["text", "image"], ["video", "audio"], image, "vision tokens"),
    ):
        with pytest.raises(ValueError, match=match):
            model.process_prompt(prompt, inputs, outputs, extra)
    model = _model()
    model.tokenizer = _Tokenizer()
    for size in ({"height": 40000, "width": 40000}, {"height": 768}, {"width": "wide", "height": 768}):
        with pytest.raises(ValueError):
            model.process_prompt("a dog", ["text", "image"], ["video", "audio"], image, **size)


class _VisionProcessor:
    """Qwen3-VL's image processor's two outputs for keyframes on a canvas it keeps as is."""

    def __call__(self, images, return_tensors):
        grids = [[1, image.size[1] // 16, image.size[0] // 16] for image in images]
        return {"image_grid_thw": torch.tensor(grids), "pixel_values": torch.zeros(sum(h * w for _, h, w in grids), 8)}


def _data_worker(model):
    """The data worker's media and prompt stage, without its threads or sockets."""
    from mstar.api_server.data_worker import PreprocessWorkerThread

    worker = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    worker.model, worker.device, worker._keyed_labels = model, "cpu", None
    return worker


def _png(path, size=(96, 64)):
    from PIL import Image

    Image.new("RGB", size, (90, 120, 30)).save(path)
    return str(path)


def test_files_the_model_cannot_take_are_refused_before_decoding(tmp_path, monkeypatch):
    model = _model()

    def decoded(*args, **kwargs):
        raise AssertionError("decoded before the file count was checked")

    for modality in ("image", "audio", "video"):
        monkeypatch.setattr(model, f"load_{modality}", decoded)
    worker = _data_worker(model)
    image = _png(tmp_path / "k.png")
    for files in ({"image": [image] * 3}, {"video": [image]}, {"audio": [image]}):
        with pytest.raises(ValueError, match="at most"):
            worker._load_media(files)


def test_the_raw_keyframes_are_not_stored(tmp_path):
    """``process_prompt`` returns the keyframes on the canvas; the decoded uploads it read
    them from would otherwise be stored, and held, for the whole request."""
    from mstar.api_server.request_types import PreprocessInput

    model = _model()
    model.tokenizer, model._image_processor = _Tokenizer(), _VisionProcessor()
    images = [_png(tmp_path / "first.png"), _png(tmp_path / "last.png")]
    prepared = _data_worker(model)._prepare_input(PreprocessInput(
        request_id="r", text="a dog", file_paths={"image": images}, input_modalities=["image", "text"],
        output_modalities=["video", "audio"], model_kwargs={"keyframe_anchors": ["first", "last"]},
    ))
    assert prepared.tensors["image_inputs"] == []
    assert prepared.tensors["keyframe_pixels"][0].shape[0] == 2 and prepared.tensors["vision_pixels"]


def test_the_row_cap_counts_every_packed_row():
    # 768x1344x124 is 37296 video and 414 audio rows
    model = _model(max_packed_rows=37 * 24 * 42 + 207 * 2 + 5)
    model.tokenizer = _Tokenizer()
    model.process_prompt("a dog", ["text"], ["video", "audio"])
    with pytest.raises(ValueError, match="max_packed_rows"):
        model.process_prompt("a dog!", ["text"], ["video", "audio"])
    with pytest.raises(ValueError, match="max_packed_rows"):
        model.resolve_request({}, text_rows=0, keyframes=1)


def test_a_prompt_is_capped_by_the_text_encoder_alone():
    # MiniMax's prompt expansion writes prompts of up to ~10k tokens, for clips of up to 15 s
    model = _model()
    model.tokenizer = _Tokenizer()
    ids = model.process_prompt("x" * 12_800, ["text"], ["video", "audio"], num_frames=243)[TEXT_INPUTS][0]
    assert ids.numel() == 12_800
    with pytest.raises(ValueError, match="positions"):
        model.process_prompt("x" * (H3Config().text_max_positions + 1), ["text"], ["video", "audio"])
    from mstar.model.components.diffusion.qwen3.weight_loading import make_text_encoder

    with torch.device("meta"):
        encoder = make_text_encoder(H3Config().text)
    assert encoder.max_positions == H3Config().text_max_positions  # it encodes every prompt admitted


def test_walk_plan_and_step_metadata():
    from mstar.conductor.conductor import _walk_io

    model = _model()
    plan = model.walk_plan(
        "default", ["text"], ["video", "audio"], {"text_inputs": [_tensor_info([37])]},
        {"height": 256, "width": 448, "num_inference_steps": 6},
    )
    assert [step.walk for step in plan] == [ENCODE_TEXT_WALK, GENERATE_AV_WALK]
    layout = layout_from_metadata(plan[0].step_metadata)
    assert layout == H3Layout(text_len=37, latent_frames=37, latent_height=16, latent_width=28, audio_latents=207)
    assert layout_from_metadata(plan[1].step_metadata) == layout
    # what the conductor sends the denoise step: the prompt's embeddings, and empty loop-backs the dit seeds
    assert {name for name, _ in _walk_io(model)[0][GENERATE_AV_WALK]} == {"text_embeds", "latents", "audio_latents"}
    for outputs, walk in ((["video"], GENERATE_VIDEO_WALK), (["audio"], GENERATE_AUDIO_WALK)):
        plan = model.walk_plan("default", ["text"], outputs, {"text_inputs": [_tensor_info([5])]}, {})
        assert plan[-1].walk == walk
    assert model.get_loop_caps(num_inference_steps=6) == {DENOISE_LOOP: 5}


def test_schedule_matches_the_reference_scheduler():
    """MiniMaxH3Scheduler's grid (values printed from diffusers 0.42.1): 50 points, 49 steps."""
    s = h3_schedule(50, 12.0, 3.0)
    assert s.num_steps == 49 and s.sigmas.shape[0] == 50 and s.audio_sigmas.shape[0] == 50
    torch.testing.assert_close(
        s.sigmas[:4], torch.tensor([1.0, 0.998266875743866, 0.9964664578437805, 0.9945946335792542]),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        s.audio_sigmas[:4], torch.tensor([1.0, 0.9931034445762634, 0.9860140085220337, 0.978723406791687]),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        s.timesteps[-3:], torch.tensor([0.5609756112098694, 0.6619718074798584, 0.800000011920929]), rtol=0, atol=0,
    )
    assert float(s.sigmas[-1]) == 0.0 and torch.all(s.cond_timesteps >= 0.999)


def test_layout_rows_tags_and_classes():
    layout = H3Layout(text_len=5, latent_frames=3, latent_height=4, latent_width=6, audio_latents=4,
                      anchors=("first", "last"))
    assert (layout.rows_per_frame, layout.num_condition_rows, layout.num_audio_rows, layout.num_video_rows) == (
        6, 12, 8, 18)
    assert layout.total_rows == 5 + 12 + 8 + 18
    tags, classes = modality_tags(layout), timestep_classes(layout)
    assert tags[:5].tolist() == [TEXT_TAG] * 5 and tags[5:17].tolist() == [VIDEO_TAG] * 12
    assert tags[17:25].tolist() == [AUDIO_TAG] * 8 and tags[25:].tolist() == [VIDEO_TAG] * 18
    assert classes[:5].tolist() == [T_VIDEO] * 5 and classes[5:17].tolist() == [T_COND] * 12
    assert classes[17:25].tolist() == [T_AUDIO] * 8 and classes[25:].tolist() == [T_VIDEO] * 18
    z = torch.randn(4, 3, 4, 6)
    rows = patchify_video(z, layout.patch)
    assert rows.shape == (18, 16)
    assert torch.equal(unpatchify_video(rows[None], layout, 4)[0], z)


def test_vae_tile_plan_covers_the_canvas():
    from mstar.model.minimax_h3.components.video_vae import H3VideoDecoder

    with torch.device("meta"):
        vae = H3VideoDecoder(H3Config().video_vae)
    for size in (1344, 768):
        starts, lengths, overlaps = vae.split_tiles(size)
        assert starts[0] == 0 and starts[-1] + lengths[-1] == size and all(o >= 64 for o in overlaps)
    assert vae.split_tiles(128) == ([0], [128], [])


# a 1 x 1 PNG
PNG_URL = ("data:image/png;base64,"
           "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")


def test_openai_adapter(tmp_path):
    """``image`` is the first frame and ``last_image`` (an ``extra_body`` field) the last."""
    from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
    from mstar.api_server.openai.protocol import VideoGenerationRequest

    adapter = ADAPTER_REGISTRY["minimax_h3"]
    args = adapter.video_to_request(
        VideoGenerationRequest(prompt="a dog", size="1344x768", seed=3, num_frames=124), None,
    )
    assert args.output_modalities == ["video", "audio"]
    with_null = adapter.video_to_request(VideoGenerationRequest(prompt="a dog", audio=None), None)
    assert with_null.output_modalities == ["video", "audio"] and "audio" not in with_null.model_kwargs
    assert {k: args.model_kwargs[k] for k in ("width", "height", "seed", "num_frames")} == {
        "width": 1344, "height": 768, "seed": 3, "num_frames": 124}
    url = PNG_URL
    args = adapter.video_to_request(VideoGenerationRequest(prompt="a dog", image=url, last_image=url), tmp_path)
    assert args.input_modalities == ["image", "text"] and len(args.file_paths["image"]) == 2
    assert args.model_kwargs["keyframe_anchors"] == ["first", "last"] and "last_image" not in args.model_kwargs
    last_only = adapter.video_to_request(VideoGenerationRequest(prompt="a dog", last_image=url), tmp_path)
    assert last_only.model_kwargs["keyframe_anchors"] == ["last"]
    for bad in (VideoGenerationRequest(prompt="a dog", fps=30), VideoGenerationRequest(prompt="a dog", video=url),
                VideoGenerationRequest(prompt="a dog", n=2), VideoGenerationRequest(prompt="a dog", audio="false")):
        with pytest.raises(ValueError):
            adapter.video_to_request(bad, tmp_path)


def test_a_refused_keyframe_leaves_no_upload(tmp_path, monkeypatch):
    """A request refused at its last keyframe removes the first one's saved upload: it is
    never submitted, so no deferred cleanup would. A non-string or empty reference is a
    400, not a text-only request."""
    from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
    from mstar.api_server.openai.protocol import VideoGenerationRequest

    monkeypatch.delenv("MSTAR_ALLOW_REMOTE", raising=False)
    adapter = ADAPTER_REGISTRY["minimax_h3"]
    for last in ("/etc/hostname", "file:///etc/hostname", "https://example.com/a.png", 123, ["x"], "", 0, []):
        with pytest.raises(ValueError):
            adapter.video_to_request(VideoGenerationRequest(prompt="a dog", image=PNG_URL, last_image=last), tmp_path)
        assert list(tmp_path.iterdir()) == [], last
    with pytest.raises(ValueError):
        adapter.video_to_request(VideoGenerationRequest(prompt="a dog", image=""), tmp_path)


def test_a_keyframe_is_never_scaled_far_past_the_canvas(monkeypatch):
    """The last keyframe is scaled to cover the canvas, then cropped; a sliver would be
    scaled by its aspect ratio (2x400 px to 1344x268800, ~1 GB) and is refused instead."""
    from PIL import Image

    real = Image.Image.resize

    def resize(self, size, *args, **kwargs):
        assert size[0] * size[1] <= 32 * 768 * 1344, f"resized to {size}"
        return real(self, size, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", resize)
    first = Image.new("RGB", (1344, 768))
    for sliver in ((2, 400), (400, 2)):
        with pytest.raises(ValueError, match="aspect"):
            fit_keyframes(first, Image.new("RGB", sliver))
    # 4:1 onto 1:4, the most the aspect range allows: 16x the canvas before the crop
    fitted = fit_keyframes(Image.new("RGB", (2048, 512)), Image.new("RGB", (512, 2048)))
    assert fitted.images[1].size == (fitted.width, fitted.height)


def test_keyframes_onto_the_canvas():
    from PIL import Image

    first, last = Image.new("RGB", (1000, 600), (10, 20, 30)), Image.new("RGB", (600, 600), (40, 50, 60))
    fitted = fit_keyframes(first, last)
    # the first keyframe's 5:3 aspect sets the canvas, short edge 768 under the 768 x 1344 area cap
    assert (fitted.height, fitted.width) == (768, 1280) and fitted.anchors == ("first", "last")
    assert all(image.size == (1280, 768) for image in fitted.images)
    last_only = fit_keyframes(None, last, 512, 512)
    assert last_only.anchors == ("last",) and last_only.images[0].size == (512, 512)
    with pytest.raises(ValueError):
        fit_keyframes(None, None)


def test_presentation_tags_vision_blocks_as_video():
    tok = _Tokenizer()
    pres = presentation(tok, "a dog", [3, 2])
    spans = vision_spans(tok, [3, 2])
    label = len("<Picture 1>: ")
    assert spans == [(label, 5), (label + 5 + label, 4)]
    for start, length in spans:
        assert pres.token_ids[start] == 900 and pres.token_ids[start + length - 1] == 902
        assert set(pres.tags[start:start + length]) == {VIDEO_TAG}
    assert pres.tags.count(VIDEO_TAG) == 9 and len(pres.token_ids) == len(pres.tags) == 2 * label + 9 + 5
    layout = H3Layout(text_len=len(pres.tags), latent_frames=3, latent_height=4, latent_width=4, audio_latents=2,
                      text_video_spans=tuple(spans))
    assert modality_tags(layout)[:layout.text_len].tolist() == pres.tags


def test_keyframe_walk_plan():
    from mstar.conductor.conductor import _walk_io

    model = _model()
    model.tokenizer = _Tokenizer()
    signals = {"text_inputs": [_tensor_info([40])], "vision_pixels": [_tensor_info([2 * 4032, 1536])],
               "keyframe_pixels": [_tensor_info([2, 3, 768, 1344])]}
    plan = model.walk_plan("default", ["image", "text"], ["video", "audio"], signals,
                           {"keyframe_anchors": ["last", "first"]})
    assert [step.walk for step in plan] == [ENCODE_KEYFRAMES_WALK, GENERATE_AV_WALK + KEYFRAMES_SUFFIX]
    request = plan[0].step_metadata
    assert request["anchors"] == ["first", "last"] and (request["height"], request["width"]) == (768, 1344)
    assert request["keyframe_grids"] == [[1, 48, 84]] * 2
    assert request["text_video_spans"] == [list(s) for s in vision_spans(model.tokenizer, [1008, 1008])]
    inputs, _, _ = _walk_io(model)
    assert set(inputs[ENCODE_KEYFRAMES_WALK]) == {
        ("text_inputs", "text_encoder"), ("vision_pixels", "text_encoder"), ("keyframe_pixels", "vae_encoder")}
    layout = layout_from_metadata(request)
    assert layout.anchors == ("first", "last") and layout.num_condition_rows == 2 * 24 * 42
    assert {name for name, _ in inputs[GENERATE_AV_WALK + KEYFRAMES_SUFFIX]} == {
        "text_embeds", "cond_latents", "latents", "audio_latents"}


def test_keyframe_noise_is_drawn_first():
    """One draw per keyframe (its latent's shape), then the video noise, then the audio's."""
    dit = H3DenoiseSubmodule(None, H3Config(), loop_name=DENOISE_LOOP, attn_resource_key=None)
    layout = H3Layout(text_len=3, latent_frames=2, latent_height=4, latent_width=6, audio_latents=3,
                      anchors=("first", "last"))
    seeded = dit.seed_loop_back(None, layout, torch.Generator().manual_seed(7))
    g = torch.Generator().manual_seed(7)
    cond = [torch.randn(1, 24, 1, 4, 6, generator=g) for _ in range(2)]
    video = torch.randn(1, 24, 2, 4, 6, generator=g)
    audio = torch.randn(6, 32, generator=g)
    want = torch.cat([patchify_video(c[0], layout.patch) for c in cond] + [patchify_video(video[0], layout.patch)])
    assert torch.equal(seeded["latents"], want) and torch.equal(seeded["audio_latents"], audio)


def _tiny_decode_case():
    """A tiny video VAE, its config, and a batch of two requests' rows: 14 latents make 3
    clips, the last padded."""
    from mstar.model.minimax_h3.components.video_vae import H3VideoDecoder
    from mstar.model.minimax_h3.config import H3VideoVAEConfig

    torch.manual_seed(0)
    vcfg = H3VideoVAEConfig(latent_channels=4, decoder_layers=1, decoder_heads=2, decoder_head_dim=32,
                            latents_mean=(0.1,) * 4, latents_std=(0.9,) * 4)
    vae = H3VideoDecoder(vcfg).eval()
    with torch.no_grad():
        for param in vae.parameters():
            param.normal_(0, 0.2)
    layout = H3Layout(text_len=3, latent_frames=14, latent_height=6, latent_width=4, audio_latents=2)
    return vae, H3Config(video_vae=vcfg), layout, torch.randn(2, layout.num_video_rows, 4 * 2 * 2)


def _sp_decode(rank: int, world: int, store: str, out: str) -> None:
    import torch.distributed as dist

    from mstar.distributed.communication import CommGroup
    from mstar.model.minimax_h3.submodules import H3VideoDecoderSubmodule

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        torch.set_num_threads(1)
        group = CommGroup(rank, rank, list(range(world)))
        group.device_group, group.initialized = dist.group.WORLD, True
        vae, config, layout, rows = _tiny_decode_case()
        calls = []
        decode = vae.decoder.forward
        vae.decoder.forward = lambda *args, **kwargs: calls.append(1) or decode(*args, **kwargs)
        with torch.no_grad():
            frames = H3VideoDecoderSubmodule(vae, config, sp_group=group).run_batch(rows, layout)["video_output"]
            torch.save(len(calls), f"{out}.{rank}.calls")
            if rank == 0:  # at the same thread count: CPU reductions round by it
                torch.save(H3VideoDecoderSubmodule(vae, config).run_batch(rows, layout)["video_output"], f"{out}.one")
        torch.save(frames, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_sequence_parallel_decode_gives_every_rank_the_one_nodes_frames(tmp_path, world):
    """``vae_decoder`` as an SP node: each rank's share, gathered, against one decode; 4 ranks
    over 3 clips leaves one rank without a share, which decodes nothing."""
    import torch.multiprocessing as mp

    out = str(tmp_path / "frames")
    mp.spawn(_sp_decode, args=(world, str(tmp_path / "store"), out), nprocs=world)
    want = torch.load(f"{out}.one")
    for rank in range(world):
        assert torch.equal(torch.load(f"{out}.{rank}"), want)
    calls = [torch.load(f"{out}.{rank}.calls") for rank in range(world)]
    assert calls[-1] == (0 if world > 3 else 1) and all(calls[:3]), calls  # one tile per clip


def _sp_cfg():
    from dataclasses import replace

    from mstar.model.minimax_h3.tests.tiny_checkpoint import CFG

    return replace(CFG, num_attention_heads=8)  # Ulysses splits the heads over the ranks


def _sp_dit(rank: int, world: int, store: str, snapshot: str, out: str, head_groups: int = 1) -> None:
    import torch.distributed as dist

    from mstar.distributed.communication import CommGroup
    from mstar.model.minimax_h3 import weight_loader
    from mstar.model.minimax_h3.components.transformer import H3Inputs

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        torch.set_num_threads(1)
        group = CommGroup(rank, rank, list(range(world)))
        group.device_group, group.initialized = dist.group.WORLD, True
        config = H3Config(transformer=_sp_cfg())
        # 31 + 12 keyframe rows: no world here divides the 43 rows
        layout = H3Layout(text_len=5, latent_frames=3, latent_height=4, latent_width=6, audio_latents=4,
                          anchors=("first", "last"))
        g = torch.Generator().manual_seed(1)
        cfg = config.transformer
        video = torch.randn(1, layout.num_condition_rows + layout.num_video_rows, cfg.patch_dim, generator=g)
        audio = torch.randn(1, layout.num_audio_rows, cfg.audio_latents_dim, generator=g)
        text = torch.randn(1, layout.text_len, cfg.hidden_size, generator=g)
        inputs = H3Inputs(video=video, audio=audio, text=text, timesteps=torch.tensor([[0.3, 0.6, 0.999]]))
        for name, sp in (("sp", group), ("one", None)):
            if sp is None and rank:
                break
            dit = weight_loader.build_transformer(config, snapshot, "cpu", sp_group=sp, dtype=torch.float32,
                                                  head_groups=head_groups if sp is not None else 1)
            node = H3DenoiseSubmodule(dit, config, loop_name=DENOISE_LOOP, attn_resource_key=None)
            step_layout = node.build_layout(layout, "cpu")
            projected = []
            dit.video_patch_proj.register_forward_hook(lambda m, args, y, seen=projected: seen.append(args[0].shape[1]))
            with torch.no_grad():
                torch.save(dit(inputs, step_layout, node.attend(step_layout)), f"{out}.{name}.{rank}")
            torch.save(sum(projected), f"{out}.{name}.{rank}.projected")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,head_groups", [(2, 1), (4, 1), (8, 1), (2, 2), (4, 2)])
def test_the_sequence_parallel_dit_gives_every_rank_the_one_ranks_velocity(tmp_path, world, head_groups):
    """``narrow``, ``rope.rows``, the Ulysses exchange and ``gather_sequence`` through ``H3DiT``,
    over rows no rank count divides (padded to one it does), against the unpadded DiT on one rank; with
    ``head_groups`` the attention runs a head group at a time (on CPU the pipeline declines)."""
    import torch.multiprocessing as mp
    from safetensors.torch import save_file

    from mstar.model.minimax_h3.tests.tiny_checkpoint import random_checkpoint

    (tmp_path / "FL2VA" / "transformer").mkdir(parents=True)
    save_file(random_checkpoint(_sp_cfg()), str(tmp_path / "FL2VA" / "transformer" / "model.safetensors"))
    out = str(tmp_path / "v")
    mp.spawn(_sp_dit, args=(world, str(tmp_path / "store"), str(tmp_path), out, head_groups), nprocs=world)
    want = torch.load(f"{out}.one.0")
    for rank in range(world):
        for got, ref in zip(torch.load(f"{out}.sp.{rank}"), want, strict=True):
            torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
    # each of the 12 keyframe and 18 generated rows through the patch projection on one rank only
    assert sum(torch.load(f"{out}.sp.{rank}.projected") for rank in range(world)) == 12 + 18


class _SPStub:
    """What ``H3DenoiseSubmodule`` reads off a transformer to lay rows out."""

    dtype = torch.float32
    blocks = ()

    def __init__(self, world: int):
        from mstar.distributed.communication import CommGroup

        self.sp_group = CommGroup(0, 0, list(range(world)))


@pytest.mark.parametrize(("concurrent", "streams"), [(True, True), (False, False)])
def test_group_streams_need_an_attention_that_runs_on_several_streams_at_once(concurrent, streams):
    """With an attention resource whose calls share scratch (FlashInfer), the head groups attend on one stream."""
    from types import SimpleNamespace

    dit = _SPStub(4)
    dit.ulysses_pipeline = SimpleNamespace(group_streams=True)
    node = H3DenoiseSubmodule(dit, H3Config(), loop_name=DENOISE_LOOP, attn_resource_key=DIT_ATTN)
    node.bind_node_resources({DIT_ATTN: SimpleNamespace(concurrent_streams=concurrent)})
    assert dit.ulysses_pipeline.group_streams is streams


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("row_bucket", [1, 256])
def test_the_padded_rows_split_evenly_over_the_sp_ranks_at_every_prompt_length(world, row_bucket):
    import math

    from mstar.model.components.distributed.sequence_parallel import sp_seq_split

    node = H3DenoiseSubmodule(_SPStub(world), H3Config(), loop_name=DENOISE_LOOP, attn_resource_key=None,
                              row_bucket=row_bucket)
    multiple = math.lcm(row_bucket, world)
    for text_len in range(1, 600):
        layout = H3Layout(text_len=text_len, latent_frames=2, latent_height=4, latent_width=6, audio_latents=3,
                          anchors=("first",) if text_len % 2 else ())
        rows, padded = layout.total_rows, node.num_tokens(layout)
        assert padded % multiple == 0 and rows <= padded < rows + multiple
        assert len(set(sp_seq_split(padded, world))) == 1
        segments = node.attention_segments(layout)
        assert segments == (("main", rows), ("main", padded - rows))[:1 + (padded > rows)]
        if text_len % 97 == 0:
            step_layout = node.build_layout(layout, "cpu")
            assert step_layout.num_rows == padded and step_layout.segments == tuple(span for _, span in segments)


def _padding_case():
    from dataclasses import replace

    from mstar.model.minimax_h3.components.transformer import H3DiT, H3Inputs
    from mstar.model.minimax_h3.tests.tiny_checkpoint import CFG

    torch.manual_seed(0)
    dit = H3DiT(CFG).eval()
    with torch.no_grad():
        for param in dit.parameters():
            param.normal_(0, 0.2)
    config = replace(H3Config(), transformer=CFG)
    layout = H3Layout(text_len=5, latent_frames=3, latent_height=4, latent_width=6, audio_latents=4,
                      anchors=("first",))
    g = torch.Generator().manual_seed(1)
    inputs = H3Inputs(
        video=torch.randn(1, layout.num_condition_rows + layout.num_video_rows, CFG.patch_dim, generator=g),
        audio=torch.randn(1, layout.num_audio_rows, CFG.audio_latents_dim, generator=g),
        text=torch.randn(1, layout.text_len, CFG.hidden_size, generator=g),
        timesteps=torch.tensor([[0.3, 0.6, 0.999]]),
    )
    return dit, config, layout, inputs


def test_padding_rows_change_no_output():
    dit, config, layout, inputs = _padding_case()
    want = None
    for row_bucket in (1, 16, 64):  # 37 rows
        node = H3DenoiseSubmodule(dit, config, loop_name=DENOISE_LOOP, attn_resource_key=None, row_bucket=row_bucket)
        step_layout = node.build_layout(layout, "cpu")
        assert (step_layout.num_pad_rows > 0) == (row_bucket > 1)
        with torch.no_grad():
            got = dit(inputs, step_layout, node.attend(step_layout))
        if want is None:
            want = got
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_padding_rows_stay_finite_and_never_reach_the_request():
    """Zeros in, the padding rows stay finite through every block; made NaN before the first
    block, they still leave every output as it was: attention keeps them in their own segment
    and no output reads them."""
    dit, config, layout, inputs = _padding_case()
    node = H3DenoiseSubmodule(dit, config, loop_name=DENOISE_LOOP, attn_resource_key=None, row_bucket=64)
    step_layout = node.build_layout(layout, "cpu")
    real = step_layout.num_rows - step_layout.num_pad_rows
    seen = []
    hooks = [block.register_forward_hook(lambda m, args, out: seen.append(out[:, real:])) for block in dit.blocks]
    hooks.append(dit.final_layer.register_forward_hook(lambda m, args, out: seen.append(out[:, real:])))
    with torch.no_grad():
        want = dit(inputs, step_layout, node.attend(step_layout))
    for hook in hooks:
        hook.remove()
    assert len(seen) == len(dit.blocks) + 1 and all(torch.isfinite(rows).all() for rows in seen)

    def poison(module, args):
        x = args[0].clone()
        x[:, real:] = float("nan")
        return (x, *args[1:])

    hook = dit.blocks[0].register_forward_pre_hook(poison)
    try:
        with torch.no_grad():
            got = dit(inputs, step_layout, node.attend(step_layout))
    finally:
        hook.remove()
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)


def test_an_sp_size_the_heads_do_not_divide_is_refused_at_load():
    from mstar.distributed.communication import CommGroup
    from mstar.model.minimax_h3.components.transformer import H3DiT

    with torch.device("meta"):
        H3DiT(_sp_cfg(), sp_group=CommGroup(0, 0, [0, 1, 2, 3]))
        with pytest.raises(ValueError, match="8 heads don't split over sp_size 3"):
            H3DiT(_sp_cfg(), sp_group=CommGroup(0, 0, [0, 1, 2]))
        with pytest.raises(ValueError, match="2 heads a rank don't split into 3 head groups"):
            H3DiT(_sp_cfg(), sp_group=CommGroup(0, 0, [0, 1, 2, 3]), head_groups=3)


def test_a_torchaudio_that_does_not_load_fails_startup_by_name(monkeypatch):
    import builtins

    from mstar.model.minimax_h3.minimax_h3_model import import_image_processor

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers" and "AutoImageProcessor" in (fromlist or ()):
            raise OSError("Could not load this library: /venv/site-packages/torchaudio/lib/libtorchaudio.so")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="torchaudio build that matches this torch"):
        import_image_processor()
    serving = H3Model(model_path_hf="test/minimax_h3")
    with pytest.raises(ValueError, match="minimax_h3.yaml: .*torchaudio"):
        serving.validate_config_yaml({}, "minimax_h3.yaml")
    _model().validate_config_yaml({}, "minimax_h3.yaml")  # dummy mode serves nothing


def test_the_vision_tower_keeps_its_frame_attention_through_resource_binding():
    """The engine binds the node's resources into every layer; the tower has none there."""
    from torch import nn

    from mstar.model.components.qwen_vision import QwenVisionConfig, QwenVisionModel
    from mstar.model.minimax_h3.submodules import H3TextEncoderSubmodule

    tower = QwenVisionModel(QwenVisionConfig(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3, patch_size=4,
        temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=24,
    ))
    node = H3TextEncoderSubmodule(nn.Identity(), nn.Identity(), attn_resource_key=REFINER_ATTN, vision=tower,
                                  config=H3Config())
    for resources in ({}, {REFINER_ATTN: object()}):
        node.bind_node_resources(resources)
        assert all(block.attn.ragged_attn is node._frames for block in tower.blocks)



class _StubDiT:
    """Records the step's inputs; returns a zero velocity, so the scheduler leaves every
    generated row where it was."""

    dtype = torch.float32
    blocks = ()  # nothing to compile

    def __init__(self):
        self.calls = []

    def __call__(self, inputs, layout, attend):
        self.calls.append(inputs)
        return torch.zeros_like(inputs.video), torch.zeros_like(inputs.audio)


def _step(node, layout, first, latents=None, cond=None):
    """One ``denoise`` call at sigma == sigma_next, where a zero velocity leaves the rows in place."""
    from mstar.model.minimax_h3.config import KEYFRAME_NOISE_AUG

    cfg = H3Config().transformer
    rows = layout.num_condition_rows + layout.num_video_rows
    latents = torch.zeros(1, rows, cfg.patch_dim) if latents is None else latents

    def scalar(value):
        return torch.tensor([value])

    return node.denoise(
        None, layout, latents, scalar(0.9), scalar(0.8), scalar(0.8),
        torch.zeros(1, layout.num_audio_rows, cfg.audio_latents_dim), scalar(0.9), scalar(0.8), scalar(0.8),
        scalar(KEYFRAME_NOISE_AUG), torch.zeros(1, 3, cfg.hidden_size), torch.tensor(first), cond_latents=cond,
    )


def test_a_compiled_dit_sees_one_timestep_table_shape():
    """Eager step 0 without keyframes embeds one timestep, as the reference; compiled, it
    keeps the three-entry table, so a row count compiles one graph, not two."""
    layout = H3Layout(text_len=3, latent_frames=2, latent_height=4, latent_width=4, audio_latents=2)
    for compiled, step0 in ((False, (1, 1)), (True, (1, 3))):
        stub = _StubDiT()
        node = H3DenoiseSubmodule(stub, H3Config(), loop_name=DENOISE_LOOP, attn_resource_key=None,
                                  compile_blocks=compiled)
        for first in (True, False):
            _step(node, layout, first)
        assert [tuple(call.timesteps.shape) for call in stub.calls] == [step0, (1, 3)]


def test_step_0_gives_the_same_velocity_on_one_timestep_or_three():
    from dataclasses import replace

    from mstar.model.minimax_h3.components.transformer import H3DiT, H3Inputs
    from mstar.model.minimax_h3.tests.tiny_checkpoint import CFG

    torch.manual_seed(0)
    dit = H3DiT(CFG).eval()
    with torch.no_grad():
        for param in dit.parameters():
            param.normal_(0, 0.2)
    node = H3DenoiseSubmodule(dit, replace(H3Config(), transformer=CFG), loop_name=DENOISE_LOOP,
                              attn_resource_key=None)
    layout = H3Layout(text_len=5, latent_frames=3, latent_height=4, latent_width=6, audio_latents=4)
    step_layout = node.build_layout(layout, "cpu")
    g = torch.Generator().manual_seed(1)
    video = torch.randn(1, layout.num_video_rows, CFG.patch_dim, generator=g)
    audio = torch.randn(1, layout.num_audio_rows, CFG.audio_latents_dim, generator=g)
    text = torch.randn(1, layout.text_len, CFG.hidden_size, generator=g)
    with torch.no_grad():
        one, three = (dit(H3Inputs(video=video, audio=audio, text=text, timesteps=t), step_layout, sdpa_attention)
                      for t in (torch.zeros(1, 1), torch.tensor([[0.0, 0.0, 0.999]])))
    for got, want in zip(three, one, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_denoise_noises_keyframe_rows_at_step_0_and_carries_them():
    from mstar.model.minimax_h3.config import KEYFRAME_NOISE_AUG

    layout = H3Layout(text_len=3, latent_frames=2, latent_height=4, latent_width=4, audio_latents=2,
                      anchors=("first",))
    stub = _StubDiT()
    node = H3DenoiseSubmodule(stub, H3Config(), loop_name=DENOISE_LOOP, attn_resource_key=None)
    n, rows, channels = layout.num_condition_rows, layout.num_video_rows, H3Config().transformer.patch_dim
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(1, n + rows, channels, generator=g)
    audio = torch.randn(1, layout.num_audio_rows, H3Config().transformer.audio_latents_dim, generator=g)
    cond = torch.randn(1, n, channels, generator=g)

    def scalar(value):
        return torch.tensor([value])

    def step(latents, first):
        # sigma == sigma_next: a zero velocity then leaves the generated rows exactly in place
        return node.denoise(
            None, layout, latents, scalar(0.9), scalar(0.8), scalar(0.8), audio, scalar(0.9), scalar(0.8),
            scalar(0.8), scalar(KEYFRAME_NOISE_AUG), torch.zeros(1, 3, H3Config().transformer.hidden_size),
            torch.tensor(first), cond_latents=cond,
        )

    t = torch.tensor(KEYFRAME_NOISE_AUG, dtype=torch.float32)
    noised = t * cond + (1.0 - t) * noise[:, :n]
    out = step(noise, True)[LATENTS]
    assert torch.equal(stub.calls[-1].video[:, :n], noised)  # the DiT sees the noised keyframes
    assert torch.equal(out[:, :n], noised) and torch.equal(out[:, n:], noise[:, n:])
    assert stub.calls[-1].timesteps.shape == (1, 3)  # keyframe rows keep their own timestep
    later = step(out, False)[LATENTS]
    assert torch.equal(later[:, :n], noised)  # carried unchanged after step 0
    assert torch.equal(stub.calls[-1].video[:, :n], noised)
