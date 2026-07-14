"""Dummy-mode tests for the Wan2.2-TI2V-5B model declaration.

Structural pieces only — no weights, no GPU, no network: graph walks, engine
types, the denoise loop's loop-back inventory, persist flags, the schedule-driven
walk transitions, and the check_stop boundary math. The model is built via
``object.__new__`` with hand-set fields, so the real constructor's lazy-tokenizer
seam is never exercised.
"""

import sys

sys.path.insert(0, ".")

from pathlib import Path

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import EngineType
from mstar.graph.base import GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.wan22.config import WAN22_VARIANT_TI2V_5B, Wan22Config
from mstar.model.wan22.submodules import DENOISE_LOOP_NAME, Wan22DitSubmodule
from mstar.model.wan22.wan22_model import Wan22Model

CONFIG_PATH = str(
    Path(__file__).resolve().parents[2] / "configs" / "wan22.yaml"
)

# The dit node's loop-back edges (see Wan22DitSubmodule for the inventory).
LOOP_BACK_NAMES = {"latents", "time_index", "unipc_model_outputs", "unipc_last_sample"}


def _make_model() -> Wan22Model:
    """Construct a Wan2.2 model without touching tokenizer or weights."""
    model = object.__new__(Wan22Model)
    model.model_path_hf = "test/wan22"
    model.cache_dir = None
    model.config = Wan22Config()
    model.skip_weight_loading = True  # dummy mode: get_submodule returns None
    model.tokenizer = None
    model._tokenizer_initialized = True  # pin the byte-fallback path
    model._submodule_cache = {}
    model._encode_vae = None
    return model


def _tensor_info(name: str) -> TensorPointerInfo:
    """Minimal TensorPointerInfo stand-in for persist-signal plumbing."""
    return TensorPointerInfo(
        dims=[1], dtype="torch.float32", nbytes=4, address=0, stride=[1],
        uuid=f"uuid-{name}", source_session_id="test:0", source_entity="test",
    )


# ----------------------------------------------------------------------
# Variant seam
# ----------------------------------------------------------------------


def test_wan22_variant_seam_rejects_non_ti2v_5b():
    with pytest.raises(NotImplementedError, match="a14b"):
        Wan22Model(model_path_hf="test/wan22", variant="a14b")


def test_wan22_constructor_accepts_ti2v_5b_without_network():
    model = Wan22Model(model_path_hf="test/wan22")
    assert model.config.variant == WAN22_VARIANT_TI2V_5B
    assert model.tokenizer is None
    assert model._submodule_cache == {}


# ----------------------------------------------------------------------
# Graph structure
# ----------------------------------------------------------------------


def test_wan22_graph_walks_have_expected_keys():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    assert set(walks.keys()) == {
        Wan22Model.ENCODE_TEXT_WALK,
        Wan22Model.ENCODE_IMAGE_WALK,
        Wan22Model.VIDEO_GEN_WALK,
        Wan22Model.VIDEO_GEN_I2V_WALK,
    }


def test_wan22_node_engine_types_all_stateless():
    model = _make_model()
    types = model.get_node_engine_types()
    assert types == {
        "text_encoder": EngineType.STATELESS,
        "vae_encoder": EngineType.STATELESS,
        "dit": EngineType.STATELESS,
        "vae_decoder": EngineType.STATELESS,
    }
    # Node names in the walks must all be registered engine types.
    walk_nodes = set()
    for walk in model.get_graph_walk_graphs().values():
        walk_nodes.update(walk.get_nodes().keys())
    assert walk_nodes == set(types.keys())


def test_wan22_kv_cache_config_is_empty():
    model = _make_model()
    assert model.get_kv_cache_config() == []


def test_wan22_encoder_walks_persist_embeddings():
    model = _make_model()
    walks = model.get_graph_walk_graphs()

    encode_text = walks[Wan22Model.ENCODE_TEXT_WALK]
    assert isinstance(encode_text, GraphNode) and encode_text.name == "text_encoder"
    assert encode_text.input_names == {"text_inputs"}
    by_name = {e.name: e for e in encode_text.outputs}
    assert set(by_name) == {"text_embeds_pos", "text_embeds_neg"}
    for edge in by_name.values():
        assert edge.persist is True
        assert edge.next_node == EMPTY_DESTINATION

    encode_image = walks[Wan22Model.ENCODE_IMAGE_WALK]
    assert isinstance(encode_image, GraphNode) and encode_image.name == "vae_encoder"
    assert encode_image.input_names == {"image_inputs"}
    (latent_edge,) = encode_image.outputs
    assert latent_edge.name == "image_latent"
    assert latent_edge.persist is True
    assert latent_edge.next_node == EMPTY_DESTINATION


@pytest.mark.parametrize("walk_name,expect_image_latent", [
    (Wan22Model.VIDEO_GEN_WALK, False),
    (Wan22Model.VIDEO_GEN_I2V_WALK, True),
])
def test_wan22_video_gen_walk_structure(walk_name, expect_image_latent):
    model = _make_model()
    walk = model.get_graph_walk_graphs()[walk_name]

    assert isinstance(walk, Sequential)
    assert len(walk.sections) == 2
    loop, decoder = walk.sections

    # -- denoise loop: named, ceiling-bounded, dit body, exact loop-backs --
    assert isinstance(loop, Loop)
    assert loop.name == DENOISE_LOOP_NAME == "denoise_loop"
    assert loop.max_iters == model.config.max_denoise_steps
    body = loop.section
    assert isinstance(body, GraphNode) and body.name == "dit"
    # Async scheduling off, or a speculative iteration overshoots the stop.
    assert body.enable_async_scheduling is False
    assert {e.name for e in body.outputs} == LOOP_BACK_NAMES
    assert all(e.next_node == "dit" for e in body.outputs)

    expected_inputs = {"text_embeds_pos", "text_embeds_neg"} | LOOP_BACK_NAMES
    if expect_image_latent:
        expected_inputs.add("image_latent")
    assert body.input_names == expected_inputs

    # -- terminal output: final latents -> vae_decoder, matched by name
    assert len(loop.outputs) == 1
    terminal = loop.outputs[0]
    assert terminal.next_node == "vae_decoder"
    assert terminal.name == "latents"
    assert terminal.name in LOOP_BACK_NAMES

    # -- vae_decoder emits the video to the client --
    assert isinstance(decoder, GraphNode) and decoder.name == "vae_decoder"
    assert decoder.input_names == {"latents"}
    (emit,) = decoder.outputs
    assert emit.next_node == EMIT_TO_CLIENT
    assert emit.name == "video_output"
    assert emit.output_modality == "video"


def test_wan22_worker_graphs_from_yaml():
    model = _make_model()
    worker_graphs = model.get_worker_graphs(CONFIG_PATH)
    walks_seen = set()
    for wg in worker_graphs:
        walks_seen.update(wg.graph_walks)
    assert walks_seen == {
        Wan22Model.ENCODE_TEXT_WALK,
        Wan22Model.ENCODE_IMAGE_WALK,
        Wan22Model.VIDEO_GEN_WALK,
        Wan22Model.VIDEO_GEN_I2V_WALK,
    }


# ----------------------------------------------------------------------
# Forward pass transitions (schedule-driven state machine)
# ----------------------------------------------------------------------


def test_wan22_initial_schedule_t2v():
    model = _make_model()
    args = model.get_initial_forward_pass_args(
        partition_name="default",
        input_modalities=["text"],
        output_modalities=["video"],
        input_signals={"text_inputs": [_tensor_info("text")]},
    )
    assert args.full_metadata.graph_walk == Wan22Model.ENCODE_TEXT_WALK
    assert args.full_metadata.is_prefill is True
    assert args.full_metadata.kwargs["walk_schedule"] == [
        Wan22Model.ENCODE_TEXT_WALK,
        Wan22Model.VIDEO_GEN_WALK,
    ]
    edge_targets = {(e.next_node, e.name) for e in args.inputs}
    assert edge_targets == {("text_encoder", "text_inputs")}
    assert args.step_metadata["num_inference_steps"] == \
        model.config.default_num_inference_steps
    assert args.step_metadata["guidance_scale"] == model.config.guidance_scale


def test_wan22_initial_schedule_i2v_inserts_encode_image():
    model = _make_model()
    args = model.get_initial_forward_pass_args(
        partition_name="default",
        input_modalities=["text", "image"],
        output_modalities=["video"],
        input_signals={
            "text_inputs": [_tensor_info("text")],
            "image_inputs": [_tensor_info("image")],
        },
    )
    assert args.full_metadata.kwargs["walk_schedule"] == [
        Wan22Model.ENCODE_TEXT_WALK,
        Wan22Model.ENCODE_IMAGE_WALK,
        Wan22Model.VIDEO_GEN_I2V_WALK,
    ]


def test_wan22_rejects_non_video_output():
    model = _make_model()
    with pytest.raises(ValueError, match="video"):
        model.get_initial_forward_pass_args(
            partition_name="default",
            input_modalities=["text"],
            output_modalities=["text"],
            input_signals={"text_inputs": [_tensor_info("text")]},
        )


def test_wan22_rejects_promptless_request():
    model = _make_model()
    with pytest.raises(ValueError, match="text prompt"):
        model.get_initial_forward_pass_args(
            partition_name="default",
            input_modalities=["image"],
            output_modalities=["video"],
            input_signals={"image_inputs": [_tensor_info("image")]},
        )


@pytest.mark.parametrize("requested,expected", [
    (1, 1),
    (50, 50),
    (100, 100),   # exactly the ceiling
    (10_000, 100),  # clamped to max_denoise_steps
    (0, 1),       # floored at one step
])
def test_wan22_num_inference_steps_clamped(requested, expected):
    model = _make_model()
    assert model.config.max_denoise_steps == 100
    args = model.get_initial_forward_pass_args(
        partition_name="default",
        input_modalities=["text"],
        output_modalities=["video"],
        input_signals={"text_inputs": [_tensor_info("text")]},
        model_kwargs={"num_inference_steps": requested},
    )
    assert args.step_metadata["num_inference_steps"] == expected


def _step(model, metadata, persist_signals):
    return model.get_partition_forward_pass_args(
        partition_name="default",
        partition_metadata=metadata,
        persist_signals=persist_signals,
    )


def test_wan22_t2v_transitions_encode_text_to_video_gen_to_done():
    model = _make_model()
    args = model.get_initial_forward_pass_args(
        partition_name="default",
        input_modalities=["text"],
        output_modalities=["video"],
        input_signals={"text_inputs": [_tensor_info("text")]},
    )
    metadata = args.full_metadata

    persist = {
        "text_embeds_pos": [_tensor_info("pos")],
        "text_embeds_neg": [_tensor_info("neg")],
    }
    result = _step(model, metadata, persist)
    assert result.full_metadata.graph_walk == Wan22Model.VIDEO_GEN_WALK
    assert result.full_metadata.is_prefill is False
    assert result.request_done is False
    edge_targets = {(e.next_node, e.name) for e in result.inputs}
    assert edge_targets == {("dit", name) for name in (
        {"text_embeds_pos", "text_embeds_neg"} | LOOP_BACK_NAMES
    )}
    # Persisted embeddings carry tensor_info; loop-seeding edges arrive empty.
    by_name = {e.name: e for e in result.inputs}
    assert by_name["text_embeds_pos"].tensor_info == persist["text_embeds_pos"]
    assert by_name["text_embeds_neg"].tensor_info == persist["text_embeds_neg"]
    for name in LOOP_BACK_NAMES:
        assert by_name[name].tensor_info == []
    # The embeddings see their last use here and must be unpersisted.
    unpersisted = {info.uuid for info in result.unpersist_tensors}
    assert {persist["text_embeds_pos"][0].uuid, persist["text_embeds_neg"][0].uuid} \
        <= unpersisted

    result = _step(model, result.full_metadata, {})
    assert result.request_done is True
    assert result.inputs == []


def test_wan22_i2v_transitions_include_encode_image_and_image_latent():
    model = _make_model()
    args = model.get_initial_forward_pass_args(
        partition_name="default",
        input_modalities=["text", "image"],
        output_modalities=["video"],
        input_signals={
            "text_inputs": [_tensor_info("text")],
            "image_inputs": [_tensor_info("image")],
        },
    )
    metadata = args.full_metadata

    # encode_text -> encode_image: the persisted raw image feeds vae_encoder.
    persist = {"image_inputs": [_tensor_info("image")]}
    result = _step(model, metadata, persist)
    assert result.full_metadata.graph_walk == Wan22Model.ENCODE_IMAGE_WALK
    assert result.full_metadata.is_prefill is True
    edge_targets = {(e.next_node, e.name) for e in result.inputs}
    assert edge_targets == {("vae_encoder", "image_inputs")}

    # encode_image -> video_gen_i2v: dit also consumes the persisted latent.
    persist = {
        "text_embeds_pos": [_tensor_info("pos")],
        "text_embeds_neg": [_tensor_info("neg")],
        "image_latent": [_tensor_info("latent")],
    }
    result = _step(model, result.full_metadata, persist)
    assert result.full_metadata.graph_walk == Wan22Model.VIDEO_GEN_I2V_WALK
    assert result.full_metadata.is_prefill is False
    by_name = {e.name: e for e in result.inputs}
    assert by_name["image_latent"].tensor_info == persist["image_latent"]

    result = _step(model, result.full_metadata, {})
    assert result.request_done is True


# ----------------------------------------------------------------------
# check_stop boundary math
# ----------------------------------------------------------------------


def _fwd_info(requested_steps: int, iter_count: int) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id="r0",
        graph_walk=Wan22Model.VIDEO_GEN_WALK,
        requires_cfg=False,
        fwd_index=0,
        random_seed=0,
        max_tokens=0,
        sampling_config={},
        step_metadata={"num_inference_steps": requested_steps},
        dynamic_loop_iter_counts={DENOISE_LOOP_NAME: iter_count},
    )


# iter_count reads k while 0-based iteration k is being postprocessed, so
# stopping at k == N - 1 runs exactly N iterations.
@pytest.mark.parametrize("requested_steps,iter_count,expect_stop", [
    (1, 0, True),     # single-step request stops on its only iteration
    (2, 0, False),
    (2, 1, True),
    (50, 0, False),
    (50, 48, False),  # one before the boundary: keep going
    (50, 49, True),   # k + 1 == N: exactly N iterations run
    (50, 50, True),   # deferred-stop overshoot (speculative iter N) still stops
    (100, 99, True),  # request at the ceiling stops on the loop's last iter
])
def test_wan22_check_stop_boundary(requested_steps, iter_count, expect_stop):
    submodule = Wan22DitSubmodule(transformer=None, config=Wan22Config())
    stops = submodule.check_stop("r0", _fwd_info(requested_steps, iter_count), outputs={})
    assert (DENOISE_LOOP_NAME in stops) == expect_stop
    if not expect_stop:
        assert stops == set()


def test_wan22_check_stop_without_step_metadata_never_stops():
    # A missing step count must not stop the loop; max_iters still bounds it.
    submodule = Wan22DitSubmodule(transformer=None, config=Wan22Config())
    info = _fwd_info(requested_steps=0, iter_count=99)
    info.step_metadata = {}
    assert submodule.check_stop("r0", info, outputs={}) == set()


# ----------------------------------------------------------------------
# Prompt processing / deferred surfaces
# ----------------------------------------------------------------------


def test_wan22_process_prompt_byte_fallback_emits_pos_and_neg():
    model = _make_model()
    result = model.process_prompt(
        prompt="a cat surfing",
        input_modalities=["text"],
        output_modalities=["video"],
        negative_prompt="blurry",
    )
    assert set(result.keys()) == {"text_inputs"}
    pos, neg = result["text_inputs"]
    assert torch.equal(pos, torch.tensor(list(b"a cat surfing"), dtype=torch.uint8))
    assert torch.equal(neg, torch.tensor(list(b"blurry"), dtype=torch.uint8))


# ----------------------------------------------------------------------
# Malformed requests must be rejected by process_prompt, which is the seam that
# produces a 400. See Wan22Model._validate_request.
# ----------------------------------------------------------------------


def test_wan22_process_prompt_without_prompt_raises():
    model = _make_model()
    with pytest.raises(ValueError, match="requires a text prompt"):
        model.process_prompt(
            prompt=None, input_modalities=["image"], output_modalities=["video"],
        )


def test_wan22_process_prompt_rejects_non_video_output():
    model = _make_model()
    with pytest.raises(ValueError, match="only generates video"):
        model.process_prompt(
            prompt="a cat", input_modalities=["text"], output_modalities=["image"],
        )


def test_wan22_process_prompt_rejects_declared_image_without_tensor():
    model = _make_model()
    with pytest.raises(ValueError, match="declared an image input"):
        model.process_prompt(
            prompt="a cat", input_modalities=["image", "text"],
            output_modalities=["video"], tensors={},
        )


def test_wan22_process_prompt_accepts_i2v_with_matching_image():
    # The conditioning image must be the request's HxW (the server does not resize).
    model = _make_model()
    result = model.process_prompt(
        prompt="a cat", input_modalities=["image", "text"], output_modalities=["video"],
        height=64, width=64, tensors={"image_inputs": [torch.zeros(3, 64, 64)]},
    )
    assert set(result.keys()) == {"text_inputs"}


def test_wan22_process_prompt_rejects_mismatched_conditioning_image():
    # A wrong-sized image is a 400 at this seam, not a shape fault on the worker.
    model = _make_model()
    with pytest.raises(ValueError, match="conditioning image is 64x64 .HxW. but the request is 96x96"):
        model.process_prompt(
            prompt="a cat", input_modalities=["image", "text"], output_modalities=["video"],
            height=96, width=96, tensors={"image_inputs": [torch.zeros(3, 64, 64)]},
        )


# ----------------------------------------------------------------------
# Size: H and W must be multiples of 32. See Wan22Config.spatial_alignment.
# ----------------------------------------------------------------------


def test_wan22_spatial_alignment_is_32_on_both_axes():
    assert Wan22Config().spatial_alignment == (32, 32)


@pytest.mark.parametrize("height,width", [
    (480, 832),    # the oracle tier
    (704, 1280),   # the 720p-class tier (704 IS the valid one, not 720)
    (1280, 704),   # portrait
    (32, 32),      # the minimum aligned grid
])
def test_wan22_process_prompt_accepts_aligned_sizes(height, width):
    model = _make_model()
    result = model.process_prompt(
        prompt="a cat surfing", input_modalities=["text"], output_modalities=["video"],
        height=height, width=width,
    )
    assert set(result.keys()) == {"text_inputs"}


@pytest.mark.parametrize("height,width,bad_axis,nearest", [
    (720, 1280, "height", "704 or 736"),   # 720/32 = 22.5 — not a size this model can make
    (480, 900, "width", "896 or 928"),
    (481, 832, "height", "480 or 512"),
    (480, 16, "width", "width: 32"),       # below one full cell: 0 is no size, so only "up" is offered
])
def test_wan22_process_prompt_rejects_unaligned_sizes(height, width, bad_axis, nearest):
    model = _make_model()
    with pytest.raises(ValueError) as exc:
        model.process_prompt(
            prompt="a cat surfing", input_modalities=["text"], output_modalities=["video"],
            height=height, width=width,
        )
    message = str(exc.value)
    assert bad_axis in message
    assert "multiple of 32" in message
    assert nearest in message


@pytest.mark.parametrize("kwargs,fragment", [
    ({"height": 0}, "must be positive"),
    ({"width": -32}, "must be positive"),
    ({"height": "tall"}, "must be an integer"),
    ({"width": None}, "must be an integer"),
])
def test_wan22_process_prompt_rejects_malformed_sizes(kwargs, fragment):
    model = _make_model()
    with pytest.raises(ValueError, match=fragment):
        model.process_prompt(
            prompt="a cat surfing", input_modalities=["text"],
            output_modalities=["video"], **kwargs,
        )


def test_wan22_process_prompt_defaults_are_aligned():
    model = _make_model()
    assert model.process_prompt(
        prompt="a cat surfing", input_modalities=["text"], output_modalities=["video"],
    )["text_inputs"]


# ----------------------------------------------------------------------
# num_frames must be 4k+1. Unaligned counts are silently floored, and
# non-positive ones crash the worker; both must be 400s instead.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("frames", [1, 33, 81, 121])  # the defaults and both tiers
def test_wan22_process_prompt_accepts_4k_plus_1_frames(frames):
    model = _make_model()
    result = model.process_prompt(
        prompt="a cat surfing", input_modalities=["text"], output_modalities=["video"],
        num_frames=frames,
    )
    assert set(result.keys()) == {"text_inputs"}


@pytest.mark.parametrize("frames,nearest", [
    (32, "29 or 33"),    # would have silently returned 29 frames
    (34, "33 or 37"),
    (80, "77 or 81"),
])
def test_wan22_process_prompt_rejects_unaligned_num_frames(frames, nearest):
    model = _make_model()
    with pytest.raises(ValueError) as exc:
        model.process_prompt(
            prompt="a cat surfing", input_modalities=["text"],
            output_modalities=["video"], num_frames=frames,
        )
    message = str(exc.value)
    assert "num_frames" in message
    assert "4k+1" in message
    assert nearest in message


@pytest.mark.parametrize("frames,fragment", [
    (0, "must be positive"),
    (-4, "must be positive"),      # would have crashed the worker on a negative dim
    ("many", "must be an integer"),
])
def test_wan22_process_prompt_rejects_malformed_num_frames(frames, fragment):
    model = _make_model()
    with pytest.raises(ValueError, match=fragment):
        model.process_prompt(
            prompt="a cat surfing", input_modalities=["text"],
            output_modalities=["video"], num_frames=frames,
        )


# ----------------------------------------------------------------------
# fps must be refused up front, not in postprocess: postprocess runs after the
# video is generated and its raise is swallowed.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fps", [24, 30, 23.976, None])  # None == unset -> config default
def test_wan22_process_prompt_accepts_valid_fps(fps):
    model = _make_model()
    result = model.process_prompt(
        prompt="a cat surfing", input_modalities=["text"],
        output_modalities=["video"], fps=fps,
    )
    assert set(result.keys()) == {"text_inputs"}


@pytest.mark.parametrize("fps,fragment", [
    (0, "must be positive"),
    (-1, "must be positive"),
    ("fast", "must be a number"),
])
def test_wan22_process_prompt_rejects_bad_fps(fps, fragment):
    model = _make_model()
    with pytest.raises(ValueError, match=fragment):
        model.process_prompt(
            prompt="a cat surfing", input_modalities=["text"],
            output_modalities=["video"], fps=fps,
        )


def test_wan22_get_submodule_dummy_mode_returns_none():
    model = _make_model()
    for node in ("text_encoder", "vae_encoder", "dit", "vae_decoder"):
        assert model.get_submodule(node) is None
        assert node in model._submodule_cache  # None is cached, not retried


def test_wan22_i2v_declared_without_image_tensor_raises():
    model = _make_model()
    with pytest.raises(ValueError, match="image_inputs"):
        model.get_initial_forward_pass_args(
            partition_name="default",
            input_modalities=["text", "image"],
            output_modalities=["video"],
            input_signals={"text_inputs": [_tensor_info("text")]},
        )


def _uint8_video(frames=4, height=64, width=96):
    return torch.randint(0, 256, (1, 3, frames, height, width), dtype=torch.uint8)


def test_wan22_postprocess_video_encodes_decodable_mp4():
    av = pytest.importorskip("av")
    import io

    model = _make_model()
    frames, height, width = 4, 64, 96
    video = _uint8_video(frames, height, width)
    # Absent kwargs and an explicit fps: null both mean "unset".
    for kwargs in (None, {"fps": None}):
        data = model.postprocess(video, modality="video", request_kwargs=kwargs)
        assert isinstance(data, bytes) and len(data) > 0
        container = av.open(io.BytesIO(data))
        decoded = [f for f in container.decode(video=0)]
        assert len(decoded) == frames
        assert (decoded[0].height, decoded[0].width) == (height, width)
        assert container.streams.video[0].average_rate == model.config.video_fps


def test_wan22_postprocess_honors_request_fps():
    av = pytest.importorskip("av")
    import io

    model = _make_model()
    request_fps = 16
    assert request_fps != model.config.video_fps  # or the assertion is vacuous
    data = model.postprocess(_uint8_video(), modality="video", request_kwargs={"fps": request_fps})
    container = av.open(io.BytesIO(data))
    assert container.streams.video[0].average_rate == request_fps


def test_wan22_postprocess_rejects_bad_fps():
    pytest.importorskip("av")
    model = _make_model()
    for bad in (0, -24, "fast"):
        with pytest.raises(ValueError, match="fps"):
            model.postprocess(_uint8_video(), modality="video", request_kwargs={"fps": bad})


def test_wan22_postprocess_rejects_non_video_modality():
    model = _make_model()
    with pytest.raises(ValueError):
        model.postprocess(torch.zeros(1), modality="text")
