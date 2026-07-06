"""CPU-only checks for the Cosmos3 OpenAI-serving entry points.

Covers the request -> model wiring that the engine relies on: prompt
tokenization into a conditional + unconditional pair, generation-parameter
resolution + step-metadata threading, and the OpenAI image adapter. No GPU and
no model weights are required. The prompt-tokenization check needs a real
tokenizer, so point ``COSMOS3_NANO_DIR`` at a Cosmos3-Nano directory to run it
(it is skipped otherwise).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mstar.model.cosmos3.cosmos3_model import Cosmos3Model

NANO_DIR = Path(
    os.environ.get(
        "COSMOS3_NANO_DIR",
        "/Users/atindrajha/Downloads/disaggregation_research/Cosmos3-Nano-hf",
    )
)


def test_adapter_registered_for_images() -> None:
    from mstar.api_server.openai.adapters import get_adapter

    adapter = get_adapter("cosmos3")
    assert adapter is not None
    assert adapter.supports_images

    class _Req:
        prompt = "a red cube"
        size = "512x512"
        seed = 7

        def __init__(self):
            self.model_extra = {"guidance_scale": 4.0}

    args = adapter.image_to_request(_Req(), upload_dir="/tmp")
    assert args.text == "a red cube"
    assert args.output_modalities == ["image"]
    assert args.model_kwargs["size"] == "512x512"
    assert args.model_kwargs["seed"] == 7
    assert args.model_kwargs["guidance_scale"] == 4.0


def test_video_adapter_t2v_and_i2v(tmp_path) -> None:
    from mstar.api_server.openai.adapters import get_adapter
    from mstar.api_server.openai.protocol import VideoGenerationRequest

    adapter = get_adapter("cosmos3")
    assert adapter is not None and adapter.supports_videos

    # text-to-video: text-only input, video output, num_frames/fps threaded.
    req = VideoGenerationRequest(
        prompt="a kite", size="256x256", seed=1, num_frames=17, fps=16.0,
        guidance_scale=6.0,
    )
    args = adapter.video_to_request(req, upload_dir=str(tmp_path))
    assert args.text == "a kite"
    assert args.input_modalities == ["text"]
    assert args.output_modalities == ["video"]
    assert args.file_paths is None
    assert args.model_kwargs["num_frames"] == 17
    assert args.model_kwargs["fps"] == 16.0
    assert args.model_kwargs["guidance_scale"] == 6.0

    # image-to-video: the conditioning image (data URI) is persisted and routed
    # in as an image input; the worker VAE-encodes it into the frame-0 anchor.
    i2v = adapter.video_to_request(
        VideoGenerationRequest(prompt="zoom in", image="data:image/png;base64,AAAA"),
        upload_dir=str(tmp_path),
    )
    assert i2v.input_modalities == ["image", "text"]
    assert i2v.output_modalities == ["video"]
    assert i2v.file_paths and i2v.file_paths["image"]


def test_gen_params_and_step_metadata() -> None:
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)

    # "size" parses to width/height; explicit width/height win; defaults applied.
    p = model._resolve_gen_params({"size": "480x256"}, ["text"], ["image"])
    assert (p["width"], p["height"]) == (480, 256)
    assert p["num_frames"] == 1 and p["has_image_condition"] is False

    # The denoise loop stops per-request (check_stop), so a per-request
    # num_inference_steps is honored, clamped to [1, max_inference_steps];
    # guidance_scale is likewise per request.
    p = model._resolve_gen_params(
        {"num_inference_steps": 3, "guidance_scale": 2.5}, ["text"], ["image"]
    )
    assert p["num_inference_steps"] == 3
    assert p["guidance_scale"] == 2.5
    # A request above the loop's upper bound is clamped; the image/video defaults
    # differ by mode.
    assert model._resolve_gen_params(
        {"num_inference_steps": 10_000}, ["text"], ["image"]
    )["num_inference_steps"] == model.config.max_inference_steps
    assert model._resolve_gen_params({}, ["text"], ["image"])[
        "num_inference_steps"
    ] == model.config.num_inference_steps
    assert model._resolve_gen_params({"num_frames": 17}, ["text"], ["video"])[
        "num_inference_steps"
    ] == model.config.num_inference_steps_video

    # i2v conditioning is inferred from the input modalities.
    p = model._resolve_gen_params({}, ["image", "text"], ["image"])
    assert p["has_image_condition"] is True

    fpa = model.get_initial_forward_pass_args(
        "p0", ["text"], ["image"], {"text_inputs": []},
        model_kwargs={"size": "256x256", "num_inference_steps": 7},
    )
    sm = fpa.step_metadata
    assert sm["is_prefill"] is True
    assert sm["height"] == 256 and sm["width"] == 256
    assert sm["num_inference_steps"] == 7


def test_dynamic_loop_check_stop_and_wasted_step() -> None:
    """The denoise loop stops at each request's own step count, and a step
    dispatched one past that count is a no-op — so the loop's single speculative
    extra iteration can't index the scheduler out of range."""
    import types

    import torch

    from mstar.model.cosmos3.submodules import (
        ACTION_GEN_LOOP,
        ACTION_GEN_WALK,
        IMAGE_GEN_LOOP,
        IMAGE_GEN_WALK,
        Cosmos3DiTSubmodule,
    )

    dit = Cosmos3DiTSubmodule(transformer=None, config=Cosmos3Model(
        model_path_hf="unused", skip_weight_loading=True).config, scheduler=None)

    class _Sched:
        def __init__(self, n):
            self.timesteps = list(range(n))

    n = 4
    dit.request_state("r").add_all(scheduler=_Sched(n), raw_action_dim=2)

    def info(walk, it):
        return types.SimpleNamespace(
            graph_walk=walk,
            dynamic_loop_iter_counts={IMAGE_GEN_LOOP: it, ACTION_GEN_LOOP: it},
        )

    # Stops only on the last real step (iter n-1), not before; routes by walk.
    assert dit.check_stop("r", info(IMAGE_GEN_WALK, n - 2), {}) == set()
    assert dit.check_stop("r", info(IMAGE_GEN_WALK, n - 1), {}) == {IMAGE_GEN_LOOP}
    assert dit.check_stop("r", info(ACTION_GEN_WALK, n - 1), {}) == {ACTION_GEN_LOOP}
    # Unknown request -> no stop.
    assert dit.check_stop("missing", info(IMAGE_GEN_WALK, 0), {}) == set()

    # A forward one past the step count returns its inputs unchanged without
    # touching the transformer or cache manager (both None here).
    lat = torch.zeros(1, 4, 1, 2, 2)
    ti = torch.tensor([n])
    out = dit._forward_image_gen(None, dit.request_states["r"], latents=lat, time_index=ti)
    assert torch.equal(out["latents"][0], lat) and torch.equal(out["time_index"][0], ti)

    act = torch.zeros(1, 3, 5)
    out = dit._forward_action_gen(
        None, dit.request_states["r"], latents=lat, action_latents=act, time_index=ti
    )
    assert torch.equal(out["latents"][0], lat)
    # The action latents (the looped self-edge the loop emits on finish) pass
    # through unchanged on the discarded extra step.
    assert torch.equal(out["action_latents"][0], act)


@pytest.mark.skipif(not NANO_DIR.exists(), reason="set COSMOS3_NANO_DIR to a Cosmos3-Nano dir")
def test_process_prompt_emits_cond_and_uncond() -> None:
    model = Cosmos3Model(model_path_hf=str(NANO_DIR))
    assert model.tokenizer is not None
    sog = model.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    eos = model.tokenizer.eos_token_id

    out = model.process_prompt("a red cube on a table", ["text"], ["image"], tensors={}, size="256x256")
    ti = out["text_inputs"]
    assert len(ti) == 2, "t2i must emit a conditional and unconditional prompt"
    cond, uncond = ti[0].tolist(), ti[1].tolist()
    assert cond[-2:] == [eos, sog]
    assert uncond[-2:] == [eos, sog]
    assert cond != uncond


def test_video_adapter_v2v(tmp_path) -> None:
    from mstar.api_server.openai.adapters import get_adapter
    from mstar.api_server.openai.protocol import VideoGenerationRequest

    adapter = get_adapter("cosmos3")
    assert adapter is not None

    # video-to-video: the conditioning video (data URI) is persisted and routed
    # in as a video input; the conditioning knobs flow through as model kwargs.
    v2v = adapter.video_to_request(
        VideoGenerationRequest(
            prompt="keep going", video="data:video/mp4;base64,AAAA",
            condition_frame_indexes_vision="0,1", condition_video_keep="last",
        ),
        upload_dir=str(tmp_path),
    )
    assert v2v.input_modalities == ["video", "text"]
    assert v2v.output_modalities == ["video"]
    assert v2v.file_paths and v2v.file_paths["video"]
    assert v2v.model_kwargs["condition_frame_indexes_vision"] == "0,1"
    assert v2v.model_kwargs["condition_video_keep"] == "last"

    with pytest.raises(ValueError, match="not both"):
        adapter.video_to_request(
            VideoGenerationRequest(
                prompt="x", image="data:image/png;base64,AAAA", video="data:video/mp4;base64,AAAA",
            ),
            upload_dir=str(tmp_path),
        )


def test_gen_params_v2v() -> None:
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)

    # Defaults follow the reference V2V recipe: pin latent frames (0, 1) from
    # the start of the input video and denoise with flow_shift 10.
    p = model._resolve_gen_params({"num_frames": 17}, ["video", "text"], ["video"])
    assert p["has_video_condition"] is True
    assert p["condition_frame_indexes_vision"] == (0, 1)
    assert p["condition_video_keep"] == "first"
    assert p["flow_shift"] == 10.0

    # Explicit values win: comma-string indexes normalize sorted + deduped.
    p = model._resolve_gen_params(
        {"num_frames": 17, "condition_frame_indexes_vision": "2, 0, 2",
         "condition_video_keep": "LAST", "flow_shift": 4.0},
        ["video", "text"], ["video"],
    )
    assert p["condition_frame_indexes_vision"] == (0, 2)
    assert p["condition_video_keep"] == "last"
    assert p["flow_shift"] == 4.0

    # 17 frames -> 5 latent frames, so index 5 is out of range.
    with pytest.raises(ValueError, match="outside the latent"):
        model._resolve_gen_params(
            {"num_frames": 17, "condition_frame_indexes_vision": [0, 5]},
            ["video", "text"], ["video"],
        )
    with pytest.raises(ValueError, match="num_frames > 1"):
        model._resolve_gen_params({"num_frames": 1}, ["video", "text"], ["video"])
    with pytest.raises(ValueError, match="'first' or 'last'"):
        model._resolve_gen_params(
            {"num_frames": 17, "condition_video_keep": "middle"},
            ["video", "text"], ["video"],
        )

    # Action requests own their video input; no V2V params are attached.
    p = model._resolve_gen_params(
        {"action_mode": "inverse_dynamics", "action_chunk_size": 8, "raw_action_dim": 7},
        ["video", "text"], ["action"],
    )
    assert "has_video_condition" not in p


def test_gen_params_max_sequence_length() -> None:
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    assert model._resolve_gen_params({}, ["text"], ["image"])["max_sequence_length"] == 4096
    assert model._resolve_gen_params(
        {"max_sequence_length": 128}, ["text"], ["image"]
    )["max_sequence_length"] == 128
    # Floor of 1: a bad value can't empty the prompt.
    assert model._resolve_gen_params(
        {"max_sequence_length": 0}, ["text"], ["image"]
    )["max_sequence_length"] == 1


@pytest.mark.skipif(not NANO_DIR.exists(), reason="set COSMOS3_NANO_DIR to a Cosmos3-Nano dir")
def test_tokenize_prompt_truncation() -> None:
    from mstar.model.cosmos3.components.packing import tokenize_prompt

    model = Cosmos3Model(model_path_hf=str(NANO_DIR))
    tok = model.tokenizer
    eos = tok.eos_token_id
    sog = tok.convert_tokens_to_ids("<|vision_start|>")
    long_prompt = "a red cube on a table " * 200

    full, _ = tokenize_prompt(tok, long_prompt, "", num_frames=1, height=256, width=256)
    capped, _ = tokenize_prompt(
        tok, long_prompt, "", num_frames=1, height=256, width=256, max_sequence_length=64
    )
    # Truncate BEFORE the two trailing markers: 64 prompt tokens + eos + sog,
    # and the kept prefix is exactly the untruncated head.
    assert len(capped) == 66
    assert capped[-2:] == [eos, sog]
    assert capped[:64] == full[:64]

    # An under-cap prompt is bit-identical with and without the cap.
    short_default, _ = tokenize_prompt(tok, "a red cube", "", num_frames=1, height=256, width=256)
    short_capped, _ = tokenize_prompt(
        tok, "a red cube", "", num_frames=1, height=256, width=256, max_sequence_length=4096
    )
    assert short_default == short_capped


def test_create_images_n_expands_requests() -> None:
    import asyncio
    import base64

    from mstar.api_server.openai.adapters import get_adapter
    from mstar.api_server.openai.protocol import ImageGenerationRequest
    from mstar.api_server.openai.serving_images import create_images
    from mstar.api_server.request_types import ResultChunk  # noqa: I001

    class _Api:
        upload_dir = "/tmp"

        def __init__(self):
            self.submits = []

        def submit_request(self, **kw):
            self.submits.append(kw)
            return kw["request_id"]

        def collect_results(self, request_id):
            return [ResultChunk(request_id=request_id, modality="image", data=request_id.encode())]

    adapter = get_adapter("cosmos3")

    # n=3 with a seed: the reference contract gives image i seed + i, so
    # image 0 bit-matches the same request with n=1; results keep seed order.
    api = _Api()
    out = asyncio.run(
        create_images(api, "cosmos3", adapter, ImageGenerationRequest(prompt="x", n=3, seed=7))
    )
    assert [s["model_kwargs"]["seed"] for s in api.submits] == [7, 8, 9]
    ids = [s["request_id"] for s in api.submits]
    assert [base64.b64decode(d["b64_json"]).decode() for d in out["data"]] == ids

    # n=1 (default): exactly one submit, kwargs untouched.
    api = _Api()
    asyncio.run(create_images(api, "cosmos3", adapter, ImageGenerationRequest(prompt="x", seed=7)))
    assert len(api.submits) == 1 and api.submits[0]["model_kwargs"]["seed"] == 7

    # Unseeded n=2: independent per-request seeds, none injected here.
    api = _Api()
    asyncio.run(create_images(api, "cosmos3", adapter, ImageGenerationRequest(prompt="x", n=2)))
    assert len(api.submits) == 2
    assert all("seed" not in s["model_kwargs"] for s in api.submits)


def test_normalize_condition_frame_indexes() -> None:
    from mstar.model.cosmos3.components.packing import normalize_condition_frame_indexes

    default = (0, 1)
    assert normalize_condition_frame_indexes(None, default) == (0, 1)
    assert normalize_condition_frame_indexes(3, default) == (3,)
    assert normalize_condition_frame_indexes("2,0, 2", default) == (0, 2)
    assert normalize_condition_frame_indexes([1, 1, 0], default) == (0, 1)
    for bad in ("", [], [-1], "a,b", object()):
        with pytest.raises(ValueError):
            normalize_condition_frame_indexes(bad, default)


def test_static_inputs_noisy_frames_subset() -> None:
    import torch

    from mstar.model.cosmos3.components.packing import build_static_inputs

    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    cfg = model.config
    latent_shape = (1, cfg.latent_channel, 5, 16, 16)
    tcf = cfg.vae.scale_factor_temporal
    ids = list(range(12))
    p = cfg.latent_patch_size
    stride = -(-16 // p) * -(-16 // p)

    # Pinning latent frames {0, 2} predicts the complement {1, 3, 4}: the noisy
    # token count shrinks and the mse indexes cover exactly those frame blocks.
    noisy = [1, 3, 4]
    st = build_static_inputs(ids, latent_shape, cfg, tcf, 16.0, "cpu", noisy_frames=noisy)
    assert st["num_vision_tokens"] == 5 * stride
    assert st["num_noisy_vision_tokens"] == 3 * stride
    assert st["vision_noisy_frame_indexes"][0].tolist() == noisy
    expected = [
        st["und_len"] + f * stride + i for f in noisy for i in range(stride)
    ]
    assert st["vision_mse_loss_indexes"].tolist() == expected

    # noisy_frames covering all-but-frame-0 reproduces the i2v layout exactly.
    st_i2v = build_static_inputs(ids, latent_shape, cfg, tcf, 16.0, "cpu", has_image_condition=True)
    st_eq = build_static_inputs(ids, latent_shape, cfg, tcf, 16.0, "cpu", noisy_frames=[1, 2, 3, 4])
    assert torch.equal(st_i2v["vision_mse_loss_indexes"], st_eq["vision_mse_loss_indexes"])
    assert torch.equal(st_i2v["vision_noisy_frame_indexes"][0], st_eq["vision_noisy_frame_indexes"][0])
    assert st_i2v["num_noisy_vision_tokens"] == st_eq["num_noisy_vision_tokens"]


def test_vae_encoder_walk_and_signal_wiring() -> None:
    """The conditioned prefill walks run the vae_encoder node in parallel with
    the DiT prefill and persist ``cond_latents`` to the conductor; the media
    input routes to the encoder; the gen transition threads the persisted
    latents (or an empty edge) into the loop's external ``cond_latents`` input."""
    from mstar.graph.base import Parallel, TensorPointerInfo
    from mstar.graph.special_destinations import EMPTY_DESTINATION
    from mstar.model.cosmos3.cosmos3_model import DIT_NODE, VAE_ENCODER_NODE

    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    walks = model.get_graph_walk_graphs()

    for walk, cond_input in (("prefill_cond", "image_inputs"), ("prefill_cond_video", "video_inputs")):
        g = walks[walk]
        assert isinstance(g, Parallel)
        enc = g.get_nodes()[VAE_ENCODER_NODE]
        assert enc.input_names == {cond_input}
        (out,) = enc.outputs
        assert (out.name, out.next_node, out.persist) == ("cond_latents", EMPTY_DESTINATION, True)

    # Every gen loop takes cond_latents as an external (non-loop-back) input.
    for walk in ("image_gen", "video_gen", "action_gen", "action_video_gen"):
        (loop,) = walks[walk].get_loops().values()
        assert ("cond_latents", DIT_NODE) in loop._external_inputs, walk

    # The conditioning media edge targets the encoder node.
    fpa = model.get_initial_forward_pass_args(
        "p0", ["image", "text"], ["video"],
        {"text_inputs": [], "image_inputs": [TensorPointerInfo(
            dims=[3, 8, 8], dtype="float32", nbytes=768, address=0, stride=[64, 8, 1],
            uuid="u-img", source_session_id="s", source_entity="api_server")]},
    )
    targets = {(e.name, e.next_node) for e in fpa.inputs}
    assert ("image_inputs", VAE_ENCODER_NODE) in targets

    # Prefill -> gen: the persisted latents ride the cond_latents edge and are
    # unpersisted with this pass; without a persist signal the edge is empty.
    info = TensorPointerInfo(
        dims=[1], dtype="bfloat16", nbytes=2, address=0, stride=[1],
        uuid="u-lat", source_session_id="s", source_entity="worker",
    )
    fpa2 = model.get_partition_forward_pass_args(
        "default", fpa.full_metadata, {"cond_latents": [info]},
    )
    cond_edges = [e for e in fpa2.inputs if e.name == "cond_latents"]
    assert len(cond_edges) == 1 and cond_edges[0].tensor_info == [info]
    assert info in fpa2.unpersist_tensors
    fpa3 = model.get_initial_forward_pass_args("p1", ["text"], ["image"], {"text_inputs": []})
    fpa4 = model.get_partition_forward_pass_args("default", fpa3.full_metadata, {})
    (empty_edge,) = [e for e in fpa4.inputs if e.name == "cond_latents"]
    assert empty_edge.tensor_info == []


def test_ingest_cond_latents_routing() -> None:
    """Gen-init state adoption: i2v anchors land under ``cond_latents``; masked
    (video/action) conditioning lands under ``cond_video_latents``; an action
    request without media pins zeros; V2V without media is an error."""
    import types

    import torch

    from mstar.model.cosmos3.submodules import Cosmos3DiTSubmodule

    dit = Cosmos3DiTSubmodule(transformer=None, config=Cosmos3Model(
        model_path_hf="unused", skip_weight_loading=True).config, scheduler=None)
    dit.transformer = types.SimpleNamespace(
        proj_in=types.SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16))
    )
    anchor = torch.ones(1, 16, 1, 2, 2)

    st = dit.request_state("i2v")
    dit._ingest_cond_latents(st, {"cond_latents": [anchor]}, "cpu")
    assert torch.equal(st["cond_latents"], anchor) and st.get("cond_video_latents") is None

    st = dit.request_state("t2v")
    dit._ingest_cond_latents(st, {"cond_latents": []}, "cpu")
    assert st.get("cond_latents") is None

    st = dit.request_state("v2v")
    st.add("vmask", torch.ones(1, 1, 1, 1, 1))
    dit._ingest_cond_latents(st, {"cond_latents": [anchor]}, "cpu")
    assert torch.equal(st["cond_video_latents"], anchor)

    st = dit.request_state("act")
    st.add_all(vmask=torch.ones(1, 1, 1, 1, 1), action_chunk=4, latent_shape=(1, 16, 1, 2, 2))
    dit._ingest_cond_latents(st, {"cond_latents": []}, "cpu")
    assert st["cond_video_latents"].abs().sum() == 0
    assert st["cond_video_latents"].dtype == torch.bfloat16

    st = dit.request_state("v2v_missing")
    st.add("vmask", torch.ones(1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="no conditioning video"):
        dit._ingest_cond_latents(st, {}, "cpu")


if __name__ == "__main__":
    test_adapter_registered_for_images()
    test_gen_params_and_step_metadata()
    test_gen_params_v2v()
    test_normalize_condition_frame_indexes()
    test_static_inputs_noisy_frames_subset()
    test_vae_encoder_walk_and_signal_wiring()
    test_ingest_cond_latents_routing()
    if NANO_DIR.exists():
        test_process_prompt_emits_cond_and_uncond()
    print("PASS")
