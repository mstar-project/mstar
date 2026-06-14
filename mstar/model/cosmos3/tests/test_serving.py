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


def test_video_adapter_t2v() -> None:
    import pytest as _pytest

    from mstar.api_server.openai.adapters import get_adapter
    from mstar.api_server.openai.protocol import VideoGenerationRequest

    adapter = get_adapter("cosmos3")
    assert adapter is not None and adapter.supports_videos

    # text-to-video: text-only input, video output, num_frames/fps threaded.
    req = VideoGenerationRequest(
        prompt="a kite", size="256x256", seed=1, num_frames=17, fps=16.0,
        guidance_scale=6.0,
    )
    args = adapter.video_to_request(req, upload_dir="/tmp")
    assert args.text == "a kite"
    assert args.input_modalities == ["text"]
    assert args.output_modalities == ["video"]
    assert args.file_paths is None
    assert args.model_kwargs["num_frames"] == 17
    assert args.model_kwargs["fps"] == 16.0
    assert args.model_kwargs["guidance_scale"] == 6.0

    # image-to-video is not wired yet; it must reject fast rather than silently
    # dropping the conditioning image.
    with _pytest.raises(NotImplementedError):
        adapter.video_to_request(
            VideoGenerationRequest(prompt="zoom in", image="data:image/png;base64,AAAA"),
            upload_dir="/tmp",
        )


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
        Cosmos3DiTSubmodule,
        IMAGE_GEN_LOOP,
        IMAGE_GEN_WALK,
    )

    dit = Cosmos3DiTSubmodule(transformer=None, config=Cosmos3Model(
        model_path_hf="unused", skip_weight_loading=True).config, scheduler=None)

    class _Sched:
        def __init__(self, n):
            self.timesteps = list(range(n))

    n = 4
    dit._req["r"] = {"scheduler": _Sched(n), "raw_action_dim": 2}

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
    out = dit._forward_image_gen(None, dit._req["r"], latents=lat, time_index=ti)
    assert torch.equal(out["latents"][0], lat) and torch.equal(out["time_index"][0], ti)

    act = torch.zeros(1, 3, 5)
    out = dit._forward_action_gen(
        None, dit._req["r"], latents=lat, action_latents=act, time_index=ti
    )
    assert torch.equal(out["latents"][0], lat)
    assert torch.equal(out["action_output"][0], act[:, :, :2])


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


if __name__ == "__main__":
    test_adapter_registered_for_images()
    test_gen_params_and_step_metadata()
    if NANO_DIR.exists():
        test_process_prompt_emits_cond_and_uncond()
    print("PASS")
