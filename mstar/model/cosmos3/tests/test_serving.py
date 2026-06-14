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


def test_gen_params_and_step_metadata() -> None:
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)

    # "size" parses to width/height; explicit width/height win; defaults applied.
    p = model._resolve_gen_params({"size": "480x256"}, ["text"], ["image"])
    assert (p["width"], p["height"]) == (480, 256)
    assert p["num_frames"] == 1 and p["has_image_condition"] is False

    # The denoise loop count is fixed at graph build, so a per-request
    # num_inference_steps must NOT change the resolved value (it would desync the
    # loop and the scheduler); guidance_scale, however, is honored per request.
    p = model._resolve_gen_params(
        {"num_inference_steps": 3, "guidance_scale": 2.5}, ["text"], ["image"]
    )
    assert p["num_inference_steps"] == model.config.num_inference_steps
    assert p["guidance_scale"] == 2.5

    # i2v conditioning is inferred from the input modalities.
    p = model._resolve_gen_params({}, ["image", "text"], ["image"])
    assert p["has_image_condition"] is True

    fpa = model.get_initial_forward_pass_args(
        "p0", ["text"], ["image"], {"text_inputs": []}, model_kwargs={"size": "256x256"}
    )
    sm = fpa.step_metadata
    assert sm["is_prefill"] is True
    assert sm["height"] == 256 and sm["width"] == 256
    assert sm["num_inference_steps"] == model.config.num_inference_steps


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
