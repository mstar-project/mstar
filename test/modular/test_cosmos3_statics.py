"""Cosmos3's DiT takes a request's statics from one memo, keyed by what builds them.

A denoise step reads the generation band's rotary table, its token layout and its clean/noisy masks. They depend
on the request's shape and its prompt's length, not on the prompt's tokens, and a label's come to ~6.4 MB at
480x832 and 121 frames. Built per request, they lived in the request's state, which the state contract leaves no
room for. From the memo two requests of one shape share them.

Every step rebuilds its settings and looks the statics up again. The memo once built a value past an eighth of its
budget per call and kept it nowhere, so a 720x1280, 289-frame label (34 MB) was built again on every step.

The prefill builds them, as it did when they lived in the request's state. Left to the first denoise step, nine
720x1280 requests' denoise steps took 110 ms each to prepare, where they had taken 0.8 ms.
"""

from types import SimpleNamespace

import torch
from diffusers import UniPCMultistepScheduler

from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.constants import PROMPT_LENGTHS
from mstar.model.cosmos3.submodules import Cosmos3DiTSubmodule, _nbytes


def _dit() -> Cosmos3DiTSubmodule:
    return Cosmos3DiTSubmodule(transformer=None, config=Cosmos3Config(compile_denoise=False), scheduler=None)


class _CountingDiT(Cosmos3DiTSubmodule):
    """Counts the statics it builds."""

    builds = 0

    def _build_static(self, *args, **kwargs):
        self.builds += 1
        return super()._build_static(*args, **kwargs)


def test_prompts_of_one_length_share_their_statics():
    dit = _dit()
    a = dit._static(list(range(7)), 64, 64, 9, 24.0, False, "cpu")
    b = dit._static(list(range(100, 107)), 64, 64, 9, 24.0, False, "cpu")
    assert a is b, "two requests of one shape would each build and hold their statics"
    assert dit._static(list(range(8)), 64, 64, 9, 24.0, False, "cpu")["und_len"] == 8


def test_the_prefill_reads_each_prompt_s_own_tokens():
    inputs = _dit()._get_prefill_node_inputs([5, 6, 7], [1, 2, 3], "cpu")
    assert torch.equal(inputs.tensor_inputs["input_ids_main"], torch.tensor([5, 6, 7]))
    assert torch.equal(inputs.tensor_inputs["input_ids_uncond"], torch.tensor([1, 2, 3])), (
        "a shared static would hand the prefill another request's prompt"
    )


def test_a_static_carries_the_layout_its_noisy_frames_give():
    static = _dit()._static(list(range(7)), 64, 64, 9, 24.0, True, "cpu")
    assert torch.equal(static["noisy_frame_mask"].flatten(), torch.tensor([0.0, 1.0, 1.0]))
    assert int(static["noisy_token_mask"].sum()) == static["num_noisy_vision_tokens"], (
        "a captured step would move the image anchor's tokens"
    )


def test_a_request_s_steps_build_its_statics_once(monkeypatch):
    label = _nbytes(_dit()._static(list(range(7)), 64, 64, 9, 24.0, False, "cpu"))
    # both labels fit, and each is past an eighth of the budget
    monkeypatch.setattr(_CountingDiT, "STATICS_BYTES", 3 * label)
    dit = _CountingDiT(
        transformer=None, config=Cosmos3Config(compile_denoise=False), scheduler=UniPCMultistepScheduler(),
    )
    md = {"height": 64, "width": 64, "num_frames": 9, "num_inference_steps": 4, PROMPT_LENGTHS: (7, 5)}
    for step in range(4):
        dit._settings(md, step, {}, "cpu")
    assert dit.builds == 2, "every step would build the request's statics again"


def test_a_request_s_denoise_steps_find_the_statics_its_prefill_built():
    dit = _CountingDiT(
        transformer=None, config=Cosmos3Config(compile_denoise=False), scheduler=UniPCMultistepScheduler(),
    )
    md = {"height": 64, "width": 64, "num_frames": 9, "num_inference_steps": 4, PROMPT_LENGTHS: (7, 5)}
    prompts = {"text_inputs": [torch.arange(7), torch.arange(5)]}
    dit._prepare_prefill(SimpleNamespace(step_metadata=md), prompts, "cpu")
    built = dit.builds
    for step in range(4):
        dit._settings(md, step, {}, "cpu")
    assert (built, dit.builds) == (2, 2), "the first denoise step would build the request's statics"


def test_a_pin_mask_is_built_once_per_layout():
    dit = _dit()
    mask = dit._pin_mask(4, range(2), "cpu", torch.float32)
    assert torch.equal(mask.flatten(), torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert dit._pin_mask(4, (0, 1), "cpu", torch.float32) is mask, "each window would build its own pin mask"
