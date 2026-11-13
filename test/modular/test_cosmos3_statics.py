"""Cosmos3's DiT takes a request's statics from one memo, keyed by what builds them.

A denoise step reads the generation band's rotary table, its token layout and its clean/noisy masks. They depend
on the request's shape and its prompt's length, not on the prompt's tokens, and a label's come to ~6.4 MB at
480x832 and 121 frames. Built per request, they lived in the request's state, which the state contract leaves no
room for. From the memo two requests of one shape share them.
"""

import torch

from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.submodules import Cosmos3DiTSubmodule


def _dit() -> Cosmos3DiTSubmodule:
    return Cosmos3DiTSubmodule(transformer=None, config=Cosmos3Config(compile_denoise=False), scheduler=None)


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


def test_a_pin_mask_is_built_once_per_layout():
    dit = _dit()
    mask = dit._pin_mask(4, range(2), "cpu", torch.float32)
    assert torch.equal(mask.flatten(), torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert dit._pin_mask(4, (0, 1), "cpu", torch.float32) is mask, "each window would build its own pin mask"
