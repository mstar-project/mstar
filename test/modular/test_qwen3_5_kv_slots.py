"""Qwen3.5 says how many slots its prompt takes in ``kv_cache``.

The pool admits a request only when its pages fit, and it can count only
what the request declares. An image's prefill writes its merged tokens plus
the two sentinels `split_around_spans` dropped from the text, and no tensor's
length says so. Declaring less lets the pool admit a request whose own
prefill then finds no page.

The LLM's own `prepare_inputs` and `declare_step` run for each scheduled walk,
so the count is checked against the spans the walks declare rather than a
formula. Needs the Qwen/Qwen3.5-0.8B tokenizer and preprocessor in the HF
cache, not the weights.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, ".")

import pytest
import torch
from api_uploads import png, preprocess
from huggingface_hub import try_to_load_from_cache
from torch import nn

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.qwen3_5.config import KV_CACHE, ROPE

REPO = "Qwen/Qwen3.5-0.8B"


class _StubLM(nn.Module):
    """Only the embedding the prefill walks read."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, 1)


class _StubRope:
    """Every request starts at position 0."""

    def position(self, rid, label):
        return 0


@pytest.fixture(scope="module")
def model():
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    for name in ("config.json", "tokenizer.json", "preprocessor_config.json"):
        path = try_to_load_from_cache(REPO, name)
        if not isinstance(path, str):
            pytest.skip(f"{REPO} {name} is not in the HF cache")
    return Qwen3_5DenseModel(os.path.dirname(path))


@pytest.fixture(scope="module")
def llm(model):
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    llm = LLMSubmodule(_StubLM(model.config.vocab_size), model.config, model.vision_config)
    llm.node_resources = {ROPE: _StubRope()}
    return llm


def _written(model, llm, modalities: list[str], tensors) -> int:
    """The slots the LLM's prefill walks declare for this prompt."""
    written = 0
    for step in model._prefill_schedule(modalities, tensors):
        inputs = {
            edge.name: edge.tensor_info
            for edge in model._walk_inputs(step) if edge.next_node == "LLM"
        }
        # the encoder's output, at least as long as any grid's merged tokens;
        # the walk cuts each image's share off it
        patches = sum(int(g.prod()) for g in inputs.get("image_grid_thw", []))
        inputs["vision_embeds"] = [torch.zeros(patches, 1)]
        fwd_info = CurrentForwardPassInfo(
            request_id="r", graph_walk=step.walk, fwd_index=0, random_seed=0,
            max_tokens=1, step_metadata={"prefill_order": step.order},
        )
        inp = llm.prepare_inputs(step.walk, fwd_info, inputs)
        written += sum(s.span for s in llm.declare_step(step.walk, ["r"], [inp]).segments)
    return written


@pytest.mark.parametrize(("prompt", "images"), [
    ("Describe a cat.", []),
    ("Compare these.", [(224, 224), (480, 640)]),
    # an OpenAI chat whose only part is an image arrives with no prompt
    (None, [(224, 224)]),
], ids=["text", "two_images", "image_no_text"])
def test_the_prompt_declares_what_its_prefill_walks_write(model, llm, prompt, images, tmp_path):
    files = [png(height, width) for height, width in images]
    tensors, kwargs = preprocess(model, None, prompt, files, ["text"], {}, tmp_path)

    modalities = ["image"] * len(images) + ["text"] * bool(prompt)
    assert (kwargs["prompt_slots"], kwargs["decode_labels"]) == (
        {KV_CACHE: {"main": _written(model, llm, modalities, tensors)}}, {KV_CACHE: ["main"]},
    ), "a prompt was declared at other than its text spans and each image's merged grid and two sentinels"
