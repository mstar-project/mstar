"""Pi0.5 says how much its ``kv`` holds, because the prompt tensor does not.

The pool admits a request only when the pages it can still take fit. Pi0.5's
prompt tensor is its text alone: SigLIP turns each camera into a grid of tokens
on the worker, and every ``action_gen`` step writes ``action_horizon`` slots past
the prefix without committing them, which admit gives pages all the same.
Under-counting lets the pool admit a request whose own step then finds no page;
over-counting refuses one that runs today.
"""

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch
from api_uploads import png, preprocess
from torch import nn

from mstar.model.pi05.components.siglip import Pi05SiglipEncoder
from mstar.model.pi05.config import LLM_KV, Pi05Config
from mstar.model.pi05.pi05_model import Pi05Model
from mstar.model.pi05.submodules import Pi05LLMSubmodule, Pi05ViTEncoderSubmodule


class _StubTokenizer:
    """One id per character."""

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        return torch.tensor([ord(c) % 64 for c in prompt], dtype=torch.long)


def _config() -> Pi05Config:
    # a one-layer SigLIP at the real image and patch size: the grid is what counts
    return Pi05Config(
        vit_hidden_size=16, vit_num_layers=1, vit_num_heads=2, vit_intermediate_size=32, hidden_size=16,
    )


def _model() -> Pi05Model:
    model = object.__new__(Pi05Model)
    model.config = _config()
    model.tokenizer = _StubTokenizer()
    return model


def _llm(config: Pi05Config) -> Pi05LLMSubmodule:
    """The LLM node with only what its prepare and declare paths read."""
    llm = object.__new__(Pi05LLMSubmodule)
    nn.Module.__init__(llm)
    llm.config = config
    llm.embed_tokens = nn.Embedding(64, config.hidden_size)
    llm._image_embed_scale = llm._text_embed_scale = 1.0
    return llm


def _written(model: Pi05Model, tensors: dict) -> int:
    """The slots the prefill and action_gen walks put on ``main``, from the walks' own code."""
    vit = Pi05ViTEncoderSubmodule(Pi05SiglipEncoder(model.config), model.config)
    pixel_values = vit.prepare_inputs("prefill", None, tensors).tensor_inputs["pixel_values"]
    img_emb = vit.forward("prefill", None, pixel_values.unsqueeze(0))["img_emb"]
    llm = _llm(model.config)
    walks = {
        "prefill": llm.prepare_inputs("prefill", None, {"img_emb": img_emb, "text_inputs": tensors["text_inputs"]}),
        "action_gen": llm.prepare_inputs("action_gen", SimpleNamespace(random_seed=0), {}),
    }
    return sum(
        segment.span
        for walk, inp in walks.items()
        for segment in llm.declare_step(walk, ["r"], [inp]).segments
    )


def test_cameras_uploaded_one_file_each_reserve_what_the_walks_write(tmp_path):
    model = _model()
    cameras = [png(224, 224) for _ in range(3)]
    kwargs = {"robot_state": torch.linspace(-1.0, 1.0, 8).tolist()}
    tensors, declared = preprocess(model, None, "pick up the block", cameras, ["action"], kwargs, tmp_path)

    # the encoder takes every camera of a request as one stacked tensor
    stacked = dict(tensors, image_inputs=[torch.stack(tensors["image_inputs"])])
    assert (declared["prompt_slots"], declared["decode_labels"]) == (
        {LLM_KV: {"main": _written(model, stacked)}}, {LLM_KV: []},
    ), "a request's three uploaded cameras reserved other than their grids, the prompt and the action chunk"
