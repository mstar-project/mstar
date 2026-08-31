"""Platform use case: safely co-schedule visual and text chat requests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.kv_store import PositionInfo
from mstar.model.qwenvl.components import QwenVLForCausalLM
from mstar.model.qwenvl.submodules import (
    QwenVLLLMSubmodule,
    QwenVLVisionSubmodule,
    qwen_vl_position_ids,
)

from ._helpers import Cache, FixedLanguageModel, tiny_config


def test_visual_chat_uses_post_merge_features_and_preserves_request_context():
    pooled = torch.ones(2, 16)

    class Vision(torch.nn.Module):
        def forward(self, pixels, grid_thw):
            self.grid = grid_thw
            return pooled, [pooled, pooled, pooled]

    vision = Vision()
    submodule = QwenVLVisionSubmodule(vision)
    inputs = submodule.prepare_inputs(
        "prefill_vision",
        None,
        {
            "pixel_values": [torch.ones(4, 12)],
            "image_grid_thw": [torch.tensor([[1, 2, 2]])],
            "text_inputs": [torch.tensor([1, 2])],
            "position_ids": [torch.tensor([[0, 1], [0, 1], [0, 1]])],
        },
    )
    output = submodule.forward("prefill_vision", None, **inputs.tensor_inputs)
    assert torch.equal(vision.grid, inputs.tensor_inputs["image_grid_thw"])
    assert torch.equal(output["vision_embeds"][0], pooled)
    assert len(output["deepstack_visual_embeds"]) == 3


def test_image_prefill_preserves_mrope_position_for_follow_up_decode():
    config = tiny_config()
    submodule = QwenVLLLMSubmodule(QwenVLForCausalLM(config), config)
    cache = Cache()
    engine = SimpleNamespace(cache_manager=cache, request_ids=["a", "b"], sampler=None)
    first = submodule.prepare_inputs(
        "prefill", None, {"text_inputs": [torch.tensor([1, 2, 3])], "position_ids": [torch.arange(3).expand(3, -1)]}
    )
    ids = torch.tensor(
        [4, config.image_token_id, config.image_token_id, config.image_token_id, config.image_token_id, 5]
    )
    image = submodule.prepare_inputs(
        "prefill_vision",
        None,
        {
            "text_inputs": [ids],
            "position_ids": [qwen_vl_position_ids(ids, torch.tensor([[1, 4, 4]]), config)],
            "vision_embeds": [torch.ones(4, 16)],
            "deepstack_visual_embeds": [torch.ones(4, 16) for _ in range(3)],
        },
    )
    packed = submodule.preprocess("prefill", engine, [first, image])
    assert packed["seq_lens"] == [3, 6]
    assert packed["position_advance"] == [3, 4]
    assert cache.custom_pos_advance == [3, 4]
    decode = submodule.prepare_inputs(
        "decode", None, {"text_inputs": [torch.tensor([7])]}, pos_info={"main": PositionInfo(position_id_start=4)}
    )
    assert decode.custom_pos_ids.tolist() == [[4], [4], [4]]
    with pytest.raises(ValueError, match="Unknown"):
        submodule.prepare_inputs("unknown", None, {"text_inputs": [torch.tensor([7])]})


def test_concurrent_visual_and_text_chats_keep_tenant_outputs_isolated():
    config = tiny_config(image_token_id=30)
    model = FixedLanguageModel(config)
    submodule = QwenVLLLMSubmodule(model, config)
    cache = Cache()
    engine = SimpleNamespace(cache_manager=cache, request_ids=["text", "image"], sampler=None)
    text = submodule.prepare_inputs(
        "prefill", None, {"text_inputs": [torch.tensor([1, 2, 3])], "position_ids": [torch.arange(3).expand(3, -1)]}
    )
    ids = torch.tensor([4, 30, 30, 30, 30, 5])
    image = submodule.prepare_inputs(
        "prefill_vision",
        None,
        {
            "text_inputs": [ids],
            "position_ids": [qwen_vl_position_ids(ids, torch.tensor([[1, 4, 4]]), config)],
            "vision_embeds": [torch.full((4, 16), 3.0)],
            "deepstack_visual_embeds": [torch.full((4, 16), 0.5) for _ in range(3)],
        },
    )
    result = submodule.forward_batched("prefill", engine, **submodule.preprocess("prefill", engine, [text, image]))
    assert set(result) == {"text", "image"}
    assert torch.equal(model.calls[-1][0][4:8], torch.full((4, 16), 3.0))

    class Sampler:
        def sample(self, request_ids, logits, apply_penalty):
            assert (request_ids, logits.shape, apply_penalty) == (
                ["text", "image"],
                (2, config.text_config.vocab_size),
                True,
            )
            return torch.tensor([8, 9])

    engine.sampler = Sampler()
    sampled = submodule.forward_batched(
        "decode",
        engine,
        text_inputs=torch.tensor([1, 2]),
        position_ids=torch.tensor([[3, 4], [3, 4], [3, 4]]),
        position_advance=[1, 1],
        seq_lens=[1, 1],
        cos_3d=torch.ones(2, 8),
        sin_3d=torch.zeros(2, 8),
    )
    assert sampled.keys() == {"text", "image"}
    submodule.postprocess("text", None, sampled["text"])
    assert torch.equal(sampled["text"]["text_inputs"][0], torch.tensor([8]))


def test_decode_serving_honors_per_request_last_token_and_stop_conditions():
    config = tiny_config()
    submodule = QwenVLLLMSubmodule(FixedLanguageModel(config), config)
    engine = SimpleNamespace(cache_manager=Cache(), request_ids=["a", "b"], sampler=None)
    output = submodule.forward_batched(
        "decode",
        engine,
        text_inputs=torch.tensor([1, 2]),
        position_ids=torch.tensor([[0, 1], [0, 1], [0, 1]]),
        position_advance=[1, 1],
        cos_3d=torch.ones(2, 8),
        sin_3d=torch.zeros(2, 8),
    )
    assert set(output) == {"a", "b"}
    direct = submodule.forward(
        "prefill",
        engine,
        text_inputs=torch.tensor([1, 2]),
        position_ids=torch.tensor([[0, 1], [0, 1], [0, 1]]),
        position_advance=[1, 1],
        seq_lens=[1, 1],
        cos_3d=torch.ones(2, 8),
        sin_3d=torch.zeros(2, 8),
    )
    assert direct["logits"][0].shape == (2, config.text_config.vocab_size)
    assert submodule.forward(
        "decode",
        engine,
        text_inputs=torch.tensor([1]),
        position_ids=torch.zeros(3, 1, dtype=torch.long),
        position_advance=1,
        cos_3d=torch.ones(1, 8),
        sin_3d=torch.zeros(1, 8),
    )["logits"][0].shape == (1, config.text_config.vocab_size)
    request = SimpleNamespace(
        graph_walk="decode",
        sampling_config={"LLM": SimpleNamespace(ignore_eos=False)},
        dynamic_loop_iter_counts={"decode_loop": 0},
        max_tokens=3,
    )
    assert submodule.check_stop("a", request, {"new_token": [torch.tensor([config.text_config.eos_token_id])]}) == {
        "decode_loop"
    }
    assert submodule.check_stop("a", request, {}) == set()
    request.graph_walk = "prefill"
    assert submodule.check_stop(
        "a", request, {"new_token": [torch.tensor([config.text_config.eos_token_id])]}
    ) == set()
    assert submodule.can_batch(None, [])
    with pytest.raises(ValueError, match="placeholder count"):
        submodule._merge_embeddings(torch.tensor([1, 2]), torch.ones(1, 16))
