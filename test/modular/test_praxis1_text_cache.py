"""Bounded prompt embedding reuse in the Praxis text node."""

from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from mstar.model.praxis1.config import Praxis1Config
from mstar.model.praxis1.submodules import Praxis1TextEncoderSubmodule


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2, num_hidden_layers=4)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, *, input_ids, attention_mask, output_hidden_states, use_cache):
        assert output_hidden_states and not use_cache
        self.calls += 1
        values = input_ids.to(self.scale.dtype) * attention_mask * self.scale
        values = torch.stack((values, values + 1), dim=-1)
        return SimpleNamespace(hidden_states=tuple(values + i for i in range(5)))


def _node() -> Praxis1TextEncoderSubmodule:
    config = replace(Praxis1Config.for_rig("single_arm"), text_hidden_dim=2)
    return Praxis1TextEncoderSubmodule(config, _Encoder().eval())


def _call(node, task=(4, 5), task_mask=(1, 1)):
    return node.forward(
        "encode_text", None,
        text_ids=torch.tensor(task, dtype=torch.long),
        text_mask=torch.tensor(task_mask, dtype=torch.long),
        blank_ids=torch.tensor([9, 0], dtype=torch.long),
        blank_mask=torch.tensor([1, 0], dtype=torch.long),
    )


def test_repeated_exact_tokens_hit_and_other_prompt_or_mask_misses():
    node = _node()
    first = _call(node)
    assert node.encoder.calls == 2
    repeated = _call(node)
    assert node.encoder.calls == 2
    for name in ("text_embeds", "text_embeds_blank"):
        assert torch.equal(first[name][0], repeated[name][0])
        assert first[name][0].data_ptr() != repeated[name][0].data_ptr()

    changed = _call(node, task=(4, 6))
    assert node.encoder.calls == 3
    assert not torch.equal(first["text_embeds"][0], changed["text_embeds"][0])
    assert torch.equal(first["text_embeds_blank"][0], changed["text_embeds_blank"][0])

    shorter = _call(node, task_mask=(1, 0))
    assert node.encoder.calls == 4
    assert shorter["text_embeds"][0].shape[1] == 1
    assert first["text_embeds"][0].shape[1] == 2


def test_entry_and_byte_limits_are_enforced():
    node = _node()
    node.CACHE_MAX_ENTRIES = 2
    _call(node)
    _call(node, task=(4, 6))
    assert len(node._text_cache) == 2
    assert node._text_cache_bytes <= node.CACHE_MAX_BYTES
    _call(node)
    assert node.encoder.calls == 4  # first task was evicted; blank caption stayed

    node._clear_text_cache()
    node.CACHE_MAX_BYTES = 1
    _call(node)
    assert len(node._text_cache) == 0
    assert node._text_cache_bytes == 0


def test_cache_clears_on_weight_load_device_apply_and_training_mode():
    node = _node()
    first = _call(node)["text_embeds"][0]
    assert len(node._text_cache) == 2

    state = {name: value.clone() for name, value in node.state_dict().items()}
    state["encoder.scale"].fill_(2)
    node.load_state_dict(state)
    assert len(node._text_cache) == 0
    second = _call(node)["text_embeds"][0]
    assert node.encoder.calls == 4
    assert not torch.equal(first, second)

    parent = nn.Module()
    parent.add_module("text", node)
    parent_state = {name: value.clone() for name, value in parent.state_dict().items()}
    parent_state["text.encoder.scale"].fill_(3)
    parent.load_state_dict(parent_state)
    assert len(node._text_cache) == 0
    assert node.encoder.calls == 4
    _call(node)
    assert node.encoder.calls == 6

    node.to(dtype=torch.float64)
    assert len(node._text_cache) == 0
    third = _call(node)["text_embeds"][0]
    assert third.dtype == torch.float64
    assert node.encoder.calls == 8

    node.train()
    assert len(node._text_cache) == 0
    _call(node)
    _call(node)
    assert node.encoder.calls == 12  # training never reads or writes the cache


def test_output_mutation_cannot_poison_later_hits():
    node = _node()
    first = _call(node)
    expected = _call(node)["text_embeds_blank"][0].clone()
    with torch.inference_mode():
        first["text_embeds_blank"][0].zero_()
    later = _call(node)
    assert torch.equal(later["text_embeds_blank"][0], expected)
    assert node.encoder.calls == 2
