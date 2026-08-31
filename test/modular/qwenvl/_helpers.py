"""Shared fixtures and CPU references for Qwen3-VL-MoE tests."""

from __future__ import annotations

import importlib.machinery
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytest.importorskip("safetensors", reason="QwenVL tests require the qwenvl model extra")

# M* installs a CPU test shim for Triton; Transformers probes its module spec.
triton = sys.modules.get("triton")
if triton is not None and getattr(triton, "__spec__", None) is None:
    triton.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)

from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (  # noqa: E402
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)


def tiny_config(*, image_token_id: int = 151655) -> Qwen3VLMoeConfig:
    text = Qwen3VLMoeTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rope_theta=10_000.0,
        max_position_embeddings=64,
        num_experts=4,
        num_experts_per_tok=2,
        # Keep this different from hidden_size / 2 so a mistaken transpose
        # of the published fused-expert layout cannot pass by shape accident.
        moe_intermediate_size=6,
        norm_topk_prob=True,
        mlp_only_layers=[],
        decoder_sparse_step=1,
        bos_token_id=0,
        eos_token_id=31,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [1, 2, 1],
            "mrope_interleaved": True,
        },
    )
    vision = Qwen3VLMoeVisionConfig(
        out_hidden_size=16,
        spatial_merge_size=2,
        deepstack_visual_indexes=[0, 1, 2],
    )
    return Qwen3VLMoeConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=image_token_id,
        tie_word_embeddings=False,
    )


class Cache:
    def set_layer_idx(self, index):
        self.index = index

    def run_attention(self, q, k, v):
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert k.shape == v.shape
        return q.new_zeros(q.shape)

    def advance_seq_lens(self, pos_id_ns=None):
        self.advanced = True
        self.position_advance = pos_id_ns

    def set_active_label(self, label):
        self.label = label

    def plan_attention(self, seq_lens, is_causal=True, label="main"):
        self.seq_lens = seq_lens

    def set_custom_pos_advance(self, pos_advance, label=None):
        self.custom_pos_advance = list(pos_advance) if pos_advance is not None else None


class CausalCache(Cache):
    def run_attention(self, q, k, v):
        from torch.nn import functional

        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        output = functional.scaled_dot_product_attention(
            q.permute(1, 0, 2).unsqueeze(0),
            k.permute(1, 0, 2).unsqueeze(0),
            v.permute(1, 0, 2).unsqueeze(0),
            is_causal=True,
        )
        return output.squeeze(0).permute(1, 0, 2)


class FakeProcessor:
    def __init__(self, input_ids: torch.Tensor, grid: torch.Tensor | None = None):
        self.input_ids = input_ids
        self.grid = grid
        self.messages = None
        self.call_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "rendered prompt"

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        result = {"input_ids": self.input_ids.unsqueeze(0)}
        if self.grid is not None:
            result.update({"pixel_values": torch.ones(self.grid.prod().item(), 12), "image_grid_thw": self.grid})
        return result

    def decode(self, output):
        return f"decoded:{output.tolist()}"


class FixedLanguageModel(torch.nn.Module):
    """Minimal text model that records the submodule's composed call."""

    def __init__(self, config: Qwen3VLMoeConfig):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(config.text_config.vocab_size, config.text_config.hidden_size)
        self.lm_head = torch.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.calls = []

    def forward(self, embeddings, cache, position_ids, **kwargs):
        self.calls.append((embeddings.detach().clone(), position_ids.detach().clone(), kwargs))
        return embeddings


def install_fake_transformers(monkeypatch, processor=None, vision_class=None, config=None):
    processor = processor or FakeProcessor(torch.tensor([1, 2]))
    vision_class = vision_class or torch.nn.Identity
    config = config or tiny_config()
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor)
    transformers.AutoConfig = SimpleNamespace(from_pretrained=lambda *args, **kwargs: config)
    models = ModuleType("transformers.models")
    qwen = ModuleType("transformers.models.qwen3_vl_moe")
    modeling = ModuleType("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe")
    modeling.Qwen3VLMoeVisionModel = vision_class
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.models", models)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_vl_moe", qwen)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe", modeling)


def qwen_transformers_or_skip():
    """Import the optional Transformers oracle under Pytest's Triton shim."""
    triton = sys.modules.get("triton")
    if triton is not None and getattr(triton, "__spec__", None) is None:
        triton.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)
    try:
        from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
            Qwen3VLMoeConfig,
            Qwen3VLMoeTextConfig,
            Qwen3VLMoeVisionConfig,
        )
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeModel,
            Qwen3VLMoeTextModel,
            Qwen3VLMoeVisionModel,
        )
    except Exception as error:
        pytest.skip(f"QwenVL parity requires a healthy Transformers extra: {error}")
    return SimpleNamespace(
        config=Qwen3VLMoeConfig,
        text_config=Qwen3VLMoeTextConfig,
        vision_config=Qwen3VLMoeVisionConfig,
        vision_model=Qwen3VLMoeVisionModel,
        model=Qwen3VLMoeModel,
        text_model=Qwen3VLMoeTextModel,
    )


def patch_cpu_rms_norm(monkeypatch):
    import mstar.model.components.norm as norm_module

    def cpu_rms_norm(input, weight, eps):
        normalized = input.float() * torch.rsqrt(input.float().square().mean(-1, keepdim=True) + eps)
        return (normalized * weight.float()).to(input.dtype)

    monkeypatch.setattr(norm_module, "run_rms_norm", cpu_rms_norm)
