"""The Thinker's layer loop with the residual adds deferred into the norms (what
AMD GPUs run) computes what the plain loop computes. On CPU the deferred add and
norm are an add then a norm, so the outputs match the plain loop bit for bit,
except where deepstack features join the stream, which reorders one sum.
Attention is a stub; the MoE, norms and router are the real ones."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch
from torch import nn

from mstar.model.qwen3_omni.components.thinker import Qwen3OmniThinkerModel
from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig

LAYERS = 5
HIDDEN = 64


class _Cursor:
    def bind_step(self, label):
        pass

    def set_layer_idx(self, idx):
        pass


class _Attention(nn.Module):
    """A token-wise linear map in place of attention, which needs the engine's resources."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.attend = _Cursor()

    def forward(self, hidden_states, cos_sin_3d=None, mrope_section=None):
        return self.proj(hidden_states)


def _model():
    cfg = Qwen3OmniModelConfig()
    tc = cfg.thinker_text
    tc.vocab_size, tc.hidden_size, tc.num_hidden_layers = 32, HIDDEN, LAYERS
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 2, 1, 32
    tc.num_experts, tc.num_experts_per_tok, tc.moe_intermediate_size = 8, 2, 16
    cfg.talker.accept_hidden_layer = 3
    torch.manual_seed(0)
    model = Qwen3OmniThinkerModel(cfg)
    for layer in model.model.layers:
        layer.self_attn = _Attention()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
                p.uniform_(0.5, 1.5)
            else:
                p.normal_(0, 0.2)
    return model.eval()


def _run(model, fused, x, deepstack=None):
    model.fused_residual = fused
    with torch.no_grad():
        return model(input_embeds=x.clone(), deepstack_visual_embeds=deepstack)


def test_the_fused_loop_is_the_plain_loop_on_cpu():
    model = _model()
    x = torch.randn(6, HIDDEN)
    plain = _run(model, False, x)
    fused = _run(model, True, x)
    assert model.accept_hidden_layer == 3 and plain[2] is not None
    for a, b in zip(plain, fused, strict=True):
        assert torch.equal(a, b)


def test_deepstack_features_reach_the_stream():
    model = _model()
    x = torch.randn(6, HIDDEN)
    deepstack = [torch.randn(6, HIDDEN) for _ in range(2)]
    plain = _run(model, False, x, deepstack)
    fused = _run(model, True, x, deepstack)
    without = _run(model, True, x)
    for a, b in zip(plain, fused, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(fused[0], without[0])


@pytest.mark.parametrize("rocm", [True, False])
def test_only_amd_gpus_take_the_fused_loop(monkeypatch, rocm):
    from mstar.model.qwen3_omni.components import thinker

    class _Platform:
        def is_rocm(self):
            return rocm

    resolved = []

    class _Dispatch:
        def resolve(self):
            resolved.append(True)

    monkeypatch.setattr(thinker, "current_platform", _Platform())
    monkeypatch.setattr(thinker, "run_fused_add_rms_norm", _Dispatch())
    cfg = Qwen3OmniModelConfig()
    cfg.thinker_text.num_hidden_layers = 1
    with torch.device("meta"):
        model = Qwen3OmniThinkerModel(cfg)
    assert model.fused_residual is rocm
    # the add+norm kernel is picked at build time, outside any compiled forward
    assert resolved == ([True] if rocm else [])
