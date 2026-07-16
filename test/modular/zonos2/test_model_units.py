"""Unit tests for Zonos2 model pieces that don't need a checkpoint or GPU.

Two of these guard fixes on this branch:
  * the MoE router selects ``get_num_experts_per_tok(layer)`` experts, so the
    ``special_topk_layers`` override actually reaches routing;
  * speaker conditioning is injected at exactly the speaker token position(s).

``Zonos2Router.forward`` and the model's ``out_norm`` route through mstar's
flashinfer RMSNorm (CUDA-only), so the router test checks ``top_k`` directly
(the resolved value *is* the routed count) and the speaker test swaps
``out_norm`` for ``Identity`` — irrelevant to the injection, and row-wise
anyway, so per-position isolation is preserved.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mstar.model.zonos2.config import Zonos2Config
from mstar.model.zonos2.components.language_model import (
    Zonos2ForCausalLM,
    Zonos2Router,
    softcap,
)


# -- softcap ----------------------------------------------------------------
def test_softcap_bounds_and_monotonic():
    cap = 15.0
    # Moderate range so tanh stays unsaturated (strictly monotonic in float32).
    x = torch.linspace(-30, 30, 301)
    y = softcap(x, cap)
    assert (y.abs() < cap).all()                       # strictly bounded
    assert (y.diff() > 0).all()                        # monotonic increasing
    assert torch.allclose(softcap(torch.zeros(1), cap), torch.zeros(1))
    # Extreme inputs saturate at (never beyond) the cap.
    assert softcap(torch.tensor(1e4), cap).item() <= cap
    assert softcap(torch.tensor(-1e4), cap).item() >= -cap


# -- MoE router top-k per layer --------------------------------------------
def _moe_cfg(**kw) -> Zonos2Config:
    base = dict(
        hidden_size=16, moe_n_experts=4, num_experts_per_tok=1,
        moe_router_dim=8, moe_start_from_layer=0,
    )
    base.update(kw)
    return Zonos2Config(**base)


def test_router_topk_honors_special_layers():
    # ``top_k`` is the count the router feeds to ``torch.topk`` when selecting
    # experts, so this pins the special-topk override to routing without
    # running forward (which routes through the CUDA-only RMSNorm).
    cfg = _moe_cfg(num_experts_per_tok=1, special_topk_layers={2: 3})
    assert Zonos2Router(cfg, layer_id=0).top_k == 1   # global default
    assert Zonos2Router(cfg, layer_id=2).top_k == 3   # per-layer override
    # String keys (checkpoint JSON form) resolve the same way.
    cfg_str = _moe_cfg(num_experts_per_tok=1, special_topk_layers={"2": 3})
    assert Zonos2Router(cfg_str, layer_id=2).top_k == 3


def test_router_topk_is_static_int():
    # CUDA-graph safety: ``top_k`` must be a plain int fixed at construction,
    # not a tensor / callable / per-token value. A future config change that
    # made routing count data-dependent would silently reintroduce dynamic
    # shapes into the (to-be-captured) MoE dispatch — this pins it.
    router = Zonos2Router(_moe_cfg(num_experts_per_tok=2), layer_id=0)
    assert isinstance(router.top_k, int)
    assert router.top_k == 2


# -- speaker conditioning injection ----------------------------------------
def _speaker_model(**kw) -> Zonos2ForCausalLM:
    cfg = Zonos2Config(
        num_layers=0, hidden_size=16, n_codebooks=3, codebook_size=8,
        text_vocab=10, moe_n_experts=1, **kw,
    )
    model = Zonos2ForCausalLM(cfg).eval()
    model.out_norm = nn.Identity()  # CUDA-only RMSNorm; irrelevant to injection
    # mstar's parallel embedding/linear layers leave weights uninitialized
    # (populated only at checkpoint load), so give them finite, deterministic
    # values before exercising the forward.
    torch.manual_seed(0)
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)
    return model


def _stub_cache() -> SimpleNamespace:
    return SimpleNamespace(advance_seq_lens=lambda: None, set_layer_idx=lambda i: None)


def _ids(model: Zonos2ForCausalLM, T: int = 4) -> torch.Tensor:
    return torch.zeros(T, model.n_codebooks + 1, dtype=torch.long)


def test_speaker_projection_shapes():
    with_lda = _speaker_model(speaker_enabled=True, speaker_embedding_dim=5, speaker_lda_dim=4)
    assert with_lda.speaker_lda_projection.in_features == 5
    assert with_lda.speaker_lda_projection.out_features == 4
    assert with_lda.speaker_projection.in_features == 4        # fed by the LDA output
    assert with_lda.speaker_projection.out_features == 16      # hidden_size

    no_lda = _speaker_model(speaker_enabled=True, speaker_embedding_dim=5, speaker_lda_dim=None)
    assert no_lda.speaker_lda_projection is None
    assert no_lda.speaker_projection.in_features == 5          # raw embedding


def test_speaker_injection_only_at_positions():
    model = _speaker_model(speaker_enabled=True, speaker_embedding_dim=5, speaker_lda_dim=4)
    ids, cache = _ids(model, T=4), _stub_cache()
    pos = 1
    with torch.no_grad():
        base = model(ids, _stub_cache())
        spk = model(
            ids, cache,
            speaker_emb_values=torch.randn(1, 5),
            speaker_token_positions=torch.tensor([pos]),
        )
    assert not torch.allclose(base[pos], spk[pos])            # injected row changed
    for i in range(base.shape[0]):
        if i != pos:
            assert torch.allclose(base[i], spk[i])            # every other row intact


def test_speaker_disabled_ignores_values():
    model = _speaker_model(speaker_enabled=False)
    assert model.speaker_projection is None
    ids = _ids(model)
    with torch.no_grad():
        base = model(ids, _stub_cache())
        # Supplying values is a harmless no-op when the model is speaker-disabled.
        out = model(
            ids, _stub_cache(),
            speaker_emb_values=torch.randn(1, 5),
            speaker_token_positions=torch.tensor([0]),
        )
    assert torch.allclose(base, out)
