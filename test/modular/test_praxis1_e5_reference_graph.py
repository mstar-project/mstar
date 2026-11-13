"""Real ported towers through the E5 graph nodes, against pinned Praxis CPU output."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("praxis", reason="the Praxis-1 reference is not on PYTHONPATH")
import praxis.layers as L
from praxis.model import SamplerSettings
from tests.tiny_model import tiny_policy

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.attn.ragged.expert_mask import cross_visibility, self_visibility
from mstar.engine.state import ExpertCross, ExpertCrossLayout, ExpertSelf, ExpertSelfLayout
from mstar.model.praxis1.components.chunk_mask import encoder_codes, visible
from mstar.model.praxis1.components.encoder import dense_chunk_causal_mask
from mstar.model.praxis1.components.sampler import ACTION_SALT, VIDEO_SALT, seeded_normal
from mstar.model.praxis1.native_towers import build_backbone, build_expert
from mstar.model.praxis1.submodules import (
    DECODER_BLANK_KV,
    DECODER_COND_KV,
    ENCODER_BLANK_KV,
    ENCODER_COND_KV,
    EXPERT_QUERIES,
    EXPERT_TRUNK_KEYS,
    Praxis1ActionExpertSubmodule,
    Praxis1BackboneSubmodule,
)
from mstar.model.praxis1.weight_loader import _local_key
from test.modular.test_praxis1_weight_loader import _tiny_config


def _eager_reference_flex(q, k, v, block_mask=None, scale=None, enable_gqa=False, kernel_options=None):
    """Evaluate the reference's own BlockMask predicate with CPU SDPA."""
    qi = torch.arange(q.shape[-2])[:, None]
    ki = torch.arange(k.shape[-2])[None, :]
    mask = block_mask.mask_mod(torch.tensor(0), torch.tensor(0), qi, ki)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=enable_gqa)


class _DenseReferenceHandle:
    def __init__(self, mask):
        self.mask = mask

    def add(self, layer, *, k, v):
        self.k, self.v = k, v

    def read(self, layer, q):
        assert self.mask.shape == (q.shape[1], self.k.shape[1]), (self.mask.shape, q.shape, self.k.shape)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            self.k.transpose(1, 2),
            self.v.transpose(1, 2),
            attn_mask=self.mask,
            enable_gqa=q.shape[2] != self.k.shape[2],
        )
        return out.transpose(1, 2)


@pytest.mark.parametrize("k", [1, 2])
def test_real_tower_graph_nodes_match_reference_cold_and_warm(k: int, monkeypatch) -> None:
    monkeypatch.setattr(L, "_compiled_flex_attention", _eager_reference_flex)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _run_case(k)
    finally:
        torch.set_num_threads(previous_threads)


def _run_case(k: int) -> None:
    _, c = _tiny_config()
    ref = tiny_policy(seed=7, dtype=torch.float32).eval()
    portb = build_backbone(c).eval()
    porte = build_expert(c).eval()
    for expected, actual in (
        (ref.backbone.encoder, portb.encoder),
        (ref.backbone.decoder, portb.decoder),
        (ref.action_expert, porte),
    ):
        actual.load_state_dict({_local_key(key): value for key, value in expected.state_dict().items()}, strict=True)
    gen = torch.Generator().manual_seed(311 + k)
    text = torch.randn(1, 3, 32, generator=gen)
    blank = torch.randn(1, 2, 32, generator=gen)
    anchor = torch.randn(1, 192, 1, 2, 4, generator=gen)
    clean = torch.randn(1, 192, (k - 1) * 6, 2, 4, generator=gen) if k > 1 else None
    pro = torch.randn(1, k, 10, generator=gen)
    settings = SamplerSettings(num_steps=2, t_shift=1.0)
    with torch.no_grad():
        vr, ar = ref.sample_window_joint(
            text,
            blank,
            anchor,
            [] if clean is None else [clean],
            pro,
            17,
            settings,
            return_window=True,
            ctx_cache=False,
            overlap_expert=False,
        )
    tpf = 8
    f = 6
    g = 4 + tpf
    nclean = (k - 1) * f
    nnoised = k * f
    enc_c = dense_chunk_causal_mask(4, tpf, text.shape[1], f * tpf, k, k - 1)
    enc_u = dense_chunk_causal_mask(4, tpf, blank.shape[1], f * tpf, k, k - 1)
    dec = visible(encoder_codes(4, 0, 0, f * tpf * 4, k, 0))
    cross = cross_visibility(
        ExpertCross(f, 2, tpf, (2 * k - 1) * f), ExpertCrossLayout(k, g + text.shape[1], nclean, nnoised)
    )
    selfm = self_visibility(ExpertSelf(f, 2), ExpertSelfLayout(k))
    handles = {
        ENCODER_COND_KV: _DenseReferenceHandle(enc_c),
        ENCODER_BLANK_KV: _DenseReferenceHandle(enc_u),
        DECODER_COND_KV: _DenseReferenceHandle(dec),
        DECODER_BLANK_KV: _DenseReferenceHandle(dec),
        EXPERT_QUERIES: _DenseReferenceHandle(selfm),
        EXPERT_TRUNK_KEYS: _DenseReferenceHandle(cross),
    }
    bn = Praxis1BackboneSubmodule(c, portb)
    en = Praxis1ActionExpertSubmodule(c, porte)
    bn.state_handles = handles
    en.state_handles = handles
    state = {
        "video_latents": [seeded_normal((1, 192, k * f, 2, 4), 17, VIDEO_SALT)],
        "action_latents": [seeded_normal((1, k * f, 40), 17, ACTION_SALT)],
    }
    with torch.no_grad():
        for step in range(2):
            info = CurrentForwardPassInfo(
                request_id="r",
                graph_walk="chunk",
                fwd_index=step,
                random_seed=17,
                max_tokens=1,
                step_metadata={
                    "num_steps": 2,
                    "t_shift": 1.0,
                    "cfg_video_scale": 4.0,
                    "cfg_scale": 1.0,
                    "text_tokens": 3,
                    "blank_tokens": 2,
                },
                dynamic_loop_iter_counts={"praxis_denoise": step},
            )
            inp = {"obs_latents": [anchor], "text_embeds": [text], "text_embeds_blank": [blank], **state}
            if clean is not None:
                inp.update(anchor_latents=[anchor], clean_latents=[clean])
            prepared = bn.prepare_inputs("chunk", info, inp)
            out = bn.forward("chunk", None, **prepared.tensor_inputs, **prepared.kwargs)
            ein = {**out, "proprio": [pro[:, -1:]]}
            if clean is not None:
                ein["proprio_all"] = [pro]
            prepared = en.prepare_inputs("chunk", info, ein)
            state = en.forward("chunk", None, **prepared.tensor_inputs, **prepared.kwargs)
    assert torch.equal(state["video_latents"][0], vr)
    assert torch.equal(state["action_latents"][0], ar)
