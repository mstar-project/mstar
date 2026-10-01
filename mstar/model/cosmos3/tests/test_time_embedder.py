"""The timestep embedder stays fp32 under any cast of the module tree that holds it.

The DiT and reasoner submodules share one transformer; the worker casts each submodule to bf16 in load order. Before
the ``_apply`` pin, a reasoner cast that ran after the DiT's re-cast the shared embedder to bf16 and the fp32
timestep features failed the matmul — batched CFG requests returned 500 and the image-gen graph capture failed, on
about half of the server launches (the node set iterates in hash order).
"""

import torch

from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer, TimestepEmbedder
from mstar.model.cosmos3.tests.test_edge import _init_all, _tiny_edge_config


def _embedder_dtypes(module):
    emb = getattr(module, "time_embedder", module)
    return {p.dtype for p in emb.parameters()}


def test_embedder_survives_casts_of_its_parent():
    model = Cosmos3OmniTransformer(_tiny_edge_config())
    _init_all(model)
    model.to(torch.bfloat16)  # the build-time pin
    assert _embedder_dtypes(model) == {torch.float32}
    # A second cast from another holder of the same transformer (the reasoner submodule's engine cast).
    model.to(dtype=torch.bfloat16)
    model.bfloat16()
    model.half()
    assert _embedder_dtypes(model) == {torch.float32}
    # Everything else did take the cast.
    assert model.proj_in.weight.dtype == torch.float16


def test_embedder_forward_takes_fp32_and_bf16_timestep_features():
    emb = TimestepEmbedder(in_channels=8, time_embed_dim=16).bfloat16()
    assert _embedder_dtypes(emb) == {torch.float32}
    feats = torch.randn(3, 8)
    out32 = emb(feats)
    out16 = emb(feats.to(torch.bfloat16))
    assert out32.dtype == torch.float32 and out16.dtype == torch.float32
    assert torch.allclose(out32, emb(feats.float()))
    torch.testing.assert_close(out16, emb(feats.to(torch.bfloat16).float()))


def test_device_move_keeps_fp32_without_a_dtype_change():
    emb = TimestepEmbedder(in_channels=8, time_embed_dim=16)
    emb.to(device="cpu", dtype=torch.bfloat16)
    assert _embedder_dtypes(emb) == {torch.float32}
    assert all(p.device.type == "cpu" for p in emb.parameters())
