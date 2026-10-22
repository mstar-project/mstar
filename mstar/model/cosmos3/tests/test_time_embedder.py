"""The timestep embedder stays fp32: the transformer's precision policy keeps it so.

The DiT and reasoner submodules share one transformer. Before node policies, the worker cast each submodule to
bf16 in load order, and a reasoner cast that ran after the DiT's re-cast the shared embedder to bf16; the fp32
timestep features then failed the matmul (batched CFG requests returned 500 and the image-gen capture failed, on
about half of the launches). Both nodes now declare one policy, which the engine applies and checks per node.
"""

import torch

from mstar.engine.precision import apply_precision, precision_mismatches
from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer, TimestepEmbedder
from mstar.model.cosmos3.cosmos3_model import DIT_NODE, REASONER_NODE, TRANSFORMER_PRECISION, Cosmos3Model
from mstar.model.cosmos3.tests.test_edge import _init_all, _tiny_edge_config


def _embedder_dtypes(module):
    emb = getattr(module, "time_embedder", module)
    return {p.dtype for p in emb.parameters()}


def test_policy_keeps_the_embedder_fp32_and_casts_the_rest():
    with torch.device("meta"):
        model = Cosmos3OmniTransformer(_tiny_edge_config())
    apply_precision(model, TRANSFORMER_PRECISION)
    assert _embedder_dtypes(model) == {torch.float32}
    assert model.proj_in.weight.dtype == torch.bfloat16
    model = model.to_empty(device="cpu")
    _init_all(model)
    assert _embedder_dtypes(model) == {torch.float32}
    assert precision_mismatches(model, TRANSFORMER_PRECISION) == []


def test_dit_and_reasoner_declare_one_policy_for_the_shared_transformer():
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    dit, reasoner = model.get_node_precision(DIT_NODE), model.get_node_precision(REASONER_NODE)
    assert dit == reasoner
    assert dit.fp32_modules == ("transformer.time_embedder",) and not dit.autocast


def test_embedder_forward_takes_fp32_and_bf16_timestep_features():
    emb = TimestepEmbedder(in_channels=8, time_embed_dim=16)
    assert _embedder_dtypes(emb) == {torch.float32}
    feats = torch.randn(3, 8)
    out32 = emb(feats)
    out16 = emb(feats.to(torch.bfloat16))
    assert out32.dtype == torch.float32 and out16.dtype == torch.float32
    assert torch.allclose(out32, emb(feats.float()))
    torch.testing.assert_close(out16, emb(feats.to(torch.bfloat16).float()))
