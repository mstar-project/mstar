"""Ragged FA4 varlen backend (``RaggedAttnBackend.FA4``): which manager it builds,
and on a Blackwell GPU with flash-attn-4 installed, eager outputs against SDPA.

The numeric tests skip off Blackwell or without ``flash_attn.cute``; the build
tests run on CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedAttnBackend,
    RaggedCrossAttentionSpec,
)
from mstar.engine.resources.attn.ragged import dense as ragged_dense
from mstar.engine.resources.attn.ragged.dense import Fa4RaggedCrossManager, Fa4RaggedManager, fa4_unavailable_reason
from mstar.engine.resources.attn.ragged.dense_wrappers import Fa4RaggedPrefillWrapper
from mstar.engine.resources.attn.ragged.flashinfer import FlashInferRaggedCrossManager, FlashInferRaggedManager
from mstar.engine.resources.base import EngineResourceInfo, build_resource

CPU = torch.device("cpu")


def _spec(spec_class):
    config = RaggedAttentionConfig(num_qo_heads=4, num_kv_heads=4, head_dim=128, backend=RaggedAttnBackend.FA4)
    return spec_class(resource_key="ragged", nodes={"node"}, config=config)


@pytest.mark.parametrize(
    ("spec_class", "manager_class"),
    [(RaggedAttentionSpec, FlashInferRaggedManager), (RaggedCrossAttentionSpec, FlashInferRaggedCrossManager)],
)
def test_fa4_falls_back_to_flashinfer_off_blackwell(spec_class, manager_class):
    manager = build_resource(_spec(spec_class), EngineResourceInfo(device=CPU, kv_dtype=torch.bfloat16))
    assert type(manager) is manager_class


@pytest.mark.parametrize(
    ("spec_class", "manager_class"),
    [(RaggedAttentionSpec, Fa4RaggedManager), (RaggedCrossAttentionSpec, Fa4RaggedCrossManager)],
)
def test_fa4_builds_when_available(spec_class, manager_class, monkeypatch):
    monkeypatch.setattr(ragged_dense, "fa4_unavailable_reason", lambda device=None: None)
    manager = build_resource(_spec(spec_class), EngineResourceInfo(device=CPU, kv_dtype=torch.bfloat16))
    assert type(manager) is manager_class


_GPU_REASON = "no CUDA" if not torch.cuda.is_available() else fa4_unavailable_reason(torch.device("cuda", 0))


@pytest.mark.skipif(_GPU_REASON is not None, reason=f"FA4 varlen unavailable: {_GPU_REASON}")
@pytest.mark.parametrize("seg_lens", [[384, 200, 40], [7, 1, 1300]], ids=["mixed", "ragged"])
def test_fa4_eager_matches_sdpa(seg_lens):
    device, heads, dim = torch.device("cuda", 0), 7, 128
    cu = torch.tensor([0, *torch.tensor(seg_lens).cumsum(0).tolist()], dtype=torch.int32)
    gen = torch.Generator(device=device).manual_seed(0)
    q, k, v = (torch.randn(sum(seg_lens), heads, dim, dtype=torch.bfloat16, device=device, generator=gen)
               for _ in range(3))
    w = Fa4RaggedPrefillWrapper(num_qo_heads=heads, num_kv_heads=heads, head_dim=dim, device=device,
                                q_data_type=torch.bfloat16)
    w.plan(cu)
    got = w.run(q, k, v)
    want = torch.empty_like(q)
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist(), strict=True):
        qs, ks, vs = (t[start:end].transpose(0, 1).unsqueeze(0).float() for t in (q, k, v))
        want[start:end] = F.scaled_dot_product_attention(qs, ks, vs).squeeze(0).transpose(0, 1).to(q.dtype)
    err = (got.float() - want.float()).abs().max().item()
    assert err < 3e-2, f"max_abs_err={err}"
